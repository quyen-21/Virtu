
import os, math, json
from pathlib import Path
import numpy as np, pandas as pd, joblib, torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_fast_solver import fast_solve_layout
from .physics_heavy_solver import needs_heavy_check, heavy_check_layout
from .layout_reranker import choose_best_layout

SERVICE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.getenv('ARTIFACT_DIR', SERVICE_ROOT / 'artifacts'))

CATEGORIES = ['sofa','chair','armchair','coffee_table','dining_table','desk','bed','nightstand','wardrobe','cabinet','shelf','tv_stand','tv','rug','bench','lamp','plant','vase','book','decor','wall_art','mirror','door','window','curtain','sink','toilet','bathtub','counter','stove','fridge','shower','ceiling_lamp','wall_planter','hanging_planter','unknown']
SUPPORTERS = {'coffee_table','dining_table','desk','nightstand','cabinet','shelf','tv_stand','counter'}
NEEDS_SUPPORT = {'tv','vase','book','decor','lamp','plant'}
WALL = {'wall_art','mirror','curtain','window','door','wall_planter','hanging_planter'}
CEILING = {'ceiling_lamp'}
UNDER = {'rug'}
LAYER = {c:'floor' for c in CATEGORIES}
for c in ['tv','vase','book','decor','lamp','plant']: LAYER[c] = 'top_surface'
for c in WALL: LAYER[c] = 'wall'
for c in CEILING: LAYER[c] = 'ceiling'
for c in UNDER: LAYER[c] = 'floor_under'

FRONT_AXIS = {
    'bed': '-Z', 'sofa': '-Z', 'tv': '+Z', 'chair': '+Z', 'armchair': '+Z',
    'wardrobe': '-Z', 'nightstand': '-Z', 'desk': '+Z', 'counter': '+Z',
    'tv_stand': '-Z', 'sink': '+Z', 'toilet': '+Z', 'stove': '+Z', 'fridge': '-Z',
}
WALL_OBJECTS = {'bed','sofa','wardrobe','cabinet','shelf','tv_stand','desk','counter','sink','toilet','bathtub','stove','fridge'}


CATEGORY_ALIASES = {
    'flower_vase': 'vase',
    'table_vase': 'vase',
    'decor_vase': 'vase',
    'vase_decor': 'vase',
    'hanging planter': 'hanging_planter',
    'hanging plant': 'hanging_planter',
    'hanging_planter': 'hanging_planter',
    'wall planter': 'wall_planter',
    'wall plant': 'wall_planter',
    'wall_planter': 'wall_planter',
    'ceiling light': 'ceiling_lamp',
    'ceiling lamp': 'ceiling_lamp',
    'pendant light': 'ceiling_lamp',
    'pendant lamp': 'ceiling_lamp',
    'pendant': 'ceiling_lamp',
    'chandelier': 'ceiling_lamp',
    'hanging light': 'ceiling_lamp',
    'hanging lamp': 'ceiling_lamp',
    'table lamp': 'lamp',
    'floor lamp': 'lamp',
    'desk lamp': 'lamp',
}


def safe(x): return str(x or 'unknown').strip().lower()

def canonical_category(value):
    s = safe(value).replace('-', ' ').replace('_', ' ').strip()
    return CATEGORY_ALIASES.get(s, CATEGORY_ALIASES.get(s.replace(' ', '_'), s.replace(' ', '_')))

def fp(p):
    f = p.get('footprint') or {}
    return {
        'widthM': float(p.get('widthM', p.get('width_m', f.get('widthM', .6)) or .6)),
        'depthM': float(p.get('depthM', p.get('depth_m', f.get('depthM', .6)) or .6)),
        'heightM': float(p.get('heightM', p.get('height_m', f.get('heightM', .6)) or .6)),
    }

class DecorTransformerV2(nn.Module):
    def __init__(self, vocab, numeric_dim, d_model=192, nhead=6, num_layers=4, dropout=.12):
        super().__init__()
        self.vocab = vocab
        def emb(name): return nn.Embedding(len(vocab[name]), d_model, padding_idx=vocab['pad_idx'].get(name, 0))
        self.cat_emb = emb('categories'); self.room_emb = emb('room_types'); self.style_emb = emb('styles')
        self.layer_emb = emb('layers'); self.anchor_emb = emb('anchors'); self.zone_emb = emb('zones')
        self.group_emb = emb('groups'); self.wall_emb = emb('nearest_walls'); self.support_emb = emb('support_states')
        self.num_proj = nn.Sequential(nn.Linear(numeric_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.pos_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3))
        self.rot_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.keep_head = nn.Linear(d_model, 1)
        self.layer_head = nn.Linear(d_model, len(vocab['layers']))
        self.zone_head = nn.Linear(d_model, len(vocab['zones']))
        self.wall_head = nn.Linear(d_model, len(vocab['nearest_walls']))
        self.group_head = nn.Linear(d_model, len(vocab['groups']))
        self.support_head = nn.Linear(d_model, len(vocab['support_states']))
    def forward(self, b):
        x = (
            self.cat_emb(b['cat']) + self.room_emb(b['room_t']) + self.style_emb(b['style']) +
            self.layer_emb(b['layer']) + self.anchor_emb(b['anchor']) + self.zone_emb(b['zone']) +
            self.group_emb(b['group']) + self.wall_emb(b['wall']) + self.support_emb(b['support']) +
            self.num_proj(b['num'])
        )
        h = self.norm(self.encoder(x, src_key_padding_mask=b['mask'] < .5))
        return {
            'pos': self.pos_head(h),
            'rot': F.normalize(self.rot_head(h), dim=-1, eps=1e-6),
            'keep_logit': self.keep_head(h).squeeze(-1),
            'layer_logit': self.layer_head(h),
            'zone_logit': self.zone_head(h),
            'wall_logit': self.wall_head(h),
            'group_logit': self.group_head(h),
            'support_logit': self.support_head(h),
        }

class HybridInteriorRuntimeV2:
    def __init__(self, artifact_dir=ARTIFACT_DIR):
        self.artifact_dir = Path(artifact_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.product_filter = self._load_joblib('product_filter.joblib')
        self.set_filter = self._load_joblib('set_filter.joblib')
        self.layout_model = None
        self.layout_vocab = None
        self.layout_cfg = {}
        self.numeric_features = []
        self.max_objects = 16
        self._load_layout_model()

    def _room_type(self, room):
        return safe(room.get('type', room.get('room_type', 'unknown')))

    def _canonical_category(self, value):
        return canonical_category(value)

    def _room_dims(self, room):
        return float(room.get('widthM', 4)), float(room.get('lengthM', 5)), float(room.get('heightM', 2.8))

    def _product_id(self, product, fallback):
        return str(product.get('productId', product.get('id', fallback)))

    def _front_axis(self, category):
        return FRONT_AXIS.get(safe(category), '+Z')

    def _model_front_offset(self, category):
        return {'+X': 0.0, '+Z': -math.pi / 2, '-X': math.pi, '-Z': math.pi / 2}.get(self._front_axis(category), -math.pi / 2)

    def _face_target_angle(self, item, target):
        ix, iz = float(item.get('x', 0.0)), float(item.get('z', 0.0))
        tx, tz = target
        return self._norm_angle(math.atan2(tz - iz, tx - ix) + self._model_front_offset(item.get('category')))

    def _norm_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _wall_outward_target(self, room, wall):
        rw, rl, _ = self._room_dims(room)
        return {
            'front': (rw / 2, rl / 2),
            'back': (rw / 2, rl / 2),
            'left': (rw / 2, rl / 2),
            'right': (rw / 2, rl / 2),
        }.get(wall, (rw / 2, rl / 2))

    def _rotation_from_wall(self, room, item, wall):
        return self._face_target_angle(item, self._wall_outward_target(room, wall))

    def _relation_graph(self, room_type):
        graphs = {
            'bedroom': {
                'primaryObject': 'bed',
                'relations': [
                    {'a': 'bed', 'relation': 'against_wall', 'b': 'back_wall'},
                    {'a': 'bed', 'relation': 'face_to', 'b': 'room_center'},
                    {'a': 'nightstand', 'relation': 'left_of', 'b': 'bed'},
                    {'a': 'nightstand', 'relation': 'right_of', 'b': 'bed'},
                    {'a': 'rug', 'relation': 'under', 'b': 'bed'},
                    {'a': 'bench', 'relation': 'in_front_of', 'b': 'bed'},
                    {'a': 'wardrobe', 'relation': 'against_wall', 'b': 'right_wall'},
                    {'a': 'chair', 'relation': 'near', 'b': 'side_table'},
                ],
            },
            'living_room': {
                'primaryObject': 'sofa',
                'relations': [
                    {'a': 'sofa', 'relation': 'against_wall', 'b': 'back_wall'},
                    {'a': 'sofa', 'relation': 'face_to', 'b': 'tv'},
                    {'a': 'tv', 'relation': 'face_to', 'b': 'sofa'},
                    {'a': 'coffee_table', 'relation': 'between', 'b': 'sofa_tv'},
                    {'a': 'coffee_table', 'relation': 'in_front_of', 'b': 'sofa'},
                    {'a': 'rug', 'relation': 'under', 'b': 'seating_group'},
                    {'a': 'armchair', 'relation': 'near', 'b': 'coffee_table'},
                    {'a': 'side_table', 'relation': 'near', 'b': 'armchair'},
                ],
            },
            'kitchen': {
                'primaryObject': 'counter',
                'relations': [
                    {'a': 'counter', 'relation': 'against_wall', 'b': 'front_wall'},
                    {'a': 'sink', 'relation': 'near', 'b': 'counter'},
                    {'a': 'stove', 'relation': 'near', 'b': 'counter'},
                    {'a': 'fridge', 'relation': 'near', 'b': 'counter'},
                    {'a': 'dining_table', 'relation': 'in_free_space', 'b': 'center_zone'},
                    {'a': 'chair', 'relation': 'face_to', 'b': 'dining_table'},
                ],
            },
            'bathroom': {
                'primaryObject': 'sink',
                'relations': [
                    {'a': 'toilet', 'relation': 'against_wall', 'b': 'back_wall'},
                    {'a': 'sink', 'relation': 'against_wall', 'b': 'front_wall'},
                    {'a': 'mirror', 'relation': 'above', 'b': 'sink'},
                    {'a': 'bathtub', 'relation': 'near', 'b': 'corner'},
                    {'a': 'cabinet', 'relation': 'against_wall', 'b': 'side_wall'},
                ],
            },
        }
        key = 'living_room' if room_type == 'livingroom' else room_type
        return {'roomType': key, **graphs.get(key, {'primaryObject': None, 'relations': []})}

    def _door_lines(self, room):
        openings = room.get('openings') or room.get('doors') or []
        lines = []
        for o in openings:
            if not isinstance(o, dict):
                continue
            kind = safe(o.get('type', 'door'))
            if kind not in {'door', 'opening', 'passage'}:
                continue
            wall = safe(o.get('wall', o.get('wallAnchor', 'front')))
            pos = float(o.get('position', o.get('center', 0.5)) or 0.5)
            span = float(o.get('widthM', o.get('width', 0.9)) or 0.9)
            lines.append({'wall': wall, 'position': max(0.05, min(0.95, pos)), 'widthM': max(0.4, span)})
        return lines

    def _door_clearance_penalty(self, room, item, door_lines):
        if not door_lines:
            return 0.0
        rw, rl, _ = self._room_dims(room)
        x, z = float(item.get('x', 0.0)), float(item.get('z', 0.0))
        w, d = self._rotated_aabb_size(item)
        pad = max(w, d) / 2 + 0.15
        penalty = 0.0
        for door in door_lines:
            if door['wall'] == 'front':
                door_x = rw * door['position']
                if z - d / 2 <= 0.35 and abs(x - door_x) < pad + door['widthM'] / 2:
                    penalty += 1.0
            elif door['wall'] == 'back':
                door_x = rw * door['position']
                if z + d / 2 >= rl - 0.35 and abs(x - door_x) < pad + door['widthM'] / 2:
                    penalty += 1.0
            elif door['wall'] == 'left':
                door_z = rl * door['position']
                if x - w / 2 <= 0.35 and abs(z - door_z) < pad + door['widthM'] / 2:
                    penalty += 1.0
            elif door['wall'] == 'right':
                door_z = rl * door['position']
                if x + w / 2 >= rw - 0.35 and abs(z - door_z) < pad + door['widthM'] / 2:
                    penalty += 1.0
        return penalty

    def _hard_validate_items(self, room, items):
        room_type = self._room_type(room)
        cats = {safe(it.get('category')) for it in items}
        reasons = []
        if room_type == 'bedroom' and 'bed' not in cats:
            reasons.append('missing_bed')
        if room_type in {'living_room', 'livingroom'} and 'sofa' not in cats:
            reasons.append('missing_sofa')
        if room_type == 'kitchen' and 'counter' not in cats:
            reasons.append('missing_counter')
        if room_type == 'bathroom' and 'sink' not in cats:
            reasons.append('missing_sink')
        if room.get('openings') or room.get('doors'):
            if any(self._door_clearance_penalty(room, it, self._door_lines(room)) > 0 for it in items if it.get('layer') == 'floor'):
                reasons.append('blocks_door')
        if room_type == 'bedroom' and 'nightstand' in cats and 'bed' not in cats:
            reasons.append('nightstand_without_bed')
        if room_type in {'living_room', 'livingroom'} and 'coffee_table' in cats and 'sofa' not in cats:
            reasons.append('coffee_table_without_sofa')
        if room_type == 'bedroom' and 'bed' in cats:
            if not any(it.get('placementZone') == 'bedside' for it in items if safe(it.get('category')) == 'nightstand'):
                reasons.append('missing_bedside_group')
            if not any(safe(it.get('category')) == 'rug' and it.get('placementZone') == 'under_bed' for it in items):
                reasons.append('missing_bed_rug_group')
        if room_type in {'living_room', 'livingroom'} and 'sofa' in cats:
            if not any(it.get('placementZone') == 'center' for it in items if safe(it.get('category')) == 'coffee_table'):
                reasons.append('missing_coffee_table_group')
            if not any(safe(it.get('category')) == 'rug' and it.get('placementZone') == 'under_seating' for it in items):
                reasons.append('missing_seating_rug_group')
            if 'armchair' in cats and not any(it.get('placementZone') == 'reading_corner' for it in items if safe(it.get('category')) in {'armchair', 'chair'}):
                reasons.append('missing_armchair_group')
        if room_type == 'kitchen' and 'counter' in cats:
            if not any(safe(it.get('category')) == 'sink' and it.get('placementZone') == 'sink_zone' for it in items):
                reasons.append('missing_sink_group')
            if not any(safe(it.get('category')) == 'stove' and it.get('placementZone') == 'cooking_zone' for it in items):
                reasons.append('missing_stove_group')
            if 'fridge' in cats and not any(it.get('placementZone') == 'entry' for it in items if safe(it.get('category')) == 'fridge'):
                reasons.append('missing_fridge_group')
        if room_type == 'bathroom' and 'sink' in cats:
            if not any(safe(it.get('category')) == 'mirror' and it.get('placementZone') == 'wall' for it in items):
                reasons.append('missing_mirror_group')
            if 'toilet' in cats and not any(it.get('placementZone') == 'toilet_zone' for it in items if safe(it.get('category')) == 'toilet'):
                reasons.append('missing_toilet_group')
            if not any(safe(it.get('category')) in {'bathtub', 'shower'} and it.get('placementZone') == 'corner' for it in items):
                reasons.append('missing_bath_group')
        return reasons

    def _composition_policy(self, room_type):
        room_type = 'living_room' if room_type == 'livingroom' else room_type
        return {
            'bedroom': {
                'primary': 'bed', 'secondary': ['nightstand', 'bench'], 'accent': ['rug', 'chair'],
                'style_bias': 'balanced_symmetry', 'density': .24, 'focal': 'back_wall', 'zones': ['back_band', 'bedside', 'foot_band', 'right_band', 'corner']
            },
            'living_room': {
                'primary': 'sofa', 'secondary': ['tv', 'coffee_table'], 'accent': ['rug', 'armchair'],
                'style_bias': 'center_axis', 'density': .22, 'focal': 'sofa_tv_axis', 'zones': ['back_band', 'front_band', 'center', 'corner', 'reading_corner']
            },
            'kitchen': {
                'primary': 'counter', 'secondary': ['sink', 'stove', 'fridge'], 'accent': ['dining_table', 'chair'],
                'style_bias': 'workflow', 'density': .26, 'focal': 'work_triangle', 'zones': ['front_band', 'work_zone', 'dining_zone', 'entry_zone', 'storage_zone']
            },
            'bathroom': {
                'primary': 'sink', 'secondary': ['toilet', 'bathtub'], 'accent': ['mirror', 'cabinet'],
                'style_bias': 'clear_path', 'density': .2, 'focal': 'wall_function', 'zones': ['front_band', 'back_band', 'corner', 'storage_zone']
            },
        }.get(room_type, {'primary': None, 'secondary': [], 'accent': [], 'style_bias': 'generic', 'density': .22, 'focal': 'center', 'zones': ['center']})

    def _intent_variants(self, room_type, policy):
        room_type = 'living_room' if room_type == 'livingroom' else room_type
        if room_type == 'bedroom':
            return [
                {'name': 'balanced_symmetry', 'bias': {'symmetry': 1.35, 'spacing': 1.0, 'decor': 0.9, 'clear_path': 1.0, 'focal': 1.0}},
                {'name': 'cozy', 'bias': {'symmetry': 0.95, 'spacing': 0.92, 'decor': 1.35, 'clear_path': 0.95, 'focal': 1.05}},
                {'name': 'minimal', 'bias': {'symmetry': 1.05, 'spacing': 1.18, 'decor': 0.55, 'clear_path': 1.2, 'focal': 0.9}},
                {'name': 'workflow', 'bias': {'symmetry': 0.9, 'spacing': 1.0, 'decor': 0.7, 'clear_path': 1.25, 'focal': 1.0}},
            ]
        if room_type in {'living_room', 'livingroom'}:
            return [
                {'name': 'balanced_symmetry', 'bias': {'symmetry': 1.3, 'spacing': 1.0, 'decor': 0.95, 'clear_path': 1.0, 'focal': 1.05}},
                {'name': 'cozy', 'bias': {'symmetry': 0.9, 'spacing': 0.92, 'decor': 1.3, 'clear_path': 0.95, 'focal': 1.1}},
                {'name': 'minimal', 'bias': {'symmetry': 1.0, 'spacing': 1.15, 'decor': 0.55, 'clear_path': 1.2, 'focal': 0.95}},
                {'name': 'workflow', 'bias': {'symmetry': 0.85, 'spacing': 1.05, 'decor': 0.75, 'clear_path': 1.25, 'focal': 1.0}},
            ]
        if room_type == 'kitchen':
            return [
                {'name': 'balanced_symmetry', 'bias': {'symmetry': 1.05, 'spacing': 1.0, 'decor': 0.7, 'clear_path': 1.0, 'focal': 1.0}},
                {'name': 'cozy', 'bias': {'symmetry': 0.9, 'spacing': 0.95, 'decor': 1.15, 'clear_path': 0.95, 'focal': 1.0}},
                {'name': 'minimal', 'bias': {'symmetry': 1.0, 'spacing': 1.15, 'decor': 0.5, 'clear_path': 1.25, 'focal': 0.95}},
                {'name': 'workflow', 'bias': {'symmetry': 0.9, 'spacing': 1.05, 'decor': 0.65, 'clear_path': 1.35, 'focal': 1.1}},
            ]
        if room_type == 'bathroom':
            return [
                {'name': 'balanced_symmetry', 'bias': {'symmetry': 1.15, 'spacing': 1.0, 'decor': 0.8, 'clear_path': 1.0, 'focal': 1.0}},
                {'name': 'cozy', 'bias': {'symmetry': 0.9, 'spacing': 0.95, 'decor': 1.15, 'clear_path': 0.98, 'focal': 1.0}},
                {'name': 'minimal', 'bias': {'symmetry': 1.0, 'spacing': 1.15, 'decor': 0.45, 'clear_path': 1.25, 'focal': 0.95}},
                {'name': 'workflow', 'bias': {'symmetry': 0.85, 'spacing': 1.0, 'decor': 0.65, 'clear_path': 1.3, 'focal': 1.05}},
            ]
        return [{'name': policy.get('style_bias', 'generic'), 'bias': {'symmetry': 1.0, 'spacing': 1.0, 'decor': 1.0, 'clear_path': 1.0, 'focal': 1.0}}]

    def _room_zones(self, room):
        rw, rl, _ = self._room_dims(room)
        return {
            'back_band': {'x1': 0.08, 'z1': rl * 0.68, 'x2': rw - 0.08, 'z2': rl - 0.08},
            'front_band': {'x1': 0.08, 'z1': 0.08, 'x2': rw - 0.08, 'z2': rl * 0.32},
            'center': {'x1': rw * 0.2, 'z1': rl * 0.28, 'x2': rw * 0.8, 'z2': rl * 0.72},
            'bedside': {'x1': 0.1, 'z1': rl * 0.45, 'x2': rw - 0.1, 'z2': rl * 0.65},
            'foot_band': {'x1': rw * 0.2, 'z1': rl * 0.18, 'x2': rw * 0.8, 'z2': rl * 0.42},
            'right_band': {'x1': rw * 0.64, 'z1': 0.1, 'x2': rw - 0.08, 'z2': rl - 0.1},
            'corner': {'x1': rw * 0.68, 'z1': rl * 0.42, 'x2': rw - 0.08, 'z2': rl - 0.08},
            'work_zone': {'x1': 0.08, 'z1': 0.08, 'x2': rw * 0.76, 'z2': rl * 0.35},
            'dining_zone': {'x1': rw * 0.56, 'z1': rl * 0.45, 'x2': rw - 0.08, 'z2': rl - 0.08},
            'entry_zone': {'x1': rw * 0.7, 'z1': 0.08, 'x2': rw - 0.08, 'z2': rl * 0.38},
            'storage_zone': {'x1': rw * 0.68, 'z1': rl * 0.2, 'x2': rw - 0.08, 'z2': rl * 0.72},
            'reading_corner': {'x1': rw * 0.68, 'z1': rl * 0.24, 'x2': rw - 0.08, 'z2': rl * 0.58},
        }

    def _zone_center(self, room, zone_name):
        zones = self._room_zones(room)
        zone = zones.get(zone_name) or zones['center']
        return (zone['x1'] + zone['x2']) / 2, (zone['z1'] + zone['z2']) / 2

    def _place_in_zone(self, room, item, zone_name, rotation_y=None, wall_anchor=None, relations=None, facing_target=None, reason=None):
        x, z = self._zone_center(room, zone_name)
        return self._place_item(room, item, x, z, rotation_y if rotation_y is not None else float(item.get('rotationY', 0.0)), wall_anchor, zone_name, relations, facing_target, reason)

    def _back_wall_center(self, room):
        rw, rl, _ = self._room_dims(room)
        return rw / 2, rl - 0.08

    def _center(self, room):
        rw, rl, _ = self._room_dims(room)
        return rw / 2, rl / 2

    def _wall_position(self, room, wall, item, t=0.5):
        rw, rl, _ = self._room_dims(room)
        f = fp(item)
        t = max(0.12, min(0.88, float(t)))
        if wall == 'left':
            return f['widthM'] / 2 + 0.08, rl * t
        if wall == 'right':
            return rw - f['widthM'] / 2 - 0.08, rl * t
        if wall == 'back':
            return rw * t, rl - f['depthM'] / 2 - 0.08
        return rw * t, f['depthM'] / 2 + 0.08

    def _rotated_aabb_size(self, item):
        f = fp(item)
        c = abs(math.cos(float(item.get('rotationY', 0.0))))
        s = abs(math.sin(float(item.get('rotationY', 0.0))))
        return f['widthM'] * c + f['depthM'] * s, f['widthM'] * s + f['depthM'] * c

    def _rotated_rect(self, item):
        f = fp(item)
        x, z, r = float(item.get('x', 0.0)), float(item.get('z', 0.0)), float(item.get('rotationY', 0.0))
        hw, hd = f['widthM'] / 2, f['depthM'] / 2
        pts = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
        c, s = math.cos(r), math.sin(r)
        return [(x + px * c - pz * s, z + px * s + pz * c) for px, pz in pts]

    def _aabb_from_poly(self, poly):
        xs = [p[0] for p in poly]; zs = [p[1] for p in poly]
        return min(xs), min(zs), max(xs), max(zs)

    def _place_item(self, room, item, x, z, rotation_y, wall_anchor=None, zone='free', relations=None, facing_target=None, reason=None):
        item['x'] = float(x)
        item['z'] = float(z)
        item['y'] = 0.0
        item['rotationY'] = self._norm_angle(float(rotation_y))
        item['wallAnchor'] = wall_anchor
        item['placementZone'] = zone
        item['facingTarget'] = facing_target
        item['relations'] = relations or []
        item['layoutReasoning'] = reason
        item['frontAxis'] = self._front_axis(item.get('category'))
        return item

    def _bedroom_template(self, room, products):
        rw, rl, _ = self._room_dims(room)
        items, used = [], set()
        bed = next((p for p in products if safe(p.get('category')) == 'bed'), None)
        if bed:
            bed = dict(bed); bed['layer'] = 'floor'; bed['footprint'] = fp(bed); bed['productId'] = self._product_id(bed, 'bed_0')
            bx = rw / 2
            bz = rl - bed['footprint']['depthM'] / 2 - 0.08
            bed_probe = {'category': 'bed', 'x': bx, 'z': bz}
            rot = self._face_target_angle(bed_probe, self._center(room))
            self._place_item(room, bed, bx, bz, rot, 'back', 'back_wall',
                             [{'type': 'against_wall', 'target': 'back_wall'}, {'type': 'face_to', 'target': 'room_center'}],
                             'room_center', 'Giường được đặt giữa tường sau, đầu giường áp tường và mặt giường hướng ra trung tâm phòng.')
            items.append(bed); used.add(bed['productId']); bed_cx, bed_cz = bx, bz; bed_f = bed['footprint']
        else:
            bed_cx, bed_cz = self._center(room); bed_f = {'widthM': 1.6, 'depthM': 2.0, 'heightM': 0.8}

        stands = [p for p in products if safe(p.get('category')) == 'nightstand' and self._product_id(p, '') not in used]
        for idx, side in enumerate(['left', 'right']):
            if idx >= len(stands):
                break
            q = dict(stands[idx]); q['layer'] = 'floor'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, f'nightstand_{idx}')
            offset = bed_f['widthM'] / 2 + q['footprint']['widthM'] / 2 + 0.12
            x = bed_cx - offset if side == 'left' else bed_cx + offset
            z = bed_cz - 0.18
            rot = self._face_target_angle({'category': 'nightstand', 'x': x, 'z': z}, self._center(room))
            self._place_item(room, q, x, z, rot, None, 'bedside', [{'type': f'{side}_of', 'target': 'bed'}],
                             'room_center', f'Tab đầu giường được đặt {side} giường để tạo bố cục cân xứng.')
            items.append(q); used.add(q['productId'])

        for cat, zone, reason in [('rug', 'under_bed', 'Thảm được căn dưới giường và lớn hơn vùng ngủ để gom nhóm không gian.'), ('bench', 'foot_of_bed', 'Ghế băng đặt ở cuối giường, không chắn lối đi chính.')]:
            p = next((x for x in products if safe(x.get('category')) == cat and self._product_id(x, '') not in used), None)
            if not p:
                continue
            q = dict(p); q['layer'] = 'floor_under' if cat == 'rug' else 'floor'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, cat)
            z = bed_cz - bed_f['depthM'] / 2 - q['footprint']['depthM'] / 2 - 0.12 if cat == 'bench' else bed_cz - 0.15
            rel = 'under' if cat == 'rug' else 'in_front_of'
            self._place_item(room, q, bed_cx, z, self._face_target_angle({'category': cat, 'x': bed_cx, 'z': z}, self._center(room)), None, zone,
                             [{'type': rel, 'target': 'bed'}], 'room_center', reason)
            items.append(q); used.add(q['productId'])

        wardrobe = next((p for p in products if safe(p.get('category')) in {'wardrobe','cabinet'} and self._product_id(p, '') not in used), None)
        if wardrobe:
            q = dict(wardrobe); q['layer'] = 'floor'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, 'wardrobe')
            x, z = self._wall_position(room, 'right', q, 0.58)
            rot = self._face_target_angle({'category': q.get('category'), 'x': x, 'z': z}, self._center(room))
            self._place_item(room, q, x, z, rot, 'right', 'side_wall', [{'type': 'against_wall', 'target': 'right_wall'}, {'type': 'face_to', 'target': 'room_center'}],
                             'room_center', 'Tủ được áp sát tường phải và mặt tủ hướng ra khoảng trống để tránh quay vào tường.')
            items.append(q); used.add(q['productId'])

        chair = next((p for p in products if safe(p.get('category')) in {'chair', 'armchair'} and self._product_id(p, '') not in used), None)
        if chair:
            q = dict(chair); q['layer'] = 'floor'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, 'chair')
            x, z = rw - q['footprint']['widthM'] / 2 - 0.35, max(q['footprint']['depthM'] / 2 + 0.35, rl * 0.32)
            rot = self._face_target_angle({'category': q.get('category'), 'x': x, 'z': z}, (bed_cx, bed_cz))
            self._place_item(room, q, x, z, rot, None, 'reading_corner', [{'type': 'near', 'target': 'side_table'}, {'type': 'face_to', 'target': 'bed'}],
                             'bed', 'Ghế thư giãn đặt ở góc trống và xoay về giường để tạo góc đọc/sinh hoạt.')
            items.append(q); used.add(q['productId'])
        return items

    def _living_room_template(self, room, products):
        rw, rl, _ = self._room_dims(room)
        items, used = [], set()
        sofa = next((p for p in products if safe(p.get('category')) == 'sofa'), None)
        tv = next((p for p in products if safe(p.get('category')) == 'tv'), None)
        coffee = next((p for p in products if safe(p.get('category')) == 'coffee_table'), None)
        rug = next((p for p in products if safe(p.get('category')) == 'rug'), None)
        armchair = next((p for p in products if safe(p.get('category')) in {'armchair', 'chair'}), None)
        side_table = next((p for p in products if safe(p.get('category')) in {'nightstand', 'cabinet', 'shelf'}), None)

        if sofa:
            q = dict(sofa); q['layer'] = 'floor'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, 'sofa')
            x = rw / 2
            z = rl - q['footprint']['depthM'] / 2 - 0.22
            rot = self._face_target_angle({'category': 'sofa', 'x': x, 'z': z}, (rw / 2, 0.5))
            self._place_item(room, q, x, z, rot, 'back', 'seating',
                             [{'type': 'face_to', 'target': 'tv'}, {'type': 'against_wall', 'target': 'back_wall'}],
                             'tv', 'Sofa đặt gần tường sau và xoay về phía TV để tạo cụm sinh hoạt trung tâm.')
            items.append(q); used.add(q['productId'])
            sofa_cx, sofa_cz = x, z
        else:
            sofa_cx, sofa_cz = rw / 2, rl * 0.7

        if tv:
            q = dict(tv); q['layer'] = 'wall'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, 'tv')
            x = rw / 2
            z = q['footprint']['depthM'] / 2 + 0.08
            rot = self._face_target_angle({'category': 'tv', 'x': x, 'z': z}, (sofa_cx, sofa_cz))
            self._place_item(room, q, x, z, rot, 'front', 'tv_wall',
                             [{'type': 'face_to', 'target': 'sofa'}, {'type': 'against_wall', 'target': 'front_wall'}],
                             'sofa', 'TV được đặt áp tường trước và hướng về sofa.')
            items.append(q); used.add(q['productId'])

        if coffee:
            q = dict(coffee); q['layer'] = 'floor'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, 'coffee_table')
            x = rw / 2
            z = (sofa_cz + rl / 2) / 2
            rot = self._face_target_angle({'category': 'coffee_table', 'x': x, 'z': z}, (sofa_cx, sofa_cz))
            self._place_item(room, q, x, z, rot, None, 'center', [{'type': 'between', 'target': 'sofa_tv'}], 'sofa', 'Bàn trà được đặt giữa sofa và TV để hoàn chỉnh cụm ngồi.')
            items.append(q); used.add(q['productId'])

        if rug:
            q = dict(rug); q['layer'] = 'floor_under'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, 'rug')
            x = rw / 2
            z = (sofa_cz + rl / 2) / 2
            self._place_item(room, q, x, z, 0.0, None, 'under_seating', [{'type': 'under', 'target': 'seating_group'}], 'sofa', 'Thảm gom nhóm ghế ngồi và tạo vùng sinh hoạt rõ ràng.')
            items.append(q); used.add(q['productId'])

        if armchair:
            q = dict(armchair); q['layer'] = 'floor'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, 'armchair')
            x = rw * 0.78
            z = rl * 0.48
            rot = self._face_target_angle({'category': q.get('category'), 'x': x, 'z': z}, (rw / 2, rl / 2))
            self._place_item(room, q, x, z, rot, None, 'reading_corner', [{'type': 'near', 'target': 'side_table'}, {'type': 'face_to', 'target': 'tv'}], 'tv', 'Ghế phụ được đặt lệch góc để cân bằng cụm phòng khách.')
            items.append(q); used.add(q['productId'])

        if side_table:
            q = dict(side_table); q['layer'] = 'floor'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, 'side_table')
            x = min(rw - q['footprint']['widthM'] / 2 - 0.2, rw * 0.86)
            z = rl * 0.55
            self._place_item(room, q, x, z, self._face_target_angle({'category': q.get('category'), 'x': x, 'z': z}, (sofa_cx, sofa_cz)), None, 'side_zone', [{'type': 'near', 'target': 'armchair'}], 'sofa', 'Bàn phụ nằm cạnh ghế để hoàn thiện góc đọc/ngồi.')
            items.append(q); used.add(q['productId'])
        return items

    def _kitchen_template(self, room, products):
        rw, rl, _ = self._room_dims(room)
        items, used = [], set()
        counter = next((p for p in products if safe(p.get('category')) == 'counter'), None)
        sink = next((p for p in products if safe(p.get('category')) == 'sink'), None)
        stove = next((p for p in products if safe(p.get('category')) == 'stove'), None)
        fridge = next((p for p in products if safe(p.get('category')) == 'fridge'), None)
        dining = next((p for p in products if safe(p.get('category')) == 'dining_table'), None)
        chair = next((p for p in products if safe(p.get('category')) == 'chair'), None)

        if counter:
            q = dict(counter); q['layer'] = 'floor'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, 'counter')
            x = rw / 2
            z = q['footprint']['depthM'] / 2 + 0.08
            rot = self._rotation_from_wall(room, {'category': 'counter', 'x': x, 'z': z}, 'front')
            self._place_item(room, q, x, z, rot, 'front', 'against_wall', [{'type': 'against_wall', 'target': 'front_wall'}], 'working_area', 'Bàn bếp áp tường để giữ mặt sử dụng quay ra lối đi.')
            items.append(q); used.add(q['productId'])
            counter_cx, counter_cz = x, z
        else:
            counter_cx, counter_cz = rw / 2, rl * 0.18

        if sink:
            q = dict(sink); q['layer'] = 'floor'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, 'sink')
            x = rw / 2 - 0.45
            z = q['footprint']['depthM'] / 2 + 0.08
            self._place_item(room, q, x, z, self._rotation_from_wall(room, {'category': 'sink', 'x': x, 'z': z}, 'front'), 'front', 'sink_zone', [{'type': 'near', 'target': 'counter'}], 'counter', 'Chậu rửa được đặt sát dải bếp để tạo workflow nấu nướng liên tục.')
            items.append(q); used.add(q['productId'])

        if stove:
            q = dict(stove); q['layer'] = 'floor'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, 'stove')
            x = rw / 2 + 0.45
            z = q['footprint']['depthM'] / 2 + 0.08
            self._place_item(room, q, x, z, self._rotation_from_wall(room, {'category': 'stove', 'x': x, 'z': z}, 'front'), 'front', 'cooking_zone', [{'type': 'near', 'target': 'counter'}], 'counter', 'Bếp nấu nằm tách nhẹ khỏi chậu rửa để an toàn và tiện thao tác.')
            items.append(q); used.add(q['productId'])

        if fridge:
            q = dict(fridge); q['layer'] = 'floor'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, 'fridge')
            x, z = rw - q['footprint']['widthM'] / 2 - 0.14, rl * 0.28
            self._place_item(room, q, x, z, self._face_target_angle({'category': 'fridge', 'x': x, 'z': z}, (rw / 2, rl / 2)), 'right', 'entry', [{'type': 'near', 'target': 'counter'}], 'counter', 'Tủ lạnh đặt gần vùng vào bếp nhưng không chắn lối đi chính.')
            items.append(q); used.add(q['productId'])

        if dining:
            q = dict(dining); q['layer'] = 'floor'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, 'dining_table')
            x = rw * 0.68
            z = rl * 0.62
            self._place_item(room, q, x, z, self._face_target_angle({'category': 'dining_table', 'x': x, 'z': z}, (rw / 2, rl / 2)), None, 'dining_zone', [{'type': 'near', 'target': 'chairs'}], 'center', 'Bàn ăn nằm trong vùng trống để tách khỏi khối bếp.')
            items.append(q); used.add(q['productId'])

        if chair:
            q = dict(chair); q['layer'] = 'floor'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, 'chair')
            x = rw * 0.58
            z = rl * 0.62 + 0.32
            self._place_item(room, q, x, z, self._face_target_angle({'category': 'chair', 'x': x, 'z': z}, (rw * 0.68, rl * 0.62)), None, 'dining_zone', [{'type': 'face_to', 'target': 'dining_table'}], 'dining_table', 'Ghế ăn quay về bàn ăn và giữ khoảng trống đủ đi lại.')
            items.append(q); used.add(q['productId'])
        return items

    def _bathroom_template(self, room, products):
        rw, rl, _ = self._room_dims(room)
        items, used = [], set()
        toilet = next((p for p in products if safe(p.get('category')) == 'toilet'), None)
        sink = next((p for p in products if safe(p.get('category')) == 'sink'), None)
        mirror = next((p for p in products if safe(p.get('category')) == 'mirror'), None)
        bathtub = next((p for p in products if safe(p.get('category')) in {'bathtub', 'shower'}), None)
        cabinet = next((p for p in products if safe(p.get('category')) in {'cabinet', 'shelf'}), None)

        if toilet:
            q = dict(toilet); q['layer'] = 'floor'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, 'toilet')
            x = rw * 0.25
            z = rl - q['footprint']['depthM'] / 2 - 0.15
            self._place_item(room, q, x, z, self._rotation_from_wall(room, {'category': 'toilet', 'x': x, 'z': z}, 'back'), 'back', 'toilet_zone', [{'type': 'against_wall', 'target': 'back_wall'}], 'back_wall', 'Toilet áp tường để giữ lối đi và vùng vệ sinh rõ ràng.')
            items.append(q); used.add(q['productId'])

        if sink:
            q = dict(sink); q['layer'] = 'floor'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, 'sink')
            x = rw * 0.72
            z = q['footprint']['depthM'] / 2 + 0.08
            self._place_item(room, q, x, z, self._rotation_from_wall(room, {'category': 'sink', 'x': x, 'z': z}, 'front'), 'front', 'sink_zone', [{'type': 'against_wall', 'target': 'front_wall'}], 'mirror', 'Lavabo đặt sát tường trước để thuận tay và dành chỗ cho gương phía trên.')
            items.append(q); used.add(q['productId'])

        if mirror:
            q = dict(mirror); q['layer'] = 'wall'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, 'mirror')
            x = rw * 0.72
            z = 0.08
            self._place_item(room, q, x, z, 0.0, 'front', 'wall', [{'type': 'above', 'target': 'sink'}], 'sink', 'Gương treo phía trên lavabo để đúng công năng sử dụng.')
            items.append(q); used.add(q['productId'])

        if bathtub:
            q = dict(bathtub); q['layer'] = 'floor'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, 'bathtub')
            x = rw * 0.62
            z = rl * 0.48
            self._place_item(room, q, x, z, self._face_target_angle({'category': q.get('category'), 'x': x, 'z': z}, (rw / 2, rl / 2)), None, 'corner', [{'type': 'near', 'target': 'corner'}], 'back_wall', 'Bồn tắm đặt ở góc rộng để không chặn khu lavabo.')
            items.append(q); used.add(q['productId'])

        if cabinet:
            q = dict(cabinet); q['layer'] = 'floor'; q['footprint'] = fp(q); q['productId'] = self._product_id(q, 'cabinet')
            x = rw * 0.78
            z = rl * 0.38
            self._place_item(room, q, x, z, self._face_target_angle({'category': q.get('category'), 'x': x, 'z': z}, (rw / 2, rl / 2)), 'right', 'storage', [{'type': 'against_wall', 'target': 'right_wall'}], 'room_center', 'Tủ phụ đặt sát tường bên để chứa đồ mà không ảnh hưởng lối di chuyển.')
            items.append(q); used.add(q['productId'])
        return items

    def _default_template(self, room, products):
        return self.prior(room, products, 0)

    def _template_layout(self, room, products):
        room_type = self._room_type(room)
        if room_type == 'bedroom':
            return self._bedroom_template(room, products)
        if room_type in {'living_room', 'livingroom'}:
            return self._living_room_template(room, products)
        if room_type == 'kitchen':
            return self._kitchen_template(room, products)
        if room_type == 'bathroom':
            return self._bathroom_template(room, products)
        return self._default_template(room, products)

    def _load_joblib(self, name):
        p = self.artifact_dir / name
        return joblib.load(p) if p.exists() else None

    def _safe_json(self, path):
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def _load_layout_model(self):
        ckpt_path = self.artifact_dir / 'decor_transformer_v2.pt'
        vocab_path = self.artifact_dir / 'decor_vocab_v2.json'
        if not ckpt_path.exists():
            return
        ck = torch.load(ckpt_path, map_location=self.device)
        vocab = ck.get('vocab') or self._safe_json(vocab_path)
        if not vocab:
            return
        cfg = ck.get('cfg', {})
        numeric_features = ck.get('numeric_features', [])
        model = DecorTransformerV2(
            vocab=vocab,
            numeric_dim=len(numeric_features),
            d_model=int(cfg.get('D_MODEL', 192)),
            nhead=int(cfg.get('NHEAD', 6)),
            num_layers=int(cfg.get('NUM_LAYERS', 4)),
            dropout=float(cfg.get('DROPOUT', 0.12))
        ).to(self.device)
        state = ck.get('model_state_dict') or ck.get('model')
        model.load_state_dict(state, strict=False)
        model.eval()
        self.layout_model = model
        self.layout_vocab = vocab
        self.layout_cfg = cfg
        self.numeric_features = numeric_features
        self.max_objects = int(cfg.get('MAX_OBJECTS_PER_ROOM', 16))

    def _token(self, group_name, value):
        vocab = self.layout_vocab or {}
        stoi = vocab.get('stoi', {}).get(group_name, {})
        return stoi.get(safe(value), stoi.get('unknown', 0))

    def item_row(self, room, p):
        cat = self._canonical_category(p.get('category', 'unknown'))
        f = fp(p)
        rw = float(room.get('widthM', 4)); rl = float(room.get('lengthM', 5)); rh = float(room.get('heightM', 2.8))
        area = max(rw * rl, 1e-6)
        obj_area = f['widthM'] * f['depthM']
        return {
            'room_type': safe(room.get('type', room.get('room_type', 'unknown'))),
            'style': safe(room.get('style', 'unknown')),
            'category': cat,
            'layer': LAYER.get(cat, 'floor'),
            'anchor': 'free',
            'source_dataset': 'runtime_be',
            'group_type': 'unknown',
            'placement_zone': 'free',
            'nearest_wall': 'front_wall',
            'nearest_wall_distance_m': 0.0,
            'room_width_m': rw,
            'room_length_m': rl,
            'room_height_m': rh,
            'room_area_m2': area,
            'product_width_m': f['widthM'],
            'product_depth_m': f['depthM'],
            'product_height_m': f['heightM'],
            'footprint_area_m2': obj_area,
            'clearance_required_m': .45,
            'support_surface_area_m2': obj_area if cat in SUPPORTERS else 0.0,
            'mass_kg_est': float(p.get('massKg', 0) or 0),
            'max_load_kg_est': float(p.get('maxLoadKg', 0) or 0),
            'area_ratio': obj_area / area,
            'w_norm': f['widthM'] / max(rw, 1e-6),
            'd_norm': f['depthM'] / max(rl, 1e-6),
            'h_norm': f['heightM'] / max(rh, 1e-6),
            'needs_support': float(cat in NEEDS_SUPPORT),
            'is_supporter': float(cat in SUPPORTERS),
            'is_wall_object': float(cat in WALL),
            'is_floor_under': float(cat in UNDER),
            'support_surface_area_norm': 1.0 if cat in SUPPORTERS else 0.0,
            'nearest_wall_distance_norm': 0.0,
            'on': 0.0, 'supported_by': 0.0, 'supports': 0.0, 'under': 0.0, 'near': 0.0,
            'against_wall': 0.0, 'wall_attached': 0.0, 'facing': 0.0, 'same_group': 0.0,
            'clearance_conflict': 0.0,
        }

    def score_products(self, room, products):
        rows = [self.item_row(room, p) for p in products]
        df = pd.DataFrame(rows)
        if self.product_filter is not None:
            try:
                scores = self.product_filter.predict_proba(df)[:, 1]
            except Exception:
                scores = np.array([float(p.get('score', .5) or .5) for p in products], dtype=float)
        else:
            scores = np.array([float(p.get('score', .5) or .5) for p in products], dtype=float)
        out = []
        for p, s in zip(products, scores):
            q = dict(p); q['keepProbability'] = float(s); out.append(q)
        return out

    def score_set(self, room, products):
        if self.set_filter is None:
            return 0.5
        try:
            rw = float(room.get('widthM', 4)); rl = float(room.get('lengthM', 5)); area = max(rw * rl, 1e-6)
            cats = [safe(p.get('category')) for p in products]
            areas = [fp(p)['widthM'] * fp(p)['depthM'] for p in products]
            row = {
                'room_type': safe(room.get('type', room.get('room_type', 'unknown'))),
                'style': safe(room.get('style', 'unknown')),
                'source_dataset': 'runtime_be',
                'num_items': len(products),
                'num_floor_items': sum(LAYER.get(c, 'floor') == 'floor' for c in cats),
                'num_wall_items': sum(c in WALL for c in cats),
                'num_support_required_items': sum(c in NEEDS_SUPPORT for c in cats),
                'num_supporters': sum(c in SUPPORTERS for c in cats),
                'num_groups': len(set(cats)),
                'area_ratio_total': sum(areas) / area,
                'largest_item_area_ratio': max(areas) / area if areas else 0,
                'room_density_score': min(1, sum(areas) / (area * .45)),
                'has_primary_anchor_item': float(any(c in {'sofa','bed','dining_table','desk'} for c in cats)),
                'has_seating_group': float(any(c in {'sofa','armchair','coffee_table'} for c in cats)),
                'has_sleeping_group': float(any(c in {'bed','nightstand'} for c in cats)),
                'has_dining_group': float(any(c in {'dining_table','chair'} for c in cats)),
                'has_work_group': float(any(c in {'desk','chair','shelf'} for c in cats)),
                'pair_compatibility_score': .5 + .1*float('sofa' in cats and 'coffee_table' in cats) + .1*float('bed' in cats and 'nightstand' in cats) + .1*float('tv' in cats and ('tv_stand' in cats or 'shelf' in cats)),
                'group_completeness_score': .5,
                'support_coverage_score': 1 if sum(c in NEEDS_SUPPORT for c in cats) <= max(1, sum(c in SUPPORTERS for c in cats)) else .35
            }
            return float(self.set_filter.predict_proba(pd.DataFrame([row]))[0, 1])
        except Exception:
            return 0.5

    def _decode_label(self, logits, group_name):
        idx = int(torch.argmax(logits).item())
        vocab = self.layout_vocab.get(group_name, []) if self.layout_vocab else []
        return vocab[idx] if idx < len(vocab) else 'unknown'

    def _build_layout_batch(self, room, products):
        n = min(len(products), self.max_objects)
        pad = self.max_objects - n
        rows = [self.item_row(room, p) for p in products[:n]]
        if not rows:
            return None, []

        cat = [self._token('categories', r['category']) for r in rows] + [self.layout_vocab['pad_idx']['categories']] * pad
        room_t = [self._token('room_types', r['room_type']) for r in rows] + [self.layout_vocab['pad_idx']['room_types']] * pad
        style = [self._token('styles', r['style']) for r in rows] + [self.layout_vocab['pad_idx']['styles']] * pad
        layer = [self._token('layers', r['layer']) for r in rows] + [self.layout_vocab['pad_idx']['layers']] * pad
        anchor = [self._token('anchors', r['anchor']) for r in rows] + [self.layout_vocab['pad_idx']['anchors']] * pad
        zone = [self._token('zones', r['placement_zone']) for r in rows] + [self.layout_vocab['pad_idx']['zones']] * pad
        group = [self._token('groups', r['group_type']) for r in rows] + [self.layout_vocab['pad_idx']['groups']] * pad
        wall = [self._token('nearest_walls', r['nearest_wall']) for r in rows] + [self.layout_vocab['pad_idx']['nearest_walls']] * pad
        support = [self._token('support_states', 'needs_support' if r['needs_support'] else ('is_supporter' if r['is_supporter'] else ('wall_mounted' if r['is_wall_object'] else 'none'))) for r in rows] + [self.layout_vocab['pad_idx']['support_states']] * pad

        num = np.array([[float(r.get(k, 0.0)) for k in self.numeric_features] for r in rows], dtype=np.float32)
        if pad > 0:
            num = np.vstack([num, np.zeros((pad, len(self.numeric_features)), np.float32)])

        mask = np.zeros(self.max_objects, np.float32); mask[:n] = 1.0

        batch = {
            'cat': torch.tensor([cat], dtype=torch.long, device=self.device),
            'room_t': torch.tensor([room_t], dtype=torch.long, device=self.device),
            'style': torch.tensor([style], dtype=torch.long, device=self.device),
            'layer': torch.tensor([layer], dtype=torch.long, device=self.device),
            'anchor': torch.tensor([anchor], dtype=torch.long, device=self.device),
            'zone': torch.tensor([zone], dtype=torch.long, device=self.device),
            'group': torch.tensor([group], dtype=torch.long, device=self.device),
            'wall': torch.tensor([wall], dtype=torch.long, device=self.device),
            'support': torch.tensor([support], dtype=torch.long, device=self.device),
            'num': torch.tensor([num], dtype=torch.float32, device=self.device),
            'mask': torch.tensor([mask], dtype=torch.float32, device=self.device),
        }
        return batch, rows

    def transformer_layout(self, room, products):
        if self.layout_model is None or self.layout_vocab is None:
            return None

        batch, rows = self._build_layout_batch(room, products)
        if batch is None:
            return []

        with torch.no_grad():
            out = self.layout_model(batch)

        rw = float(room.get('widthM', 4)); rl = float(room.get('lengthM', 5)); rh = float(room.get('heightM', 2.8))
        items = []
        for i, p in enumerate(products[:len(rows)]):
            f = fp(p)
            pos = out['pos'][0, i].detach().cpu().numpy()
            rot = out['rot'][0, i].detach().cpu().numpy()
            keep_model = float(torch.sigmoid(out['keep_logit'][0, i]).item())

            layer_pred = self._decode_label(out['layer_logit'][0, i], 'layers')
            zone_pred = self._decode_label(out['zone_logit'][0, i], 'zones')
            wall_pred = self._decode_label(out['wall_logit'][0, i], 'nearest_walls')
            group_pred = self._decode_label(out['group_logit'][0, i], 'groups')
            support_pred = self._decode_label(out['support_logit'][0, i], 'support_states')

            x = float(np.clip(pos[0], 0.05, 0.95) * rw)
            z = float(np.clip(pos[2], 0.05, 0.95) * rl)
            y = float(max(0.0, np.clip(pos[1], 0.0, 1.0) * rh))
            rotation_y = float(math.atan2(rot[0], rot[1]))

            item = dict(p)
            item['productId'] = str(p.get('productId', p.get('id', f'item_{i}')))
            item['category'] = self._canonical_category(p.get('category', 'unknown'))
            item['footprint'] = f
            item['keepProbability'] = float((float(p.get('keepProbability', .5) or .5) + keep_model) / 2.0)
            item['x'] = x; item['y'] = y; item['z'] = z
            item['rotationY'] = rotation_y
            item['layer'] = layer_pred if layer_pred not in {'<pad>', 'unknown'} else LAYER.get(item['category'], 'floor')
            item['placementZone'] = zone_pred
            item['predictedGroup'] = group_pred
            item['predictedSupportState'] = support_pred

            if item['layer'] == 'wall' or item['category'] in WALL:
                item['wallAnchor'] = wall_pred if wall_pred not in {'<pad>', 'unknown'} else 'front'
            elif zone_pred in {'front_wall','back_wall','left_wall','right_wall'}:
                item['wallAnchor'] = zone_pred.replace('_wall', '')
            items.append(item)
        return items

    def prior(self, room, products, k=0):
        rw = float(room.get('widthM', 4)); rl = float(room.get('lengthM', 5)); items = []; i = 0
        for p in products:
            cat = self._canonical_category(p.get('category')); f = fp(p); it = dict(p); it['category'] = cat; it['footprint'] = f; it['productId'] = str(p.get('productId', p.get('id', f'item_{len(items)}'))); it['layer'] = LAYER.get(cat, 'floor')
            if it['layer'] == 'wall':
                it.update(x=rw/2, z=.05, y=1.4, wallAnchor=['front','back','left','right'][k%4], placementZone='front_wall')
            elif it['layer'] == 'floor_under':
                it.update(x=rw/2, z=rl/2, y=.01, placementZone='under_group')
            elif cat in ['sofa','bed','wardrobe','cabinet','shelf','tv_stand','desk']:
                it.update(x=rw/2, z=f['depthM']/2+.05, y=0, wallAnchor='front', placementZone='front_wall')
            else:
                cols = max(1, int(max(1, len(products)) ** .5)); row = i // cols; col = i % cols; i += 1
                it.update(x=(col+1)*rw/(cols+1), z=min(rl-f['depthM']/2, (row+1)*rl/(cols+2)), y=0, placementZone='center')
            it['rotationY'] = float((k % 4) * math.pi / 2); items.append(it)
        return items

    def _template_variant(self, room, base_items, variant_idx):
        items = []
        room_type = self._room_type(room)
        door_lines = self._door_lines(room)
        for it in base_items:
            q = dict(it)
            cat = safe(q.get('category'))
            if cat in {'bed', 'sofa', 'counter', 'sink', 'toilet', 'tv'}:
                q['rotationY'] = self._norm_angle(float(q.get('rotationY', 0.0)))
            elif q.get('layer') == 'ceiling':
                q['x'], q['z'] = self._center(room)
                q['y'] = self._room_dims(room)[2] - fp(q)['heightM'] / 2 - 0.12
                q['rotationY'] = 0.0
            elif q.get('layer') == 'wall':
                wall = q.get('wallAnchor') or 'front'
                q['x'], q['z'] = self._wall_position(room, wall, q, 0.5)
                q['y'] = max(float(q.get('y', 1.4)), 1.2)
                q['rotationY'] = self._rotation_from_wall(room, q, wall)
            elif q.get('layer') == 'floor':
                dx = ((variant_idx % 3) - 1) * 0.05
                dz = (((variant_idx // 3) % 3) - 1) * 0.05
                if room_type == 'living_room' and cat in {'coffee_table', 'armchair', 'side_table'}:
                    dx *= 1.4; dz *= 1.4
                elif room_type == 'kitchen' and cat in {'dining_table', 'chair'}:
                    dx *= 1.3; dz *= 1.3
                elif room_type == 'bathroom' and cat in {'cabinet', 'bathtub'}:
                    dx *= 0.7; dz *= 0.7
                q['x'] = float(q.get('x', 0.0)) + dx
                q['z'] = float(q.get('z', 0.0)) + dz
                if q.get('facingTarget') == 'room_center':
                    q['rotationY'] = self._face_target_angle(q, self._center(room))
            elif q.get('layer') == 'wall' and variant_idx > 0:
                q['wallAnchor'] = ['front', 'right', 'back', 'left'][variant_idx % 4]
            if self._door_clearance_penalty(room, q, door_lines) > 0:
                q['x'] = float(q.get('x', 0.0)) + (0.18 if variant_idx % 2 == 0 else -0.18)
                q['z'] = float(q.get('z', 0.0)) + (0.12 if variant_idx % 2 == 0 else -0.12)
            items.append(q)
        return items

    def _door_metadata(self, room):
        openings = room.get('openings') or room.get('doors') or []
        out = []
        for o in openings:
            if not isinstance(o, dict):
                continue
            kind = safe(o.get('type', 'door'))
            if kind not in {'door', 'opening', 'passage'}:
                continue
            out.append({
                'wall': safe(o.get('wall', o.get('wallAnchor', 'front'))),
                'position': float(o.get('position', o.get('center', 0.5)) or 0.5),
                'widthM': float(o.get('widthM', o.get('width', 0.9)) or 0.9),
            })
        return out

    def _avoid_door_zones(self, room, item, door_meta):
        if not door_meta:
            return item
        rw, rl, _ = self._room_dims(room)
        x, z = float(item.get('x', 0.0)), float(item.get('z', 0.0))
        w, d = self._rotated_aabb_size(item)
        for door in door_meta:
            wall = door['wall']
            pos = max(0.05, min(0.95, float(door['position'])))
            width = max(0.4, float(door['widthM']))
            if wall == 'front' and z - d / 2 < 0.42:
                x = x + (width / 2 + w / 2 + 0.25) if x < rw * pos else x - (width / 2 + w / 2 + 0.25)
                z = max(z, d / 2 + 0.55)
            elif wall == 'back' and z + d / 2 > rl - 0.42:
                x = x + (width / 2 + w / 2 + 0.25) if x < rw * pos else x - (width / 2 + w / 2 + 0.25)
                z = min(z, rl - d / 2 - 0.55)
            elif wall == 'left' and x - w / 2 < 0.42:
                z = z + (width / 2 + d / 2 + 0.25) if z < rl * pos else z - (width / 2 + d / 2 + 0.25)
                x = max(x, w / 2 + 0.55)
            elif wall == 'right' and x + w / 2 > rw - 0.42:
                z = z + (width / 2 + d / 2 + 0.25) if z < rl * pos else z - (width / 2 + d / 2 + 0.25)
                x = min(x, rw - w / 2 - 0.55)
        item['x'], item['z'] = clamp(x, w / 2 + 0.05, rw - w / 2 - 0.05), clamp(z, d / 2 + 0.05, rl - d / 2 - 0.05)
        return item

    def _apply_intent_bias(self, room, item, intent, policy):
        cat = safe(item.get('category'))
        bias = intent.get('bias', {})
        room_type = self._room_type(room)
        if cat in {'bed', 'sofa', 'counter', 'sink', 'tv'}:
            item['rotationY'] = self._norm_angle(float(item.get('rotationY', 0.0)) + (0.0 if bias.get('focal', 1.0) >= 1.0 else 0.08))
        if intent['name'] == 'balanced_symmetry':
            if room_type == 'bedroom' and cat == 'nightstand':
                item['z'] = float(item.get('z', 0.0)) + (-0.02 if item.get('placementZone') == 'bedside' else 0.02)
            if room_type in {'living_room', 'livingroom'} and cat in {'armchair', 'chair'}:
                item['x'] = float(item.get('x', 0.0)) + (0.03 if item.get('placementZone') == 'reading_corner' else -0.03)
        elif intent['name'] == 'cozy':
            if cat in {'rug', 'lamp', 'plant', 'chair', 'armchair'}:
                item['x'] = float(item.get('x', 0.0)) + 0.04
                item['z'] = float(item.get('z', 0.0)) - 0.03
        elif intent['name'] == 'minimal':
            if cat in {'rug', 'decor', 'plant', 'lamp'}:
                item['x'] = float(item.get('x', 0.0)) - 0.05
                item['z'] = float(item.get('z', 0.0)) + 0.02
        elif intent['name'] == 'workflow':
            if room_type == 'kitchen' and cat in {'sink', 'stove', 'fridge', 'counter'}:
                item['z'] = float(item.get('z', 0.0)) - 0.02
        return item

    def _make_candidates_from_transformer(self, room, selected, top_k):
        template = self._template_layout(room, selected)
        base = template or self.transformer_layout(room, selected)
        if not base:
            return [self.prior(room, selected, k) for k in range(max(1, top_k))]
        room_type = self._room_type(room)
        door_meta = self._door_metadata(room)
        policy = self._composition_policy(room_type)
        intents = self._intent_variants(room_type, policy)
        candidates = []
        per_intent = max(1, top_k // max(1, len(intents)))
        for intent_idx, intent in enumerate(intents):
            for k in range(per_intent):
                variant_idx = intent_idx * per_intent + k
                cand = self._template_variant(room, base, variant_idx)
                for it in cand:
                    cat = safe(it.get('category'))
                    if room_type in {'living_room', 'livingroom'} and cat == 'coffee_table':
                        it['x'] = float(it.get('x', 0.0)) + ((variant_idx % 2) * 0.04)
                        it['z'] = float(it.get('z', 0.0)) - ((variant_idx % 2) * 0.03)
                    elif room_type == 'kitchen' and cat in {'dining_table', 'chair'}:
                        it['x'] = float(it.get('x', 0.0)) + ((variant_idx % 3) - 1) * 0.08
                        it['z'] = float(it.get('z', 0.0)) + (((variant_idx // 3) % 2) - 0.5) * 0.06
                    elif room_type == 'bathroom' and cat in {'cabinet', 'bathtub', 'mirror'}:
                        it['z'] = float(it.get('z', 0.0)) + ((variant_idx % 2) * 0.03)
                    self._apply_intent_bias(room, it, intent, policy)
                    if self._door_clearance_penalty(room, it, door_meta) > 0:
                        self._avoid_door_zones(room, it, door_meta)
                cand_score_hint = intent['name']
                for it in cand:
                    it['designIntent'] = cand_score_hint
                candidates.append(cand)
        return candidates

    def _relation_score_for_item(self, room, item, items):
        cat = safe(item.get('category'))
        room_type = self._room_type(room)
        score = 0.5
        if room_type == 'bedroom':
            if cat == 'bed' and item.get('wallAnchor') == 'back':
                score += 0.25
            if cat == 'nightstand' and item.get('placementZone') == 'bedside':
                score += 0.2
            if cat == 'rug' and item.get('placementZone') == 'under_bed':
                score += 0.25
            if cat == 'bench' and item.get('placementZone') == 'foot_of_bed':
                score += 0.2
            if cat in {'wardrobe', 'cabinet'} and item.get('wallAnchor') == 'right':
                score += 0.15
        elif room_type in {'living_room', 'livingroom'}:
            if cat == 'sofa' and item.get('wallAnchor') == 'back':
                score += 0.2
            if cat == 'tv' and item.get('wallAnchor') == 'front':
                score += 0.22
            if cat == 'coffee_table' and item.get('placementZone') in {'center', 'side_zone'}:
                score += 0.18
            if cat == 'rug' and item.get('placementZone') == 'under_seating':
                score += 0.22
            if cat in {'armchair', 'chair'} and item.get('placementZone') == 'reading_corner':
                score += 0.16
        elif room_type == 'kitchen':
            if cat == 'counter' and item.get('wallAnchor') == 'front':
                score += 0.22
            if cat in {'sink', 'stove'} and item.get('placementZone') in {'sink_zone', 'cooking_zone'}:
                score += 0.2
            if cat == 'fridge' and item.get('placementZone') == 'entry':
                score += 0.15
            if cat == 'dining_table' and item.get('placementZone') == 'dining_zone':
                score += 0.18
            if cat == 'chair' and item.get('placementZone') == 'dining_zone':
                score += 0.16
        elif room_type == 'bathroom':
            if cat == 'toilet' and item.get('wallAnchor') == 'back':
                score += 0.22
            if cat == 'sink' and item.get('wallAnchor') == 'front':
                score += 0.2
            if cat == 'mirror' and item.get('placementZone') == 'wall':
                score += 0.16
            if cat in {'bathtub', 'shower'} and item.get('placementZone') == 'corner':
                score += 0.18
            if cat in {'cabinet', 'shelf'} and item.get('placementZone') in {'storage', 'side_wall'}:
                score += 0.12
        if item.get('facingTarget') == 'room_center':
            score += 0.05
        if item.get('relations'):
            score += min(0.1, 0.02 * len(item['relations']))
        return round(min(1.0, score), 4)

    def _layout_quality_metrics(self, room, items, base_metrics):
        rw, rl, _ = self._room_dims(room)
        room_type = self._room_type(room)
        door_lines = self._door_lines(room)
        collision_count, outside_count, door_penalty = 0, 0, 0
        floor = [it for it in items if it.get('layer', 'floor') == 'floor']
        item_relation_scores = []
        for it in floor:
            x1, z1, x2, z2 = self._aabb_from_poly(self._rotated_rect(it))
            outside_count += int(x1 < -1e-3 or z1 < -1e-3 or x2 > rw + 1e-3 or z2 > rl + 1e-3)
            door_penalty += int(self._door_clearance_penalty(room, it, door_lines) > 0)
            item_relation_scores.append(self._relation_score_for_item(room, it, items))
        for i in range(len(floor)):
            ax1, az1, ax2, az2 = self._aabb_from_poly(self._rotated_rect(floor[i]))
            for j in range(i + 1, len(floor)):
                bx1, bz1, bx2, bz2 = self._aabb_from_poly(self._rotated_rect(floor[j]))
                collision_count += int(min(ax2, bx2) > max(ax1, bx1) and min(az2, bz2) > max(az1, bz1))

        relation_graph = self._relation_graph(room_type)
        relation_total = max(1, len(relation_graph.get('relations', [])))
        relation_hits = 0
        for rel in relation_graph.get('relations', []):
            a = next((it for it in items if safe(it.get('category')) == safe(rel.get('a'))), None)
            b = next((it for it in items if safe(it.get('category')) == safe(rel.get('b')) or safe(it.get('target')) == safe(rel.get('b'))), None)
            if a and b:
                relation_hits += 1
        layout_relation_score = max(0.0, min(1.0, relation_hits / relation_total - 0.08 * base_metrics.get('penalties', {}).get('wall', 0) - 0.12 * outside_count - 0.1 * door_penalty))
        relation_score = max(layout_relation_score, sum(item_relation_scores) / max(1, len(item_relation_scores)))

        facing_items = [it for it in items if it.get('facingTarget')]
        facing_score = 1.0
        for it in facing_items:
            target = self._center(room) if it.get('facingTarget') == 'room_center' else None
            if target:
                expected = self._face_target_angle(it, target)
                err = abs(self._norm_angle(float(it.get('rotationY', 0.0)) - expected)) / math.pi
                facing_score -= err / max(1, len(facing_items))
        clearance_score = max(0.0, 1.0 - 0.12 * collision_count - 0.08 * outside_count - 0.08 * door_penalty - 0.04 * len(items) / max(rw * rl, 1.0))
        wall_alignment_score = max(0.0, 1.0 - 0.08 * sum(1 for it in items if safe(it.get('category')) in WALL_OBJECTS and not it.get('wallAnchor')))
        symmetry_score = 0.75
        stands = [it for it in items if safe(it.get('category')) == 'nightstand']
        beds = [it for it in items if safe(it.get('category')) == 'bed']
        if len(stands) >= 2 and beds:
            bx = float(beds[0].get('x', 0.0))
            ds = sorted(abs(float(s.get('x', 0.0)) - bx) for s in stands[:2])
            symmetry_score = max(0.0, 1.0 - abs(ds[0] - ds[1]))
        room_balance_score = max(0.0, 1.0 - abs(sum(float(it.get('x', rw / 2)) for it in items) / max(1, len(items)) - rw / 2) / max(rw / 2, 1e-6)) if items else 0.0
        aesthetic_score = max(0.0, min(1.0, 0.22 * relation_score + 0.22 * facing_score + 0.18 * clearance_score + 0.16 * wall_alignment_score + 0.12 * symmetry_score + 0.10 * room_balance_score))
        item_scores = {str(it.get('productId', i)): self._relation_score_for_item(room, it, items) for i, it in enumerate(items)}
        return {
            'collisionCount': collision_count,
            'outsideCount': outside_count,
            'doorPenaltyCount': door_penalty,
            'relationScore': round(relation_score, 4),
            'facingScore': round(max(0.0, min(1.0, facing_score)), 4),
            'clearanceScore': round(clearance_score, 4),
            'wallAlignmentScore': round(wall_alignment_score, 4),
            'symmetryScore': round(symmetry_score, 4),
            'roomBalanceScore': round(room_balance_score, 4),
            'aestheticScore': round(aesthetic_score, 4),
            'relationCount': relation_total,
            'roomType': room_type,
            'itemRelationScores': item_scores,
        }

    def _camera_positions(self, room):
        rw, rl, _ = self._room_dims(room)
        return {
            'front': (rw / 2, -max(rw, rl) * 0.55),
            'back': (rw / 2, rl + max(rw, rl) * 0.55),
            'left': (-max(rw, rl) * 0.55, rl / 2),
            'right': (rw + max(rw, rl) * 0.55, rl / 2),
            'corner': (-max(rw, rl) * 0.35, -max(rw, rl) * 0.35),
        }

    def _visibility_from_camera(self, room, items, camera_name='front'):
        cams = self._camera_positions(room)
        cam = cams.get(camera_name, cams['front'])
        rw, rl, _ = self._room_dims(room)
        visible = 0
        total = 0
        for it in items:
            if it.get('layer') == 'wall':
                continue
            total += 1
            x, z = float(it.get('x', 0.0)), float(it.get('z', 0.0))
            dist = ((x - cam[0]) ** 2 + (z - cam[1]) ** 2) ** 0.5
            center_bias = 1.0 - abs(x - rw / 2) / max(rw / 2, 1e-6)
            depth_bias = 1.0 - abs(z - rl / 2) / max(rl / 2, 1e-6)
            facing_bonus = 0.08 if it.get('facingTarget') in {'room_center', 'tv', 'sofa', 'bed'} else 0.0
            score = max(0.0, min(1.0, 1.0 / (1.0 + dist) + 0.25 * center_bias + 0.25 * depth_bias + facing_bonus))
            visible += score
        return round(visible / max(1, total), 4)

    def _format_item_for_fe(self, item):
        q = dict(item)
        f = fp(q)
        q['footprint'] = f
        q['position'] = {'x': float(q.get('x', 0.0)), 'y': float(q.get('y', 0.0)), 'z': float(q.get('z', 0.0))}
        q['rotationY'] = float(q.get('rotationY', 0.0))
        q['category'] = safe(q.get('category'))
        q['productId'] = self._product_id(q, q.get('category', 'item'))
        q['frontAxis'] = self._front_axis(q['category'])
        q['canAgainstWall'] = q['category'] in WALL_OBJECTS or bool(q.get('wallAnchor'))
        q['needsSupport'] = q['category'] in NEEDS_SUPPORT
        q['supportSurface'] = q['category'] in SUPPORTERS
        q['footprintPolygon'] = [{'x': x, 'z': z} for x, z in self._rotated_rect(q)]
        return q

    def generate_layout(self, payload):
        room = payload.get('room', {})
        products = payload.get('products') or payload.get('items') or []
        opts = payload.get('options', {})
        min_score = float(opts.get('minScore', .35))
        top_k = int(opts.get('topK', 8))
        room_type = self._room_type(room)
        relation_graph = self._relation_graph(room_type)
        composition_policy = self._composition_policy(room_type)
        room_zones = self._room_zones(room)

        scored = self.score_products(room, products)
        selected = [p for p in scored if p.get('keepProbability', 0) >= min_score] or sorted(scored, key=lambda x: x.get('keepProbability', 0), reverse=True)[:min(6, len(scored))]
        set_score = self.score_set(room, selected)

        raw_candidates = self._make_candidates_from_transformer(room, selected, top_k)
        candidates = []
        for raw_items in raw_candidates:
            items, rejected, metrics = fast_solve_layout(room, raw_items)
            hard_failures = self._hard_validate_items(room, items)
            metrics.update(self._layout_quality_metrics(room, items, metrics))
            metrics['setProbability'] = set_score
            metrics['usedLayoutModel'] = bool(self.layout_model is not None)
            metrics['usedRoomTemplate'] = bool(self._template_layout(room, selected))
            metrics['hasDoorMetadata'] = bool(self._door_metadata(room))
            metrics['compositionPolicy'] = composition_policy
            metrics['roomZones'] = room_zones
            metrics['hardFailures'] = hard_failures
            metrics['hardPass'] = not hard_failures
            metrics['hardRejected'] = bool(hard_failures)
            metrics['visibilityFront'] = self._visibility_from_camera(room, items, 'front')
            metrics['visibilityCorner'] = self._visibility_from_camera(room, items, 'corner')
            metrics['cameraVisibilityScore'] = round((metrics['visibilityFront'] + metrics['visibilityCorner']) / 2, 4)
            if needs_heavy_check(items, metrics):
                ok, heavy = heavy_check_layout(room, items)
                metrics['heavyPhysics'] = heavy
                metrics['heavyPassed'] = ok
            else:
                metrics['heavyPhysics'] = {'skipped': True}
                metrics['heavyPassed'] = True
            if hard_failures:
                metrics['layoutScore'] = 0.0
                metrics['hardRejected'] = True
            candidates.append({'items': items, 'rejected': rejected, 'metrics': metrics})

        best = choose_best_layout(candidates)
        best_items = [self._format_item_for_fe(it) for it in best.get('items', [])]
        return {
            'room': room,
            'relationGraph': relation_graph,
            'items': best_items,
            'rejected': best.get('rejected', []),
            'metrics': best.get('metrics', {}),
            'candidateSummary': [
                {
                    'layoutScore': c['metrics'].get('layoutScore'),
                    'itemCount': len(c.get('items', [])),
                    'rejectedCount': len(c.get('rejected', [])),
                    'usedLayoutModel': c['metrics'].get('usedLayoutModel', False),
                } for c in candidates
            ],
            'summary': {
                'inputProducts': len(products),
                'selectedProducts': len(selected),
                'setProbability': set_score,
                'runtime': 'HybridInteriorRuntimeV2',
                'usedLayoutModel': bool(self.layout_model is not None),
                'compositionPolicy': composition_policy,
                'roomZones': room_zones,
            }
        }
