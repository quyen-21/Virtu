
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

CATEGORIES = ['sofa','chair','armchair','coffee_table','dining_table','desk','bed','nightstand','wardrobe','cabinet','shelf','tv_stand','tv','rug','lamp','plant','vase','book','decor','wall_art','mirror','door','window','curtain','sink','toilet','bathtub','counter','unknown']
SUPPORTERS = {'coffee_table','dining_table','desk','nightstand','cabinet','shelf','tv_stand','counter'}
NEEDS_SUPPORT = {'tv','vase','book','decor','lamp','plant'}
WALL = {'wall_art','mirror','curtain','window','door'}
UNDER = {'rug'}
LAYER = {c:'floor' for c in CATEGORIES}
for c in ['tv','vase','book','decor','lamp','plant']: LAYER[c] = 'top_surface'
for c in WALL: LAYER[c] = 'wall'
for c in UNDER: LAYER[c] = 'floor_under'

def safe(x): return str(x or 'unknown').strip().lower()

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
        cat = safe(p.get('category', 'unknown'))
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
            item['category'] = safe(p.get('category', 'unknown'))
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
            cat = safe(p.get('category')); f = fp(p); it = dict(p); it['category'] = cat; it['footprint'] = f; it['productId'] = str(p.get('productId', p.get('id', f'item_{len(items)}'))); it['layer'] = LAYER.get(cat, 'floor')
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

    def _make_candidates_from_transformer(self, room, selected, top_k):
        base = self.transformer_layout(room, selected)
        if not base:
            return [self.prior(room, selected, k) for k in range(max(1, top_k))]
        candidates = []
        for k in range(max(1, top_k)):
            items = []
            for idx, it in enumerate(base):
                q = dict(it)
                if q.get('layer') == 'floor':
                    # small deterministic jitter for diverse candidates
                    dx = ((k % 3) - 1) * 0.08
                    dz = (((k // 3) % 3) - 1) * 0.08
                    q['x'] = float(q['x']) + dx
                    q['z'] = float(q['z']) + dz
                elif q.get('layer') == 'wall' and k > 0:
                    q['wallAnchor'] = ['front','right','back','left'][k % 4]
                q['rotationY'] = float(q.get('rotationY', 0.0) + (0.0 if q.get('layer') == 'wall' else (k % 4) * math.pi / 8))
                items.append(q)
            candidates.append(items)
        return candidates

    def generate_layout(self, payload):
        room = payload.get('room', {})
        products = payload.get('products') or payload.get('items') or []
        opts = payload.get('options', {})
        min_score = float(opts.get('minScore', .35))
        top_k = int(opts.get('topK', 8))

        scored = self.score_products(room, products)
        selected = [p for p in scored if p.get('keepProbability', 0) >= min_score] or sorted(scored, key=lambda x: x.get('keepProbability', 0), reverse=True)[:min(6, len(scored))]
        set_score = self.score_set(room, selected)

        raw_candidates = self._make_candidates_from_transformer(room, selected, top_k)
        candidates = []
        for raw_items in raw_candidates:
            items, rejected, metrics = fast_solve_layout(room, raw_items)
            metrics['setProbability'] = set_score
            metrics['usedLayoutModel'] = bool(self.layout_model is not None)
            if needs_heavy_check(items, metrics):
                ok, heavy = heavy_check_layout(room, items)
                metrics['heavyPhysics'] = heavy
                metrics['heavyPassed'] = ok
            else:
                metrics['heavyPhysics'] = {'skipped': True}
                metrics['heavyPassed'] = True
            candidates.append({'items': items, 'rejected': rejected, 'metrics': metrics})

        best = choose_best_layout(candidates)
        return {
            'room': room,
            'items': best.get('items', []),
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
            }
        }
