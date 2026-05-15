"""
inference.py – Trained LayoutTransformer model loading & position prediction.

This module ONLY handles:
  1) Loading the trained decor_transformer.pt model and product_filter.joblib
  2) Providing predict_positions(room, products) for the layout engine
  3) Providing score_be_candidates(payload) for product filtering

The actual layout orchestration (template generation, Shapely scoring, etc.)
is handled by layout_engine/engine.py which calls these functions.
"""

import re, json, math, os
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn

# ============================================================================
# Paths & device
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
PRODUCT_FILTER_PATH = MODELS_DIR / 'product_filter.joblib'
CATEGORY_PRIORS_PATH = MODELS_DIR / 'category_priors.csv'
CATEGORY_DIM_MEDIANS_PATH = MODELS_DIR / 'category_dim_medians.csv'
FILTER_CONFIG_PATH = MODELS_DIR / 'product_filter_config.json'
DECOR_MODEL_PATH = MODELS_DIR / 'decor_transformer.pt'
DECOR_VOCAB_PATH = MODELS_DIR / 'decor_vocab.json'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass


# ============================================================================
# Text canonicalization
# ============================================================================

def canonical_text(x):
    if x is None: return ""
    x = str(x).strip().lower()
    x = re.sub(r"[_\-]+", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x

ROOM_TYPE_ALIAS = {
    'living room': 'living_room', 'livingroom': 'living_room', 'phòng khách': 'living_room',
    'bed room': 'bedroom', 'master bedroom': 'bedroom', 'kids room': 'bedroom', 'phòng ngủ': 'bedroom',
    'dining room': 'dining_room', 'phòng ăn': 'dining_room',
    'study room': 'office', 'office': 'office', 'phòng làm việc': 'office',
    'kitchen': 'kitchen', 'phòng bếp': 'kitchen',
    'bathroom': 'bathroom', 'phòng tắm': 'bathroom',
}

CATEGORY_ALIAS = {
    'sofa': 'sofa', 'sectional sofa': 'sofa', 'loveseat': 'sofa', 'couch': 'sofa',
    'coffee table': 'coffee_table', 'tea table': 'coffee_table',
    'side table': 'side_table', 'end table': 'side_table',
    'nightstand': 'nightstand', 'bedside table': 'nightstand',
    'tv stand': 'tv_stand', 'media console': 'tv_stand',
    'armchair': 'armchair', 'lounge chair': 'armchair', 'recliner': 'armchair',
    'chair': 'chair', 'dining chair': 'dining_chair', 'office chair': 'office_chair',
    'desk': 'desk', 'dining table': 'dining_table', 'table': 'table',
    'bed': 'bed', 'wardrobe': 'wardrobe', 'closet': 'wardrobe',
    'cabinet': 'cabinet', 'bookshelf': 'bookshelf', 'shelf': 'bookshelf',
    'drawer': 'drawer', 'dresser': 'dresser',
    'rug': 'rug', 'carpet': 'rug',
    'lamp': 'lamp', 'floor lamp': 'floor_lamp', 'ceiling lamp': 'ceiling_lamp',
    'plant': 'plant', 'mirror': 'mirror',
    'painting': 'wall_art', 'picture': 'wall_art', 'wall art': 'wall_art', 'wall decor': 'wall_art',
    'stool': 'stool', 'bench': 'bench',
    # Vietnamese aliases
    'ghế sofa': 'sofa', 'ghe sofa': 'sofa',
    'bàn nước': 'coffee_table', 'ban nuoc': 'coffee_table', 'bàn trà': 'coffee_table', 'ban tra': 'coffee_table',
    'ghế thư giãn': 'armchair', 'ghe thu gian': 'armchair', 'ghế đơn': 'armchair', 'ghe don': 'armchair',
    'ghế': 'chair', 'bàn đầu giường': 'nightstand', 'ban dau giuong': 'nightstand',
    'tủ tivi': 'tv_stand', 'tu tivi': 'tv_stand', 'kệ tivi': 'tv_stand', 'ke tivi': 'tv_stand',
    'kệ tv': 'tv_stand', 'ke tv': 'tv_stand',
    'bàn ăn': 'dining_table', 'ban an': 'dining_table',
    'ghế ăn': 'dining_chair', 'ghe an': 'dining_chair',
    'bàn làm việc': 'desk', 'ban lam viec': 'desk',
    'ghế văn phòng': 'office_chair', 'ghe van phong': 'office_chair',
    'giường': 'bed', 'giuong': 'bed',
    'tủ quần áo': 'wardrobe', 'tu quan ao': 'wardrobe',
    'thảm': 'rug', 'tham': 'rug', 'đèn': 'lamp', 'den': 'lamp',
    'tranh': 'wall_art', 'kệ sách': 'bookshelf', 'ke sach': 'bookshelf',
    'tủ': 'cabinet', 'tu': 'cabinet',
    'cây trang trí': 'plant', 'cay trang tri': 'plant',
    'gương': 'mirror', 'guong': 'mirror',
    'bàn bên': 'side_table', 'ban ben': 'side_table',
    'bàn cạnh': 'side_table', 'ban canh': 'side_table',
    'bàn sofa': 'coffee_table', 'ban sofa': 'coffee_table',
    'bàn console': 'side_table', 'ban console': 'side_table',
    'console': 'side_table', 'console table': 'side_table',
    'kệ phòng khách': 'bookshelf', 'ke phong khach': 'bookshelf',
    'kệ hangar': 'bookshelf', 'ke hangar': 'bookshelf',
    'kệ': 'bookshelf', 'ke': 'bookshelf',
    'đèn trang trí': 'lamp', 'den trang tri': 'lamp',
    'đèn bàn': 'lamp', 'den ban': 'lamp',
}


def canonical_room_type(x):
    x = canonical_text(x)
    if x in ROOM_TYPE_ALIAS: return ROOM_TYPE_ALIAS[x]
    for k, v in ROOM_TYPE_ALIAS.items():
        if k in x: return v
    return x if x else "unknown"


def canonical_category(x):
    x = canonical_text(x)
    if x in CATEGORY_ALIAS: return CATEGORY_ALIAS[x]
    for k, v in CATEGORY_ALIAS.items():
        if k in x: return v
    return x if x else "unknown"


def _safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default if default is None else float(default)
        v = float(value)
        if not math.isfinite(v):
            return default if default is None else float(default)
        return v
    except Exception:
        return default if default is None else float(default)


def cm_to_m(x):
    """Accept both cm and m. Values > 10 are treated as cm."""
    v = _safe_float(x, 0.0)
    if v <= 0:
        return 0.0
    return v / 100.0 if v > 10 else v


# ============================================================================
# LayoutTransformer – trained model architecture (DO NOT MODIFY)
# ============================================================================

class LayoutTransformer(nn.Module):
    def __init__(self, num_categories, num_room_types, item_num_dim=4, room_num_dim=5,
                 cat_emb_dim=48, room_emb_dim=16, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.cat_emb = nn.Embedding(num_categories + 1, cat_emb_dim, padding_idx=0)
        self.room_emb = nn.Embedding(num_room_types + 1, room_emb_dim, padding_idx=0)
        in_dim = cat_emb_dim + room_emb_dim + item_num_dim + room_num_dim
        self.input_mlp = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.ReLU()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pose_head = nn.Linear(d_model, 4)
        self.anchor_head = nn.Linear(d_model, 2)

    def forward(self, room_type_id, room_num, cat_ids, item_num, mask):
        B, T = cat_ids.shape
        cat_e = self.cat_emb(cat_ids)
        room_e = self.room_emb(room_type_id).unsqueeze(1).repeat(1, T, 1)
        room_num_rep = room_num.unsqueeze(1).repeat(1, T, 1)
        x = torch.cat([cat_e, room_e, item_num, room_num_rep], dim=-1)
        x = self.input_mlp(x)
        h = self.encoder(x, src_key_padding_mask=~mask)
        pose_raw = self.pose_head(h)
        anchor_logits = self.anchor_head(h)
        xz = torch.sigmoid(pose_raw[:, :, :2])
        rot = torch.tanh(pose_raw[:, :, 2:])
        return torch.cat([xz, rot], dim=-1), anchor_logits


# ============================================================================
# Load trained model + filter
# ============================================================================

filter_config = json.load(open(FILTER_CONFIG_PATH, 'r', encoding='utf-8'))
category_priors_df = pd.read_csv(CATEGORY_PRIORS_PATH)
decor_vocab = json.load(open(DECOR_VOCAB_PATH, 'r', encoding='utf-8'))

room_cat_prob_map = {
    (str(r['room_type']), str(r['category'])): float(r['room_cat_prob'])
    for r in category_priors_df.to_dict('records')
}

# Product filter model
try:
    loaded_filter = joblib.load(PRODUCT_FILTER_PATH)
    FILTER_LOAD_ERROR = ""
except Exception as _filter_error:
    loaded_filter = None
    FILTER_LOAD_ERROR = str(_filter_error)

# LayoutTransformer model
cfg = decor_vocab['model_config']
decor_model = LayoutTransformer(
    num_categories=len(decor_vocab['cat2id']),
    num_room_types=len(decor_vocab['room2id']),
    item_num_dim=cfg['item_num_dim'],
    room_num_dim=cfg['room_num_dim'],
    cat_emb_dim=cfg['cat_emb_dim'],
    room_emb_dim=cfg['room_emb_dim'],
    d_model=cfg['d_model'],
    nhead=cfg['nhead'],
    num_layers=cfg['num_layers'],
    dropout=cfg['dropout'],
).to(device)

ckpt = torch.load(DECOR_MODEL_PATH, map_location=device)
if isinstance(ckpt, dict):
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
else:
    state_dict = ckpt

expected = decor_model.state_dict()
missing_keys = [k for k in expected.keys() if k not in state_dict]
unexpected_keys = [k for k in state_dict.keys() if k not in expected]
shape_mismatch = [
    k for k in expected.keys()
    if k in state_dict and tuple(state_dict[k].shape) != tuple(expected[k].shape)
]
if missing_keys or unexpected_keys or shape_mismatch:
    raise RuntimeError(
        'Incompatible decor_transformer.pt. '
        f'missing={missing_keys[:5]}, '
        f'unexpected={unexpected_keys[:5]}, '
        f'shape_mismatch={shape_mismatch[:5]}'
    )
decor_model.load_state_dict(state_dict)
decor_model.eval()

for name, param in decor_model.named_parameters():
    if not torch.isfinite(param).all():
        raise RuntimeError(f'decor_transformer.pt contains NaN/Inf parameter: {name}')

MODEL_LOADED = True
print(f"[inference] LayoutTransformer loaded on {device}, filter={'OK' if loaded_filter else 'FALLBACK'}")


# ============================================================================
# Product filtering & scoring
# ============================================================================

def _first_present(mapping, keys, default=None):
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping and mapping.get(key) not in (None, ""):
            return mapping.get(key)
    return default


def get_item_dims(p):
    """Extract (w, d, h) in meters from a product dict, handling cm/m ambiguity."""
    if all(k in p for k in ("item_width_m", "item_depth_m", "item_height_m")):
        return _safe_float(p["item_width_m"]), _safe_float(p["item_depth_m"]), _safe_float(p["item_height_m"])
    dims = p.get("dimensions") or {}
    w = cm_to_m(_first_present(p, ["widthM", "width_m", "width"],
                _first_present(dims, ["widthM", "width_m", "width", "w"], 0)))
    d = cm_to_m(_first_present(p, ["depthM", "depth_m", "depth", "lengthM", "length_m"],
                _first_present(dims, ["depthM", "depth_m", "depth", "lengthM", "length_m", "length", "d"], 0)))
    h = cm_to_m(_first_present(p, ["heightM", "height_m", "height"],
                _first_present(dims, ["heightM", "height_m", "height", "h"], 0)))
    return w, d, h


def make_filter_feature_row(room_type, style_hint, room_w, room_l, room_h, category, item_w, item_d, item_h):
    room_area = max(room_w * room_l, 1e-6)
    item_area = max(item_w * item_d, 1e-6)
    item_vol = max(item_w * item_d * item_h, 1e-6)
    return {
        "room_type": room_type,
        "style_hint": style_hint if isinstance(style_hint, str) else "",
        "room_width_m": float(room_w), "room_length_m": float(room_l),
        "room_height_m": float(room_h), "room_area_m2": float(room_area),
        "room_aspect_ratio": float(room_w / max(room_l, 1e-6)),
        "category": category,
        "item_width_m": float(item_w), "item_depth_m": float(item_d), "item_height_m": float(item_h),
        "item_area_m2": float(item_area), "item_volume_m3": float(item_vol),
        "width_ratio": float(item_w / max(room_w, 1e-6)),
        "depth_ratio": float(item_d / max(room_l, 1e-6)),
        "height_ratio": float(item_h / max(room_h, 1e-6)),
        "area_ratio": float(item_area / room_area),
        "fit_margin_w": float(room_w - item_w),
        "fit_margin_d": float(room_l - item_d),
        "fit_margin_h": float(room_h - item_h),
        "min_fit_margin": float(min(room_w - item_w, room_l - item_d, room_h - item_h)),
        "room_cat_prior": float(room_cat_prob_map.get((room_type, category), 0.0)),
    }


def build_infer_rows_from_be_payload(payload):
    room = payload["room"]
    room_type = canonical_room_type(room.get("type", "unknown"))
    style_hint = str(room.get("style", "") or "")
    room_w = float(room.get("widthM", 4.0))
    room_l = float(room.get("lengthM", 5.0))
    room_h = float(room.get("heightM", 3.0))
    rows = []
    for p in payload.get("recommendation", {}).get("products", []):
        category = canonical_category(p.get("category", "unknown"))
        item_w, item_d, item_h = get_item_dims(p)
        row = make_filter_feature_row(room_type, style_hint, room_w, room_l, room_h, category, item_w, item_d, item_h)
        row["product_id"] = p.get("id") or p.get("productId")
        row["name"] = p.get("name", "")
        row["raw_category"] = p.get("category", "")
        row["ranking_score"] = float(p.get("ranking_score", p.get("score", 0.0)) or 0.0)
        row["style_score"] = float(p.get("style_score", 0.0) or 0.0)
        row["color_score"] = float(p.get("color_score", 0.0) or 0.0)
        row["source_reasoning"] = p.get("reasoning", "")
        row["raw"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def score_be_candidates(payload):
    """Score candidate products using the trained filter model or heuristic fallback."""
    infer_df = build_infer_rows_from_be_payload(payload)
    if len(infer_df) == 0:
        return infer_df
    if loaded_filter is not None:
        try:
            X_infer = infer_df[filter_config["cat_cols"] + filter_config["num_cols"]].copy()
            infer_df["keep_probability"] = loaded_filter.predict_proba(X_infer)[:, 1]
        except Exception:
            infer_df["keep_probability"] = 0.65
    else:
        room_area = (infer_df["room_width_m"] * infer_df["room_length_m"]).clip(lower=1e-6)
        fit = (infer_df["min_fit_margin"] >= 0).astype(float)
        area_ok = (infer_df["item_area_m2"] / room_area <= 0.35).astype(float)
        prior = infer_df.get("room_cat_prior", 0.0)
        infer_df["keep_probability"] = (0.35 + 0.25 * fit + 0.20 * area_ok + 0.20 * prior).clip(0.05, 0.98)

    infer_df["final_score"] = (
        0.60 * infer_df["keep_probability"]
        + 0.20 * infer_df.get("ranking_score", 0.0)
        + 0.10 * infer_df.get("style_score", 0.0)
        + 0.10 * infer_df.get("color_score", 0.0)
    )
    return infer_df.sort_values("final_score", ascending=False).reset_index(drop=True)


# ============================================================================
# LayoutTransformer position prediction – PUBLIC API
# ============================================================================

def predict_positions(room_dict, products):
    """
    Use the trained LayoutTransformer to predict (x, z, rotation, anchor)
    for each product in the given room.

    Parameters
    ----------
    room_dict : dict with keys widthM, lengthM, heightM, type
    products : list of dicts with keys: category, widthM/dimensions, ...

    Returns
    -------
    list of dicts with keys: product_id, pred_x_m, pred_z_m, pred_rotation_y_deg, pred_anchor
    """
    cat2id = decor_vocab["cat2id"]
    room2id = decor_vocab["room2id"]
    id2anchor = {int(k): v for k, v in decor_vocab["id2anchor"].items()}
    max_items = int(decor_vocab["max_items"])
    room_num_mean = np.array(decor_vocab["room_num_mean"], dtype=np.float32)
    room_num_std = np.array(decor_vocab["room_num_std"], dtype=np.float32)
    item_num_mean = np.array(decor_vocab["item_num_mean"], dtype=np.float32)
    item_num_std = np.array(decor_vocab["item_num_std"], dtype=np.float32)

    room_type = canonical_room_type(room_dict.get("type", "unknown"))
    room_w = float(room_dict.get("widthM", 4.0))
    room_l = float(room_dict.get("lengthM", 5.0))
    room_h = float(room_dict.get("heightM", 3.0))
    room_num_raw = np.array([room_w, room_l, room_h, room_w * room_l, room_w / max(room_l, 1e-6)], dtype=np.float32)
    room_num = (room_num_raw - room_num_mean) / room_num_std

    enriched = []
    for idx, p in enumerate(products):
        category = canonical_category(p.get("category", "unknown"))
        item_w, item_d, item_h = get_item_dims(p)
        enriched.append({
            "idx": idx, "product_id": p.get("id") or p.get("productId") or f"p-{idx}",
            "category": category, "w": item_w, "d": item_d, "h": item_h,
            "area_ratio": (item_w * item_d) / max(room_w * room_l, 1e-6),
        })
    enriched = sorted(enriched, key=lambda x: (x["w"] * x["d"], x["h"]), reverse=True)[:max_items]

    cat_ids = np.zeros((max_items,), dtype=np.int64)
    item_num = np.zeros((max_items, 4), dtype=np.float32)
    mask = np.zeros((max_items,), dtype=np.bool_)
    for i, p in enumerate(enriched):
        cat_ids[i] = int(cat2id.get(p["category"], 0))
        item_raw = np.array([
            p["w"] / max(room_w, 1e-6), p["d"] / max(room_l, 1e-6),
            p["h"] / max(room_h, 1e-6), p["area_ratio"]
        ], dtype=np.float32)
        item_num[i] = (item_raw - item_num_mean) / item_num_std
        mask[i] = True

    room_type_id = torch.tensor([room2id.get(room_type, 0)], dtype=torch.long).to(device)
    room_num_t = torch.tensor([room_num], dtype=torch.float32).to(device)
    cat_ids_t = torch.tensor([cat_ids], dtype=torch.long).to(device)
    item_num_t = torch.tensor([item_num], dtype=torch.float32).to(device)
    mask_t = torch.tensor([mask], dtype=torch.bool).to(device)

    with torch.inference_mode():
        pose_pred, anchor_logits = decor_model(room_type_id, room_num_t, cat_ids_t, item_num_t, mask_t)

    pose_pred = pose_pred[0].cpu().numpy()
    anchor_pred = anchor_logits[0].argmax(dim=-1).cpu().numpy()

    results = []
    for i, p in enumerate(enriched):
        x_n, z_n = float(pose_pred[i, 0]), float(pose_pred[i, 1])
        rot_sin, rot_cos = float(pose_pred[i, 2]), float(pose_pred[i, 3])
        x_m = (x_n - 0.5) * room_w
        z_m = (z_n - 0.5) * room_l
        rot_deg = math.degrees(math.atan2(rot_sin, rot_cos))
        results.append({
            "product_id": p["product_id"],
            "category": p["category"],
            "pred_x_m": float(x_m),
            "pred_z_m": float(z_m),
            "pred_rotation_y_deg": float(rot_deg),
            "pred_anchor": id2anchor.get(int(anchor_pred[i]), "floating"),
            "original_index": p["idx"],
        })
    return sorted(results, key=lambda x: x["original_index"])


@lru_cache(maxsize=1)
def pybullet_probe():
    try:
        import pybullet as p
        cid = p.connect(p.DIRECT)
        p.disconnect(cid)
        return True, ""
    except Exception as e:
        return False, str(e)


def model_info():
    """Return metadata about the loaded models."""
    return {
        "layoutModel": "decor_transformer.pt",
        "filterModel": "product_filter.joblib",
        "filterLoaded": loaded_filter is not None,
        "filterLoadError": FILTER_LOAD_ERROR,
        "device": device,
        "vocabCategories": list(decor_vocab["cat2id"].keys()),
        "vocabRoomTypes": list(decor_vocab["room2id"].keys()),
        "maxItems": decor_vocab["max_items"],
    }
