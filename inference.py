import re, json, math, os
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn

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

def canonical_text(x):
    if x is None: return ""
    x = str(x).strip().lower()
    x = re.sub(r"[_\-]+", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x

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

def cm_to_m(x):
    try: return float(x) / 100.0
    except: return 0.0

class LayoutTransformer(nn.Module):
    def __init__(self, num_categories, num_room_types, item_num_dim=4, room_num_dim=5, cat_emb_dim=48, room_emb_dim=16, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.cat_emb = nn.Embedding(num_categories + 1, cat_emb_dim, padding_idx=0)
        self.room_emb = nn.Embedding(num_room_types + 1, room_emb_dim, padding_idx=0)
        in_dim = cat_emb_dim + room_emb_dim + item_num_dim + room_num_dim
        self.input_mlp = nn.Sequential(nn.Linear(in_dim, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.ReLU())
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True, activation="gelu")
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

def make_filter_feature_row(room_type, style_hint, room_w, room_l, room_h, category, item_w, item_d, item_h):
    room_area = max(room_w * room_l, 1e-6)
    item_area = max(item_w * item_d, 1e-6)
    item_vol = max(item_w * item_d * item_h, 1e-6)
    width_ratio = item_w / max(room_w, 1e-6)
    depth_ratio = item_d / max(room_l, 1e-6)
    height_ratio = item_h / max(room_h, 1e-6)
    area_ratio = item_area / room_area
    room_aspect = room_w / max(room_l, 1e-6)
    fit_margin_w = room_w - item_w
    fit_margin_d = room_l - item_d
    fit_margin_h = room_h - item_h
    min_fit_margin = min(fit_margin_w, fit_margin_d, fit_margin_h)
    room_cat_prior = room_cat_prob_map.get((room_type, category), 0.0)
    return {
        "room_type": room_type, "style_hint": style_hint if isinstance(style_hint, str) else "", "room_width_m": float(room_w),
        "room_length_m": float(room_l), "room_height_m": float(room_h), "room_area_m2": float(room_area),
        "room_aspect_ratio": float(room_aspect), "category": category, "item_width_m": float(item_w),
        "item_depth_m": float(item_d), "item_height_m": float(item_h), "item_area_m2": float(item_area),
        "item_volume_m3": float(item_vol), "width_ratio": float(width_ratio), "depth_ratio": float(depth_ratio),
        "height_ratio": float(height_ratio), "area_ratio": float(area_ratio), "fit_margin_w": float(fit_margin_w),
        "fit_margin_d": float(fit_margin_d), "fit_margin_h": float(fit_margin_h), "min_fit_margin": float(min_fit_margin),
        "room_cat_prior": float(room_cat_prior),
    }

def get_item_dims_from_payload_product(p):
    if "item_width_m" in p and "item_depth_m" in p and "item_height_m" in p:
        return float(p["item_width_m"]), float(p["item_depth_m"]), float(p["item_height_m"])
    dims = p.get("dimensions", {})
    return cm_to_m(dims.get("width", 0)), cm_to_m(dims.get("depth", 0)), cm_to_m(dims.get("height", 0))

def build_infer_rows_from_be_payload(payload):
    room = payload["room"]
    room_type = canonical_room_type(room.get("type", "unknown"))
    style_hint = str(room.get("style", "") or "")
    room_w, room_l, room_h = float(room["widthM"]), float(room["lengthM"]), float(room["heightM"])
    rows = []
    for p in payload.get("recommendation", {}).get("products", []):
        category = canonical_category(p.get("category", "unknown"))
        item_w, item_d, item_h = get_item_dims_from_payload_product(p)
        row = make_filter_feature_row(room_type, style_hint, room_w, room_l, room_h, category, item_w, item_d, item_h)
        row["product_id"] = p.get("id")
        row["name"] = p.get("name", "")
        row["raw_category"] = p.get("category", "")
        row["ranking_score"] = float(p.get("ranking_score", 0.0))
        row["style_score"] = float(p.get("style_score", 0.0))
        row["color_score"] = float(p.get("color_score", 0.0))
        row["source_reasoning"] = p.get("reasoning", "")
        row["raw"] = p
        rows.append(row)
    return pd.DataFrame(rows)

def score_be_candidates(payload):
    infer_df = build_infer_rows_from_be_payload(payload)
    if len(infer_df) == 0: return infer_df
    X_infer = infer_df[filter_config["cat_cols"] + filter_config["num_cols"]].copy()
    infer_df["keep_probability"] = loaded_filter.predict_proba(X_infer)[:, 1]
    infer_df["final_score"] = 0.60 * infer_df["keep_probability"] + 0.20 * infer_df["ranking_score"] + 0.10 * infer_df["style_score"] + 0.10 * infer_df["color_score"]
    return infer_df.sort_values("final_score", ascending=False).reset_index(drop=True)

def select_products_for_layout(payload, top_k=8, threshold=None):
    if threshold is None: threshold = float(filter_config.get("recommended_threshold", 0.5))
    scored = score_be_candidates(payload)
    if len(scored) == 0: return [], []
    selected, rejected, used_categories = [], [], set()
    for _, row in scored.iterrows():
        rec = row.to_dict()
        if float(rec["keep_probability"]) < float(threshold):
            rejected.append({"productId": rec["product_id"], "name": rec["name"], "category": rec["category"], "reason": "low_keep_probability", "keepProbability": float(rec["keep_probability"])})
            continue
        if rec["category"] not in used_categories or len(selected) < min(3, top_k):
            selected.append(rec); used_categories.add(rec["category"])
        elif len(selected) < top_k:
            selected.append(rec)
        else:
            rejected.append({"productId": rec["product_id"], "name": rec["name"], "category": rec["category"], "reason": "beyond_top_k", "keepProbability": float(rec["keep_probability"])})
        if len(selected) >= top_k: break
    selected_ids = {x["product_id"] for x in selected}
    for _, row in scored.iterrows():
        if row["product_id"] not in selected_ids and not any(r["productId"] == row["product_id"] for r in rejected):
            rejected.append({"productId": row["product_id"], "name": row["name"], "category": row["category"], "reason": "not_selected", "keepProbability": float(row["keep_probability"])})
    return selected, rejected

def predict_layout_for_selected(room, selected_products):
    cat2id = decor_vocab["cat2id"]
    room2id = decor_vocab["room2id"]
    id2anchor = {int(k): v for k, v in decor_vocab["id2anchor"].items()}
    max_items = int(decor_vocab["max_items"])
    room_num_mean = np.array(decor_vocab["room_num_mean"], dtype=np.float32)
    room_num_std = np.array(decor_vocab["room_num_std"], dtype=np.float32)
    item_num_mean = np.array(decor_vocab["item_num_mean"], dtype=np.float32)
    item_num_std = np.array(decor_vocab["item_num_std"], dtype=np.float32)

    room_type = canonical_room_type(room.get("type", "unknown"))
    room_w, room_l, room_h = float(room["widthM"]), float(room["lengthM"]), float(room["heightM"])
    room_num_raw = np.array([room_w, room_l, room_h, room_w * room_l, room_w / max(room_l, 1e-6)], dtype=np.float32)
    room_num = (room_num_raw - room_num_mean) / room_num_std

    enriched = []
    for idx, p in enumerate(selected_products):
        category = canonical_category(p.get("category", "unknown"))
        item_w, item_d, item_h = get_item_dims_from_payload_product(p)
        enriched.append({**p, "_orig_idx": idx, "_category_norm": category, "_w_m": item_w, "_d_m": item_d, "_h_m": item_h, "_area_ratio": (item_w * item_d) / max(room_w * room_l, 1e-6)})
    enriched = sorted(enriched, key=lambda x: (x["_w_m"] * x["_d_m"], x["_h_m"]), reverse=True)[:max_items]

    cat_ids = np.zeros((max_items,), dtype=np.int64)
    item_num = np.zeros((max_items, 4), dtype=np.float32)
    mask = np.zeros((max_items,), dtype=np.bool_)
    for i, p in enumerate(enriched):
        cat_ids[i] = int(cat2id.get(p["_category_norm"], 0))
        item_raw = np.array([p["_w_m"] / max(room_w, 1e-6), p["_d_m"] / max(room_l, 1e-6), p["_h_m"] / max(room_h, 1e-6), p["_area_ratio"]], dtype=np.float32)
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

    out_items = []
    for i, p in enumerate(enriched):
        x_n, z_n = float(pose_pred[i, 0]), float(pose_pred[i, 1])
        rot_sin, rot_cos = float(pose_pred[i, 2]), float(pose_pred[i, 3])
        x_m = (x_n - 0.5) * room_w
        z_m = (z_n - 0.5) * room_l
        rot_deg = math.degrees(math.atan2(rot_sin, rot_cos))
        out_items.append({**p, "pred_x_m": float(x_m), "pred_z_m": float(z_m), "pred_rotation_y_deg": float(rot_deg), "pred_anchor": id2anchor.get(int(anchor_pred[i]), "floating")})
    return sorted(out_items, key=lambda x: x["_orig_idx"])

def item_bbox_2d(item):
    hw, hd = item["item_width_m"] / 2.0, item["item_depth_m"] / 2.0
    return item["x_m"] - hw, item["x_m"] + hw, item["z_m"] - hd, item["z_m"] + hd

def overlap_area_2d(a, b):
    ax1, ax2, az1, az2 = item_bbox_2d(a)
    bx1, bx2, bz1, bz2 = item_bbox_2d(b)
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(az2, bz2) - max(az1, bz1))

def is_overlapping(a, b, eps=1e-6):
    return overlap_area_2d(a, b) > eps

def clamp_inside_room(item, room):
    hw, hd = item["item_width_m"] / 2.0, item["item_depth_m"] / 2.0
    x_min, x_max = -room["widthM"] / 2.0 + hw, room["widthM"] / 2.0 - hw
    z_min, z_max = -room["lengthM"] / 2.0 + hd, room["lengthM"] / 2.0 - hd
    old_x, old_z = item["x_m"], item["z_m"]
    item["x_m"] = min(max(item["x_m"], x_min), x_max)
    item["z_m"] = min(max(item["z_m"], z_min), z_max)
    return int(abs(item["x_m"] - old_x) > 1e-9) + int(abs(item["z_m"] - old_z) > 1e-9)

def nearest_wall_side(item, room):
    hw, hd = item["item_width_m"] / 2.0, item["item_depth_m"] / 2.0
    gaps = {
        "left": (item["x_m"] - hw) - (-room["widthM"] / 2.0),
        "right": (room["widthM"] / 2.0) - (item["x_m"] + hw),
        "back": (item["z_m"] - hd) - (-room["lengthM"] / 2.0),
        "front": (room["lengthM"] / 2.0) - (item["z_m"] + hd),
    }
    return min(gaps, key=gaps.get)

def snap_to_wall(item, room, wall=None, margin=0.08):
    if wall is None: wall = nearest_wall_side(item, room)
    hw, hd = item["item_width_m"] / 2.0, item["item_depth_m"] / 2.0
    if wall == "left":
        item["x_m"] = -room["widthM"] / 2.0 + hw + margin; item["rotation_y_deg"] = 90.0
    elif wall == "right":
        item["x_m"] = room["widthM"] / 2.0 - hw - margin; item["rotation_y_deg"] = 90.0
    elif wall == "back":
        item["z_m"] = -room["lengthM"] / 2.0 + hd + margin; item["rotation_y_deg"] = 0.0
    elif wall == "front":
        item["z_m"] = room["lengthM"] / 2.0 - hd - margin; item["rotation_y_deg"] = 0.0

def resolve_overlaps(items, room, max_iter=40):
    collision_fixes = 0
    for _ in range(max_iter):
        changed = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a.get("layer", "floor") != "floor" or b.get("layer", "floor") != "floor": continue
                if not is_overlapping(a, b): continue
                changed, collision_fixes = True, collision_fixes + 1
                dx, dz = b["x_m"] - a["x_m"], b["z_m"] - a["z_m"]
                if abs(dx) >= abs(dz):
                    shift = ((a["item_width_m"] + b["item_width_m"]) / 2.0 - abs(dx)) / 2.0 + 0.05
                    if dx >= 0: a["x_m"] -= shift; b["x_m"] += shift
                    else: a["x_m"] += shift; b["x_m"] -= shift
                else:
                    shift = ((a["item_depth_m"] + b["item_depth_m"]) / 2.0 - abs(dz)) / 2.0 + 0.05
                    if dz >= 0: a["z_m"] -= shift; b["z_m"] += shift
                    else: a["z_m"] += shift; b["z_m"] -= shift
                clamp_inside_room(a, room); clamp_inside_room(b, room)
        if not changed: break
    return collision_fixes

def place_coffee_table_near_sofa(items, room):
    fixes = 0
    sofas = [x for x in items if x["category"] == "sofa" and x.get("layer", "floor") == "floor"]
    tables = [x for x in items if x["category"] == "coffee_table" and x.get("layer", "floor") == "floor"]
    if sofas and tables:
        sofa, table = sofas[0], tables[0]
        table["x_m"] = sofa["x_m"]
        table["z_m"] = sofa["z_m"] + (sofa["item_depth_m"] / 2.0 + table["item_depth_m"] / 2.0 + 0.22)
        table["rotation_y_deg"] = sofa["rotation_y_deg"]
        fixes += clamp_inside_room(table, room) + 1
    return fixes

def place_armchair_near_sofa(items, room):
    fixes = 0
    sofas = [x for x in items if x["category"] == "sofa" and x.get("layer", "floor") == "floor"]
    chairs = [x for x in items if x["category"] == "armchair" and x.get("layer", "floor") == "floor"]
    if sofas and chairs:
        sofa = sofas[0]
        for idx, chair in enumerate(chairs[:2]):
            side = -1 if idx % 2 == 0 else 1
            chair["x_m"] = sofa["x_m"] + side * (sofa["item_width_m"] / 2.0 + chair["item_width_m"] / 2.0 + 0.25)
            chair["z_m"] = sofa["z_m"] + 0.15
            fixes += clamp_inside_room(chair, room) + 1
    return fixes

def place_rug_under_group(items, room):
    fixes = 0
    rugs = [x for x in items if x["category"] in RUG_CATEGORIES]
    sofas = [x for x in items if x["category"] == "sofa" and x.get("layer", "floor") == "floor"]
    tables = [x for x in items if x["category"] == "coffee_table" and x.get("layer", "floor") == "floor"]
    if rugs and sofas:
        rug, sofa = rugs[0], sofas[0]
        rug["x_m"] = sofa["x_m"]; rug["z_m"] = sofa["z_m"] + 0.15
        need_w = sofa["item_width_m"] + 0.6
        need_d = sofa["item_depth_m"] + (tables[0]["item_depth_m"] if tables else 0.2) + 0.6
        rug["item_width_m"] = max(rug["item_width_m"], need_w)
        rug["item_depth_m"] = max(rug["item_depth_m"], need_d)
        rug["item_height_m"] = min(rug["item_height_m"], 0.03)
        rug["rotation_y_deg"] = 0.0; rug["layer"] = "floor"
        fixes += clamp_inside_room(rug, room) + 1
    return fixes

def place_wall_items(items, room):
    fixes = 0
    for idx, item in enumerate([x for x in items if x["category"] in WALL_MOUNT_CATEGORIES]):
        wall = "back" if idx % 2 == 0 else "left"
        snap_to_wall(item, room, wall=wall, margin=0.02)
        item["layer"] = "wall"; item["y_m"] = min(room["heightM"] * 0.62, 1.55)
        fixes += 1
    return fixes

def place_top_surface_decor(items):
    fixes = 0
    supports = [x for x in items if x["category"] in SUPPORT_SURFACE_CATEGORIES and x.get("layer", "floor") == "floor"]
    decors = [x for x in items if x["category"] in TOP_SURFACE_DECOR_CATEGORIES]
    if supports and decors:
        support = supports[0]
        for idx, item in enumerate(decors[:2]):
            item["x_m"] = support["x_m"] + (0.08 if idx % 2 == 0 else -0.08)
            item["z_m"] = support["z_m"]
            item["y_m"] = support["item_height_m"] + item["item_height_m"] / 2.0
            item["layer"] = "top_surface"; item["supportSurfaceId"] = support["productId"]; item["rotation_y_deg"] = 0.0
            fixes += 1
    return fixes

def apply_wall_priors(items, room):
    fixes = 0
    for item in items:
        if item["category"] in WALL_FAVOR_CATEGORIES or item.get("anchor", "") == "against_wall":
            snap_to_wall(item, room); fixes += 1
    return fixes

def default_y_for_item(item):
    if item.get("layer") == "wall": return item.get("y_m", 1.5)
    if item.get("layer") == "top_surface": return item.get("y_m", item["item_height_m"] / 2.0)
    return item["item_height_m"] / 2.0

def build_layout_reasoning(item):
    cat = item["category"]
    if cat == "sofa": return "Sofa được đặt sát tường để mở lối đi, giữ trọng tâm phòng khách rõ ràng và tạo bố cục ngồi thực tế."
    if cat == "coffee_table": return "Bàn nước được đặt gần sofa để hỗ trợ sinh hoạt của seating group và giữ quan hệ sử dụng hợp lý."
    if cat == "armchair": return "Ghế thư giãn được đặt như ghế phụ cạnh nhóm ngồi chính, tránh chắn lối đi trung tâm."
    if cat == "rug": return "Thảm được đưa xuống dưới nhóm ghế-bàn để neo thị giác khu vực tiếp khách."
    if cat in {"tv_stand","bed","wardrobe","cabinet","bookshelf","desk"}: return "Vật lớn được kéo về gần tường để giảm chiếm vùng di chuyển giữa phòng và bám theo prior của layout thật."
    if cat in {"wall_art","mirror"}: return "Vật treo tường được neo lên mặt tường thay vì đặt dưới sàn để giữ đúng ngữ cảnh sử dụng."
    if cat in {"lamp","plant"} and item.get("layer") == "top_surface": return "Decor nhỏ được đặt trên mặt bàn/tủ để hợp lý hơn về support surface."
    return "Vật phẩm được đặt theo prior học từ layout thật và đã qua bước sửa ràng buộc không gian."

@lru_cache(maxsize=1)
def pybullet_probe():
    try:
        import pybullet as p
        cid = p.connect(p.DIRECT)
        p.disconnect(cid)
        return True, ""
    except Exception as e:
        return False, str(e)

def finalize_layout(payload):
    room = payload["room"]
    room = {"widthM": float(room["widthM"]), "lengthM": float(room["lengthM"]), "heightM": float(room["heightM"]), "type": canonical_room_type(room.get("type", "unknown")), "style": room.get("style", "")}
    top_k = int(payload.get("topK", 8))
    threshold = float(payload.get("minScore", filter_config.get("recommended_threshold", 0.5)))
    model_url_by_id = payload.get("modelUrlById", {})

    selected, rejected = select_products_for_layout(payload=payload, top_k=top_k, threshold=threshold)
    placed = predict_layout_for_selected(room, selected)

    items, out_of_room_fixed = [], 0
    for p in placed:
        raw = p.get("raw", {})
        item = {
            "productId": p.get("product_id"), "name": p.get("name", ""), "category": p.get("_category_norm", canonical_category(p.get("category", "unknown"))),
            "score": float(p.get("final_score", 0.0)), "modelUrl": model_url_by_id.get(p.get("product_id"), ""),
            "keepProbability": float(p.get("keep_probability", 0.0)), "rotation_y_deg": float(p.get("pred_rotation_y_deg", 0.0)),
            "x_m": float(p.get("pred_x_m", 0.0)), "z_m": float(p.get("pred_z_m", 0.0)),
            "item_width_m": float(p.get("_w_m", 0.0)), "item_depth_m": float(p.get("_d_m", 0.0)), "item_height_m": float(p.get("_h_m", 0.0)),
            "anchor": p.get("pred_anchor", "floating"), "layer": "floor",
            "sourceRecommendationReasoning": raw.get("reasoning", p.get("source_reasoning", "")),
        }
        if item["category"] in WALL_MOUNT_CATEGORIES: item["layer"] = "wall"
        out_of_room_fixed += clamp_inside_room(item, room)
        items.append(item)

    wall_prior_fixes = apply_wall_priors(items, room)
    coffee_fixes = place_coffee_table_near_sofa(items, room)
    armchair_fixes = place_armchair_near_sofa(items, room)
    rug_fixes = place_rug_under_group(items, room)
    wall_item_fixes = place_wall_items(items, room)
    top_surface_fixes = place_top_surface_decor(items)
    collision_fixes = resolve_overlaps(items, room)

    final_output_items = []
    for item in items:
        y_m = default_y_for_item(item)
        final_output_items.append({
            "productId": item["productId"], "name": item["name"], "category": item["category"], "score": float(item["score"]),
            "modelUrl": item["modelUrl"],
            "position": {"x": float(item["x_m"]), "y": float(y_m), "z": float(item["z_m"])},
            "rotationY": float(item["rotation_y_deg"]),
            "keepProbability": float(item["keepProbability"]),
            "footprint": {"widthM": float(item["item_width_m"]), "depthM": float(item["item_depth_m"]), "heightM": float(item["item_height_m"])},
            "layer": item.get("layer", "floor"),
            "anchor": item.get("anchor", "floating"),
            "supportSurfaceId": item.get("supportSurfaceId"),
            "layoutReasoning": build_layout_reasoning(item),
            "sourceRecommendationReasoning": item.get("sourceRecommendationReasoning", ""),
        })

    pybullet_ok, pybullet_note = pybullet_probe()
    return {
        "room": room, "items": final_output_items, "rejected": rejected,
        "metrics": {
            "selectedCount": len(final_output_items), "rejectedCount": len(rejected),
            "outOfRoomFixed": int(out_of_room_fixed), "wallPriorFixes": int(wall_prior_fixes),
            "coffeeTableFixes": int(coffee_fixes), "armchairFixes": int(armchair_fixes), "rugFixes": int(rug_fixes),
            "wallItemFixes": int(wall_item_fixes), "topSurfaceFixes": int(top_surface_fixes), "collisionsResolved": int(collision_fixes),
            "pybulletAvailable": bool(pybullet_ok), "pybulletNote": pybullet_note,
            "layoutModel": "decor_transformer.pt", "filterModel": "product_filter.joblib",
        }
    }

ROOM_TYPE_ALIAS = {'living room': 'living_room', 'livingroom': 'living_room', 'phòng khách': 'living_room', 'bed room': 'bedroom', 'master bedroom': 'bedroom', 'kids room': 'bedroom', 'phòng ngủ': 'bedroom', 'dining room': 'dining_room', 'phòng ăn': 'dining_room', 'study room': 'office', 'office': 'office', 'phòng làm việc': 'office', 'kitchen': 'kitchen', 'phòng bếp': 'kitchen', 'bathroom': 'bathroom', 'phòng tắm': 'bathroom'}
CATEGORY_ALIAS = {'sofa': 'sofa', 'sectional sofa': 'sofa', 'loveseat': 'sofa', 'couch': 'sofa', 'coffee table': 'coffee_table', 'tea table': 'coffee_table', 'side table': 'side_table', 'end table': 'side_table', 'nightstand': 'nightstand', 'bedside table': 'nightstand', 'tv stand': 'tv_stand', 'media console': 'tv_stand', 'armchair': 'armchair', 'lounge chair': 'armchair', 'recliner': 'armchair', 'chair': 'chair', 'dining chair': 'dining_chair', 'office chair': 'office_chair', 'desk': 'desk', 'dining table': 'dining_table', 'table': 'table', 'bed': 'bed', 'wardrobe': 'wardrobe', 'closet': 'wardrobe', 'cabinet': 'cabinet', 'bookshelf': 'bookshelf', 'shelf': 'bookshelf', 'drawer': 'drawer', 'dresser': 'dresser', 'rug': 'rug', 'carpet': 'rug', 'lamp': 'lamp', 'floor lamp': 'floor_lamp', 'ceiling lamp': 'ceiling_lamp', 'plant': 'plant', 'mirror': 'mirror', 'painting': 'wall_art', 'picture': 'wall_art', 'wall art': 'wall_art', 'wall decor': 'wall_art', 'stool': 'stool', 'bench': 'bench', 'ghế sofa': 'sofa', 'bàn nước': 'coffee_table', 'bàn trà': 'coffee_table', 'ghế thư giãn': 'armchair', 'ghế': 'chair', 'bàn đầu giường': 'nightstand', 'tủ tivi': 'tv_stand', 'bàn ăn': 'dining_table', 'ghế ăn': 'dining_chair', 'bàn làm việc': 'desk', 'ghế văn phòng': 'office_chair', 'giường': 'bed', 'tủ quần áo': 'wardrobe', 'thảm': 'rug', 'đèn': 'lamp', 'tranh': 'wall_art', 'kệ sách': 'bookshelf', 'tủ': 'cabinet', 'cây trang trí': 'plant', 'gương': 'mirror'}
WALL_FAVOR_CATEGORIES = {'wardrobe', 'bookshelf', 'cabinet', 'sofa', 'bed', 'desk', 'tv_stand'}
SUPPORT_SURFACE_CATEGORIES = {'dining_table', 'cabinet', 'nightstand', 'desk', 'side_table', 'coffee_table'}
TOP_SURFACE_DECOR_CATEGORIES = {'lamp', 'plant'}
WALL_MOUNT_CATEGORIES = {'mirror', 'wall_art'}
RUG_CATEGORIES = {'rug'}


loaded_filter = joblib.load(PRODUCT_FILTER_PATH)
filter_config = json.load(open(FILTER_CONFIG_PATH, 'r', encoding='utf-8'))
category_priors_df = pd.read_csv(CATEGORY_PRIORS_PATH)
decor_vocab = json.load(open(DECOR_VOCAB_PATH, 'r', encoding='utf-8'))

room_cat_prob_map = {
    (str(r['room_type']), str(r['category'])): float(r['room_cat_prob'])
    for r in category_priors_df.to_dict('records')
}

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
        'Do not use decor_transformer_v2.pt with this service. '
        f'missing={missing_keys[:5]}, '
        f'unexpected={unexpected_keys[:5]}, '
        f'shape_mismatch={shape_mismatch[:5]}'
    )

decor_model.load_state_dict(state_dict)
decor_model.eval()

for name, param in decor_model.named_parameters():
    if not torch.isfinite(param).all():
        raise RuntimeError(f'decor_transformer.pt contains NaN/Inf parameter: {name}')

