from __future__ import annotations

"""
Runtime patches for layout_engine.engine.

Purpose:
- keep FastAPI as a layout-only service;
- inject the trained LayoutTransformer prediction as an extra layout candidate;
- fix graph-relation scoring when multiple products share the same category;
- add optional reserved-zone constraints for doors/windows/walkways;
- make density caps more room-aware.

This file intentionally wraps the existing engine instead of replacing it, so the
large template/rule/Shapely/scoring pipeline remains stable.
"""

import math
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from layout_engine import engine as _engine

_ORIGINAL_GENERATE_CANDIDATES = _engine.generate_candidates
_ORIGINAL_REPAIR_LAYOUT = _engine.repair_layout
_ORIGINAL_SCORE_LAYOUT = _engine.score_layout
_ORIGINAL_FINALIZE_LAYOUT = _engine.finalize_layout

_CURRENT_PAYLOAD: Dict[str, Any] = {}
_PATCH_INSTALLED = False
_LAST_MODEL_CANDIDATE_STATUS: Dict[str, Any] = {"attempted": False, "injected": False, "error": ""}


# ---------------------------------------------------------------------------
# Product selection: keep the old logic, but use room-aware density caps.
# ---------------------------------------------------------------------------

def _room_density_cap(room_type: str, density: str, requested_top_k: int) -> int:
    d = _engine.canonical_text(density)
    is_sparse = "sparse" in d or "thưa" in d or "thua" in d or "ít" in d or "it" in d
    is_dense = "dense" in d or "dày" in d or "day" in d or "nhiều" in d or "nhieu" in d

    caps = {
        "living_room": (6, 9, 11),
        "bedroom": (5, 7, 9),
        "dining_room": (5, 8, 10),
        "office": (5, 7, 9),
        "kitchen": (5, 7, 10),
        "bathroom": (4, 5, 6),
    }
    sparse_cap, medium_cap, dense_cap = caps.get(room_type, (5, 8, 10))
    cap = sparse_cap if is_sparse else dense_cap if is_dense else medium_cap
    return max(1, min(int(requested_top_k), cap))


def select_products(room: Any, products: List[Any], payload: Dict[str, Any]):
    top_k = int(payload.get("topK") or payload.get("top_k") or 8)
    density = str(payload.get("furnitureDensity") or payload.get("furniture_density") or "medium")
    top_k = _room_density_cap(room.type, density, max(1, top_k))

    comp = _engine.ROOM_COMPOSITIONS.get(room.type, _engine.ROOM_COMPOSITIONS["living_room"])
    allowed = comp["allowed"]
    quota = _engine.apply_style_quota(dict(comp["quota"]), room.style)
    priority = list(comp["priority"])

    predictor = _engine._get_model_predictor()
    if predictor.get("available"):
        try:
            scored_df = predictor["score"](payload)
            if scored_df is not None and len(scored_df) > 0:
                prob_by_id = {}
                for _, row in scored_df.iterrows():
                    pid = str(row.get("product_id", ""))
                    if pid and "keep_probability" in row:
                        prob_by_id[pid] = float(row["keep_probability"])
                for p in products:
                    if p.productId in prob_by_id:
                        p.sourceScore = 0.6 * prob_by_id[p.productId] + 0.4 * p.sourceScore
        except Exception:
            pass

    rejected: List[Dict[str, Any]] = []
    valid: List[Any] = []
    room_area = room.widthM * room.lengthM

    for p in products:
        if p.category not in allowed:
            rejected.append({"productId": p.productId, "name": p.name, "category": p.category, "reason": "category_not_allowed_for_room"})
            continue
        if p.widthM > room.widthM * 0.92 or p.depthM > room.lengthM * 0.92 or (p.widthM * p.depthM) > room_area * 0.45:
            rejected.append({"productId": p.productId, "name": p.name, "category": p.category, "reason": "too_large_for_room"})
            continue
        valid.append(p)

    selected: List[Any] = []
    used = set()
    count_by_cat: Dict[str, int] = {}

    def p_score(p: Any) -> float:
        dim_compat = _engine._dim_compatibility(p, room)
        return (
            0.50 * p.sourceScore
            + 0.25 * _engine.style_match_score(room.style, p.styles)
            + 0.15 * min(1.0, (p.widthM * p.depthM) / max(0.1, room_area * 0.08))
            + 0.10 * dim_compat
        )

    for cat in priority:
        candidates = [p for p in valid if p.category == cat and p.productId not in used]
        candidates.sort(key=p_score, reverse=True)
        for p in candidates[: int(quota.get(cat, 0))]:
            if len(selected) >= top_k:
                break
            selected.append(p)
            used.add(p.productId)
            count_by_cat[cat] = count_by_cat.get(cat, 0) + 1
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for p in sorted(valid, key=p_score, reverse=True):
            if p.productId in used:
                continue
            if count_by_cat.get(p.category, 0) >= int(quota.get(p.category, 1)):
                continue
            selected.append(p)
            used.add(p.productId)
            count_by_cat[p.category] = count_by_cat.get(p.category, 0) + 1
            if len(selected) >= top_k:
                break

    for p in valid:
        if p.productId not in used:
            reason = "category_quota_exceeded" if count_by_cat.get(p.category, 0) >= int(quota.get(p.category, 1)) else "not_selected"
            rejected.append({"productId": p.productId, "name": p.name, "category": p.category, "reason": reason})

    missing = []
    if room.type == "living_room":
        for cat in ["sofa", "coffee_table", "tv_stand"]:
            if not any(p.category == cat for p in selected):
                missing.append(cat)
    elif room.type == "bedroom":
        if not any(p.category == "bed" for p in selected):
            missing.append("bed")
    elif room.type == "dining_room":
        if not any(p.category == "dining_table" for p in selected):
            missing.append("dining_table")
    elif room.type == "office":
        if not any(p.category == "desk" for p in selected):
            missing.append("desk")

    return selected, rejected, missing


# ---------------------------------------------------------------------------
# Trained LayoutTransformer candidate injection.
# ---------------------------------------------------------------------------

def _product_to_model_payload(p: Any) -> Dict[str, Any]:
    raw = dict(p.raw) if isinstance(getattr(p, "raw", None), dict) else {}
    raw.update({
        "id": p.productId,
        "productId": p.productId,
        "name": p.name,
        "category": p.category,
        "widthM": p.widthM,
        "depthM": p.depthM,
        "heightM": p.heightM,
        "modelUrl": p.modelUrl,
        "imageUrl": p.imageUrl,
        "dimensions": {
            "widthM": p.widthM,
            "depthM": p.depthM,
            "heightM": p.heightM,
            "width": p.widthM,
            "depth": p.depthM,
            "height": p.heightM,
        },
    })
    return raw


def _nearest_wall(room: Any, x: float, z: float) -> str:
    distances = {
        "LEFT_WALL": abs(x + room.widthM / 2.0),
        "RIGHT_WALL": abs(room.widthM / 2.0 - x),
        "BACK_WALL": abs(z + room.lengthM / 2.0),
        "FRONT_WALL": abs(room.lengthM / 2.0 - z),
    }
    return min(distances, key=distances.get)


def _anchor_to_wall(anchor: Any, room: Any, x: float, z: float) -> Optional[str]:
    a = str(anchor or "").strip().upper().replace("-", "_").replace(" ", "_")
    mapping = {
        "LEFT": "LEFT_WALL", "LEFT_WALL": "LEFT_WALL", "WEST": "LEFT_WALL", "WEST_WALL": "LEFT_WALL",
        "RIGHT": "RIGHT_WALL", "RIGHT_WALL": "RIGHT_WALL", "EAST": "RIGHT_WALL", "EAST_WALL": "RIGHT_WALL",
        "BACK": "BACK_WALL", "BACK_WALL": "BACK_WALL", "NORTH": "BACK_WALL", "NORTH_WALL": "BACK_WALL",
        "FRONT": "FRONT_WALL", "FRONT_WALL": "FRONT_WALL", "SOUTH": "FRONT_WALL", "SOUTH_WALL": "FRONT_WALL",
    }
    if a in mapping:
        return mapping[a]
    if "WALL" in a:
        return _nearest_wall(room, x, z)
    return None


def _place_wall_mount_from_prediction(item: Any, room: Any, prediction: Dict[str, Any], anchor_wall: Optional[str]) -> None:
    wall = anchor_wall or _nearest_wall(room, float(prediction.get("pred_x_m", 0.0)), float(prediction.get("pred_z_m", 0.0)))
    along = float(prediction.get("pred_z_m", 0.0)) if wall in {"LEFT_WALL", "RIGHT_WALL"} else float(prediction.get("pred_x_m", 0.0))
    _engine.place_wall_art(item, room, wall, along=along)
    item.layoutReasoning = "Tranh/gương được treo tường theo vị trí dự đoán từ LayoutTransformer."
    item.relations.append({"type": "ai_predicted_pose", "target": "layout_transformer"})


def _generate_model_candidate(room: Any, products: List[Any]) -> Optional[Dict[str, Any]]:
    global _LAST_MODEL_CANDIDATE_STATUS
    _LAST_MODEL_CANDIDATE_STATUS = {"attempted": True, "injected": False, "error": ""}

    predictor = _engine._get_model_predictor()
    if not predictor.get("available"):
        _LAST_MODEL_CANDIDATE_STATUS["error"] = str(predictor.get("error", "model_not_available"))
        return None

    try:
        room_dict = asdict(room)
        model_products = [_product_to_model_payload(p) for p in products]
        predictions = predictor["predict"](room_dict, model_products) or []
        if not predictions:
            _LAST_MODEL_CANDIDATE_STATUS["error"] = "empty_model_predictions"
            return None

        pred_by_id = {str(pred.get("product_id")): pred for pred in predictions if pred.get("product_id") is not None}
        items = [_engine.item_from_product(p) for p in products]
        fallback_walls = _engine.default_focal_walls(room)

        for index, item in enumerate(items):
            pred = pred_by_id.get(str(item.productId))
            if pred is None:
                wall = fallback_walls[index % len(fallback_walls)]
                _engine.place_against_wall(item, room, wall, along=0.0)
                item.layoutReasoning = "Fallback placement vì model không trả vị trí cho sản phẩm này."
                item.relations.append({"type": "fallback_after_model", "target": wall})
                continue

            x = float(pred.get("pred_x_m", 0.0))
            z = float(pred.get("pred_z_m", 0.0))
            rotation = float(pred.get("pred_rotation_y_deg", 0.0)) % 360.0
            anchor_wall = _anchor_to_wall(pred.get("pred_anchor"), room, x, z)

            if item.category in _engine.WALL_MOUNT_CATEGORIES:
                _place_wall_mount_from_prediction(item, room, pred, anchor_wall)
                continue

            item.x = x
            item.z = z
            item.rotationY = rotation
            item.facingDirection = _engine.rotation_to_facing(rotation)
            item.anchorWall = anchor_wall or "MODEL_FLOATING"
            item.layer = "floor"
            item.y = 0.01 if item.category == "rug" else item.heightM / 2.0
            item.facingTarget = "room_center"
            item.layoutReasoning = "Vị trí ban đầu được dự đoán bởi LayoutTransformer, sau đó được Shapely/rule repair kiểm tra."
            item.relations.append({"type": "ai_predicted_pose", "target": "layout_transformer"})
            if anchor_wall:
                item.relations.append({"type": "against_wall", "target": anchor_wall})
            _engine.clamp_inside_room(item, room)

        _LAST_MODEL_CANDIDATE_STATUS = {"attempted": True, "injected": True, "error": ""}
        return {
            "items": items,
            "template": "model_layout_transformer_candidate",
            "focalWall": "MODEL_PREDICTED",
            "activeZone": _engine.active_zone(room, len(items)),
            "layoutAxis": "MODEL",
        }
    except Exception as exc:
        _LAST_MODEL_CANDIDATE_STATUS = {"attempted": True, "injected": False, "error": str(exc)}
        return None


def generate_candidates(room: Any, products: List[Any]) -> List[Dict[str, Any]]:
    candidates = _ORIGINAL_GENERATE_CANDIDATES(room, products)
    model_candidate = _generate_model_candidate(room, products)
    if model_candidate is not None:
        candidates.append(model_candidate)
    return candidates


# ---------------------------------------------------------------------------
# Optional reserved zones: doors, windows, walkways, explicit no-place areas.
# ---------------------------------------------------------------------------

def _zone_from_wall(room: Any, wall: str, along: float, width: float, depth: float, kind: str) -> Dict[str, Any]:
    wall = str(wall or "").upper().replace("-", "_").replace(" ", "_")
    if wall in {"LEFT", "WEST", "LEFT_WALL", "WEST_WALL"}:
        return {"x": -room.widthM / 2.0 + depth / 2.0, "z": along, "widthM": depth, "depthM": width, "kind": kind}
    if wall in {"RIGHT", "EAST", "RIGHT_WALL", "EAST_WALL"}:
        return {"x": room.widthM / 2.0 - depth / 2.0, "z": along, "widthM": depth, "depthM": width, "kind": kind}
    if wall in {"FRONT", "SOUTH", "FRONT_WALL", "SOUTH_WALL"}:
        return {"x": along, "z": room.lengthM / 2.0 - depth / 2.0, "widthM": width, "depthM": depth, "kind": kind}
    return {"x": along, "z": -room.lengthM / 2.0 + depth / 2.0, "widthM": width, "depthM": depth, "kind": kind}


def _normalize_reserved_zones(payload: Dict[str, Any], room: Any) -> List[Dict[str, Any]]:
    constraints = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}
    raw_zones: List[Dict[str, Any]] = []
    for key in ("reservedZones", "noPlaceZones", "walkways"):
        value = payload.get(key) or constraints.get(key)
        if isinstance(value, list):
            raw_zones.extend([z for z in value if isinstance(z, dict)])

    zones: List[Dict[str, Any]] = []
    for z in raw_zones:
        zones.append({
            "x": _engine.as_float(z.get("x"), 0.0),
            "z": _engine.as_float(z.get("z"), 0.0),
            "widthM": max(0.1, _engine.normalize_dim(z.get("widthM") or z.get("width") or z.get("w"), 0.8)),
            "depthM": max(0.1, _engine.normalize_dim(z.get("depthM") or z.get("lengthM") or z.get("depth") or z.get("length") or z.get("d"), 0.8)),
            "kind": str(z.get("kind") or z.get("type") or "reserved_zone"),
        })

    openings: List[Dict[str, Any]] = []
    for key in ("openings", "doors", "windows"):
        value = payload.get(key) or constraints.get(key)
        if isinstance(value, list):
            openings.extend([o for o in value if isinstance(o, dict)])

    for op in openings:
        kind = str(op.get("kind") or op.get("type") or "opening")
        wall = op.get("wall") or op.get("anchorWall") or "BACK_WALL"
        along = _engine.as_float(op.get("along") or op.get("center") or op.get("offset"), 0.0)
        width = max(0.3, _engine.normalize_dim(op.get("widthM") or op.get("width"), 0.9 if "door" in kind.lower() else 1.2))
        depth = max(0.25, _engine.normalize_dim(op.get("clearanceM") or op.get("depthM") or op.get("depth"), 0.9 if "door" in kind.lower() else 0.45))
        zones.append(_zone_from_wall(room, wall, along, width, depth, kind))

    return zones


def _zone_polygon(zone: Dict[str, Any]):
    if _engine.box is None:
        return None
    w = float(zone["widthM"])
    d = float(zone["depthM"])
    x = float(zone["x"])
    z = float(zone["z"])
    return _engine.box(x - w / 2.0, z - d / 2.0, x + w / 2.0, z + d / 2.0)


def _reserved_zone_overlap(items: List[Any], zones: List[Dict[str, Any]]) -> float:
    if _engine.Polygon is None or _engine.box is None or not zones:
        return 0.0
    total = 0.0
    zone_polys = [_zone_polygon(z) for z in zones]
    zone_polys = [z for z in zone_polys if z is not None]
    for item in items:
        if item.layer != "floor" or item.category == "rug":
            continue
        item_poly = _engine.oriented_box_polygon(item)
        for zone_poly in zone_polys:
            total += item_poly.intersection(zone_poly).area
    return total


def _repair_reserved_zones(candidate: Dict[str, Any], room: Any, zones: List[Dict[str, Any]]) -> int:
    if _engine.Polygon is None or _engine.box is None or not zones:
        return 0
    fixed = 0
    items: List[Any] = candidate.get("items", [])
    zone_polys = [(z, _zone_polygon(z)) for z in zones]
    zone_polys = [(z, poly) for z, poly in zone_polys if poly is not None]

    for _ in range(80):
        changed = False
        for item in items:
            if item.layer != "floor" or item.category == "rug":
                continue
            item_poly = _engine.oriented_box_polygon(item)
            for zone, zone_poly in zone_polys:
                overlap = item_poly.intersection(zone_poly).area
                if overlap <= 1e-5:
                    continue
                dx = item.x - float(zone["x"])
                dz = item.z - float(zone["z"])
                if abs(dx) < 1e-6 and abs(dz) < 1e-6:
                    dx, dz = 1.0, 0.35
                norm = max(math.hypot(dx, dz), 1e-6)
                push = min(1.0, max(0.20, math.sqrt(overlap) + 0.15))
                item.x += dx / norm * push
                item.z += dz / norm * push
                _engine.clamp_inside_room(item, room)
                fixed += 1
                changed = True
                item.relations.append({"type": "avoid_reserved_zone", "target": str(zone.get("kind", "reserved_zone"))})
        if not changed:
            break
    return fixed


def repair_layout(candidate: Dict[str, Any], room: Any):
    repaired, metrics = _ORIGINAL_REPAIR_LAYOUT(candidate, room)
    zones = _normalize_reserved_zones(_CURRENT_PAYLOAD, room)
    fixed = _repair_reserved_zones(repaired, room, zones)
    if fixed:
        repaired, second_metrics = _ORIGINAL_REPAIR_LAYOUT(repaired, room)
        for key, value in second_metrics.items():
            metrics[key] = metrics.get(key, 0) + value
    metrics["reservedZoneFixed"] = metrics.get("reservedZoneFixed", 0) + fixed
    return repaired, metrics


def score_layout(candidate: Dict[str, Any], room: Any, products: List[Any]) -> Dict[str, Any]:
    result = _ORIGINAL_SCORE_LAYOUT(candidate, room, products)
    zones = _normalize_reserved_zones(_CURRENT_PAYLOAD, room)
    penalty = _reserved_zone_overlap(candidate.get("items", []), zones)
    if zones:
        reserved_score = max(0.0, 1.0 - penalty / 0.50)
        result["breakdown"]["reservedZoneClearance"] = round(reserved_score, 4)
        result["rawPenalties"]["reservedZoneOverlapArea"] = round(penalty, 6)
        result["total"] = round(max(0.0, float(result["total"]) - min(25.0, penalty * 40.0)), 4)
    return result


# ---------------------------------------------------------------------------
# Graph relation scoring: support duplicate categories, e.g. 2 chairs/nightstands.
# ---------------------------------------------------------------------------

def _group_items(items: Iterable[Any]) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    by_id: Dict[str, Any] = {}
    by_cat: Dict[str, List[Any]] = {}
    for item in items:
        by_id[str(item.productId)] = item
        by_cat.setdefault(item.category, []).append(item)
    return by_id, by_cat


def _resolve_targets(target: str, by_id: Dict[str, Any], by_cat: Dict[str, List[Any]]) -> List[Any]:
    if target in by_id:
        return [by_id[target]]
    return list(by_cat.get(target, []))


def _best_relation_score(item: Any, targets: List[Any], score_fn) -> Optional[float]:
    values = [score_fn(target) for target in targets if target.productId != item.productId]
    return max(values) if values else None


def score_graph_relations(items: List[Any]) -> Dict[str, float]:
    facing_total = 0.0
    facing_count = 0
    relation_total = 0.0
    relation_count = 0
    by_id, by_cat = _group_items(items)

    for item in items:
        if item.facingTarget and item.facingTarget != "room_center":
            targets = _resolve_targets(item.facingTarget, by_id, by_cat)
            value = _best_relation_score(item, targets, lambda target: max(0.0, _engine.dot_facing_to_target(item, target)))
            if value is not None:
                facing_total += value
                facing_count += 1
        elif item.facingTarget == "room_center":
            dx, dz = -item.x, -item.z
            norm = max(math.hypot(dx, dz), 1e-6)
            fx, fz = _engine.facing_vector(item.facingDirection)
            facing_total += max(0.0, (fx * dx + fz * dz) / norm)
            facing_count += 1

        for rel in item.relations:
            rtype = rel.get("type")
            target = rel.get("target")

            if rtype == "against_wall":
                relation_total += 1.0 if "WALL" in str(item.anchorWall) else 0.0
                relation_count += 1
                continue
            if not target:
                continue

            targets = _resolve_targets(target, by_id, by_cat)
            if rtype == "near":
                value = _best_relation_score(
                    item,
                    targets,
                    lambda target_item: 1.0 if _engine.distance(item, target_item) <= 2.5 else max(0.0, 1.0 - (_engine.distance(item, target_item) - 2.5) / 2.0),
                )
            elif rtype == "under":
                value = _best_relation_score(item, targets, lambda target_item: 1.0 if _engine.distance(item, target_item) <= 1.5 else 0.0)
            elif rtype == "face_to":
                value = _best_relation_score(item, targets, lambda target_item: max(0.0, _engine.dot_facing_to_target(item, target_item)))
            else:
                value = None

            if value is not None:
                relation_total += value
                relation_count += 1

    return {
        "facingScore": facing_total / max(1, facing_count),
        "relationScore": relation_total / max(1, relation_count),
    }


# ---------------------------------------------------------------------------
# Public wrapper used by FastAPI.
# ---------------------------------------------------------------------------

def finalize_layout(payload: Dict[str, Any]) -> Dict[str, Any]:
    global _CURRENT_PAYLOAD
    previous_payload = _CURRENT_PAYLOAD
    _CURRENT_PAYLOAD = payload if isinstance(payload, dict) else {}
    try:
        result = _ORIGINAL_FINALIZE_LAYOUT(payload)
        metrics = result.setdefault("metrics", {})
        predictor = _engine._get_model_predictor()
        metrics["modelPredictionAvailable"] = bool(predictor.get("available"))
        metrics["modelCandidateAttempted"] = bool(_LAST_MODEL_CANDIDATE_STATUS.get("attempted"))
        metrics["modelCandidateInjected"] = bool(_LAST_MODEL_CANDIDATE_STATUS.get("injected"))
        if _LAST_MODEL_CANDIDATE_STATUS.get("error"):
            metrics["modelCandidateError"] = _LAST_MODEL_CANDIDATE_STATUS["error"]
        metrics["layoutServiceRole"] = "layout_only"
        return result
    finally:
        _CURRENT_PAYLOAD = previous_payload


def model_patch_status() -> Dict[str, Any]:
    predictor = _engine._get_model_predictor()
    return {
        "patchInstalled": _PATCH_INSTALLED,
        "modelAvailable": bool(predictor.get("available")),
        "modelError": predictor.get("error", ""),
        "lastModelCandidate": dict(_LAST_MODEL_CANDIDATE_STATUS),
    }


def install_engine_patches() -> None:
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return
    _engine.select_products = select_products
    _engine.generate_candidates = generate_candidates
    _engine.repair_layout = repair_layout
    _engine.score_layout = score_layout
    _engine.score_graph_relations = score_graph_relations
    _engine.finalize_layout = finalize_layout
    _PATCH_INSTALLED = True


install_engine_patches()
