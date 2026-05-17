from __future__ import annotations

"""
Small-room capacity patch.

Installed after bedroom_variants_patch.

Problem fixed:
- users can enter very small rooms, e.g. < 10 m², then choose medium/dense;
- previous selection may still keep too many products and Shapely has to repair
  heavily, causing crowded or overlapping layouts.

This patch applies a physical room capacity limit before candidate generation:
- area < 6 m²: tiny room cap;
- area < 10 m²: small room cap;
- medium/dense cannot exceed the safe product count for the room area;
- essential products are kept first;
- decorative / duplicate / low-priority products are rejected with a clear reason.
"""

from typing import Any, Dict, List, Tuple

from layout_engine import engine as _engine
from layout_engine import bedroom_variants_patch as _variants_patch

_BASE_SELECT_PRODUCTS = _engine.select_products
_BASE_FINALIZE_LAYOUT = _engine.finalize_layout
_BASE_MODEL_PATCH_STATUS = _variants_patch.model_patch_status

_SMALL_ROOM_PATCH_INSTALLED = False


# Floor-heavy categories increase collision risk in small rooms.
FLOOR_HEAVY_CATEGORIES = {
    "bed",
    "sofa",
    "armchair",
    "chair",
    "desk",
    "dining_table",
    "dining_chair",
    "wardrobe",
    "cabinet",
    "bookshelf",
    "tv_stand",
    "coffee_table",
    "side_table",
    "bench",
    "stool",
}

LIGHT_CATEGORIES = {
    "rug",
    "mirror",
    "wall_art",
    "ceiling_lamp",
    "lamp",
    "plant",
}

ROOM_PRIORITY = {
    "bedroom": [
        "bed", "rug", "nightstand", "ceiling_lamp", "mirror",
        "wardrobe", "cabinet", "bench", "desk", "chair", "armchair",
        "bookshelf", "lamp", "plant", "wall_art", "side_table",
    ],
    "living_room": [
        "sofa", "coffee_table", "tv_stand", "rug", "ceiling_lamp",
        "armchair", "chair", "side_table", "bookshelf", "cabinet",
        "mirror", "plant", "lamp", "wall_art",
    ],
    "office": [
        "desk", "chair", "rug", "ceiling_lamp", "bookshelf", "cabinet",
        "side_table", "plant", "lamp", "wall_art",
    ],
    "dining_room": [
        "dining_table", "chair", "dining_chair", "rug", "ceiling_lamp",
        "cabinet", "side_table", "plant", "wall_art",
    ],
    "kitchen": [
        "cabinet", "dining_table", "chair", "rug", "ceiling_lamp",
        "side_table", "plant", "wall_art",
    ],
    "bathroom": [
        "mirror", "cabinet", "rug", "ceiling_lamp", "plant", "wall_art",
    ],
}


def _density(payload: Dict[str, Any]) -> str:
    return _engine.canonical_text(
        payload.get("furnitureDensity")
        or payload.get("furniture_density")
        or payload.get("density")
        or "medium"
    )


def _density_level(payload: Dict[str, Any]) -> str:
    d = _density(payload)
    if "sparse" in d or "thưa" in d or "thua" in d or "ít" in d or "it" in d:
        return "sparse"
    if "dense" in d or "dày" in d or "day" in d or "nhiều" in d or "nhieu" in d:
        return "dense"
    return "medium"


def _room_area(room: Any) -> float:
    try:
        return float(room.widthM) * float(room.lengthM)
    except Exception:
        return 0.0


def _capacity_for(room: Any, payload: Dict[str, Any]) -> Tuple[int, int]:
    """Return (max_total_items, max_floor_heavy_items)."""
    area = _room_area(room)
    density = _density_level(payload)

    if area <= 0:
        return 8, 6

    if area < 6.0:
        # Example: very small bedroom/office/bathroom. Keep only essentials.
        if density == "sparse":
            return 3, 2
        return 4, 2

    if area < 10.0:
        if density == "sparse":
            return 4, 2
        if density == "medium":
            return 5, 3
        return 6, 3

    # Outside the requested small-room range: do not interfere.
    return 99, 99


def _priority_index(room_type: str, category: str) -> int:
    priority = ROOM_PRIORITY.get(room_type, ROOM_PRIORITY.get("living_room", []))
    try:
        return priority.index(category)
    except ValueError:
        return 999


def _item_area(item: Any) -> float:
    try:
        return float(item.widthM) * float(item.depthM)
    except Exception:
        return 999.0


def _is_floor_heavy(item: Any) -> bool:
    if item.category in FLOOR_HEAVY_CATEGORIES:
        return True
    if getattr(item, "layer", "floor") == "floor" and item.category not in LIGHT_CATEGORIES:
        return True
    return False


def _sort_key(room: Any, item: Any):
    # Keep essentials first. For same category priority, prefer smaller footprint
    # in small rooms to reduce overlap risk.
    return (
        _priority_index(room.type, item.category),
        0 if item.category in LIGHT_CATEGORIES else 1,
        _item_area(item),
        -float(getattr(item, "sourceScore", 0.0) or 0.0),
    )


def _cap_selected_for_small_room(room: Any, selected: List[Any], payload: Dict[str, Any]) -> Tuple[List[Any], List[Dict[str, Any]], Dict[str, Any]]:
    max_total, max_floor_heavy = _capacity_for(room, payload)
    area = _room_area(room)
    info = {
        "enabled": area < 10.0,
        "roomAreaM2": round(area, 4),
        "maxTotalItems": max_total,
        "maxFloorHeavyItems": max_floor_heavy,
        "density": _density_level(payload),
    }

    if not info["enabled"]:
        return selected, [], info

    sorted_items = sorted(selected, key=lambda item: _sort_key(room, item))
    kept: List[Any] = []
    rejected: List[Dict[str, Any]] = []
    floor_heavy_count = 0

    for item in sorted_items:
        is_floor_heavy = _is_floor_heavy(item)
        if len(kept) >= max_total:
            rejected.append({
                "productId": item.productId,
                "name": item.name,
                "category": item.category,
                "reason": "small_room_area_total_capacity_exceeded",
                "roomAreaM2": round(area, 4),
                "maxTotalItems": max_total,
            })
            continue
        if is_floor_heavy and floor_heavy_count >= max_floor_heavy:
            rejected.append({
                "productId": item.productId,
                "name": item.name,
                "category": item.category,
                "reason": "small_room_floor_capacity_exceeded",
                "roomAreaM2": round(area, 4),
                "maxFloorHeavyItems": max_floor_heavy,
            })
            continue
        kept.append(item)
        if is_floor_heavy:
            floor_heavy_count += 1

    info["keptItems"] = len(kept)
    info["keptFloorHeavyItems"] = floor_heavy_count
    info["rejectedBySmallRoomCap"] = len(rejected)
    return kept, rejected, info


def select_products(room: Any, products: List[Any], payload: Dict[str, Any]):
    selected, rejected, missing = _BASE_SELECT_PRODUCTS(room, products, payload)
    capped_selected, small_room_rejected, cap_info = _cap_selected_for_small_room(room, selected, payload)

    if cap_info.get("enabled"):
        kept_ids = {item.productId for item in capped_selected}
        # Add any selected-but-capped items to rejected if not already reported.
        existing_rejected_ids = {str(r.get("productId")) for r in rejected}
        for r in small_room_rejected:
            if str(r.get("productId")) not in existing_rejected_ids:
                rejected.append(r)
        selected = capped_selected
        selected_categories = {item.category for item in selected}
        missing = [m for m in missing if m not in selected_categories]

    return selected, rejected, missing


def finalize_layout(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = _BASE_FINALIZE_LAYOUT(payload)
    room = result.get("room") or {}
    metrics = result.setdefault("metrics", {})

    try:
        area = float(room.get("widthM", 0)) * float(room.get("lengthM", 0))
    except Exception:
        area = 0.0

    metrics["smallRoomCapacityPatchInstalled"] = True
    metrics["smallRoomCapacityRule"] = "area_under_10m2_density_safe_cap_v1"
    metrics["smallRoomAreaM2"] = round(area, 4)
    metrics["smallRoomCapacityApplied"] = area > 0 and area < 10.0
    return result


def model_patch_status() -> Dict[str, Any]:
    status = _BASE_MODEL_PATCH_STATUS()
    status.update({
        "smallRoomCapacityPatchInstalled": _SMALL_ROOM_PATCH_INSTALLED,
        "smallRoomCapacityRule": "area_under_10m2_density_safe_cap_v1",
    })
    return status


def install_small_room_capacity_patch() -> None:
    global _SMALL_ROOM_PATCH_INSTALLED
    if _SMALL_ROOM_PATCH_INSTALLED:
        return
    _engine.select_products = select_products
    _engine.finalize_layout = finalize_layout
    _SMALL_ROOM_PATCH_INSTALLED = True


install_small_room_capacity_patch()
