from __future__ import annotations

"""
Category density quota patch.

Installed after small_room_capacity_patch.

Problem fixed:
- external AI recommender can return many products from the same category;
- layout should not blindly place many repeated items;
- for every room/style/size/density, keep about 1-2 products per useful category,
  while still allowing special pairs like 2 nightstands or 2 dining chairs.

This patch applies a final category quota after all previous selection logic:
- sparse: mostly 1 item per category;
- medium: 1 item per category, 2 only for pair-friendly categories;
- dense: up to 2 for pair-friendly categories, but still area-aware;
- small rooms remain stricter because small_room_capacity_patch already runs first.
"""

from typing import Any, Dict, List, Tuple

from layout_engine import engine as _engine
from layout_engine import small_room_capacity_patch as _small_room_patch

_BASE_SELECT_PRODUCTS = _engine.select_products
_BASE_FINALIZE_LAYOUT = _engine.finalize_layout
_BASE_MODEL_PATCH_STATUS = _small_room_patch.model_patch_status

_CATEGORY_DENSITY_QUOTA_PATCH_INSTALLED = False

# Some categories should almost never appear more than once in one layout.
SINGLETON_CATEGORIES = {
    "bed",
    "sofa",
    "coffee_table",
    "tv_stand",
    "dining_table",
    "desk",
    "wardrobe",
    "rug",
    "mirror",
    "bookshelf",
}

# Pair-friendly categories may appear twice, but only when density/area allow it.
PAIR_FRIENDLY_CATEGORIES = {
    "nightstand",
    "armchair",
    "chair",
    "dining_chair",
    "side_table",
    "ceiling_lamp",
    "lamp",
    "plant",
    "wall_art",
    "cabinet",
    "stool",
    "bench",
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
    "dining_room": [
        "dining_table", "dining_chair", "chair", "rug", "ceiling_lamp",
        "cabinet", "side_table", "plant", "wall_art",
    ],
    "office": [
        "desk", "chair", "rug", "ceiling_lamp", "bookshelf",
        "cabinet", "side_table", "plant", "lamp", "wall_art",
    ],
    "kitchen": [
        "cabinet", "dining_table", "chair", "dining_chair", "rug",
        "ceiling_lamp", "side_table", "plant", "wall_art",
    ],
    "bathroom": [
        "mirror", "cabinet", "rug", "ceiling_lamp", "plant", "wall_art",
    ],
}

STYLE_DECOR_CATEGORIES = {"plant", "wall_art", "lamp", "ceiling_lamp", "mirror"}
MINIMAL_STYLE_KEYWORDS = {"minimal", "minimalist", "tối giản", "toi gian", "scandinavian"}
LUXURY_STYLE_KEYWORDS = {"luxury", "sang trọng", "sang trong", "classic", "tân cổ điển", "tan co dien"}


def _canonical(value: Any) -> str:
    return _engine.canonical_text(value or "")


def _density_level(payload: Dict[str, Any]) -> str:
    d = _canonical(
        payload.get("furnitureDensity")
        or payload.get("furniture_density")
        or payload.get("density")
        or "medium"
    )
    if "sparse" in d or "thưa" in d or "thua" in d or "ít" in d or "it" in d:
        return "sparse"
    if "dense" in d or "dày" in d or "day" in d or "nhiều" in d or "nhieu" in d:
        return "dense"
    return "medium"


def _style_text(room: Any, payload: Dict[str, Any]) -> str:
    return _canonical(
        getattr(room, "style", "")
        or payload.get("style")
        or payload.get("roomStyle")
        or payload.get("designStyle")
        or ""
    )


def _room_area(room: Any) -> float:
    try:
        return float(room.widthM) * float(room.lengthM)
    except Exception:
        return 0.0


def _area_tier(area: float) -> str:
    if area < 6.0:
        return "tiny"
    if area < 10.0:
        return "small"
    if area < 18.0:
        return "compact"
    if area < 35.0:
        return "normal"
    return "large"


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


def _max_total_by_area_density(area: float, density: str) -> int:
    tier = _area_tier(area)
    if tier == "tiny":
        return 3 if density == "sparse" else 4
    if tier == "small":
        return 4 if density == "sparse" else 5 if density == "medium" else 6
    if tier == "compact":
        return 5 if density == "sparse" else 7 if density == "medium" else 8
    if tier == "normal":
        return 7 if density == "sparse" else 9 if density == "medium" else 11
    return 8 if density == "sparse" else 11 if density == "medium" else 13


def _base_category_quota(category: str, room_type: str, density: str, area: float, style: str) -> int:
    tier = _area_tier(area)

    if category in SINGLETON_CATEGORIES:
        return 1

    if category not in PAIR_FRIENDLY_CATEGORIES:
        return 1

    # Tiny rooms should avoid pairs except required bedroom nightstands can still
    # become one only, because fitting two often causes overlap.
    if tier == "tiny":
        return 1

    if density == "sparse":
        return 1

    # Room-specific natural pairs.
    if room_type == "bedroom" and category == "nightstand":
        return 2 if area >= 10.0 and density in {"medium", "dense"} else 1
    if room_type == "dining_room" and category in {"chair", "dining_chair"}:
        if area < 10.0:
            return 2
        return 2 if density in {"medium", "dense"} else 1
    if room_type == "living_room" and category in {"armchair", "chair", "side_table"}:
        return 2 if density == "dense" and area >= 14.0 else 1
    if category == "ceiling_lamp":
        return 2 if density == "dense" and area >= 18.0 else 1

    # Style can allow one extra decor item in bigger rooms, not floor-heavy items.
    if category in STYLE_DECOR_CATEGORIES:
        if any(k in style for k in MINIMAL_STYLE_KEYWORDS):
            return 1
        if any(k in style for k in LUXURY_STYLE_KEYWORDS) and density == "dense" and area >= 18.0:
            return 2

    return 2 if density == "dense" and area >= 18.0 else 1


def _sort_for_keep(room: Any, item: Any):
    # Keep essential categories first, then higher source score, then smaller area.
    return (
        _priority_index(room.type, item.category),
        -float(getattr(item, "sourceScore", 0.0) or 0.0),
        _item_area(item),
    )


def _apply_category_density_quota(room: Any, selected: List[Any], payload: Dict[str, Any]) -> Tuple[List[Any], List[Dict[str, Any]], Dict[str, Any]]:
    area = _room_area(room)
    density = _density_level(payload)
    style = _style_text(room, payload)
    max_total = _max_total_by_area_density(area, density)

    info = {
        "roomAreaM2": round(area, 4),
        "density": density,
        "style": style,
        "maxTotalItems": max_total,
        "categoryQuotaMode": "one_to_two_per_category_area_density_style_v1",
    }

    sorted_items = sorted(selected, key=lambda item: _sort_for_keep(room, item))
    kept: List[Any] = []
    rejected: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}

    for item in sorted_items:
        quota = _base_category_quota(item.category, room.type, density, area, style)
        current = counts.get(item.category, 0)

        if current >= quota:
            rejected.append({
                "productId": item.productId,
                "name": item.name,
                "category": item.category,
                "reason": "category_density_quota_exceeded",
                "categoryQuota": quota,
                "density": density,
                "roomAreaM2": round(area, 4),
            })
            continue

        if len(kept) >= max_total:
            rejected.append({
                "productId": item.productId,
                "name": item.name,
                "category": item.category,
                "reason": "room_density_total_quota_exceeded",
                "maxTotalItems": max_total,
                "density": density,
                "roomAreaM2": round(area, 4),
            })
            continue

        kept.append(item)
        counts[item.category] = current + 1

    info["keptItems"] = len(kept)
    info["rejectedByCategoryQuota"] = len(rejected)
    info["categoryCounts"] = dict(counts)
    return kept, rejected, info


def select_products(room: Any, products: List[Any], payload: Dict[str, Any]):
    selected, rejected, missing = _BASE_SELECT_PRODUCTS(room, products, payload)
    capped_selected, quota_rejected, quota_info = _apply_category_density_quota(room, selected, payload)

    selected = capped_selected
    existing_rejected_ids = {str(r.get("productId")) for r in rejected}
    for r in quota_rejected:
        if str(r.get("productId")) not in existing_rejected_ids:
            rejected.append(r)

    selected_categories = {item.category for item in selected}
    missing = [m for m in missing if m not in selected_categories]

    # Store quota info on payload so finalize_layout can expose it in metrics.
    try:
        payload["_categoryDensityQuotaInfo"] = quota_info
    except Exception:
        pass

    return selected, rejected, missing


def finalize_layout(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = _BASE_FINALIZE_LAYOUT(payload)
    metrics = result.setdefault("metrics", {})
    quota_info = payload.get("_categoryDensityQuotaInfo") if isinstance(payload, dict) else None

    metrics["categoryDensityQuotaPatchInstalled"] = True
    metrics["categoryDensityQuotaRule"] = "one_to_two_per_category_area_density_style_v1"
    if isinstance(quota_info, dict):
        metrics["categoryDensityQuota"] = quota_info
    return result


def model_patch_status() -> Dict[str, Any]:
    status = _BASE_MODEL_PATCH_STATUS()
    status.update({
        "categoryDensityQuotaPatchInstalled": _CATEGORY_DENSITY_QUOTA_PATCH_INSTALLED,
        "categoryDensityQuotaRule": "one_to_two_per_category_area_density_style_v1",
    })
    return status


def install_category_density_quota_patch() -> None:
    global _CATEGORY_DENSITY_QUOTA_PATCH_INSTALLED
    if _CATEGORY_DENSITY_QUOTA_PATCH_INSTALLED:
        return
    _engine.select_products = select_products
    _engine.finalize_layout = finalize_layout
    _CATEGORY_DENSITY_QUOTA_PATCH_INSTALLED = True


install_category_density_quota_patch()
