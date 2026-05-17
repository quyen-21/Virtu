from __future__ import annotations

"""
Category balanced selection patch.

Installed after category_density_quota_patch.

Problem fixed:
- the recommender may return products across 3+ categories;
- previous selection/quota could still end up keeping products from only one
  category because it capped the already-selected list instead of re-balancing
  from the original recommendation pool;
- this patch tries to keep at least one representative product from each useful
  recommended category before allowing a second product from any category.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from layout_engine import engine as _engine
from layout_engine import category_density_quota_patch as _quota_patch

_BASE_SELECT_PRODUCTS = _engine.select_products
_BASE_FINALIZE_LAYOUT = _engine.finalize_layout
_BASE_MODEL_PATCH_STATUS = _quota_patch.model_patch_status

_CATEGORY_BALANCED_SELECTION_PATCH_INSTALLED = False

# Categories that are basically duplicates from a visual/layout point of view.
# We still keep the original category in JSON, but balance by family to avoid
# four chair-like objects being treated as four different useful categories.
CATEGORY_FAMILY = {
    "chair": "seating",
    "armchair": "seating",
    "dining_chair": "seating",
    "stool": "seating",
    "bench": "seating",
    "sofa": "sofa",
    "bed": "bed",
    "nightstand": "bedside",
    "side_table": "small_table",
    "coffee_table": "coffee_table",
    "tv_stand": "media",
    "cabinet": "storage",
    "wardrobe": "storage",
    "bookshelf": "storage",
    "desk": "work_table",
    "dining_table": "dining_table",
    "rug": "rug",
    "mirror": "mirror",
    "ceiling_lamp": "lighting",
    "lamp": "lighting",
    "plant": "decor",
    "wall_art": "decor",
}


def _family(category: str) -> str:
    return CATEGORY_FAMILY.get(category, category)


def _room_area(room: Any) -> float:
    try:
        return float(room.widthM) * float(room.lengthM)
    except Exception:
        return 0.0


def _density_level(payload: Dict[str, Any]) -> str:
    return _quota_patch._density_level(payload)


def _max_total(room: Any, payload: Dict[str, Any]) -> int:
    return _quota_patch._max_total_by_area_density(_room_area(room), _density_level(payload))


def _style(room: Any, payload: Dict[str, Any]) -> str:
    return _quota_patch._style_text(room, payload)


def _quota_for(room: Any, payload: Dict[str, Any], category: str) -> int:
    return _quota_patch._base_category_quota(category, room.type, _density_level(payload), _room_area(room), _style(room, payload))


def _allowed_categories(room: Any) -> set[str]:
    comp = _engine.ROOM_COMPOSITIONS.get(room.type) or _engine.ROOM_COMPOSITIONS.get("living_room") or {}
    return set(comp.get("allowed", set()))


def _priority(room: Any, category: str) -> int:
    return _quota_patch._priority_index(room.type, category)


def _item_area(item: Any) -> float:
    return _quota_patch._item_area(item)


def _source_score(item: Any) -> float:
    try:
        return float(getattr(item, "sourceScore", 0.0) or 0.0)
    except Exception:
        return 0.0


def _is_eligible(room: Any, item: Any) -> bool:
    allowed = _allowed_categories(room)
    if item.category not in allowed:
        return False
    # Avoid extremely large items for the room even if recommender returns them.
    area = max(_room_area(room), 0.01)
    if _item_area(item) > area * 0.42:
        return False
    return True


def _best_item_for_category(room: Any, products: List[Any], category: str, used_ids: set[str]) -> Optional[Any]:
    candidates = [p for p in products if p.productId not in used_ids and p.category == category and _is_eligible(room, p)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (-_source_score(p), _item_area(p), _priority(room, p.category)))
    return candidates[0]


def _counts_by_category(items: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        counts[item.category] = counts.get(item.category, 0) + 1
    return counts


def _counts_by_family(items: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        fam = _family(item.category)
        counts[fam] = counts.get(fam, 0) + 1
    return counts


def _recommended_categories(room: Any, products: List[Any]) -> List[str]:
    categories = []
    seen = set()
    for item in sorted(products, key=lambda p: (_priority(room, p.category), -_source_score(p), _item_area(p))):
        if not _is_eligible(room, item):
            continue
        if item.category in seen:
            continue
        seen.add(item.category)
        categories.append(item.category)
    return categories


def _drop_candidate(room: Any, selected: List[Any], protected_ids: set[str]) -> Optional[Any]:
    if not selected:
        return None
    cat_counts = _counts_by_category(selected)
    fam_counts = _counts_by_family(selected)

    candidates = [item for item in selected if item.productId not in protected_ids]
    if not candidates:
        candidates = selected[:]

    # Drop duplicates/family duplicates first, then lowest priority / lowest score.
    candidates.sort(
        key=lambda item: (
            0 if cat_counts.get(item.category, 0) > 1 else 1,
            0 if fam_counts.get(_family(item.category), 0) > 1 else 1,
            -_priority(room, item.category),
            _source_score(item),
            -_item_area(item),
        )
    )
    return candidates[0] if candidates else None


def _append_or_replace(room: Any, selected: List[Any], item: Any, payload: Dict[str, Any], protected_ids: set[str]) -> Tuple[List[Any], Optional[Dict[str, Any]]]:
    max_total = _max_total(room, payload)
    if len(selected) < max_total:
        selected.append(item)
        protected_ids.add(item.productId)
        return selected, None

    drop = _drop_candidate(room, selected, protected_ids)
    if drop is None:
        return selected, {
            "productId": item.productId,
            "name": item.name,
            "category": item.category,
            "reason": "category_balance_no_replace_slot_available",
        }

    selected = [x for x in selected if x.productId != drop.productId]
    selected.append(item)
    protected_ids.add(item.productId)
    return selected, {
        "productId": drop.productId,
        "name": drop.name,
        "category": drop.category,
        "reason": "category_balance_replaced_by_missing_recommend_category",
        "replacedByProductId": item.productId,
        "replacedByCategory": item.category,
    }


def _balance_from_recommend_categories(room: Any, selected: List[Any], products: List[Any], payload: Dict[str, Any]) -> Tuple[List[Any], List[Dict[str, Any]], Dict[str, Any]]:
    recommended_categories = _recommended_categories(room, products)
    max_total = _max_total(room, payload)
    density = _density_level(payload)

    info = {
        "recommendedCategories": recommended_categories,
        "recommendedCategoryCount": len(recommended_categories),
        "maxTotalItems": max_total,
        "density": density,
        "mode": "represent_each_recommended_category_before_second_item_v1",
    }

    if not recommended_categories:
        info["keptCategories"] = sorted({x.category for x in selected})
        info["balancedAdded"] = 0
        return selected, [], info

    selected_ids = {x.productId for x in selected}
    selected_categories = {x.category for x in selected}
    protected_ids: set[str] = set()
    balance_rejected: List[Dict[str, Any]] = []
    added = 0

    # First pass: ensure each recommended category has at least one representative,
    # limited by total capacity.
    for category in recommended_categories:
        if category in selected_categories:
            for item in selected:
                if item.category == category:
                    protected_ids.add(item.productId)
                    break
            continue

        candidate = _best_item_for_category(room, products, category, selected_ids)
        if candidate is None:
            continue

        selected, replaced = _append_or_replace(room, selected, candidate, payload, protected_ids)
        selected_ids = {x.productId for x in selected}
        selected_categories = {x.category for x in selected}
        if replaced:
            balance_rejected.append(replaced)
        added += 1

    # Second pass: avoid one visual family dominating the whole layout.
    # Example: chair + armchair + stool + bench can otherwise look like 4 same things.
    fam_counts = _counts_by_family(selected)
    max_family = 2 if density == "dense" and _room_area(room) >= 14.0 else 1
    for fam, count in list(fam_counts.items()):
        while count > max_family:
            duplicate_items = [x for x in selected if _family(x.category) == fam and x.productId not in protected_ids]
            if not duplicate_items:
                break
            duplicate_items.sort(key=lambda item: (_source_score(item), -_item_area(item), -_priority(room, item.category)))
            drop = duplicate_items[0]
            selected = [x for x in selected if x.productId != drop.productId]
            balance_rejected.append({
                "productId": drop.productId,
                "name": drop.name,
                "category": drop.category,
                "reason": "category_family_dominance_exceeded",
                "family": fam,
                "maxFamilyItems": max_family,
            })
            count -= 1

    # Final re-apply strict category quota from v2.8 logic.
    selected, quota_rejected, quota_info = _quota_patch._apply_category_density_quota(room, selected, payload)
    balance_rejected.extend(quota_rejected)

    info["balancedAdded"] = added
    info["keptCategories"] = sorted({x.category for x in selected})
    info["keptFamilies"] = dict(_counts_by_family(selected))
    info["quotaInfo"] = quota_info
    return selected, balance_rejected, info


def select_products(room: Any, products: List[Any], payload: Dict[str, Any]):
    selected, rejected, missing = _BASE_SELECT_PRODUCTS(room, products, payload)
    balanced_selected, balance_rejected, balance_info = _balance_from_recommend_categories(room, selected, products, payload)

    selected = balanced_selected
    existing_rejected_ids = {str(r.get("productId")) for r in rejected}
    for r in balance_rejected:
        if str(r.get("productId")) not in existing_rejected_ids:
            rejected.append(r)

    selected_categories = {item.category for item in selected}
    missing = [m for m in missing if m not in selected_categories]

    try:
        payload["_categoryBalancedSelectionInfo"] = balance_info
    except Exception:
        pass

    return selected, rejected, missing


def finalize_layout(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = _BASE_FINALIZE_LAYOUT(payload)
    metrics = result.setdefault("metrics", {})
    balance_info = payload.get("_categoryBalancedSelectionInfo") if isinstance(payload, dict) else None

    metrics["categoryBalancedSelectionPatchInstalled"] = True
    metrics["categoryBalancedSelectionRule"] = "represent_each_recommended_category_before_second_item_v1"
    if isinstance(balance_info, dict):
        metrics["categoryBalancedSelection"] = balance_info
    return result


def model_patch_status() -> Dict[str, Any]:
    status = _BASE_MODEL_PATCH_STATUS()
    status.update({
        "categoryBalancedSelectionPatchInstalled": _CATEGORY_BALANCED_SELECTION_PATCH_INSTALLED,
        "categoryBalancedSelectionRule": "represent_each_recommended_category_before_second_item_v1",
    })
    return status


def install_category_balanced_selection_patch() -> None:
    global _CATEGORY_BALANCED_SELECTION_PATCH_INSTALLED
    if _CATEGORY_BALANCED_SELECTION_PATCH_INSTALLED:
        return
    _engine.select_products = select_products
    _engine.finalize_layout = finalize_layout
    _CATEGORY_BALANCED_SELECTION_PATCH_INSTALLED = True


install_category_balanced_selection_patch()
