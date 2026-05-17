from __future__ import annotations

"""
Living-room semantic role patch.

This layer is installed after layout_quality_patch and focuses on a common
recommendation/layout mismatch:

Recommendation may return Vietnamese/e-commerce categories such as:
- Bàn bên
- Bàn console
- Tủ lưu trữ
- Kệ phòng khách

But the layout template needs semantic roles:
- coffee_table
- tv_stand

The patch maps product categories and product names to semantic room roles before
selection, so layout does not reject useful products only because the category
label is not exactly the expected template role.
"""

from typing import Any, Dict, Iterable, List, Optional

from layout_engine import engine as _engine
from layout_engine import layout_quality_patch as _quality_patch

_BASE_NORMALIZE_PRODUCT = _engine.normalize_product
_BASE_SELECT_PRODUCTS = _engine.select_products
_BASE_FINALIZE_LAYOUT = _engine.finalize_layout
_BASE_MODEL_PATCH_STATUS = _quality_patch.model_patch_status

_SEMANTIC_PATCH_INSTALLED = False


UNDERSCORE_CATEGORY_ALIAS = {
    "ban_console": "side_table",
    "bàn_console": "side_table",
    "console_table": "side_table",
    "ban_ben": "side_table",
    "bàn_bên": "side_table",
    "ban_phu": "side_table",
    "bàn_phụ": "side_table",

    "ke_phong_khach": "bookshelf",
    "kệ_phòng_khách": "bookshelf",
    "ke_hangar": "bookshelf",
    "kệ_hangar": "bookshelf",
    "ke_luu_tru": "bookshelf",
    "kệ_lưu_trữ": "bookshelf",
    "ke_sach": "bookshelf",
    "kệ_sách": "bookshelf",
    "tu_trung_bay": "bookshelf",
    "tủ_trưng_bày": "bookshelf",

    "tu_luu_tru": "cabinet",
    "tủ_lưu_trữ": "cabinet",
    "hoc_keo": "cabinet",
    "hộc_kéo": "cabinet",
    "ngan_keo": "cabinet",
    "ngăn_kéo": "cabinet",

    "sofa_goc": "sofa",
    "sofa_góc": "sofa",
    "tham": "rug",
    "thảm": "rug",
    "guong": "mirror",
    "gương": "mirror",
}


def _canonical_text(value: Any) -> str:
    return _engine.canonical_text(value or "")


def _text_variants(*values: Any) -> str:
    parts: List[str] = []
    for value in values:
        text = _canonical_text(value)
        parts.append(text)
        parts.append(text.replace("_", " "))
    return " ".join(parts)


def _contains(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _raw_category_and_name(product: Any) -> str:
    raw = product.raw if isinstance(getattr(product, "raw", None), dict) else {}
    return _text_variants(
        product.category,
        product.name,
        raw.get("category"),
        raw.get("type"),
        raw.get("productType"),
        raw.get("name"),
        raw.get("title"),
    )


def _looks_like_console(text: str) -> bool:
    return _contains(text, ["console", "bàn console", "ban console", "bàn_console", "ban_console"])


def _looks_like_living_shelf(text: str) -> bool:
    return _contains(
        text,
        [
            "kệ phòng khách", "ke phong khach", "kệ_phòng_khách", "ke_phong_khach",
            "kệ hangar", "ke hangar", "kệ_hangar", "ke_hangar",
            "tủ trưng bày", "tu trung bay", "tủ_trưng_bày", "tu_trung_bay",
            "media", "tv", "tivi", "television",
        ],
    )


def _looks_like_side_table(text: str) -> bool:
    return _contains(
        text,
        [
            "bàn bên", "ban ben", "bàn_bên", "ban_ben",
            "bàn phụ", "ban phu", "bàn_phụ", "ban_phu",
            "side table", "side_table", "stellar", "magazine",
        ],
    )


def _looks_like_storage(text: str) -> bool:
    return _contains(
        text,
        [
            "tủ lưu trữ", "tu luu tru", "tủ_lưu_trữ", "tu_luu_tru",
            "hộc kéo", "hoc keo", "hộc_kéo", "hoc_keo",
            "ngăn kéo", "ngan keo", "ngăn_kéo", "ngan_keo",
            "cabinet", "stocker",
        ],
    )


def _is_low_long(product: Any) -> bool:
    long_side = max(product.widthM, product.depthM)
    short_side = min(product.widthM, product.depthM)
    return long_side >= 0.75 and short_side >= 0.25 and product.heightM <= 0.90


def _is_small_table(product: Any) -> bool:
    area = product.widthM * product.depthM
    return 0.12 <= area <= 1.40 and 0.25 <= product.heightM <= 0.75


def normalize_product(product: Dict[str, Any], index: int, model_url_by_id: Optional[Dict[str, str]] = None):
    p = _BASE_NORMALIZE_PRODUCT(product, index, model_url_by_id)
    text = _raw_category_and_name(p)
    raw_cat = _canonical_text((p.raw or {}).get("category", ""))

    # Fix category labels that were already transformed into underscore forms
    # before alias matching, e.g. bàn_console, kệ_phòng_khách.
    if p.category in UNDERSCORE_CATEGORY_ALIAS:
        p.category = UNDERSCORE_CATEGORY_ALIAS[p.category]
    elif raw_cat in UNDERSCORE_CATEGORY_ALIAS:
        p.category = UNDERSCORE_CATEGORY_ALIAS[raw_cat]
    elif _looks_like_console(text):
        p.category = "side_table"
    elif _looks_like_living_shelf(text):
        p.category = "bookshelf"
    elif _looks_like_storage(text):
        p.category = "cabinet"
    elif _looks_like_side_table(text):
        p.category = "side_table"

    return p


def _boost_semantic_score(product: Any, amount: float) -> None:
    try:
        product.sourceScore = min(1.0, float(product.sourceScore) + amount)
    except Exception:
        product.sourceScore = amount


def _mark_semantic_role(product: Any, new_category: str, reason: str) -> None:
    old_category = product.category
    product.category = new_category
    raw = product.raw if isinstance(getattr(product, "raw", None), dict) else {}
    raw["layoutSemanticRole"] = new_category
    raw["layoutOriginalCategory"] = old_category
    raw["layoutSemanticReason"] = reason
    product.raw = raw
    _boost_semantic_score(product, 0.10)


def _choose_best_coffee_table(products: List[Any], used_ids: set[str]) -> Optional[Any]:
    candidates = []
    for p in products:
        if p.productId in used_ids:
            continue
        text = _raw_category_and_name(p)
        if p.category in {"coffee_table", "table"}:
            candidates.append((3.0, p))
        elif p.category == "side_table" and _is_small_table(p):
            name_bonus = 0.8 if _looks_like_side_table(text) else 0.0
            console_penalty = -0.6 if _looks_like_console(text) else 0.0
            size_score = min(1.0, (p.widthM * p.depthM) / 0.45)
            candidates.append((2.0 + name_bonus + console_penalty + size_score + p.sourceScore, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _choose_best_tv_stand(products: List[Any], used_ids: set[str]) -> Optional[Any]:
    candidates = []
    for p in products:
        if p.productId in used_ids:
            continue
        text = _raw_category_and_name(p)
        if p.category == "tv_stand":
            candidates.append((5.0 + p.sourceScore, p))
            continue

        # Console, media shelf, low living-room shelf, or low display cabinet can
        # work as tv stand. Tall bookcases should remain bookshelf.
        candidate_role = _looks_like_console(text) or _looks_like_living_shelf(text) or p.category in {"bookshelf", "cabinet", "side_table"}
        if not candidate_role:
            continue

        low_long = _is_low_long(p)
        height_ok = p.heightM <= 1.10
        if not (low_long and height_ok):
            continue

        score = p.sourceScore
        if _looks_like_console(text):
            score += 2.2
        if _looks_like_living_shelf(text):
            score += 1.8
        if p.category == "cabinet":
            score += 0.6
        if p.category == "bookshelf":
            score += 0.4
        score += min(1.2, max(p.widthM, p.depthM) / 1.2)
        candidates.append((score, p))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _ensure_living_room_semantic_roles(products: List[Any]) -> Dict[str, str]:
    notes: Dict[str, str] = {}
    used_ids: set[str] = set()

    has_coffee = any(p.category == "coffee_table" for p in products)
    has_tv = any(p.category == "tv_stand" for p in products)

    if not has_tv:
        tv_product = _choose_best_tv_stand(products, used_ids)
        if tv_product is not None:
            used_ids.add(tv_product.productId)
            _mark_semantic_role(
                tv_product,
                "tv_stand",
                "Mapped from console/storage/living-room shelf because living_room needs a focal TV/media stand.",
            )
            notes[tv_product.productId] = "semantic_tv_stand"

    if not has_coffee:
        coffee_product = _choose_best_coffee_table(products, used_ids)
        if coffee_product is not None:
            used_ids.add(coffee_product.productId)
            _mark_semantic_role(
                coffee_product,
                "coffee_table",
                "Mapped from side table/table because living_room needs a central coffee table.",
            )
            notes[coffee_product.productId] = "semantic_coffee_table"

    return notes


def _allow_living_room_semantic_categories() -> None:
    living = _engine.ROOM_COMPOSITIONS.get("living_room")
    if not living:
        return
    for cat in ["coffee_table", "tv_stand", "side_table", "cabinet", "bookshelf", "mirror", "rug"]:
        living["allowed"].add(cat)
    living["quota"]["coffee_table"] = max(int(living["quota"].get("coffee_table", 1)), 1)
    living["quota"]["tv_stand"] = max(int(living["quota"].get("tv_stand", 1)), 1)
    living["quota"]["side_table"] = max(int(living["quota"].get("side_table", 2)), 2)
    living["quota"]["cabinet"] = max(int(living["quota"].get("cabinet", 1)), 1)
    living["quota"]["bookshelf"] = max(int(living["quota"].get("bookshelf", 1)), 1)
    living["quota"]["mirror"] = max(int(living["quota"].get("mirror", 1)), 1)

    preferred = ["sofa", "coffee_table", "tv_stand", "rug", "armchair", "chair", "side_table", "cabinet", "bookshelf", "mirror"]
    old_priority = [cat for cat in living.get("priority", []) if cat not in preferred]
    living["priority"] = preferred + old_priority


def select_products(room: Any, products: List[Any], payload: Dict[str, Any]):
    semantic_notes: Dict[str, str] = {}
    if room.type == "living_room":
        semantic_notes = _ensure_living_room_semantic_roles(products)

    selected, rejected, missing = _BASE_SELECT_PRODUCTS(room, products, payload)

    if room.type == "living_room":
        selected_categories = {p.category for p in selected}
        missing = [m for m in missing if m not in selected_categories]
        for item in selected:
            note = semantic_notes.get(item.productId)
            if note:
                raw = item.raw if isinstance(getattr(item, "raw", None), dict) else {}
                item.raw = raw

        # If real semantic products were found, remove stale rejection entries for
        # their old category labels to avoid confusing the API consumer.
        semantic_ids = set(semantic_notes.keys())
        rejected = [r for r in rejected if str(r.get("productId")) not in semantic_ids]

    return selected, rejected, missing


def finalize_layout(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = _BASE_FINALIZE_LAYOUT(payload)
    metrics = result.setdefault("metrics", {})
    metrics["livingRoomSemanticPatchInstalled"] = True
    metrics["semanticRoleMapping"] = "console_side_storage_to_coffee_table_tv_stand_v1"
    return result


def model_patch_status() -> Dict[str, Any]:
    status = _BASE_MODEL_PATCH_STATUS()
    status.update({
        "livingRoomSemanticPatchInstalled": _SEMANTIC_PATCH_INSTALLED,
        "semanticRoleMapping": "console_side_storage_to_coffee_table_tv_stand_v1",
    })
    return status


def install_living_room_semantic_patch() -> None:
    global _SEMANTIC_PATCH_INSTALLED
    if _SEMANTIC_PATCH_INSTALLED:
        return
    _engine.CATEGORY_ALIAS.update(UNDERSCORE_CATEGORY_ALIAS)
    _allow_living_room_semantic_categories()
    _engine.normalize_product = normalize_product
    _engine.select_products = select_products
    _engine.finalize_layout = finalize_layout
    _SEMANTIC_PATCH_INSTALLED = True


install_living_room_semantic_patch()
