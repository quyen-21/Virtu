from __future__ import annotations

"""
Additional quality patches installed after engine_patch.

This layer fixes issues found in local visual tests:
- product dimensions from DB are often cm/mm, not meters;
- Vietnamese categories need richer aliases;
- ceiling lamps must be placed on the ceiling, not on nightstands;
- mattress products should not be selected as a separate bed when a bed exists;
- high layout scores must be capped when essential room composition is missing;
- large/dense bedrooms need secondary zones instead of all furniture being visually sparse.
"""

import math
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from layout_engine import engine as _engine
from layout_engine import engine_patch as _base_patch

_BASE_NORMALIZE_PRODUCT = _engine.normalize_product
_BASE_NORMALIZE_PAYLOAD = _engine.normalize_payload
_BASE_SELECT_PRODUCTS = _engine.select_products
_BASE_GENERATE_CANDIDATES = _engine.generate_candidates
_BASE_SCORE_LAYOUT = _engine.score_layout
_BASE_FINALIZE_LAYOUT = _engine.finalize_layout
_BASE_MODEL_PATCH_STATUS = _base_patch.model_patch_status

_CURRENT_PAYLOAD: Dict[str, Any] = {}
_QUALITY_PATCH_INSTALLED = False


# ---------------------------------------------------------------------------
# 1) Category aliases and room compositions
# ---------------------------------------------------------------------------

EXTRA_CATEGORY_ALIAS = {
    # living room / storage
    "sofa góc": "sofa",
    "sofa goc": "sofa",
    "sectional": "sofa",
    "bàn console": "side_table",
    "ban console": "side_table",
    "console table": "side_table",
    "console": "side_table",
    "kệ phòng khách": "bookshelf",
    "ke phong khach": "bookshelf",
    "kệ hangar": "bookshelf",
    "ke hangar": "bookshelf",
    "kệ lưu trữ": "bookshelf",
    "ke luu tru": "bookshelf",
    "kệ": "bookshelf",
    "ke": "bookshelf",
    "tủ lưu trữ": "cabinet",
    "tu luu tru": "cabinet",
    "hộc kéo": "cabinet",
    "hoc keo": "cabinet",
    "ngăn kéo": "cabinet",
    "ngan keo": "cabinet",
    "pallet": "cabinet",

    # bedroom
    "nệm": "mattress",
    "nem": "mattress",
    "mattress": "mattress",
    "đệm": "mattress",
    "dem": "mattress",
    "bàn đầu giường": "nightstand",
    "ban dau giuong": "nightstand",
    "tủ đầu giường": "nightstand",
    "tu dau giuong": "nightstand",

    # lighting
    "đèn trần": "ceiling_lamp",
    "den tran": "ceiling_lamp",
    "đèn áp trần": "ceiling_lamp",
    "den ap tran": "ceiling_lamp",
    "áp trần": "ceiling_lamp",
    "ap tran": "ceiling_lamp",
    "ceiling lamp": "ceiling_lamp",
    "pendant lamp": "ceiling_lamp",
    "pendant": "ceiling_lamp",
    "chandelier": "ceiling_lamp",
    "đèn chùm": "ceiling_lamp",
    "den chum": "ceiling_lamp",
    "đèn trang trí": "lamp",
    "den trang tri": "lamp",

    # wall / mirror
    "gương đứng": "mirror",
    "guong dung": "mirror",
    "gương treo": "mirror",
    "guong treo": "mirror",
}


def _install_aliases_and_compositions() -> None:
    _engine.CATEGORY_ALIAS.update(EXTRA_CATEGORY_ALIAS)
    _engine.DEFAULT_DIMS.update({
        "mattress": (1.80, 2.00, 0.25),
        "ceiling_lamp": (0.50, 0.50, 0.30),
    })

    for room_type in ["living_room", "bedroom", "dining_room", "office", "kitchen", "bathroom"]:
        comp = _engine.ROOM_COMPOSITIONS.get(room_type)
        if not comp:
            continue
        comp["allowed"].add("ceiling_lamp")
        comp["quota"]["ceiling_lamp"] = 2 if room_type in {"living_room", "bedroom", "dining_room"} else 1
        if "ceiling_lamp" not in comp["priority"]:
            # Put ceiling lighting after the essential furniture, before small decor.
            insert_at = max(1, len(comp["priority"]) - 3)
            comp["priority"].insert(insert_at, "ceiling_lamp")

    bedroom = _engine.ROOM_COMPOSITIONS.get("bedroom")
    if bedroom:
        # Dense bedroom should prefer storage before loose decor.
        for cat in ["cabinet", "bookshelf"]:
            bedroom["allowed"].add(cat)
            bedroom["quota"][cat] = max(int(bedroom["quota"].get(cat, 1)), 1)
        bedroom["priority"] = [
            "bed", "nightstand", "wardrobe", "rug", "cabinet", "bookshelf",
            "desk", "chair", "armchair", "mirror", "ceiling_lamp", "lamp",
            "side_table", "tv_stand", "plant", "wall_art",
        ]


# ---------------------------------------------------------------------------
# 2) Robust product normalization
# ---------------------------------------------------------------------------

def _text(x: Any) -> str:
    return _engine.canonical_text(x)


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(n in text for n in needles)


def _infer_category(product: Dict[str, Any]) -> str:
    raw_cat = _engine.first_value(product.get("category"), product.get("type"), product.get("productType"), default="unknown")
    name = product.get("name") or product.get("title") or ""
    combined = f"{raw_cat} {name}"
    ctext = _text(combined)

    if _contains_any(ctext, ["đèn trần", "den tran", "đèn áp trần", "den ap tran", "áp trần", "ap tran", "ceiling", "pendant", "chandelier", "đèn chùm", "den chum"]):
        return "ceiling_lamp"
    if _contains_any(ctext, ["nệm", "nem", "mattress", "đệm", "dem"]):
        return "mattress"
    if _contains_any(ctext, ["sofa góc", "sofa goc", "sectional"]):
        return "sofa"
    if _contains_any(ctext, ["bàn console", "ban console", "console"]):
        return "side_table"
    if _contains_any(ctext, ["kệ phòng khách", "ke phong khach", "kệ lưu trữ", "ke luu tru", "kệ hangar", "ke hangar"]):
        return "bookshelf"
    if _contains_any(ctext, ["tủ lưu trữ", "tu luu tru", "hộc kéo", "hoc keo", "ngăn kéo", "ngan keo"]):
        return "cabinet"

    return _engine.canonical_category(raw_cat)


def _raw_dimension_value(product: Dict[str, Any], keys: Iterable[str], dims: Dict[str, Any], default: float) -> Any:
    for key in keys:
        if key in product and product.get(key) not in (None, ""):
            return product.get(key)
    for key in keys:
        if key in dims and dims.get(key) not in (None, ""):
            return dims.get(key)
    return default


def _product_dim_to_m(value: Any, default: float) -> float:
    """
    Product DB values are usually cm, sometimes mm, while API values can be meters.
    Use product-only heuristic; do not apply this to room dimensions.
    """
    v = _engine.as_float(value, default)
    if v <= 0:
        return default
    if v > 500:
        return v / 1000.0  # millimeters, e.g. 1800 -> 1.8m
    if v > 10:
        return v / 100.0   # centimeters, e.g. 175 -> 1.75m, 12 -> 0.12m
    return v               # already meters, e.g. 1.8, 0.44


def normalize_product(product: Dict[str, Any], index: int, model_url_by_id: Optional[Dict[str, str]] = None):
    model_url_by_id = model_url_by_id or {}
    category = _infer_category(product)
    default_w, default_d, default_h = _engine.DEFAULT_DIMS.get(category, (0.80, 0.60, 0.70))
    dims = product.get("dimensions") or product.get("dimension") or {}
    if not isinstance(dims, dict):
        dims = {}

    raw_w = _raw_dimension_value(product, ["widthM", "width_m", "item_width_m", "width", "w"], dims, default_w)
    raw_d = _raw_dimension_value(product, ["depthM", "depth_m", "lengthM", "length_m", "item_depth_m", "depth", "length", "d"], dims, default_d)
    raw_h = _raw_dimension_value(product, ["heightM", "height_m", "item_height_m", "height", "h"], dims, default_h)

    width_m = max(0.03, _product_dim_to_m(raw_w, default_w))
    depth_m = max(0.02, _product_dim_to_m(raw_d, default_d))
    height_m = max(0.01, _product_dim_to_m(raw_h, default_h))

    # Keep wall objects thin even if raw depth is noisy.
    if category in {"mirror", "wall_art"}:
        depth_m = min(depth_m, 0.08)
    if category == "ceiling_lamp":
        height_m = min(max(height_m, 0.12), 0.60)
        width_m = min(max(width_m, 0.20), 1.20)
        depth_m = min(max(depth_m, 0.20), 1.20)

    pid = str(_engine.first_value(product.get("id"), product.get("productId"), product.get("product_id"), default=f"ai-product-{index + 1}"))
    styles = product.get("styles") or product.get("style") or []
    if isinstance(styles, str):
        styles = [styles]
    score = _engine.as_float(
        _engine.first_value(product.get("final_score"), product.get("ranking_score"), product.get("score"), product.get("style_score"), default=0.5),
        0.5,
    )

    return _engine.Product(
        productId=pid,
        name=str(_engine.first_value(product.get("name"), product.get("title"), default=category)),
        category=category,
        widthM=width_m,
        depthM=depth_m,
        heightM=height_m,
        modelUrl=str(_engine.first_value(product.get("modelUrl"), product.get("model_url"), model_url_by_id.get(pid), default="") or ""),
        imageUrl=str(_engine.first_value(product.get("imageUrl"), product.get("image_url"), product.get("thumbnail"), default="") or ""),
        styles=list(styles),
        price=product.get("price"),
        sourceScore=score,
        raw=product,
    )


def normalize_payload(payload: Dict[str, Any]):
    room = _engine.normalize_room(payload)
    model_url_by_id = payload.get("modelUrlById") or {}
    products = [normalize_product(p, i, model_url_by_id) for i, p in enumerate(_engine.extract_products(payload))]
    return room, products


# ---------------------------------------------------------------------------
# 3) Selection quality: mattress handling and fallback room essentials
# ---------------------------------------------------------------------------

def _make_virtual_product(room_type: str, category: str) -> Any:
    dims = _engine.DEFAULT_DIMS.get(category, (0.8, 0.6, 0.7))
    names = {
        "coffee_table": "Virtual Coffee Table",
        "tv_stand": "Virtual TV Stand",
        "nightstand": "Virtual Nightstand",
    }
    return _engine.Product(
        productId=f"virtual-{room_type}-{category}",
        name=names.get(category, f"Virtual {category}"),
        category=category,
        widthM=dims[0],
        depthM=dims[1],
        heightM=dims[2],
        modelUrl="",
        imageUrl="",
        styles=["virtual"],
        sourceScore=0.15,
        raw={"virtual": True, "reason": "fallback_required_room_essential"},
    )


def select_products(room: Any, products: List[Any], payload: Dict[str, Any]):
    # Do not allow mattress as a separate visible item if an actual bed exists.
    has_real_bed = any(p.category == "bed" for p in products)
    filtered = []
    extra_rejected: List[Dict[str, Any]] = []
    for product in products:
        if product.category == "mattress" and has_real_bed:
            extra_rejected.append({
                "productId": product.productId,
                "name": product.name,
                "category": product.category,
                "reason": "mattress_merged_into_bed",
            })
            continue
        filtered.append(product)

    selected, rejected, missing = _BASE_SELECT_PRODUCTS(room, filtered, payload)
    rejected = extra_rejected + rejected

    # Living room templates need these anchors. Add virtual low-score placeholders
    # only when recommendation did not provide a real product; this keeps layout coherent.
    if room.type == "living_room":
        present = {p.category for p in selected}
        for required in ["coffee_table", "tv_stand"]:
            if required not in present:
                selected.append(_make_virtual_product(room.type, required))
                present.add(required)
        missing = [m for m in missing if m not in {"coffee_table", "tv_stand"}]

    return selected, rejected, missing


# ---------------------------------------------------------------------------
# 4) Candidate post-processing: ceiling lamps and large/dense bedrooms
# ---------------------------------------------------------------------------

def _items_by_category(items: Iterable[Any]) -> Dict[str, List[Any]]:
    grouped: Dict[str, List[Any]] = {}
    for item in items:
        grouped.setdefault(item.category, []).append(item)
    return grouped


def _place_ceiling_lamps(room: Any, items: List[Any]) -> None:
    grouped = _items_by_category(items)
    lamps = grouped.get("ceiling_lamp", [])
    if not lamps:
        return
    bed = grouped.get("bed", [None])[0]
    table = grouped.get("dining_table", [None])[0] or grouped.get("coffee_table", [None])[0]
    anchor_x = bed.x if bed else table.x if table else 0.0
    anchor_z = bed.z if bed else table.z if table else 0.0
    offsets = [0.0, -0.9, 0.9]
    for idx, lamp in enumerate(lamps):
        lamp.layer = "ceiling"
        lamp.anchorWall = "CEILING"
        lamp.supportSurfaceId = None
        lamp.x = max(-room.widthM / 2.0 + 0.35, min(room.widthM / 2.0 - 0.35, anchor_x))
        lamp.z = max(-room.lengthM / 2.0 + 0.35, min(room.lengthM / 2.0 - 0.35, anchor_z + offsets[min(idx, len(offsets) - 1)]))
        lamp.y = max(2.05, min(room.heightM - lamp.heightM / 2.0 - 0.08, room.heightM - 0.25))
        lamp.rotationY = 0.0
        lamp.facingDirection = "SOUTH"
        lamp.facingTarget = ""
        lamp.layoutReasoning = "Đèn trần được treo trên trần tại vùng trung tâm/giường, không đặt trên tủ đầu giường."
        lamp.relations.append({"type": "ceiling_centered_over", "target": "bed" if bed else "room_center"})


def _place_bedroom_secondary_zone(room: Any, items: List[Any]) -> None:
    if room.type != "bedroom" or room.widthM * room.lengthM < 36:
        return
    grouped = _items_by_category(items)
    bed = grouped.get("bed", [None])[0]
    chair = (grouped.get("armchair") or grouped.get("chair") or [None])[0]
    storage = (grouped.get("wardrobe") or grouped.get("cabinet") or grouped.get("bookshelf") or [None])[0]
    mirror = grouped.get("mirror", [None])[0]

    bed_wall = bed.anchorWall if bed else "RIGHT_WALL"
    opposite = _engine.opposite_wall(bed_wall)
    side_a, side_b = _engine.side_walls_for_wall(bed_wall)

    if storage:
        wall = opposite if storage.category in {"wardrobe", "cabinet", "bookshelf"} else side_a
        _engine.place_against_wall(storage, room, wall, along=-room.lengthM * 0.22 if wall in {"LEFT_WALL", "RIGHT_WALL"} else -room.widthM * 0.22)
        storage.layoutReasoning = "Tủ/kệ được đưa về vùng lưu trữ để phòng lớn không bị trống."
        storage.facingTarget = "room_center"
        storage.relations.append({"type": "storage_zone", "target": wall})

    if chair:
        # Create a reading corner away from the bed but still inside the room.
        chair.x = -room.widthM / 2.0 + max(0.75, chair.widthM / 2.0 + 0.35)
        chair.z = room.lengthM / 2.0 - max(0.85, chair.depthM / 2.0 + 0.55)
        chair.rotationY = 135.0
        chair.facingDirection = "SOUTH"
        chair.anchorWall = "READING_CORNER"
        chair.facingTarget = "room_center"
        chair.layoutReasoning = "Ghế được chuyển thành góc đọc sách để lấp không gian phòng ngủ dài."
        chair.relations.append({"type": "reading_corner", "target": "bedroom_secondary_zone"})
        _engine.clamp_inside_room(chair, room)

    if mirror:
        wall = side_a if side_a != bed_wall else side_b
        _engine.place_wall_art(mirror, room, wall, along=0.0)
        mirror.layoutReasoning = "Gương đặt ở tường phụ, tạo vùng thay đồ riêng thay vì dồn sát giường."
        mirror.relations.append({"type": "dressing_zone", "target": wall})


def _postprocess_candidate(candidate: Dict[str, Any], room: Any) -> Dict[str, Any]:
    items = candidate.get("items", [])
    _place_ceiling_lamps(room, items)
    _place_bedroom_secondary_zone(room, items)
    return candidate


def generate_candidates(room: Any, products: List[Any]) -> List[Dict[str, Any]]:
    candidates = _BASE_GENERATE_CANDIDATES(room, products)
    return [_postprocess_candidate(candidate, room) for candidate in candidates]


# ---------------------------------------------------------------------------
# 5) Scoring caps for fake-high scores
# ---------------------------------------------------------------------------

def _density() -> str:
    return _text(_CURRENT_PAYLOAD.get("furnitureDensity") or _CURRENT_PAYLOAD.get("furniture_density") or "medium")


def _cap_score(result: Dict[str, Any], cap: float, reason: str) -> None:
    old = float(result.get("total", 0.0))
    if old > cap:
        result["total"] = round(cap, 4)
        result.setdefault("breakdown", {})[f"scoreCap_{reason}"] = round(cap / 100.0, 4)


def score_layout(candidate: Dict[str, Any], room: Any, products: List[Any]) -> Dict[str, Any]:
    result = _BASE_SCORE_LAYOUT(candidate, room, products)
    items = candidate.get("items", [])
    cats = {item.category for item in items}
    floor_items = [item for item in items if item.layer == "floor" and item.category != "rug"]

    if room.type == "living_room":
        missing = [cat for cat in ["sofa", "coffee_table", "tv_stand"] if cat not in cats]
        completeness = 1.0 - len(missing) / 3.0
        result.setdefault("breakdown", {})["essentialCompleteness"] = round(completeness, 4)
        if len(missing) >= 2:
            _cap_score(result, 48.0, "missingLivingEssentials")
        elif len(missing) == 1:
            _cap_score(result, 68.0, "missingLivingEssential")

    if room.type == "bedroom":
        area = room.widthM * room.lengthM
        is_dense = "dense" in _density() or "dày" in _density() or "day" in _density() or "nhiều" in _density()
        storage_present = bool(cats.intersection({"wardrobe", "cabinet", "bookshelf"}))
        if area >= 40 and is_dense and not storage_present:
            _cap_score(result, 82.0, "largeDenseBedroomNoStorage")
        if area >= 40 and len(floor_items) <= 4:
            _cap_score(result, 78.0, "largeBedroomTooSparse")

    # Penalize excessive repair even when final collision is zero; many repairs mean the first layout was unstable.
    # The original repair counters are added after scoring, so this part uses geometric spread instead.
    if floor_items:
        xs = [item.x for item in floor_items]
        zs = [item.z for item in floor_items]
        spread_x = (max(xs) - min(xs)) / max(room.widthM, 1e-6)
        spread_z = (max(zs) - min(zs)) / max(room.lengthM, 1e-6)
        spread = max(spread_x, spread_z)
        result.setdefault("breakdown", {})["layoutSpread"] = round(spread, 4)
        if room.widthM * room.lengthM >= 40 and spread < 0.28:
            _cap_score(result, 80.0, "largeRoomPoorSpread")

    return result


# ---------------------------------------------------------------------------
# 6) Public wrapper and installer
# ---------------------------------------------------------------------------

def finalize_layout(payload: Dict[str, Any]) -> Dict[str, Any]:
    global _CURRENT_PAYLOAD
    previous = _CURRENT_PAYLOAD
    _CURRENT_PAYLOAD = payload if isinstance(payload, dict) else {}
    try:
        result = _BASE_FINALIZE_LAYOUT(payload)
        metrics = result.setdefault("metrics", {})
        metrics["qualityPatchInstalled"] = True
        metrics["dimensionNormalization"] = "product_cm_mm_to_m_v2"
        metrics["categoryAliasPatch"] = "vi_furniture_aliases_v2"
        metrics["scoreQualityCaps"] = True
        return result
    finally:
        _CURRENT_PAYLOAD = previous


def model_patch_status() -> Dict[str, Any]:
    status = _BASE_MODEL_PATCH_STATUS()
    status.update({
        "qualityPatchInstalled": _QUALITY_PATCH_INSTALLED,
        "dimensionNormalization": "product_cm_mm_to_m_v2",
        "categoryAliasPatch": "vi_furniture_aliases_v2",
        "scoreQualityCaps": True,
    })
    return status


def install_quality_patches() -> None:
    global _QUALITY_PATCH_INSTALLED
    if _QUALITY_PATCH_INSTALLED:
        return
    _install_aliases_and_compositions()
    _engine.normalize_product = normalize_product
    _engine.normalize_payload = normalize_payload
    _engine.select_products = select_products
    _engine.generate_candidates = generate_candidates
    _engine.score_layout = score_layout
    _engine.finalize_layout = finalize_layout
    _QUALITY_PATCH_INSTALLED = True


install_quality_patches()
