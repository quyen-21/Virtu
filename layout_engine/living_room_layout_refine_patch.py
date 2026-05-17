from __future__ import annotations

"""
Living-room role-specific layout refinement patch.

Installed after living_room_semantic_patch.

Goal:
- keep semantic mapping from e-commerce categories to layout roles;
- then place each role in a more natural interior-design position:
  - tv_stand / console / low media cabinet: against focal wall;
  - sofa: opposite focal wall;
  - coffee_table: between sofa and focal wall, not replaced by tall console;
  - armchairs: diagonal/side seating around coffee table;
  - side tables: near sofa ends;
  - bookshelf / display shelf: side wall, not inside seating cluster;
  - small cabinet / drawer: wall/corner zone, not floating alone.
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from layout_engine import engine as _engine
from layout_engine import living_room_semantic_patch as _semantic_patch

_BASE_SELECT_PRODUCTS = _engine.select_products
_BASE_GENERATE_CANDIDATES = _engine.generate_candidates
_BASE_FINALIZE_LAYOUT = _engine.finalize_layout
_BASE_MODEL_PATCH_STATUS = _semantic_patch.model_patch_status

_REFINEMENT_PATCH_INSTALLED = False


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _text(value: Any) -> str:
    return _engine.canonical_text(value or "")


def _item_text(item: Any) -> str:
    raw = item.raw if isinstance(getattr(item, "raw", None), dict) else {}
    parts = [
        item.category,
        item.name,
        raw.get("category"),
        raw.get("name"),
        raw.get("title"),
        raw.get("layoutOriginalCategory"),
        raw.get("layoutSemanticRole"),
    ]
    text = " ".join(_text(p) for p in parts if p is not None)
    return f"{text} {text.replace('_', ' ')}"


def _contains(text: str, keywords: Iterable[str]) -> bool:
    return any(k in text for k in keywords)


def _long_side(item: Any) -> float:
    return max(float(item.widthM), float(item.depthM))


def _short_side(item: Any) -> float:
    return min(float(item.widthM), float(item.depthM))


def _area(item: Any) -> float:
    return float(item.widthM) * float(item.depthM)


def _is_console_like(item: Any) -> bool:
    text = _item_text(item)
    return _contains(text, ["console", "bàn console", "ban console", "bàn_console", "ban_console"])


def _is_side_table_like(item: Any) -> bool:
    text = _item_text(item)
    return _contains(text, ["bàn bên", "ban ben", "bàn_bên", "ban_ben", "bàn phụ", "ban phu", "side table", "side_table"])


def _is_shelf_like(item: Any) -> bool:
    text = _item_text(item)
    return _contains(text, ["kệ", "ke ", "bookshelf", "bookcase", "tủ trưng bày", "tu trung bay", "display"])


def _is_good_coffee_table(item: Any) -> bool:
    # A coffee table should be low and not too long. Console tables are usually
    # too high/long and should become tv_stand/focal furniture instead.
    if _is_console_like(item):
        return False
    if item.heightM > 0.58:
        return False
    if _long_side(item) > 1.30:
        return False
    if not (0.16 <= _area(item) <= 1.25):
        return False
    return item.category in {"coffee_table", "side_table", "table"} or _is_side_table_like(item)


def _is_good_tv_stand(item: Any) -> bool:
    # Console / low shelf / low cabinet can work as a TV stand. Tall shelves stay
    # as bookshelf and should not become focal media furniture.
    if item.heightM > 1.10:
        return False
    if _long_side(item) < 0.80:
        return False
    if _short_side(item) < 0.20:
        return False
    text = _item_text(item)
    return (
        item.category in {"tv_stand", "cabinet", "bookshelf", "side_table"}
        or _is_console_like(item)
        or _contains(text, ["media", "tivi", "tv", "television", "kệ phòng khách", "ke phong khach"])
    )


def _mark_role(item: Any, category: str, reason: str) -> None:
    old_category = item.category
    item.category = category
    raw = item.raw if isinstance(getattr(item, "raw", None), dict) else {}
    raw["layoutSemanticRole"] = category
    raw.setdefault("layoutOriginalCategory", old_category)
    raw["layoutRefineReason"] = reason
    item.raw = raw


# ---------------------------------------------------------------------------
# Product role cleanup before template generation
# ---------------------------------------------------------------------------

def _find_best(items: Iterable[Any], predicate, used: set[str]) -> Optional[Any]:
    candidates: List[Tuple[float, Any]] = []
    for item in items:
        if item.productId in used:
            continue
        if not predicate(item):
            continue
        score = float(getattr(item, "sourceScore", 0.0) or 0.0)
        # Prefer real products with image/model over virtual fallbacks.
        if getattr(item, "modelUrl", "") or getattr(item, "imageUrl", ""):
            score += 0.2
        # Prefer visually usable proportions.
        score += min(0.4, _area(item))
        candidates.append((score, item))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _cleanup_living_room_roles(selected: List[Any], all_products: List[Any]) -> List[Any]:
    used_ids = {item.productId for item in selected}
    selected_by_id = {item.productId: item for item in selected}

    coffee_items = [item for item in selected if item.category == "coffee_table"]
    tv_items = [item for item in selected if item.category == "tv_stand"]

    # If a tall/long console was accidentally mapped to coffee_table, demote it
    # and choose a better side table as coffee_table.
    bad_coffee = coffee_items[0] if coffee_items and not _is_good_coffee_table(coffee_items[0]) else None
    if bad_coffee is not None:
        alt_pool = list(selected) + [p for p in all_products if p.productId not in selected_by_id]
        alt = _find_best(alt_pool, _is_good_coffee_table, used={bad_coffee.productId})
        if alt is not None:
            _mark_role(alt, "coffee_table", "Selected as central coffee table because it is low and compact.")
            if alt.productId not in selected_by_id:
                selected.append(alt)
                selected_by_id[alt.productId] = alt
            if _is_good_tv_stand(bad_coffee):
                _mark_role(bad_coffee, "tv_stand", "Tall/long console demoted from coffee_table and reused as TV/media stand.")
            else:
                _mark_role(bad_coffee, "side_table", "Tall/long product is not suitable as coffee table, demoted to side table.")

    # Ensure there is a real focal tv_stand if possible.
    if not any(item.category == "tv_stand" for item in selected):
        alt_pool = list(selected) + [p for p in all_products if p.productId not in selected_by_id]
        tv = _find_best(alt_pool, _is_good_tv_stand, used=set())
        if tv is not None:
            _mark_role(tv, "tv_stand", "Selected as focal TV/media stand from console/low cabinet/low shelf.")
            if tv.productId not in selected_by_id:
                selected.append(tv)
                selected_by_id[tv.productId] = tv

    # Ensure there is a usable central coffee table if possible.
    if not any(item.category == "coffee_table" for item in selected):
        alt_pool = list(selected) + [p for p in all_products if p.productId not in selected_by_id]
        coffee = _find_best(alt_pool, _is_good_coffee_table, used={item.productId for item in selected if item.category == "tv_stand"})
        if coffee is not None:
            _mark_role(coffee, "coffee_table", "Selected as central coffee table from low side table/table product.")
            if coffee.productId not in selected_by_id:
                selected.append(coffee)
                selected_by_id[coffee.productId] = coffee

    # Keep the scene from getting too crowded if semantic additions exceeded cap.
    # Preserve essentials first.
    priority = {"sofa": 0, "coffee_table": 1, "tv_stand": 2, "rug": 3, "armchair": 4, "chair": 5, "side_table": 6, "bookshelf": 7, "cabinet": 8, "mirror": 9}
    selected.sort(key=lambda item: (priority.get(item.category, 99), -float(getattr(item, "sourceScore", 0.0) or 0.0)))
    return selected[:10]


def select_products(room: Any, products: List[Any], payload: Dict[str, Any]):
    selected, rejected, missing = _BASE_SELECT_PRODUCTS(room, products, payload)
    if room.type != "living_room":
        return selected, rejected, missing

    selected = _cleanup_living_room_roles(selected, products)
    selected_ids = {item.productId for item in selected}
    selected_categories = {item.category for item in selected}
    missing = [m for m in missing if m not in selected_categories]

    # Remove rejected entries for products that were reused through refinement.
    rejected = [r for r in rejected if str(r.get("productId")) not in selected_ids]
    return selected, rejected, missing


# ---------------------------------------------------------------------------
# Geometry helpers for role-specific placement
# ---------------------------------------------------------------------------

def _wall_vector(wall: str) -> Tuple[float, float]:
    # Direction from room center/seating area toward the focal wall.
    wall = str(wall or "BACK_WALL").upper()
    if wall == "FRONT_WALL":
        return 0.0, 1.0
    if wall == "LEFT_WALL":
        return -1.0, 0.0
    if wall == "RIGHT_WALL":
        return 1.0, 0.0
    return 0.0, -1.0


def _rotation_towards(source: Any, target_x: float, target_z: float) -> float:
    dx = target_x - source.x
    dz = target_z - source.z
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return float(getattr(source, "rotationY", 0.0) or 0.0)
    return math.degrees(math.atan2(dx, dz)) % 360.0


def _facing_direction(rotation: float) -> str:
    return _engine.rotation_to_facing(rotation)


def _set_floor_item(item: Any, x: float, z: float, rotation: float, room: Any) -> None:
    item.x = x
    item.z = z
    item.rotationY = rotation % 360.0
    item.facingDirection = _facing_direction(item.rotationY)
    item.layer = "floor"
    item.y = 0.01 if item.category == "rug" else item.heightM / 2.0
    _engine.clamp_inside_room(item, room)


def _by_category(items: Iterable[Any]) -> Dict[str, List[Any]]:
    grouped: Dict[str, List[Any]] = {}
    for item in items:
        grouped.setdefault(item.category, []).append(item)
    return grouped


def _choose_focal_wall(candidate: Dict[str, Any], tv: Optional[Any]) -> str:
    wall = str(candidate.get("focalWall") or "").upper()
    if wall in {"BACK_WALL", "FRONT_WALL", "LEFT_WALL", "RIGHT_WALL"}:
        return wall
    if tv and str(getattr(tv, "anchorWall", "")).upper() in {"BACK_WALL", "FRONT_WALL", "LEFT_WALL", "RIGHT_WALL"}:
        return str(tv.anchorWall).upper()
    return "BACK_WALL"


def _opposite_rotation_to_wall(wall: str) -> float:
    # Rotation for a sofa/chair that faces the focal wall.
    wall = str(wall or "BACK_WALL").upper()
    if wall == "FRONT_WALL":
        return 0.0
    if wall == "LEFT_WALL":
        return 270.0
    if wall == "RIGHT_WALL":
        return 90.0
    return 180.0


def _place_bookshelf_side_wall(item: Any, room: Any, focal_wall: str, index: int) -> None:
    side_walls = _engine.side_walls_for_wall(focal_wall)
    wall = side_walls[index % len(side_walls)] if side_walls else "LEFT_WALL"
    along = -room.lengthM * 0.22 if wall in {"LEFT_WALL", "RIGHT_WALL"} else -room.widthM * 0.22
    _engine.place_against_wall(item, room, wall, along=along)
    item.facingTarget = "room_center"
    item.layoutReasoning = "Kệ lớn được đưa sát tường phụ để không chen vào cụm sofa."
    item.relations.append({"type": "side_wall_storage_zone", "target": wall})


def _place_small_cabinet(item: Any, room: Any, focal_wall: str, index: int) -> None:
    side_walls = _engine.side_walls_for_wall(focal_wall)
    wall = side_walls[(index + 1) % len(side_walls)] if side_walls else "RIGHT_WALL"
    along = room.lengthM * 0.30 if wall in {"LEFT_WALL", "RIGHT_WALL"} else room.widthM * 0.30
    _engine.place_against_wall(item, room, wall, along=along)
    item.facingTarget = "room_center"
    item.layoutReasoning = "Tủ nhỏ/hộc kéo được đưa vào vùng tường/corner thay vì đứng lẻ loi giữa phòng."
    item.relations.append({"type": "corner_storage_zone", "target": wall})


# ---------------------------------------------------------------------------
# Living-room candidate refinement
# ---------------------------------------------------------------------------

def _refine_living_room_candidate(candidate: Dict[str, Any], room: Any) -> Dict[str, Any]:
    if room.type != "living_room":
        return candidate

    items = candidate.get("items", [])
    if not items:
        return candidate

    grouped = _by_category(items)
    sofa = (grouped.get("sofa") or [None])[0]
    coffee = (grouped.get("coffee_table") or [None])[0]
    tv = (grouped.get("tv_stand") or [None])[0]
    rug = (grouped.get("rug") or [None])[0]
    armchairs = grouped.get("armchair", []) + grouped.get("chair", [])
    side_tables = [item for item in grouped.get("side_table", []) if item is not coffee and item is not tv]
    bookshelves = grouped.get("bookshelf", [])
    cabinets = [item for item in grouped.get("cabinet", []) if item is not tv and item is not coffee]

    if sofa is None:
        return candidate

    focal_wall = _choose_focal_wall(candidate, tv)
    fx, fz = _wall_vector(focal_wall)
    px, pz = -fz, fx  # perpendicular to focal axis

    # 1) Focal tv/media stand stays against the focal wall.
    if tv is not None:
        _engine.place_against_wall(tv, room, focal_wall, along=0.0)
        tv.facingTarget = "sofa"
        tv.layoutReasoning = "TV/media stand hoặc console được đặt sát focal wall để tạo điểm nhìn chính."
        tv.relations.append({"type": "focal_media_wall", "target": focal_wall})

    # 2) Sofa is centered opposite focal wall.
    target_x = tv.x if tv is not None else fx * min(room.widthM, room.lengthM) * 0.30
    target_z = tv.z if tv is not None else fz * min(room.widthM, room.lengthM) * 0.30
    viewing_distance = min(max(2.45, sofa.depthM + 1.65), max(room.widthM, room.lengthM) * 0.42)
    sofa_x = target_x - fx * viewing_distance
    sofa_z = target_z - fz * viewing_distance
    _set_floor_item(sofa, sofa_x, sofa_z, _opposite_rotation_to_wall(focal_wall), room)
    sofa.facingTarget = "tv_stand" if tv is not None else "room_center"
    sofa.layoutReasoning = "Sofa được căn đối diện focal wall, tạo trục nhìn chính của phòng khách."

    # 3) Coffee table must sit between sofa and focal point.
    if coffee is not None:
        gap = 0.42
        forward = sofa.depthM / 2.0 + coffee.depthM / 2.0 + gap
        coffee_x = sofa.x + fx * forward
        coffee_z = sofa.z + fz * forward
        _set_floor_item(coffee, coffee_x, coffee_z, sofa.rotationY, room)
        coffee.facingTarget = ""
        coffee.layoutReasoning = "Coffee table được đặt giữa sofa và focal wall, không dùng console cao như bàn trà."
        coffee.relations.append({"type": "center_between", "target": "sofa_tv_axis"})

    center_x = coffee.x if coffee is not None else sofa.x + fx * 1.15
    center_z = coffee.z if coffee is not None else sofa.z + fz * 1.15

    # 4) Rug anchors the seating group.
    if rug is not None:
        rug_x = (sofa.x + center_x) / 2.0
        rug_z = (sofa.z + center_z) / 2.0
        _set_floor_item(rug, rug_x, rug_z, sofa.rotationY, room)
        rug.facingTarget = ""
        rug.layoutReasoning = "Thảm được căn dưới cụm sofa/coffee table để gom nhóm sinh hoạt."

    # 5) Armchairs sit diagonally/side around coffee table, facing the table.
    for idx, chair in enumerate(armchairs[:3]):
        sign = -1.0 if idx % 2 == 0 else 1.0
        side_gap = 0.62 + min(0.45, chair.widthM / 2.0)
        back_offset = -0.10 if idx < 2 else 0.65
        chair_x = center_x + px * sign * (0.95 + side_gap) - fx * back_offset
        chair_z = center_z + pz * sign * (0.95 + side_gap) - fz * back_offset
        _set_floor_item(chair, chair_x, chair_z, 0.0, room)
        chair.rotationY = _rotation_towards(chair, center_x, center_z)
        chair.facingDirection = _facing_direction(chair.rotationY)
        chair.facingTarget = "coffee_table" if coffee is not None else "sofa"
        chair.layoutReasoning = "Ghế phụ được đặt chéo quanh coffee table để tạo cụm trò chuyện tự nhiên."
        chair.relations.append({"type": "conversational_seating", "target": "coffee_table" if coffee else "sofa"})

    # 6) Side tables go next to sofa ends, not in the middle of seating axis.
    for idx, table in enumerate(side_tables[:2]):
        sign = -1.0 if idx == 0 else 1.0
        table_x = sofa.x + px * sign * (sofa.widthM / 2.0 + table.widthM / 2.0 + 0.22)
        table_z = sofa.z + pz * sign * (sofa.widthM / 2.0 + table.widthM / 2.0 + 0.22)
        _set_floor_item(table, table_x, table_z, sofa.rotationY, room)
        table.facingTarget = ""
        table.layoutReasoning = "Bàn bên được đặt cạnh đầu sofa, không chen vào vị trí coffee table."
        table.relations.append({"type": "beside", "target": "sofa"})

    # 7) Bookshelves and tall display furniture go to side walls.
    for idx, shelf in enumerate(bookshelves[:2]):
        _place_bookshelf_side_wall(shelf, room, focal_wall, idx)

    # 8) Small cabinets/drawers go to wall/corner zone.
    for idx, cabinet in enumerate(cabinets[:2]):
        _place_small_cabinet(cabinet, room, focal_wall, idx)

    candidate["template"] = str(candidate.get("template", "living_room")) + "_role_refined"
    candidate["layoutRefinement"] = "living_room_role_specific_v1"
    return candidate


def generate_candidates(room: Any, products: List[Any]) -> List[Dict[str, Any]]:
    candidates = _BASE_GENERATE_CANDIDATES(room, products)
    if room.type != "living_room":
        return candidates
    return [_refine_living_room_candidate(candidate, room) for candidate in candidates]


# ---------------------------------------------------------------------------
# Public wrapper / health status
# ---------------------------------------------------------------------------

def finalize_layout(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = _BASE_FINALIZE_LAYOUT(payload)
    metrics = result.setdefault("metrics", {})
    metrics["livingRoomLayoutRefinePatchInstalled"] = True
    metrics["livingRoomRolePlacement"] = "role_specific_focal_wall_seating_group_v1"
    return result


def model_patch_status() -> Dict[str, Any]:
    status = _BASE_MODEL_PATCH_STATUS()
    status.update({
        "livingRoomLayoutRefinePatchInstalled": _REFINEMENT_PATCH_INSTALLED,
        "livingRoomRolePlacement": "role_specific_focal_wall_seating_group_v1",
    })
    return status


def install_living_room_layout_refine_patch() -> None:
    global _REFINEMENT_PATCH_INSTALLED
    if _REFINEMENT_PATCH_INSTALLED:
        return
    _engine.select_products = select_products
    _engine.generate_candidates = generate_candidates
    _engine.finalize_layout = finalize_layout
    _REFINEMENT_PATCH_INSTALLED = True


install_living_room_layout_refine_patch()
