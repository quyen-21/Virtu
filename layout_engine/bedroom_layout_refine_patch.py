from __future__ import annotations

"""
Bedroom role-specific layout refinement patch.

Installed after living_room_layout_refine_patch.

Goal:
- keep previous layout patches for layout-only service, quality fixes, semantic
  mapping, and living-room refinement;
- add bedroom-specific placement rules:
  - bed anchors to the main wall and is centered;
  - rug sits under the lower 2/3 of the bed;
  - nightstands are symmetrical beside the bed;
  - bench is placed at the foot of the bed, not floating in a corner;
  - mirror is placed on a side wall as a dressing zone;
  - ceiling lamps are centered over the bed or room center, away from mirror;
  - cabinet / wardrobe / bookshelf move to wall/corner storage zones.
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from layout_engine import engine as _engine
from layout_engine import living_room_layout_refine_patch as _living_patch

_BASE_SELECT_PRODUCTS = _engine.select_products
_BASE_GENERATE_CANDIDATES = _engine.generate_candidates
_BASE_FINALIZE_LAYOUT = _engine.finalize_layout
_BASE_MODEL_PATCH_STATUS = _living_patch.model_patch_status

_BEDROOM_PATCH_INSTALLED = False


# ---------------------------------------------------------------------------
# Category and text helpers
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


def _is_bench_like(item: Any) -> bool:
    text = _item_text(item)
    return _contains(text, ["bench", "ghế dài", "ghe dai", "đôn", "don", "stool"])


def _is_ceiling_lamp(item: Any) -> bool:
    return item.category == "ceiling_lamp" or _contains(_item_text(item), ["đèn trần", "den tran", "ceiling", "pendant", "chandelier"])


def _long_side(item: Any) -> float:
    return max(float(item.widthM), float(item.depthM))


def _short_side(item: Any) -> float:
    return min(float(item.widthM), float(item.depthM))


def _mark_role(item: Any, category: str, reason: str) -> None:
    old_category = item.category
    item.category = category
    raw = item.raw if isinstance(getattr(item, "raw", None), dict) else {}
    raw.setdefault("layoutOriginalCategory", old_category)
    raw["layoutSemanticRole"] = category
    raw["bedroomRefineReason"] = reason
    item.raw = raw


# ---------------------------------------------------------------------------
# Selection cleanup for bedroom
# ---------------------------------------------------------------------------

def _cleanup_bedroom_roles(selected: List[Any], all_products: List[Any]) -> List[Any]:
    # Convert bench-like chairs/stools into a bedroom bench role. The base engine
    # can still treat it geometrically as floor furniture, but our placement
    # refinement can put it at the foot of the bed.
    for item in selected:
        if item.category in {"chair", "armchair", "stool", "side_table"} and _is_bench_like(item):
            _mark_role(item, "bench", "Bench-like product is used as foot-of-bed bench in bedroom.")

    # Do not keep too many loose chairs in bedroom unless no bench exists.
    has_bench = any(item.category == "bench" for item in selected)
    if has_bench:
        loose_chairs = [item for item in selected if item.category in {"chair", "armchair"}]
        keep_loose = set(item.productId for item in loose_chairs[:1])
        selected = [item for item in selected if item.category not in {"chair", "armchair"} or item.productId in keep_loose]

    priority = {
        "bed": 0,
        "nightstand": 1,
        "rug": 2,
        "bench": 3,
        "wardrobe": 4,
        "cabinet": 5,
        "bookshelf": 6,
        "mirror": 7,
        "ceiling_lamp": 8,
        "desk": 9,
        "chair": 10,
        "armchair": 11,
        "lamp": 12,
        "side_table": 13,
        "plant": 14,
        "wall_art": 15,
    }
    selected.sort(key=lambda item: (priority.get(item.category, 99), -float(getattr(item, "sourceScore", 0.0) or 0.0)))
    return selected[:10]


def select_products(room: Any, products: List[Any], payload: Dict[str, Any]):
    selected, rejected, missing = _BASE_SELECT_PRODUCTS(room, products, payload)
    if room.type != "bedroom":
        return selected, rejected, missing

    selected = _cleanup_bedroom_roles(selected, products)
    selected_ids = {item.productId for item in selected}
    selected_categories = {item.category for item in selected}
    missing = [m for m in missing if m not in selected_categories]
    rejected = [r for r in rejected if str(r.get("productId")) not in selected_ids]
    return selected, rejected, missing


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _by_category(items: Iterable[Any]) -> Dict[str, List[Any]]:
    grouped: Dict[str, List[Any]] = {}
    for item in items:
        grouped.setdefault(item.category, []).append(item)
    return grouped


def _wall_vector(wall: str) -> Tuple[float, float]:
    wall = str(wall or "BACK_WALL").upper()
    if wall == "FRONT_WALL":
        return 0.0, 1.0
    if wall == "LEFT_WALL":
        return -1.0, 0.0
    if wall == "RIGHT_WALL":
        return 1.0, 0.0
    return 0.0, -1.0


def _perp_vector(fx: float, fz: float) -> Tuple[float, float]:
    return -fz, fx


def _bed_rotation_for_head_wall(wall: str) -> float:
    wall = str(wall or "BACK_WALL").upper()
    # Bed headboard is against wall, so bed faces away from that wall.
    if wall == "FRONT_WALL":
        return 180.0
    if wall == "LEFT_WALL":
        return 90.0
    if wall == "RIGHT_WALL":
        return 270.0
    return 0.0


def _place_floor(item: Any, x: float, z: float, rotation: float, room: Any) -> None:
    item.x = x
    item.z = z
    item.rotationY = rotation % 360.0
    item.facingDirection = _engine.rotation_to_facing(item.rotationY)
    item.layer = "floor"
    item.y = 0.01 if item.category == "rug" else item.heightM / 2.0
    _engine.clamp_inside_room(item, room)


def _rotation_towards(item: Any, target_x: float, target_z: float) -> float:
    dx = target_x - item.x
    dz = target_z - item.z
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return float(getattr(item, "rotationY", 0.0) or 0.0)
    return math.degrees(math.atan2(dx, dz)) % 360.0


def _choose_bed_wall(candidate: Dict[str, Any], room: Any) -> str:
    # Prefer a long wall for a large bedroom; otherwise keep the template focal wall.
    wall = str(candidate.get("focalWall") or "").upper()
    if wall in {"BACK_WALL", "FRONT_WALL", "LEFT_WALL", "RIGHT_WALL"}:
        return wall
    return "BACK_WALL" if room.lengthM >= room.widthM else "LEFT_WALL"


def _wall_center_position(room: Any, wall: str, item: Any, clearance: float = 0.08) -> Tuple[float, float]:
    wall = str(wall or "BACK_WALL").upper()
    if wall == "FRONT_WALL":
        return 0.0, room.lengthM / 2.0 - item.depthM / 2.0 - clearance
    if wall == "LEFT_WALL":
        return -room.widthM / 2.0 + item.depthM / 2.0 + clearance, 0.0
    if wall == "RIGHT_WALL":
        return room.widthM / 2.0 - item.depthM / 2.0 - clearance, 0.0
    return 0.0, -room.lengthM / 2.0 + item.depthM / 2.0 + clearance


def _place_storage_wall(item: Any, room: Any, bed_wall: str, index: int) -> None:
    side_walls = _engine.side_walls_for_wall(bed_wall)
    wall = side_walls[index % len(side_walls)] if side_walls else "LEFT_WALL"
    along = room.lengthM * 0.22 if wall in {"LEFT_WALL", "RIGHT_WALL"} else room.widthM * 0.22
    _engine.place_against_wall(item, room, wall, along=along)
    item.facingTarget = "room_center"
    item.layoutReasoning = "Tủ/kệ được đưa sát tường phụ để tạo storage zone cho phòng ngủ."
    item.relations.append({"type": "bedroom_storage_wall", "target": wall})


def _place_dressing_mirror(mirror: Any, room: Any, bed_wall: str) -> None:
    side_walls = _engine.side_walls_for_wall(bed_wall)
    wall = side_walls[0] if side_walls else "LEFT_WALL"
    _engine.place_wall_art(mirror, room, wall, along=-room.lengthM * 0.18 if wall in {"LEFT_WALL", "RIGHT_WALL"} else -room.widthM * 0.18)
    mirror.layoutReasoning = "Gương đặt ở tường phụ làm dressing zone, tránh nằm ngay sau đèn/đầu giường."
    mirror.relations.append({"type": "bedroom_dressing_zone", "target": wall})


def _place_ceiling_lamps(lamps: List[Any], room: Any, bed: Any, fx: float, fz: float, px: float, pz: float) -> None:
    # Put lamps above the lower/middle part of bed, not directly in front of mirror.
    if not lamps:
        return
    base_x = bed.x + fx * min(0.45, bed.depthM * 0.25)
    base_z = bed.z + fz * min(0.45, bed.depthM * 0.25)
    offsets = [0.0, 0.55, -0.55]
    for idx, lamp in enumerate(lamps[:3]):
        offset = offsets[idx] if idx < len(offsets) else 0.0
        lamp.category = "ceiling_lamp"
        lamp.layer = "ceiling"
        lamp.anchorWall = "CEILING"
        lamp.supportSurfaceId = None
        lamp.x = base_x + px * offset
        lamp.z = base_z + pz * offset
        lamp.y = max(2.15, min(room.heightM - lamp.heightM / 2.0 - 0.08, room.heightM - 0.25))
        lamp.rotationY = 0.0
        lamp.facingDirection = "SOUTH"
        lamp.facingTarget = "bed"
        _engine.clamp_inside_room(lamp, room)
        lamp.layoutReasoning = "Đèn trần được căn theo vùng giường, không đặt thấp/gần gương."
        lamp.relations.append({"type": "ceiling_over_bed", "target": "bed"})


# ---------------------------------------------------------------------------
# Bedroom candidate refinement
# ---------------------------------------------------------------------------

def _refine_bedroom_candidate(candidate: Dict[str, Any], room: Any) -> Dict[str, Any]:
    if room.type != "bedroom":
        return candidate

    items = candidate.get("items", [])
    if not items:
        return candidate

    # Normalize ceiling lamps in case previous alias/model candidate still gave lamp.
    for item in items:
        if _is_ceiling_lamp(item):
            item.category = "ceiling_lamp"

    grouped = _by_category(items)
    bed = (grouped.get("bed") or [None])[0]
    if bed is None:
        return candidate

    nightstands = grouped.get("nightstand", [])[:2]
    rug = (grouped.get("rug") or [None])[0]
    mirrors = grouped.get("mirror", [])
    lamps = grouped.get("ceiling_lamp", [])
    benches = grouped.get("bench", [])
    chairs = grouped.get("chair", []) + grouped.get("armchair", [])
    storage_items = grouped.get("wardrobe", []) + grouped.get("cabinet", []) + grouped.get("bookshelf", [])
    desks = grouped.get("desk", [])

    bed_wall = _choose_bed_wall(candidate, room)
    fx, fz = _wall_vector(bed_wall)        # from room center toward bed head wall
    px, pz = _perp_vector(fx, fz)
    bed_rotation = _bed_rotation_for_head_wall(bed_wall)

    # 1) Bed headboard against wall, centered on the main axis.
    bed_x, bed_z = _wall_center_position(room, bed_wall, bed, clearance=0.12)
    _place_floor(bed, bed_x, bed_z, bed_rotation, room)
    bed.anchorWall = bed_wall
    bed.facingTarget = "room_center"
    bed.layoutReasoning = "Giường được neo vào tường chính và căn giữa để làm trọng tâm phòng ngủ."
    bed.relations.append({"type": "headboard_against_wall", "target": bed_wall})

    # 2) Rug under the lower 2/3 of bed.
    if rug is not None:
        rug_x = bed.x - fx * min(0.35, bed.depthM * 0.18)
        rug_z = bed.z - fz * min(0.35, bed.depthM * 0.18)
        _place_floor(rug, rug_x, rug_z, bed.rotationY, room)
        rug.facingTarget = ""
        rug.layoutReasoning = "Thảm được đặt dưới 2/3 phần giường để gom vùng ngủ."
        rug.relations.append({"type": "under", "target": "bed"})

    # 3) Nightstands symmetric beside bed head area.
    for idx, ns in enumerate(nightstands):
        sign = -1.0 if idx == 0 else 1.0
        side_offset = bed.widthM / 2.0 + ns.widthM / 2.0 + 0.16
        head_offset = bed.depthM * 0.25
        ns_x = bed.x + px * sign * side_offset + fx * head_offset
        ns_z = bed.z + pz * sign * side_offset + fz * head_offset
        _place_floor(ns, ns_x, ns_z, bed.rotationY, room)
        ns.anchorWall = bed_wall
        ns.facingTarget = "bed"
        ns.layoutReasoning = "Tủ đầu giường được đặt đối xứng hai bên đầu giường."
        ns.relations.append({"type": "beside", "target": "bed"})

    # 4) Bench at foot of bed. If no explicit bench exists but there is one loose
    # chair/armchair, keep it as a reading chair instead of forcing it.
    if benches:
        bench = benches[0]
        foot_gap = 0.28
        bench_x = bed.x - fx * (bed.depthM / 2.0 + bench.depthM / 2.0 + foot_gap)
        bench_z = bed.z - fz * (bed.depthM / 2.0 + bench.depthM / 2.0 + foot_gap)
        _place_floor(bench, bench_x, bench_z, bed.rotationY, room)
        bench.facingTarget = "bed"
        bench.layoutReasoning = "Bench được đặt ở cuối giường thay vì lệch vào góc phòng."
        bench.relations.append({"type": "foot_of_bed", "target": "bed"})

    # 5) Mirror creates dressing zone on side wall.
    for mirror in mirrors[:1]:
        _place_dressing_mirror(mirror, room, bed_wall)

    # 6) Ceiling lamps align with bed/room center.
    _place_ceiling_lamps(lamps, room, bed, -fx, -fz, px, pz)

    # 7) Storage goes to side walls/corners.
    for idx, storage in enumerate(storage_items[:3]):
        _place_storage_wall(storage, room, bed_wall, idx)

    # 8) Desk/vanity or remaining chair becomes a secondary zone away from bed.
    secondary_wall = _engine.opposite_wall(bed_wall)
    for idx, desk in enumerate(desks[:1]):
        _engine.place_against_wall(desk, room, secondary_wall, along=0.0)
        desk.facingTarget = "room_center"
        desk.layoutReasoning = "Bàn làm việc/vanity được đặt ở tường đối diện để tạo zone phụ."
        desk.relations.append({"type": "bedroom_secondary_desk_zone", "target": secondary_wall})

    for idx, chair in enumerate(chairs[:1]):
        # Reading chair near side/corner, oriented toward bed/room center.
        side_walls = _engine.side_walls_for_wall(bed_wall)
        wall = side_walls[-1] if side_walls else "RIGHT_WALL"
        if wall in {"LEFT_WALL", "RIGHT_WALL"}:
            chair_x = -room.widthM / 2.0 + chair.widthM / 2.0 + 0.45 if wall == "LEFT_WALL" else room.widthM / 2.0 - chair.widthM / 2.0 - 0.45
            chair_z = -room.lengthM * 0.08
        else:
            chair_x = room.widthM * 0.18
            chair_z = room.lengthM / 2.0 - chair.depthM / 2.0 - 0.45 if wall == "FRONT_WALL" else -room.lengthM / 2.0 + chair.depthM / 2.0 + 0.45
        _place_floor(chair, chair_x, chair_z, 0.0, room)
        chair.rotationY = _rotation_towards(chair, bed.x, bed.z)
        chair.facingDirection = _engine.rotation_to_facing(chair.rotationY)
        chair.facingTarget = "bed"
        chair.layoutReasoning = "Ghế rời được đặt thành góc đọc sách phụ, không đứng lạc giữa phòng."
        chair.relations.append({"type": "bedroom_reading_corner", "target": "bed"})

    candidate["template"] = str(candidate.get("template", "bedroom")) + "_bedroom_refined"
    candidate["bedroomLayoutRefinement"] = "bedroom_role_specific_v1"
    return candidate


def generate_candidates(room: Any, products: List[Any]) -> List[Dict[str, Any]]:
    candidates = _BASE_GENERATE_CANDIDATES(room, products)
    if room.type != "bedroom":
        return candidates
    return [_refine_bedroom_candidate(candidate, room) for candidate in candidates]


# ---------------------------------------------------------------------------
# Public wrapper / health status
# ---------------------------------------------------------------------------

def finalize_layout(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = _BASE_FINALIZE_LAYOUT(payload)
    metrics = result.setdefault("metrics", {})
    metrics["bedroomLayoutRefinePatchInstalled"] = True
    metrics["bedroomRolePlacement"] = "bed_wall_nightstand_rug_bench_storage_v1"
    return result


def model_patch_status() -> Dict[str, Any]:
    status = _BASE_MODEL_PATCH_STATUS()
    status.update({
        "bedroomLayoutRefinePatchInstalled": _BEDROOM_PATCH_INSTALLED,
        "bedroomRolePlacement": "bed_wall_nightstand_rug_bench_storage_v1",
    })
    return status


def install_bedroom_layout_refine_patch() -> None:
    global _BEDROOM_PATCH_INSTALLED
    if _BEDROOM_PATCH_INSTALLED:
        return
    # Add bench as an allowed bedroom role if the base engine does not know it.
    _engine.DEFAULT_DIMS.setdefault("bench", (1.20, 0.42, 0.45))
    bedroom = _engine.ROOM_COMPOSITIONS.get("bedroom")
    if bedroom:
        bedroom["allowed"].add("bench")
        bedroom["quota"]["bench"] = max(int(bedroom["quota"].get("bench", 1)), 1)
        if "bench" not in bedroom["priority"]:
            bedroom["priority"].insert(4, "bench")

    _engine.select_products = select_products
    _engine.generate_candidates = generate_candidates
    _engine.finalize_layout = finalize_layout
    _BEDROOM_PATCH_INSTALLED = True


install_bedroom_layout_refine_patch()
