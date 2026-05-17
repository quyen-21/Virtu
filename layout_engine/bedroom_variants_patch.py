from __future__ import annotations

"""
Bedroom variant generation patch.

Installed after bedroom_layout_refine_patch.

Problem fixed:
- bedroom_layout_refine_patch made bedrooms safer, but too deterministic;
- many bedroom tests looked nearly identical because all candidates were refined
  into the same bed-wall-centered pattern.

This patch generates multiple bedroom layout variants before final scoring:
- different headboard walls;
- centered / left-biased / right-biased bed placement;
- different dressing/storage/reading zones;
- different ceiling-lamp alignment.

The existing Shapely repair and score_layout pipeline still chooses the best
candidate, so this patch adds diversity without removing collision protection.
"""

import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from layout_engine import engine as _engine
from layout_engine import bedroom_layout_refine_patch as _bedroom_patch

_BASE_GENERATE_CANDIDATES = _engine.generate_candidates
_BASE_SCORE_LAYOUT = _engine.score_layout
_BASE_FINALIZE_LAYOUT = _engine.finalize_layout
_BASE_MODEL_PATCH_STATUS = _bedroom_patch.model_patch_status

_BEDROOM_VARIANTS_PATCH_INSTALLED = False


# ---------------------------------------------------------------------------
# Helpers
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


def _wall_center(room: Any, wall: str, item: Any, along_offset: float = 0.0, wall_clearance: float = 0.12) -> Tuple[float, float]:
    wall = str(wall or "BACK_WALL").upper()
    along_offset = max(-0.38, min(0.38, along_offset))
    if wall == "FRONT_WALL":
        x = along_offset * room.widthM
        z = room.lengthM / 2.0 - item.depthM / 2.0 - wall_clearance
    elif wall == "LEFT_WALL":
        x = -room.widthM / 2.0 + item.depthM / 2.0 + wall_clearance
        z = along_offset * room.lengthM
    elif wall == "RIGHT_WALL":
        x = room.widthM / 2.0 - item.depthM / 2.0 - wall_clearance
        z = along_offset * room.lengthM
    else:
        x = along_offset * room.widthM
        z = -room.lengthM / 2.0 + item.depthM / 2.0 + wall_clearance
    return x, z


def _wall_fits_bed(room: Any, wall: str, bed: Any) -> bool:
    # Conservative fit check. Bed width lies roughly along the wall.
    wall = str(wall or "BACK_WALL").upper()
    along_space = room.widthM if wall in {"BACK_WALL", "FRONT_WALL"} else room.lengthM
    return bed.widthM + 0.80 <= along_space


def _candidate_walls(room: Any, bed: Any) -> List[str]:
    walls = ["BACK_WALL", "RIGHT_WALL", "LEFT_WALL", "FRONT_WALL"]
    fit = [wall for wall in walls if _wall_fits_bed(room, wall, bed)]
    return fit or ["BACK_WALL"]


def _side_walls(wall: str) -> Tuple[str, str]:
    return _engine.side_walls_for_wall(wall)


def _opposite_wall(wall: str) -> str:
    return _engine.opposite_wall(wall)


def _place_wall_item(item: Any, room: Any, wall: str, along: float = 0.0) -> None:
    if item.category in {"mirror", "wall_art"}:
        _engine.place_wall_art(item, room, wall, along=along)
    else:
        _engine.place_against_wall(item, room, wall, along=along)
    item.facingTarget = "room_center"


def _text(item: Any) -> str:
    raw = item.raw if isinstance(getattr(item, "raw", None), dict) else {}
    values = [item.category, item.name, raw.get("category"), raw.get("name"), raw.get("title")]
    return " ".join(_engine.canonical_text(v or "") for v in values)


def _is_lamp(item: Any) -> bool:
    text = _text(item)
    return item.category == "ceiling_lamp" or "đèn trần" in text or "den tran" in text or "ceiling" in text or "pendant" in text


# ---------------------------------------------------------------------------
# Variant placement
# ---------------------------------------------------------------------------

def _apply_bedroom_variant(candidate: Dict[str, Any], room: Any, variant: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cand = copy.deepcopy(candidate)
    items = cand.get("items", [])
    grouped = _by_category(items)
    bed = (grouped.get("bed") or [None])[0]
    if bed is None:
        return None

    wall = str(variant.get("bedWall", "BACK_WALL"))
    if not _wall_fits_bed(room, wall, bed):
        return None

    along = float(variant.get("along", 0.0))
    zone = str(variant.get("zone", "balanced"))
    lamp_mode = str(variant.get("lampMode", "over_bed"))

    fx, fz = _wall_vector(wall)
    px, pz = _perp_vector(fx, fz)
    rotation = _bed_rotation_for_head_wall(wall)

    # 1) Bed position variant.
    bed_x, bed_z = _wall_center(room, wall, bed, along_offset=along, wall_clearance=0.12)
    _place_floor(bed, bed_x, bed_z, rotation, room)
    bed.anchorWall = wall
    bed.facingTarget = "room_center"
    bed.layoutReasoning = f"Bedroom variant: bed đặt ở {wall}, offset {along:.2f}, mode {zone}."
    bed.relations.append({"type": "variant_headboard_wall", "target": wall})

    # 2) Rug under lower/middle bed.
    rug = (grouped.get("rug") or [None])[0]
    if rug is not None:
        rug_x = bed.x - fx * min(0.35, bed.depthM * 0.18)
        rug_z = bed.z - fz * min(0.35, bed.depthM * 0.18)
        _place_floor(rug, rug_x, rug_z, rotation, room)
        rug.layoutReasoning = "Variant: thảm căn theo trục giường."
        rug.relations.append({"type": "under", "target": "bed"})

    # 3) Nightstands beside headboard.
    nightstands = grouped.get("nightstand", [])[:2]
    for idx, ns in enumerate(nightstands):
        sign = -1.0 if idx == 0 else 1.0
        side_offset = bed.widthM / 2.0 + ns.widthM / 2.0 + 0.16
        head_offset = bed.depthM * 0.23
        ns_x = bed.x + px * sign * side_offset + fx * head_offset
        ns_z = bed.z + pz * sign * side_offset + fz * head_offset
        _place_floor(ns, ns_x, ns_z, rotation, room)
        ns.anchorWall = wall
        ns.facingTarget = "bed"
        ns.layoutReasoning = "Variant: nightstand đối xứng hai bên đầu giường."
        ns.relations.append({"type": "beside", "target": "bed"})

    # 4) Bench / bench-like item at foot of bed. If there is no bench category,
    # use one selected chair/armchair as a reading chair, not foot bench.
    benches = grouped.get("bench", [])
    if benches:
        bench = benches[0]
        gap = 0.28
        bx = bed.x - fx * (bed.depthM / 2.0 + bench.depthM / 2.0 + gap)
        bz = bed.z - fz * (bed.depthM / 2.0 + bench.depthM / 2.0 + gap)
        _place_floor(bench, bx, bz, rotation, room)
        bench.facingTarget = "bed"
        bench.layoutReasoning = "Variant: bench đặt cuối giường."
        bench.relations.append({"type": "foot_of_bed", "target": "bed"})

    # 5) Mirror / dressing zone changes by variant.
    mirrors = grouped.get("mirror", [])
    left_wall, right_wall = _side_walls(wall)
    opposite = _opposite_wall(wall)
    dressing_wall = left_wall if zone in {"balanced", "left_storage"} else right_wall
    if zone == "opposite_dressing":
        dressing_wall = opposite
    for mirror in mirrors[:1]:
        along_m = -room.lengthM * 0.18 if dressing_wall in {"LEFT_WALL", "RIGHT_WALL"} else -room.widthM * 0.18
        _place_wall_item(mirror, room, dressing_wall, along=along_m)
        mirror.layoutReasoning = f"Variant: gương đặt ở {dressing_wall} làm dressing zone."
        mirror.relations.append({"type": "variant_dressing_zone", "target": dressing_wall})

    # 6) Storage zone changes by variant.
    storage_items = grouped.get("wardrobe", []) + grouped.get("cabinet", []) + grouped.get("bookshelf", [])
    if zone == "left_storage":
        storage_wall = left_wall
    elif zone == "right_storage":
        storage_wall = right_wall
    else:
        storage_wall = opposite if storage_items and storage_items[0].category == "wardrobe" else right_wall
    for idx, storage in enumerate(storage_items[:3]):
        along_s = (idx - 1) * 0.75
        if storage_wall in {"LEFT_WALL", "RIGHT_WALL"}:
            along_val = max(-room.lengthM * 0.35, min(room.lengthM * 0.35, along_s))
        else:
            along_val = max(-room.widthM * 0.35, min(room.widthM * 0.35, along_s))
        _place_wall_item(storage, room, storage_wall, along=along_val)
        storage.layoutReasoning = f"Variant: storage đặt sát {storage_wall}."
        storage.relations.append({"type": "variant_storage_zone", "target": storage_wall})

    # 7) Desk / chair / reading zone.
    desks = grouped.get("desk", [])
    chairs = grouped.get("chair", []) + grouped.get("armchair", [])
    reading_wall = right_wall if zone in {"balanced", "left_storage"} else left_wall
    for desk in desks[:1]:
        _place_wall_item(desk, room, opposite, along=room.widthM * 0.18 if opposite in {"BACK_WALL", "FRONT_WALL"} else room.lengthM * 0.18)
        desk.layoutReasoning = f"Variant: desk/vanity đặt ở {opposite}."
        desk.relations.append({"type": "variant_secondary_desk_zone", "target": opposite})

    for chair in chairs[:1]:
        # Reading chair near side wall, looking back to the bed.
        if reading_wall == "LEFT_WALL":
            cx = -room.widthM / 2.0 + chair.widthM / 2.0 + 0.45
            cz = bed.z - fz * min(1.40, room.lengthM * 0.16)
        elif reading_wall == "RIGHT_WALL":
            cx = room.widthM / 2.0 - chair.widthM / 2.0 - 0.45
            cz = bed.z - fz * min(1.40, room.lengthM * 0.16)
        elif reading_wall == "FRONT_WALL":
            cx = bed.x + px * min(1.20, room.widthM * 0.18)
            cz = room.lengthM / 2.0 - chair.depthM / 2.0 - 0.45
        else:
            cx = bed.x + px * min(1.20, room.widthM * 0.18)
            cz = -room.lengthM / 2.0 + chair.depthM / 2.0 + 0.45
        _place_floor(chair, cx, cz, 0.0, room)
        chair.rotationY = _rotation_towards(chair, bed.x, bed.z)
        chair.facingDirection = _engine.rotation_to_facing(chair.rotationY)
        chair.facingTarget = "bed"
        chair.layoutReasoning = f"Variant: ghế đọc sách đặt gần {reading_wall}."
        chair.relations.append({"type": "variant_reading_corner", "target": reading_wall})

    # 8) Ceiling lamps. Use mode to avoid all lamps appearing in exactly same spot.
    lamps = [item for item in items if _is_lamp(item)]
    if lamps:
        if lamp_mode == "room_center":
            base_x, base_z = 0.0, 0.0
        elif lamp_mode == "foot_bed":
            base_x = bed.x - fx * min(0.55, bed.depthM * 0.30)
            base_z = bed.z - fz * min(0.55, bed.depthM * 0.30)
        else:
            base_x = bed.x
            base_z = bed.z
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
            lamp.facingTarget = "bed" if lamp_mode != "room_center" else "room_center"
            _engine.clamp_inside_room(lamp, room)
            lamp.layoutReasoning = f"Variant: đèn trần đặt theo mode {lamp_mode}."
            lamp.relations.append({"type": "variant_ceiling_lamp", "target": lamp.facingTarget})

    cand["template"] = f"{cand.get('template', 'bedroom')}_variant_{wall}_{zone}_{lamp_mode}_{along:.2f}"
    cand["bedroomVariant"] = {
        "bedWall": wall,
        "along": along,
        "zone": zone,
        "lampMode": lamp_mode,
    }
    cand["bedroomLayoutRefinement"] = "bedroom_multi_variant_v1"
    return cand


def _variant_specs(room: Any, bed: Any) -> List[Dict[str, Any]]:
    walls = _candidate_walls(room, bed)
    offsets = [0.0]
    # Wide/long rooms can support asymmetric bed placement.
    if room.widthM >= 4.2 or room.lengthM >= 6.5:
        offsets = [0.0, -0.16, 0.16]

    zones = ["balanced", "left_storage", "right_storage", "opposite_dressing"]
    lamp_modes = ["over_bed", "foot_bed", "room_center"]

    specs: List[Dict[str, Any]] = []
    for wall in walls:
        for offset in offsets:
            for zone in zones:
                # Avoid too many candidates while keeping diversity.
                lamp_mode = lamp_modes[(len(specs) + int(abs(offset) * 100)) % len(lamp_modes)]
                specs.append({"bedWall": wall, "along": offset, "zone": zone, "lampMode": lamp_mode})
                if len(specs) >= 24:
                    return specs
    return specs


def generate_candidates(room: Any, products: List[Any]) -> List[Dict[str, Any]]:
    base_candidates = _BASE_GENERATE_CANDIDATES(room, products)
    if room.type != "bedroom" or not base_candidates:
        return base_candidates

    # Use one representative candidate as the source of selected items, then
    # generate controlled variants from it. Keep the original candidates too.
    source = base_candidates[0]
    grouped = _by_category(source.get("items", []))
    bed = (grouped.get("bed") or [None])[0]
    if bed is None:
        return base_candidates

    variants: List[Dict[str, Any]] = []
    seen_templates = set()
    for spec in _variant_specs(room, bed):
        candidate = _apply_bedroom_variant(source, room, spec)
        if candidate is None:
            continue
        key = candidate.get("template")
        if key in seen_templates:
            continue
        seen_templates.add(key)
        variants.append(candidate)

    # Put variants first so scoring sees them early, but still keep base/model
    # candidates for fallback.
    return variants + base_candidates


# ---------------------------------------------------------------------------
# Scoring tweaks: reward diversity and penalize repetitive/empty bedroom layouts
# ---------------------------------------------------------------------------

def _floor_items(candidate: Dict[str, Any]) -> List[Any]:
    return [it for it in candidate.get("items", []) if getattr(it, "layer", "floor") == "floor" and it.category != "rug"]


def score_layout(candidate: Dict[str, Any], room: Any, products: List[Any]) -> Dict[str, Any]:
    result = _BASE_SCORE_LAYOUT(candidate, room, products)
    if room.type != "bedroom":
        return result

    items = candidate.get("items", [])
    grouped = _by_category(items)
    bed = (grouped.get("bed") or [None])[0]
    floor_items = _floor_items(candidate)
    breakdown = result.setdefault("breakdown", {})

    if bed is not None:
        # Prefer bed not always exactly at geometric center in large rooms; slight
        # offset can create more natural zone planning, but not too much.
        center_offset = min(1.0, (abs(bed.x) / max(room.widthM / 2.0, 1e-6) + abs(bed.z) / max(room.lengthM / 2.0, 1e-6)) / 2.0)
        variant_balance = 1.0 - abs(center_offset - 0.16) / 0.45
        variant_balance = max(0.0, min(1.0, variant_balance))
        breakdown["bedroomVariantBalance"] = round(variant_balance, 4)
        result["total"] = round(float(result.get("total", 0.0)) + 0.8 * variant_balance, 4)

    # Reward actual zones: storage, dressing, reading/desk.
    relation_text = " ".join(str(rel.get("type", "")) for it in items for rel in getattr(it, "relations", []))
    zone_score = 0.0
    if "storage" in relation_text:
        zone_score += 0.25
    if "dressing" in relation_text:
        zone_score += 0.25
    if "reading" in relation_text or "desk" in relation_text:
        zone_score += 0.25
    if "ceiling" in relation_text:
        zone_score += 0.25
    breakdown["bedroomZoneScore"] = round(zone_score, 4)
    result["total"] = round(float(result.get("total", 0.0)) + 1.2 * zone_score, 4)

    # Penalize very empty large bedrooms.
    if room.widthM * room.lengthM >= 35 and len(floor_items) <= 3:
        breakdown["scoreCap_bedroomTooEmpty"] = 0.82
        result["total"] = round(min(float(result.get("total", 0.0)), 82.0), 4)

    return result


# ---------------------------------------------------------------------------
# Public wrapper / health status
# ---------------------------------------------------------------------------

def finalize_layout(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = _BASE_FINALIZE_LAYOUT(payload)
    metrics = result.setdefault("metrics", {})
    metrics["bedroomVariantsPatchInstalled"] = True
    metrics["bedroomVariantGeneration"] = "multi_wall_zone_lamp_variants_v1"
    return result


def model_patch_status() -> Dict[str, Any]:
    status = _BASE_MODEL_PATCH_STATUS()
    status.update({
        "bedroomVariantsPatchInstalled": _BEDROOM_VARIANTS_PATCH_INSTALLED,
        "bedroomVariantGeneration": "multi_wall_zone_lamp_variants_v1",
    })
    return status


def install_bedroom_variants_patch() -> None:
    global _BEDROOM_VARIANTS_PATCH_INSTALLED
    if _BEDROOM_VARIANTS_PATCH_INSTALLED:
        return
    _engine.generate_candidates = generate_candidates
    _engine.score_layout = score_layout
    _engine.finalize_layout = finalize_layout
    _BEDROOM_VARIANTS_PATCH_INSTALLED = True


install_bedroom_variants_patch()
