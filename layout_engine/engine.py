from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from shapely.geometry import Polygon, box
except Exception:  # pragma: no cover
    Polygon = None
    box = None

# Lazy-loaded trained model predictions
_model_predictor = None

def _get_model_predictor():
    global _model_predictor
    if _model_predictor is None:
        try:
            from inference import predict_positions, score_be_candidates, model_info
            _model_predictor = {
                "predict": predict_positions,
                "score": score_be_candidates,
                "info": model_info,
                "available": True,
            }
        except Exception as e:
            _model_predictor = {"available": False, "error": str(e)}
    return _model_predictor


# ============================================================
# 1) Canonical mapping
# ============================================================

ROOM_TYPE_ALIAS = {
    "living room": "living_room", "livingroom": "living_room", "phòng khách": "living_room", "phong khach": "living_room",
    "bed room": "bedroom", "bedroom": "bedroom", "master bedroom": "bedroom", "phòng ngủ": "bedroom", "phong ngu": "bedroom",
    "dining room": "dining_room", "dining": "dining_room", "phòng ăn": "dining_room", "phong an": "dining_room",
    "study room": "office", "office room": "office", "office": "office", "phòng làm việc": "office", "phong lam viec": "office",
    "kitchen": "kitchen", "phòng bếp": "kitchen", "phong bep": "kitchen",
}

CATEGORY_ALIAS = {
    "sofa": "sofa", "sectional sofa": "sofa", "loveseat": "sofa", "couch": "sofa", "ghế sofa": "sofa", "ghe sofa": "sofa",
    "coffee table": "coffee_table", "tea table": "coffee_table", "bàn nước": "coffee_table", "ban nuoc": "coffee_table", "bàn trà": "coffee_table", "ban tra": "coffee_table", "bàn sofa": "coffee_table", "ban sofa": "coffee_table",
    "side table": "side_table", "end table": "side_table", "bàn bên": "side_table", "ban ben": "side_table", "bàn cạnh": "side_table", "ban canh": "side_table",
    "tv stand": "tv_stand", "media console": "tv_stand", "tủ tivi": "tv_stand", "tu tivi": "tv_stand", "kệ tivi": "tv_stand", "ke tivi": "tv_stand", "kệ tv": "tv_stand", "ke tv": "tv_stand",
    "armchair": "armchair", "lounge chair": "armchair", "recliner": "armchair", "ghế thư giãn": "armchair", "ghe thu gian": "armchair", "ghế đơn": "armchair", "ghe don": "armchair",
    "chair": "chair", "ghế": "chair", "ghe": "chair",
    "dining chair": "dining_chair", "ghế ăn": "dining_chair", "ghe an": "dining_chair",
    "office chair": "office_chair", "ghế văn phòng": "office_chair", "ghe van phong": "office_chair",
    "desk": "desk", "bàn làm việc": "desk", "ban lam viec": "desk",
    "dining table": "dining_table", "bàn ăn": "dining_table", "ban an": "dining_table",
    "table": "table",
    "bed": "bed", "giường": "bed", "giuong": "bed",
    "nightstand": "nightstand", "bedside table": "nightstand", "bàn đầu giường": "nightstand", "ban dau giuong": "nightstand",
    "wardrobe": "wardrobe", "closet": "wardrobe", "tủ quần áo": "wardrobe", "tu quan ao": "wardrobe",
    "cabinet": "cabinet", "tủ": "cabinet", "tu": "cabinet",
    "bookshelf": "bookshelf", "shelf": "bookshelf", "kệ sách": "bookshelf", "ke sach": "bookshelf",
    "drawer": "drawer", "dresser": "dresser",
    "rug": "rug", "carpet": "rug", "thảm": "rug", "tham": "rug",
    "lamp": "lamp", "floor lamp": "lamp", "table lamp": "lamp", "đèn": "lamp", "den": "lamp",
    "plant": "plant", "cây trang trí": "plant", "cay trang tri": "plant", "cây": "plant", "cay": "plant",
    "mirror": "mirror", "gương": "mirror", "guong": "mirror",
    "painting": "wall_art", "picture": "wall_art", "wall art": "wall_art", "wall decor": "wall_art", "tranh": "wall_art",
    "stool": "stool", "bench": "bench",
}

DEFAULT_DIMS = {
    "sofa": (2.40, 0.90, 0.80),
    "coffee_table": (1.20, 0.65, 0.42),
    "tv_stand": (1.80, 0.45, 0.55),
    "armchair": (0.85, 0.85, 0.90),
    "chair": (0.55, 0.55, 0.85),
    "side_table": (0.55, 0.45, 0.50),
    "rug": (2.40, 1.70, 0.03),
    "lamp": (0.35, 0.35, 1.45),
    "plant": (0.55, 0.55, 1.30),
    "wall_art": (0.80, 0.05, 0.70),
    "mirror": (0.70, 0.05, 1.00),
    "bed": (1.80, 2.00, 0.55),
    "nightstand": (0.50, 0.45, 0.55),
    "wardrobe": (1.60, 0.60, 2.20),
    "cabinet": (1.20, 0.45, 0.90),
    "desk": (1.40, 0.70, 0.75),
    "office_chair": (0.65, 0.65, 0.95),
    "bookshelf": (0.90, 0.35, 1.80),
    "dining_table": (1.60, 0.90, 0.75),
    "dining_chair": (0.50, 0.55, 0.85),
    "table": (1.20, 0.70, 0.75),
    "stool": (0.40, 0.40, 0.50),
    "bench": (1.20, 0.40, 0.45),
    "drawer": (0.80, 0.45, 0.75),
    "dresser": (1.20, 0.50, 0.80),
    "floor_lamp": (0.40, 0.40, 1.60),
    "ceiling_lamp": (0.50, 0.50, 0.30),
}

# category priority: smaller number = more important, less likely to be pushed around
PRIORITY = {
    "bed": 1, "sofa": 1, "dining_table": 1, "desk": 1,
    "tv_stand": 2, "coffee_table": 2, "wardrobe": 2,
    "nightstand": 3, "dining_chair": 3, "office_chair": 3,
    "armchair": 4, "chair": 4, "side_table": 5, "cabinet": 5, "bookshelf": 5,
    "stool": 5, "bench": 5, "drawer": 5, "dresser": 5,
    "rug": 6, "lamp": 7, "plant": 7, "floor_lamp": 7, "ceiling_lamp": 7,
    "wall_art": 8, "mirror": 8,
}

WALL_CATEGORIES = {"sofa", "tv_stand", "bed", "wardrobe", "cabinet", "bookshelf", "desk"}
WALL_MOUNT_CATEGORIES = {"wall_art", "mirror"}
DECOR_CATEGORIES = {"plant", "lamp", "wall_art", "mirror"}
TOP_SURFACE_CATEGORIES = {"side_table", "coffee_table", "tv_stand", "desk", "nightstand", "cabinet"}

# Minimum gap (meters) enforced between any two floor items after collision resolution.
MIN_ITEM_GAP = 0.05


# ============================================================
# 2) Data classes
# ============================================================

@dataclass
class Room:
    widthM: float = 4.0
    lengthM: float = 5.0
    heightM: float = 2.8
    type: str = "living_room"
    style: str = "modern"

@dataclass
class Product:
    productId: str
    name: str
    category: str
    widthM: float
    depthM: float
    heightM: float
    modelUrl: str = ""
    imageUrl: str = ""
    styles: List[str] = field(default_factory=list)
    price: Any = None
    sourceScore: float = 0.5
    raw: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LayoutItem:
    productId: str
    name: str
    category: str
    widthM: float
    depthM: float
    heightM: float
    x: float = 0.0
    z: float = 0.0
    y: float = 0.0
    rotationY: float = 0.0
    facingDirection: str = "SOUTH"
    facingTarget: str = ""
    anchorWall: str = "FLOATING"
    layer: str = "floor"
    modelUrl: str = ""
    imageUrl: str = ""
    sourceScore: float = 0.5
    layoutReasoning: str = ""
    supportSurfaceId: Optional[str] = None
    relations: List[Dict[str, str]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def output(self) -> Dict[str, Any]:
        return {
            "productId": self.productId,
            "name": self.name,
            "category": self.category,
            "score": round(float(self.sourceScore), 4),
            "modelUrl": self.modelUrl,
            "imageUrl": self.imageUrl,
            "position": {"x": round(float(self.x), 4), "y": round(float(self.y), 4), "z": round(float(self.z), 4)},
            "rotationY": round(float(self.rotationY % 360), 4),
            "facingDirection": self.facingDirection,
            "facingTarget": self.facingTarget,
            "anchorWall": self.anchorWall,
            "footprint": {"widthM": round(float(self.widthM), 4), "depthM": round(float(self.depthM), 4), "heightM": round(float(self.heightM), 4)},
            "layer": self.layer,
            "supportSurfaceId": self.supportSurfaceId,
            "layoutReasoning": self.layoutReasoning,
            "relations": self.relations,
        }


# ============================================================
# 3) Utility helpers
# ============================================================

def canonical_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def canonical_room_type(x: Any) -> str:
    s = canonical_text(x)
    if s in ROOM_TYPE_ALIAS:
        return ROOM_TYPE_ALIAS[s]
    for k, v in ROOM_TYPE_ALIAS.items():
        if k in s:
            return v
    return s.replace(" ", "_") if s else "living_room"


def canonical_category(x: Any) -> str:
    s = canonical_text(x)
    if s in CATEGORY_ALIAS:
        return CATEGORY_ALIAS[s]
    for k, v in CATEGORY_ALIAS.items():
        if k in s:
            return v
    return s.replace(" ", "_") if s else "unknown"


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def normalize_dim(value: Any, default: float) -> float:
    """Accept meters or centimeters. Values > 15 are treated as centimeters."""
    v = as_float(value, default)
    if v <= 0:
        return default
    # Most furniture dimensions in meters are below 15. Values like 120, 240 are centimeters.
    if v > 15:
        return v / 100.0
    return v


def nested_get(d: Dict[str, Any], *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def first_value(*values: Any, default: Any = None) -> Any:
    for v in values:
        if v is not None and v != "":
            return v
    return default


def facing_to_rotation(direction: str) -> float:
    return {"SOUTH": 0.0, "EAST": 90.0, "NORTH": 180.0, "WEST": 270.0}.get(direction, 0.0)


def rotation_to_facing(rotation_y: float) -> str:
    r = int(round(rotation_y / 90.0) * 90) % 360
    return {0: "SOUTH", 90: "EAST", 180: "NORTH", 270: "WEST"}.get(r, "SOUTH")


def facing_vector(direction: str) -> Tuple[float, float]:
    return {"SOUTH": (0.0, 1.0), "EAST": (1.0, 0.0), "NORTH": (0.0, -1.0), "WEST": (-1.0, 0.0)}.get(direction, (0.0, 1.0))


def opposite_facing(direction: str) -> str:
    return {"SOUTH": "NORTH", "NORTH": "SOUTH", "EAST": "WEST", "WEST": "EAST"}.get(direction, "NORTH")


def perpendicular_directions(direction: str) -> Tuple[str, str]:
    if direction in {"EAST", "WEST"}:
        return "NORTH", "SOUTH"
    return "WEST", "EAST"


def signed_distance_to_wall(room: Room, wall: str) -> float:
    if wall == "LEFT_WALL":
        return -room.widthM / 2.0
    if wall == "RIGHT_WALL":
        return room.widthM / 2.0
    if wall == "BACK_WALL":
        return -room.lengthM / 2.0
    if wall == "FRONT_WALL":
        return room.lengthM / 2.0
    return 0.0


def wall_to_facing(wall: str) -> str:
    return {
        "LEFT_WALL": "EAST", "RIGHT_WALL": "WEST",
        "BACK_WALL": "SOUTH", "FRONT_WALL": "NORTH",
    }.get(wall, "SOUTH")


def opposite_wall(wall: str) -> str:
    return {
        "LEFT_WALL": "RIGHT_WALL", "RIGHT_WALL": "LEFT_WALL",
        "BACK_WALL": "FRONT_WALL", "FRONT_WALL": "BACK_WALL",
    }.get(wall, "FRONT_WALL")


def side_walls_for_wall(wall: str) -> Tuple[str, str]:
    if wall in {"LEFT_WALL", "RIGHT_WALL"}:
        return "BACK_WALL", "FRONT_WALL"
    return "LEFT_WALL", "RIGHT_WALL"


def axis_for_wall(wall: str) -> str:
    return "X" if wall in {"LEFT_WALL", "RIGHT_WALL"} else "Z"


def style_match_score(room_style: str, styles: Iterable[str]) -> float:
    if not room_style:
        return 0.5
    rs = canonical_text(room_style)
    joined = " ".join(canonical_text(s) for s in styles or [])
    if not joined:
        return 0.5
    return 1.0 if rs in joined or joined in rs else 0.65


# ============================================================
# 4) Normalize BE/recommend payload without changing BE JSON
# ============================================================

def normalize_room(payload: Dict[str, Any]) -> Room:
    room = payload.get("room") or payload.get("roomInfo") or {}
    width = first_value(room.get("widthM"), room.get("width_m"), room.get("width"), payload.get("widthM"), payload.get("width"), default=4.0)
    length = first_value(room.get("lengthM"), room.get("length_m"), room.get("length"), room.get("depthM"), payload.get("lengthM"), payload.get("length"), default=5.0)
    height = first_value(room.get("heightM"), room.get("height_m"), room.get("height"), payload.get("heightM"), payload.get("height"), default=2.8)
    return Room(
        widthM=max(1.5, normalize_dim(width, 4.0)),
        lengthM=max(1.5, normalize_dim(length, 5.0)),
        heightM=max(2.0, normalize_dim(height, 2.8)),
        type=canonical_room_type(first_value(room.get("type"), room.get("room_type"), room.get("roomType"), payload.get("roomType"), default="living_room")),
        style=str(first_value(room.get("style"), payload.get("style"), default="modern") or "modern"),
    )


def extract_products(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rec = payload.get("recommendation") or {}
    products = rec.get("products") or payload.get("products") or payload.get("recommendedProducts") or []
    if isinstance(products, dict):
        products = list(products.values())
    return [p for p in products if isinstance(p, dict)]


def normalize_product(p: Dict[str, Any], index: int, model_url_by_id: Optional[Dict[str, str]] = None) -> Product:
    model_url_by_id = model_url_by_id or {}
    category = canonical_category(first_value(p.get("category"), p.get("type"), p.get("productType"), default="unknown"))
    default_w, default_d, default_h = DEFAULT_DIMS.get(category, (0.80, 0.60, 0.70))
    dims = p.get("dimensions") or p.get("dimension") or {}

    width = first_value(
        p.get("widthM"), p.get("width_m"), p.get("item_width_m"),
        dims.get("widthM"), dims.get("width_m"), dims.get("width"), dims.get("w"),
        default=default_w,
    )
    depth = first_value(
        p.get("depthM"), p.get("depth_m"), p.get("lengthM"), p.get("length_m"), p.get("item_depth_m"),
        dims.get("depthM"), dims.get("depth_m"), dims.get("depth"), dims.get("lengthM"), dims.get("length_m"), dims.get("length"), dims.get("d"),
        default=default_d,
    )
    height = first_value(
        p.get("heightM"), p.get("height_m"), p.get("item_height_m"),
        dims.get("heightM"), dims.get("height_m"), dims.get("height"), dims.get("h"),
        default=default_h,
    )

    pid = str(first_value(p.get("id"), p.get("productId"), p.get("product_id"), default=f"ai-product-{index+1}"))
    styles = p.get("styles") or p.get("style") or []
    if isinstance(styles, str):
        styles = [styles]
    score = as_float(first_value(p.get("final_score"), p.get("ranking_score"), p.get("score"), p.get("style_score"), default=0.5), 0.5)
    return Product(
        productId=pid,
        name=str(first_value(p.get("name"), p.get("title"), default=category)),
        category=category,
        widthM=max(0.05, normalize_dim(width, default_w)),
        depthM=max(0.03, normalize_dim(depth, default_d)),
        heightM=max(0.02, normalize_dim(height, default_h)),
        modelUrl=str(first_value(p.get("modelUrl"), p.get("model_url"), model_url_by_id.get(pid), default="") or ""),
        imageUrl=str(first_value(p.get("imageUrl"), p.get("image_url"), p.get("thumbnail"), default="") or ""),
        styles=list(styles),
        price=p.get("price"),
        sourceScore=score,
        raw=p,
    )


def normalize_payload(payload: Dict[str, Any]) -> Tuple[Room, List[Product]]:
    room = normalize_room(payload)
    model_url_by_id = payload.get("modelUrlById") or {}
    products = [normalize_product(p, i, model_url_by_id) for i, p in enumerate(extract_products(payload))]
    return room, products


# ============================================================
# 5) Composition selector
# ============================================================

ROOM_COMPOSITIONS = {
    "living_room": {
        "allowed": {"sofa", "coffee_table", "tv_stand", "armchair", "chair", "side_table", "rug", "lamp", "plant", "wall_art", "mirror", "bookshelf", "cabinet"},
        "priority": ["sofa", "coffee_table", "tv_stand", "rug", "armchair", "chair", "side_table", "lamp", "plant", "wall_art", "mirror", "bookshelf", "cabinet"],
        "quota": {"sofa": 1, "coffee_table": 1, "tv_stand": 1, "rug": 1, "armchair": 2, "chair": 4, "side_table": 2, "lamp": 2, "plant": 2, "wall_art": 2, "mirror": 1, "bookshelf": 1, "cabinet": 1},
    },
    "bedroom": {
        "allowed": {"bed", "nightstand", "wardrobe", "cabinet", "desk", "chair", "armchair", "side_table", "tv_stand", "rug", "lamp", "plant", "mirror", "wall_art", "bookshelf"},
        "priority": ["bed", "nightstand", "wardrobe", "rug", "desk", "chair", "armchair", "tv_stand", "side_table", "lamp", "plant", "mirror", "wall_art", "bookshelf", "cabinet"],
        "quota": {"bed": 1, "nightstand": 2, "wardrobe": 1, "rug": 1, "desk": 1, "chair": 1, "armchair": 1, "side_table": 1, "tv_stand": 1, "lamp": 2, "plant": 1, "mirror": 1, "wall_art": 1, "bookshelf": 1, "cabinet": 1},
    },
    "dining_room": {
        "allowed": {"dining_table", "dining_chair", "cabinet", "side_table", "rug", "lamp", "plant", "wall_art", "mirror"},
        "priority": ["dining_table", "dining_chair", "rug", "cabinet", "side_table", "lamp", "plant", "wall_art", "mirror"],
        "quota": {"dining_table": 1, "dining_chair": 8, "rug": 1, "cabinet": 1, "side_table": 1, "lamp": 2, "plant": 2, "wall_art": 1, "mirror": 1},
    },
    "office": {
        "allowed": {"desk", "office_chair", "chair", "armchair", "sofa", "bookshelf", "cabinet", "rug", "lamp", "plant", "wall_art"},
        "priority": ["desk", "office_chair", "chair", "armchair", "sofa", "bookshelf", "cabinet", "rug", "lamp", "plant", "wall_art"],
        "quota": {"desk": 1, "office_chair": 1, "chair": 2, "armchair": 2, "sofa": 1, "bookshelf": 2, "cabinet": 1, "rug": 1, "lamp": 2, "plant": 1, "wall_art": 1},
    },
    "kitchen": {
        "allowed": {"cabinet", "dining_table", "dining_chair", "chair", "stool", "plant", "lamp", "side_table"},
        "priority": ["cabinet", "dining_table", "dining_chair", "chair", "stool", "side_table", "plant", "lamp"],
        "quota": {"cabinet": 3, "dining_table": 1, "dining_chair": 4, "chair": 4, "stool": 2, "side_table": 1, "plant": 1, "lamp": 1},
    },
    "bathroom": {
        "allowed": {"cabinet", "mirror", "lamp", "plant", "side_table", "stool"},
        "priority": ["cabinet", "mirror", "side_table", "lamp", "plant", "stool"],
        "quota": {"cabinet": 2, "mirror": 1, "side_table": 1, "lamp": 1, "plant": 1, "stool": 1},
    },
}

# Style-aware quota adjustments
STYLE_QUOTA_MODS = {
    "minimal": {"_remove": {"plant", "wall_art", "side_table"}, "_reduce": {"armchair": 1, "chair": 1, "lamp": 1, "cabinet": 0}},
    "japandi": {"_remove": {"wall_art", "side_table"}, "_reduce": {"armchair": 1, "chair": 1, "cabinet": 0}},
    "scandinavian": {"_add": {"plant": 1}},
    "luxury": {"_add": {"plant": 1, "wall_art": 1, "side_table": 1, "lamp": 1}},
    "boho": {"_add": {"plant": 2, "wall_art": 1, "lamp": 1}},
    "industrial": {"_reduce": {"plant": 0, "wall_art": 0}},
}

def apply_style_quota(quota: Dict[str, int], style: str) -> Dict[str, int]:
    s = canonical_text(style)
    for alias, canon in {"modern": "modern", "contemporary": "modern", "minimalist": "minimal",
                         "simple": "minimal", "nordic": "scandinavian", "premium": "luxury",
                         "classic": "luxury", "loft": "industrial", "eclectic": "boho"}.items():
        if alias in s:
            s = canon
            break
    mods = STYLE_QUOTA_MODS.get(s)
    if not mods:
        return quota
    q = dict(quota)
    for cat in mods.get("_remove", set()):
        q[cat] = 0
    for cat, val in mods.get("_reduce", {}).items():
        if cat in q:
            q[cat] = min(q[cat], val)
    for cat, val in mods.get("_add", {}).items():
        q[cat] = q.get(cat, 0) + val
    return q


def density_cap(density: str, requested_top_k: int) -> int:
    d = canonical_text(density)
    if "sparse" in d or "thưa" in d or "thua" in d:
        return min(requested_top_k, 6)
    if "dense" in d or "dày" in d or "day" in d:
        return min(requested_top_k, 12)
    return min(requested_top_k, 9)


def _dim_compatibility(p: Product, room: Room) -> float:
    """Score 0-1: how well the item dimensions fit the room proportionally."""
    area_ratio = (p.widthM * p.depthM) / max(room.widthM * room.lengthM, 1e-6)
    # Items taking > 25% of floor are too dominant
    if area_ratio > 0.25:
        return 0.1
    # Very small items (lamps, plants) are generally fine
    if area_ratio < 0.005:
        return 0.7
    # Sweet spot: 1-12% of room area
    if 0.01 <= area_ratio <= 0.12:
        return 1.0
    return max(0.2, 1.0 - abs(area_ratio - 0.06) * 6)


def select_products(room: Room, products: List[Product], payload: Dict[str, Any]) -> Tuple[List[Product], List[Dict[str, Any]], List[str]]:
    top_k = int(payload.get("topK") or payload.get("top_k") or 8)
    density = str(payload.get("furnitureDensity") or payload.get("furniture_density") or "medium")
    top_k = density_cap(density, max(1, top_k))

    comp = ROOM_COMPOSITIONS.get(room.type, ROOM_COMPOSITIONS["living_room"])
    allowed = comp["allowed"]
    quota = apply_style_quota(dict(comp["quota"]), room.style)
    priority = list(comp["priority"])

    # --- Integrate trained filter scores if available ---
    predictor = _get_model_predictor()
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
                        # Blend trained filter probability with original source score
                        p.sourceScore = 0.6 * prob_by_id[p.productId] + 0.4 * p.sourceScore
        except Exception:
            pass

    rejected: List[Dict[str, Any]] = []
    valid: List[Product] = []
    room_area = room.widthM * room.lengthM

    for p in products:
        if p.category not in allowed:
            rejected.append({"productId": p.productId, "name": p.name, "category": p.category, "reason": "category_not_allowed_for_room"})
            continue
        # hard scale guard: avoid items that are impossible in the room
        if p.widthM > room.widthM * 0.92 or p.depthM > room.lengthM * 0.92 or (p.widthM * p.depthM) > room_area * 0.45:
            rejected.append({"productId": p.productId, "name": p.name, "category": p.category, "reason": "too_large_for_room"})
            continue
        valid.append(p)

    selected: List[Product] = []
    used = set()
    count_by_cat: Dict[str, int] = {}

    def p_score(p: Product) -> float:
        dim_compat = _dim_compatibility(p, room)
        return 0.50 * p.sourceScore + 0.25 * style_match_score(room.style, p.styles) + 0.15 * min(1.0, (p.widthM * p.depthM) / max(0.1, room_area * 0.08)) + 0.10 * dim_compat

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

    # Fill remaining with valid but still respect quota
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


# ============================================================
# 6) Room planner
# ============================================================

def active_zone(room: Room, item_count: int) -> Dict[str, float]:
    area = room.widthM * room.lengthM
    # For large rooms with few items, use a smaller visual living zone.
    if area > 45 and item_count <= 9:
        zw = min(room.widthM - 0.8, max(4.2, min(6.8, room.widthM * 0.65)))
        zl = min(room.lengthM - 0.8, max(3.8, min(6.2, room.lengthM * 0.55)))
    else:
        zw = max(1.6, room.widthM - 0.8)
        zl = max(1.6, room.lengthM - 0.8)
    return {"x": -zw / 2.0, "z": -zl / 2.0, "widthM": zw, "lengthM": zl, "centerX": 0.0, "centerZ": 0.0}


def zone_bounds(zone: Dict[str, float]) -> Tuple[float, float, float, float]:
    x1 = zone["centerX"] - zone["widthM"] / 2.0
    x2 = zone["centerX"] + zone["widthM"] / 2.0
    z1 = zone["centerZ"] - zone["lengthM"] / 2.0
    z2 = zone["centerZ"] + zone["lengthM"] / 2.0
    return x1, x2, z1, z2


def default_focal_walls(room: Room) -> List[str]:
    # Try all walls, but put the wall that produces a compact group first.
    if room.widthM >= room.lengthM:
        return ["BACK_WALL", "FRONT_WALL", "LEFT_WALL", "RIGHT_WALL"]
    return ["LEFT_WALL", "RIGHT_WALL", "BACK_WALL", "FRONT_WALL"]


# ============================================================
# 7) Geometry using Shapely
# ============================================================

def oriented_box_polygon(item: LayoutItem):
    # For wall-mounted and top-surface items, use a small footprint only for boundary; collisions ignored later.
    theta = math.radians(item.rotationY % 360)
    # facing vector: rotationY 0 -> +Z, 90 -> +X
    fx, fz = math.sin(theta), math.cos(theta)
    rx, rz = math.cos(theta), -math.sin(theta)
    hw, hd = item.widthM / 2.0, item.depthM / 2.0
    pts = []
    for sw, sd in [(1, 1), (1, -1), (-1, -1), (-1, 1)]:
        x = item.x + sw * hw * rx + sd * hd * fx
        z = item.z + sw * hw * rz + sd * hd * fz
        pts.append((x, z))
    if Polygon is None:
        return pts
    return Polygon(pts)


def room_polygon(room: Room):
    if box is None:
        return None
    return box(-room.widthM / 2.0, -room.lengthM / 2.0, room.widthM / 2.0, room.lengthM / 2.0)


def polygon_bounds(poly) -> Tuple[float, float, float, float]:
    if Polygon is not None and hasattr(poly, "bounds"):
        minx, minz, maxx, maxz = poly.bounds
        return minx, maxx, minz, maxz
    xs = [p[0] for p in poly]
    zs = [p[1] for p in poly]
    return min(xs), max(xs), min(zs), max(zs)


def clamp_inside_room(item: LayoutItem, room: Room, margin: float = 0.05) -> int:
    old_x, old_z = item.x, item.z
    poly = oriented_box_polygon(item)
    minx, maxx, minz, maxz = polygon_bounds(poly)
    room_minx, room_maxx = -room.widthM / 2.0 + margin, room.widthM / 2.0 - margin
    room_minz, room_maxz = -room.lengthM / 2.0 + margin, room.lengthM / 2.0 - margin
    if minx < room_minx:
        item.x += room_minx - minx
    if maxx > room_maxx:
        item.x -= maxx - room_maxx
    if minz < room_minz:
        item.z += room_minz - minz
    if maxz > room_maxz:
        item.z -= maxz - room_maxz
    return int(abs(old_x - item.x) > 1e-6 or abs(old_z - item.z) > 1e-6)


def inside_room_penalty(items: List[LayoutItem], room: Room) -> float:
    rp = room_polygon(room)
    if rp is None:
        return 0.0
    penalty = 0.0
    for it in items:
        if it.layer not in {"floor", "top_surface"}:
            continue
        poly = oriented_box_polygon(it)
        if not rp.contains(poly):
            penalty += poly.area - rp.intersection(poly).area
    return penalty


def collision_area(items: List[LayoutItem]) -> float:
    area = 0.0
    polys = []
    for it in items:
        if it.layer != "floor" or it.category == "rug":
            continue
        polys.append((it, oriented_box_polygon(it)))
    if Polygon is None:
        return 0.0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            a, pa = polys[i]
            b, pb = polys[j]
            if a.supportSurfaceId == b.productId or b.supportSurfaceId == a.productId:
                continue
            inter = pa.intersection(pb).area
            if inter > 1e-6:
                area += inter
    return area


def min_pair_gap(items: List[LayoutItem]) -> float:
    if Polygon is None:
        return 0.0
    polys = [(it, oriented_box_polygon(it)) for it in items if it.layer == "floor" and it.category != "rug"]
    if len(polys) < 2:
        return 9.9
    d = 9.9
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            d = min(d, polys[i][1].distance(polys[j][1]))
    return d


def front_clearance_penalty(items: List[LayoutItem], room: Room, clear_m: float = 0.55) -> float:
    if Polygon is None:
        return 0.0
    penalty = 0.0
    floor_polys = [(it, oriented_box_polygon(it)) for it in items if it.layer == "floor" and it.category != "rug"]
    for it, poly in floor_polys:
        if it.category not in {"sofa", "armchair", "chair", "office_chair", "wardrobe", "cabinet", "desk", "tv_stand"}:
            continue
        fx, fz = facing_vector(it.facingDirection)
        # front rectangle centered in front of item
        front = deepcopy(it)
        front.widthM = max(0.4, it.widthM * 0.75)
        front.depthM = clear_m
        front.x = it.x + fx * (it.depthM / 2.0 + clear_m / 2.0)
        front.z = it.z + fz * (it.depthM / 2.0 + clear_m / 2.0)
        fpoly = oriented_box_polygon(front)
        for other, opoly in floor_polys:
            if other.productId == it.productId or other.category == "rug":
                continue
            if other.supportSurfaceId == it.productId or it.supportSurfaceId == other.productId:
                continue
            penalty += fpoly.intersection(opoly).area
    return penalty


# ============================================================
# 8) Candidate generation: templates per room
# ============================================================

def item_from_product(p: Product) -> LayoutItem:
    return LayoutItem(
        productId=p.productId,
        name=p.name,
        category=p.category,
        widthM=p.widthM,
        depthM=p.depthM,
        heightM=p.heightM,
        y=p.heightM / 2.0,
        modelUrl=p.modelUrl,
        imageUrl=p.imageUrl,
        sourceScore=p.sourceScore,
        raw=p.raw,
    )


def get_first(items: List[LayoutItem], *cats: str) -> Optional[LayoutItem]:
    for c in cats:
        for it in items:
            if it.category == c:
                return it
    return None


def get_many(items: List[LayoutItem], *cats: str) -> List[LayoutItem]:
    return [it for it in items if it.category in cats]


def set_facing(item: LayoutItem, direction: str):
    item.facingDirection = direction
    item.rotationY = facing_to_rotation(direction)


def place_against_wall(item: LayoutItem, room: Room, wall: str, along: float = 0.0, margin: float = 0.08):
    facing = wall_to_facing(wall)
    set_facing(item, facing)
    item.anchorWall = wall
    item.layer = "floor"
    if wall == "LEFT_WALL":
        item.x = -room.widthM / 2.0 + item.depthM / 2.0 + margin
        item.z = max(-room.lengthM / 2.0 + item.widthM / 2.0 + margin, min(room.lengthM / 2.0 - item.widthM / 2.0 - margin, along))
    elif wall == "RIGHT_WALL":
        item.x = room.widthM / 2.0 - item.depthM / 2.0 - margin
        item.z = max(-room.lengthM / 2.0 + item.widthM / 2.0 + margin, min(room.lengthM / 2.0 - item.widthM / 2.0 - margin, along))
    elif wall == "BACK_WALL":
        item.z = -room.lengthM / 2.0 + item.depthM / 2.0 + margin
        item.x = max(-room.widthM / 2.0 + item.widthM / 2.0 + margin, min(room.widthM / 2.0 - item.widthM / 2.0 - margin, along))
    elif wall == "FRONT_WALL":
        item.z = room.lengthM / 2.0 - item.depthM / 2.0 - margin
        item.x = max(-room.widthM / 2.0 + item.widthM / 2.0 + margin, min(room.widthM / 2.0 - item.widthM / 2.0 - margin, along))


def offset_item_from(base: LayoutItem, item: LayoutItem, direction: str, distance: float):
    fx, fz = facing_vector(direction)
    item.x = base.x + fx * distance
    item.z = base.z + fz * distance


def face_target(item: LayoutItem, target: LayoutItem):
    dx, dz = target.x - item.x, target.z - item.z
    if abs(dx) >= abs(dz):
        set_facing(item, "EAST" if dx > 0 else "WEST")
    else:
        set_facing(item, "SOUTH" if dz > 0 else "NORTH")



def apply_fallback_placement(items: List[LayoutItem], room: Room, focal_wall: str, zone: Dict[str, float]):
    unplaced = [it for it in items if not it.layoutReasoning and it.category not in {"rug", "ceiling_lamp"}]
    if not unplaced:
        return
    walls = [opposite_wall(focal_wall), *side_walls_for_wall(focal_wall), focal_wall]
    for idx, it in enumerate(unplaced):
        wall = walls[idx % len(walls)]
        place_against_wall(it, room, wall, along=(idx * 0.4) % max(1.0, room.widthM/2))
        it.layoutReasoning = "Sắp xếp tự động (fallback) vào sát tường để tránh đè lấn ở giữa phòng."
        it.facingTarget = "room_center"
        it.relations.append({"type": "against_wall", "target": wall})
        it.relations.append({"type": "face_to", "target": "room_center"})

def generate_living_candidate(room: Room, products: List[Product], focal_wall: str, variant: int = 0) -> Dict[str, Any]:
    items = [item_from_product(p) for p in products]
    zone = active_zone(room, len(items))
    _, _, z1, z2 = zone_bounds(zone)
    x1, x2, _, _ = zone_bounds(zone)
    z_center = zone["centerZ"]
    x_center = zone["centerX"]

    tv = get_first(items, "tv_stand")
    sofa = get_first(items, "sofa")
    table = get_first(items, "coffee_table", "table")
    rug = get_first(items, "rug")
    chairs = get_many(items, "armchair", "chair")
    sides = get_many(items, "side_table")
    plants = get_many(items, "plant")
    lamps = get_many(items, "lamp")
    arts = get_many(items, "wall_art", "mirror")
    cabinets = get_many(items, "bookshelf", "cabinet")

    tv_like = tv or (cabinets[0] if cabinets else None)
    sofa_wall = opposite_wall(focal_wall)
    center_along = z_center if focal_wall in {"LEFT_WALL", "RIGHT_WALL"} else x_center
    offset_along = (variant - 1) * 0.35

    main_seating = sofa
    if not main_seating and chairs:
        main_seating = chairs.pop(0)

    if tv_like:
        place_against_wall(tv_like, room, focal_wall, along=center_along + offset_along)
        tv_like.layoutReasoning = "Vật focal được đặt sát tường chính để tạo điểm nhìn rõ ràng."
        tv_like.facingTarget = "sofa"
        tv_like.relations.append({"type": "against_wall", "target": focal_wall})
    if main_seating and tv_like:
        tv_facing = tv_like.facingDirection
        seating_facing = opposite_facing(tv_facing)
        set_facing(main_seating, seating_facing)
        axis_len = room.widthM if focal_wall in {"LEFT_WALL", "RIGHT_WALL"} else room.lengthM
        target_distance = min(4.2, max(2.6, axis_len * 0.38))
        fx, fz = facing_vector(tv_facing)
        main_seating.x = tv_like.x + fx * target_distance
        main_seating.z = tv_like.z + fz * target_distance
        main_seating.anchorWall = "FLOATING_ZONE"
        main_seating.layoutReasoning = "Ghế chính được đặt đối diện focal wall."
        main_seating.facingTarget = "tv_stand"
        main_seating.relations.append({"type": "face_to", "target": "tv_stand"})
        clamp_inside_room(main_seating, room)
    elif main_seating:
        place_against_wall(main_seating, room, sofa_wall, along=center_along)
        main_seating.layoutReasoning = "Ghế chính được đặt sát tường để mở lối đi."
        main_seating.facingTarget = "room_center"
        main_seating.relations.append({"type": "against_wall", "target": sofa_wall})

    if main_seating and table:
        direction = main_seating.facingDirection
        distance = main_seating.depthM / 2.0 + table.depthM / 2.0 + 0.55
        offset_item_from(main_seating, table, direction, distance)
        set_facing(table, main_seating.facingDirection)
        table.anchorWall = "FLOATING"
        table.layoutReasoning = "Bàn nước nằm trước ghế chính."
        table.relations.append({"type": "in_front_of", "target": main_seating.category})
    elif table:
        table.x, table.z = x_center, z_center
        set_facing(table, "SOUTH")
        table.layoutReasoning = "Bàn nước đặt giữa phòng."
        table.relations.append({"type": "center_of", "target": "room"})

    if table:
        side_a, side_b = perpendicular_directions(main_seating.facingDirection if main_seating else "SOUTH")
        for idx, ch in enumerate(chairs[:2]):
            side_dir = side_a if idx == 0 else side_b
            gap = 0.45 + ch.depthM / 2.0 + table.widthM / 2.0
            offset_item_from(table, ch, side_dir, gap)
            face_target(ch, table)
            ch.anchorWall = "FLOATING"
            ch.layoutReasoning = "Ghế phụ được xoay về bàn nước."
            ch.facingTarget = "coffee_table"
            ch.relations.append({"type": "face_to", "target": "coffee_table"})
            ch.relations.append({"type": "near", "target": "coffee_table"})
    elif main_seating:
        for idx, ch in enumerate(chairs[:2]):
            side_dir = "NORTH" if idx == 0 else "SOUTH"
            offset_item_from(main_seating, ch, side_dir, main_seating.widthM / 2.0 + ch.depthM / 2.0 + 0.35)
            face_target(ch, main_seating)
            ch.layoutReasoning = "Ghế phụ xoay về ghế chính."
            ch.facingTarget = main_seating.category
            ch.relations.append({"type": "face_to", "target": main_seating.category})
            ch.relations.append({"type": "near", "target": main_seating.category})

    if main_seating:
        side_a, side_b = perpendicular_directions(main_seating.facingDirection)
        for idx, st in enumerate(sides[:2]):
            side = side_a if idx == 0 else side_b
            offset_item_from(main_seating, st, side, main_seating.widthM / 2.0 + st.widthM / 2.0 + 0.18)
            set_facing(st, main_seating.facingDirection)
            st.layoutReasoning = "Bàn phụ đặt cạnh ghế chính."
            st.relations.append({"type": "near", "target": main_seating.category})

    if rug:
        anchor = table or main_seating
        if anchor:
            rug.x = anchor.x
            rug.z = anchor.z
            rug.widthM = max(rug.widthM, (main_seating.widthM if main_seating else 1.8) + 0.6)
            rug.depthM = max(rug.depthM, (main_seating.depthM if main_seating else 0.8) + (table.depthM if table else 0.6) + 1.0)
        else:
            rug.x, rug.z = x_center, z_center
        rug.heightM = min(rug.heightM, 0.04)
        rug.y = 0.01
        rug.layer = "floor"
        set_facing(rug, main_seating.facingDirection if main_seating else "SOUTH")
        rug.layoutReasoning = "Thảm được đặt dưới cụm sinh hoạt chính."
        rug.relations.append({"type": "under", "target": "seating_group"})

    corner_positions: List[Tuple[float, float]] = []
    if focal_wall in {"LEFT_WALL", "RIGHT_WALL"}:
        x_corner = -room.widthM / 2.0 + 0.55 if focal_wall == "LEFT_WALL" else room.widthM / 2.0 - 0.55
        corner_positions = [(x_corner, z1 + 0.55), (x_corner, z2 - 0.55), (-x_corner, z2 - 0.55)]
    else:
        z_corner = -room.lengthM / 2.0 + 0.55 if focal_wall == "BACK_WALL" else room.lengthM / 2.0 - 0.55
        corner_positions = [(x1 + 0.55, z_corner), (x2 - 0.55, z_corner), (x2 - 0.55, -z_corner)]
    for idx, deco in enumerate(plants + lamps):
        if idx >= len(corner_positions):
            break
        deco.x, deco.z = corner_positions[idx]
        deco.anchorWall = "CORNER"
        set_facing(deco, "SOUTH")
        deco.layoutReasoning = "Decor được đưa về góc/tường để làm đầy không gian."
        deco.facingTarget = "room_center"
        deco.relations.append({"type": "near", "target": "corner"})
        deco.relations.append({"type": "face_to", "target": "room_center"})

    for idx, art in enumerate(arts):
        wall = sofa_wall if idx == 0 else focal_wall
        place_wall_art(art, room, wall, along=center_along + (idx - 0.5) * 0.7)

    for idx, cab in enumerate([c for c in cabinets if c is not tv_like]):
        wall = side_walls_for_wall(focal_wall)[idx % 2]
        place_against_wall(cab, room, wall, along=0.0)
        cab.layoutReasoning = "Tủ/kệ được kéo sát tường."
        cab.facingTarget = "room_center"
        cab.relations.append({"type": "against_wall", "target": wall})
        cab.relations.append({"type": "face_to", "target": "room_center"})

    apply_fallback_placement(items, room, focal_wall, zone)
    return {"items": items, "template": f"living_{focal_wall.lower()}_v{variant}", "focalWall": focal_wall, "activeZone": zone, "layoutAxis": "WIDTH" if focal_wall in {"LEFT_WALL", "RIGHT_WALL"} else "LENGTH"}


def place_wall_art(item: LayoutItem, room: Room, wall: str, along: float = 0.0):
    item.layer = "wall"
    item.anchorWall = wall
    set_facing(item, wall_to_facing(wall))
    if wall == "LEFT_WALL":
        item.x = -room.widthM / 2.0 + 0.03
        item.z = max(-room.lengthM / 2.0 + item.widthM / 2.0, min(room.lengthM / 2.0 - item.widthM / 2.0, along))
    elif wall == "RIGHT_WALL":
        item.x = room.widthM / 2.0 - 0.03
        item.z = max(-room.lengthM / 2.0 + item.widthM / 2.0, min(room.lengthM / 2.0 - item.widthM / 2.0, along))
    elif wall == "BACK_WALL":
        item.z = -room.lengthM / 2.0 + 0.03
        item.x = max(-room.widthM / 2.0 + item.widthM / 2.0, min(room.widthM / 2.0 - item.widthM / 2.0, along))
    else:
        item.z = room.lengthM / 2.0 - 0.03
        item.x = max(-room.widthM / 2.0 + item.widthM / 2.0, min(room.widthM / 2.0 - item.widthM / 2.0, along))
    item.y = min(room.heightM * 0.58, 1.55)
    item.layoutReasoning = "Tranh/gương được treo trên tường ở cao độ xem hợp lý, không đặt dưới sàn."


def generate_bedroom_candidate(room: Room, products: List[Product], focal_wall: str, variant: int = 0) -> Dict[str, Any]:
    items = [item_from_product(p) for p in products]
    zone = active_zone(room, len(items))
    tv = get_first(items, "tv_stand")
    bed = get_first(items, "bed")
    nights = get_many(items, "nightstand")
    wardrobe = get_first(items, "wardrobe", "cabinet")
    desk = get_first(items, "desk")
    chair = get_first(items, "chair", "office_chair")
    armchair = get_first(items, "armchair")
    if not armchair and chair and desk is None:
        armchair = chair
        chair = None
    side_table = get_first(items, "side_table")
    rug = get_first(items, "rug")
    lamps = get_many(items, "lamp")
    arts = get_many(items, "wall_art", "mirror")
    plant = get_first(items, "plant")

    if bed:
        place_against_wall(bed, room, focal_wall, along=0.0, margin=0.10)
        bed.layoutReasoning = "Đầu giường được neo sát tường."
        bed.facingTarget = "room_center"
        bed.relations.append({"type": "against_wall", "target": focal_wall})
        bed.relations.append({"type": "face_to", "target": "room_center"})
        side_a, side_b = perpendicular_directions(bed.facingDirection)
        for idx, nt in enumerate(nights[:2]):
            side = side_a if idx == 0 else side_b
            offset_item_from(bed, nt, side, bed.widthM / 2.0 + nt.widthM / 2.0 + 0.18)
            set_facing(nt, bed.facingDirection)
            nt.layoutReasoning = "Tủ đầu giường đặt cạnh giường."
            nt.facingTarget = "room_center"
            nt.relations.append({"type": "near", "target": "bed"})
            nt.relations.append({"type": "face_to", "target": "room_center"})

    if tv:
        place_against_wall(tv, room, opposite_wall(focal_wall), along=0.0)
        tv.layoutReasoning = "Kệ TV đặt đối diện giường."
        tv.facingTarget = "bed"
        tv.relations.append({"type": "against_wall", "target": opposite_wall(focal_wall)})
        tv.relations.append({"type": "face_to", "target": "bed"})

    if wardrobe:
        wall = side_walls_for_wall(focal_wall)[0]
        place_against_wall(wardrobe, room, wall, along=0.0)
        wardrobe.layoutReasoning = "Tủ quần áo sát tường phụ."
        wardrobe.facingTarget = "room_center"
        wardrobe.relations.append({"type": "against_wall", "target": wall})
        wardrobe.relations.append({"type": "face_to", "target": "room_center"})

    if desk:
        wall = side_walls_for_wall(focal_wall)[1]
        place_against_wall(desk, room, wall, along=0.6)
        desk.layoutReasoning = "Bàn làm việc đặt sát tường."
        desk.facingTarget = "chair"
        desk.relations.append({"type": "against_wall", "target": wall})
        if chair:
            offset_item_from(desk, chair, desk.facingDirection, desk.depthM / 2.0 + chair.depthM / 2.0 + 0.45)
            face_target(chair, desk)
            chair.layoutReasoning = "Ghế làm việc."
            chair.facingTarget = "desk"
            chair.relations.append({"type": "face_to", "target": "desk"})
            chair.relations.append({"type": "near", "target": "desk"})

    if armchair:
        w = side_walls_for_wall(focal_wall)[1] if not desk else side_walls_for_wall(focal_wall)[0]
        place_against_wall(armchair, room, w, along=-1.0)
        armchair.facingDirection = "SOUTH"
        armchair.rotationY = 45.0
        armchair.layoutReasoning = "Ghế thư giãn tạo góc đọc sách."
        armchair.facingTarget = "room_center"
        armchair.relations.append({"type": "near", "target": "corner"})
        armchair.relations.append({"type": "face_to", "target": "room_center"})
        if side_table:
            offset_item_from(armchair, side_table, "EAST" if armchair.x < 0 else "WEST", 0.7)
            set_facing(side_table, "SOUTH")
            side_table.layoutReasoning = "Bàn phụ đặt cạnh ghế thư giãn."
            side_table.relations.append({"type": "near", "target": "armchair"})

    if rug:
        anchor = bed or desk
        rug.x, rug.z = (anchor.x, anchor.z + 0.35) if anchor else (0.0, 0.0)
        rug.widthM = max(rug.widthM, 1.8)
        rug.depthM = max(rug.depthM, 1.4)
        rug.y = 0.01
        rug.layer = "floor"
        rug.layoutReasoning = "Thảm trải sàn."
        rug.relations.append({"type": "under", "target": "bed" if bed else "desk"})

    for idx, lamp in enumerate(lamps):
        if idx < len(nights):
            lamp.x, lamp.z = nights[idx].x, nights[idx].z
            lamp.y = nights[idx].heightM + lamp.heightM / 2.0
            lamp.layer = "top_surface"
            lamp.supportSurfaceId = nights[idx].productId
            lamp.layoutReasoning = "Đèn ngủ."
            lamp.relations.append({"type": "on_top_of", "target": "nightstand"})

    for idx, art in enumerate(arts):
        place_wall_art(art, room, opposite_wall(focal_wall) if not tv else focal_wall, along=(idx - 0.5) * 0.6)

    if plant:
        plant.x, plant.z = room.widthM / 2.0 - 0.6, room.lengthM / 2.0 - 0.6
        plant.anchorWall = "CORNER"
        plant.layoutReasoning = "Cây trang trí."
        plant.facingTarget = "room_center"
        plant.relations.append({"type": "near", "target": "corner"})

    apply_fallback_placement(items, room, focal_wall, zone)
    return {"items": items, "template": f"bedroom_{focal_wall.lower()}_v{variant}", "focalWall": focal_wall, "activeZone": zone, "layoutAxis": "WIDTH" if focal_wall in {"LEFT_WALL", "RIGHT_WALL"} else "LENGTH"}


def generate_dining_candidate(room: Room, products: List[Product], focal_wall: str, variant: int = 0) -> Dict[str, Any]:
    items = [item_from_product(p) for p in products]
    zone = active_zone(room, len(items))
    table = get_first(items, "dining_table", "table")
    chairs = get_many(items, "dining_chair", "chair")
    cabinet = get_first(items, "cabinet")
    rug = get_first(items, "rug")
    lamps = get_many(items, "lamp")
    plants = get_many(items, "plant")
    arts = get_many(items, "wall_art", "mirror")
    if table:
        table.x, table.z = 0.0, 0.0
        set_facing(table, "SOUTH" if variant % 2 == 0 else "EAST")
        table.layoutReasoning = "Bàn ăn được đặt làm tâm bố cục, đảm bảo ghế và lối đi xung quanh."
    if table:
        positions = ["NORTH", "SOUTH", "WEST", "EAST", "NORTH", "SOUTH", "WEST", "EAST"]
        counts = {"NORTH": 0, "SOUTH": 0, "WEST": 0, "EAST": 0}
        for idx, ch in enumerate(chairs[:8]):
            side = positions[idx]
            counts[side] += 1
            extra = (counts[side] - 1) * 0.6 - 0.3 if counts[side] > 1 else 0.0
            if side in {"NORTH", "SOUTH"}:
                ch.x = table.x + extra
                ch.z = table.z + (-1 if side == "NORTH" else 1) * (table.depthM / 2.0 + ch.depthM / 2.0 + 0.25)
            else:
                ch.x = table.x + (-1 if side == "WEST" else 1) * (table.widthM / 2.0 + ch.depthM / 2.0 + 0.25)
                ch.z = table.z + extra
            face_target(ch, table)
            ch.layoutReasoning = "Ghế ăn được đặt xung quanh bàn và quay vào bàn."
    if rug and table:
        rug.x, rug.z = table.x, table.z
        rug.widthM = max(rug.widthM, table.widthM + 1.2)
        rug.depthM = max(rug.depthM, table.depthM + 1.2)
        rug.y = 0.01
    if cabinet:
        place_against_wall(cabinet, room, focal_wall, along=0.0)
    for idx, lamp in enumerate(lamps):
        if table and idx == 0:
            lamp.x, lamp.z = table.x, table.z
            lamp.y = min(room.heightM - 0.5, 2.1)
            lamp.layer = "ceiling"
            lamp.anchorWall = "CEILING"
            lamp.layoutReasoning = "Đèn được đặt ở vùng trung tâm bàn ăn để tạo điểm nhấn."
    for idx, plant in enumerate(plants[:2]):
        plant.x = -room.widthM / 2.0 + 0.6 if idx == 0 else room.widthM / 2.0 - 0.6
        plant.z = room.lengthM / 2.0 - 0.6
        plant.anchorWall = "CORNER"
    for idx, art in enumerate(arts):
        place_wall_art(art, room, opposite_wall(focal_wall), along=(idx - 0.5) * 0.7)
    apply_fallback_placement(items, room, focal_wall, zone)
    return {"items": items, "template": f"dining_{focal_wall.lower()}_v{variant}", "focalWall": focal_wall, "activeZone": zone, "layoutAxis": "CENTER"}


def generate_office_candidate(room: Room, products: List[Product], focal_wall: str, variant: int = 0) -> Dict[str, Any]:
    items = [item_from_product(p) for p in products]
    zone = active_zone(room, len(items))
    desk = get_first(items, "desk")
    chair = get_first(items, "office_chair", "chair")
    client_chairs = [ch for ch in items if ch.category in {"chair", "armchair"} and ch is not chair]
    shelves = get_many(items, "bookshelf", "cabinet")
    sofa = get_first(items, "sofa")
    rug = get_first(items, "rug")
    lamps = get_many(items, "lamp")
    plants = get_many(items, "plant")
    arts = get_many(items, "wall_art", "mirror")

    if desk:
        if variant == 2:
            desk.x, desk.z = 0.0, 0.5
            set_facing(desk, "SOUTH")
            desk.layoutReasoning = "Bàn làm việc đặt giữa phòng."
            desk.facingTarget = "door"
            desk.relations.append({"type": "center_of", "target": "room"})
        else:
            place_against_wall(desk, room, focal_wall, along=0.0)
            desk.layoutReasoning = "Bàn làm việc sát tường."
            desk.facingTarget = "chair"
            desk.relations.append({"type": "against_wall", "target": focal_wall})
        
        if chair:
            offset_item_from(desk, chair, desk.facingDirection, desk.depthM / 2.0 + chair.depthM / 2.0 + 0.50)
            face_target(chair, desk)
            chair.layoutReasoning = "Ghế người ngồi làm việc."
            chair.facingTarget = "desk"
            chair.relations.append({"type": "face_to", "target": "desk"})
            chair.relations.append({"type": "near", "target": "desk"})
            
        for idx, cch in enumerate(client_chairs[:2]):
            offset_item_from(desk, cch, opposite_facing(desk.facingDirection), desk.depthM / 2.0 + cch.depthM / 2.0 + 0.6)
            cch.x += -0.6 if idx == 0 else 0.6
            face_target(cch, desk)
            cch.layoutReasoning = "Ghế khách."
            cch.facingTarget = "desk"
            cch.relations.append({"type": "face_to", "target": "desk"})
            cch.relations.append({"type": "near", "target": "desk"})

    if sofa:
        place_against_wall(sofa, room, side_walls_for_wall(focal_wall)[0], along=0.0)
        sofa.layoutReasoning = "Sofa nghỉ ngơi đặt sát tường bên."
        sofa.facingTarget = "room_center"
        sofa.relations.append({"type": "against_wall", "target": side_walls_for_wall(focal_wall)[0]})
        sofa.relations.append({"type": "face_to", "target": "room_center"})

    for idx, sh in enumerate(shelves[:2]):
        wall = side_walls_for_wall(focal_wall)[1] if idx == 0 else opposite_wall(focal_wall)
        place_against_wall(sh, room, wall, along=0.0)
        sh.layoutReasoning = "Kệ/tủ hồ sơ sát tường."
        sh.facingTarget = "room_center"
        sh.relations.append({"type": "against_wall", "target": wall})
        sh.relations.append({"type": "face_to", "target": "room_center"})

    if rug:
        rug.x, rug.z = (desk.x, desk.z) if desk else (0.0, 0.0)
        rug.widthM = max(rug.widthM, 1.6)
        rug.depthM = max(rug.depthM, 1.3)
        rug.y = 0.01
        rug.layer = "floor"
        rug.layoutReasoning = "Thảm lót sàn."
        rug.relations.append({"type": "under", "target": "desk"})

    for idx, lamp in enumerate(lamps):
        if desk and idx == 0:
            lamp.x, lamp.z = desk.x, desk.z
            lamp.y = desk.heightM + lamp.heightM / 2.0
            lamp.layer = "top_surface"
            lamp.supportSurfaceId = desk.productId
            lamp.layoutReasoning = "Đèn bàn."
            lamp.relations.append({"type": "on_top_of", "target": "desk"})

    for idx, plant in enumerate(plants[:1]):
        plant.x, plant.z = room.widthM / 2.0 - 0.6, room.lengthM / 2.0 - 0.6
        plant.anchorWall = "CORNER"

    for idx, art in enumerate(arts):
        place_wall_art(art, room, opposite_wall(focal_wall), along=0.0)

    apply_fallback_placement(items, room, focal_wall, zone)
    return {"items": items, "template": f"office_{focal_wall.lower()}_v{variant}", "focalWall": focal_wall, "activeZone": zone, "layoutAxis": "WORK_WALL"}


def generate_generic_candidate(room: Room, products: List[Product], focal_wall: str, variant: int = 0) -> Dict[str, Any]:
    items = [item_from_product(p) for p in products]
    zone = active_zone(room, len(items))
    x1, x2, z1, z2 = zone_bounds(zone)
    wall_items = [it for it in items if it.category in WALL_CATEGORIES]
    float_items = [it for it in items if it.category not in WALL_CATEGORIES and it.category not in WALL_MOUNT_CATEGORIES]
    walls = [focal_wall, opposite_wall(focal_wall), *side_walls_for_wall(focal_wall)]
    for idx, it in enumerate(wall_items):
        place_against_wall(it, room, walls[idx % len(walls)], along=0.0)
    for idx, it in enumerate(float_items):
        cols = max(1, int(math.ceil(math.sqrt(len(float_items)))))
        row, col = divmod(idx, cols)
        it.x = x1 + (col + 0.5) * zone["widthM"] / cols
        it.z = z1 + (row + 0.5) * zone["lengthM"] / max(1, math.ceil(len(float_items) / cols))
        set_facing(it, "SOUTH")
    for art in [it for it in items if it.category in WALL_MOUNT_CATEGORIES]:
        place_wall_art(art, room, focal_wall, along=0.0)
    apply_fallback_placement(items, room, focal_wall, zone)
    return {"items": items, "template": f"generic_{focal_wall.lower()}_v{variant}", "focalWall": focal_wall, "activeZone": zone, "layoutAxis": "GENERIC"}


def generate_kitchen_candidate(room: Room, products: List[Product], focal_wall: str, variant: int = 0) -> Dict[str, Any]:
    items = [item_from_product(p) for p in products]
    zone = active_zone(room, len(items))
    cabinets = get_many(items, "cabinet")
    table = get_first(items, "dining_table", "table")
    chairs = get_many(items, "dining_chair", "chair", "stool")
    lamps = get_many(items, "lamp")
    plants = get_many(items, "plant")
    sides = get_many(items, "side_table")

    # Cabinets against focal wall and side walls
    walls = [focal_wall, *side_walls_for_wall(focal_wall)]
    for idx, cab in enumerate(cabinets[:3]):
        place_against_wall(cab, room, walls[idx % len(walls)], along=(idx - 1) * 0.8)
        cab.layoutReasoning = "Tủ bếp được đặt sát tường để tối ưu không gian sử dụng."

    if table:
        table.x, table.z = 0.0, 0.3 * (1 if variant % 2 == 0 else -1)
        set_facing(table, "SOUTH" if variant % 2 == 0 else "EAST")
        table.layoutReasoning = "Bàn ăn đặt ở trung tâm bếp để tiện di chuyển xung quanh."
    if table and chairs:
        positions = ["NORTH", "SOUTH", "WEST", "EAST"]
        for idx, ch in enumerate(chairs[:4]):
            side = positions[idx % 4]
            if side in {"NORTH", "SOUTH"}:
                ch.x = table.x
                ch.z = table.z + (-1 if side == "NORTH" else 1) * (table.depthM / 2.0 + ch.depthM / 2.0 + 0.25)
            else:
                ch.x = table.x + (-1 if side == "WEST" else 1) * (table.widthM / 2.0 + ch.depthM / 2.0 + 0.25)
                ch.z = table.z
            face_target(ch, table)
            ch.layoutReasoning = "Ghế đặt quanh bàn ăn và quay vào bàn."

    for idx, st in enumerate(sides[:1]):
        st.x = room.widthM / 2.0 - st.widthM / 2.0 - 0.15
        st.z = room.lengthM / 2.0 - st.depthM / 2.0 - 0.15
        st.layoutReasoning = "Bàn phụ đặt ở góc bếp."
    for idx, plant in enumerate(plants[:1]):
        plant.x = -room.widthM / 2.0 + 0.5
        plant.z = room.lengthM / 2.0 - 0.5
        plant.anchorWall = "CORNER"
    for idx, lamp in enumerate(lamps[:1]):
        if table:
            lamp.x, lamp.z = table.x, table.z
            lamp.y = min(room.heightM - 0.4, 2.1)
            lamp.layer = "ceiling"
            lamp.layoutReasoning = "Đèn chiếu sáng vùng bàn ăn."
    apply_fallback_placement(items, room, focal_wall, zone)
    return {"items": items, "template": f"kitchen_{focal_wall.lower()}_v{variant}", "focalWall": focal_wall, "activeZone": zone, "layoutAxis": "CENTER"}


def generate_bathroom_candidate(room: Room, products: List[Product], focal_wall: str, variant: int = 0) -> Dict[str, Any]:
    items = [item_from_product(p) for p in products]
    zone = active_zone(room, len(items))
    cabinets = get_many(items, "cabinet")
    mirror = get_first(items, "mirror")
    sides = get_many(items, "side_table")
    lamps = get_many(items, "lamp")
    plants = get_many(items, "plant")
    stools = get_many(items, "stool")

    # Cabinet against focal wall
    for idx, cab in enumerate(cabinets[:2]):
        wall = focal_wall if idx == 0 else side_walls_for_wall(focal_wall)[0]
        place_against_wall(cab, room, wall, along=0.0)
        cab.layoutReasoning = "Tủ phòng tắm sát tường để tối ưu không gian."
        cab.facingTarget = "room_center"
        cab.relations.append({"type": "against_wall", "target": wall})

    if mirror:
        wall = focal_wall
        place_wall_art(mirror, room, wall, along=0.0)
        mirror.layoutReasoning = "Gương treo trên tường phía trên tủ, đúng vị trí sử dụng."

    for idx, st in enumerate(sides[:1]):
        wall = side_walls_for_wall(focal_wall)[1]
        place_against_wall(st, room, wall, along=0.0)
        st.layoutReasoning = "Bàn phụ đặt cạnh tường để không chắn lối đi."
    for idx, stool in enumerate(stools[:1]):
        stool.x = 0.0
        stool.z = room.lengthM / 4.0
        stool.layoutReasoning = "Ghế đẩu đặt ở vùng trung tâm phòng tắm."
    for idx, plant in enumerate(plants[:1]):
        plant.x = room.widthM / 2.0 - 0.45
        plant.z = room.lengthM / 2.0 - 0.45
        plant.anchorWall = "CORNER"
    for idx, lamp in enumerate(lamps[:1]):
        if cabinets:
            lamp.x, lamp.z = cabinets[0].x, cabinets[0].z
            lamp.y = cabinets[0].heightM + lamp.heightM / 2.0
            lamp.layer = "top_surface"
            lamp.supportSurfaceId = cabinets[0].productId
    apply_fallback_placement(items, room, focal_wall, zone)
    return {"items": items, "template": f"bathroom_{focal_wall.lower()}_v{variant}", "focalWall": focal_wall, "activeZone": zone, "layoutAxis": "COMPACT"}


def generate_candidates(room: Room, products: List[Product]) -> List[Dict[str, Any]]:
    generator = {
        "living_room": generate_living_candidate,
        "bedroom": generate_bedroom_candidate,
        "dining_room": generate_dining_candidate,
        "office": generate_office_candidate,
        "kitchen": generate_kitchen_candidate,
        "bathroom": generate_bathroom_candidate,
    }.get(room.type, generate_generic_candidate)
    candidates: List[Dict[str, Any]] = []
    for focal_wall in default_focal_walls(room):
        for variant in range(3):
            cand = generator(room, products, focal_wall, variant)
            candidates.append(cand)
    return candidates


# ============================================================
# 9) Constraint repair and scoring
# ============================================================

def repair_layout(candidate: Dict[str, Any], room: Room) -> Tuple[Dict[str, Any], Dict[str, int]]:
    cand = deepcopy(candidate)
    items: List[LayoutItem] = cand["items"]
    metrics = {"clamped": 0, "collisionsResolved": 0, "gapFixed": 0}

    # Assign default y and clamp.
    for it in items:
        if it.layer == "floor":
            it.y = 0.01 if it.category == "rug" else it.heightM / 2.0
        elif it.layer == "wall":
            it.y = it.y or min(room.heightM * 0.58, 1.55)
        elif it.layer == "ceiling":
            it.y = it.y or min(room.heightM - 0.4, 2.1)
        elif it.layer == "top_surface":
            it.y = max(it.y, it.heightM / 2.0)
        metrics["clamped"] += clamp_inside_room(it, room)

    # --- Phase 1: Resolve actual overlaps (area intersection > 0) ---
    if Polygon is not None:
        for iteration in range(250):
            changed = False
            floor = [it for it in items if it.layer == "floor" and it.category != "rug"]
            for i in range(len(floor)):
                for j in range(i + 1, len(floor)):
                    a, b = floor[i], floor[j]
                    # Skip items on support surfaces of each other
                    if a.supportSurfaceId == b.productId or b.supportSurfaceId == a.productId:
                        continue
                    pa, pb = oriented_box_polygon(a), oriented_box_polygon(b)
                    inter = pa.intersection(pb).area
                    if inter <= 1e-5:
                        continue
                    metrics["collisionsResolved"] += 1
                    # Move lower priority item away from the higher priority item.
                    move = b if PRIORITY.get(b.category, 9) >= PRIORITY.get(a.category, 9) else a
                    anchor = a if move is b else b
                    dx, dz = move.x - anchor.x, move.z - anchor.z
                    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
                        # Push outwards strongly when exactly stacked to force separation
                        dx, dz = (move.widthM + anchor.widthM) / 2.0, (move.depthM + anchor.depthM) / 2.0
                    norm = max(math.hypot(dx, dz), 1e-6)
                    # Adaptive step: much larger push for larger overlaps
                    overlap_size = math.sqrt(max(inter, 0.0))
                    step = min(1.2, max(0.25, 0.2 + overlap_size * 2.0))
                    move.x += dx / norm * step
                    move.z += dz / norm * step
                    clamp_inside_room(move, room)
                    changed = True
            if not changed:
                break

        # --- Phase 2: Enforce minimum gap between all floor items ---
        for _ in range(60):
            gap_changed = False
            floor = [it for it in items if it.layer == "floor" and it.category != "rug"]
            for i in range(len(floor)):
                for j in range(i + 1, len(floor)):
                    a, b = floor[i], floor[j]
                    if a.supportSurfaceId == b.productId or b.supportSurfaceId == a.productId:
                        continue
                    pa, pb = oriented_box_polygon(a), oriented_box_polygon(b)
                    gap = pa.distance(pb)
                    if gap >= MIN_ITEM_GAP:
                        continue
                    # Push apart to achieve minimum gap
                    metrics["gapFixed"] += 1
                    move = b if PRIORITY.get(b.category, 9) >= PRIORITY.get(a.category, 9) else a
                    anchor = a if move is b else b
                    dx, dz = move.x - anchor.x, move.z - anchor.z
                    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
                        dx, dz = 1.0, 0.3
                    norm = max(math.hypot(dx, dz), 1e-6)
                    push = MIN_ITEM_GAP - gap + 0.02  # small extra to avoid re-triggering
                    move.x += dx / norm * push
                    move.z += dz / norm * push
                    clamp_inside_room(move, room)
                    gap_changed = True
            if not gap_changed:
                break

    return cand, metrics


def distance(a: LayoutItem, b: LayoutItem) -> float:
    return math.hypot(a.x - b.x, a.z - b.z)


def dot_facing_to_target(a: LayoutItem, b: LayoutItem) -> float:
    fx, fz = facing_vector(a.facingDirection)
    dx, dz = b.x - a.x, b.z - a.z
    norm = max(math.hypot(dx, dz), 1e-6)
    return (fx * dx + fz * dz) / norm


def is_between(mid: LayoutItem, a: LayoutItem, b: LayoutItem, tolerance: float = 0.55) -> bool:
    # Distance sum close to direct distance means mid is between a and b.
    return abs((distance(a, mid) + distance(mid, b)) - distance(a, b)) <= tolerance


def score_room_relations(room: Room, items: List[LayoutItem]) -> Tuple[float, Dict[str, float]]:
    breakdown: Dict[str, float] = {}
    score = 0.0
    max_score = 1e-6
    if room.type == "living_room":
        sofa = get_first(items, "sofa")
        tv = get_first(items, "tv_stand")
        table = get_first(items, "coffee_table", "table")
        chairs = get_many(items, "armchair", "chair")
        rug = get_first(items, "rug")
        if sofa and tv:
            facing = max(0.0, (dot_facing_to_target(sofa, tv) + dot_facing_to_target(tv, sofa)) / 2.0)
            dist = distance(sofa, tv)
            dist_score = 1.0 if 2.0 <= dist <= 5.2 else max(0.0, 1.0 - abs(dist - 3.4) / 4.0)
            score += 2.0 * facing + 1.0 * dist_score
            max_score += 3.0
            breakdown["sofaTvFacing"] = facing
            breakdown["sofaTvDistance"] = dist_score
        if sofa and table:
            d = distance(sofa, table)
            near = 1.0 if 0.8 <= d <= 1.9 else max(0.0, 1.0 - abs(d - 1.25) / 2.0)
            score += 1.5 * near
            max_score += 1.5
            breakdown["sofaCoffeeDistance"] = near
        if sofa and tv and table:
            between = 1.0 if is_between(table, sofa, tv, 0.8) else 0.0
            score += 1.5 * between
            max_score += 1.5
            breakdown["coffeeBetweenSofaTv"] = between
        if table and chairs:
            chair_score = sum(max(0.0, dot_facing_to_target(ch, table)) for ch in chairs) / max(1, len(chairs))
            score += 1.0 * chair_score
            max_score += 1.0
            breakdown["chairsFaceTable"] = chair_score
        if rug and (sofa or table):
            anchor = table or sofa
            rug_near = 1.0 if distance(rug, anchor) <= 0.8 else 0.3
            score += 0.8 * rug_near
            max_score += 0.8
            breakdown["rugAnchorsGroup"] = rug_near
    elif room.type == "bedroom":
        bed = get_first(items, "bed")
        nights = get_many(items, "nightstand")
        if bed:
            # bed should be against a wall
            against = 1.0 if bed.anchorWall.endswith("WALL") else 0.0
            score += 2.0 * against
            max_score += 2.0
            breakdown["bedAgainstWall"] = against
        if bed and nights:
            near = sum(1.0 if distance(bed, n) <= bed.widthM / 2 + 1.1 else 0.0 for n in nights) / len(nights)
            score += 1.2 * near
            max_score += 1.2
            breakdown["nightstandsNearBed"] = near
    elif room.type == "dining_room":
        table = get_first(items, "dining_table", "table")
        chairs = get_many(items, "dining_chair", "chair")
        if table and chairs:
            face = sum(max(0.0, dot_facing_to_target(ch, table)) for ch in chairs) / len(chairs)
            score += 2.0 * face
            max_score += 2.0
            breakdown["chairsFaceDiningTable"] = face
    elif room.type == "office":
        desk = get_first(items, "desk")
        chair = get_first(items, "office_chair", "chair")
        if desk and chair:
            face = max(0.0, dot_facing_to_target(chair, desk))
            d = distance(desk, chair)
            near = 1.0 if 0.7 <= d <= 1.4 else max(0.0, 1.0 - abs(d - 1.0) / 2.0)
            score += 1.5 * face + 1.0 * near
            max_score += 2.5
            breakdown["chairFacesDesk"] = face
            breakdown["deskChairDistance"] = near
    return min(1.0, score / max_score), breakdown


def score_graph_relations(items: List[LayoutItem]) -> Dict[str, float]:
    facing_total = 0.0
    facing_count = 0
    relation_total = 0.0
    relation_count = 0

    item_dict = {it.category: it for it in items}

    for it in items:
        # Facing Score
        if it.facingTarget and it.facingTarget != "room_center":
            target_item = item_dict.get(it.facingTarget)
            if target_item:
                facing_total += max(0.0, dot_facing_to_target(it, target_item))
                facing_count += 1
        elif it.facingTarget == "room_center":
            dx, dz = -it.x, -it.z
            norm = max(math.hypot(dx, dz), 1e-6)
            fx, fz = facing_vector(it.facingDirection)
            facing_total += max(0.0, (fx * dx + fz * dz) / norm)
            facing_count += 1

        # Relation Score
        for rel in it.relations:
            rtype = rel.get("type")
            target = rel.get("target")
            if not target: continue

            if rtype == "near":
                target_item = item_dict.get(target)
                if target_item:
                    dist = distance(it, target_item)
                    relation_total += 1.0 if dist <= 2.5 else max(0.0, 1.0 - (dist - 2.5) / 2.0)
                    relation_count += 1
            elif rtype == "under":
                target_item = item_dict.get(target)
                if target_item:
                    dist = distance(it, target_item)
                    relation_total += 1.0 if dist <= 1.5 else 0.0
                    relation_count += 1
            elif rtype == "face_to":
                target_item = item_dict.get(target)
                if target_item:
                    relation_total += max(0.0, dot_facing_to_target(it, target_item))
                    relation_count += 1
            elif rtype == "against_wall":
                relation_total += 1.0 if "WALL" in it.anchorWall else 0.0
                relation_count += 1

    return {
        "facingScore": facing_total / max(1, facing_count),
        "relationScore": relation_total / max(1, relation_count)
    }

def score_balance(candidate: Dict[str, Any], room: Room) -> float:
    items: List[LayoutItem] = [it for it in candidate["items"] if it.layer == "floor" and it.category != "rug"]
    if not items:
        return 0.0
    cx = sum(it.x for it in items) / len(items)
    cz = sum(it.z for it in items) / len(items)
    # For a planned active zone centered in room, group center should not be too far away.
    offset = math.hypot(cx, cz) / max(1.0, math.hypot(room.widthM / 2.0, room.lengthM / 2.0))
    return max(0.0, 1.0 - offset * 1.8)


def score_scale_fit(room: Room, items: List[LayoutItem]) -> float:
    area = room.widthM * room.lengthM
    furniture_area = sum(it.widthM * it.depthM for it in items if it.layer == "floor" and it.category != "rug")
    ratio = furniture_area / max(area, 1e-6)
    # Avoid both extremely empty and overfull. For large room with few items, active zone handles grouping.
    if 0.04 <= ratio <= 0.28:
        return 1.0
    if ratio < 0.04:
        return max(0.45, ratio / 0.04)
    return max(0.0, 1.0 - (ratio - 0.28) / 0.25)


def score_style(room: Room, products: List[Product]) -> float:
    if not products:
        return 0.0
    return sum(style_match_score(room.style, p.styles) for p in products) / len(products)


def score_layout(candidate: Dict[str, Any], room: Room, products: List[Product]) -> Dict[str, Any]:
    items: List[LayoutItem] = candidate["items"]
    collision = collision_area(items)
    out = inside_room_penalty(items, room)
    clear = front_clearance_penalty(items, room)
    relation, rel_breakdown = score_room_relations(room, items)
    balance = score_balance(candidate, room)
    scale = score_scale_fit(room, items)
    style = score_style(room, products)
    gap = min_pair_gap(items)
    gap_score = 1.0 if gap >= 0.05 else 0.0

    graph_scores = score_graph_relations(items)
    facingScore = graph_scores["facingScore"]
    relationScore = graph_scores["relationScore"]
    aestheticScore = (balance + scale + style) / 3.0

    # Heavily penalize collisions: a small 0.15m2 overlap will drop the score to 0
    no_collision = max(0.0, 1.0 - collision / 0.15)
    inside = max(0.0, 1.0 - out / 1.0)
    clearance = max(0.0, 1.0 - clear / 0.8)

    total = (
        3.5 * no_collision +
        2.5 * inside +
        1.4 * clearance +
        1.0 * relation +
        1.0 * relationScore +
        0.5 * facingScore +
        1.0 * balance +
        0.8 * scale +
        0.5 * style +
        0.6 * gap_score
    )
    max_total = 12.8
    return {
        "total": round(total / max_total * 100.0, 4),
        "breakdown": {
            "noCollision": round(no_collision, 4),
            "insideRoom": round(inside, 4),
            "clearance": round(clearance, 4),
            "roomRelations": round(relation, 4),
            "balance": round(balance, 4),
            "scaleFit": round(scale, 4),
            "style": round(style, 4),
            "gap": round(gap_score, 4),
            "facingScore": round(facingScore, 4),
            "relationScore": round(relationScore, 4),
            "aestheticScore": round(aestheticScore, 4),
            **{k: round(v, 4) for k, v in rel_breakdown.items()},
        },
        "rawPenalties": {
            "collisionArea": round(collision, 6),
            "outOfRoomArea": round(out, 6),
            "frontClearanceOverlapArea": round(clear, 6),
            "minPairGapM": round(gap, 4),
        },
    }


# ============================================================
# 10) Engine API
# ============================================================

def compute_active_zone_from_items(items: List[LayoutItem], room: Room, padding: float = 0.65) -> Dict[str, float]:
    floor = [it for it in items if it.layer == "floor" and it.category != "rug"]
    if not floor:
        return active_zone(room, 0)
    minx, maxx = 999.0, -999.0
    minz, maxz = 999.0, -999.0
    for it in floor:
        poly = oriented_box_polygon(it)
        bx1, bx2, bz1, bz2 = polygon_bounds(poly)
        minx, maxx = min(minx, bx1), max(maxx, bx2)
        minz, maxz = min(minz, bz1), max(maxz, bz2)
    minx = max(-room.widthM / 2.0, minx - padding)
    maxx = min(room.widthM / 2.0, maxx + padding)
    minz = max(-room.lengthM / 2.0, minz - padding)
    maxz = min(room.lengthM / 2.0, maxz + padding)
    return {
        "x": round(minx, 4),
        "z": round(minz, 4),
        "widthM": round(max(0.1, maxx - minx), 4),
        "lengthM": round(max(0.1, maxz - minz), 4),
        "centerX": round((minx + maxx) / 2.0, 4),
        "centerZ": round((minz + maxz) / 2.0, 4),
    }


def build_warnings(room: Room, selected: List[Product], missing: List[str]) -> List[str]:
    warnings = []
    if missing:
        warnings.append("Thiếu nhóm sản phẩm quan trọng cho %s: %s" % (room.type, ", ".join(missing)))
    if room.widthM * room.lengthM > 45 and len(selected) <= 7:
        warnings.append("Phòng lớn nhưng số sản phẩm ít, engine đã tạo activeZone nhỏ hơn để tránh bố cục bị rời rạc.")
    if Polygon is None:
        warnings.append("Shapely chưa khả dụng, collision/clearance sẽ yếu hơn. Hãy cài shapely trong requirements.")
    return warnings


def finalize_layout(payload: Dict[str, Any]) -> Dict[str, Any]:
    import json
    try:
        with open("last_engine_payload.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    room, products = normalize_payload(payload)
    selected, rejected, missing = select_products(room, products, payload)

    if not selected:
        return {
            "success": False,
            "room": asdict(room),
            "items": [],
            "layout": {"items": []},
            "rejected": rejected,
            "warnings": ["Không có sản phẩm hợp lệ để layout."],
            "metrics": {"engine": "rule_template_shapely_scoring_v2", "candidateCount": 0},
        }

    # Generate template-based candidates for this room type
    candidates = generate_candidates(room, selected)

    # Also generate a candidate from the trained LayoutTransformer if available
    model_pred_available = False
    predictor = _get_model_predictor()
    if predictor.get("available"):
        pass

    # Score all candidates (templates + model prediction)
    scored_candidates = []
    total_repairs = {"clamped": 0, "collisionsResolved": 0, "gapFixed": 0}
    for cand in candidates:
        repaired, repair_metrics = repair_layout(cand, room)
        s = score_layout(repaired, room, selected)
        repaired["score"] = s["total"]
        repaired["scoreBreakdown"] = s["breakdown"]
        repaired["rawPenalties"] = s["rawPenalties"]
        scored_candidates.append(repaired)
        for k, v in repair_metrics.items():
            total_repairs[k] = total_repairs.get(k, 0) + v

    best = max(scored_candidates, key=lambda c: c.get("score", 0.0))

    items: List[LayoutItem] = best["items"]

    # Final safety clamp after choosing the best candidate.
    for it in items:
        clamp_inside_room(it, room)

    output_items = [it.output() for it in items]
    layout = {
        "layoutAxis": best.get("layoutAxis"),
        "focalWall": best.get("focalWall"),
        "activeZone": compute_active_zone_from_items(items, room),
        "template": best.get("template"),
        "score": best.get("score"),
        "scoreBreakdown": best.get("scoreBreakdown"),
        "rawPenalties": best.get("rawPenalties"),
        "items": output_items,
        "warnings": build_warnings(room, selected, missing),
        "rejected": rejected,
    }

    return {
        "success": True,
        "room": asdict(room),
        "items": output_items,
        "layout": layout,
        "rejected": rejected,
        "warnings": layout["warnings"],
        "metrics": {
            "engine": "rule_template_shapely_scoring_v2",
            "candidateCount": len(candidates),
            "selectedCount": len(output_items),
            "rejectedCount": len(rejected),
            "missingCategories": missing,
            "repairs": total_repairs,
            "shapelyAvailable": Polygon is not None,
            "modelPredictionAvailable": model_pred_available,
            "chosenTemplate": best.get("template", "unknown"),
        },
    }

