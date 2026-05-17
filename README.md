# VirtuSpace Hybrid Layout AI V2

This FastAPI service is **layout-only**.

It receives a full recommendation JSON from the external AI recommendation service and returns 3D furniture placement data. Product recommendation must be handled by the separate recommender service.

## Run

```bash
pip install -r requirements.txt
bash start.sh
```

## Endpoints

### Health check

```http
GET /health
```

Expected important fields after the small-room capacity patch:

```json
{
  "service": "layout_only",
  "version": "2.7.0",
  "removedEndpoints": ["POST /api/v1/recommend"],
  "patch": {
    "patchInstalled": true,
    "qualityPatchInstalled": true,
    "livingRoomSemanticPatchInstalled": true,
    "livingRoomLayoutRefinePatchInstalled": true,
    "bedroomLayoutRefinePatchInstalled": true,
    "bedroomVariantsPatchInstalled": true,
    "smallRoomCapacityPatchInstalled": true,
    "smallRoomCapacityRule": "area_under_10m2_density_safe_cap_v1",
    "dimensionNormalization": "product_cm_mm_to_m_v2",
    "categoryAliasPatch": "vi_furniture_aliases_v2",
    "semanticRoleMapping": "console_side_storage_to_coffee_table_tv_stand_v1",
    "livingRoomRolePlacement": "role_specific_focal_wall_seating_group_v1",
    "bedroomRolePlacement": "bed_wall_nightstand_rug_bench_storage_v1",
    "bedroomVariantGeneration": "multi_wall_zone_lamp_variants_v1",
    "scoreQualityCaps": true
  }
}
```

### Generate 3D layout

```http
POST /api/ai/layout/generate
Content-Type: application/json
```

### Debug layout generation

```http
POST /api/ai/layout/generate-debug
Content-Type: application/json
```

### Alias for backend integration

```http
POST /api/ai/layout/generate-from-recommendation
Content-Type: application/json
```

## Removed endpoint

```http
POST /api/v1/recommend
```

This endpoint was removed from FastAPI because `/api/v1/recommend` belongs to the external AI recommendation service, not the layout service.

## Expected payload shape

```json
{
  "room": {
    "type": "living_room",
    "style": "modern",
    "widthM": 5,
    "lengthM": 7,
    "heightM": 3
  },
  "recommendation": {
    "analysis": {},
    "products": []
  },
  "furnitureDensity": "medium",
  "topK": 8
}
```

## Frontend render fields

FE should render `response.items` or `response.layout.items` using:

```text
modelUrl
position.x
position.y
position.z
rotationY
footprint.widthM
footprint.depthM
footprint.heightM
```

## Layout engine strategy

The service uses a hybrid pipeline:

```text
recommendation JSON
  -> normalize room/products
  -> Vietnamese category alias patch
  -> living-room semantic role mapping
  -> product cm/mm/m dimension normalization
  -> room-aware product selection
  -> small-room capacity cap when area < 10m²
  -> template candidates
  -> trained LayoutTransformer candidate when model is available
  -> living-room role-specific placement refinement
  -> bedroom role-specific placement refinement
  -> bedroom multi-variant generation
  -> Shapely collision/clearance repair
  -> scoring/ranking with quality caps and bedroom zone score
  -> best layout response
```

## Important fixes in v2.2.0

- Converts product dimensions from cm/mm to meters more safely, including small values like `height: 12` cm.
- Adds aliases for `nệm`, `đèn trần`, `kệ lưu trữ`, `tủ lưu trữ`, `bàn console`, `sofa góc`, and related Vietnamese names.
- Prevents mattress products from being selected as a separate visible bed when a real bed is already selected.
- Places `ceiling_lamp` on the ceiling instead of on nightstands.
- Caps fake-high scores when essential living room items are missing or when a large/dense bedroom is too sparse.
- Creates a secondary zone for large bedrooms, such as reading/storage/dressing zones.

## Important fixes in v2.3.0

- Adds underscore aliases such as `bàn_console`, `kệ_phòng_khách`, `tủ_lưu_trữ`, `hộc_kéo`, `sofa_góc`.
- Maps living-room semantic roles when recommender returns related but non-exact categories:
  - `Bàn bên` / suitable side table -> `coffee_table` when no real coffee table exists.
  - `Bàn console` / low cabinet / living-room shelf -> `tv_stand` when no real TV stand exists.
  - `Kệ phòng khách` / `Tủ trưng bày` -> `bookshelf` or `tv_stand` depending on size.
  - `Tủ lưu trữ` / `Hộc kéo` -> `cabinet` or fallback `tv_stand` only when low and long enough.
- Removes confusing rejected entries for products that were re-used by semantic role mapping.

## Important fixes in v2.4.0

- Refines living-room placement by role after semantic mapping.
- Keeps `tv_stand` / console / low media cabinet against the focal wall.
- Places sofa opposite the focal wall and centers it on the main viewing axis.
- Places `coffee_table` between sofa and focal wall; tall/long console is no longer allowed to behave like a coffee table.
- Places armchairs diagonally around the coffee table to create a conversational seating group.
- Moves side tables to sofa ends instead of letting them occupy the center.
- Moves bookshelves/display shelves to side walls.
- Moves small cabinets/drawers to wall/corner zones instead of leaving them floating alone.

## Important fixes in v2.5.0

- Adds bedroom role-specific placement refinement.
- Anchors bed to the main wall and centers it as the bedroom focal object.
- Places rug under the lower two-thirds of the bed.
- Places nightstands symmetrically beside the bed head area.
- Converts bench-like chair/stool products into a `bench` role and places them at the foot of the bed.
- Moves mirror to a side wall as a dressing zone, away from the lamp/headboard axis.
- Centers ceiling lamps over the bed/room zone, avoiding the previous low/gương-overlap look.
- Moves wardrobe/cabinet/bookshelf products to side wall or corner storage zones.
- Places desk/vanity/loose chair as a secondary bedroom zone instead of leaving it floating.

## Important fixes in v2.6.0

- Adds bedroom multi-variant generation instead of forcing one fixed bedroom template.
- Generates variants across different bed headboard walls: back, front, left, right when dimensions allow.
- Generates centered and slightly offset bed positions for larger rooms.
- Varies dressing/storage/reading zones so bedrooms do not all look identical.
- Varies ceiling lamp modes: over bed, foot of bed, or room center.
- Keeps original/model candidates as fallback, then lets Shapely repair and scoring choose the best layout.
- Adds bedroom scoring signals for zone quality and avoids rewarding very empty large bedrooms.

## Important fixes in v2.7.0

- Adds small-room capacity cap when room area is under `10m²`.
- If area is under `6m²`, keeps only 3–4 products depending on density.
- If area is from `6m²` to under `10m²`, keeps about 4–6 products depending on density.
- For `medium` or `dense`, the engine no longer blindly keeps many items; it caps total products and floor-heavy products separately.
- Prioritizes essential products first, such as bed/sofa/desk/rug/mirror/light, depending on room type.
- Rejects excess products with reasons like `small_room_area_total_capacity_exceeded` or `small_room_floor_capacity_exceeded`.
- Adds metrics: `smallRoomCapacityPatchInstalled`, `smallRoomCapacityApplied`, and `smallRoomAreaM2`.

Optional layout constraints can be sent through `constraints`, for example `doors`, `windows`, `walkways`, `reservedZones`, or `noPlaceZones`.
