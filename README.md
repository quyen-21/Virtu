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

Expected important fields after the living-room refinement patch:

```json
{
  "service": "layout_only",
  "version": "2.4.0",
  "removedEndpoints": ["POST /api/v1/recommend"],
  "patch": {
    "patchInstalled": true,
    "qualityPatchInstalled": true,
    "livingRoomSemanticPatchInstalled": true,
    "livingRoomLayoutRefinePatchInstalled": true,
    "dimensionNormalization": "product_cm_mm_to_m_v2",
    "categoryAliasPatch": "vi_furniture_aliases_v2",
    "semanticRoleMapping": "console_side_storage_to_coffee_table_tv_stand_v1",
    "livingRoomRolePlacement": "role_specific_focal_wall_seating_group_v1",
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
  -> template candidates
  -> trained LayoutTransformer candidate when model is available
  -> living-room role-specific placement refinement
  -> ceiling lamp / secondary-zone postprocess
  -> Shapely collision/clearance repair
  -> scoring/ranking with quality caps
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

Optional layout constraints can be sent through `constraints`, for example `doors`, `windows`, `walkways`, `reservedZones`, or `noPlaceZones`.
