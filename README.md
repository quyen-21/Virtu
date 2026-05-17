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

Expected important fields after the quality patch:

```json
{
  "service": "layout_only",
  "version": "2.2.0",
  "removedEndpoints": ["POST /api/v1/recommend"],
  "patch": {
    "patchInstalled": true,
    "qualityPatchInstalled": true,
    "dimensionNormalization": "product_cm_mm_to_m_v2",
    "categoryAliasPatch": "vi_furniture_aliases_v2",
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
  -> room-aware product selection
  -> Vietnamese category alias patch
  -> product cm/mm/m dimension normalization
  -> template candidates
  -> trained LayoutTransformer candidate when model is available
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

Optional layout constraints can be sent through `constraints`, for example `doors`, `windows`, `walkways`, `reservedZones`, or `noPlaceZones`.
