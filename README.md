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
  -> template candidates
  -> trained LayoutTransformer candidate
  -> Shapely collision/clearance repair
  -> scoring/ranking
  -> best layout response
```

Optional layout constraints can be sent through `constraints`, for example `doors`, `windows`, `walkways`, `reservedZones`, or `noPlaceZones`.
