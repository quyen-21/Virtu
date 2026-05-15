from fastapi import FastAPI, Form, UploadFile, File
from typing import Dict, Any, Optional
import json

from fastapi.responses import JSONResponse
import traceback

from layout_engine.engine import finalize_layout

app = FastAPI(title="VirtuSpace AI Layout Service", version="2.0.0")


@app.get("/health")
def health():
    try:
        from inference import model_info
        info = model_info()
    except Exception:
        info = {"error": "model not loaded"}
    return {"ok": True, "model": info}


@app.post("/api/ai/layout/generate")
def generate_layout(payload: Dict[str, Any]):
    return finalize_layout(payload)

@app.post("/api/ai/layout/generate-debug")
def generate_layout_debug(payload: Dict[str, Any]):
    try:
        return finalize_layout(payload)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

@app.post("/api/ai/layout/generate-from-recommendation")
def generate_layout_from_recommendation(payload: Dict[str, Any]):
    return finalize_layout(payload)


@app.post("/api/v1/recommend")
async def recommend_compat(
    room_type: str = Form("living_room"),
    style: str = Form(""),
    width: float = Form(4.0),
    length: float = Form(5.0),
    height: float = Form(3.0),
    furniture_density: str = Form("medium"),
    gender: str = Form(""),
    age: int = Form(25),
    user_id: str = Form(""),
    products_json: str = Form("[]"),
    image: Optional[UploadFile] = File(None),
):
    """
    Compatibility endpoint for current Spring Boot backend.
    Backend calls ai.api.url using multipart/form-data.
    """

    try:
        products = json.loads(products_json) if products_json else []
    except Exception:
        products = []

    # Dump for debugging
    try:
        with open("last_frontend_request.json", "w", encoding="utf-8") as f:
            json.dump({"room": room_type, "width": width, "length": length, "products": products}, f, ensure_ascii=False, indent=2)
    except:
        pass

    payload = {
        "room": {
            "type": room_type,
            "room_type": room_type,
            "style": style,
            "widthM": width,
            "lengthM": length,
            "heightM": height,
            "width_m": width,
            "length_m": length,
            "height_m": height,
        },
        "recommendation": {
            "products": products
        },
        "topK": 8,
        "minScore": 0.50,
        "furnitureDensity": furniture_density,
    }

    layout_result = finalize_layout(payload)

    items = (
        layout_result.get("items")
        or layout_result.get("layout", {}).get("items")
        or []
    )

    product_map = {
        str(p.get("id") or p.get("product_id") or p.get("productId")): p
        for p in products
        if isinstance(p, dict)
    }

    response_products = []

    for index, item in enumerate(items):
        item_id = str(
            item.get("productId")
            or item.get("id")
            or item.get("product_id")
            or f"ai-product-{index + 1}"
        )

        src = product_map.get(item_id, {})

        dimensions = src.get("dimensions") or item.get("dimensions") or {}
        footprint = item.get("footprint") or {}

        response_products.append({
            "id": item_id,
            "name": src.get("name") or item.get("name") or item.get("category") or "AI Product",
            "category": src.get("category") or item.get("category") or "Furniture",
            "styles": src.get("styles") or ([style] if style else []),
            "price": src.get("price"),
            "dimensions": {
                "width": dimensions.get("width") or footprint.get("widthM") or dimensions.get("width_m"),
                "depth": dimensions.get("depth") or footprint.get("depthM") or dimensions.get("depth_m"),
                "height": dimensions.get("height") or footprint.get("heightM") or dimensions.get("height_m"),
            },
            "colors": src.get("colors") or [],
            "imageUrl": src.get("imageUrl") or src.get("image_url") or item.get("imageUrl") or "",
            "facingTarget": item.get("facingTarget") or "",
            "relations": item.get("relations") or [],
            "reasoning": item.get("layoutReasoning") or item.get("reasoning") or "AI selected this product for the room layout.",
        })

    # Extract detailed metrics
    score_breakdown = layout_result.get("layout", {}).get("scoreBreakdown", {})
    metrics = {
        "layoutScore": layout_result.get("layout", {}).get("score", 0.0) / 100.0,
        "collisionCount": layout_result.get("metrics", {}).get("repairs", {}).get("collisionsResolved", 0),
        "relationScore": score_breakdown.get("relationScore", 0.0),
        "facingScore": score_breakdown.get("facingScore", 0.0),
        "clearanceScore": score_breakdown.get("clearance", 0.0),
        "aestheticScore": score_breakdown.get("aestheticScore", 0.0),
    }

    return {
        "analysis": {
            "reasoning": "AI analyzed room dimensions, style, product compatibility, and layout constraints.",
            "imageAnalysis": {
                "dominantColors": [],
                "colorTone": style,
                "detectedStyle": style,
                "lightingType": "",
                "existingFurnitureCategories": [],
            },
        },
        "metrics": metrics,
        "products": response_products,
        "layout": layout_result,
    }