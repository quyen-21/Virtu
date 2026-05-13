from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Dict, Any
import traceback
import json
import hashlib
import logging

from inference import (
    finalize_layout,
    select_products_for_layout,
    canonical_room_type,
)

APP_VERSION = "layout-guard-2026-05-13-v3"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("virtu-layout-service")

app = FastAPI(
    title="VirtuSpace AI Layout Service",
    version=APP_VERSION,
    description="AI layout service only. This service receives recommended products and returns 3D layout positions."
)


def payload_hash(payload: Dict[str, Any]) -> str:
    """
    Tạo hash để kiểm tra 2 lần test có thật sự cùng payload hay không.
    Nếu payload giống 100%, hash phải giống nhau.
    """
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def get_products(payload: Dict[str, Any]):
    return payload.get("recommendation", {}).get("products", [])


def validate_layout_payload(payload: Dict[str, Any]):
    """
    Validate nhẹ để tránh lỗi khó hiểu khi thiếu room hoặc products.
    """
    if not isinstance(payload, dict):
        return "Payload must be a JSON object."

    if "room" not in payload:
        return "Missing required field: room."

    room = payload.get("room") or {}

    required_room_fields = ["widthM", "lengthM", "heightM"]
    missing = [field for field in required_room_fields if field not in room]

    if missing:
        return f"Missing required room field(s): {', '.join(missing)}."

    if "recommendation" not in payload:
        return "Missing required field: recommendation."

    products = get_products(payload)

    if not isinstance(products, list):
        return "recommendation.products must be a list."

    return None


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "VirtuSpace AI Layout Service",
        "version": APP_VERSION,
        "message": "Use POST /api/ai/layout/generate for layout generation."
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "VirtuSpace AI Layout Service",
        "version": APP_VERSION,
        "endpoints": {
            "layout": "/api/ai/layout/generate",
            "layoutDebug": "/api/ai/layout/generate-debug",
            "selectionDebug": "/api/ai/layout/debug-selection"
        }
    }


@app.post("/api/ai/layout/generate")
def generate_layout(payload: Dict[str, Any]):
    """
    Endpoint chính cho Spring BE gọi.

    Input đúng:
    {
      "room": {...},
      "recommendation": {
        "products": [...]
      },
      "topK": 8,
      "minScore": 0.2
    }

    Output:
    {
      "room": {...},
      "items": [
        {
          "productId": "...",
          "position": {"x": ..., "y": ..., "z": ...},
          "rotationY": ...
        }
      ],
      "rejected": [...],
      "metrics": {...}
    }
    """
    err = validate_layout_payload(payload)
    if err:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": err,
                "payloadHash": payload_hash(payload) if isinstance(payload, dict) else None
            }
        )

    hash_value = payload_hash(payload)
    products = get_products(payload)

    logger.info("POST /api/ai/layout/generate")
    logger.info("payloadHash=%s", hash_value)
    logger.info("productCount=%s", len(products))

    try:
        result = finalize_layout(payload)

        if isinstance(result, dict):
            metrics = result.setdefault("metrics", {})
            metrics["payloadHash"] = hash_value
            metrics["inputProductCount"] = len(products)
            metrics["serviceVersion"] = APP_VERSION

        return result

    except Exception as e:
        logger.exception("Layout generation failed. payloadHash=%s", hash_value)

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "payloadHash": hash_value,
                "productCount": len(products),
                "message": "Layout generation failed. Use /api/ai/layout/generate-debug for traceback."
            }
        )


@app.post("/api/ai/layout/generate-debug")
def generate_layout_debug(payload: Dict[str, Any]):
    """
    Endpoint debug. Dùng khi muốn xem traceback đầy đủ.
    Không nên để FE gọi endpoint này trong production.
    """
    hash_value = payload_hash(payload) if isinstance(payload, dict) else None
    products = get_products(payload) if isinstance(payload, dict) else []

    try:
        err = validate_layout_payload(payload)
        if err:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": err,
                    "payloadHash": hash_value,
                    "productCount": len(products),
                    "serviceVersion": APP_VERSION
                }
            )

        result = finalize_layout(payload)

        if isinstance(result, dict):
            metrics = result.setdefault("metrics", {})
            metrics["payloadHash"] = hash_value
            metrics["inputProductCount"] = len(products)
            metrics["serviceVersion"] = APP_VERSION

        return result

    except Exception as e:
        logger.exception("Debug layout generation failed. payloadHash=%s", hash_value)

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "payloadHash": hash_value,
                "productCount": len(products),
                "serviceVersion": APP_VERSION
            }
        )


@app.post("/api/ai/layout/generate-from-recommendation")
def generate_layout_from_recommendation(payload: Dict[str, Any]):
    """
    Alias endpoint nếu bạn vẫn còn test bằng tên cũ.
    Logic giống /api/ai/layout/generate.
    """
    return generate_layout(payload)


@app.post("/api/ai/layout/debug-selection")
def debug_selection(payload: Dict[str, Any]):
    """
    Debug bước filter trước khi gọi model Transformer.

    Dùng endpoint này để biết:
    - Payload có bao nhiêu product.
    - Bao nhiêu product được selected.
    - Bao nhiêu product bị rejected.
    - Lý do reject là gì.

    Nếu selectedCount = 0 thì layout sẽ không có items.
    """
    err = validate_layout_payload(payload)
    if err:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": err,
                "payloadHash": payload_hash(payload) if isinstance(payload, dict) else None,
                "serviceVersion": APP_VERSION
            }
        )

    hash_value = payload_hash(payload)
    products = get_products(payload)

    try:
        top_k = int(payload.get("topK", 8))
        threshold = float(payload.get("minScore", 0.20))

        selected, rejected = select_products_for_layout(
            payload=payload,
            top_k=top_k,
            threshold=threshold
        )

        room = payload.get("room", {}) or {}
        room_type = canonical_room_type(room.get("type", "unknown"))

        selected_view = []
        for item in selected:
            selected_view.append({
                "productId": item.get("product_id"),
                "name": item.get("name"),
                "category": item.get("category"),
                "rawCategory": item.get("raw_category"),
                "keepProbability": float(item.get("keep_probability", 0.0)),
                "finalScore": float(item.get("final_score", 0.0)),
                "rankingScore": float(item.get("ranking_score", 0.0)),
                "styleScore": float(item.get("style_score", 0.0)),
                "colorScore": float(item.get("color_score", 0.0))
            })

        return {
            "success": True,
            "serviceVersion": APP_VERSION,
            "payloadHash": hash_value,
            "roomType": room_type,
            "topK": top_k,
            "threshold": threshold,
            "productCount": len(products),
            "selectedCount": len(selected),
            "rejectedCount": len(rejected),
            "selected": selected_view,
            "rejected": rejected
        }

    except Exception as e:
        logger.exception("Debug selection failed. payloadHash=%s", hash_value)

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "payloadHash": hash_value,
                "productCount": len(products),
                "serviceVersion": APP_VERSION
            }
        )