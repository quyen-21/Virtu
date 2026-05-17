from typing import Any, Dict
import traceback

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Importing living_room_semantic_patch installs:
# - base engine patches
# - quality fixes for aliases, product units, ceiling lamps, scoring caps
# - living room semantic role mapping for console/side/storage/shelf products
from layout_engine.living_room_semantic_patch import finalize_layout, model_patch_status

app = FastAPI(title="VirtuSpace AI Layout Service", version="2.3.0")


@app.get("/health")
def health():
    try:
        from inference import model_info
        info = model_info()
    except Exception as exc:
        info = {"error": "model not loaded", "detail": str(exc)}
    return {
        "ok": True,
        "service": "layout_only",
        "version": "2.3.0",
        "availableEndpoints": [
            "POST /api/ai/layout/generate",
            "POST /api/ai/layout/generate-debug",
            "POST /api/ai/layout/generate-from-recommendation",
        ],
        "removedEndpoints": ["POST /api/v1/recommend"],
        "model": info,
        "patch": model_patch_status(),
    }


@app.post("/api/ai/layout/generate")
def generate_layout(payload: Dict[str, Any]):
    """Generate a 3D layout from an AI recommendation JSON payload."""
    return finalize_layout(payload)


@app.post("/api/ai/layout/generate-debug")
def generate_layout_debug(payload: Dict[str, Any]):
    """Debug endpoint that returns traceback details when layout generation fails."""
    try:
        return finalize_layout(payload)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


@app.post("/api/ai/layout/generate-from-recommendation")
def generate_layout_from_recommendation(payload: Dict[str, Any]):
    """Alias kept for Spring BE when it forwards the full recommendation JSON."""
    return finalize_layout(payload)
