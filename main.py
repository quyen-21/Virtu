from typing import Any, Dict
import traceback

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from layout_engine.runtime import (
    AVAILABLE_ENDPOINTS,
    REMOVED_ENDPOINTS,
    SERVICE_ROLE,
    SERVICE_VERSION,
    finalize_layout,
    model_patch_status,
)

app = FastAPI(title="VirtuSpace AI Layout Service", version=SERVICE_VERSION)


def _safe_model_info() -> Dict[str, Any]:
    try:
        from inference import model_info
        return model_info()
    except Exception as exc:
        return {"error": "model not loaded", "detail": str(exc)}


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": SERVICE_ROLE,
        "version": SERVICE_VERSION,
        "availableEndpoints": AVAILABLE_ENDPOINTS,
        "removedEndpoints": REMOVED_ENDPOINTS,
        "model": _safe_model_info(),
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
