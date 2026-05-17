from __future__ import annotations

"""
Clean runtime entrypoint for the VirtuSpace layout service.

This module is intentionally small and stable. It hides the internal patch stack
from FastAPI so `main.py` does not need to know which patch file is currently the
last one in the chain.

Business behavior is not changed here. Importing
`category_balanced_selection_patch` still installs the full existing patch stack
in the same order as before:

1. engine_patch
2. layout_quality_patch
3. living_room_semantic_patch
4. living_room_layout_refine_patch
5. bedroom_layout_refine_patch
6. bedroom_variants_patch
7. small_room_capacity_patch
8. category_density_quota_patch
9. category_balanced_selection_patch

Public API exported to FastAPI:
- finalize_layout(payload)
- model_patch_status()
- PATCH_STACK
- SERVICE_VERSION
"""

from typing import Any, Dict, List

from layout_engine.category_balanced_selection_patch import (
    finalize_layout as _finalize_layout,
    model_patch_status as _model_patch_status,
)

SERVICE_VERSION = "2.9.0"
SERVICE_ROLE = "layout_only"

PATCH_STACK: List[str] = [
    "engine_patch",
    "layout_quality_patch",
    "living_room_semantic_patch",
    "living_room_layout_refine_patch",
    "bedroom_layout_refine_patch",
    "bedroom_variants_patch",
    "small_room_capacity_patch",
    "category_density_quota_patch",
    "category_balanced_selection_patch",
]

AVAILABLE_ENDPOINTS: List[str] = [
    "POST /api/ai/layout/generate",
    "POST /api/ai/layout/generate-debug",
    "POST /api/ai/layout/generate-from-recommendation",
]

REMOVED_ENDPOINTS: List[str] = [
    "POST /api/v1/recommend",
]


def finalize_layout(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate final 3D layout using the installed patch stack."""
    return _finalize_layout(payload)


def model_patch_status() -> Dict[str, Any]:
    """Return model and runtime patch status for health/debug endpoints."""
    status = _model_patch_status()
    status["runtimeEntrypoint"] = "layout_engine.runtime"
    status["patchStack"] = PATCH_STACK
    return status
