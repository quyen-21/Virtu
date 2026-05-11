from fastapi import FastAPI
from typing import Dict, Any
from inference import finalize_layout

app = FastAPI(title="VirtuSpace AI Layout Service", version="1.0.0")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/ai/layout/generate")
def generate_layout(payload: Dict[str, Any]):
    return finalize_layout(payload)
