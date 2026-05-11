from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .runtime_v2 import HybridInteriorRuntimeV2

app = FastAPI(title='VirtuSpace Hybrid Layout AI V2', version='2.5.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

runtime = HybridInteriorRuntimeV2()

class RoomIn(BaseModel):
    widthM: float = 4.0
    lengthM: float = 5.0
    heightM: float = 2.8
    type: str = 'living_room'
    style: str = 'modern'

class ProductIn(BaseModel):
    productId: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    category: str = 'unknown'
    score: Optional[float] = 0.5
    widthM: Optional[float] = 0.6
    depthM: Optional[float] = 0.6
    heightM: Optional[float] = 0.6
    modelUrl: Optional[str] = None
    massKg: Optional[float] = None
    maxLoadKg: Optional[float] = None

class GenerateRequest(BaseModel):
    room: RoomIn
    products: List[ProductIn] = []
    options: Dict[str, Any] = {}

@app.get('/health')
def health():
    return {'ok': True, 'service': 'VirtuSpace Hybrid Layout AI V2'}

@app.post('/api/ai/layout/generate')
def generate_layout(req: GenerateRequest):
    return runtime.generate_layout(req.model_dump())
