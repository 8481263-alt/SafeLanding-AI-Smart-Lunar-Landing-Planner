"""SafeLanding AI — backend API."""
import json
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from schemas import SiteReport, VisionResult

ROOT = Path(__file__).resolve().parent
USE_FALLBACK = os.getenv("USE_FALLBACK", "0") == "1"

app = FastAPI(title="SafeLanding AI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_fallback(name: str) -> dict:
    with open(ROOT / "fallback" / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


@app.get("/")
def root():
    return {"name": "SafeLanding AI", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "fallback": USE_FALLBACK}


@app.post("/api/vision/analyze", response_model=VisionResult)
async def analyze(file: UploadFile = File(...)):
    """Аналіз зображення нейромережею (Людина 1)."""
    if USE_FALLBACK:
        return load_fallback("vision_sample")
    from modules.vision import predict  # імпортується лише тут, бо torch важкий

    data = await file.read()
    return predict(data)


@app.get("/api/sites/{site_id}/sun-earth", response_model=SiteReport)
def sun_earth(site_id: str, start: str = "2026-11-14", end: str = "2026-11-21"):
    """Положення Сонця й Землі та вікна видимості (Людина 3). site_id: site_a | site_b."""
    if USE_FALLBACK:
        return load_fallback(f"sun_earth_{site_id}")
    from modules.astro import compute_series

    return compute_series(site_id, start, end)


@app.get("/api/compare")
def compare(a: str = "site_a", b: str = "site_b"):
    """Порівняння двох ділянок за однаковий період."""
    from modules.astro import compare_sites

    return compare_sites(a, b)