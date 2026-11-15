"""SafeLanding AI — backend API."""
import json
import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
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
    """Analyze an image with the neural-network detector."""
    if USE_FALLBACK:
        return load_fallback("vision_sample")
    from modules.vision import predict  # Load PyTorch only for vision requests.

    data = await file.read()
    return predict(data)


@app.get("/api/sites/{site_id}/sun-earth", response_model=SiteReport)
def sun_earth(site_id: str, start: str = "2026-11-14", end: str = "2026-11-21"):
    """Return Sun and Earth positions and visibility windows for Site A or B."""
    if USE_FALLBACK:
        return load_fallback(f"sun_earth_{site_id}")
    from modules.astro import compute_series

    return compute_series(site_id, start, end)


@app.get("/api/compare")
def compare(a: str = "site_a", b: str = "site_b"):
    """Compare two sites over the same period."""
    from modules.astro import compare_sites

    return compare_sites(a, b)


@app.get("/api/planner/sites")
def planner_sites():
    """List catalogued lunar locations and coordinate sources."""
    from modules.planner import SITES

    return [asdict(site) for site in SITES.values()]


@app.get("/api/planner/visibility")
def planner_visibility(
    start: str, end: str, site_id: str | None = None,
    lat: float | None = None, lon: float | None = None,
    step_minutes: int = 30, horizon: str = "auto",
):
    """Plan from UTC timestamps using a catalog ID or east-positive coordinates."""
    from modules.lunar_visibility import parse_utc
    from modules.planner import SITES, Site, plan_visibility

    if site_id is not None:
        if lat is not None or lon is not None:
            raise HTTPException(422, "Choose a site ID or supply both coordinates.")
        if site_id not in SITES:
            raise HTTPException(404, f"Unknown lunar site: {site_id}")
        site = SITES[site_id]
    elif lat is not None and lon is not None:
        site = Site("custom", "Custom coordinates", lat, lon, "Custom")
    else:
        raise HTTPException(422, "Choose a site ID or supply both coordinates.")
    try:
        return plan_visibility(site, parse_utc(start), parse_utc(end), step_minutes, horizon)
    except (ValueError, OverflowError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
