"""Єдиний контракт даних між модулями (довідкові схеми для документації API)."""
from typing import List, Optional

from pydantic import BaseModel


class Hazard(BaseModel):
    bbox: List[float]          # [x1, y1, x2, y2] у пікселях оригінального зображення
    cls: str                   # "hazard" (поточна модель) або crater | rock | shadow
    confidence: float          # 0..1


class VisionResult(BaseModel):
    hazards: List[Hazard]
    image_width: int
    image_height: int
    inference_ms: float
    device: str                # cpu | npu


class SunEarthPoint(BaseModel):
    time_utc: str
    sun_elevation_deg: float
    sun_azimuth_deg: float
    earth_elevation_deg: float
    earth_azimuth_deg: float
    sun_visible: bool
    earth_visible: bool


class SiteReport(BaseModel):
    site_id: str
    lat: float
    lon: float
    series: List[SunEarthPoint]
    illuminated_hours: float
    earth_visible_hours: float
    simultaneous_hours: float
    horizon_assumption: str
    caveats: List[str]


class LandingRecommendation(BaseModel):
    point_xy: Optional[List[int]]
    safety_score: float
    reason: str