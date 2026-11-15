"""Live lunar visibility planning shared by Streamlit and FastAPI."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from functools import lru_cache
from math import ceil, isfinite
from pathlib import Path

from modules.lunar_visibility import (
    Position, UTC, analyze, download_ephemerides, flat_horizon, interpolate_horizon,
)
from modules.terrain import HorizonProfile, OutsideDEM, build_lola_horizon

APOLLO_SOURCE = "https://nssdc.gsfc.nasa.gov/planetary/lunar/lunar_sites.html"
LANDER_SOURCE = "https://nssdc.gsfc.nasa.gov/planetary/lunar/lunar_artifact_impacts.html"
RIDGE_SOURCE = "https://pgda.gsfc.nasa.gov/products/78"
LUNAR_SOLAR_DAY = timedelta(days=29, hours=12, minutes=44)
MAX_PERIOD = timedelta(days=62)
SAMPLING_STEPS = (5, 15, 30, 60, 120)


@dataclass(frozen=True)
class Site:
    id: str
    name: str
    latitude_deg: float
    longitude_east_deg: float
    group: str
    source: str | None = None


SITES = {
    site.id: site for site in (
        Site("connecting_ridge", "Connecting Ridge", -89.46315776, 222.49801509,
             "Polar exploration site", RIDGE_SOURCE),
        Site("apollo_11", "Apollo 11", 0.67416, 23.47314, "Apollo · USA", APOLLO_SOURCE),
        Site("apollo_12", "Apollo 12", -3.0128, -23.4219, "Apollo · USA", APOLLO_SOURCE),
        Site("apollo_14", "Apollo 14", -3.64589, -17.47194, "Apollo · USA", APOLLO_SOURCE),
        Site("apollo_15", "Apollo 15", 26.13239, 3.63330, "Apollo · USA", APOLLO_SOURCE),
        Site("apollo_16", "Apollo 16", -8.9734, 15.5011, "Apollo · USA", APOLLO_SOURCE),
        Site("apollo_17", "Apollo 17", 20.1911, 30.7723, "Apollo · USA", APOLLO_SOURCE),
        Site("im_1", "IM-1 Odysseus", -80.13, 1.44, "CLPS · USA", LANDER_SOURCE),
        Site("im_2", "IM-2 Athena", -84.7906, 29.1957, "CLPS · USA", LANDER_SOURCE),
        Site("blue_ghost_1", "Blue Ghost 1", 18.562, 61.810, "CLPS · USA", LANDER_SOURCE),
        Site("chang_e_4", "Chang’e 4", -45.4561, 177.5885, "China", LANDER_SOURCE),
        Site("chang_e_6", "Chang’e 6", -41.6385, -153.9852, "China", LANDER_SOURCE),
        Site("chandrayaan_3", "Chandrayaan-3 Vikram", -69.3741, 32.32, "India", LANDER_SOURCE),
        Site("slim", "SLIM", -13.3160, 25.2510, "Japan", LANDER_SOURCE),
        Site("luna_16", "Luna 16", -0.5137, 56.3638, "USSR", LANDER_SOURCE),
        Site("site_a", "Site A", -89.5, 0.0, "Comparison point"),
        Site("site_b", "Site B", -89.0, 45.0, "Comparison point"),
    )
}


@lru_cache(maxsize=64)
def _horizon(latitude: float, longitude: float, dem: Path | None, revision: int = 0) -> HorizonProfile:
    return build_lola_horizon(latitude, longitude, dem)


def _terrain(latitude: float, longitude: float) -> HorizonProfile | None:
    local = os.environ.get("SAFELANDING_LOLA_DEM")
    if local:
        path = Path(local).expanduser().resolve()
        try:
            return _horizon(latitude, longitude, path, path.stat().st_mtime_ns)
        except OutsideDEM:
            pass
        except OSError as exc:
            raise RuntimeError(f"Configured LOLA file cannot be read: {path.name}") from exc
    try:
        return _horizon(latitude, longitude, None)
    except OutsideDEM:
        return None


def plan_visibility(
    site: Site, start: datetime, end: datetime, step_minutes: int = 30,
    terrain_mode: str = "auto",
) -> dict:
    """Compute a UTC timeline with the local terrain and the 0° baseline."""
    latitude, longitude = site.latitude_deg, site.longitude_east_deg
    if not (-90 <= latitude <= 90 and isfinite(longitude) and -360 <= longitude <= 360):
        raise ValueError("Use latitude −90…90° and east-positive longitude −360…360°.")
    if (
        start.tzinfo is None or end.tzinfo is None
        or start.utcoffset() != timedelta(0) or end.utcoffset() != timedelta(0)
        or start.second or end.second or start.microsecond or end.microsecond
    ):
        raise ValueError("Use UTC times at minute precision.")
    if not timedelta(minutes=1) <= end - start <= MAX_PERIOD:
        raise ValueError("Choose a period from 1 minute to 62 Earth days.")
    if step_minutes not in SAMPLING_STEPS or terrain_mode not in ("auto", "flat"):
        raise ValueError("Invalid sampling step or horizon mode.")

    start, end = start.astimezone(UTC), end.astimezone(UTC)
    longitude %= 360
    profile = _terrain(latitude, longitude) if terrain_mode == "auto" else None
    horizon = interpolate_horizon(list(profile.samples)) if profile else flat_horizon
    step = timedelta(minutes=step_minutes)
    grid_end = start + ceil((end - start) / step) * step
    sun, sun_source = download_ephemerides(
        "10", longitude, latitude, profile.surface_height_km if profile else 0, start, grid_end, step_minutes,
    )
    earth, earth_source = download_ephemerides(
        "399", longitude, latitude, profile.surface_height_km if profile else 0, start, grid_end, step_minutes,
    )
    sun, earth = _clip_end(sun, end), _clip_end(earth, end)
    rows, summary = analyze(sun, earth, horizon)
    return {
        "site": asdict(replace(site, longitude_east_deg=longitude)),
        "rows": rows, "summary": summary,
        "terrain": asdict(profile) if profile else None,
        "terrain_status": "lola" if profile else ("outside_coverage" if terrain_mode == "auto" else "flat"),
        "ephemerides": {"sun": sun_source, "earth": earth_source},
        "step_minutes": step_minutes,
    }


def _clip_end(positions: list[Position], end: datetime) -> list[Position]:
    if positions[-1].time == end:
        return positions
    a, b = positions[-2:]
    fraction = (end - a.time) / (b.time - a.time)
    d_az = (b.azimuth_deg - a.azimuth_deg + 180) % 360 - 180
    return [
        *positions[:-1],
        Position(end, (a.azimuth_deg + d_az * fraction) % 360,
                 a.elevation_deg + (b.elevation_deg - a.elevation_deg) * fraction),
    ]
