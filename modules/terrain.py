"""Azimuthal horizon from NASA LOLA elevations in the lunar ME frame."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import aiohttp
import fsspec
import numpy as np
import rasterio
from pyproj import CRS, Geod, Transformer
from rasterio.errors import RasterioError
from rasterio.windows import Window

MOON_RADIUS_M = 1_737_400.0
LOLA_PRODUCT = "https://pgda.gsfc.nasa.gov/products/90"
LOLA_URL = "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LDEM_80S_80MPP_ADJ.TIF"
LOLA_NORTH_LIMIT = -80.0
POLAR_RADIUS_KM = 40.0
LOCAL_RADIUS_KM = 4.0
AZIMUTH_STEP_DEG = 1
MAX_WINDOW_PIXELS = 12_000_000
REMOTE_LOCK = Lock()


class OutsideDEM(ValueError):
    """The observer lies outside the advertised or raster coverage."""


@dataclass(frozen=True)
class HorizonProfile:
    samples: tuple[tuple[float, float], ...]
    surface_height_km: float
    radius_km: float
    requested_radius_km: float
    pixel_size_m: float
    minimum_valid_fraction: float
    source: str


def build_lola_horizon(
    latitude: float, longitude_east: float, dem_path: Path | None = None,
) -> HorizonProfile:
    """Sample a local horizon along lunar geodesics using a bounded raster window."""
    if dem_path is None and latitude > LOLA_NORTH_LIMIT:
        raise OutsideDEM("The default LOLA DEM covers 80–90°S.")

    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            if dem_path is not None:
                with rasterio.open(dem_path) as dem:
                    return _sample_horizon(dem, latitude, longitude_east, dem_path)
            with REMOTE_LOCK:
                _, filesystem = _remote_cache()
                with rasterio.open(LOLA_URL, opener=filesystem) as dem:
                    return _sample_horizon(dem, latitude, longitude_east, None)
    except (RasterioError, OSError, aiohttp.ClientError) as exc:
        raise RuntimeError("Could not read LOLA elevations. Retry or use a local DEM or the 0° horizon.") from exc


@lru_cache(maxsize=1)
def _remote_cache():
    """Share downloaded ranges between nearby sites for this process."""
    directory = TemporaryDirectory(prefix="safelanding-lola-")
    filesystem = fsspec.filesystem(
        "blockcache", target_protocol="https", cache_storage=directory.name,
        target_options={
            "block_size": 1 << 20,
            "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=20), "trust_env": True},
        },
    )
    return directory, filesystem


def _sample_horizon(dem, latitude: float, longitude: float, path: Path | None) -> HorizonProfile:
    if not dem.crs or not dem.count:
        raise ValueError("LOLA DEM needs an elevation band and a projected lunar CRS.")
    projected = CRS.from_user_input(dem.crs)
    geographic = projected.geodetic_crs
    if (
        not projected.is_projected or geographic is None
        or abs(geographic.ellipsoid.semi_major_metre - MOON_RADIUS_M) > 2_000
        or abs(geographic.ellipsoid.semi_minor_metre - MOON_RADIUS_M) > 2_000
        or any(abs(axis.unit_conversion_factor - 1) > 1e-6 for axis in projected.axis_info[:2])
    ):
        raise ValueError("Use a lunar projected DEM with horizontal coordinates in metres.")
    project = Transformer.from_crs(geographic, projected, always_xy=True)
    geodesic = Geod(a=MOON_RADIUS_M, b=MOON_RADIUS_M)
    x0, y0 = project.transform(longitude, latitude)
    col0, row0 = ~dem.transform * (x0, y0)
    if not (0 <= col0 < dem.width and 0 <= row0 < dem.height):
        raise OutsideDEM("Observer is outside the DEM footprint.")

    polar_mosaic = path is None or path.name.upper() == LOLA_URL.rsplit("/", 1)[-1]
    pixel_size = min(dem.res)
    requested_radius = POLAR_RADIUS_KM if polar_mosaic else LOCAL_RADIUS_KM
    distance_step = max(1, pixel_size / 2)
    distances = np.arange(distance_step, requested_radius * 1000 + 1e-8, distance_step)
    azimuths = np.arange(0, 360, AZIMUTH_STEP_DEG)
    bearings, ranges = np.meshgrid(azimuths, distances, indexing="ij")
    lons, lats, _ = geodesic.fwd(
        np.full(bearings.shape, longitude), np.full(bearings.shape, latitude), bearings, ranges,
    )
    xs, ys = project.transform(lons, lats)
    cols, rows = ~dem.transform * (xs, ys)

    # All bearings must have the same usable radius. Leave a pixel for the
    # bilinear interpolation of the elevation grid.
    covered = (cols >= 0.5) & (cols < dem.width - 0.5) & (rows >= 0.5) & (rows < dem.height - 0.5)
    if polar_mosaic:
        covered &= lats <= LOLA_NORTH_LIMIT
    gaps = np.flatnonzero(~covered.all(axis=0))
    count = int(gaps[0]) if gaps.size else len(distances)
    if count < 4:
        raise ValueError("Too little 360° DEM coverage around the observer.")
    distances, cols, rows = distances[:count], cols[:, :count], rows[:, :count]
    if not (0.5 <= col0 < dem.width - 0.5 and 0.5 <= row0 < dem.height - 0.5):
        raise ValueError("Observer is too close to the DEM edge.")
    left, top = floor(min(cols.min(), col0) - 0.5), floor(min(rows.min(), row0) - 0.5)
    right, bottom = ceil(max(cols.max(), col0) + 1.5), ceil(max(rows.max(), row0) + 1.5)
    window = Window(left, top, right - left, bottom - top)
    if window.width * window.height > MAX_WINDOW_PIXELS:
        raise ValueError("DEM window is too large. Use a lower-resolution raster.")
    cells = dem.read(1, window=window, masked=True).astype(np.float64).filled(np.nan)
    cells = cells * dem.scales[0] + dem.offsets[0]
    surface = float(_bilinear(cells, col0 - left, row0 - top))
    if not np.isfinite(surface):
        raise ValueError("Observer elevation is missing from the LOLA DEM.")
    elevations = _bilinear(cells, cols - left, rows - top)
    valid = np.isfinite(elevations).mean(axis=1)
    if valid.min() < 0.98:
        raise ValueError("LOLA has missing elevations along one or more azimuths.")

    # Same local cross-section geometry as the supplied LOLA prototype.
    angle = distances / MOON_RADIUS_M
    horizontal = (MOON_RADIUS_M + elevations) * np.sin(angle)
    vertical = (MOON_RADIUS_M + elevations) * np.cos(angle) - (MOON_RADIUS_M + surface)
    # Use the tangent as a conservative floor for a finite-radius profile.
    peaks = np.maximum(0, np.nanmax(np.degrees(np.arctan2(vertical, horizontal)), axis=1))
    return HorizonProfile(
        tuple((float(az), float(peak)) for az, peak in zip(azimuths, peaks)),
        surface / 1000, float(distances[-1]) / 1000, requested_radius,
        pixel_size, float(valid.min()), path.name if path is not None else LOLA_URL.rsplit("/", 1)[-1],
    )


def _bilinear(cells: np.ndarray, col, row):
    x, y = np.asarray(col) - 0.5, np.asarray(row) - 0.5
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = x - ix, y - iy
    return (
        cells[iy, ix] * (1 - dx) * (1 - dy)
        + cells[iy, ix + 1] * dx * (1 - dy)
        + cells[iy + 1, ix] * (1 - dx) * dy
        + cells[iy + 1, ix + 1] * dx * dy
    )
