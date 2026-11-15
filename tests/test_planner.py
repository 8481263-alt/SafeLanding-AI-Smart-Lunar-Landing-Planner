import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

from modules import lunar_visibility as visibility
from modules import planner
from modules.terrain import build_lola_horizon

START = visibility.parse_utc("2026-11-15T00:00Z")


class VisibilityTests(unittest.TestCase):
    def test_terrain_blocks_positive_elevation_and_produces_a_set(self):
        times = [START + timedelta(hours=i) for i in range(3)]
        sun = [visibility.Position(t, 90, angle) for t, angle in zip(times, (3, 1, 1))]
        earth = [visibility.Position(t, 10, 1) for t in times]
        rows, summary = visibility.analyze(sun, earth, lambda az: 2 if az == 90 else 0)
        crossing = "2026-11-15T00:30:00Z"
        self.assertEqual(summary["sun"]["set_times_utc"], [crossing])
        self.assertEqual(summary["sun"]["blocked_windows"][0]["start_utc"], crossing)
        self.assertEqual(summary["sun"]["terrain_blocked_hours"], 1.5)
        self.assertEqual(summary["sun_and_earth"]["overlap_hours"], 0.5)
        self.assertEqual(summary["earth"]["set_times_utc"], [])
        self.assertFalse(rows[-1]["sun_center_visible"])

    def test_horizon_touch_creates_no_darkness_window(self):
        times = [START + timedelta(hours=i) for i in range(3)]
        result = visibility.visibility_intervals(times, [1, 0, 1])
        self.assertEqual(result["set_times_utc"], [])
        self.assertEqual(result["visible_hours"], 2)
        self.assertEqual(result["blocked_windows"], [])

    def test_wrapped_azimuth_and_unaligned_end(self):
        end = START + timedelta(minutes=7)
        end_of_grid = START + timedelta(minutes=15)
        samples = [visibility.Position(START, 359, 3), visibility.Position(end_of_grid, 1, 3)]
        with patch.object(planner, "download_ephemerides", return_value=(samples, "live")) as fetch:
            with patch.dict(os.environ, {"SAFELANDING_LOLA_DEM": ""}):
                report = planner.plan_visibility(planner.SITES["apollo_12"], START, end, 15)
        self.assertEqual(report["terrain_status"], "outside_coverage")
        self.assertEqual(report["summary"]["end_utc"], visibility.iso(end))
        self.assertAlmostEqual(report["rows"][-1]["sun_azimuth_deg"], 359 + 14 / 15, places=5)
        self.assertEqual(report["site"]["longitude_east_deg"], 360 - 23.4219)
        self.assertEqual(fetch.call_args.args[5], end_of_grid)

    def test_horizons_reuses_only_a_valid_cache_entry(self):
        raw = "\n".join([
            "$$SOE", "2026-Nov-15 00:00,*, , 20.0, 2.0,",
            "2026-Nov-15 00:15,*, , 21.0, 3.0,", "$$EOE",
        ])
        response = Mock()
        response.json.return_value = {"result": raw}
        end = START + timedelta(minutes=15)
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ, {"SAFELANDING_HORIZONS_CACHE": folder},
        ), patch.object(visibility.requests, "get", return_value=response) as get:
            args = ("10", 0, -85, 0, START, end, 15)
            self.assertEqual(visibility.download_ephemerides(*args)[1], "live")
            self.assertEqual(visibility.download_ephemerides(*args)[1], "cache")
            self.assertEqual(get.call_count, 1)
            next(Path(folder).glob("*.txt")).write_text("truncated", encoding="utf-8")
            self.assertEqual(visibility.download_ephemerides(*args)[1], "live")
            self.assertEqual(get.call_count, 2)


class TerrainTests(unittest.TestCase):
    def test_ridge_is_east_of_observer_in_lunar_coordinates(self):
        elevations = np.zeros((220, 200), dtype=np.float32)
        elevations[84:96, 110:125] = 200
        lunar_crs = "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +R=1737400 +units=m"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ridge.tif"
            with rasterio.open(
                path, "w", driver="GTiff", width=200, height=220, count=1, dtype="float32",
                crs=lunar_crs, transform=from_origin(-5000, 6000, 50, 50),
            ) as raster:
                raster.write(elevations, 1)
            profile = build_lola_horizon(-89.95, 0, path)
        self.assertEqual(profile.radius_km, 4)
        self.assertGreater(dict(profile.samples)[90], 5)
        self.assertEqual(dict(profile.samples)[270], 0)
        self.assertEqual(profile.surface_height_km, 0)


if __name__ == "__main__":
    unittest.main()
