import csv
import json
from pathlib import Path

from fastapi import HTTPException

DATA = Path(__file__).resolve().parent.parent / "data" / "astro"

def _load(site_id: str):
    folder = DATA / site_id
    if not folder.is_dir():
        available = sorted(p.name for p in DATA.iterdir() if p.is_dir())
        raise HTTPException(404, f"Unknown site '{site_id}'. Available: {available}")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    with (folder / "timeline.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return summary, rows


def compute_series(site_id: str, start: str, end: str) -> dict:
    summary, rows = _load(site_id)
    if len(end) == 10:  # "2026-11-21" -> включно до кінця доби
        end += "T23:59:59Z"
    step_h = summary["model"]["sampling_step_minutes"] / 60

    series = [
        {
            "time_utc": r["timestamp_utc"],
            "sun_elevation_deg": float(r["sun_elevation_deg"]),
            "sun_azimuth_deg": float(r["sun_azimuth_deg"]),
            "earth_elevation_deg": float(r["earth_elevation_deg"]),
            "earth_azimuth_deg": float(r["earth_azimuth_deg"]),
            "sun_visible": r["sun_center_visible"] == "True",
            "earth_visible": r["earth_center_visible"] == "True",
        }
        for r in rows
        if start <= r["timestamp_utc"] <= end or r["timestamp_utc"].startswith(start)
    ]
    if not series:
        raise HTTPException(
            404,
            f"No data in {start}..{end}. Available: {summary['start_utc']} .. {summary['end_utc']}",
        )

    return {
        "site_id": site_id,
        "lat": summary["site"]["latitude_deg"],
        "lon": summary["site"]["longitude_east_deg"],
        "series": series,
        "illuminated_hours": round(sum(p["sun_visible"] for p in series) * step_h, 2),
        "earth_visible_hours": round(sum(p["earth_visible"] for p in series) * step_h, 2),
        "simultaneous_hours": round(
            sum(p["sun_visible"] and p["earth_visible"] for p in series) * step_h, 2
        ),
        "horizon_assumption": summary["model"]["horizon"],
        "caveats": summary["model"]["caveats"],
    }

def compare_sites(a: str, b: str) -> dict:
    from modules.compare_scenarios import compare

    sa, _ = _load(a)
    sb, _ = _load(b)
    return compare(sa, sb)