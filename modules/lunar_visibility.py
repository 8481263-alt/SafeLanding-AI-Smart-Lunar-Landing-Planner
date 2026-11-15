"""JPL Horizons observer ephemerides and interpolated lunar visibility windows."""
from __future__ import annotations

import csv
import json
import math
import os
import re
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from time import time
from typing import Callable

import requests

HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"
UTC = timezone.utc
HORIZONS_DATE = re.compile(r"(?:A\.D\.\s*)?(\d{4})-([A-Za-z]{3})-(\d{2})\s+(\d{2}:\d{2}(?::\d{2})?)")
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "horizons"
CACHE_TTL_S = 7 * 24 * 3600
CACHE_ENTRIES = 256
REQUEST_LOCK = Lock()


@dataclass(frozen=True)
class Position:
    time: datetime
    azimuth_deg: float
    elevation_deg: float


def quote(value: str) -> str:
    return f"'{value}'"


def horizons_params(
    target: str, lon_east_deg: float, lat_deg: float, height_km: float,
    start: datetime, end: datetime, step_minutes: int,
) -> dict[str, str]:
    if target not in ("10", "399"):
        raise ValueError("Target must be 10 (Sun) or 399 (Earth)")
    return {
        "format": "json", "COMMAND": quote(target),
        "MAKE_EPHEM": quote("YES"), "EPHEM_TYPE": quote("OBSERVER"),
        "OBJ_DATA": quote("NO"), "CENTER": quote("coord@301"),
        "COORD_TYPE": quote("GEODETIC"),
        "SITE_COORD": quote(f"{lon_east_deg:.7f},{lat_deg:.7f},{height_km:.5f}"),
        "START_TIME": quote(start.strftime("%Y-%m-%d %H:%M")),
        "STOP_TIME": quote(end.strftime("%Y-%m-%d %H:%M")),
        "STEP_SIZE": quote(f"{step_minutes} m"),
        "QUANTITIES": quote("4"),  # apparent azimuth and elevation
        "APPARENT": quote("AIRLESS"), "TIME_TYPE": quote("UT"),
        "TIME_ZONE": quote("+00:00"), "TIME_DIGITS": quote("MINUTES"),
        "CAL_FORMAT": quote("CAL"), "CSV_FORMAT": quote("YES"),
        "ANG_FORMAT": quote("DEG"), "ELEV_CUT": quote("-90"),
        "SKIP_DAYLT": quote("NO"),
    }


def parse_horizons_table(text: str) -> list[Position]:
    """Parse Horizons CSV QUANTITIES=4 (date, flags, azimuth, elevation).

    Numeric fields after the date are only azimuth and elevation because the
    requested quantity list contains only #4. Variable visibility markers are
    ignored. Any API-format change fails loudly rather than silently guessing.
    """
    if "$$SOE" not in text or "$$EOE" not in text:
        raise ValueError("Horizons response lacks $$SOE/$$EOE markers: " + text[:220])
    body = text.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    result: list[Position] = []
    for row in csv.reader(body.splitlines(), skipinitialspace=True):
        if not row or not row[0].strip():
            continue
        match = HORIZONS_DATE.search(row[0])
        if not match:
            raise ValueError(f"Unexpected Horizons timestamp field: {row[0]!r}")
        year, month, day, clock = match.groups()
        instant = datetime.fromisoformat(
            f"{year}-{MONTHS.index(month.title()) + 1:02d}-{day}T{clock}+00:00"
        )
        numeric: list[float] = []
        for field in row[1:]:
            value = field.strip()
            if not value:
                continue
            try:
                numeric.append(float(value))
            except ValueError:
                # Horizons may include single-character Sun/target flags.
                if len(value) > 3:
                    raise ValueError(f"Unexpected Horizons field: {value!r}")
        if len(numeric) != 2:
            raise ValueError(f"Expected only azimuth/elevation; got {numeric} from {row!r}")
        az, el = numeric
        if not 0.0 <= az <= 360.0 or not -90.0 <= el <= 90.0:
            raise ValueError(f"Out-of-range azimuth/elevation: {az}, {el}")
        result.append(Position(instant, az % 360.0, el))
    if len(result) < 2:
        raise ValueError("Horizons returned fewer than two observations")
    if any(b.time <= a.time for a, b in zip(result, result[1:])):
        raise ValueError("Horizons time series is not strictly increasing")
    return result


def download_ephemerides(
    target: str, lon: float, lat: float, height_km: float,
    start: datetime, end: datetime, step_minutes: int,
) -> tuple[list[Position], str]:
    """Fetch one target, returning positions and their provenance."""
    params = horizons_params(target, lon, lat, height_km, start, end, step_minutes)
    key = sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    path = Path(os.environ.get("SAFELANDING_HORIZONS_CACHE", CACHE_DIR)) / f"{key}.txt"
    step = timedelta(minutes=step_minutes)

    def parse(raw: str) -> list[Position]:
        positions = parse_horizons_table(raw)
        if (
            positions[0].time != start or positions[-1].time != end
            or any(b.time - a.time != step for a, b in zip(positions, positions[1:]))
        ):
            raise ValueError("Horizons did not return the requested UTC sampling grid")
        return positions

    # JPL asks clients to submit one request at a time. Recheck the cache inside
    # the lock so concurrent users of this process can share the first response.
    with REQUEST_LOCK:
        try:
            if time() - path.stat().st_mtime < CACHE_TTL_S:
                return parse(path.read_text(encoding="utf-8")), "cache"
        except (OSError, ValueError):
            pass
        try:
            response = requests.get(HORIZONS_API, params=params, timeout=(8, 30))
            response.raise_for_status()
            doc = response.json()
            if not isinstance(doc, dict):
                raise ValueError("Missing ephemeris in the Horizons response")
            if doc.get("error"):
                raise ValueError(str(doc["error"]))
            raw = doc.get("result")
            if not isinstance(raw, str):
                raise ValueError("Missing ephemeris in the Horizons response")
            positions = parse(raw)
        except (requests.RequestException, ValueError) as err:
            raise RuntimeError(f"NASA/JPL Horizons ({'Sun' if target == '10' else 'Earth'}): {err}") from err
        save_ephemeris(path, raw)
        return positions, "live"


def save_ephemeris(path: Path, raw: str) -> None:
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False, mode="w", encoding="utf-8") as stream:
            temporary = Path(stream.name)
            stream.write(raw)
        temporary.replace(path)
        entries = sorted(path.parent.glob("*.txt"), key=lambda entry: entry.stat().st_mtime, reverse=True)
        for old in entries[CACHE_ENTRIES:]:
            old.unlink(missing_ok=True)
    except OSError:
        # Cache storage is optional on read-only deployments.
        pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def flat_horizon(_: float) -> float:
    return 0.0


def interpolate_horizon(rows: list[tuple[float, float]]) -> Callable[[float], float]:
    rows = sorted((az % 360, elevation) for az, elevation in rows)
    if any(not math.isfinite(az) or not -90 <= elevation <= 90 for az, elevation in rows):
        raise ValueError("Invalid horizon angles")
    if len(rows) < 3 or len({r[0] for r in rows}) != len(rows):
        raise ValueError("Horizon requires >=3 distinct azimuth samples")
    azimuths = [p[0] for p in rows]
    gaps = [(azimuths[(i + 1) % len(rows)] - azimuths[i]) % 360 for i in range(len(rows))]
    if max(gaps) > 30:
        raise ValueError("Horizon has an azimuth gap >30 degrees")

    azimuths = [azimuths[-1] - 360, *azimuths, azimuths[0] + 360]
    elevations = [rows[-1][1], *(elevation for _, elevation in rows), rows[0][1]]

    def horizon(azimuth: float) -> float:
        angle = azimuth % 360.0
        i = bisect_right(azimuths, angle)
        fraction = (angle - azimuths[i - 1]) / (azimuths[i] - azimuths[i - 1])
        return elevations[i - 1] + fraction * (elevations[i] - elevations[i - 1])
    return horizon


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def crossings(times: list[datetime], margins: list[float]) -> list[tuple[datetime, bool]]:
    """Piecewise-linear threshold crossings; approximate, sampling-limited."""
    if len(times) != len(margins) or len(times) < 2:
        raise ValueError("Time and margin arrays must have >=2 matching entries")
    if any(t2 <= t1 for t1, t2 in zip(times, times[1:])):
        raise ValueError("Timestamps must be increasing")
    events = []
    for i in range(len(times) - 1):
        t0, t1 = times[i], times[i + 1]
        m0, m1 = margins[i], margins[i + 1]
        v0, v1 = m0 > 0, m1 > 0
        if v0 == v1:
            continue
        fraction = -m0 / (m1 - m0)
        instant = t0 + (t1 - t0) * fraction
        # When a curve touches the horizon and rebounds at the same instant,
        # there is no interval of changed visibility.
        if events and events[-1][0] == instant:
            events.pop()
        else:
            events.append((instant, v1))
    return events


def interval(start: datetime, end: datetime) -> dict:
    return {"start_utc": iso(start), "end_utc": iso(end),
            "duration_h": round((end - start).total_seconds() / 3600, 6)}


def visibility_intervals(times: list[datetime], margins: list[float]) -> dict:
    events = crossings(times, margins)
    visible = margins[0] > 0
    start = times[0]
    windows, blocked = [], []
    for end, next_visible in [*events, (times[-1], False)]:
        if end > start:
            (windows if visible else blocked).append(interval(start, end))
        start, visible = end, next_visible
    return {
        "windows": windows, "blocked_windows": blocked,
        "visible_hours": round(hours(windows), 6),
        "longest_blocked_hours": max((w["duration_h"] for w in blocked), default=0),
        "rise_times_utc": [iso(t) for t, rising in events if rising],
        "set_times_utc": [iso(t) for t, rising in events if not rising],
    }


def overlap_windows(a: list[dict], b: list[dict]) -> list[dict]:
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        x, y = a[i], b[j]
        x_end, y_end = parse_utc(x["end_utc"]), parse_utc(y["end_utc"])
        begin = max(parse_utc(x["start_utc"]), parse_utc(y["start_utc"]))
        end = min(x_end, y_end)
        if end > begin:
            result.append(interval(begin, end))
        i += x_end <= y_end
        j += y_end <= x_end
    return result


def parse_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise ValueError("Use UTC timestamps with Z or +00:00, e.g. 2026-11-14T00:00Z")
    return dt.astimezone(UTC)


def hours(windows: list[dict]) -> float:
    return sum((parse_utc(w["end_utc"]) - parse_utc(w["start_utc"])).total_seconds() / 3600 for w in windows)


def analyze(
    sun: list[Position], earth: list[Position],
    horizon: Callable[[float], float] = flat_horizon, clearance_deg: float = 0.0,
) -> tuple[list[dict], dict]:
    if len(sun) != len(earth) or any(a.time != b.time for a, b in zip(sun, earth)):
        raise ValueError("Sun and Earth timestamps do not match")
    times = [r.time for r in sun]
    rows: list[dict] = []
    for s, e in zip(sun, earth):
        sh = horizon(s.azimuth_deg)
        eh = horizon(e.azimuth_deg)
        sm = s.elevation_deg - sh - clearance_deg
        em = e.elevation_deg - eh - clearance_deg
        rows.append({
            "timestamp_utc": iso(s.time),
            "sun_azimuth_deg": round(s.azimuth_deg, 6),
            "sun_elevation_deg": round(s.elevation_deg, 6),
            "earth_azimuth_deg": round(e.azimuth_deg, 6),
            "earth_elevation_deg": round(e.elevation_deg, 6),
            "sun_horizon_deg": round(sh, 6), "earth_horizon_deg": round(eh, 6),
            "sun_visibility_margin_deg": sm,
            "earth_visibility_margin_deg": em,
            "sun_center_visible": sm > 0, "earth_center_visible": em > 0,
        })
    sun_result = visibility_intervals(times, [r["sun_visibility_margin_deg"] for r in rows])
    earth_result = visibility_intervals(times, [r["earth_visibility_margin_deg"] for r in rows])
    both_windows = overlap_windows(sun_result["windows"], earth_result["windows"])
    full_h = (times[-1] - times[0]).total_seconds() / 3600
    for target, result in (("sun", sun_result), ("earth", earth_result)):
        tangent = visibility_intervals(times, [r[f"{target}_elevation_deg"] - clearance_deg for r in rows])
        result["terrain_blocked_hours"] = max(0, round(tangent["visible_hours"] - result["visible_hours"], 6))
    summary = {
        "start_utc": iso(times[0]), "end_utc": iso(times[-1]), "period_hours": round(full_h, 4),
        "sun": {**sun_result, "visible_fraction": round(hours(sun_result["windows"]) / full_h, 6)},
        "earth": {**earth_result, "visible_fraction": round(hours(earth_result["windows"]) / full_h, 6)},
        "sun_and_earth": {"overlap_hours": round(hours(both_windows), 4),
                          "windows": both_windows},
    }
    return rows, summary
