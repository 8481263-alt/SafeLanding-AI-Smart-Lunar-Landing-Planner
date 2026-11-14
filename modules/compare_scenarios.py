import argparse
import json
from pathlib import Path


def summarize(s: dict) -> dict:
    return {
        "site": s.get("site", {}),
        "period_hours": s["period_hours"],
        "sun_visible_percent": round(100 * s["sun"]["visible_fraction"], 2),
        "earth_visible_percent": round(100 * s["earth"]["visible_fraction"], 2),
        "longest_dark_hours": s["sun"]["longest_dark_hours"],
        "longest_direct_los_outage_hours": s["earth"]["longest_no_direct_los_hours"],
        "simultaneous_visibility_hours": s["sun_and_earth"]["overlap_hours"],
        "horizon_assumption": s.get("model", {}).get("horizon", "UNKNOWN"),
        "is_synthetic": s.get("model", {}).get("synthetic_demo", False),
    }


def compare(a: dict, b: dict) -> dict:
    if (a["start_utc"], a["end_utc"]) != (b["start_utc"], b["end_utc"]):
        raise ValueError("Different time periods; do not compare percentages without labeling")
    return {"scenario_a": summarize(a), "scenario_b": summarize(b),
            "note": "Comparison of Sun/Earth center geometric visibility, not landing safety or RF link budget."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site_a", type=Path)
    ap.add_argument("site_b", type=Path)
    ap.add_argument("--out", type=Path, default=Path("comparison.json"))
    args = ap.parse_args()
    a = json.loads(args.site_a.read_text(encoding="utf-8"))
    b = json.loads(args.site_b.read_text(encoding="utf-8"))
    report = compare(a, b)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.out}")
    for name in ["scenario_a", "scenario_b"]:
        c = report[name]
        print(f"{name}: Sun {c['sun_visible_percent']}%, Earth {c['earth_visible_percent']}%, simultaneous {c['simultaneous_visibility_hours']}h")


if __name__ == "__main__":
    main()