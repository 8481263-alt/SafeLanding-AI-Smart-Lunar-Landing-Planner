"""Тимчасова заміна модуля safety_algorithm (поки немає коду Людини 4).

Інтерфейс — той, що очікує frontend/app.py:
    analyze_landing_zones(request: dict) -> dict

request: image_width, image_height, hazards=[{bbox_xyxy, type, confidence}],
         landing_radius_px, safety_margin_px, top_n
result:  status ("ok" | "no_safe_zone"), candidates=[{x, y, clearance_px, score}], warnings=[str]

Логіка: центр модуля дозволений там, де відстань до найближчої небезпеки не менша
за радіус + запас, а до краю кадру — не менша за радіус. Кандидати вибираються за
найбільшим просвітом, кола не перекриваються. Це оцінка по одному 2D-кадру.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["analyze_landing_zones"]


def analyze_landing_zones(request: dict) -> dict:
    w = int(request["image_width"])
    h = int(request["image_height"])
    radius = float(request.get("landing_radius_px", 20.0))
    margin = float(request.get("safety_margin_px", 0.0))
    top_n = max(1, int(request.get("top_n", 3)))
    need = radius + margin
    warnings = ["Stand-in algorithm (distance transform on one 2D frame); "
                "slope and undetected hazards are not considered."]

    k = min(1.0, 400.0 / max(w, h))                       # рахуємо на зменшеній сітці
    gw, gh = max(1, round(w * k)), max(1, round(h * k))
    free = np.ones((gh, gw), dtype=bool)
    for hz in request.get("hazards", []):
        x1, y1, x2, y2 = (float(v) * k for v in hz["bbox_xyxy"])
        free[max(0, int(np.floor(y1))):min(gh, int(np.ceil(y2))),
             max(0, int(np.floor(x1))):min(gw, int(np.ceil(x2)))] = False

    if free.all():
        dist_hz = np.full((gh, gw), np.inf)
        warnings.append("No hazards were detected in the frame.")
    else:
        dist_hz = ndimage.distance_transform_edt(free) / k          # у пікселях оригіналу

    ys, xs = np.mgrid[0:gh, 0:gw]
    cx, cy = (xs + 0.5) / k, (ys + 0.5) / k                       # центри клітинок, пікселі оригіналу
    dist_edge = np.minimum.reduce([cx, cy, w - cx, h - cy])
    clearance = np.minimum(dist_hz, dist_edge)
    valid = (dist_hz >= need) & (dist_edge >= radius) & free

    candidates = []
    for _ in range(top_n):
        if not valid.any():
            break
        masked = np.where(valid, clearance, -1.0)
        iy, ix = np.unravel_index(int(np.argmax(masked)), masked.shape)
        c = float(clearance[iy, ix])
        extra = max(0.0, c - need) / max(need, 1e-6)
        candidates.append({"x": float(cx[iy, ix]), "y": float(cy[iy, ix]),
                           "clearance_px": c, "score": 100.0 * extra / (1.0 + extra)})
        valid &= np.hypot(cx - cx[iy, ix], cy - cy[iy, ix]) >= 2 * radius   # кола не перекриваються

    return {"status": "ok" if candidates else "no_safe_zone",
            "candidates": candidates, "warnings": warnings}