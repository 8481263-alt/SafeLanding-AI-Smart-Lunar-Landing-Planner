"""Streamlit controls and charts for the live mission planner."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pyproj import Geod

from modules.lunar_visibility import UTC, parse_utc
from modules.planner import LUNAR_SOLAR_DAY, MAX_PERIOD, SAMPLING_STEPS, SITES, Site, plan_visibility
from modules.terrain import LOLA_PRODUCT, MOON_RADIUS_M

SUN = "#F3CA73"
EARTH = "#65B6FF"
HORIZON = "#EB8B82"
BODY_NAMES = {"sun": "Sun", "earth": "Earth"}


def render_planner() -> None:
    st.subheader("Mission planner")
    st.caption("Find sunlight and direct Earth line-of-sight windows for a lunar site and time.")
    location, period = st.columns(2)
    with location:
        selected = st.multiselect(
            "Landing sites", list(SITES), default=["connecting_ridge"],
            format_func=lambda key: f"{SITES[key].name} · {SITES[key].group}",
            max_selections=4,
        )
        sites = [SITES[key] for key in selected]
        if st.checkbox("Add custom coordinates"):
            lat_column, lon_column = st.columns(2)
            lat = lat_column.number_input("Latitude (°N)", -90.0, 90.0, -89.45, step=0.0001, format="%.6f")
            lon = lon_column.number_input(
                "East longitude (°)", -360.0, 360.0, 222.5, step=0.0001, format="%.6f",
                help="Use a negative angle for west longitude. Coordinates use the lunar mean Earth/polar axis frame.",
            )
            sites.append(Site("custom", "Custom coordinates", lat, lon, "Custom"))
        if st.checkbox("Compare with a nearby point", value=True):
            offset = st.number_input("Distance north (m)", 10, 5000, 500, step=10)
            if sites:
                site = sites[0]
                lon, lat, _ = Geod(a=MOON_RADIUS_M, b=MOON_RADIUS_M).fwd(
                    site.longitude_east_deg, site.latitude_deg, 0, offset,
                )
                sites.append(Site("nearby", f"{offset:,} m from {site.name}", lat, lon, "Nearby comparison"))
    with period:
        day_min, day_max = date(1600, 1, 1), date(2500, 12, 31)
        today = datetime.now(UTC).date()
        start_day_col, start_clock_col = st.columns(2)
        start_day = start_day_col.date_input("Start date (UTC)", today, day_min, day_max)
        start_clock = start_clock_col.time_input("Start time (UTC)", time.min, step=60)
        start = datetime.combine(start_day, start_clock, UTC)
        duration = st.radio(
            "Planning period", ["Custom end", "One lunar solar day", "Two lunar solar days"], horizontal=True,
        )
        if duration == "Custom end":
            end_day_col, end_clock_col = st.columns(2)
            end_day = end_day_col.date_input("End date (UTC)", today + timedelta(days=7), day_min, day_max)
            end_clock = end_clock_col.time_input("End time (UTC)", time.min, step=60)
            end = datetime.combine(end_day, end_clock, UTC)
        else:
            end = start + LUNAR_SOLAR_DAY * (1 if duration == "One lunar solar day" else 2)
            st.caption(f"Ends {_utc_label(end)}. A mean lunar solar day lasts about 29.53 Earth days.")
    settings = st.columns(2)
    mode = settings[0].selectbox(
        "Horizon", ["auto", "flat"],
        format_func={"auto": "LOLA where covered", "flat": "Geometry only (0° tangent)"}.get,
        help="The default LOLA map covers 80–90°S. Sites beyond the coverage use a labelled 0° tangent.",
    )
    step = settings[1].selectbox("Sampling interval (minutes)", SAMPLING_STEPS, index=2)
    query = (tuple(sites), start, end, step, mode)
    if st.button("Calculate visibility", type="primary"):
        st.session_state.pop("planner_results", None)
        st.session_state.pop("planner_moment", None)
        if not 1 <= len(sites) <= 4:
            st.error("Choose one to four sites in total, including custom and nearby points.")
            return
        if not timedelta(minutes=1) <= end - start <= MAX_PERIOD:
            st.error("Choose an end after the start and a period of at most 62 Earth days.")
            return
        reports, errors = [], []
        for site in sites:
            with st.spinner(f"Calculating {site.name}…"):
                try:
                    reports.append(plan_visibility(site, start, end, step, mode))
                except (RuntimeError, ValueError, OverflowError) as exc:
                    errors.append((site.name, str(exc)))
        st.session_state["planner_results"] = (query, reports, errors)

    if "planner_results" not in st.session_state:
        return
    previous, reports, errors = st.session_state["planner_results"]
    if query != previous:
        st.info("Inputs have changed. Select Calculate visibility to update the results.")
    for name, message in errors:
        st.error(f"{name} · {message}")
    if reports:
        _render_results(reports)


def _utc_label(instant: datetime) -> str:
    return instant.strftime("%d %b %Y %H:%M UTC")


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 1:
        return "<1 min"
    days, remainder = divmod(minutes, 24 * 60)
    hours, minutes = divmod(remainder, 60)
    return " ".join(f"{number} {unit}" for number, unit in ((days, "d"), (hours, "h"), (minutes, "min")) if number)


def _render_results(reports: list[dict]) -> None:
    times = [parse_utc(row["timestamp_utc"]) for row in reports[0]["rows"]]
    st.divider()
    st.caption(f"{_utc_label(times[0])} → {_utc_label(times[-1])} · {reports[0]['step_minutes']} min sampling")
    st.subheader("Time to horizon crossing")
    moment = st.select_slider("Planning moment (UTC)", options=times, format_func=_utc_label, key="planner_moment")
    index = times.index(moment)
    cols = st.columns(len(reports)) if len(reports) <= 2 else st.columns(2)
    for i, report in enumerate(reports):
        with cols[i % len(cols)]:
            _site_status(report, index, moment)

    chart_tab, sensitivity_tab, window_tab = st.tabs(["Elevation & horizon", "Sensitivity heatmap", "Visibility windows"])
    with chart_tab:
        lookup = {report["site"]["id"]: report for report in reports}
        site_id = st.selectbox("Plot site", lookup, format_func=lambda key: lookup[key]["site"]["name"])
        _plot_horizon(lookup[site_id], times, index)
    with sensitivity_tab:
        _sensitivity(reports, times, moment)
    with window_tab:
        _windows(reports)
    with st.expander("Data and model"):
        st.markdown(
            "Visibility uses the centres of the Sun and Earth. Sunrise and set times are interpolated between "
            "samples. Short events can fall between samples. The LOLA profile is limited to the displayed radius "
            "and floored at the 0° tangent. Earth visibility estimates geometric line of sight. "
            "Sun visibility estimates geometric exposure."
        )
        st.markdown(
            f"[NASA/JPL Horizons](https://ssd-api.jpl.nasa.gov/doc/horizons.html) · "
            f"[NASA LOLA polar DEM]({LOLA_PRODUCT})"
        )


def _site_status(report: dict, index: int, moment: datetime) -> None:
    site = report["site"]
    st.markdown(f"**{site['name']}**")
    details = f"{site['latitude_deg']:+.6f}° lat, {site['longitude_east_deg']:.6f}° E"
    if site["source"]:
        details += f" · [coordinates]({site['source']})"
    st.caption(details)
    terrain = report["terrain"]
    if terrain:
        st.caption(
            f"LOLA · {terrain['pixel_size_m']:g} m/pixel · {terrain['radius_km']:.2f} km radius "
            f"· {terrain['source']}"
        )
        if terrain["radius_km"] + terrain["pixel_size_m"] / 1000 < terrain["requested_radius_km"]:
            st.caption("The DEM edge limits the radius at this site.")
    elif report["terrain_status"] == "outside_coverage":
        st.caption("0° tangent · outside available LOLA coverage")
    else:
        st.caption("0° tangent · geometry only")
    sources = report["ephemerides"]
    st.caption(f"Horizons · Sun {sources['sun']} · Earth {sources['earth']}")
    row = report["rows"][index]
    for body, name in BODY_NAMES.items():
        summary = report["summary"][body]
        visible = row[f"{body}_center_visible"]
        margin = row[f"{body}_visibility_margin_deg"]
        st.write(f"**{name}** · {'visible' if visible else 'below or on horizon'} · {margin:+.2f}° clearance")
        st.caption(_next_event(summary["set_times_utc"], moment, "Sets", "No set in the remaining period."))
        if not visible:
            st.caption(_next_event(summary["rise_times_utc"], moment, "Rises", "No rise in the remaining period."))


def _next_event(events: list[str], moment: datetime, title: str, absent: str) -> str:
    for event in events:
        instant = parse_utc(event)
        if instant >= moment:
            wait = "now" if instant == moment else f"in {_duration((instant - moment).total_seconds())}"
            return f"{title} {wait} · {_utc_label(instant + timedelta(seconds=30))}"
    return absent


def _plot_horizon(report: dict, times: list[datetime], index: int) -> None:
    body = st.radio("Body", list(BODY_NAMES), format_func=BODY_NAMES.get, horizontal=True, key="planner_body")
    rows = report["rows"]
    point = rows[index]
    fig = go.Figure()
    fig.add_scatter(
        x=times, y=[r[f"{body}_elevation_deg"] for r in rows], name=f"{BODY_NAMES[body]} elevation",
        mode="lines", line_color=SUN if body == "sun" else EARTH,
    )
    fig.add_scatter(
        x=times, y=[r[f"{body}_horizon_deg"] for r in rows],
        name="Terrain at body azimuth" if report["terrain"] else "0° tangent", mode="lines", line_color=HORIZON,
    )
    if report["terrain"]:
        fig.add_hline(y=0, line_dash="dot", line_color="#8FA6C8", annotation_text="0° tangent")
    fig.add_vline(x=times[index], line_dash="dot", line_color="#8FA6C8")
    fig.update_layout(
        height=335, margin=dict(t=24, b=46), xaxis_title="UTC", yaxis_title="Elevation (°)",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig)
    if report["terrain"]:
        with st.expander("Terrain profile by azimuth", expanded=True):
            samples = report["terrain"]["samples"]
            azimuths, elevations = zip(*samples, (360, samples[0][1]))
            profile = go.Figure()
            profile.add_scatter(x=azimuths, y=elevations, name="Terrain", fill="tozeroy", line_color=HORIZON)
            profile.add_scatter(
                x=[point[f"{body}_azimuth_deg"]], y=[point[f"{body}_elevation_deg"]],
                mode="markers", marker=dict(size=12, color=SUN if body == "sun" else EARTH),
                name=f"{BODY_NAMES[body]} at planning moment",
            )
            profile.update_layout(
                height=300, margin=dict(t=24, b=46), xaxis_title="Azimuth (°)", yaxis_title="Elevation (°)",
                legend=dict(orientation="h"),
            )
            profile.update_xaxes(range=[0, 360], tickvals=[0, 90, 180, 270, 360], ticktext=["N", "E", "S", "W", "N"])
            st.plotly_chart(profile)


def _sensitivity(reports: list[dict], times: list[datetime], moment: datetime) -> None:
    body = st.radio(
        "Heatmap body", ["sun", "earth", "both"], horizontal=True,
        format_func={"sun": "Sun", "earth": "Earth", "both": "Sun and Earth"}.get,
    )
    rows = []
    for report in reports:
        if body == "both":
            rows.append([min(r["sun_visibility_margin_deg"], r["earth_visibility_margin_deg"]) for r in report["rows"]])
        else:
            rows.append([r[f"{body}_visibility_margin_deg"] for r in report["rows"]])
    fig = go.Figure(go.Heatmap(
        x=times, y=[report["site"]["name"] for report in reports], z=rows,
        zmin=-3, zmax=3,
        colorscale=[
            (0, "#AD392E"), (0.499, "#E6A29C"), (0.5, "#F2E9D8"), (0.501, "#92DBD1"), (1, "#146B61"),
        ],
        colorbar=dict(title="Clearance (°)", tickvals=[-3, 0, 3], thickness=14),
        hovertemplate="%{y}<br>%{x|%d %b %Y %H:%M} UTC<br>%{z:+.2f}° clearance<extra></extra>",
    ))
    fig.add_vline(x=moment, line_dash="dot", line_color="#D4E4FF")
    fig.update_layout(height=max(205, 100 + 44 * len(reports)), margin=dict(t=18, b=56), xaxis_title="UTC")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig)
    st.caption(
        "Clearance is body elevation minus the local horizon at its azimuth. Positive values indicate visibility. "
        "Near 0°, a small change in date or terrain can move a body below the horizon. "
        "For both bodies, the colour shows the smaller clearance. Hover for the angle beyond the ±3° colour scale."
    )


def _windows(reports: list[dict]) -> None:
    overview, windows = [], []
    has_terrain = any(report["terrain"] for report in reports)
    for report in reports:
        name, summary = report["site"]["name"], report["summary"]
        metrics = {
            "Site": name, "Sun visible (%)": round(100 * summary["sun"]["visible_fraction"], 2),
            "Earth visible (%)": round(100 * summary["earth"]["visible_fraction"], 2),
            "Longest darkness (h)": round(summary["sun"]["longest_blocked_hours"], 2),
            "Longest Earth outage (h)": round(summary["earth"]["longest_blocked_hours"], 2),
            "Both visible (h)": round(summary["sun_and_earth"]["overlap_hours"], 2),
        }
        if has_terrain:
            metrics["Sun hidden by terrain (h)"] = round(summary["sun"]["terrain_blocked_hours"], 2) if report["terrain"] else None
            metrics["Earth hidden by terrain (h)"] = round(summary["earth"]["terrain_blocked_hours"], 2) if report["terrain"] else None
        overview.append(metrics)
        for label, intervals in (
            ("Sun visible", summary["sun"]["windows"]),
            ("Darkness", summary["sun"]["blocked_windows"]),
            ("Earth visible", summary["earth"]["windows"]),
            ("Earth below horizon", summary["earth"]["blocked_windows"]),
            ("Sun and Earth visible", summary["sun_and_earth"]["windows"]),
        ):
            for window in intervals:
                windows.append({
                    "Site": name, "Window": label, "Start (UTC)": _utc_label(parse_utc(window["start_utc"])),
                    "End (UTC)": _utc_label(parse_utc(window["end_utc"])), "Hours": round(window["duration_h"], 2),
                })
    st.dataframe(pd.DataFrame(overview), hide_index=True)
    st.dataframe(pd.DataFrame(windows), hide_index=True)
