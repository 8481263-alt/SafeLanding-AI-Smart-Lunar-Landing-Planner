# SafeLanding AI

A mission planner for the [Lunar Landing Site Navigator](https://www.spaceappschallenge.org/2026/challenges/lunar-landing-site-navigator/) challenge. Compare Sun and Earth visibility at up to four locations. Choose Apollo, CLPS, or other mission sites, enter your own coordinates, or add a nearby comparison point.

## Run on Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r frontend/requirements.txt
Set-Location frontend
python -m streamlit run app.py
```

Open [Streamlit](http://localhost:8501). For a quick comparison, choose **Connecting Ridge**, a **500 m** offset, **15–22 November 2026 UTC**, and **LOLA where covered**. Select **Calculate visibility**. **Elevation & horizon** shows where the body's path crosses the terrain profile. **Sensitivity heatmap** compares angular clearance above the horizon across dates and sites. **Visibility windows** lists darkness, loss of direct Earth visibility, and periods when both bodies are visible.

Enter dates and times in UTC. Choose a custom period of up to 62 Earth days, or choose one or two mean lunar solar days. The **Planning moment** slider shows the time to the next set and, when the body is below the horizon, the time to the next rise within the selected period.

## Data and model

- Sun and Earth positions come from [NASA/JPL Horizons](https://ssd-api.jpl.nasa.gov/doc/horizons.html). Valid responses are cached for seven days in `.cache/horizons`. Use `SAFELANDING_HORIZONS_CACHE` to set a different cache directory.
- The planner builds the terrain horizon from [NASA LOLA LDEM_80S_80MPP_ADJ](https://pgda.gsfc.nasa.gov/products/90) within 80–90°S. It reads a window from the remote GeoTIFF, samples lunar geodesics at 1° azimuth intervals for up to 40 km, and accounts for lunar curvature. Outside the DEM coverage, the interface labels the horizon as a 0° tangent.
- To use a local lunar GeoTIFF in the MOON_ME frame, set `$env:SAFELANDING_LOLA_DEM = "C:\path\to\model.tif"` before starting the app. A local 5 m/pixel tile from [NASA PGDA](https://pgda.gsfc.nasa.gov/products/78) uses a radius of up to 4 km. Keep the filename `LDEM_80S_80MPP_ADJ.TIF` when using a local copy of the standard polar DEM.
- Visibility crossings for the centres of the Sun and Earth are interpolated between samples. The terrain profile uses the sampled DEM radius and a 0° tangent as its lower bound. Earth visibility estimates geometric line of sight. Solar panel output and radio links require separate models.
- Site coordinates use the lunar mean Earth/polar axis frame with east-positive longitude. [NASA Apollo Landing Sites](https://nssdc.gsfc.nasa.gov/planetary/lunar/lunar_sites.html), [NASA Lunar Spacecraft Coordinates](https://nssdc.gsfc.nasa.gov/planetary/lunar/lunar_artifact_impacts.html)

## FastAPI

Streamlit runs independently. For Swagger and HazardNet image analysis, install `requirements.txt` and run `.\run.ps1` or `./run.sh` from the repository root. HazardNet uses `training/best.pt` when the checkpoint is available.

- `GET /api/planner/sites` returns the site catalogue with coordinate sources.
- `GET /api/planner/visibility?site_id=connecting_ridge&start=2026-11-15T00:00Z&end=2026-11-22T00:00Z` returns positions, the terrain profile, and visibility windows. Use `lat` and `lon` for custom coordinates. Use `step_minutes` and `horizon=auto|flat` to set the sampling interval and horizon model.
- `GET /api/sites/{site_id}/sun-earth` and `GET /api/compare` serve the archived Site A/B scenarios.

## Camera and AI

The **Camera & AI** tab supports camera capture, file uploads, and sample frames. For privacy reasons, this repository includes only part of the source code for the camera and AI features.

## Tests

```powershell
python -m unittest discover -s tests -v
```
