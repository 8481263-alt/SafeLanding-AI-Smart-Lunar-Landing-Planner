"""SafeLanding AI Streamlit entry point."""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent  # Repository root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # Make modules in the repository root importable.

from frontend.mission_planner import render_planner
from modules.image_processing import digitize_surface_photo

try:
    from safety_algorithm import analyze_landing_zones  # Team member 4's module.
    HAS_SAFETY = True
except Exception:  # Show detected hazards when landing-zone analysis is unavailable.
    HAS_SAFETY = False

DEFAULT_API = os.getenv("SAFELANDING_API", "http://localhost:8000")
RED, GREEN, AMBER = (235, 70, 70), (60, 220, 120), (240, 166, 58)

st.set_page_config(page_title="SafeLanding AI", page_icon="🌙", layout="wide")


# Use the team's presentation colors and fonts.
# Keep star positions stable across page reruns.
STARS_A = "1326px 308px 0 0.6px rgba(234,241,255,0.9),197px 148px 0 0.6px rgba(234,241,255,0.35),1497px 1193px 0 0.6px rgba(234,241,255,0.35),2078px 439px 0 0.6px rgba(234,241,255,0.35),352px 888px 0 0.6px rgba(234,241,255,0.9),286px 492px 0 0.6px rgba(234,241,255,0.35),2257px 869px 0 0.6px rgba(234,241,255,0.35),2316px 253px 0 0.6px rgba(234,241,255,0.5),2387px 126px 0 0.6px rgba(234,241,255,0.9),203px 452px 0 0.6px rgba(234,241,255,0.35),2280px 272px 0 0.6px rgba(234,241,255,0.7),1716px 295px 0 0.6px rgba(234,241,255,0.35),2338px 631px 0 0.6px rgba(234,241,255,0.5),422px 1191px 0 0.6px rgba(234,241,255,0.5),1525px 199px 0 0.6px rgba(234,241,255,0.35),2311px 122px 0 0.6px rgba(234,241,255,0.5),2033px 1393px 0 0.6px rgba(234,241,255,0.9),1286px 953px 0 0.6px rgba(234,241,255,0.9),1481px 613px 0 0.6px rgba(234,241,255,0.5),736px 1431px 0 0.6px rgba(234,241,255,0.5),335px 1176px 0 0.6px rgba(234,241,255,0.7),2151px 1013px 0 0.6px rgba(234,241,255,0.7),1838px 589px 0 0.6px rgba(234,241,255,0.35),483px 1048px 0 0.6px rgba(234,241,255,0.9),675px 1550px 0 0.6px rgba(234,241,255,0.7),622px 1001px 0 0.6px rgba(234,241,255,0.9),160px 1368px 0 0.6px rgba(234,241,255,0.35),2285px 1173px 0 0.6px rgba(234,241,255,0.7),1393px 1423px 0 0.6px rgba(234,241,255,0.7),2034px 1187px 0 0.6px rgba(234,241,255,0.9),281px 191px 0 0.6px rgba(234,241,255,0.7),1941px 1427px 0 0.6px rgba(234,241,255,0.35),248px 1497px 0 0.6px rgba(234,241,255,0.7),2367px 1395px 0 0.6px rgba(234,241,255,0.9),1165px 1467px 0 0.6px rgba(234,241,255,0.9),1421px 46px 0 0.6px rgba(234,241,255,0.9),1455px 344px 0 0.6px rgba(234,241,255,0.35),2022px 120px 0 0.6px rgba(234,241,255,0.5),1177px 264px 0 0.6px rgba(234,241,255,0.5),1629px 800px 0 0.6px rgba(234,241,255,0.9),330px 340px 0 0.6px rgba(234,241,255,0.9),1645px 1125px 0 0.6px rgba(234,241,255,0.7),560px 881px 0 0.6px rgba(234,241,255,0.7),1701px 734px 0 0.6px rgba(234,241,255,0.9),945px 309px 0 0.6px rgba(234,241,255,0.35),721px 309px 0 0.6px rgba(234,241,255,0.5),955px 24px 0 0.6px rgba(234,241,255,0.9),746px 538px 0 0.6px rgba(234,241,255,0.7),16px 298px 0 0.6px rgba(234,241,255,0.9),2189px 756px 0 0.6px rgba(234,241,255,0.7),514px 1414px 0 0.6px rgba(234,241,255,0.35),1870px 1597px 0 0.6px rgba(234,241,255,0.9),1630px 817px 0 0.6px rgba(234,241,255,0.9),424px 986px 0 0.6px rgba(234,241,255,0.9),254px 390px 0 0.6px rgba(234,241,255,0.35),855px 902px 0 0.6px rgba(234,241,255,0.5),450px 696px 0 0.6px rgba(234,241,255,0.35),419px 0px 0 0.6px rgba(234,241,255,0.5),2197px 207px 0 0.6px rgba(234,241,255,0.7),104px 144px 0 0.6px rgba(234,241,255,0.5),1541px 304px 0 0.6px rgba(234,241,255,0.7),1422px 1233px 0 0.6px rgba(234,241,255,0.7),1942px 251px 0 0.6px rgba(234,241,255,0.35),1999px 954px 0 0.6px rgba(234,241,255,0.9),1981px 638px 0 0.6px rgba(234,241,255,0.35),590px 209px 0 0.6px rgba(234,241,255,0.7),1084px 980px 0 0.6px rgba(234,241,255,0.5),2114px 47px 0 0.6px rgba(234,241,255,0.5),2163px 740px 0 0.6px rgba(234,241,255,0.5),2224px 55px 0 0.6px rgba(234,241,255,0.7),372px 1425px 0 0.6px rgba(234,241,255,0.7),2123px 751px 0 0.6px rgba(234,241,255,0.5),1456px 1580px 0 0.6px rgba(234,241,255,0.5),2181px 1109px 0 0.6px rgba(234,241,255,0.7),913px 1255px 0 0.6px rgba(234,241,255,0.5),980px 820px 0 0.6px rgba(234,241,255,0.5),818px 1060px 0 0.6px rgba(234,241,255,0.9),1456px 1497px 0 0.6px rgba(234,241,255,0.35),114px 572px 0 0.6px rgba(234,241,255,0.9),1061px 396px 0 0.6px rgba(234,241,255,0.7),1831px 1480px 0 0.6px rgba(234,241,255,0.7),1493px 164px 0 0.6px rgba(234,241,255,0.5),418px 464px 0 0.6px rgba(234,241,255,0.9),805px 691px 0 0.6px rgba(234,241,255,0.5),1976px 1278px 0 0.6px rgba(234,241,255,0.35),1963px 1337px 0 0.6px rgba(234,241,255,0.7),347px 1352px 0 0.6px rgba(234,241,255,0.35),1591px 1457px 0 0.6px rgba(234,241,255,0.5),1958px 365px 0 0.6px rgba(234,241,255,0.9),1361px 177px 0 0.6px rgba(234,241,255,0.9),1897px 822px 0 0.6px rgba(234,241,255,0.35),650px 348px 0 0.6px rgba(234,241,255,0.5),112px 309px 0 0.6px rgba(234,241,255,0.9),598px 1252px 0 0.6px rgba(234,241,255,0.9),1435px 319px 0 0.6px rgba(234,241,255,0.5),87px 29px 0 0.6px rgba(234,241,255,0.35),2156px 1534px 0 0.6px rgba(234,241,255,0.5),1776px 398px 0 0.6px rgba(234,241,255,0.5),114px 515px 0 0.6px rgba(234,241,255,0.5),1199px 1026px 0 0.6px rgba(234,241,255,0.5),1335px 531px 0 0.6px rgba(234,241,255,0.9),536px 124px 0 0.6px rgba(234,241,255,0.7),1876px 1356px 0 0.6px rgba(234,241,255,0.9),2054px 267px 0 0.6px rgba(234,241,255,0.5),2144px 1045px 0 0.6px rgba(234,241,255,0.35),1802px 1590px 0 0.6px rgba(234,241,255,0.5),16px 1589px 0 0.6px rgba(234,241,255,0.5),705px 289px 0 0.6px rgba(234,241,255,0.9),492px 1139px 0 0.6px rgba(234,241,255,0.35),1335px 1397px 0 0.6px rgba(234,241,255,0.9)"
STARS_B = "434px 1147px 0 1.0px rgba(234,241,255,0.35),1017px 391px 0 1.0px rgba(234,241,255,0.7),172px 1581px 0 1.0px rgba(234,241,255,0.35),2079px 926px 0 1.0px rgba(234,241,255,0.35),259px 907px 0 1.0px rgba(234,241,255,0.7),2070px 1241px 0 1.0px rgba(234,241,255,0.5),1135px 926px 0 1.0px rgba(234,241,255,0.9),2079px 507px 0 1.0px rgba(234,241,255,0.7),2291px 414px 0 1.0px rgba(234,241,255,0.9),561px 853px 0 1.0px rgba(234,241,255,0.35),1607px 905px 0 1.0px rgba(234,241,255,0.7),297px 1374px 0 1.0px rgba(234,241,255,0.5),1754px 149px 0 1.0px rgba(234,241,255,0.5),1240px 250px 0 1.0px rgba(234,241,255,0.5),1499px 292px 0 1.0px rgba(234,241,255,0.7),562px 957px 0 1.0px rgba(234,241,255,0.5),385px 815px 0 1.0px rgba(234,241,255,0.9),666px 1367px 0 1.0px rgba(234,241,255,0.5),661px 1446px 0 1.0px rgba(234,241,255,0.9),2111px 827px 0 1.0px rgba(234,241,255,0.7),1725px 400px 0 1.0px rgba(234,241,255,0.7),1304px 188px 0 1.0px rgba(234,241,255,0.7),79px 692px 0 1.0px rgba(234,241,255,0.9),1804px 1440px 0 1.0px rgba(234,241,255,0.35),1574px 678px 0 1.0px rgba(234,241,255,0.7),2098px 131px 0 1.0px rgba(234,241,255,0.35),936px 214px 0 1.0px rgba(234,241,255,0.35),1087px 556px 0 1.0px rgba(234,241,255,0.35),743px 553px 0 1.0px rgba(234,241,255,0.5),1729px 1384px 0 1.0px rgba(234,241,255,0.7),1662px 305px 0 1.0px rgba(234,241,255,0.9),1339px 183px 0 1.0px rgba(234,241,255,0.7),235px 1409px 0 1.0px rgba(234,241,255,0.5),1742px 148px 0 1.0px rgba(234,241,255,0.7),68px 1299px 0 1.0px rgba(234,241,255,0.35),1067px 171px 0 1.0px rgba(234,241,255,0.5),272px 541px 0 1.0px rgba(234,241,255,0.35),1858px 23px 0 1.0px rgba(234,241,255,0.7),2265px 855px 0 1.0px rgba(234,241,255,0.7),529px 88px 0 1.0px rgba(234,241,255,0.5),448px 330px 0 1.0px rgba(234,241,255,0.7),206px 370px 0 1.0px rgba(234,241,255,0.5),1277px 1287px 0 1.0px rgba(234,241,255,0.7),2175px 1555px 0 1.0px rgba(234,241,255,0.5),1187px 912px 0 1.0px rgba(234,241,255,0.5),1108px 710px 0 1.0px rgba(234,241,255,0.35),1025px 75px 0 1.0px rgba(234,241,255,0.35),75px 1501px 0 1.0px rgba(234,241,255,0.5),2106px 972px 0 1.0px rgba(234,241,255,0.5),1831px 217px 0 1.0px rgba(234,241,255,0.9),2027px 1118px 0 1.0px rgba(234,241,255,0.9),2075px 630px 0 1.0px rgba(234,241,255,0.5),940px 701px 0 1.0px rgba(234,241,255,0.5),572px 828px 0 1.0px rgba(234,241,255,0.7),222px 265px 0 1.0px rgba(234,241,255,0.35),289px 1280px 0 1.0px rgba(234,241,255,0.7),1764px 334px 0 1.0px rgba(234,241,255,0.35),346px 1362px 0 1.0px rgba(234,241,255,0.9),2072px 1373px 0 1.0px rgba(234,241,255,0.7),992px 1418px 0 1.0px rgba(234,241,255,0.7)"
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Vollkorn:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
:root{--sl-bg:#020B18;--sl-panel:rgba(10,26,48,.82);--sl-line:#1C3350;--sl-text:#EAF1FF;--sl-muted:#8FA6C8;
      --sl-cream:#FDFDE8;--sl-blue:#4A9BE8;--sl-sky:#A4C6F2;--sl-bad:#F08A8A;
      --sl-display:'Vollkorn',Georgia,'Times New Roman',serif;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stSidebar"]{font-family:'Inter',system-ui,sans-serif;}
[data-testid="stAppDeployButton"],[data-testid="stMainMenu"]{display:none;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stMainBlockContainer"]{padding-top:1.6rem;max-width:1320px;}
[data-testid="stSidebar"]{border-right:1px solid var(--sl-line);}

/* Two slowly twinkling layers of stars. */
[data-testid="stAppViewContainer"]{background:radial-gradient(1200px 700px at 78% -10%,#0B2A52 0%,rgba(2,11,24,0) 60%),var(--sl-bg);}
[data-testid="stAppViewContainer"]::before,[data-testid="stAppViewContainer"]::after{content:"";position:fixed;top:0;left:0;
  width:1px;height:1px;border-radius:50%;pointer-events:none;z-index:0;}
[data-testid="stAppViewContainer"]::before{box-shadow:__STARS_A__;animation:sl-twinkle 5s ease-in-out infinite alternate;}
[data-testid="stAppViewContainer"]::after{box-shadow:__STARS_B__;animation:sl-twinkle 8s ease-in-out -3s infinite alternate;}
@keyframes sl-twinkle{from{opacity:.35}to{opacity:1}}
[data-testid="stMain"]{position:relative;z-index:1;background:transparent;}

[data-testid="stMain"] h1,[data-testid="stMain"] h2,[data-testid="stMain"] h3{font-family:var(--sl-display) !important;letter-spacing:0;}
[data-testid="stMain"] h3{font-size:1.6rem;font-weight:600;margin-top:1.8rem;color:var(--sl-sky) !important;}

/* Landing video header. */
.sl-hero{position:relative;overflow:hidden;border-radius:18px;border:1px solid var(--sl-line);margin-bottom:1.2rem;
  min-height:clamp(260px,34vw,420px);display:flex;align-items:flex-end;background:#000;}
.sl-hero video{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;}
.sl-hero::after{content:"";position:absolute;inset:0;
  background:linear-gradient(90deg,rgba(2,11,24,.92) 0%,rgba(2,11,24,.55) 38%,rgba(2,11,24,0) 65%);}
.sl-hero-text{position:relative;z-index:2;padding:clamp(1.2rem,3vw,2.4rem);max-width:34rem;}
[data-testid="stMain"] .sl-hero h1{color:var(--sl-cream) !important;}
.sl-hero h1{font-size:clamp(2.4rem,5.2vw,4.2rem);font-weight:700;line-height:1;margin:0;padding:0;color:var(--sl-cream);}
.sl-hero p{margin:.8rem 0 0;font-size:1.08rem;line-height:1.5;color:var(--sl-sky);}
.sl-shoot{position:absolute;top:14%;left:-12%;width:130px;height:2px;z-index:1;border-radius:2px;opacity:0;
  background:linear-gradient(90deg,rgba(253,253,232,0),#FDFDE8);transform:rotate(-14deg);
  animation:sl-shoot 11s ease-in 2s infinite;}
@keyframes sl-shoot{0%{left:-12%;top:30%;opacity:0}3%{opacity:.9}14%{left:108%;top:2%;opacity:0}100%{left:108%;top:2%;opacity:0}}
@media (prefers-reduced-motion:reduce){.sl-shoot{display:none}
  [data-testid="stAppViewContainer"]::before,[data-testid="stAppViewContainer"]::after{animation:none}}

[data-testid="stTabs"] [role="tablist"]{gap:.6rem !important;border-bottom:none !important;padding-bottom:.4rem;}
[data-testid="stTabs"] [role="tab"]{height:auto !important;padding:.6rem 1.2rem !important;border:1px solid var(--sl-line) !important;
  border-radius:999px !important;background:rgba(10,26,48,.6) !important;color:var(--sl-muted) !important;}
[data-testid="stTabs"] [role="tab"] p{font-size:.95rem;font-weight:500;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{background:var(--sl-blue) !important;border-color:var(--sl-blue) !important;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p{color:#021526;font-weight:600;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none !important;}
[data-testid="stTabs"] [role="tab"]:focus-visible{outline:2px solid var(--sl-cream);outline-offset:2px;}

[data-testid="stMetric"]{background:var(--sl-panel);border:1px solid var(--sl-line);border-radius:14px;padding:.9rem 1rem;}
[data-testid="stMetricValue"]{font-family:var(--sl-display);font-weight:600;font-size:1.9rem;color:var(--sl-cream);}

.sl-verdict{margin-top:1rem;padding:1rem 1.2rem;border-radius:14px;border:1px solid var(--sl-blue);
  background:rgba(74,155,232,.1);font-family:var(--sl-display);font-size:1.2rem;color:var(--sl-cream);}
.sl-zone{background:var(--sl-panel);border:1px solid var(--sl-line);border-left:4px solid var(--c);border-radius:14px;
  padding:.7rem .9rem;margin-bottom:.6rem;}
.sl-zone b{font-family:var(--sl-display);font-size:1.15rem;color:var(--c);}
.sl-zone .tag{float:right;font-size:.72rem;font-weight:600;padding:.1rem .5rem;border-radius:999px;
  background:rgba(60,220,120,.16);color:#7BE3A6;}
.sl-zone dl{display:grid;grid-template-columns:auto 1fr;gap:.1rem .7rem;margin:.35rem 0 0;font-size:.9rem;}
.sl-zone dt{color:var(--sl-muted);} .sl-zone dd{margin:0;color:var(--sl-text);font-variant-numeric:tabular-nums;}
.sl-verdict.bad{border-color:var(--sl-bad);background:rgba(240,138,138,.1);}
.sl-note{font-size:.85rem;color:var(--sl-muted);margin:.2rem 0 .6rem;}
</style>
""".replace("__STARS_A__", STARS_A).replace("__STARS_B__", STARS_B)
st.markdown(CSS, unsafe_allow_html=True)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def num(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}"


# Backend requests.
@st.cache_data(ttl=10, show_spinner=False)
def api_health(api: str) -> dict | None:
    try:
        r = requests.get(f"{api}/health", timeout=2)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def vision_api(api: str, data: bytes, name: str) -> dict:
    """Request a VisionResult from HazardNet through the backend."""
    r = requests.post(f"{api}/api/vision/analyze",
                      files={"file": (name, data, "application/octet-stream")}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


def vision_builtin(img: Image.Image) -> dict:
    """Find strong local contrast, such as shadows and sunlit crater slopes.

    This detector supports the demo when the HazardNet backend is unavailable.
    """
    t0 = time.perf_counter()
    w, h = img.size
    k = 320 / max(w, h)  # Resize the frame to speed up detection.
    small = img.convert("L").resize((max(1, round(w * k)), max(1, round(h * k))), Image.Resampling.BILINEAR)
    g = np.asarray(small, dtype=np.float32)
    dev = np.abs(g - ndimage.gaussian_filter(g, 24))  # Difference from the local background.
    mask = dev > max(18.0, 1.6 * float(dev.std()))
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=6))  # Join a shadow and adjoining slope.
    labels, _ = ndimage.label(mask)
    hazards = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        blob = labels[sl] == i
        if blob.sum() < 25:
            continue
        ys, xs = sl
        conf = float(np.clip(dev[sl][blob].mean() / 110, 0.3, 0.99))
        hazards.append({"bbox": [xs.start / k, ys.start / k, min(w, xs.stop / k), min(h, ys.stop / k)],
                        "cls": "hazard", "confidence": conf})
    return {"hazards": hazards, "image_width": w, "image_height": h,
            "inference_ms": (time.perf_counter() - t0) * 1000, "device": "built-in"}


# Frame overlays.
def _font(px: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", px)
    except OSError:
        try:
            return ImageFont.truetype("arialbd.ttf", px)      # Windows
        except OSError:
            return ImageFont.load_default()


def keep_out_mask(res: dict, radius_px: float, margin_px: float, scale: float) -> np.ndarray:
    """Approximate where the lander centre is too close to a hazard or frame edge.

    Use a reduced grid for display. The safety algorithm computes landing zones.
    """
    w, h = res["image_width"], res["image_height"]
    gw, gh = max(1, round(w * scale)), max(1, round(h * scale))
    free = np.ones((gh, gw), dtype=bool)
    for hz in res["hazards"]:
        x1, y1, x2, y2 = (v * scale for v in hz["bbox"])
        free[int(max(0, y1)):int(np.ceil(y2)), int(max(0, x1)):int(np.ceil(x2))] = False
    dist = ndimage.distance_transform_edt(free) if not free.all() else np.full((gh, gw), 1e9)
    blocked = dist < (radius_px + margin_px) * scale
    r = int(np.ceil(radius_px * scale))
    if r > 0:
        blocked[:r, :] = blocked[-r:, :] = True
        blocked[:, :r] = blocked[:, -r:] = True
    return blocked


def draw_overlay(img: Image.Image, res: dict, zones: dict | None, radius_px: float, margin_px: float,
                 step: int) -> Image.Image:
    """Draw the frame, then add hazards, keep-out areas, and zones as step increases."""
    k = max(1.0, 1000 / img.width)  # Enlarge small frames for readable annotations.
    out = img.convert("RGB").resize((round(img.width * k), round(img.height * k)), Image.Resampling.LANCZOS)
    if step >= 2:
        blocked = keep_out_mask(res, radius_px, margin_px, min(1.0, 400 / max(img.size)))
        alpha = Image.fromarray((blocked * 90).astype(np.uint8)).resize(out.size, Image.Resampling.BILINEAR)
        out = Image.composite(Image.new("RGB", out.size, (235, 70, 70)), out, alpha)
    d = ImageDraw.Draw(out)
    lw, font = max(2, out.width // 300), _font(max(14, out.width // 55))
    if step >= 1:
        for h in res["hazards"]:
            x1, y1, x2, y2 = (v * k for v in h["bbox"])
            d.rectangle([x1, y1, x2, y2], outline=RED, width=lw)
            d.text((x1 + 4, max(0, y1 - font.size - 4)), f'{h["confidence"]:.2f}', fill=RED, font=font)
    if step >= 3 and zones and zones.get("candidates"):
        for i, c in reversed(list(enumerate(zones["candidates"]))):  # Draw the top-ranked zone last.
            x, y, r = c["x"] * k, c["y"] * k, radius_px * k
            color = GREEN if i == 0 else AMBER
            d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=lw + (2 if i == 0 else 0))
            d.text((x, y), str(i + 1), fill=color, font=font, anchor="mm")
    return out


def find_zones(res: dict, radius_px: float, margin_px: float) -> dict:
    """Clip bounding boxes to the frame before calling the landing-zone module."""
    w, h = res["image_width"], res["image_height"]
    hazards = []
    for hz in res["hazards"]:
        x1, y1, x2, y2 = hz["bbox"]
        x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
        y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
        if x2 > x1 and y2 > y1:
            hazards.append({"bbox_xyxy": [x1, y1, x2, y2], "type": hz.get("cls") or "hazard",
                            "confidence": max(0.0, min(1.0, hz["confidence"]))})
    return analyze_landing_zones({"image_width": w, "image_height": h, "hazards": hazards,
                                  "landing_radius_px": radius_px, "safety_margin_px": margin_px,
                                  "top_n": 3})


# Header and navigation.
html('<div class="sl-hero"><video autoplay muted loop playsinline preload="auto" src="app/static/landing.mp4"></video>'
     '<span class="sl-shoot"></span><div class="sl-hero-text"><h1>SafeLanding AI</h1>'
     '<p>Plan sunlight and direct Earth visibility on the Moon.</p></div></div>')
page = st.radio("View", ["Mission planner", "Camera & AI"],
                horizontal=True, label_visibility="collapsed")
if page == "Mission planner":
    render_planner()
    st.stop()

# Camera and AI hazard detection.
SAMPLES = {"high_sun": "High Sun", "low_sun": "Low Sun, long shadows", "dense_craters": "Dense crater field"}
SAMPLE_DIR = Path(__file__).resolve().parent / "static" / "samples"
STEPS = ["Digitized frame", "Hazards", "Keep-out area", "Landing zones"]

if page == "Camera & AI":
    st.info("For privacy reasons, this repository includes only part of the source code for the camera and AI features.")
    api_url = DEFAULT_API.rstrip("/")
    health = None if os.getenv("SAFELANDING_OFFLINE", "0") == "1" else api_health(api_url)
    API = api_url if health else None

    # Frame sources include samples, file uploads, and the camera.
    st.markdown("**Pick a sample frame or add your own**")
    cols = st.columns([1, 1, 1, 2])
    for col, (key, title) in zip(cols, SAMPLES.items()):
        path = SAMPLE_DIR / f"{key}.jpg"
        if path.is_file():
            col.image(str(path), width="stretch", alt=f"{title} synthetic lunar terrain sample")
            chosen_now = st.session_state.get("sample") == key
            if col.button(title, key=f"sample_{key}", width="stretch", type="primary" if chosen_now else "secondary"):
                st.session_state["sample"] = key
                st.rerun()  # Highlight the newly selected sample.
    with cols[3]:
        up = st.file_uploader("Surface image or a photo of the 3D-printed terrain", type=["png", "jpg", "jpeg", "bmp"])
        if st.toggle("Take a frame with this computer's camera", False):
            up = st.camera_input("Frame") or up
    html('<p class="sl-note">Sample frames are synthetic terrain renders made for this demo, not lunar photographs.</p>')

    # Keep scale and lander dimensions in a collapsible panel.
    frame_cm = st.session_state.get("frame_cm", 40.0)
    lander_cm = st.session_state.get("lander_cm", 11.0)
    margin_cm = st.session_state.get("margin_cm", 2.0)
    with st.expander(f"Scale and lander size: frame {num(frame_cm, 0)} cm wide, lander {num(lander_cm, 0)} cm, "
                     f"margin {num(margin_cm, 0)} cm"):
        s1, s2, s3 = st.columns(3)
        s1.number_input("Surface width in the frame, cm", 5.0, 500.0, 40.0, 1.0, key="frame_cm",
                        help="The printed terrain is 40 × 40 cm. If the whole model fits the frame, keep 40.")
        s2.number_input("Lander footprint diameter, cm", 1.0, 100.0, 11.0, 0.5, key="lander_cm",
                        help="The lander body is 110 × 110 mm")
        s3.number_input("Safety margin around hazards, cm", 0.0, 50.0, 2.0, 0.5, key="margin_cm")

    # Uploaded frames take priority over samples.
    if up is not None:
        data, name = up.getvalue(), getattr(up, "name", "frame.png")
    elif st.session_state.get("sample") in SAMPLES and (SAMPLE_DIR / f"{st.session_state['sample']}.jpg").is_file():
        name = f"{st.session_state['sample']}.jpg"
        data = (SAMPLE_DIR / name).read_bytes()
    else:
        data = None

    if data is None:
        st.info("Choose a sample frame above or upload an image. The detector marks hazards and the algorithm "
                "proposes landing zones.")
    else:
        try:
            with Image.open(io.BytesIO(data)) as uploaded_image:
                original_img, img = digitize_surface_photo(uploaded_image)
        except (OSError, ValueError) as exc:
            st.error(f"Could not read the selected image: {exc}")
            st.stop()

        digitized_buffer = io.BytesIO()
        img.save(digitized_buffer, format="PNG", optimize=True)
        digitized_data = digitized_buffer.getvalue()
        digitized_name = f"{Path(name).stem or 'surface'}_digitized.png"
        st.caption(
            "The image is converted to grayscale, uneven lighting is normalized, and local contrast is enhanced "
            "before detection. This does not reconstruct 3D terrain."
        )
        with st.expander("Original and digitized photo", expanded=False):
            original_col, digitized_col = st.columns(2)
            original_col.image(
                original_img,
                caption="Original photo",
                alt="Original surface photo before image processing",
                width="stretch",
            )
            digitized_col.image(
                img,
                caption="Digitized photo used for detection",
                alt="Grayscale surface photo with normalized lighting and enhanced crater contrast",
                width="stretch",
            )
            st.download_button(
                "Download digitized photo",
                data=digitized_data,
                file_name=digitized_name,
                mime="image/png",
                width="stretch",
            )

        res, source_note = None, None
        if API and not (health or {}).get("fallback"):
            try:
                with st.spinner("HazardNet is analysing the frame…"):
                    res = vision_api(API, digitized_data, digitized_name)
            except Exception as e:
                source_note = f"HazardNet backend did not respond ({e}). "
        if res is None:  # Use the built-in detector when the backend is unavailable.
            res = vision_builtin(img)
            st.warning((source_note or "HazardNet backend is not connected. ") +
                       "Hazards below come from the **built-in contrast detector**, a demo fallback, not the neural network.")

        px_per_cm = res["image_width"] / frame_cm
        radius_px, margin_px = lander_cm / 2 * px_per_cm, margin_cm * px_per_cm
        zones, zone_err = None, None
        if HAS_SAFETY:
            try:
                zones = find_zones(res, radius_px, margin_px)
            except Exception as e:
                zone_err = str(e)

        # Show each analysis step.
        step_name = st.segmented_control("Show", STEPS, default=STEPS[-1], key="ai_step") or STEPS[-1]
        step = STEPS.index(step_name)

        pic, info = st.columns([3, 2])
        captions = ["Digitized image with normalized lighting and enhanced local contrast.",
                    "Red boxes are detected hazards; the number is the detector's confidence.",
                    "Red tint: the lander centre cannot be here, a hazard or the frame edge is too close.",
                    "Green circle is the recommended landing zone, amber circles are alternatives."]
        pic.image(draw_overlay(img, res, zones, radius_px, margin_px, step), width="stretch", caption=captions[step])
        with info:
            # Landing verdict and candidate zone cards.
            if zones and zones["status"] == "ok":
                n = len(zones["candidates"])
                html(f'<div class="sl-verdict" style="margin-top:0"><b>Landing is possible.</b> '
                     f'{n} candidate zone{"s" if n != 1 else ""} found.</div>')
                cards = ""
                for i, c in enumerate(zones["candidates"]):
                    color = "#3CDC78" if i == 0 else "#F0A63A"
                    cards += (f'<div class="sl-zone" style="--c:{color}"><b>Zone {i + 1}</b>'
                              + ('<span class="tag">recommended</span>' if i == 0 else "")
                              + f'<dl><dt>Centre</dt><dd>{num(c["x"] / px_per_cm)} cm right, {num(c["y"] / px_per_cm)} cm down</dd>'
                                f'<dt>Clearance</dt><dd>{num(c["clearance_px"] / px_per_cm)} cm to the nearest hazard or edge</dd>'
                                f'<dt>Rank score</dt><dd>{num(c["score"], 0)} / 100</dd></dl></div>')
                html('<div style="height:.7rem"></div>' + cards)
            elif zones:
                html('<div class="sl-verdict bad" style="margin-top:0"><b>No safe zone.</b> The lander with this '
                     'margin does not fit anywhere in the frame. Try a smaller margin or another frame.</div>')
            elif zone_err:
                st.error(f"Zone algorithm error: {zone_err}")
            else:
                st.info("The `safety_algorithm` module was not found, so only hazards are shown. "
                        "Copy the `safety_algorithm/` folder into the repository root.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Hazards", len(res["hazards"]))
            m2.metric("Frame time", f'{res["inference_ms"]:.0f} ms')
            m3.metric("Speed", f'{1000 / max(res["inference_ms"], 1e-6):.0f} fps')
            detector = "built-in contrast detector" if res["device"] == "built-in" else f'HazardNet on {res["device"].upper()}'
            st.caption(f'Detector: {detector} · frame {res["image_width"]}×{res["image_height"]} px')
        with st.expander("Limitations"):
            st.markdown("- Centimetres are correct only if the camera looks straight down and the surface width "
                        "in the frame is set correctly.\n"
                        "- This is a single 2D frame estimate: surface slope and hazards the detector missed are not considered.\n"
                        "- The rank score orders zones within one frame; it is not a probability of a successful landing.\n"
                        "- The keep-out tint is an approximation for display; zones are computed by the safety algorithm.")
            if zones:
                st.caption("safety_algorithm warnings: " + " · ".join(zones["warnings"]))
