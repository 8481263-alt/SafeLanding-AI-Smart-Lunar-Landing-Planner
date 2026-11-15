"""SafeLanding AI — frontend (Streamlit), Людина 5. Інтерфейс сайту англійською.

Працює з реальним FastAPI з репозиторію команди (main.py):
    GET  /health
    GET  /api/sites/{site_id}/sun-earth?start=YYYY-MM-DD&end=YYYY-MM-DD   -> SiteReport
    POST /api/vision/analyze  (multipart, поле "file")                     -> VisionResult

Файл лежить у <корінь репозиторію>/frontend/app.py.
Якщо API недоступний, сторінка бере збережені дані з <корінь>/fallback/*.json
і позначає це в підписі під таблицею. Вкладка AI без backend використовує вбудований запасний детектор.

Вибір посадкової зони рахує модуль Дмитра (safety_algorithm), якщо він встановлений
або його папка safety_algorithm/ лежить у корені репозиторію.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent          # корінь репозиторію
FALLBACK_DIR = ROOT / "fallback"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))                      # щоб знайти safety_algorithm/ у корені

from modules.image_processing import digitize_surface_photo

try:
    from safety_algorithm import analyze_landing_zones  # модуль Людини 4
    HAS_SAFETY = True
except Exception:                                       # не встановлено — покажемо лише небезпеки
    HAS_SAFETY = False

DEFAULT_API = os.getenv("SAFELANDING_API", "http://localhost:8000")
SITES = {"site_a": "Site A", "site_b": "Site B"}
COLORS = {"site_a": "#F3E2A0", "site_b": "#4A9BE8"}   # A — місячне світло, B — синій
DATA_START, DATA_END = date(2026, 11, 14), date(2026, 11, 21)   # період, який порахувала Людина 3
PLAY_STEP, PLAY_DELAY = 8, 0.25      # програвання: крок 8 точок (4 години), пауза 0,25 с
RED, GREEN, AMBER = (235, 70, 70), (60, 220, 120), (240, 166, 58)

st.set_page_config(page_title="SafeLanding AI", page_icon="🌙", layout="wide")


# ───────────────────────── Оформлення (кольори й шрифти з презентації команди) ─────────────────────────
# Координати зірок згенеровано один раз (фіксовані, щоб небо не "стрибало" при оновленні сторінки)
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

/* зоряне небо: два шари зірок, що повільно мерехтять */
[data-testid="stAppViewContainer"]{background:radial-gradient(1200px 700px at 78% -10%,#0B2A52 0%,rgba(2,11,24,0) 60%),var(--sl-bg);}
[data-testid="stAppViewContainer"]::before,[data-testid="stAppViewContainer"]::after{content:"";position:fixed;top:0;left:0;
  width:1px;height:1px;border-radius:50%;pointer-events:none;z-index:0;}
[data-testid="stAppViewContainer"]::before{box-shadow:__STARS_A__;animation:sl-twinkle 5s ease-in-out infinite alternate;}
[data-testid="stAppViewContainer"]::after{box-shadow:__STARS_B__;animation:sl-twinkle 8s ease-in-out -3s infinite alternate;}
@keyframes sl-twinkle{from{opacity:.35}to{opacity:1}}
[data-testid="stMain"]{position:relative;z-index:1;background:transparent;}

[data-testid="stMain"] h1,[data-testid="stMain"] h2,[data-testid="stMain"] h3{font-family:var(--sl-display) !important;letter-spacing:0;}
[data-testid="stMain"] h3{font-size:1.6rem;font-weight:600;margin-top:1.8rem;color:var(--sl-sky) !important;}

/* шапка з відео посадки */
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

.sl-site{background:var(--sl-panel);border:1px solid var(--sl-line);border-left:4px solid var(--c);border-radius:14px;
  padding:.85rem 1rem;margin-bottom:.7rem;}
.sl-site-name{font-family:var(--sl-display);font-size:1.15rem;font-weight:600;color:var(--c);margin-bottom:.35rem;}
.sl-site-grid{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;}
.sl-lab{display:block;font-size:.8rem;color:var(--sl-muted);}
.sl-val{font-family:var(--sl-display);font-weight:600;font-size:1.7rem;font-variant-numeric:tabular-nums lining-nums;
  color:var(--sl-cream);margin-right:.5rem;}
.sl-ico{display:inline-block;width:.95rem;height:.95rem;border-radius:50%;margin-right:.4rem;vertical-align:-.08rem;
  background:#3A4A66;}
.sl-ico.sun.on{background:#FFD76A;box-shadow:0 0 10px 3px rgba(255,215,106,.55);animation:sl-glow 2.6s ease-in-out infinite alternate;}
.sl-ico.earth.on{background:radial-gradient(circle at 35% 35%,#A4C6F2,#4A9BE8 60%,#1E4E8C);box-shadow:0 0 10px 3px rgba(74,155,232,.55);
  animation:sl-glow 2.6s ease-in-out infinite alternate;}
@keyframes sl-glow{from{filter:brightness(.9)}to{filter:brightness(1.25)}}
.sl-site{transition:transform .15s ease,border-color .15s ease;}
.sl-site:hover{transform:translateY(-2px);border-color:var(--c);}
@media (prefers-reduced-motion:reduce){.sl-ico{animation:none !important}.sl-site{transition:none}}
.sl-pill{display:inline-block;font-size:.75rem;font-weight:600;padding:.12rem .55rem;border-radius:999px;vertical-align:middle;}
.sl-pill.ok{background:rgba(74,155,232,.2);color:var(--sl-sky);}
.sl-pill.no{background:rgba(240,138,138,.16);color:var(--sl-bad);}

.sl-ribbon{background:var(--sl-panel);border:1px solid var(--sl-line);border-radius:14px;padding:1rem 1.1rem .6rem;margin:.4rem 0 1rem;}
.sl-rib-title{font-weight:600;margin:0 0 .5rem;color:var(--sl-text);}
.sl-rib-title span{font-weight:400;color:var(--sl-muted);}
.sl-rib-row{display:grid;grid-template-columns:5.5rem 1fr 5.5rem;align-items:center;gap:.8rem;margin:.35rem 0;}
.sl-rib-name{font-size:.9rem;color:var(--sl-muted);}
.sl-rib-val{font-size:.9rem;font-variant-numeric:tabular-nums;text-align:right;color:var(--sl-text);}
.sl-track{position:relative;height:18px;border-radius:5px;background:#0F2542;overflow:hidden;}
.sl-track i{position:absolute;top:0;bottom:0;background:var(--c);}
.sl-track b{position:absolute;top:0;bottom:0;width:2px;background:#fff;}
.sl-days{position:relative;height:1.2rem;margin:.1rem 6.3rem 0 6.3rem;font-size:.75rem;color:var(--sl-muted);}
.sl-days span{position:absolute;transform:translateX(-50%);white-space:nowrap;}
.sl-gap{height:.6rem;}

.sl-table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;background:var(--sl-panel);}
.sl-table th,.sl-table td{padding:.7rem .9rem;border:none;border-bottom:1px solid var(--sl-line);text-align:right;}
.sl-table th:first-child,.sl-table td:first-child{text-align:left;}
.sl-table th{font-weight:600;font-size:.9rem;color:var(--sl-muted);}
.sl-table td:first-child{color:var(--sl-muted);}
.sl-table td{font-size:1.05rem;color:var(--sl-text);}
.sl-table td.best{color:var(--sl-cream);font-weight:600;background:rgba(74,155,232,.14);}
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
@media (max-width:700px){.sl-site-grid{grid-template-columns:1fr;}.sl-rib-row{grid-template-columns:4rem 1fr 4rem;}}
</style>
""".replace("__STARS_A__", STARS_A).replace("__STARS_B__", STARS_B)
st.markdown(CSS, unsafe_allow_html=True)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def num(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}"


# ───────────────────────── Доступ до даних ─────────────────────────
@st.cache_data(ttl=10, show_spinner=False)
def api_health(api: str) -> dict | None:
    try:
        r = requests.get(f"{api}/health", timeout=2)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def site_report(api: str | None, site_id: str, start: str, end: str) -> dict:
    """SiteReport з API або зі збереженого fallback-файлу (api=None)."""
    if api:
        r = requests.get(f"{api}/api/sites/{site_id}/sun-earth",
                         params={"start": start, "end": end}, timeout=15)
        if r.status_code != 200:
            raise RuntimeError(r.json().get("detail", r.text) if r.content else r.reason)
        return r.json()
    rep = json.loads((FALLBACK_DIR / f"sun_earth_{site_id}.json").read_text(encoding="utf-8"))
    end_full = end + "T23:59:59Z"
    rep["series"] = [p for p in rep["series"] if start <= p["time_utc"] <= end_full]
    if not rep["series"]:
        raise RuntimeError(f"Saved data has no samples for {start}..{end}")
    return rep


def vision_api(api: str, data: bytes, name: str) -> dict:
    """VisionResult від нейромережі HazardNet через backend."""
    r = requests.post(f"{api}/api/vision/analyze",
                      files={"file": (name, data, "application/octet-stream")}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


def vision_builtin(img: Image.Image) -> dict:
    """Запасний детектор без нейромережі: шукає ділянки, що сильно темніші або світліші за фон
    (тіні й освітлені схили кратерів). Це НЕ HazardNet — лише щоб демо працювало без backend."""
    t0 = time.perf_counter()
    w, h = img.size
    k = 320 / max(w, h)                                   # рахуємо на зменшеній копії — так швидше
    small = img.convert("L").resize((max(1, round(w * k)), max(1, round(h * k))), Image.Resampling.BILINEAR)
    g = np.asarray(small, dtype=np.float32)
    dev = np.abs(g - ndimage.gaussian_filter(g, 24))      # відхилення від місцевого фону
    mask = dev > max(18.0, 1.6 * float(dev.std()))
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=6))   # тінь і світлий схил — одна пляма
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


# ───────────────────────── Розрахунки для таблиці ─────────────────────────
def to_frame(rep: dict) -> pd.DataFrame:
    df = pd.DataFrame(rep["series"])
    df["time"] = pd.to_datetime(df["time_utc"], utc=True)
    return df


def windows(df: pd.DataFrame, flag: str, elev: str | None) -> list[tuple[float, float]]:
    """Вікна видимості в годинах. Межі — лінійна інтерполяція висоти до 0° між сусідніми точками
    (як у модулі Людини 3 для горизонту 0°); якщо горизонт не нульовий — середина інтервалу."""
    t = ((df["time"] - df["time"].iat[0]).dt.total_seconds() / 3600).tolist()   # години від початку
    f = df[flag].astype(bool).tolist()
    out, start = [], (t[0] if f[0] else None)
    for i in range(len(t) - 1):
        if f[i] == f[i + 1]:
            continue
        frac = 0.5
        if elev:
            e0, e1 = df[elev].iat[i], df[elev].iat[i + 1]
            if e0 * e1 < 0:
                frac = abs(e0) / (abs(e0) + abs(e1))
        cross = t[i] + frac * (t[i + 1] - t[i])
        if f[i]:
            out.append((start, cross))
            start = None
        else:
            start = cross
    if start is not None:
        out.append((start, t[-1]))
    return out


def total_and_gap(wins: list[tuple[float, float]], t0: float, t1: float) -> tuple[float, float]:
    """(сумарні години видимості, найдовша перерва без видимості)."""
    edges = [t0] + [x for w in wins for x in w] + [t1]
    gaps = [edges[i + 1] - edges[i] for i in range(0, len(edges), 2)]
    return sum(e - s for s, e in wins), max(gaps)


def intersect(a: list, b: list) -> float:
    return sum(max(0.0, min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b)


def site_metrics(rep: dict) -> dict:
    df = to_frame(rep)
    t0, t1 = 0.0, (df["time"].iat[-1] - df["time"].iat[0]).total_seconds() / 3600
    flat = "ZERO" in rep["horizon_assumption"].upper()
    sun = windows(df, "sun_visible", "sun_elevation_deg" if flat else None)
    earth = windows(df, "earth_visible", "earth_elevation_deg" if flat else None)
    sun_h, dark = total_and_gap(sun, t0, t1)
    earth_h, outage = total_and_gap(earth, t0, t1)
    return {"period": t1 - t0, "sun_h": sun_h, "earth_h": earth_h, "dark": dark,
            "outage": outage, "both_h": intersect(sun, earth)}



def ribbon(frames: dict, reports: dict, moment) -> str:
    """Стрічка часу: коли Сонце й Земля над горизонтом у кожній точці; біла риска — обраний момент."""
    first = next(iter(frames.values()))
    t0, t1 = first["time"].iat[0], first["time"].iat[-1]
    period = (t1 - t0).total_seconds() / 3600
    now = 100 * (moment - t0).total_seconds() / 3600 / period
    out = ['<div class="sl-ribbon">']
    for flag, elev, title, note in (
            ("sun_visible", "sun_elevation_deg", "Sun above the horizon", "power for solar panels"),
            ("earth_visible", "earth_elevation_deg", "Earth above the horizon", "direct link possible")):
        out.append(f'<p class="sl-rib-title">{title} <span>— {note}</span></p>')
        for sid, df in frames.items():
            flat = "ZERO" in reports[sid]["horizon_assumption"].upper()
            wins = windows(df, flag, elev if flat else None)
            segs = "".join(f'<i style="left:{100 * a / period:.3f}%;width:{100 * (b - a) / period:.3f}%"></i>'
                           for a, b in wins)
            total = sum(b - a for a, b in wins)
            out.append(f'<div class="sl-rib-row" style="--c:{COLORS[sid]}"><span class="sl-rib-name">{SITES[sid]}</span>'
                       f'<div class="sl-track">{segs}<b style="left:{now:.3f}%"></b></div>'
                       f'<span class="sl-rib-val">{num(total)} h</span></div>')
        out.append('<div class="sl-gap"></div>')
    days = pd.date_range(t0.normalize(), t1, freq="D")
    ticks = "".join(f'<span style="left:{100 * (d - t0).total_seconds() / 3600 / period:.3f}%">{d.strftime("%d %b")}</span>'
                    for d in days if t0 <= d <= t1)
    out.append(f'<div class="sl-days">{ticks}</div></div>')
    return "".join(out)

# ───────────────────────── Графіки ─────────────────────────
def style(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=height, title=title,
                      font=dict(family="Inter, system-ui, sans-serif", color="#EAF1FF"),
                      title_font=dict(family="Vollkorn, Georgia, serif", size=20, color="#A4C6F2"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def elevation_chart(frames: dict, col: str, title: str, moment) -> go.Figure:
    """Висота світила над горизонтом. Заливка — час, коли воно над горизонтом; темна смуга — під ним."""
    fig = go.Figure()
    lo = min(float(df[col].min()) for df in frames.values())
    if lo < 0:
        fig.add_hrect(y0=lo * 1.08, y1=0, fillcolor="rgba(240,138,138,0.07)", line_width=0,
                      annotation_text="below the horizon", annotation_position="bottom right",
                      annotation_font_color="#F08A8A")
    for sid, df in frames.items():
        fig.add_trace(go.Scatter(x=df["time"], y=df[col].clip(lower=0), mode="lines", line=dict(width=0),
                                 fill="tozeroy", fillcolor=rgba(COLORS[sid], 0.16),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=df["time"], y=df[col], name=SITES[sid], mode="lines",
                                 line=dict(color=COLORS[sid], width=2.5),
                                 hovertemplate="%{x|%d %b %H:%M} UTC<br>%{y:.2f}°<extra>" + SITES[sid] + "</extra>"))
        now = df.loc[df["time"] == moment]
        if len(now):                                    # точка на кривій в обраний момент
            fig.add_trace(go.Scatter(x=now["time"], y=now[col], mode="markers", showlegend=False, hoverinfo="skip",
                                     marker=dict(size=11, color=COLORS[sid], line=dict(color="#FDFDE8", width=2))))
    fig.add_hline(y=0, line_dash="dash", line_color="#8FA6C8",
                  annotation_text="horizon 0°", annotation_font_color="#8FA6C8")
    fig.add_vline(x=moment, line_color="#FDFDE8", line_width=1)
    fig.update_yaxes(title_text="elevation above horizon, °")
    fig.update_xaxes(tickformat="%d %b")
    return style(fig, 320, title)


def sky_chart(frames: dict, moment) -> go.Figure:
    """Панорама горизонту: по горизонталі — сторона світу, по вертикалі — висота над горизонтом.
    Диск Сонця жовтий, Землі — синій; літера на диску — точка посадки. Тонка лінія — шлях за період."""
    SUN, EARTH = "#FFD76A", "#4A9BE8"
    fig = go.Figure()
    fig.add_hrect(y0=-6.5, y1=0, fillcolor="rgba(120,130,150,0.22)", line_width=0)      # поверхня Місяця
    fig.add_hline(y=0, line_color="#A4C6F2", line_width=1.5)
    for body, color, label in (("sun", SUN, "Sun"), ("earth", EARTH, "Earth")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                                 marker=dict(size=13, color=color)))             # запис для легенди
        for sid, df in frames.items():
            az = [a - 360 if a > 180 else a for a in df[f"{body}_azimuth_deg"]]     # −180…180, північ у центрі
            el = df[f"{body}_elevation_deg"].tolist()
            xs, ys = [az[0]], [el[0]]
            for i in range(1, len(az)):                      # розрив лінії при переході через південь
                if abs(az[i] - az[i - 1]) > 180:
                    xs.append(None); ys.append(None)
                xs.append(az[i]); ys.append(el[i])
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False, hoverinfo="skip",
                                     line=dict(color=rgba(color, 0.4), width=1.5, dash="dot")))
            i_now = df.index[df["time"] == moment][0]
            x, y = az[df.index.get_loc(i_now)], el[df.index.get_loc(i_now)]
            up = y > 0
            fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", showlegend=False, hoverinfo="skip",
                                     marker=dict(size=46, color=rgba(color, 0.18 if up else 0.05))))   # сяйво
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers+text", showlegend=False,
                text=[SITES[sid][-1]], textfont=dict(color="#021526", size=13, family="Inter"),
                marker=dict(size=24, color=rgba(color, 1.0 if up else 0.35),
                            line=dict(color=COLORS[sid], width=3)),
                hovertemplate=f"{label}, {SITES[sid]}<br>elevation %{{y:.2f}}°<extra></extra>"))
    fig.add_annotation(x=176, y=-5.8, text="lunar surface", showarrow=False, xanchor="right",
                       font=dict(color="#8FA6C8", size=12))
    fig.update_xaxes(range=[-180, 180], tickmode="array", tickvals=[-180, -90, 0, 90, 180],
                     ticktext=["S", "W", "N", "E", "S"], showgrid=False, zeroline=False)
    fig.update_yaxes(range=[-6.5, 6.5], ticksuffix="°", dtick=4, zeroline=False, gridcolor="rgba(143,166,200,0.12)")
    return style(fig, 360)


def pole_map(reports: dict) -> go.Figure:
    """Вигляд на південний полюс: відстань від центру = градуси від полюса."""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[0], theta=[0], mode="markers+text", text=["South Pole"],
                                  textposition="bottom center", marker=dict(size=8, color="#8FA6C8"),
                                  showlegend=False, hoverinfo="skip"))
    for sid, rep in reports.items():
        fig.add_trace(go.Scatterpolar(
            r=[90 + rep["lat"]], theta=[rep["lon"]], mode="markers+text", text=[SITES[sid]],
            textposition="top center", marker=dict(size=16, color=COLORS[sid]), name=SITES[sid],
            hovertemplate=f"{SITES[sid]}<br>latitude {rep['lat']:.2f}°<br>longitude {rep['lon']:.2f}° E<extra></extra>"))
    fig.update_layout(showlegend=False, polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, 1.5], tickvals=[0.5, 1.0, 1.5],
                        ticktext=["89.5° S", "89° S", "88.5° S"]),
        angularaxis=dict(ticksuffix="° E", dtick=45)))
    return style(fig, 380)


# ───────────────────────── Розмітка кадру ─────────────────────────
def _font(px: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", px)
    except OSError:
        try:
            return ImageFont.truetype("arialbd.ttf", px)      # Windows
        except OSError:
            return ImageFont.load_default()


def keep_out_mask(res: dict, radius_px: float, margin_px: float, scale: float) -> np.ndarray:
    """Де НЕ може стояти центр модуля: ближче ніж радіус+запас до небезпеки або ближче ніж радіус до краю.
    Наближена карта для показу (рахується на зменшеній сітці); самі зони рахує safety_algorithm."""
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
    """step: 0 — чистий кадр, 1 — небезпеки, 2 — ще й заборонена область, 3 — ще й зони посадки."""
    k = max(1.0, 1000 / img.width)                 # малі кадри збільшуємо, щоб розмітка була чіткою
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
        for i, c in reversed(list(enumerate(zones["candidates"]))):   # №1 малюємо останнім, поверх
            x, y, r = c["x"] * k, c["y"] * k, radius_px * k
            color = GREEN if i == 0 else AMBER
            d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=lw + (2 if i == 0 else 0))
            d.text((x, y), str(i + 1), fill=color, font=font, anchor="mm")
    return out


def find_zones(res: dict, radius_px: float, margin_px: float) -> dict:
    """Виклик модуля Дмитра. bbox обрізаємо до меж кадру — його валідатор цього не робить."""
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


# ───────────────────────── Підключення до backend ─────────────────────────
# Блоку налаштувань на сторінці немає. Адреса backend береться зі змінної SAFELANDING_API
# (за замовчуванням http://localhost:8000). Якщо backend не відповідає, сайт сам показує
# збережені дані з папки fallback/. Примусово ввімкнути збережені дані: SAFELANDING_OFFLINE=1.
api_url = DEFAULT_API.rstrip("/")
force_offline = os.getenv("SAFELANDING_OFFLINE", "0") == "1"
health = None if force_offline else api_health(api_url)
API = api_url if health else None
if not API and not FALLBACK_DIR.is_dir():
    st.error(f"Folder {FALLBACK_DIR} not found. Put app.py into frontend/ inside the team repository.")
    st.stop()

# Шапка: відео посадки від команди (frontend/static/landing.mp4). Якщо файлу немає — лишається темне небо.
html('<div class="sl-hero"><video autoplay muted loop playsinline preload="auto" src="app/static/landing.mp4"></video>'
     '<span class="sl-shoot"></span><div class="sl-hero-text"><h1>SafeLanding AI</h1>'
     '<p>When and where it is safe to land at the lunar South Pole.</p></div></div>')
tab_plan, tab_ai = st.tabs(["When to land: mission planner", "Where to land: onboard AI"])

# ───────────────────────── Вкладка 1: планувальник ─────────────────────────
with tab_plan:
    c1, c2 = st.columns([2, 3])
    chosen = c1.multiselect("Landing sites", list(SITES), default=list(SITES), format_func=SITES.get)
    period = c2.date_input("Period (UTC)", (DATA_START, DATA_END), min_value=DATA_START,
                           max_value=DATA_END, format="DD.MM.YYYY")
    if not chosen:
        st.info("Choose at least one site.")
    elif not isinstance(period, tuple) or len(period) != 2:
        st.info("Choose the start and end date of the period.")
    else:
        try:
            reports = {s: site_report(API, s, period[0].isoformat(), period[1].isoformat()) for s in chosen}
        except Exception as e:
            st.error(f"Could not load the data: {e}")
            st.stop()
        frames = {s: to_frame(r) for s, r in reports.items()}
        times = list(next(iter(frames.values()))["time"])
        st.session_state["_times"] = times

        # Повзунок часу та перемикач «Play».
        # Значення віджетів не можна міняти після їх появи на сторінці, тому крок програвання
        # кладеться в "_next_moment"/"_stop_play" і переноситься сюди на початку наступного проходу.
        if "_next_moment" in st.session_state:
            st.session_state["moment"] = st.session_state.pop("_next_moment")
        if st.session_state.pop("_stop_play", False):
            st.session_state["playing"] = False
        if st.session_state.get("moment") not in times:           # період змінився — починаємо спочатку
            st.session_state["moment"] = times[0]
            st.session_state["playing"] = False
        sl_col, btn_col = st.columns([5, 1], vertical_alignment="bottom")
        was_playing = st.session_state.get("_was_playing", False)
        playing = btn_col.toggle("▶ Play", key="playing", help="Run through the selected period automatically")
        if playing and not was_playing and st.session_state["moment"] == times[-1]:
            st.session_state["moment"] = times[0]                 # з кінця періоду починаємо знову
        st.session_state["_was_playing"] = playing
        moment = sl_col.select_slider("Moment (UTC)", options=times, key="moment",
                                      format_func=lambda t: t.strftime("%d %b %H:%M"))

        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(elevation_chart(frames, "sun_elevation_deg", "Sun — power", moment), width="stretch")
            st.plotly_chart(elevation_chart(frames, "earth_elevation_deg", "Earth — direct link", moment),
                            width="stretch")
        with right:
            st.markdown(f"**Selected moment: {moment.strftime('%d %b %Y, %H:%M')} UTC**")
            for sid, df in frames.items():
                row = df.loc[df["time"] == moment].iloc[0]
                cells = ""
                for body, label in (("sun", "Sun"), ("earth", "Earth")):
                    vis = bool(row[f"{body}_visible"])
                    val = f"{row[f'{body}_elevation_deg']:+.2f}".replace("-", "−")
                    cells += (f'<div><span class="sl-lab"><i class="sl-ico {body}{" on" if vis else ""}"></i>{label}</span>'
                              f'<span class="sl-val">{val}°</span>'
                              f'<span class="sl-pill {"ok" if vis else "no"}">{"visible" if vis else "below horizon"}</span></div>')
                html(f'<div class="sl-site" style="--c:{COLORS[sid]}"><div class="sl-site-name">{SITES[sid]}</div>'
                     f'<div class="sl-site-grid">{cells}</div></div>')
            view = st.radio("View", ["Horizon at this moment", "Site map"], horizontal=True,
                            label_visibility="collapsed")
            st.plotly_chart(sky_chart(frames, moment) if view == "Horizon at this moment" else pole_map(reports),
                            width="stretch")

        st.subheader("Visibility windows over the period")
        html(ribbon(frames, reports, moment))

        st.subheader("Site comparison")
        mets = {s: site_metrics(r) for s, r in reports.items()}
        rows = [  # (назва, значення, більше — краще?)
            ("Sun above the horizon, %", lambda m: 100 * m["sun_h"] / m["period"], True),
            ("Earth above the horizon, %", lambda m: 100 * m["earth_h"] / m["period"], True),
            ("Longest darkness, h", lambda m: m["dark"], False),
            ("Longest direct-link outage, h", lambda m: m["outage"], False),
            ("Sun and Earth together, h", lambda m: m["both_h"], True),
        ]
        head = "".join(f'<th style="color:{COLORS[s]}">{SITES[s]}<br><span style="font-weight:400;color:var(--sl-muted)">'
                       f'{num(abs(rep["lat"]), 2)}° S, {num(rep["lon"], 2)}° E</span></th>'
                       for s, rep in reports.items())
        body = ""
        for name, fn, more_is_better in rows:
            vals = {s: round(fn(mets[s]), 1) for s in reports}
            best = (max if more_is_better else min)(vals.values())
            mark = len(set(vals.values())) > 1          # виділяємо, лише коли точки відрізняються
            body += f"<tr><td>{name}</td>" + "".join(
                f'<td class="{"best" if mark and v == best else ""}">{num(v)}</td>' for v in vals.values()) + "</tr>"
        html(f'<table class="sl-table"><thead><tr><th>Metric</th>{head}</tr></thead><tbody>{body}</tbody></table>')
        if len(mets) == 2:
            a, b = chosen
            diff = mets[b]["earth_h"] - mets[a]["earth_h"]
            if abs(diff) >= 0.05:
                best = b if diff > 0 else a
                html(f'<div class="sl-verdict"><b>Verdict.</b> {SITES[best]} has a direct-link window to Earth '
                     f'that is {num(abs(diff))} h longer over this period.</div>')
        rep0 = next(iter(reports.values()))
        source = "live from the backend API" if API else "saved copy in the repository (backend offline)"
        st.caption(f"Data: NASA/JPL Horizons, period {mets[chosen[0]]['period']:.0f} h, 30 min step, {source}. "
                   f"Horizon model: {rep0['horizon_assumption']}.")
        with st.expander("Model limitations"):
            st.markdown("- The Sun's **centre** being visible is not the same as solar-panel power.\n"
                        "- The Earth's **centre** being visible is geometric line of sight only, not a guaranteed radio link.\n"
                        "- With an idealised 0° horizon the terrain is ignored; real terrain makes the windows shorter.\n"
                        "- This is a planning demonstrator, not certified landing software.")

# ───────────────────────── Вкладка 2: бортовий AI ─────────────────────────
SAMPLES = {"high_sun": "High Sun", "low_sun": "Low Sun, long shadows", "dense_craters": "Dense crater field"}
SAMPLE_DIR = Path(__file__).resolve().parent / "static" / "samples"
STEPS = ["Digitized frame", "Hazards", "Keep-out area", "Landing zones"]

with tab_ai:
    # 1) Джерело кадру: готові приклади, власний файл або камера
    st.markdown("**Pick a sample frame or add your own**")
    cols = st.columns([1, 1, 1, 2])
    for col, (key, title) in zip(cols, SAMPLES.items()):
        path = SAMPLE_DIR / f"{key}.jpg"
        if path.is_file():
            col.image(str(path), width="stretch", alt=f"{title} synthetic lunar terrain sample")
            chosen_now = st.session_state.get("sample") == key
            if col.button(title, key=f"sample_{key}", width="stretch", type="primary" if chosen_now else "secondary"):
                st.session_state["sample"] = key
                st.rerun()                                 # щоб обрана кнопка одразу підсвітилась
    with cols[3]:
        up = st.file_uploader("Surface image or a photo of the 3D-printed terrain", type=["png", "jpg", "jpeg", "bmp"])
        if st.toggle("Take a frame with this computer's camera", False):
            up = st.camera_input("Frame") or up
    html('<p class="sl-note">Sample frames are synthetic terrain renders made for this demo, not lunar photographs.</p>')

    # 7) Масштаб і розмір модуля — у згорнутому блоці
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

    # Який кадр аналізуємо: завантажений має пріоритет над прикладом
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
        if res is None:                                   # запасний детектор, щоб демо працювало без backend
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

        # 2) Покроковий показ
        step_name = st.segmented_control("Show", STEPS, default=STEPS[-1], key="ai_step") or STEPS[-1]
        step = STEPS.index(step_name)

        pic, info = st.columns([3, 2])
        captions = ["Digitized image with normalized lighting and enhanced local contrast.",
                    "Red boxes are detected hazards; the number is the detector's confidence.",
                    "Red tint: the lander centre cannot be here, a hazard or the frame edge is too close.",
                    "Green circle is the recommended landing zone, amber circles are alternatives."]
        pic.image(draw_overlay(img, res, zones, radius_px, margin_px, step), width="stretch", caption=captions[step])
        with info:
            # 4) Вердикт і картки зон
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

# ───────────────────────── Програвання тижня ─────────────────────────
# Стоїть у самому кінці, щоб обидві вкладки встигли намалюватися перед наступним кроком.
if st.session_state.get("playing"):
    _times = st.session_state.get("_times", [])
    if st.session_state.get("moment") in _times:
        _i = _times.index(st.session_state["moment"]) + PLAY_STEP
        if _i >= len(_times):
            _i = len(_times) - 1
            st.session_state["_stop_play"] = True
        st.session_state["_next_moment"] = _times[_i]
    else:
        st.session_state["_stop_play"] = True
    time.sleep(PLAY_DELAY)
    st.rerun()
