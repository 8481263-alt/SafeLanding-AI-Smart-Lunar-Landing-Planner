# SafeLanding AI

Backend (FastAPI) + frontend (Streamlit) для вибору дати й місця посадки на південному полюсі Місяця.

## Запуск (Windows)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r frontend/requirements.txt
```

Термінал 1 — бекенд:
```powershell
.\run.ps1            # або .\run.ps1 -Fallback
```
Swagger: http://127.0.0.1:8000/docs

Термінал 2 — сайт (запускати саме з папки frontend):
```powershell
cd frontend
streamlit run app.py
```
Сайт: http://localhost:8501

## Потрібно від команди
- `train_hazard.py` і `training/best.pt` (Людина 1) — для /api/vision/analyze
- папка `safety_algorithm/` (Людина 4) — для зон посадки на вкладці AI

## Застереження
Дані Сонця/Землі — геометрія за JPL Horizons з нульовим горизонтом (без рельєфу).
Це не оцінка безпеки посадки.