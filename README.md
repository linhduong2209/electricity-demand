# Singapore Electricity Demand Forecasting

Machine learning pipeline + web portal for **daily electricity demand forecasting in Singapore**, using weather variables (Open-Meteo), calendar features, and autoregressive demand history.

Eight model configurations are compared under a chronological 80/20 split: Lag-1 Baseline, Linear Regression, Random Forest, XGBoost, CatBoost — plus **PPSO-tuned variants of every tree-based model** (RF-PPSO, XGB-PPSO, CatBoost-PPSO). Phasor Particle Swarm Optimisation (PPSO) is applied with an **identical budget to all tunable models** to guarantee a fair comparison. The best model (CatBoost-PPSO, R² ≈ 0.914) powers a web portal (FastAPI + React/Vite) with historical analysis and weather-driven demand forecasts.

## Project Structure

```
electricity-demand/
├── main.py                 # Training pipeline entry point
├── configs/config.yaml     # Central configuration (data, features, models, PPSO)
├── ema_daily_demand.csv    # Cleaned daily demand data (output of tools/parse.py)
├── requirements.txt        # Python dependencies
├── src/
│   ├── data_loader.py      # Demand CSV + Open-Meteo weather loading
│   ├── preprocessing.py    # Feature engineering (CCI, lags, rolling stats)
│   ├── model.py            # Models + PPSO optimiser + PPSOTunedModel
│   ├── trainer.py          # Unified train/predict/save wrapper
│   ├── evaluator.py        # Metrics (MAE, RMSE, R², MAPE, RAE, WI) + plots
│   └── visualizer.py       # EDA plots
├── tools/
│   ├── parse.py            # Parse raw EMA weekly .xls files → ema_daily_demand.csv
│   └── check.py            # Validate cleaned CSV (missing dates check)
├── dashboard/
│   ├── backend/app.py      # FastAPI REST service (loads best model)
│   └── frontend/           # React + Vite portal (History / Trends pages)
├── data/
│   ├── ema/                # Raw EMA weekly demand .xls files (2023–2025)
│   └── processed/          # Processed history for the dashboard
├── models/                 # Trained model pickles (*.pkl)
├── results/                # model_leaderboard.csv, summary.yaml, plots/
├── notebooks/              # Exploration notebook
└── docs/                   # Reference papers (incl. CatBoost-PPSO study)
```

## 1. Environment Setup

Requires **Python 3.9+** and **Node.js 18+**.

```bash
# From the project root (electricity-demand/)

# Create a virtual environment (skip if ./electricity-demand/ venv already exists)
python3 -m venv electricity-demand

# Activate it
source electricity-demand/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

> All commands below assume the venv is activated (or replace `python` with `./electricity-demand/bin/python`).

## 2. Data Preparation (optional — cleaned CSV is included)

Raw EMA weekly demand spreadsheets live in `data/ema/<year>/`. To rebuild the cleaned dataset:

```bash
# Parse all raw .xls files → ema_daily_demand.csv
python tools/parse.py

# Validate: check for missing dates in the expected range
python tools/check.py
```

Weather data does **not** need downloading — the pipeline fetches it automatically from the Open-Meteo API during training.

## 3. Train Models

```bash
python main.py
```

This runs the full pipeline: data loading → feature engineering (16 features) → EDA plots → chronological split → training of all 8 models (PPSO tuning for RF/XGB/CatBoost, ~30–60 min) → evaluation.

**Outputs:**

| Path | Content |
|---|---|
| `results/model_leaderboard.csv` | All models ranked by RMSE |
| `results/summary.yaml` | Best model + key metrics |
| `results/plots/` | EDA, feature importance, prediction plots |
| `models/*.pkl` | Serialised trained models |

## 4. Run the Backend (FastAPI, port 8000)

```bash
cd dashboard/backend
python -m uvicorn app:app --port 8000
```

Endpoints: `/api/summary`, `/api/range`, `/api/predict`, `/api/forecast`.
Check: `curl http://127.0.0.1:8000/api/summary`

## 5. Run the Frontend (React + Vite, port 5173)

```bash
cd dashboard/frontend
npm install        # first time only
npm run dev
```

Open http://localhost:5173 — the portal offers a **History Report** page (retrospective accuracy over a chosen period) and a **Demand Trends** page (forward forecasts driven by live Open-Meteo weather).
