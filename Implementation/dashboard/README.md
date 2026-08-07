# Electricity Demand Dashboard

Two-page portal for the Singapore electricity demand forecasting project:

- **History Report** — actual vs model-predicted demand for a selected period
  (date range / month / year, minimum 30 days).
- **Demand Trends** — forward-looking outlook (next 7/16 days, month, or season,
  up to 6 months ahead) combining the trained **CatBoost-PPSO** model with
  live weather forecasts from [Open-Meteo](https://open-meteo.com/).

## Run

### 1. Backend (FastAPI)

```bash
cd dashboard/backend
../../electricity-demand/bin/python -m uvicorn app:app --reload --port 8000
```

Endpoints: `/api/summary`, `/api/history`, `/api/range?start=&end=`,
`/api/predict?date=`, `/api/forecast?mode=days|month|season&...`.

On first start the backend rebuilds the processed history (fetches weather from
Open-Meteo archive) and caches it at `data/processed/dashboard_history.csv`.

### 2. Frontend (React + Vite, requires Node 20)

```bash
nvm use 20
cd dashboard/frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173 — API calls are proxied to the backend on port 8000.

## Forward forecasting logic

- **Weather** — Open-Meteo forecast API (`forecast_days=16`, models
  `best_match` or `ecmwf_ifs025`) for the short range; Open-Meteo climate API
  for monthly/seasonal outlooks; monthly climatology as final fallback.
- **Demand** — recursive prediction: the model predicts one day at a time and
  feeds each prediction back into the lag/rolling features of following days.
- **Accuracy note** (shown in the UI): weather forecasts are ~85–95% accurate
  for days 1–3, ~70–80% for days 4–7, ~50–60% (indicative) for days 8–16.
  Demand estimates degrade further with horizon due to recursive error.
