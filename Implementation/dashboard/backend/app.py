"""
FastAPI backend for the Electricity Demand Forecasting Dashboard.

Endpoints:
  GET /api/summary   -> model metadata + test metrics + feature importance
  GET /api/history   -> historical daily demand + weather series
  GET /api/predict   -> demand forecast for a user-supplied date (?date=YYYY-MM-DD)

Run:  uvicorn app:app --reload --port 8000   (from dashboard/backend)
"""
import os
import sys
import pickle
import logging
from typing import Optional

import pandas as pd
import requests
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Make project root importable so that pickled CatBoostPPSOModel (src.model) resolves
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))  # pickled model references top-level 'model' module

from src.data_loader import load_all_data  # noqa: E402
from src.preprocessing import DataPreprocessor  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard")

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "catboost_ppso_model.pkl")
SUMMARY_PATH = os.path.join(PROJECT_ROOT, "results", "summary.yaml")
LEADERBOARD_PATH = os.path.join(PROJECT_ROOT, "results", "model_comparison.csv")
CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "dashboard_history.csv")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.yaml")

FEATURE_COLS = [
    "temperature", "humidity", "wind_speed", "solar_radiation",
    "cci", "season",
    "day_of_week", "month", "is_weekend", "is_holiday",
    "demand_lag_1", "demand_lag_7",
    "demand_rolling_mean_3", "demand_rolling_mean_7",
    "demand_rolling_std_3", "demand_rolling_std_7",
]

app = FastAPI(title="SG Electricity Demand Forecast API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {"model": None, "df": None, "preprocessor": None, "climatology": None}


def _load_history() -> pd.DataFrame:
    """Load processed history from cache, or rebuild it from the pipeline."""
    if os.path.exists(CACHE_PATH):
        df = pd.read_csv(CACHE_PATH, parse_dates=["date"])
        logger.info("Loaded cached history (%d rows)", len(df))
        return df

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    weather = config["data"]["weather"]
    demand_csv = os.path.join(PROJECT_ROOT, "ema_daily_demand.csv")
    df, _ = load_all_data(
        demand_csv_path=demand_csv,
        start_date=weather["start_date"],
        end_date=weather["end_date"],
        latitude=weather["latitude"],
        longitude=weather["longitude"],
    )
    df = STATE["preprocessor"].process(df, config)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    logger.info("Built and cached history (%d rows)", len(df))
    return df


@app.on_event("startup")
def startup():
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    STATE["model"] = data["model"]
    STATE["preprocessor"] = DataPreprocessor()
    STATE["df"] = _load_history().sort_values("date").reset_index(drop=True)
    df = STATE["df"]
    # Monthly climatology for weather imputation
    STATE["climatology"] = (
        df.assign(m=df["date"].dt.month)
          .groupby("m")[["temperature", "humidity", "wind_speed", "solar_radiation"]]
          .mean()
    )
    logger.info("Startup complete. Model + %d history rows ready.", len(df))


@app.get("/api/summary")
def summary():
    with open(SUMMARY_PATH) as f:
        s = yaml.safe_load(f)
    leaderboard = []
    if os.path.exists(LEADERBOARD_PATH):
        lb = pd.read_csv(LEADERBOARD_PATH)
        leaderboard = lb.to_dict(orient="records")

    fi = []
    model = STATE["model"]
    try:
        importances = model.final_model.feature_importances_
        fi = sorted(
            [{"feature": f, "importance": round(float(v), 3)}
             for f, v in zip(FEATURE_COLS, importances)],
            key=lambda x: -x["importance"],
        )
    except Exception:  # pragma: no cover
        logger.warning("Feature importance unavailable")

    return {"summary": s, "leaderboard": leaderboard, "feature_importance": fi}


@app.get("/api/history")
def history():
    df = STATE["df"]
    return {
        "min_date": df["date"].min().strftime("%Y-%m-%d"),
        "max_date": df["date"].max().strftime("%Y-%m-%d"),
    }


@app.get("/api/range")
def range_stats(start: str = Query(...), end: str = Query(...)):
    """Series + user-friendly statistics for a selected period (min 30 days)."""
    try:
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    except Exception:
        raise HTTPException(400, "Invalid date format, expected YYYY-MM-DD.")
    if t1 < t0:
        raise HTTPException(400, "End date must be after start date.")
    if (t1 - t0).days + 1 < 30:
        raise HTTPException(400, "Please select a period of at least 30 days.")

    df = STATE["df"]
    sel = df[(df["date"] >= t0) & (df["date"] <= t1)].copy()
    if len(sel) < 30:
        raise HTTPException(
            400,
            f"Only {len(sel)} days of data available in this period. "
            f"Data covers {df['date'].min().date()} to {df['date'].max().date()}.",
        )

    # Model predictions for the selected period
    X = sel[FEATURE_COLS]
    sel["predicted_mwh"] = STATE["model"].predict(X).round(1)

    # Friendly stats
    peak = sel.loc[sel["demand_mwh"].idxmax()]
    low = sel.loc[sel["demand_mwh"].idxmin()]
    mape = float((abs(sel["demand_mwh"] - sel["predicted_mwh"]) / sel["demand_mwh"]).mean() * 100)
    weekday_avg = float(sel.loc[sel["is_weekend"] == 0, "demand_mwh"].mean())
    weekend_avg = float(sel.loc[sel["is_weekend"] == 1, "demand_mwh"].mean())

    dow = sel.groupby("day_of_week")["demand_mwh"].mean().round(1).reindex(range(7)).tolist()

    out = sel[["date", "demand_mwh", "predicted_mwh", "temperature", "humidity"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    return {
        "series": out.to_dict(orient="records"),
        "stats": {
            "days": int(len(sel)),
            "avg_demand": round(float(sel["demand_mwh"].mean()), 1),
            "peak_demand": round(float(peak["demand_mwh"]), 1),
            "peak_date": peak["date"].strftime("%Y-%m-%d"),
            "low_demand": round(float(low["demand_mwh"]), 1),
            "low_date": low["date"].strftime("%Y-%m-%d"),
            "accuracy_pct": round(100 - mape, 1),
            "avg_temperature": round(float(sel["temperature"].mean()), 1),
            "weekday_avg": round(weekday_avg, 1),
            "weekend_avg": round(weekend_avg, 1),
        },
        "avg_by_dow": dow,
    }


def _build_feature_row(target: pd.Timestamp) -> tuple[dict, dict]:
    """Reconstruct the 16-feature vector for an arbitrary date."""
    df = STATE["df"]
    pre = STATE["preprocessor"]
    clim = STATE["climatology"]

    # Weather: actual if available, otherwise monthly climatology
    row_actual = df.loc[df["date"] == target]
    weather_source = "actual"
    if not row_actual.empty:
        w = row_actual.iloc[0]
        weather = {k: float(w[k]) for k in ["temperature", "humidity", "wind_speed", "solar_radiation"]}
    else:
        weather_source = "monthly climatology"
        c = clim.loc[target.month]
        weather = {k: float(c[k]) for k in ["temperature", "humidity", "wind_speed", "solar_radiation"]}

    T, RH = weather["temperature"], weather["humidity"]
    cci = T + 0.55 * (1 - RH / 100.0) * (T - 14.5)

    # Temporal features
    dow = target.dayofweek
    month = target.month
    feats = {
        **weather,
        "cci": cci,
        "season": (month % 12 // 3 + 1),
        "day_of_week": dow,
        "month": month,
        "is_weekend": int(dow in (5, 6)),
        "is_holiday": int(target in pre.sg_holidays),
    }

    # Autoregressive features from most recent demand history strictly before target
    hist = df.loc[df["date"] < target, ["date", "demand_mwh"]]
    if len(hist) < 8:
        raise HTTPException(400, "Date too early: not enough demand history before this date.")
    d = hist["demand_mwh"]
    feats["demand_lag_1"] = float(d.iloc[-1])
    feats["demand_lag_7"] = float(d.iloc[-7])
    feats["demand_rolling_mean_3"] = float(d.iloc[-3:].mean())
    feats["demand_rolling_mean_7"] = float(d.iloc[-7:].mean())
    feats["demand_rolling_std_3"] = float(d.iloc[-3:].std())
    feats["demand_rolling_std_7"] = float(d.iloc[-7:].std())

    context = {
        "weather_source": weather_source,
        "last_history_date": hist["date"].iloc[-1].strftime("%Y-%m-%d"),
    }
    return feats, context


@app.get("/api/predict")
def predict(date: str = Query(..., description="YYYY-MM-DD")):
    try:
        target = pd.Timestamp(date)
    except Exception:
        raise HTTPException(400, "Invalid date format, expected YYYY-MM-DD.")

    feats, context = _build_feature_row(target)
    X = pd.DataFrame([[feats[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
    pred = float(STATE["model"].predict(X)[0])

    df = STATE["df"]
    actual_row = df.loc[df["date"] == target]
    actual = float(actual_row["demand_mwh"].iloc[0]) if not actual_row.empty else None

    return {
        "date": target.strftime("%Y-%m-%d"),
        "prediction_mwh": round(pred, 1),
        "actual_mwh": actual,
        "features": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in feats.items()},
        **context,
    }


# ---------------------------------------------------------------------------
# Demand trend forecasting (Open-Meteo weather + recursive model prediction)
# ---------------------------------------------------------------------------

WEATHER_KEYS = ["temperature", "humidity", "wind_speed", "solar_radiation"]


def _fetch_openmeteo_forecast(start: pd.Timestamp, end: pd.Timestamp, model: str) -> dict:
    """Short/medium-range weather forecast (max 16 days ahead), daily resolution."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 1.3521,
        "longitude": 103.8198,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                 "relative_humidity_2m_mean,wind_speed_10m_max,shortwave_radiation_sum",
        "forecast_days": 16,
        "timezone": "Asia/Singapore",
        "models": model,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    d = r.json()["daily"]
    out = {}
    for i, ds in enumerate(d["time"]):
        t = pd.Timestamp(ds)
        if t < start or t > end:
            continue
        tmax, tmin = d["temperature_2m_max"][i], d["temperature_2m_min"][i]
        out[ds] = {
            "temperature": (tmax + tmin) / 2 if tmax is not None and tmin is not None else None,
            "temperature_max": tmax,
            "temperature_min": tmin,
            "precipitation": d["precipitation_sum"][i],
            "humidity": d["relative_humidity_2m_mean"][i],
            "wind_speed": d["wind_speed_10m_max"][i],
            "solar_radiation": d["shortwave_radiation_sum"][i],
        }
    return out


def _fetch_openmeteo_climate(start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Long-range climate trend forecast (CFSv2 seasonal model), daily resolution."""
    url = "https://climate-api.open-meteo.com/v1/climate"
    params = {
        "latitude": 1.3521,
        "longitude": 103.8198,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "models": "CMCC_CM2_VHR4",
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,"
                 "wind_speed_10m_mean,shortwave_radiation_sum,precipitation_sum",
        "timezone": "Asia/Singapore",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        d = r.json()["daily"]
        out = {}
        for i, ds in enumerate(d["time"]):
            out[ds] = {
                "temperature": d["temperature_2m_mean"][i],
                "humidity": d["relative_humidity_2m_mean"][i],
                "wind_speed": d["wind_speed_10m_mean"][i],
                "solar_radiation": d["shortwave_radiation_sum"][i],
                "precipitation": (d.get("precipitation_sum") or [None] * len(d["time"]))[i],
            }
        return out
    except Exception as e:  # Climate API can be flaky -> fall back to climatology
        logger.warning("Climate API unavailable (%s); falling back to climatology.", e)
        return {}


def _weather_for(target: pd.Timestamp, forecast: dict, climate: dict) -> tuple[dict, str]:
    """Resolve daily weather with priority: history > forecast API > climate API > climatology."""
    df = STATE["df"]
    row = df.loc[df["date"] == target]
    if not row.empty:
        w = row.iloc[0]
        return {k: float(w[k]) for k in WEATHER_KEYS}, "history"
    key = target.strftime("%Y-%m-%d")
    for src, name in ((forecast, "open-meteo forecast"), (climate, "open-meteo climate")):
        if key in src and src[key].get("temperature") is not None:
            c = STATE["climatology"].loc[target.month]
            return {k: (float(src[key][k]) if src[key].get(k) is not None else float(c[k]))
                    for k in WEATHER_KEYS}, name
    c = STATE["climatology"].loc[target.month]
    return {k: float(c[k]) for k in WEATHER_KEYS}, "monthly climatology"


SEASONS = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}  # quarter -> (start month, end month)


@app.get("/api/forecast")
def forecast(
    mode: str = Query("days", pattern="^(days|month|season)$"),
    days: int = Query(7, ge=1, le=16),
    month: Optional[str] = Query(None, description="YYYY-MM"),
    season: Optional[int] = Query(None, ge=1, le=4),
    year: Optional[int] = Query(None),
    weather_model: str = Query("best_match", pattern="^(best_match|ecmwf_ifs025)$"),
):
    today = pd.Timestamp.now(tz="Asia/Singapore").tz_localize(None).normalize()

    # Resolve requested window
    if mode == "days":
        w_start, w_end = today, today + pd.Timedelta(days=days - 1)
    elif mode == "month":
        if not month:
            raise HTTPException(400, "Parameter 'month' (YYYY-MM) is required for month mode.")
        w_start = pd.Timestamp(f"{month}-01")
        w_end = w_start + pd.offsets.MonthEnd(0)
    else:
        if not season or not year:
            raise HTTPException(400, "Parameters 'season' (1-4) and 'year' are required for season mode.")
        m0, m1 = SEASONS[season]
        w_start = pd.Timestamp(year=year, month=m0, day=1)
        w_end = pd.Timestamp(year=year, month=m1, day=1) + pd.offsets.MonthEnd(0)

    if w_end > today + pd.DateOffset(months=6):
        raise HTTPException(400, "Forecasts are limited to 6 months ahead.")

    df = STATE["df"]
    last_hist = df["date"].max()
    if w_end <= last_hist:
        raise HTTPException(400, "The selected period is entirely in the past — use the History Report page instead.")

    # Fetch weather sources
    fc = {}
    if w_start <= today + pd.Timedelta(days=15):
        try:
            fc = _fetch_openmeteo_forecast(w_start, w_end, weather_model)
        except Exception as e:
            logger.warning("Forecast API failed: %s", e)
    cl = {}
    if w_end > today + pd.Timedelta(days=15):
        cl = _fetch_openmeteo_climate(max(w_start, today), w_end)

    # Recursive demand prediction from the day after history ends up to w_end
    pre = STATE["preprocessor"]
    model = STATE["model"]
    demand = list(df["demand_mwh"].iloc[-30:])  # rolling buffer of recent demand
    cur = last_hist + pd.Timedelta(days=1)
    results = []
    sources_used = set()

    while cur <= w_end:
        weather, src = _weather_for(cur, fc, cl)
        T, RH = weather["temperature"], weather["humidity"]
        row = {
            **weather,
            "cci": T + 0.55 * (1 - RH / 100.0) * (T - 14.5),
            "season": (cur.month % 12 // 3 + 1),
            "day_of_week": cur.dayofweek,
            "month": cur.month,
            "is_weekend": int(cur.dayofweek in (5, 6)),
            "is_holiday": int(cur in pre.sg_holidays),
        }
        s = pd.Series(demand)
        row["demand_lag_1"] = float(s.iloc[-1])
        row["demand_lag_7"] = float(s.iloc[-7])
        row["demand_rolling_mean_3"] = float(s.iloc[-3:].mean())
        row["demand_rolling_mean_7"] = float(s.iloc[-7:].mean())
        row["demand_rolling_std_3"] = float(s.iloc[-3:].std())
        row["demand_rolling_std_7"] = float(s.iloc[-7:].std())

        X = pd.DataFrame([[row[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
        pred = float(model.predict(X)[0])
        demand.append(pred)
        demand = demand[-30:]

        if w_start <= cur <= w_end:
            days_ahead = (cur - today).days
            entry = {
                "date": cur.strftime("%Y-%m-%d"),
                "predicted_mwh": round(pred, 1),
                "temperature": round(T, 1),
                "humidity": round(RH, 1),
                "weather_source": src,
                "days_ahead": days_ahead,
            }
            key = cur.strftime("%Y-%m-%d")
            if key in fc:
                entry["temperature_max"] = fc[key].get("temperature_max")
                entry["temperature_min"] = fc[key].get("temperature_min")
                entry["precipitation"] = fc[key].get("precipitation")
            elif key in cl and cl[key].get("precipitation") is not None:
                entry["precipitation"] = round(float(cl[key]["precipitation"]), 1)
            results.append(entry)
            sources_used.add(src)
        cur += pd.Timedelta(days=1)

    if not results:
        raise HTTPException(400, "No days to forecast in the selected period.")

    ser = pd.DataFrame(results)
    trend = ser["predicted_mwh"].rolling(7, min_periods=1).mean().round(1).tolist()
    for i, t in enumerate(trend):
        results[i]["trend_mwh"] = t

    return {
        "mode": mode,
        "start": w_start.strftime("%Y-%m-%d"),
        "end": w_end.strftime("%Y-%m-%d"),
        "weather_model": weather_model,
        "weather_sources": sorted(sources_used),
        "stats": {
            "days": len(results),
            "avg_demand": round(float(ser["predicted_mwh"].mean()), 1),
            "peak_demand": round(float(ser["predicted_mwh"].max()), 1),
            "peak_date": ser.loc[ser["predicted_mwh"].idxmax(), "date"],
            "low_demand": round(float(ser["predicted_mwh"].min()), 1),
            "low_date": ser.loc[ser["predicted_mwh"].idxmin(), "date"],
            "avg_temperature": round(float(ser["temperature"].mean()), 1),
        },
        "series": results,
    }
