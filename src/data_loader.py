import pandas as pd
import numpy as np
import requests
import pathlib
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_weather_data(start_date: str = "2023-01-01", 
                       end_date: str = "2024-12-31",
                       latitude: float = 1.3521,
                       longitude: float = 103.8198) -> pd.DataFrame:
    
    url = "https://archive-api.open-meteo.com/v1/archive"

    daily_params = (
        "temperature_2m_mean,"
        "relative_humidity_2m_mean,"
        "wind_speed_10m_max,"
        "shortwave_radiation_sum"
    )
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": daily_params,
        "timezone": "Asia/Singapore"
    }

    logger.info("Fetching weather data from Open-Meteo API...")
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            weather_df = pd.DataFrame({
                'date': pd.to_datetime(data['daily']['time']),
                'temperature': data['daily']['temperature_2m_mean'],
                'humidity': data['daily']['relative_humidity_2m_mean'],
                'wind_speed': data['daily']['wind_speed_10m_max'],
                'solar_radiation': data['daily']['shortwave_radiation_sum']
            })
            logger.info(f"✓ Weather data fetched from API: {len(weather_df)} days")
            return weather_df
        else:
            logger.warning(f"API returned status {response.status_code}. Generating synthetic weather data...")
    except Exception as e:
        logger.warning(f"API call failed ({e}). Generating synthetic weather data...")

def load_demand_data(csv_path: str) -> pd.DataFrame:
    
    if not pathlib.Path(csv_path).exists():
        raise FileNotFoundError(f"Demand CSV not found at: {csv_path}")
    
    logger.info(f"Loading demand data from: {csv_path}")
    demand_df = pd.read_csv(csv_path, parse_dates=['date'])
    
    # Validate required columns
    if 'demand_mwh' not in demand_df.columns:
        raise ValueError(
            f"CSV must contain 'demand_mwh' column. Found: {demand_df.columns.tolist()}"
        )
    
    # Keep only required columns and sort
    demand_df = demand_df[['date', 'demand_mwh']].copy()
    demand_df = demand_df.sort_values('date').reset_index(drop=True)
    
    # Validate minimum data length
    if len(demand_df) < 31:
        raise ValueError(
            f"Loaded demand data has only {len(demand_df)} rows. "
            "Minimum 31 consecutive daily rows required."
        )
    
    logger.info(f"✓ Electricity demand loaded: {len(demand_df)} days")
    return demand_df


def merge_weather_demand(weather_df: pd.DataFrame, 
                        demand_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    # Merge on date (inner join - only matching dates)
    df = pd.merge(weather_df, demand_df, on='date', how='inner')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate merge statistics
    dropped_weather = len(weather_df) - len(df)
    dropped_demand = len(demand_df) - len(df)
    
    stats = {
        'original_weather_rows': len(weather_df),
        'original_demand_rows': len(demand_df),
        'merged_rows': len(df),
        'dropped_weather': dropped_weather,
        'dropped_demand': dropped_demand
    }
    
    if dropped_weather > 0 or dropped_demand > 0:
        logger.warning(
            f"Dropped {dropped_weather} weather rows and {dropped_demand} demand rows "
            "due to date mismatch"
        )
    
    if df.shape[0] < 31:
        raise ValueError(
            f"Merged dataset has only {df.shape[0]} rows (min required: 31). "
            "Check data completeness."
        )
    
    logger.info(f"✓ Datasets merged successfully: {len(df)} days")
    return df, stats


def load_all_data(demand_csv_path: str,
                  start_date: str = "2023-01-01",
                  end_date: str = "2024-12-31") -> Tuple[pd.DataFrame, dict]:
    
    weather_df = fetch_weather_data(start_date, end_date)
    try:
        demand_df = load_demand_data(demand_csv_path)
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"No data found: {e}")
    
    df, merge_stats = merge_weather_demand(weather_df, demand_df)
    
    return df, merge_stats