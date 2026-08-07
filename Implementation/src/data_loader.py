import pandas as pd
import requests
import pathlib
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_weather_data(start_date: str, 
                       end_date: str,
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

    logger.info(f"Fetching weather data from Open-Meteo ({start_date} to {end_date})...")
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        weather_df = pd.DataFrame({
            'date': pd.to_datetime(data['daily']['time']),
            'temperature': data['daily']['temperature_2m_mean'],
            'humidity': data['daily']['relative_humidity_2m_mean'],
            'wind_speed': data['daily']['wind_speed_10m_max'],
            'solar_radiation': data['daily']['shortwave_radiation_sum']
        })
        
        weather_df.ffill(inplace=True) 
        logger.info(f"✓ Weather data fetched: {len(weather_df)} days")
        return weather_df
        
    except Exception as e:
        logger.error(f"API call failed: {e}")
        raise

def load_demand_data(csv_path: str) -> pd.DataFrame:
    if not pathlib.Path(csv_path).exists():
        raise FileNotFoundError(f"Demand CSV not found at: {csv_path}")
    
    logger.info(f"Loading demand data from: {csv_path}")
    demand_df = pd.read_csv(csv_path, parse_dates=['date'])
    
    if 'demand_mwh' not in demand_df.columns:
        raise ValueError("CSV must contain 'demand_mwh' column.")
    
    demand_df = demand_df[['date', 'demand_mwh']].copy()
    demand_df = demand_df.sort_values('date').reset_index(drop=True)
    return demand_df


def load_all_data(demand_csv_path: str,
                  start_date: str = "2023-01-01",
                  end_date: str = "2024-12-31",
                  latitude: float = 1.3521,
                  longitude: float = 103.8198) -> Tuple[pd.DataFrame, dict]:
    weather_df = fetch_weather_data(start_date, end_date, latitude, longitude)
    demand_df = load_demand_data(demand_csv_path)
    
    df = pd.merge(weather_df, demand_df, on='date', how='inner')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    stats = {
        'original_weather_rows': len(weather_df),
        'original_demand_rows': len(demand_df),
        'merged_rows': len(df)
    }
    
    if df.shape[0] < 31:
        raise ValueError("Merged dataset has too few rows (min: 31). Check dates.")
        
    logger.info(f"✓ Datasets merged successfully: {len(df)} days")
    return df, stats