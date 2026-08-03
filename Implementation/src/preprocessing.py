import pandas as pd
import numpy as np
import holidays
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.sg_holidays = holidays.SG(years=[2023, 2024, 2025, 2026])
        logger.info("DataPreprocessor initialized")
    
    def add_weather_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract engineered weather features"""
        df = df.copy()
        
        if 'temperature' in df.columns and 'humidity' in df.columns:
            T = df['temperature']
            RH = df['humidity']
            df['cci'] = T + 0.55 * (1 - (RH / 100.0)) * (T - 14.5)
            
        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add categorical temporal features."""
        df = df.copy()
        
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_holiday'] = df['date'].apply(lambda x: int(x in self.sg_holidays))
        
        df['season'] = (df['month'] % 12 // 3 + 1).astype(int)
        
        return df
        
    def add_lag_and_rolling_features(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Calculate historical demand data as prediction features."""
        df = df.copy()
        lag_periods = config.get('lag_periods', [1, 7])
        rolling_windows = config.get('rolling_windows', [3, 7])
        
        for lag in lag_periods:
            df[f'demand_lag_{lag}'] = df['demand_mwh'].shift(lag)
            
        for window in rolling_windows:
            df[f'demand_rolling_mean_{window}'] = df['demand_mwh'].shift(1).rolling(window=window).mean()
            df[f'demand_rolling_std_{window}'] = df['demand_mwh'].shift(1).rolling(window=window).std()
            
        return df

    def process(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        logger.info("Starting preprocessing pipeline...")
        df = df.sort_values('date').reset_index(drop=True)
        
        df = self.add_weather_engineered_features(df)
        df = self.add_temporal_features(df)
        df = self.add_lag_and_rolling_features(df, config.get('preprocessing', {}))
        
        before_len = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.info(f"✓ Feature engineering applied. Dropped {before_len - len(df)} initial rows due to lag/rolling NaNs.")
        
        return df


def prepare_train_test_split(df: pd.DataFrame, 
                             feature_cols: List[str], 
                             target_col: str = 'demand_mwh', 
                             train_split_ratio: float = 0.8) -> Tuple:

    if not all(col in df.columns for col in feature_cols + [target_col]):
        missing = [c for c in feature_cols + [target_col] if c not in df.columns]
        raise ValueError(f"Missing columns in dataframe: {missing}")
        
    split_index = int(len(df) * train_split_ratio)
    
    X_train = df[feature_cols].iloc[:split_index]
    X_test = df[feature_cols].iloc[split_index:]
    
    y_train = df[target_col].iloc[:split_index]
    y_test = df[target_col].iloc[split_index:]
    
    dates_train = df['date'].iloc[:split_index]
    dates_test = df['date'].iloc[split_index:]
    
    logger.info(f"✓ Train/Test split: Train({len(X_train)}), Test({len(X_test)})")
    
    return X_train, X_test, y_train, y_test, dates_train, dates_test