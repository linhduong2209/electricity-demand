"""
Preprocessing Module
====================

Feature engineering and data preparation for ML models.

Key responsibilities:
- Extract temporal features (day of week, month, holidays, etc.)
- Create lag and rolling features
- Generate interaction features
- Handle data cleaning and validation

Author: Energy Forecast Team
Date: 2024
"""

import pandas as pd
import numpy as np
import holidays
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles all data preprocessing and feature engineering tasks.
    
    Attributes:
        sg_holidays: Singapore holidays object
    """
    
    def __init__(self):
        """Initialize preprocessor with Singapore holidays."""
        self.sg_holidays = holidays.SG(years=[2023, 2024, 2025, 2026])
        logger.info("DataPreprocessor initialized")
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features based on date column.
        
        Features added:
        - is_holiday: Binary indicator for Singapore public holidays
        - is_weekend: Binary indicator for Saturday/Sunday
        - day_of_week: Day number (0=Monday, 6=Sunday)
        - month: Month number (1-12)
        - day_of_year: Day number in year (1-366)
        - week_of_year: ISO week number (1-53)
        - day_name: Day name string (Monday, Tuesday, ...)
        - is_day_before_holiday: Binary indicator
        - is_day_after_holiday: Binary indicator
        
        Args:
            df (pd.DataFrame): DataFrame with 'date' column (datetime)
        
        Returns:
            pd.DataFrame: DataFrame with added temporal features
        """
        df = df.copy()
        
        # Holiday indicators
        df['is_holiday'] = df['date'].isin(self.sg_holidays).astype(int)
        df['is_day_before_holiday'] = df['is_holiday'].shift(-1).fillna(0).astype(int)
        df['is_day_after_holiday'] = df['is_holiday'].shift(1).fillna(0).astype(int)
        
        # Weekend indicator
        df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Day names
        day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                     4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        df['day_name'] = df['day_of_week'].map(day_names)
        
        # Temporal features
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        
        n_holidays = df['is_holiday'].sum()
        logger.info(f"✓ Temporal features added ({n_holidays} holidays detected)")
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, 
                        lags: List[int] = [1, 7]) -> pd.DataFrame:
        """
        Create lag features for demand and weather variables.
        
        Lag features represent past values of a variable.
        Example: demand_lag_1 = demand from yesterday
        
        Args:
            df (pd.DataFrame): DataFrame with 'demand_mwh' and weather columns
            lags (List[int]): List of lag periods (default: [1, 7])
        
        Returns:
            pd.DataFrame: DataFrame with lag features added
        """
        df = df.copy()
        
        # Demand lags
        for lag in lags:
            df[f'demand_lag_{lag}'] = df['demand_mwh'].shift(lag)
        
        # Weather lags (typically only 1-day lag)
        if 1 in lags:
            df['temp_mean_lag_1'] = df['temp_mean'].shift(1)
            df['temp_max_lag_1'] = df['temp_max'].shift(1)
        
        logger.info(f"✓ Lag features added: {lags}")
        
        return df
    
    def add_rolling_features(self, df: pd.DataFrame,
                           windows: List[int] = [3, 7, 30]) -> pd.DataFrame:
        """
        Create rolling average (moving average) features.
        
        Rolling features smooth out noise and capture recent trends.
        Example: demand_rolling_7 = average demand over last 7 days
        
        Args:
            df (pd.DataFrame): DataFrame with numeric columns
            windows (List[int]): List of rolling window sizes (default: [3, 7, 30])
        
        Returns:
            pd.DataFrame: DataFrame with rolling features added
        """
        df = df.copy()
        
        # Demand rolling averages (shift by 1 to avoid future data leakage)
        for window in windows:
            df[f'demand_rolling_{window}'] = (
                df['demand_mwh'].shift(1).rolling(window=window, min_periods=1).mean()
            )
        
        # Weather rolling averages (no shift needed - weather is known today)
        if 3 in windows:
            df['temp_mean_rolling_3'] = df['temp_mean'].rolling(window=3, min_periods=1).mean()
            df['humidity_rolling_3'] = df['humidity_max'].rolling(window=3, min_periods=1).mean()
        
        logger.info(f"✓ Rolling features added: windows {windows}")
        
        return df
    
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features capturing combined effects of variables.
        
        Features:
        - temp_humidity_interaction: Combined effect of temperature and humidity
                                     (higher values = more discomfort = higher AC load)
        
        Args:
            df (pd.DataFrame): DataFrame with relevant columns
        
        Returns:
            pd.DataFrame: DataFrame with interaction features added
        """
        df = df.copy()
        
        # Temperature-humidity interaction (heat index proxy)
        # Higher temp * high humidity = more extreme conditions = higher cooling demand
        df['temp_humidity_interaction'] = (df['temp_mean'] * df['humidity_max']) / 100
        
        logger.info("✓ Interaction features added")
        
        return df
    
    def create_target_classification(self, df: pd.DataFrame,
                                    lower_percentile: float = 33.33,
                                    upper_percentile: float = 66.67) -> pd.DataFrame:
        """
        Create target variable for classification task.
        
        Categorizes demand into three levels:
        - Low: demand < 33.33rd percentile
        - Medium: 33.33rd ≤ demand < 66.67th percentile  
        - High: demand ≥ 66.67th percentile
        
        Args:
            df (pd.DataFrame): DataFrame with 'demand_mwh' column
            lower_percentile (float): Lower percentile threshold
            upper_percentile (float): Upper percentile threshold
        
        Returns:
            pd.DataFrame: DataFrame with added 'demand_class' column
        """
        df = df.copy()
        
        lower_thresh = df['demand_mwh'].quantile(lower_percentile / 100)
        upper_thresh = df['demand_mwh'].quantile(upper_percentile / 100)
        
        df['demand_class'] = pd.cut(
            df['demand_mwh'],
            bins=[df['demand_mwh'].min() - 1, lower_thresh, upper_thresh, df['demand_mwh'].max() + 1],
            labels=['Low', 'Medium', 'High']
        )
        
        class_dist = df['demand_class'].value_counts()
        logger.info(f"✓ Classification target created:\n{class_dist}")
        
        return df
    
    def preprocess_pipeline(self, df: pd.DataFrame,
                          add_classification: bool = False) -> Tuple[pd.DataFrame, int]:
        """
        Complete preprocessing pipeline: temporal → lag → rolling → interaction features.
        
        Args:
            df (pd.DataFrame): Raw data with 'date', 'demand_mwh', and weather columns
            add_classification (bool): Whether to add classification target
        
        Returns:
            Tuple[pd.DataFrame, int]: 
                - Processed DataFrame (rows with NaN removed)
                - Number of rows removed due to NaN
        
        Example:
            >>> df, removed_rows = preprocessor.preprocess_pipeline(raw_df)
            >>> print(f"Processed {len(df)} rows, removed {removed_rows} due to NaN")
        """
        original_shape = df.shape[0]
        
        # Apply all transformations
        df = self.add_temporal_features(df)
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        df = self.add_interaction_features(df)
        
        if add_classification:
            df = self.create_target_classification(df)
        
        # Remove rows with NaN (created by lag and rolling features)
        df_clean = df.dropna().reset_index(drop=True)
        removed_rows = original_shape - len(df_clean)
        
        logger.info(f"✓ Preprocessing complete: {len(df_clean)} rows (removed {removed_rows} NaN rows)")
        
        return df_clean, removed_rows
    
    def get_feature_columns(self, 
                           include_interaction: bool = True,
                           include_lag: bool = True,
                           include_rolling: bool = True) -> List[str]:
        """
        Get list of all engineered features.
        
        Args:
            include_interaction (bool): Include interaction features
            include_lag (bool): Include lag features
            include_rolling (bool): Include rolling features
        
        Returns:
            List[str]: List of feature column names for model input
        """
        features = [
            # Weather features
            'temp_max', 'temp_min', 'temp_mean', 'humidity_max',
            'precipitation', 'wind_speed', 'apparent_temp_max',
            
            # Temporal features
            'is_weekend', 'is_holiday', 'day_of_week',
            'month', 'day_of_year', 'week_of_year',
            'is_day_before_holiday', 'is_day_after_holiday',
        ]
        
        if include_lag:
            features.extend([
                'demand_lag_1', 'demand_lag_7',
                'temp_mean_lag_1', 'temp_max_lag_1',
            ])
        
        if include_rolling:
            features.extend([
                'demand_rolling_7', 'demand_rolling_30',
                'temp_mean_rolling_3', 'humidity_rolling_3',
            ])
        
        if include_interaction:
            features.append('temp_humidity_interaction')
        
        return features
    
    def get_categorical_features(self, feature_cols: List[str]) -> List[int]:
        """
        Identify categorical feature indices for CatBoost.
        
        CatBoost handles categorical features natively - no One-Hot Encoding needed!
        Categorical features in this dataset are temporal features:
        - day_of_week: Integer 0-6
        - month: Integer 1-12
        - week_of_year: Integer 1-53
        - is_weekend, is_holiday, is_day_before_holiday, is_day_after_holiday: Binary
        
        Args:
            feature_cols (List[str]): List of all feature column names
        
        Returns:
            List[int]: Indices of categorical features in feature_cols
        """
        cat_feature_names = [
            'day_of_week', 'month', 'week_of_year',
            'is_weekend', 'is_holiday', 
            'is_day_before_holiday', 'is_day_after_holiday'
        ]
        
        # Map feature names to indices
        cat_indices = [
            i for i, col in enumerate(feature_cols) 
            if col in cat_feature_names
        ]
        
        logger.info(f"✓ Categorical features identified: {cat_feature_names} (indices: {cat_indices})")
        
        return cat_indices if cat_indices else None


def prepare_train_test_split(df: pd.DataFrame,
                            feature_cols: List[str],
                            target_col: str = 'demand_mwh',
                            train_split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                                      pd.Series, pd.Series,
                                                                      pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Prepare data for model training with chronological time-series split.
    
    Chronological split is critical for time-series to avoid forward-looking bias.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame with all features
        feature_cols (List[str]): List of feature column names
        target_col (str): Name of target column (default: 'demand_mwh')
        train_split_ratio (float): Fraction for training (default: 0.8)
    
    Returns:
        Tuple containing:
            - X_train, X_test: Feature DataFrames
            - y_train, y_test: Target Series
            - dates_train, dates_test: Date indices
    
    Raises:
        ValueError: If required columns missing or invalid split ratio
    """
    
    if not all(col in df.columns for col in feature_cols + [target_col]):
        raise ValueError(f"Required columns not found in DataFrame")
    
    if not (0 < train_split_ratio < 1):
        raise ValueError(f"train_split_ratio must be between 0 and 1, got {train_split_ratio}")
    
    split_index = int(len(df) * train_split_ratio)
    
    X_train = df[feature_cols].iloc[:split_index]
    X_test = df[feature_cols].iloc[split_index:]
    
    y_train = df[target_col].iloc[:split_index]
    y_test = df[target_col].iloc[split_index:]
    
    dates_train = df['date'].iloc[:split_index]
    dates_test = df['date'].iloc[split_index:]
    
    logger.info(
        f"✓ Train/Test split (ratio={train_split_ratio}):\n"
        f"  Train: {len(X_train)} samples ({dates_train.iloc[0].date()} → {dates_train.iloc[-1].date()})\n"
        f"  Test:  {len(X_test)} samples ({dates_test.iloc[0].date()} → {dates_test.iloc[-1].date()})"
    )
    
    return X_train, X_test, y_train, y_test, dates_train, dates_test
