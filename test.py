#!/usr/bin/env python3
"""
Main Entry Point - Singapore Energy Demand Forecast Pipeline
=============================================================

Complete ML pipeline:
1. Data Loading (weather + demand)
2. Preprocessing (feature engineering)
3. Train/Test Split
4. Model Training & Tuning
5. Evaluation & Visualization

Usage:
    python main.py

Author: Energy Forecast Team
Date: 2024
"""

import sys
import os
import logging
import yaml
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_all_data
from preprocessing import DataPreprocessor, prepare_train_test_split
from model import (
    BaselineModel, StandardScalerWrapper, CatBoostPPSOModel,
    create_model
)
from trainer import ModelTrainer
from evaluator import RegressionEvaluator, FeatureImportanceVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to config file
    
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"✓ Configuration loaded from {config_path}")
    return config


def create_output_directories(config: dict):
    """Create necessary output directories."""
    dirs = [
        config['output']['models_dir'],
        config['output']['results_dir'],
        config['output']['plots_dir']
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    logger.info("✓ Output directories created")


def main():
    
    print("\n" + "="*70)
    print("SINGAPORE ENERGY DEMAND FORECAST - ML PIPELINE".center(70))
    print("="*70 + "\n")
    
    # ========== STEP 1: Load Configuration ==========
    logger.info("STEP 1: Loading Configuration...")
    config = load_config("configs/config.yaml")
    create_output_directories(config)
    
    # ========== STEP 2: Load Data ==========
    logger.info("\nSTEP 2: Loading Data...")
    demand_csv = config['data']['demand_csv_path']
    
    df, load_stats = load_all_data(
        demand_csv_path=demand_csv,
        start_date=config['data']['weather']['start_date'],
        end_date=config['data']['weather']['end_date']
    )
    
    logger.info(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
     # ========== STEP 3: Preprocessing ==========
    logger.info("\nSTEP 3: Feature Engineering & Preprocessing...")
    preprocessor = DataPreprocessor()
    
    df_processed = preprocessor.process(df, config)

    feature_cols = [
        'temperature', 'humidity', 'wind_speed', 'solar_radiation',
        
        'cci', 'season', 
        
        'day_of_week', 'month', 'is_weekend', 'is_holiday',
        
        'demand_lag_1', 'demand_lag_7', 
        'demand_rolling_mean_3', 'demand_rolling_mean_7',
        'demand_rolling_std_3', 'demand_rolling_std_7'
    ]
    
    logger.info(f"Processed dataset: {df_processed.shape[0]} rows ready for training.")

    # ========== STEP 4: Train/Test ==========
    logger.info("\nSTEP 4: Train/Test Split (Chronological)...")
    X_train, X_test, y_train, y_test, dates_train, dates_test = prepare_train_test_split(
        df_processed,
        feature_cols=feature_cols,
        target_col='demand_mwh',
        train_split_ratio=config['train_test_split']['train_ratio']
    )
    # ========== STEP 5: Huấn luyện & Tối ưu mô hình (Training & Tuning) ==========
    logger.info("\nSTEP 5: Model Training & Tuning...")
    
    # Danh sách các thuật toán chạy thử nghiệm so sánh
    model_types = ['baseline', 'linear', 'rf', 'xgb', 'catboost', 'catboost_ppso']
    trained_models = {}
    
    for m_type in model_types:
        logger.info(f"\n--- Training model: {m_type.upper()} ---")
        try:
            # Khởi tạo mô hình qua Factory (catboost đã được ép tham số cat_features tự động)
            model_obj = create_model(m_type)
            
            # Cấu hình scaler: Chỉ Linear Regression thực sự cần scaler cho lượng dữ liệu MWh lớn.
            # Các mô hình dạng cây quyết định (RF, XGB, CatBoost) hoạt động tốt hơn nếu không scale.
            use_scaler = True if m_type in ['linear'] else False
            trainer = ModelTrainer(model_obj, use_scaler=use_scaler)
            
            if m_type == 'catboost_ppso':
                # Truyền Search Space (param_bounds) cho PPSO dựa theo bài báo
                ppso_bounds = {
                    'iterations': (100, 1000),
                    'learning_rate': (0.01, 0.2),
                    'depth': (4, 10),
                    'l2_leaf_reg': (1, 10)
                }
                trainer.fit(X_train, y_train, param_bounds=ppso_bounds)
            else:
                trainer.fit(X_train, y_train)
            
            trained_models[m_type] = {
                'model': trainer.model,
                'trainer': trainer
            }
        except Exception as e:
            logger.error(f"Failed to train model {m_type}: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
