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
    
#     # ========== STEP 3: Preprocessing ==========
#     logger.info("\nSTEP 3: Feature Engineering & Preprocessing...")
#     preprocessor = DataPreprocessor()
#     df_processed, removed_rows = preprocessor.preprocess_pipeline(
#         df, 
#         add_classification=False
#     )
    
#     logger.info(f"Processed dataset: {df_processed.shape[0]} rows (removed {removed_rows} NaN rows)")
    
#     # ========== STEP 4: Train/Test Split ==========
#     logger.info("\nSTEP 4: Train/Test Split...")
#     feature_cols = preprocessor.get_feature_columns()
    
#     X_train, X_test, y_train, y_test, dates_train, dates_test = prepare_train_test_split(
#         df_processed,
#         feature_cols=feature_cols,
#         target_col='demand_mwh',
#         train_split_ratio=config['train_test_split']['train_ratio']
#     )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
