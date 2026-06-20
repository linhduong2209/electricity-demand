#!/usr/bin/env python3
"""
Main Entry Point - Singapore Energy Demand Forecast Pipeline
=============================================================

Complete ML pipeline based on CatBoost-PPSO paper method:
1. Data Loading (weather from Open-Meteo + demand)
2. Preprocessing & Feature Engineering (CCI, Season, Lags, Rolling windows)
3. Train/Test Split (Time-series chronological split)
4. Model Training & Hyperparameter Tuning (including PPSO)
5. Evaluation & Saving Summary

Usage:
    python main.py
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_all_data
from preprocessing import DataPreprocessor, prepare_train_test_split
from model import create_model
from trainer import ModelTrainer
from evaluator import RegressionEvaluator, FeatureImportanceVisualizer
from visualizer import run_eda

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if not os.path.exists(config_path):
        if os.path.exists("config.yaml"):
            config_path = "config.yaml"
        else:
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)
            
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.info(f"✓ Configuration loaded from {config_path}")
    return config


def main():
    print("="*70)
    print("STARTING ENERGY DEMAND FORECAST PIPELINE".center(70))
    print("="*70)

    # ========== STEP 1: Config ==========
    config = load_config()
    
    os.makedirs(config['output']['models_dir'], exist_ok=True)
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    os.makedirs(config['output']['plots_dir'], exist_ok=True)

    # ========== STEP 2: Data Loading ==========
    logger.info("\nSTEP 2: Loading Data...")
    demand_csv = config['data']['demand_csv_path']
    weather_cfg = config['data']['weather']
    
    df, data_stats = load_all_data(
        demand_csv_path=demand_csv,
        start_date=weather_cfg['start_date'],
        end_date=weather_cfg['end_date'],
        latitude=weather_cfg['latitude'],
        longitude=weather_cfg['longitude']
    )
    
    # ========== STEP 3: Preprocessing (Feature Engineering) ==========
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

    # ========== STEP 3.5: EDA Visualization (on processed data with day-type labels) ==========
    logger.info("\nSTEP 3.5: EDA Visualization...")
    plots_dir = config['output']['plots_dir']
    run_eda(df_processed, save_dir=plots_dir)

    # ========== STEP 4: Split Train/Test ==========
    logger.info("\nSTEP 4: Train/Test Split (Chronological)...")
    X_train, X_test, y_train, y_test, dates_train, dates_test = prepare_train_test_split(
        df_processed,
        feature_cols=feature_cols,
        target_col='demand_mwh',
        train_split_ratio=config['train_test_split']['train_ratio']
    )

    # ========== STEP 5: Training & Tuning ==========
    logger.info("\nSTEP 5: Model Training & Tuning...")
    
    model_types = ['baseline', 'linear', 'rf', 'xgb', 'catboost', 'catboost_ppso']
    trained_models = {}
    
    for m_type in model_types:
        logger.info(f"\n--- Training model: {m_type.upper()} ---")
        try:
            model_obj = create_model(m_type)
            
            use_scaler = True if m_type in ['linear'] else False
            trainer = ModelTrainer(model_obj, use_scaler=use_scaler)
            
            if m_type == 'catboost_ppso':
            
                ppso_bounds = {
                    'iterations':    (200, 1000),  
                    'learning_rate': (0.01,  0.15), 
                    'depth':         (4,     8),    
                    'l2_leaf_reg':   (1.0,  20.0)   
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

    # ========== STEP 6: Evaluation ==========
    logger.info("\nSTEP 6: Evaluating Models & Generating Leaderboard...")

    evaluator = RegressionEvaluator()
    metrics_report = []
    
    for m_type, m_data in trained_models.items():
        y_pred = m_data['trainer'].predict(X_test)
        
        model_metrics = evaluator.evaluate(y_test, y_pred, model_name=m_type)
        metrics_report.append(model_metrics)
        
    df_metrics = pd.DataFrame(metrics_report)
    df_metrics = df_metrics.sort_values(by='RMSE').reset_index(drop=True)
    
    metrics_csv_path = f"{config['output']['results_dir']}/model_leaderboard.csv"
    df_metrics.to_csv(metrics_csv_path, index=False)
    logger.info(f"✓ Leaderboard saved to {metrics_csv_path}")

    comparison_csv_path = f"{config['output']['results_dir']}/model_comparison.csv"
    df_metrics.to_csv(comparison_csv_path, index=False)
    logger.info(f"✓ Model comparison saved to {comparison_csv_path}")
    
    print("\n" + "="*50)
    print("MODEL LEADERBOARD (Sorted by RMSE)".center(50))
    print("="*50)
    print(df_metrics.to_string(index=False))
    print("="*50)

    if config['output']['save_models']:
        for model_name, model_data in trained_models.items():
            model_path = f"{config['output']['models_dir']}/{model_name}_model.pkl"
            model_data['trainer'].save_model(model_path)

    # ========== STEP 7: Feature Importance & Prediction Plots ==========
    logger.info("\nSTEP 7: Generating evaluation plots...")
    fi_viz = FeatureImportanceVisualizer()

    # Models that support feature importance
    importance_models = ['catboost', 'catboost_ppso', 'xgb', 'rf', 'linear']
    for m_type in importance_models:
        if m_type not in trained_models:
            continue
        m_data = trained_models[m_type]
        raw_model = m_data['model']
        # For CatBoostPPSOModel, the underlying CatBoost is stored in .final_model
        if hasattr(raw_model, 'final_model'):
            raw_model = raw_model.final_model
        save_path = os.path.join(plots_dir, f'feature_importance_{m_type}.png')
        fi_viz.plot_importance(
            model=raw_model,
            feature_names=feature_cols,
            model_name=m_type.upper(),
            top_n=len(feature_cols),
            save_path=save_path
        )
        if os.path.exists(save_path):
            logger.info(f"✓ Feature importance plot saved → {save_path}")

    # Actual vs Predicted plot for best model
    best_model_name = str(df_metrics.iloc[0]['Model'])
    if best_model_name in trained_models:
        y_pred_best = trained_models[best_model_name]['trainer'].predict(X_test)
        pred_save_path = os.path.join(plots_dir, f'predictions_{best_model_name}.png')
        evaluator.plot_predictions(
            y_test, y_pred_best,
            model_name=best_model_name.upper(),
            save_path=pred_save_path
        )
        logger.info(f"✓ Prediction plot (best model) saved → {pred_save_path}")
            
    summary = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': len(feature_cols),
        'best_model': str(df_metrics.iloc[0]['Model']),
        'best_mae': float(df_metrics.iloc[0]['MAE']),
        'best_rmse': float(df_metrics.iloc[0]['RMSE']),
        'best_r2': float(df_metrics.iloc[0]['R²']),
        'date_range': f"{df_processed['date'].min().date()} to {df_processed['date'].max().date()}"
    }
    
    summary_path = f"{config['output']['results_dir']}/summary.yaml"
    with open(summary_path, 'w', encoding='utf-8') as f:
        yaml.dump(summary, f)
    
    logger.info("✓ Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()