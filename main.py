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
    """
    Main pipeline execution.
    
    Steps:
    1. Load configuration
    2. Load and preprocess data
    3. Train multiple models
    4. Evaluate and compare
    5. Save results and models
    """
    
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
    df_processed, removed_rows = preprocessor.preprocess_pipeline(
        df, 
        add_classification=False
    )
    
    logger.info(f"Processed dataset: {df_processed.shape[0]} rows (removed {removed_rows} NaN rows)")
    
    # ========== STEP 4: Train/Test Split ==========
    logger.info("\nSTEP 4: Train/Test Split...")
    feature_cols = preprocessor.get_feature_columns()
    
    X_train, X_test, y_train, y_test, dates_train, dates_test = prepare_train_test_split(
        df_processed,
        feature_cols=feature_cols,
        target_col='demand_mwh',
        train_split_ratio=config['train_test_split']['train_ratio']
    )
    
    # ========== STEP 5: Model Training ==========
    logger.info("\nSTEP 5: Training Models...")
    trained_models = {}
    predictions_dict = {}
    
    for model_name in config['models']['train_models']:
        logger.info(f"\n--- Training {model_name.upper()} ---")
        
        # Create model
        if model_name == 'baseline':
            model = BaselineModel()
            predictions_dict[model_name] = model.predict(X_test)
            trained_models[model_name] = {'model': model}
        
        elif model_name == 'linear':
            trainer = ModelTrainer(create_model('linear'), use_scaler=True)
            trainer.fit(X_train, y_train)
            predictions_dict[model_name] = trainer.predict(X_test)
            trained_models[model_name] = {'trainer': trainer}
        
        elif model_name == 'rf':
            model = create_model('rf', n_estimators=200, max_depth=15)
            model.fit(X_train, y_train)
            predictions_dict[model_name] = model.predict(X_test)
            trained_models[model_name] = {'model': model}
            logger.info("✓ Random Forest trained")
        
        elif model_name == 'xgb':
            model = create_model('xgb', n_estimators=200, max_depth=6, learning_rate=0.05)
            model.fit(X_train, y_train)
            predictions_dict[model_name] = model.predict(X_test)
            trained_models[model_name] = {'model': model}
            logger.info("✓ XGBoost trained")
        
        elif model_name == 'catboost':
            model = create_model('catboost', iterations=200, depth=6, learning_rate=0.05)
            model.fit(X_train, y_train)
            predictions_dict[model_name] = model.predict(X_test)
            trained_models[model_name] = {'model': model}
            logger.info("✓ CatBoost trained")
        
        elif model_name == 'catboost_ppso':
            ppso_config = config['models']['catboost_ppso']
            # Convert param_bounds dict to list of lists
            param_bounds = list(ppso_config['param_bounds'].values())
            
            # This improves performance and avoids data leakage
            preprocessor = DataPreprocessor()  # Use preprocessor to get cat features
            feature_cols = preprocessor.get_feature_columns()
            cat_features = preprocessor.get_categorical_features(feature_cols)
            
            logger.info(f"CatBoost-PPSO: Using {len(cat_features) if cat_features else 0} categorical features")
            
            ppso_model = CatBoostPPSOModel(cat_features=cat_features, k_folds=3)
            ppso_model.fit(
                X_train=X_train,
                y_train=y_train,
                param_bounds=param_bounds,
                particles=ppso_config['particles'],
                max_iter=ppso_config['max_iterations'],
                early_stopping_rounds=ppso_config.get('early_stopping_rounds', 10),
                random_state=42
            )
            predictions_dict[model_name] = ppso_model.predict(X_test)
            trained_models[model_name] = {'model': ppso_model}
    
    # ========== STEP 6: Evaluation ==========
    logger.info("\nSTEP 6: Model Evaluation & Comparison...")
    evaluator = RegressionEvaluator()
    
    df_metrics = evaluator.compare_models(
        predictions_dict=predictions_dict,
        y_true=y_test.values
    )
    
    # Display leaderboard
    print("\n" + "🏆 MODEL LEADERBOARD 🏆".center(80))
    print("="*85)
    print(df_metrics.to_string(index=False))
    print("="*85 + "\n")
    
    # Save leaderboard
    df_metrics.to_csv(f"{config['output']['results_dir']}/model_comparison.csv", index=False)
    logger.info(f"✓ Leaderboard saved to {config['output']['results_dir']}/model_comparison.csv")
    
    # ========== STEP 7: Visualization ==========
    logger.info("\nSTEP 7: Generating Visualizations...")
    
    if config['evaluation']['plot_leaderboard']:
        evaluator.plot_leaderboard(df_metrics, top_n=10)
    
    if config['evaluation']['plot_predictions']:
        best_model_name = df_metrics.iloc[0]['Model']
        best_predictions = predictions_dict[best_model_name]
        evaluator.plot_predictions(
            y_true=y_test.values,
            y_pred=best_predictions,
            dates=dates_test,
            model_name=best_model_name
        )
    
    if config['evaluation']['plot_residuals']:
        best_model_name = df_metrics.iloc[0]['Model']
        best_predictions = predictions_dict[best_model_name]
        evaluator.plot_residuals(
            y_true=y_test.values,
            y_pred=best_predictions,
            model_name=best_model_name
        )
    
    if config['evaluation']['plot_feature_importance']:
        for model_name, model_data in trained_models.items():
            try:
                if 'trainer' in model_data:
                    model = model_data['trainer'].model
                else:
                    model = model_data['model']
                
                if model_name != 'baseline':
                    FeatureImportanceVisualizer.plot_importance(
                        model=model,
                        feature_names=feature_cols,
                        model_name=model_name,
                        top_n=config['evaluation']['top_features']
                    )
            except Exception as e:
                logger.warning(f"Could not plot importance for {model_name}: {e}")
    
    # ========== STEP 8: Save Results ==========
    logger.info("\nSTEP 8: Saving Results...")
    
    if config['output']['save_models']:
        for model_name, model_data in trained_models.items():
            if 'trainer' in model_data:
                model_path = f"{config['output']['models_dir']}/{model_name}_model.pkl"
                model_data['trainer'].save_model(model_path)
    
    # Save training summary
    summary = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': len(feature_cols),
        'best_model': df_metrics.iloc[0]['Model'],
        'best_mae': float(df_metrics.iloc[0]['MAE']),
        'best_rmse': float(df_metrics.iloc[0]['RMSE']),
        'best_r2': float(df_metrics.iloc[0]['R²']),
        'date_range': f"{df['date'].min()} to {df['date'].max()}"
    }
    
    with open(f"{config['output']['results_dir']}/summary.yaml", 'w') as f:
        yaml.dump(summary, f)
    
    logger.info(f"✓ Results saved to {config['output']['results_dir']}/")
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY ✓".center(70))
    print("="*70)
    print(f"\nBest Model: {summary['best_model']}")
    print(f"Test MAE: {summary['best_mae']:.2f} MWh")
    print(f"Test RMSE: {summary['best_rmse']:.2f} MWh")
    print(f"Test R²: {summary['best_r2']:.4f}")
    print(f"\nResults saved to: {config['output']['results_dir']}/")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
