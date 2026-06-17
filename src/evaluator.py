"""
Evaluator Module
================

Model evaluation, metrics computation, and visualization.

Components:
- Regression metrics: R², RMSE, MAE, MAPE, RAE, WI
- Classification metrics: Accuracy, F1, Confusion Matrix
- Visualization: Leaderboard, predictions vs actual, feature importance

Author: Energy Forecast Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


class RegressionEvaluator:
    """
    Comprehensive regression model evaluation.
    
    Metrics:
    - MAE: Mean Absolute Error (↓ lower better)
    - RMSE: Root Mean Squared Error (↓ lower better)
    - R²: Coefficient of Determination (↑ higher better, max 1.0)
    - MAPE: Mean Absolute Percentage Error in % (↓ lower better)
    - RAE: Relative Absolute Error (↓ lower better)
    - WI: Willmott Index (↑ higher better, max 1.0)
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics_history = {}
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       model_name: str = "Model") -> Dict[str, float]:
        """
        Compute comprehensive regression metrics.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Model name for logging
        
        Returns:
            Dict: Dictionary of metrics
        """
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Percentage error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        # Relative Absolute Error (mean absolute error / mean absolute deviation)
        y_mean = np.mean(y_true)
        rae = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - y_mean))
        
        # Willmott Index (1 = perfect, 0 = poor)
        numerator = np.sum((y_true - y_pred)**2)
        denominator = np.sum((np.abs(y_pred - y_mean) + np.abs(y_true - y_mean))**2)
        wi = 1 - (numerator / (denominator + 1e-10))
        
        metrics = {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape,
            'RAE': rae,
            'WI': wi
        }
        
        return metrics
    
    def compare_models(self, predictions_dict: Dict[str, np.ndarray],
                      y_true: np.ndarray) -> pd.DataFrame:
        """
        Compare multiple models and return sorted leaderboard.
        
        Args:
            predictions_dict (Dict): {model_name: predictions}
            y_true (np.ndarray): True values
        
        Returns:
            pd.DataFrame: Leaderboard sorted by MAE
        """
        
        results = []
        for model_name, y_pred in predictions_dict.items():
            metrics = self.compute_metrics(y_true, y_pred, model_name)
            results.append(metrics)
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('MAE', ascending=True)
        
        self.metrics_history = {row['Model']: row.to_dict() 
                               for _, row in df_results.iterrows()}
        
        return df_results
    
    @staticmethod
    def plot_leaderboard(df_metrics: pd.DataFrame, top_n: int = 15):
        """
        Plot model comparison leaderboard.
        
        Args:
            df_metrics (pd.DataFrame): Metrics DataFrame from compare_models()
            top_n (int): Number of top models to display
        """
        
        df_top = df_metrics.head(top_n)
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Plot 1: Error metrics (RMSE vs MAE)
        ax1 = axes[0]
        x = np.arange(len(df_top))
        width = 0.35
        ax1.bar(x - width/2, df_top['RMSE'], width, label='RMSE', color='#e74c3c', alpha=0.8)
        ax1.bar(x + width/2, df_top['MAE'], width, label='MAE', color='#f39c12', alpha=0.8)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Error (MWh)')
        ax1.set_title('Error Metrics Comparison (↓ lower better)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_top['Model'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Goodness of fit (R² vs WI)
        ax2 = axes[1]
        ax2.scatter(df_top['R²'], df_top['WI'], s=100, alpha=0.7, color='#2ecc71')
        for idx, row in df_top.iterrows():
            ax2.annotate(row['Model'], (row['R²'], row['WI']), 
                        fontsize=8, ha='right', va='bottom')
        ax2.set_xlabel('R² Score')
        ax2.set_ylabel('Willmott Index')
        ax2.set_title('Fit Quality (↑ higher better)')
        ax2.set_xlim(-0.1, 1.1)
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: MAPE comparison
        ax3 = axes[2]
        ax3.barh(range(len(df_top)), df_top['MAPE'], color='#3498db', alpha=0.8)
        ax3.set_yticks(range(len(df_top)))
        ax3.set_yticklabels(df_top['Model'])
        ax3.set_xlabel('MAPE (%)')
        ax3.set_title('Mean Absolute Percentage Error (↓ lower better)')
        ax3.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                        dates: Optional[pd.DatetimeIndex] = None,
                        model_name: str = "Model"):
        """
        Plot actual vs predicted values over time.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            dates (pd.DatetimeIndex): Date index (optional)
            model_name (str): Model name for title
        """
        
        if dates is None:
            dates = np.arange(len(y_true))
        
        plt.figure(figsize=(14, 5))
        plt.plot(dates, y_true, 'o-', linewidth=2, markersize=4, 
                label='Actual Demand', color='steelblue', alpha=0.7)
        plt.plot(dates, y_pred, 's-', linewidth=2, markersize=4,
                label=f'Predicted ({model_name})', color='darkorange', alpha=0.7)
        
        plt.title(f'Electricity Demand: Actual vs Predicted\nModel: {model_name}',
                 fontsize=12, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Demand (MWh)')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                      model_name: str = "Model"):
        """
        Plot residual analysis: residuals vs predicted and residual distribution.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Model name for title
        """
        
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Residuals vs Predictions
        ax1 = axes[0]
        ax1.scatter(y_pred, residuals, alpha=0.6, color='steelblue', s=30)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Values (MWh)')
        ax1.set_ylabel('Residuals (MWh)')
        ax1.set_title(f'Residual Plot - {model_name}')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Residual Distribution
        ax2 = axes[1]
        ax2.hist(residuals, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax2.set_xlabel('Residuals (MWh)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Residual Distribution - {model_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class ClassificationEvaluator:
    """
    Classification model evaluation for demand level prediction.
    """
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       model_name: str = "Model") -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            y_true (np.ndarray): True class labels
            y_pred (np.ndarray): Predicted class labels
            model_name (str): Model name for logging
        
        Returns:
            Dict: Classification metrics
        """
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy,
            'F1-Score': f1
        }
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             model_name: str = "Model"):
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            class_names (List[str]): Class names (default: numeric labels)
            model_name (str): Model name for title
        """
        
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(cm))]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}', fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()


class FeatureImportanceVisualizer:
    """
    Visualize feature importance from tree-based models.
    """
    
    @staticmethod
    def plot_importance(model: object, feature_names: List[str],
                       model_name: str = "Model", top_n: int = 15):
        """
        Plot feature importance scores.
        
        Supports:
        - Tree-based models (Random Forest, XGBoost, CatBoost): feature_importances_
        - Linear models (Linear Regression, Ridge, Lasso): abs(coef_)
        - Custom models with get_feature_importance()
        
        Args:
            model (object): Trained model
            feature_names (List[str]): Feature column names
            model_name (str): Model name for title
            top_n (int): Number of top features to display
        """
        
        # Extract importances
        importances = None
        
        # Try different attribute names
        if hasattr(model, 'final_model') and hasattr(model.final_model, 'feature_importances_'):
            # Custom wrapper model
            importances = model.final_model.feature_importances_
        elif hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_)
            if len(importances.shape) > 1:
                importances = np.mean(importances, axis=0)
        elif hasattr(model, 'get_feature_importance'):
            # Custom get_feature_importance() method
            importance_df = model.get_feature_importance(top_n=top_n)
            return FeatureImportanceVisualizer._plot_importance_df(
                importance_df, model_name
            )
        
        if importances is None:
            logger.warning(f"Cannot extract feature importance from {model_name}")
            return
        
        # Create DataFrame
        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)
        
        FeatureImportanceVisualizer._plot_importance_df(df_importance, model_name)
    
    @staticmethod
    def _plot_importance_df(df_importance: pd.DataFrame, model_name: str):
        """Helper to plot importance DataFrame."""
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Importance', y='Feature', data=df_importance,
                        palette='viridis', orient='h')
        
        plt.title(f'Top {len(df_importance)} Feature Importance\nModel: {model_name}',
                 fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Importance Score', fontsize=11)
        plt.ylabel('Features', fontsize=11)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
        
        plt.tight_layout()
        plt.show()
