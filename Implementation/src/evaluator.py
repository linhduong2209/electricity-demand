import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class RegressionEvaluator:
    def __init__(self):
        self.metrics_history = {}

    def evaluate(self, y_true, y_pred, model_name="Model"):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        epsilon = np.finfo(np.float64).eps

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        rae = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true - np.mean(y_true))) + epsilon)
        
        wi_num = np.sum((y_pred - y_true)**2)
        wi_den = np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true)))**2)
        wi = 1 - (wi_num / (wi_den + epsilon))

        metrics = {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE (%)': mape,
            'RAE': rae,
            'WI': wi
        }
        self.metrics_history[model_name] = metrics
        return metrics

    def plot_predictions(self, y_true, y_pred, model_name="Model", save_path=None):
        plt.figure(figsize=(12, 6))
        plt.plot(np.array(y_true), label='Actual', alpha=0.7)
        plt.plot(np.array(y_pred), label='Predicted', alpha=0.7)
        plt.title(f'{model_name} - Actual vs Predicted')
        plt.ylabel('Demand')
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_residuals(self, y_true, y_pred, model_name="Model", save_path=None):
        residuals = np.array(y_true) - np.array(y_pred)
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.title(f'{model_name} - Residuals Distribution')
        plt.xlabel('Residual Error')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()


class FeatureImportanceVisualizer:
    @staticmethod
    def plot_importance(model, feature_names, model_name="Model", top_n=15, save_path=None):
        importances = None
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)

        if importances is None:
            return

        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x='Importance', 
            y='Feature', 
            data=df_importance, 
            palette='viridis', 
            hue='Feature', 
            legend=False
        )
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()