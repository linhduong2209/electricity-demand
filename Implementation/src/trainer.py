import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

logger = logging.getLogger(__name__)


class ModelTrainer:
    
    def __init__(self, model: object, use_scaler: bool = True):
        self.model = model
        self.scaler = StandardScaler() if use_scaler else None
        self.training_history = {}
        self.feature_names = None
        logger.info(f"Trainer initialized with model: {type(model).__name__}")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
           X_val: Optional[pd.DataFrame] = None, 
           y_val: Optional[pd.Series] = None, **kwargs) -> Dict[str, float]:
        
        self.feature_names = X_train.columns.tolist()
        
        # Scale features if using StandardScaler
        X_train_processed = X_train.copy()
        if self.scaler is not None:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_train_processed = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        # Train model
        logger.info(f"Training {type(self.model).__name__}...")
        self.model.fit(X_train_processed, y_train, **kwargs)
        logger.info("✓ Training complete")
        
        # Compute training metrics
        y_train_pred = self.model.predict(X_train_processed)
        train_metrics = self._compute_metrics(y_train, y_train_pred, prefix='train')
        
        self.training_history.update(train_metrics)
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            X_val_processed = X_val.copy()
            if self.scaler is not None:
                X_val_scaled = self.scaler.transform(X_val)
                X_val_processed = pd.DataFrame(X_val_scaled, columns=X_val.columns)
            
            y_val_pred = self.model.predict(X_val_processed)
            val_metrics = self._compute_metrics(y_val, y_val_pred, prefix='val')
            self.training_history.update(val_metrics)
        
        return self.training_history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_processed = X.copy()
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X_processed = pd.DataFrame(X_scaled, columns=X.columns)
        
        return self.model.predict(X_processed)
    
    @staticmethod
    def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray, 
                        prefix: str = 'test') -> Dict[str, float]:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        return {
            f'{prefix}_mae': mae,
            f'{prefix}_rmse': rmse,
            f'{prefix}_r2': r2,
            f'{prefix}_mape': mape
        }
    
    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'history': self.training_history
            }, f)
        logger.info(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.training_history = data['history']
        
        logger.info(f"✓ Model loaded from {filepath}")


class HyperparameterTuner:
    def __init__(self, model: object, param_grid: Dict[str, list],
                search_method: str = 'grid', cv_splits: int = 3,
                scoring: str = 'neg_mean_absolute_error'):
        self.model = model
        self.param_grid = param_grid
        self.search_method = search_method
        self.cv_splits = cv_splits
        self.scoring = scoring
        self.best_model = None
        self.best_params = None
        self.cv_results = None
    
    def tune(self, X_train: pd.DataFrame, y_train: pd.Series,
            time_series: bool = True) -> 'HyperparameterTuner':
        
        # Choose CV strategy
        if time_series:
            cv = TimeSeriesSplit(n_splits=self.cv_splits)
        else:
            cv = self.cv_splits
        
        # Choose search method
        if self.search_method == 'grid':
            search = GridSearchCV(
                estimator=self.model,
                param_grid=self.param_grid,
                cv=cv,
                scoring=self.scoring,
                verbose=1,
                n_jobs=-1
            )
        elif self.search_method == 'random':
            search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=self.param_grid,
                n_iter=10,
                cv=cv,
                scoring=self.scoring,
                verbose=1,
                n_jobs=-1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown search method: {self.search_method}")
        
        logger.info(f"Running {self.search_method.upper()} search...")
        search.fit(X_train, y_train)
        
        self.best_model = search.best_estimator_
        self.best_params = search.best_params_
        self.cv_results = pd.DataFrame(search.cv_results_)
        
        logger.info(f"✓ Best parameters: {self.best_params}")
        logger.info(f"✓ Best CV score: {search.best_score_:.4f}")
        
        return self
    
    def get_best_model(self) -> object:
        """Get best model after tuning."""
        if self.best_model is None:
            raise RuntimeError("Tuning not completed. Call tune() first.")
        return self.best_model
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get tuning results as DataFrame."""
        if self.cv_results is None:
            raise RuntimeError("Tuning not completed. Call tune() first.")
        return self.cv_results
