"""
Model Module
============

Model definitions including standard ML models and advanced optimization techniques.

Components:
- Standard models: Linear Regression, Random Forest, XGBoost, CatBoost
- Advanced optimization: Particle Swarm Optimization (PSO) for hyperparameter tuning
- PPSO (Periodic Particle Swarm Optimization) for CatBoost

Author: Energy Forecast Team
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Callable
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import catboost as cb

logger = logging.getLogger(__name__)


class BaselineModel:
    """
    Simple baseline: predict today's demand equals yesterday's demand (lag-1).
    
    Used to establish a minimum performance threshold for comparison.
    """
    
    def __init__(self):
        """Initialize baseline model."""
        self.name = "Baseline (Lag-1)"
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return yesterday's demand (lag-1 feature) as prediction.
        
        Args:
            X (pd.DataFrame): Feature DataFrame with 'demand_lag_1' column
        
        Returns:
            np.ndarray: Predictions
        """
        if 'demand_lag_1' not in X.columns:
            raise ValueError("Feature 'demand_lag_1' not found in X")
        return X['demand_lag_1'].values
    
    def __repr__(self):
        return self.name


class StandardScalerWrapper:
    """
    Wrapper for StandardScaler to handle pandas DataFrames while preserving column names.
    
    Attributes:
        scaler: sklearn StandardScaler instance
        feature_names: List of column names
    """
    
    def __init__(self):
        """Initialize scaler wrapper."""
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit scaler and transform data.
        
        Args:
            X (pd.DataFrame): Input features
        
        Returns:
            np.ndarray: Scaled features
        """
        self.feature_names = X.columns.tolist()
        return self.scaler.fit_transform(X)
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted scaler.
        
        Args:
            X (pd.DataFrame): Input features
        
        Returns:
            np.ndarray: Scaled features
        """
        return self.scaler.transform(X)
    
    def get_scaler(self):
        """Get underlying StandardScaler object."""
        return self.scaler


class PPSO:
    """
    Periodic Particle Swarm Optimization (PPSO).
    
    Advanced optimization algorithm for hyperparameter tuning.
    Combines PSO with periodic constraints to prevent premature convergence.
    Features: Early stopping, random seeding, convergence tolerance.
    
    Attributes:
        obj_func: Objective function to minimize
        bounds: Parameter bounds [[min1, max1], [min2, max2], ...]
        num_particles: Number of particles in swarm
        max_iter: Maximum iterations
        early_stopping_rounds: Stop if no improvement after N iterations
        random_state: Random seed for reproducibility
    """
    
    def __init__(self, obj_func: Callable, bounds: List[Tuple], 
                 num_particles: int = 15, max_iter: int = 20,
                 early_stopping_rounds: int = 10, random_state: int = 42):
        """
        Initialize PPSO optimizer.
        
        Args:
            obj_func: Objective function f(params) → score (minimize)
            bounds: List of [min, max] bounds for each parameter
            num_particles: Number of particles in swarm (default: 15)
            max_iter: Maximum iterations (default: 20)
            early_stopping_rounds: Stop if no improvement (default: 10)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.obj_func = obj_func
        self.bounds = np.array(bounds, dtype=float)
        # Ensure bounds is 2D
        if self.bounds.ndim == 1:
            self.bounds = self.bounds.reshape(1, -1)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.dim = len(bounds)
        
        # Set random seed for reproducibility
        np.random.seed(random_state)
    
    def optimize(self) -> np.ndarray:
        """
        Run PPSO optimization with early stopping.
        
        Returns:
            np.ndarray: Best parameter values found
        """
        # Initialize particles with fixed seed
        X = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                            (self.num_particles, self.dim))
        V = np.zeros_like(X)
        Theta = np.random.uniform(0, 2 * np.pi, (self.num_particles, self.dim))

        # Evaluate initial positions
        pbest_X = np.copy(X)
        pbest_scores = np.array([self.obj_func(x) for x in X])

        gbest_idx = np.argmin(pbest_scores)
        gbest_X = np.copy(pbest_X[gbest_idx])
        gbest_score = pbest_scores[gbest_idx]

        logger.info("--- Starting PPSO Optimization ---")
        
        # Early stopping tracking
        iterations_without_improvement = 0
        best_score_history = []
        
        # Main optimization loop
        for t in range(self.max_iter):
            for i in range(self.num_particles):
                # ===== PPSO PHASOR UPDATE (Key difference from standard PSO) =====
                # PPSO uses periodic angle (phase) to balance exploration vs exploitation
                # Instead of linear inertia weight w, PPSO modulates velocity using cos/sin
                
                cos_t = np.cos(Theta[i])  # Cosine component
                sin_t = np.sin(Theta[i])  # Sine component
                
                # These control the balance between:
                # - (abs(cos_t)**2) * sin_t: Exploration (cognitive)
                # - (abs(sin_t)**2) * cos_t: Exploitation (social)
                # As phase angle changes, this balance shifts periodically

                # ===== VELOCITY UPDATE (Periodic Particle Swarm) =====
                # v_i = [|cos(θ)|² · sin(θ) · (pbest - x)] + [|sin(θ)|² · cos(θ) · (gbest - x)]
                # This is different from PSO: w·v + c1·r1·(pbest - x) + c2·r2·(gbest - x)
                
                term1 = (np.abs(cos_t)**2) * sin_t * (pbest_X[i] - X[i])  # Cognitive component
                term2 = (np.abs(sin_t)**2) * cos_t * (gbest_X - X[i])     # Social component
                V[i] = term1 + term2

                # ===== VELOCITY CLAMPING (Prevent velocity explosion) =====
                # v_max is also modulated by phase angle to adapt search intensity
                v_max = (np.abs(cos_t)**2) * (self.bounds[:, 1] - self.bounds[:, 0])
                V[i] = np.clip(V[i], -v_max, v_max)

                # ===== POSITION UPDATE (Chronological) =====
                X[i] = np.clip(X[i] + V[i], self.bounds[:, 0], self.bounds[:, 1])
                
                # ===== PHASE ANGLE UPDATE (Periodic rotation) =====
                # θ_i = θ_i + |cos(θ) + sin(θ)| · 2π
                # This creates periodic oscillation in the phase angle
                Theta[i] = Theta[i] + np.abs(cos_t + sin_t) * (2 * np.pi)

                # ===== EVALUATE NEW POSITION =====
                # Fitness = validation RMSE from cross-validation (NOT test set!)
                score = self.obj_func(X[i])

                # Update personal best
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_X[i] = np.copy(X[i])
                    
                    # Update global best
                    if score < gbest_score:
                        gbest_score = score
                        gbest_X = np.copy(X[i])
                        iterations_without_improvement = 0  # Reset counter
                    else:
                        iterations_without_improvement += 1
                else:
                    iterations_without_improvement += 1

            best_score_history.append(gbest_score)
            logger.info(f"  Iteration {t+1}/{self.max_iter} | Best Score: {gbest_score:.4f} | No improve: {iterations_without_improvement}")
            
            # Early stopping: If no improvement for N iterations, stop
            if iterations_without_improvement >= self.early_stopping_rounds:
                logger.info(f"  ⚠ Early stopping at iteration {t+1}: No improvement for {self.early_stopping_rounds} iterations")
                break

        return gbest_X


class CatBoostPPSOModel:
    """
    CatBoost Regressor optimized with PPSO for hyperparameter tuning.
    
    Combines CatBoost's efficiency with PPSO's advanced optimization.
    
    KEY PRINCIPLE: 
    - Fitness function uses VALIDATION set (not test set!)
    - PPSO searches hyperparameter space: [iterations, depth, learning_rate, l2_leaf_reg]
    - Each particle represents a set of hyperparameters
    - Objective: Minimize CV RMSE on validation folds
    
    Attributes:
        cat_features: List of categorical feature indices (for CatBoost native handling)
        k_folds: K-fold cross-validation splits (default: 3)
        best_params: Dictionary of optimized hyperparameters
        final_model: Trained CatBoost model
    """
    
    def __init__(self, cat_features: List[int] = None, k_folds: int = 3):
        """
        Initialize CatBoost-PPSO model.
        
        Args:
            cat_features: List of categorical feature indices (CatBoost handles them natively!)
                         DO NOT use One-Hot Encoding with CatBoost
                         Example: [3, 5, 7] means columns at indices 3, 5, 7 are categorical
            k_folds: K-fold cross-validation splits (default: 3)
        """
        self.cat_features = cat_features
        self.k_folds = k_folds
        self.best_params = {}
        self.final_model = None
        logger.info(f"CatBoostPPSOModel initialized with {k_folds}-fold CV and cat_features={cat_features}")
    
    def _objective_function(self, params: np.ndarray) -> float:
        """
        Objective function for PPSO optimization.
        
        CRITICAL POINT: Uses VALIDATION set, NOT test set!
        Each particle position represents [iterations, depth, learning_rate, l2_leaf_reg]
        
        Algorithm:
        1. Extract hyperparameters from particle position
        2. Type casting: iterations and depth MUST be integers
        3. K-fold cross-validation on TRAINING set only
        4. For each fold:
           - Train on fold_train
           - Validate on fold_val
           - Compute RMSE on fold_val (NOT fold_train to avoid overfitting)
        5. Return mean RMSE as fitness score
        
        Args:
            params: Particle position [iterations, depth, learning_rate, l2_leaf_reg]
                   All values are floats from PPSO
        
        Returns:
            float: Mean validation RMSE (to minimize)
        
        Raises:
            Exception: If validation shapes are incorrect
        """
        # ===== TYPE CASTING: PPSO generates floats, CatBoost needs ints =====
        n_estimators = int(np.round(params[0]))  # iterations: must be int
        depth = int(np.round(params[1]))          # depth: must be int
        learning_rate = float(params[2])           # learning_rate: float is OK
        l2_leaf_reg = float(params[3])             # l2_leaf_reg: float is OK

        # Validate ranges (defensive programming)
        n_estimators = max(50, min(1000, n_estimators))  # Clamp to reasonable range
        depth = max(2, min(15, depth))
        learning_rate = max(0.001, min(0.5, learning_rate))
        l2_leaf_reg = max(0.1, min(50.0, l2_leaf_reg))

        # ===== K-FOLD CROSS-VALIDATION ON TRAINING SET ONLY =====
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        cv_rmse_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.X_train)):
            # Split training data into fold_train and fold_val
            X_fold_train = self.X_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            y_fold_val = self.y_train.iloc[val_idx]

            # ===== TRAIN on fold_train, EVALUATE on fold_val =====
            model = cb.CatBoostRegressor(
                iterations=n_estimators,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                cat_features=self.cat_features,  # CatBoost handles categorical features natively
                loss_function='RMSE',
                verbose=0,
                random_state=42
            )
            
            # Train on fold_train
            model.fit(X_fold_train, y_fold_train, 
                     eval_set=(X_fold_val, y_fold_val),
                     early_stopping_rounds=20, 
                     verbose=0)
            
            # Predict on fold_val (NOT fold_train!)
            y_fold_pred = model.predict(X_fold_val)
            
            # Compute RMSE on validation fold
            fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_fold_pred))
            cv_rmse_scores.append(fold_rmse)
            
            logger.debug(f"  Fold {fold_idx+1}/{self.k_folds}: RMSE = {fold_rmse:.2f}")

        # ===== AVERAGE CV RMSE IS FITNESS SCORE =====
        mean_cv_rmse = np.mean(cv_rmse_scores)
        
        return mean_cv_rmse
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
           param_bounds: List[Tuple],
           particles: int = 15, max_iter: int = 20,
           early_stopping_rounds: int = 10, random_state: int = 42):
        """
        Optimize hyperparameters using PPSO and train final model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            param_bounds: [[iter_min, iter_max], [depth_min, depth_max], ...]
            particles (int): Number of particles in swarm
            max_iter (int): Maximum PPSO iterations
            early_stopping_rounds (int): Stop if no improvement for N iterations
            random_state (int): Random seed for reproducibility
        """
        self.X_train = X_train
        self.y_train = y_train

        # Run PPSO optimization with early stopping
        optimizer = PPSO(
            obj_func=self._objective_function,
            bounds=param_bounds,
            num_particles=particles,
            max_iter=max_iter,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state
        )
        best_X = optimizer.optimize()

        # Store best parameters
        self.best_params = {
            'iterations': int(best_X[0]),
            'depth': int(best_X[1]),
            'learning_rate': float(best_X[2]),
            'l2_leaf_reg': float(best_X[3])
        }
        
        logger.info(f"✓ Best parameters found: {self.best_params}")
        
        # Train final model with best parameters (with random_state for reproducibility)
        self.final_model = cb.CatBoostRegressor(
            **self.best_params,
            cat_features=self.cat_features,
            loss_function='RMSE',
            random_state=random_state,
            verbose=0
        )
        self.final_model.fit(X_train, y_train)
        logger.info("✓ Final model trained successfully")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            X (pd.DataFrame): Features to predict
        
        Returns:
            np.ndarray: Predictions
        """
        if self.final_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.final_model.predict(X)
    
    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            top_n (int): Number of top features to return
        
        Returns:
            pd.DataFrame: DataFrame with feature names and importance scores
        """
        if self.final_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        importances = self.final_model.feature_importances_
        feature_names = list(range(len(importances)))
        
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)
        
        return df


def create_model(model_type: str, **kwargs) -> object:
    """
    Factory function to create model instances.
    
    Args:
        model_type (str): Type of model ('baseline', 'linear', 'rf', 'xgb', 'catboost', 'catboost_ppso')
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        object: Model instance
    
    Raises:
        ValueError: If model_type not recognized
    """
    
    models = {
        'baseline': BaselineModel,
        'linear': LinearRegression,
        'rf': RandomForestRegressor,
        'xgb': xgb.XGBRegressor,
        'catboost': cb.CatBoostRegressor,
        'catboost_ppso': CatBoostPPSOModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    ModelClass = models[model_type]
    
    # Filter kwargs based on model constructor
    if model_type == 'baseline':
        return ModelClass()
    elif model_type == 'linear':
        return ModelClass()
    elif model_type == 'rf':
        return ModelClass(random_state=42, n_jobs=-1, **kwargs)
    elif model_type == 'xgb':
        return ModelClass(random_state=42, n_jobs=-1, **kwargs)
    elif model_type == 'catboost':
        return ModelClass(verbose=0, **kwargs)
    elif model_type == 'catboost_ppso':
        return ModelClass(**kwargs)
