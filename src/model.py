import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Callable
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import catboost as cb

logger = logging.getLogger(__name__)


class BaselineModel:
    
    def __init__(self):
        """Initialize baseline model."""
        self.name = "Baseline (Lag-1)"
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if 'demand_lag_1' not in X.columns:
            raise ValueError("Feature 'demand_lag_1' not found in X")
        return X['demand_lag_1'].values
    
    def __repr__(self):
        return self.name


class StandardScalerWrapper:
    
    def __init__(self):
        """Initialize scaler wrapper."""
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.feature_names = X.columns.tolist()
        return self.scaler.fit_transform(X)
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(X)
    
    def get_scaler(self):
        """Get underlying StandardScaler object."""
        return self.scaler


class PPSO:
    
    def __init__(self, obj_func: Callable, bounds: List[Tuple], 
                 num_particles: int = 15, max_iter: int = 20,
                 early_stopping_rounds: int = 10, random_state: int = 42):
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
        
        iterations_without_improvement = 0
        prev_gbest_score = gbest_score
        best_score_history = []
        
        # Main optimization loop
        for t in range(self.max_iter):
            for i in range(self.num_particles):
                # ===== PPSO PHASOR UPDATE (Key difference from standard PSO) =====
                
                cos_t = np.cos(Theta[i])  # Cosine component
                sin_t = np.sin(Theta[i])  # Sine component

                # ===== VELOCITY UPDATE (Periodic Particle Swarm) =====
                
                term1 = (np.abs(cos_t)**2) * sin_t * (pbest_X[i] - X[i])  # Cognitive component
                term2 = (np.abs(sin_t)**2) * cos_t * (gbest_X - X[i])     # Social component
                V[i] = term1 + term2

                # ===== VELOCITY CLAMPING (Prevent velocity explosion) =====
                v_max = (np.abs(cos_t)**2) * (self.bounds[:, 1] - self.bounds[:, 0])
                # v_max must be > 0; fallback to 10% of range if cosine is near 0
                v_max = np.where(v_max < 1e-6, 0.1 * (self.bounds[:, 1] - self.bounds[:, 0]), v_max)
                V[i] = np.clip(V[i], -v_max, v_max)

                # ===== POSITION UPDATE (Chronological) =====
                X[i] = np.clip(X[i] + V[i], self.bounds[:, 0], self.bounds[:, 1])
                
                # ===== PHASE ANGLE UPDATE (Periodic rotation) =====
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

            best_score_history.append(gbest_score)
            
            # ===== EARLY STOPPING: đếm theo iteration (sau khi tất cả particle đã cập nhật) =====
            if gbest_score < prev_gbest_score - 1e-6:
                iterations_without_improvement = 0
                prev_gbest_score = gbest_score
            else:
                iterations_without_improvement += 1
            
            logger.info(f"  Iteration {t+1}/{self.max_iter} | Best Score: {gbest_score:.4f} | No improve: {iterations_without_improvement}")
            
            if iterations_without_improvement >= self.early_stopping_rounds:
                logger.info(f"  ⚠ Early stopping at iteration {t+1}: No improvement for {self.early_stopping_rounds} iterations")
                break

        return gbest_X


class CatBoostPPSOModel:
    
    def __init__(self, n_particles=15, n_iterations=10, cv_splits=3, **kwargs):
        self.num_particles = n_particles
        self.max_iter = n_iterations
        self.cv_splits = cv_splits
        self.model = None
        self.name = "catboost_ppso"
        
        self.cat_features = kwargs.get('cat_features', ['season', 'day_of_week', 'month', 'is_weekend', 'is_holiday'])
        logger.info(f"CatBoostPPSOModel initialized with TimeSeriesSplit({self.cv_splits}) and cat_features")

    def _objective_function(self, params: np.ndarray) -> float:
        n_estimators = max(50, min(1000, int(np.round(params[0]))))
        learning_rate = max(0.001, min(0.5, float(params[1])))    
        depth = max(2, min(12, int(np.round(params[2]))))           
        l2_leaf_reg = max(2.0, min(50.0, float(params[3])))

        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        cv_rmse_scores = []
        
        valid_cats = [c for c in self.cat_features if c in self.X_train.columns]

        for train_idx, val_idx in tscv.split(self.X_train):
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            model = cb.CatBoostRegressor(
                iterations=n_estimators,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                cat_features=valid_cats if valid_cats else None,
                loss_function='RMSE',
                bootstrap_type='No',
                allow_writing_files=False,
                verbose=0,
                random_state=42
            )
            
            try:
                model.fit(X_fold_train, y_fold_train, 
                          eval_set=(X_fold_val, y_fold_val),
                          early_stopping_rounds=20, 
                          verbose=0)
                
                y_fold_pred = model.predict(X_fold_val)
                fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_fold_pred))
                
                if np.isnan(fold_rmse) or np.isinf(fold_rmse):
                    cv_rmse_scores.append(999999.0)
                else:
                    cv_rmse_scores.append(fold_rmse)
                    
            except Exception as e:
                cv_rmse_scores.append(999999.0)

        return np.mean(cv_rmse_scores)

    def fit(self, X: pd.DataFrame, y: pd.Series, param_bounds: dict = None, **kwargs):
        self.X_train = X.reset_index(drop=True)
        self.y_train = y.reset_index(drop=True)
        
        if param_bounds is None:
            param_bounds = {
                'iterations': (100, 800),
                'learning_rate': (0.01, 0.2),
                'depth': (4, 10),
                'l2_leaf_reg': (3, 20)
            }

        self.param_keys = list(param_bounds.keys())
        bounds_list = [param_bounds[k] for k in self.param_keys]
        
        print(f"\n[PPSO] Optimization Phasor PSO for CatBoost...")
        
        ppso_optimizer = PPSO(
            obj_func=self._objective_function,
            bounds=bounds_list,
            num_particles=self.num_particles,
            max_iter=self.max_iter,
            early_stopping_rounds=5,
            random_state=42
        )
        
        best_params_array = ppso_optimizer.optimize()
        
        raw = {self.param_keys[i]: best_params_array[i] for i in range(len(self.param_keys))}
        
        best_iterations    = max(50,  min(2000, int(np.round(raw['iterations']))))
        best_learning_rate = max(0.001, min(0.3, float(raw['learning_rate'])))
        best_depth         = max(3,   min(10,  int(np.round(raw['depth']))))
        best_l2_leaf_reg   = max(1.0, min(50.0, float(raw['l2_leaf_reg'])))
        
        print(f"[PPSO] Best params -> iterations: {best_iterations}, depth: {best_depth}, lr: {best_learning_rate:.4f}, l2: {best_l2_leaf_reg:.4f}")
        
        valid_cats = [c for c in self.cat_features if c in self.X_train.columns]
        
        self.final_model = cb.CatBoostRegressor(
            iterations=best_iterations,
            depth=best_depth,
            learning_rate=best_learning_rate,
            l2_leaf_reg=best_l2_leaf_reg,
            cat_features=valid_cats if valid_cats else None,
            loss_function='RMSE',
            bootstrap_type='No',
            allow_writing_files=False,
            verbose=0,
            random_state=42
        )
        
        self.final_model.fit(self.X_train, self.y_train, verbose=0)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.final_model is None:
            raise RuntimeError("Call fit() first.")
        return self.final_model.predict(X)

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        if self.final_model is None:
            raise RuntimeError("Call fit() first.")
        return pd.DataFrame({
            'Feature': self.final_model.feature_names_,
            'Importance': self.final_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)


def create_model(model_type: str, **kwargs) -> object:
    
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
    
    if model_type == 'baseline':
        return ModelClass()
    elif model_type == 'linear':
        return ModelClass()
    elif model_type == 'rf':
        return ModelClass(random_state=42, n_jobs=-1, **kwargs)
    elif model_type == 'xgb':
        return ModelClass(random_state=42, n_jobs=-1, **kwargs)
    elif model_type == 'catboost':
        categorical_cols = ['season', 'day_of_week', 'month', 'is_weekend', 'is_holiday']
        return ModelClass(
            random_state=42, 
            cat_features=categorical_cols,
            **kwargs
        )
    elif model_type == 'catboost_ppso':
        return ModelClass(**kwargs)
