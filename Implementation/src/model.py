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


class PPSOTunedModel:
    """Generic PPSO-tuned regressor.

    Applies Phasor Particle Swarm Optimisation (PPSO) hyperparameter search to
    any tree-based model (Random Forest, XGBoost, CatBoost) so that all
    tunable models are compared under the same optimisation budget (fair
    comparison). Fitness = mean RMSE over TimeSeriesSplit cross-validation.
    """

    # Per-model search space: name -> (bounds, is_integer)
    DEFAULT_SPACES = {
        'rf': {
            'n_estimators':      ((100, 800), True),
            'max_depth':         ((4, 25), True),
            'min_samples_split': ((2, 20), True),
            'min_samples_leaf':  ((1, 10), True),
        },
        'xgb': {
            'n_estimators':  ((100, 1000), True),
            'max_depth':     ((3, 10), True),
            'learning_rate': ((0.01, 0.2), False),
            'reg_lambda':    ((1.0, 20.0), False),
        },
        'catboost': {
            'iterations':    ((100, 800), True),
            'learning_rate': ((0.01, 0.2), False),
            'depth':         ((4, 10), True),
            'l2_leaf_reg':   ((1.0, 20.0), False),
        },
    }

    def __init__(self, base_model_type: str = 'catboost',
                 n_particles=15, n_iterations=10, cv_splits=3, **kwargs):
        if base_model_type not in self.DEFAULT_SPACES:
            raise ValueError(f"PPSO tuning not supported for '{base_model_type}'")
        self.base_model_type = base_model_type
        self.num_particles = n_particles
        self.max_iter = n_iterations
        self.cv_splits = cv_splits
        self.final_model = None
        self.name = f"{base_model_type}_ppso"

        self.cat_features = kwargs.get(
            'cat_features',
            ['season', 'day_of_week', 'month', 'is_weekend', 'is_holiday']
        )
        logger.info(f"PPSOTunedModel({base_model_type}) initialized with TimeSeriesSplit({self.cv_splits})")

    def _decode_params(self, params: np.ndarray) -> dict:
        space = self.DEFAULT_SPACES[self.base_model_type]
        decoded = {}
        for i, key in enumerate(self.param_keys):
            (lo, hi), is_int = space[key]
            val = float(np.clip(params[i], lo, hi))
            decoded[key] = int(np.round(val)) if is_int else val
        return decoded

    def _build_model(self, hp: dict):
        if self.base_model_type == 'rf':
            return RandomForestRegressor(random_state=42, n_jobs=-1, **hp)
        elif self.base_model_type == 'xgb':
            return xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0, **hp)
        else:  # catboost
            valid_cats = [c for c in self.cat_features if c in self.X_train.columns]
            return cb.CatBoostRegressor(
                cat_features=valid_cats if valid_cats else None,
                loss_function='RMSE',
                bootstrap_type='No',
                allow_writing_files=False,
                verbose=0,
                random_state=42,
                **hp
            )

    def _objective_function(self, params: np.ndarray) -> float:
        hp = self._decode_params(params)
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        cv_rmse_scores = []

        for train_idx, val_idx in tscv.split(self.X_train):
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            try:
                model = self._build_model(hp)
                if self.base_model_type == 'catboost':
                    model.fit(X_fold_train, y_fold_train,
                              eval_set=(X_fold_val, y_fold_val),
                              early_stopping_rounds=20,
                              verbose=0)
                else:
                    model.fit(X_fold_train, y_fold_train)

                y_fold_pred = model.predict(X_fold_val)
                fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_fold_pred))
                cv_rmse_scores.append(999999.0 if (np.isnan(fold_rmse) or np.isinf(fold_rmse)) else fold_rmse)
            except Exception:
                cv_rmse_scores.append(999999.0)

        return np.mean(cv_rmse_scores)

    def fit(self, X: pd.DataFrame, y: pd.Series, param_bounds: dict = None, **kwargs):
        self.X_train = X.reset_index(drop=True)
        self.y_train = y.reset_index(drop=True)

        space = self.DEFAULT_SPACES[self.base_model_type]
        if param_bounds is None:
            param_bounds = {k: v[0] for k, v in space.items()}
        # Only tune keys known to the search space
        param_bounds = {k: v for k, v in param_bounds.items() if k in space}

        self.param_keys = list(param_bounds.keys())
        bounds_list = [param_bounds[k] for k in self.param_keys]

        print(f"\n[PPSO] Phasor PSO optimisation for {self.base_model_type.upper()}...")

        ppso_optimizer = PPSO(
            obj_func=self._objective_function,
            bounds=bounds_list,
            num_particles=self.num_particles,
            max_iter=self.max_iter,
            early_stopping_rounds=5,
            random_state=42
        )

        best_params_array = ppso_optimizer.optimize()
        self.best_params = self._decode_params(best_params_array)

        print(f"[PPSO] Best params for {self.base_model_type}: {self.best_params}")

        self.final_model = self._build_model(self.best_params)
        if self.base_model_type == 'catboost':
            self.final_model.fit(self.X_train, self.y_train, verbose=0)
        else:
            self.final_model.fit(self.X_train, self.y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.final_model is None:
            raise RuntimeError("Call fit() first.")
        return self.final_model.predict(X)

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        if self.final_model is None:
            raise RuntimeError("Call fit() first.")
        if self.base_model_type == 'catboost':
            feature_names = self.final_model.feature_names_
        else:
            feature_names = list(self.X_train.columns)
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.final_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)


class CatBoostPPSOModel(PPSOTunedModel):
    """Backward-compatible alias: CatBoost tuned by PPSO (kept so existing
    pickled models and the dashboard backend keep working)."""

    def __init__(self, n_particles=15, n_iterations=10, cv_splits=3, **kwargs):
        super().__init__(base_model_type='catboost',
                         n_particles=n_particles,
                         n_iterations=n_iterations,
                         cv_splits=cv_splits, **kwargs)
        self.name = "catboost_ppso"


def create_model(model_type: str, **kwargs) -> object:
    
    models = {
        'baseline': BaselineModel,
        'linear': LinearRegression,
        'rf': RandomForestRegressor,
        'xgb': xgb.XGBRegressor,
        'catboost': cb.CatBoostRegressor,
        'rf_ppso': PPSOTunedModel,
        'xgb_ppso': PPSOTunedModel,
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
    elif model_type in ('rf_ppso', 'xgb_ppso'):
        return ModelClass(base_model_type=model_type.split('_')[0], **kwargs)
    elif model_type == 'catboost_ppso':
        return ModelClass(**kwargs)
