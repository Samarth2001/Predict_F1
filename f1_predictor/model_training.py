# f1_predictor/model_training.py

import pandas as pd
import numpy as np
import logging
import joblib
import json
from typing import Tuple, Dict, Optional, List, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

# Hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available. Hyperparameter optimization disabled.")

from . import config
from .config_manager import get_config
from .mlflow_integration import start_mlflow_run, log_model_training, end_mlflow_run

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class F1ModelTrainer:
    """Comprehensive model training for F1 predictions with ensemble capabilities."""
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.training_history = []
        self.scalers = {}
        
    def train_model(self, 
                   features: pd.DataFrame,
                   target_column_name: str,
                   model_type: str = "ensemble") -> Tuple[Any, List[str], Dict[str, float]]:
        """
        Train F1 prediction model with advanced ensemble capabilities.
        
        Args:
            features: Feature matrix with target column
            target_column_name: Name of target column
            model_type: Type of model ('qualifying', 'race', 'ensemble')
            
        Returns:
            Tuple of (trained_model, feature_names, imputation_values)
        """
        logger.info(f"Starting {model_type} model training...")
        
        # Start MLflow run
        run_id = start_mlflow_run(f"{model_type}_training")
        
        try:
            # Prepare data
            X, y, feature_names, imputation_values = self._prepare_training_data(features, target_column_name)
            
            if len(X) == 0:
                logger.error("No training data available.")
                return None, [], {}
            
            # Create feature hash for versioning
            import hashlib
            feature_str = '|'.join(sorted(feature_names))
            feature_hash = hashlib.sha256(feature_str.encode()).hexdigest()
            
            # Split data with time-aware splitting
            X_train, X_test, y_train, y_test = self._split_data_time_aware(X, y, features)
            
            # Train model based on type
            default_type = get_config('model.default_type', 'ensemble')
            if default_type == 'ensemble':
                model = self._train_ensemble_model(X_train, y_train, X_test, y_test, model_type)
                algorithm = 'ensemble'
            else:
                model = self._train_single_model(X_train, y_train, X_test, y_test, default_type, model_type)
                algorithm = default_type
            
            if model is None:
                logger.error("Model training failed.")
                return None, [], {}
            
            # Evaluate model
            performance_metrics = self._evaluate_model(model, X_test, y_test, model_type)
            
            # Feature importance analysis
            self._analyze_feature_importance(model, feature_names, model_type)
            
            # Create model metadata
            class SimpleMetadata:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
                        
            model_metadata = SimpleMetadata(
                model_type=model_type,
                algorithm=algorithm,
                feature_hash=feature_hash,
                feature_names=feature_names,
                hyperparameters=get_config(f'model.{algorithm}', {}),
                performance_metrics=performance_metrics or {},
                imputation_values=imputation_values,
                version="3.0.0"
            )
            
            # Log to MLflow
            log_model_training(
                model=model,
                model_metadata=model_metadata,
                feature_names=feature_names,
                hyperparameters=get_config(f'model.{algorithm}', {}),
                performance_metrics=performance_metrics or {}
            )
            
            # Save model
            self._save_model(model, feature_names, imputation_values, model_type)
            
            logger.info(f"{model_type} model training completed successfully.")
            return model, feature_names, imputation_values
            
        finally:
            # End MLflow run
            end_mlflow_run()
    
    def _prepare_training_data(self, features: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """Prepare data for training with proper preprocessing."""
        logger.info("Preparing training data...")
        
        if target_column not in features.columns:
            logger.error(f"Target column '{target_column}' not found in features.")
            return np.array([]), np.array([]), [], {}
        
        # Remove rows with missing target
        data_clean = features.dropna(subset=[target_column]).copy()
        
        if len(data_clean) == 0:
            logger.error("No valid training samples after removing missing targets.")
            return np.array([]), np.array([]), [], {}
        
        # Separate features and target
        feature_columns = [col for col in data_clean.columns 
                          if col not in [target_column, 'Quali_Pos'] or col == target_column]
        
        # Select only numeric features
        numeric_features = data_clean[feature_columns].select_dtypes(include=[np.number])
        
        # Remove target from features if present
        if target_column in numeric_features.columns:
            X = numeric_features.drop(columns=[target_column])
        else:
            X = numeric_features
        
        y = data_clean[target_column].values
        feature_names = list(X.columns)
        
        # Calculate imputation values
        imputation_values = {}
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                imputation_values[col] = X[col].median()
                X[col] = X[col].fillna(imputation_values[col])
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        return X.values, y, feature_names, imputation_values
    
    def _split_data_time_aware(self, X: np.ndarray, y: np.ndarray, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data with time awareness to prevent data leakage."""
        if 'Year' in features.columns and 'Race_Num' in features.columns:
            # Sort by time
            time_idx = features[['Year', 'Race_Num']].apply(lambda x: x['Year'] * 100 + x['Race_Num'], axis=1)
            sorted_indices = time_idx.argsort()
            
            # Use last 15% of data for testing
            split_idx = int(len(sorted_indices) * (1 - config.TEST_SIZE))
            train_indices = sorted_indices[:split_idx]
            test_indices = sorted_indices[split_idx:]
            
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
        else:
            # Fallback to random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
            )
        
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test
    
    def _train_ensemble_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                            X_test: np.ndarray, y_test: np.ndarray, model_type: str) -> Any:
        """Train ensemble model with multiple algorithms."""
        logger.info("Training ensemble model...")
        
        base_models = {}
        
        # Train LightGBM
        if 'lightgbm' in config.MODEL_TYPES:
            lgbm_model = self._train_lightgbm(X_train, y_train, X_test, y_test)
            if lgbm_model is not None:
                base_models['lightgbm'] = lgbm_model
        
        # Train XGBoost
        if 'xgboost' in config.MODEL_TYPES:
            xgb_model = self._train_xgboost(X_train, y_train, X_test, y_test)
            if xgb_model is not None:
                base_models['xgboost'] = xgb_model
        
        # Train Random Forest
        if 'random_forest' in config.MODEL_TYPES:
            rf_model = self._train_random_forest(X_train, y_train)
            if rf_model is not None:
                base_models['random_forest'] = rf_model
        
        # Train Neural Network
        if 'neural_network' in config.MODEL_TYPES:
            nn_model = self._train_neural_network(X_train, y_train)
            if nn_model is not None:
                base_models['neural_network'] = nn_model
        
        if not base_models:
            logger.error("No base models trained successfully.")
            return None
        
        # Create ensemble
        ensemble_estimators = [(name, model) for name, model in base_models.items()]
        ensemble_model = VotingRegressor(
            estimators=ensemble_estimators,
            weights=[config.ENSEMBLE_WEIGHTS.get(name, 1.0) for name, _ in ensemble_estimators]
        )
        
        try:
            ensemble_model.fit(X_train, y_train)
            self.models[f'{model_type}_ensemble'] = ensemble_model
            
            # Optimize ensemble weights
            optimized_weights = self._optimize_ensemble_weights(base_models, X_test, y_test)
            if optimized_weights:
                ensemble_model.weights = optimized_weights
            
            return ensemble_model
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            # Return best single model as fallback
            return self._select_best_model(base_models, X_test, y_test)
    
    def _train_single_model(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray, 
                          model_name: str, model_type: str) -> Any:
        """Train single model type."""
        logger.info(f"Training {model_name} model...")
        
        if model_name == 'lightgbm':
            return self._train_lightgbm(X_train, y_train, X_test, y_test)
        elif model_name == 'xgboost':
            return self._train_xgboost(X_train, y_train, X_test, y_test)
        elif model_name == 'random_forest':
            return self._train_random_forest(X_train, y_train)
        elif model_name == 'neural_network':
            return self._train_neural_network(X_train, y_train)
        else:
            logger.error(f"Unknown model type: {model_name}")
            return None
    
    def _train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> lgb.LGBMRegressor:
        """Train LightGBM model with hyperparameter optimization."""
        try:
            if config.ENABLE_HYPERPARAMETER_OPTIMIZATION and OPTUNA_AVAILABLE:
                params = self._optimize_lightgbm_params(X_train, y_train)
            else:
                params = config.LGBM_PARAMS.copy()
            
            model = lgb.LGBMRegressor(**params)
            
            # Train with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS), lgb.log_evaluation(0)]
            )
            
            logger.info("LightGBM training completed")
            return model
            
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            return None
    
    def _train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray) -> xgb.XGBRegressor:
        """Train XGBoost model with hyperparameter optimization."""
        try:
            if config.ENABLE_HYPERPARAMETER_OPTIMIZATION and OPTUNA_AVAILABLE:
                params = self._optimize_xgboost_params(X_train, y_train)
            else:
                params = config.XGB_PARAMS.copy()
            
            model = xgb.XGBRegressor(**params)
            
            # Train with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
                verbose=False
            )
            
            logger.info("XGBoost training completed")
            return model
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            return None
    
    def _train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
        """Train Random Forest model."""
        try:
            params = config.RF_PARAMS.copy()
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            
            logger.info("Random Forest training completed")
            return model
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            return None
    
    def _train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray) -> MLPRegressor:
        """Train Neural Network model."""
        try:
            # Scale features for neural network
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers['neural_network'] = scaler
            
            params = config.NN_PARAMS.copy()
            model = MLPRegressor(**params)
            model.fit(X_train_scaled, y_train)
            
            logger.info("Neural Network training completed")
            return model
            
        except Exception as e:
            logger.error(f"Neural Network training failed: {e}")
            return None
    
    def _optimize_lightgbm_params(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Optimize LightGBM hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'random_state': config.RANDOM_STATE,
                'n_jobs': -1,
                'verbose': -1
            }
            
            # Cross-validation
            scores = cross_val_score(
                lgb.LGBMRegressor(**params),
                X_train, y_train,
                cv=config.CV_FOLDS,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            return scores.mean()
        
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS, timeout=config.OPTUNA_TIMEOUT)
            
            logger.info(f"Best LightGBM params: {study.best_params}")
            return study.best_params
            
        except Exception as e:
            logger.warning(f"Hyperparameter optimization failed: {e}. Using default params.")
            return config.LGBM_PARAMS.copy()
    
    def _optimize_xgboost_params(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Optimize XGBoost hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'objective': 'reg:absoluteerror',
                'eval_metric': 'mae',
                'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'alpha': trial.suggest_float('alpha', 0.0, 1.0),
                'lambda': trial.suggest_float('lambda', 0.0, 1.0),
                'random_state': config.RANDOM_STATE,
                'n_jobs': -1,
                'verbosity': 0
            }
            
            # Cross-validation
            scores = cross_val_score(
                xgb.XGBRegressor(**params),
                X_train, y_train,
                cv=config.CV_FOLDS,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            return scores.mean()
        
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS, timeout=config.OPTUNA_TIMEOUT)
            
            logger.info(f"Best XGBoost params: {study.best_params}")
            return study.best_params
            
        except Exception as e:
            logger.warning(f"Hyperparameter optimization failed: {e}. Using default params.")
            return config.XGB_PARAMS.copy()
    
    def _optimize_ensemble_weights(self, base_models: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Optional[List[float]]:
        """Optimize ensemble weights based on validation performance."""
        if len(base_models) < 2:
            return None
        
        try:
            from scipy.optimize import minimize
            
            def objective(weights):
                weights = np.array(weights) / np.sum(weights)  # Normalize
                predictions = np.zeros(len(y_test))
                
                for i, (name, model) in enumerate(base_models.items()):
                    pred = model.predict(X_test)
                    predictions += weights[i] * pred
                
                return mean_absolute_error(y_test, predictions)
            
            # Initial equal weights
            initial_weights = [1.0] * len(base_models)
            bounds = [(0.1, 2.0)] * len(base_models)
            
            result = minimize(objective, initial_weights, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                optimized_weights = result.x / np.sum(result.x)
                logger.info(f"Optimized ensemble weights: {dict(zip(base_models.keys(), optimized_weights))}")
                return optimized_weights.tolist()
            
        except Exception as e:
            logger.warning(f"Ensemble weight optimization failed: {e}")
        
        return None
    
    def _select_best_model(self, models: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Any:
        """Select best performing model from multiple trained models."""
        best_model = None
        best_score = float('inf')
        
        for name, model in models.items():
            try:
                if name == 'neural_network' and 'neural_network' in self.scalers:
                    X_test_scaled = self.scalers['neural_network'].transform(X_test)
                    pred = model.predict(X_test_scaled)
                else:
                    pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, pred)
                
                if mae < best_score:
                    best_score = mae
                    best_model = model
                    
                logger.info(f"{name} MAE: {mae:.4f}")
                
            except Exception as e:
                logger.warning(f"Error evaluating {name}: {e}")
        
        return best_model
    
    def _evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, model_type: str) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        try:
            # Handle neural network scaling
            if hasattr(model, 'predict'):
                if isinstance(model, MLPRegressor) and 'neural_network' in self.scalers:
                    X_test_scaled = self.scalers['neural_network'].transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                else:
                    y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # F1-specific metrics
            top3_accuracy = np.mean(np.abs(y_pred - y_test) <= 3)
            top5_accuracy = np.mean(np.abs(y_pred - y_test) <= 5)
            
            metrics = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'top3_accuracy': top3_accuracy,
                'top5_accuracy': top5_accuracy
            }
            
            # Log results
            logger.info(f"{model_type} Model Evaluation:")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  Top-3 Accuracy: {top3_accuracy:.4f}")
            logger.info(f"  Top-5 Accuracy: {top5_accuracy:.4f}")
            
            # Save metrics
            self._save_evaluation_metrics(metrics, model_type)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {}
    
    def _analyze_feature_importance(self, model: Any, feature_names: List[str], model_type: str):
        """Analyze and save feature importance."""
        try:
            importance_scores = None
            
            # Extract feature importance based on model type
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
            elif hasattr(model, 'estimators_'):  # Ensemble model
                # Average importance across estimators
                importances = []
                for estimator in model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances.append(estimator.feature_importances_)
                if importances:
                    importance_scores = np.mean(importances, axis=0)
            
            if importance_scores is not None:
                # Create feature importance dataframe
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance_scores
                }).sort_values('importance', ascending=False)
                
                self.feature_importance[model_type] = importance_df
                
                # Log top features
                logger.info(f"Top 10 features for {model_type}:")
                for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                    logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
                
                # Save feature importance
                importance_path = f"{config.EVALUATION_DIR}/feature_importance_{model_type}_{datetime.now().strftime('%Y%m%d')}.csv"
                importance_df.to_csv(importance_path, index=False)
                
        except Exception as e:
            logger.warning(f"Feature importance analysis failed: {e}")
    
    def _save_model(self, model: Any, feature_names: List[str], imputation_values: Dict, model_type: str):
        """Save trained model and metadata."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = f"{config.MODELS_DIR}/{model_type}_model_{timestamp}.joblib"
            metadata_path = f"{config.MODELS_DIR}/{model_type}_metadata_{timestamp}.json"
            
            # Save model
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata = {
                'model_type': model_type,
                'feature_names': feature_names,
                'imputation_values': imputation_values,
                'training_timestamp': timestamp,
                'config_version': config.MODEL_VERSION,
                'scalers': {}
            }
            
            # Include scaler info if available
            if 'neural_network' in self.scalers:
                scaler_path = f"{config.MODELS_DIR}/{model_type}_scaler_{timestamp}.joblib"
                joblib.dump(self.scalers['neural_network'], scaler_path)
                metadata['scalers']['neural_network'] = scaler_path
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def _save_evaluation_metrics(self, metrics: Dict, model_type: str):
        """Save evaluation metrics to file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metrics_path = f"{config.EVALUATION_DIR}/metrics_{model_type}_{timestamp}.json"
            
            metrics_data = {
                'model_type': model_type,
                'timestamp': timestamp,
                'metrics': metrics
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")

# Backward compatibility function
def train_model(features: pd.DataFrame, target_column_name: str, model_type: str) -> Tuple[Any, List[str], Dict[str, float]]:
    """Train model function for backward compatibility."""
    trainer = F1ModelTrainer()
    return trainer.train_model(features, target_column_name, model_type)
