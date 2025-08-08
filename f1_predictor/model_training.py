# f1_predictor/model_training.py

import pandas as pd
import numpy as np
import logging
import joblib
import os
from typing import Tuple, Dict, Optional, List, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin

from .config import config

# Set up logging
logger = logging.getLogger(__name__)

class EnsembleModel(BaseEstimator, RegressorMixin):
    """Simple ensemble that averages predictions from multiple models."""
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=1)

class F1ModelTrainer:
    """Trains and saves F1 prediction models."""
    
    def train_model(self, 
                   features: pd.DataFrame,
                   target_column_name: str,
                   model_type: str) -> Tuple[Any, List[str], Dict[str, float]]:
        """
        Train an F1 prediction model.
        """
        logger.info(f"Starting {model_type} model training...")
        
        try:
            # Prepare data
            X, y, feature_names, _ = self._prepare_training_data(features, target_column_name)
            
            if len(X) == 0:
                logger.error("No training data available.")
                return None, [], {}
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model_params = config.get(f'models.{model_type}_params', {})
            
            # Train base models
            lgb_model = lgb.LGBMRegressor(**model_params)
            xgb_model = xgb.XGBRegressor(**model_params)
            
            # Create ensemble
            model = EnsembleModel([lgb_model, xgb_model])
            model.fit(X_train, y_train)
            
            self._save_model(model, model_type)
            
            logger.info(f"{model_type} model training completed successfully.")
            return model, feature_names, {} # Returning empty dict for imputation values for now
            
        except Exception as e:
            logger.error(f"Model training failed: {e}", exc_info=True)
            return None, [], {}

    def _prepare_training_data(self, features: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict]:
        """Prepare data for training."""
        logger.info("Preparing training data...")
        
        data_clean = features.dropna(subset=[target_column]).copy()
        
        y = data_clean[target_column]
        X = data_clean.drop(columns=[target_column, 'Quali_Pos', 'Position'], errors='ignore')
        
        # Select only numeric features
        X = X.select_dtypes(include=np.number)
        feature_names = list(X.columns)
        
        # Impute any remaining NaNs
        imputation_values = {col: X[col].median() for col in X.columns if X[col].isnull().any()}
        X = X.fillna(imputation_values)
        X = X.fillna(0) # Fallback
        
        logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_names, imputation_values
    
    def _save_model(self, model: Any, model_type: str):
        """Save trained model."""
        try:
            models_dir = config.get('paths.models_dir')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f"{model_type}_model.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Model saved to: {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
