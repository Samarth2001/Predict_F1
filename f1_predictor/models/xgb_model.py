"""XGBoost model implementation"""

import numpy as np
import pandas as pd
import xgboost as xgb

from .. import config

def train(X_train, y_train, X_val, y_val, feature_names):
    """Train an XGBoost model with compatibility for different versions"""
    xgb_model = xgb.XGBRegressor(**config.XGB_PARAMS)
    
    # Try different XGBoost APIs for compatibility
    try:
        # For older XGBoost versions
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
            verbose=True
        )
    except TypeError:
        try:
            # For newer XGBoost versions
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='mae',
                verbose=True
            )
        except Exception as e:
            print(f"XGBoost training error: {e}")
            return None, None
    
    # Get feature importance
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Evaluate model
    val_pred = xgb_model.predict(X_val)
    mae = np.mean(np.abs(val_pred - y_val))
    print(f"XGBoost Validation MAE: {mae:.4f}")
    
    return xgb_model, importance