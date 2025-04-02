"""LightGBM model implementation"""

import numpy as np
import pandas as pd
import lightgbm as lgb

from .. import config

def train(X_train, y_train, X_val, y_val, feature_names):
    """Train a LightGBM model"""
    lgb_model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
    
    try:
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
            verbose=100
        )
    except Exception as e:
        print(f"LightGBM training error: {e}")
        return None, None
    
    # Get feature importance
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': lgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Evaluate model
    val_pred = lgb_model.predict(X_val)
    mae = np.mean(np.abs(val_pred - y_val))
    print(f"LightGBM Validation MAE: {mae:.4f}")
    
    return lgb_model, importance