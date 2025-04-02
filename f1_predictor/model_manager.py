"""
Model manager for F1 prediction project.
Provides a unified interface for training different model types.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
import matplotlib.pyplot as plt
import time

from . import config
from .models import lgbm_model, xgb_model

def train_model(features, target_column_name=None, model_type='qualifying', algorithm=None):
    """
    Unified model training function that works with both prediction types and algorithms.
    
    Args:
        features: DataFrame with features or (X, y) tuple for direct training
        target_column_name: Column name for target variable (used in qualifying/race mode)
        model_type: 'qualifying'/'race' OR algorithm specification if tuple is passed
        algorithm: Override for algorithm selection ('lightgbm', 'xgboost', 'ensemble')
        
    Returns:
        For qualifying/race: (model, feature_cols, imputation_values)
        For algorithm mode: (model, feature_importance)
    """
    # Determine if we're using legacy mode or algorithm mode
    if isinstance(features, tuple) and len(features) == 2:
        # Algorithm mode - (X, y) format
        X, y = features
        features_list = X.columns.tolist()
        algorithm = algorithm or model_type or config.MODEL_TYPE
        
        if algorithm not in ['lightgbm', 'xgboost', 'ensemble']:
            print(f"Warning: Unknown algorithm '{algorithm}'. Using 'lightgbm'.")
            algorithm = 'lightgbm'
            
        return _train_algorithm(X, y, features_list, algorithm)
    else:
        # Legacy mode - DataFrame with target column
        return _train_prediction_type(features, target_column_name, model_type)

def _train_algorithm(X, y, feature_names, algorithm='lightgbm'):
    """Train model using specific algorithm"""
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    print(f"Training set: {X_train.shape[0]} examples")
    print(f"Validation set: {X_val.shape[0]} examples")
    
    start_time = time.time()
    
    # Dispatch to appropriate model trainer
    if algorithm == 'lightgbm':
        model, importance = lgbm_model.train(
            X_train, y_train, X_val, y_val, feature_names
        )
        _save_model(model, 'lightgbm_model.pkl')
        
    elif algorithm == 'xgboost':
        model, importance = xgb_model.train(
            X_train, y_train, X_val, y_val, feature_names
        )
        _save_model(model, 'xgboost_model.pkl')
        
    elif algorithm == 'ensemble':
        models = {}
        importances = {}
        
        # Train both models
        models['lightgbm'], importances['lightgbm'] = lgbm_model.train(
            X_train, y_train, X_val, y_val, feature_names
        )
        
        models['xgboost'], importances['xgboost'] = xgb_model.train(
            X_train, y_train, X_val, y_val, feature_names
        )
        
        _save_model(models, 'ensemble_models.pkl')
        return models, importances
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    _plot_feature_importance(importance, algorithm)
    
    return model, importance

def _train_prediction_type(features, target_column_name, model_type='qualifying'):
    """Legacy mode training for qualifying/race prediction"""
    print(f"\nTraining {model_type} prediction model...")

    # Determine target based on model type
    target = target_column_name
    
    # Drop rows with missing target or essential encoded features
    essential_features = [col for col in features.columns if '_Encoded' in col]
    features = features.dropna(subset=[target] + essential_features)

    if features.empty:
        print(f"No data available to train {model_type} model after dropping essential NaNs.")
        return None, [], None  # Model, features, imputation values

    # --- Define Features (X) ---
    feature_cols = [col for col in features.columns if '_Encoded' in col or '_Avg_' in col]
    
    # Add position-based features depending on model type
    if model_type == 'qualifying':
        if 'Quali_Time_Seconds' in features.columns:
            feature_cols.append('Quali_Time_Seconds')
    elif model_type == 'race':
        if 'Grid_Pos' in features.columns:
            feature_cols.append('Grid_Pos')
        feature_cols = [f for f in feature_cols if f != 'Quali_Pos']

    # Ensure all selected feature columns actually exist
    feature_cols = [f for f in feature_cols if f in features.columns]
    # Remove the target column from the features
    feature_cols = [f for f in feature_cols if f != target]

    print(f"Using features for {model_type} model: {feature_cols}")
    X = features[feature_cols]
    y = features[target]

    # Handle potential NaN values in features (impute with median)
    imputation_values = X.median()
    X = X.fillna(imputation_values)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # Train LightGBM model
    print("Training LightGBM model for prediction...")
    model, importance = lgbm_model.train(X_train, y_train, X_test, y_test, feature_cols)
    
    # Evaluate on test set
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"{model_type.capitalize()} Model MAE on Test Set: {mae:.4f}")
    
    return model, feature_cols, imputation_values

def _save_model(model, filename):
    """Save model to disk"""
    model_path = os.path.join(config.BASE_DIR, 'models', filename)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def _plot_feature_importance(importance_df, model_name):
    """Plot feature importance"""
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
    plt.title(f'Top 15 Feature Importances ({model_name.capitalize()})')
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plot_path = os.path.join(config.BASE_DIR, 'models', f'{model_name}_importance.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Feature importance plot saved to {plot_path}")

# For backward compatibility
train_advanced_model = train_model