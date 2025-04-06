# f1_predictor/model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import pickle
import os
import matplotlib.pyplot as plt
import time

# Import from config within the package
from . import config

def train_model(features, target_column_name=None, model_type='qualifying', algorithm=None):
    """
    Unified model training function that handles both traditional and advanced models.
    
    Args:
        features: DataFrame containing features or X matrix directly
        target_column_name: Name of target column (used in legacy mode)
        model_type: 'qualifying' or 'race' (legacy) OR 'lightgbm', 'xgboost', 'ensemble' (advanced)
        algorithm: Override config.MODEL_TYPE if specified
    
    Returns:
        For legacy mode: (model, feature_cols, imputation_values)
        For advanced mode: (models_dict, feature_importances)
    """
    # Determine if we're in legacy mode (qualifying/race) or advanced mode (specific algorithm)
    legacy_mode = model_type in ['qualifying', 'race']
    
    # For legacy mode, use the original function flow
    if legacy_mode:
        return _train_legacy_model(features, target_column_name, model_type)
    
    # For advanced mode, use the new approach with algorithm selection
    algorithm = algorithm or config.MODEL_TYPE
    if algorithm not in ['lightgbm', 'xgboost', 'ensemble']:
        print(f"Warning: Unknown algorithm '{algorithm}'. Using 'lightgbm' instead.")
        algorithm = 'lightgbm'
    
    # Expect X, y format for advanced mode
    if isinstance(features, tuple) and len(features) == 2:
        X, y = features
        feature_names = X.columns.tolist()
    else:
        print("Error: Advanced mode requires (X, y) tuple input")
        return None, None
    
    return _train_advanced_model(X, y, feature_names, algorithm)

def _train_legacy_model(features, target_column_name, model_type='qualifying'):
    """Legacy model training function - kept for backward compatibility"""
    print(f"\nTraining {model_type} prediction model...")

    # Determine target based on model type
    target = target_column_name # e.g., 'Quali_Pos' or 'Finish_Pos_Clean'

    # Drop rows with missing target or essential encoded features
    essential_features = [col for col in features.columns if '_Encoded' in col]
    features = features.dropna(subset=[target] + essential_features)

    if features.empty:
        print(f"No data available to train {model_type} model after dropping essential NaNs.")
        return None, [], None # Model, features, imputation values

    # --- Define Features (X) ---
    # Start with encoded categoricals and average performance features
    feature_cols = [col for col in features.columns if '_Encoded' in col or '_Avg_' in col]

    # Add position-based features depending on model type
    if model_type == 'qualifying':
        # Quali model should NOT use Grid_Pos, Finish_Pos, Quali_Pos (target)
        # Keep features derived from past performance
        if 'Quali_Time_Seconds' in features.columns:
             feature_cols.append('Quali_Time_Seconds')

    elif model_type == 'race':
        # Race model SHOULD use Grid_Pos (actual quali result for that race)
        if 'Grid_Pos' in features.columns:
            feature_cols.append('Grid_Pos')
            print("Grid_Pos added as feature for race prediction")
        else:
             print("Warning: Grid_Pos column not found for Race model training. It's a crucial feature.")
        # Remove Quali_Pos if Grid_Pos is used
        feature_cols = [f for f in feature_cols if f != 'Quali_Pos']

    # Ensure all selected feature columns actually exist
    feature_cols = [f for f in feature_cols if f in features.columns]
    # Remove the target column from the features
    feature_cols = [f for f in feature_cols if f != target]

    if not feature_cols:
        print(f"Error: No valid feature columns identified for {model_type} model.")
        return None, [], None

    print(f"Using features for {model_type} model: {feature_cols}")
    X = features[feature_cols]
    y = features[target]

    # Handle potential NaN values in features (impute with median)
    imputation_values = X.median()
    X = X.fillna(imputation_values)

    if X.empty or len(X) != len(y):
        print(f"Feature matrix X or target vector y is invalid for {model_type} model.")
        return None, [], None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, shuffle=True, random_state=config.RANDOM_STATE
    )

    # --- Train LightGBM Model ---
    print("Training LightGBM Regressor...")
    
    # Set up model parameters - add special weight to Grid_Pos for race model
    lgbm_params = config.LGBM_PARAMS.copy()
    
    # Add feature importance weight for race model to prioritize Grid_Pos
    if model_type == 'race' and 'Grid_Pos' in X.columns:
        print("Enhancing Grid_Pos importance for race predictions")
        feature_weights = None
        # Create feature importance weights - Grid_Pos gets 3x default weight
        if hasattr(lgb, 'create_feature_weights'):
            feature_weights = lgb.create_feature_weights()
            for i, col in enumerate(X.columns):
                if col == 'Grid_Pos':
                    feature_weights[i] = 3.0  # 3x higher importance
        
        if feature_weights is not None:
            lgbm_params['feature_weights'] = feature_weights

    model = lgb.LGBMRegressor(**lgbm_params)
    
    # Extremely robust LightGBM training with multiple fallback options
    try_methods = [
        # Method 1: Modern API with callbacks
        lambda: model.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
                          callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS)] 
                          if hasattr(lgb, 'early_stopping') else None),
                          
        # Method 2: Legacy API with early_stopping_rounds
        lambda: model.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
                          early_stopping_rounds=config.EARLY_STOPPING_ROUNDS),
                          
        # Method 3: Just eval_set without early stopping or verbose
        lambda: model.fit(X_train, y_train, eval_set=[(X_test, y_test)]),
        
        # Method 4: Absolute minimum - no optional parameters
        lambda: model.fit(X_train, y_train)
    ]
    
    # Try each method until one works
    for i, method in enumerate(try_methods):
        try:
            method()
            print(f"Successfully trained model using method {i+1}")
            
            # For race model, check if Grid_Pos is among top features
            if model_type == 'race':
                importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                print("\nFeature Importances:")
                print(importance.head(10))
                
                # Check if Grid_Pos is important enough
                if 'Grid_Pos' in importance['Feature'].values:
                    grid_pos_rank = importance[importance['Feature'] == 'Grid_Pos'].index[0] + 1
                    grid_pos_importance = importance[importance['Feature'] == 'Grid_Pos']['Importance'].values[0]
                    print(f"Grid_Pos importance rank: {grid_pos_rank}, value: {grid_pos_importance:.4f}")
            
            break
        except Exception as e:
            if i == len(try_methods) - 1:
                print(f"All training methods failed. Error: {e}")
                return None, [], None  # Return failure
            # Otherwise continue to next method

    # Evaluate on test set
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"{model_type.capitalize()} Model MAE on Test Set: {mae:.4f}")
    
    return model, feature_cols, imputation_values

def _train_advanced_model(X, y, feature_names, algorithm='lightgbm'):
    """Internal function for training advanced models with multiple algorithm options"""
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    print(f"Training set: {X_train.shape[0]} examples")
    print(f"Validation set: {X_val.shape[0]} examples")
    
    models = {}
    feature_importances = {}
    
    start_time = time.time()
    
    # Train LightGBM if requested
    if algorithm in ['lightgbm', 'ensemble']:
        print("\nTraining LightGBM model...")
        models['lightgbm'], feature_importances['lightgbm'] = _train_lgbm(
            X_train, X_val, y_train, y_val, feature_names
        )
    
    # Train XGBoost if requested
    if algorithm in ['xgboost', 'ensemble']:
        print("\nTraining XGBoost model...")
        models['xgboost'], feature_importances['xgboost'] = _train_xgboost(
            X_train, X_val, y_train, y_val, feature_names
        )
    
    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    
    # Save model(s)
    os.makedirs(os.path.join(config.BASE_DIR, 'models'), exist_ok=True)
    
    if algorithm == 'lightgbm':
        _save_model(models['lightgbm'], 'lightgbm_model.pkl')
    elif algorithm == 'xgboost':
        _save_model(models['xgboost'], 'xgboost_model.pkl')
    elif algorithm == 'ensemble':
        _save_model(models, 'ensemble_models.pkl')
    
    # Plot feature importances
    _plot_feature_importances(feature_importances)
    
    # Return single model if not ensemble, otherwise return dict
    if algorithm != 'ensemble':
        return models[algorithm], feature_importances[algorithm]
    return models, feature_importances

def _train_lgbm(X_train, X_val, y_train, y_val, feature_names):
    """Train a LightGBM model with robust error handling"""
    lgb_model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
    
    # Similar robust approach
    try_methods = [
        # Method 1: Standard with early_stopping_rounds
        lambda: lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                             early_stopping_rounds=config.EARLY_STOPPING_ROUNDS),
                             
        # Method 2: Without verbose
        lambda: lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                             early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
                             verbose=False),
                             
        # Method 3: Just eval_set
        lambda: lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)]),
        
        # Method 4: No optional parameters  
        lambda: lgb_model.fit(X_train, y_train)
    ]
    
    # Try each method until one works
    for i, method in enumerate(try_methods):
        try:
            method()
            print(f"Successfully trained LightGBM model using method {i+1}")
            break
        except Exception as e:
            if i == len(try_methods) - 1:
                print(f"All LightGBM training methods failed. Error: {e}")
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

def _train_xgboost(X_train, X_val, y_train, y_val, feature_names):
    """Train an XGBoost model with compatibility for different versions"""
    xgb_model = xgb.XGBRegressor(**config.XGB_PARAMS)
    
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
            # For newer XGBoost versions with callbacks
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='mae',
                verbose=True,
                callbacks=[xgb.callback.EarlyStopping(rounds=config.EARLY_STOPPING_ROUNDS)]
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

def _save_model(model, filename):
    """Save model to disk"""
    model_path = os.path.join(config.BASE_DIR, 'models', filename)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def _plot_feature_importances(feature_importances):
    """Plot feature importances for all models"""
    for model_name, importance_df in feature_importances.items():
        if importance_df is None:
            continue
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
        plt.title(f'Top 15 Feature Importances ({model_name.capitalize()})')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        plot_path = os.path.join(config.BASE_DIR, 'models', f'{model_name}_feature_importance.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Feature importance plot saved to {plot_path}")

# For backwards compatibility
train_advanced_model = train_model
train_model_adapter = train_model
train_single_model = _train_legacy_model
