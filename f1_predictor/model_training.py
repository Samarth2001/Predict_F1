# f1_predictor/model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# Import from config within the package
from . import config

def train_model(features, target_column_name, model_type='qualifying'):
    """Trains a LightGBM model for Qualifying or Race prediction."""
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
        pass # Base feature_cols are likely okay, maybe add Quali_Time_Seconds if available and relevant
        if 'Quali_Time_Seconds' in features.columns:
             feature_cols.append('Quali_Time_Seconds') # Example: Use raw time as a feature

    elif model_type == 'race':
        # Race model SHOULD use Grid_Pos (actual quali result for that race)
        # It should NOT use Finish_Pos_Clean (target) or Quali_Pos (often redundant with Grid_Pos)
        if 'Grid_Pos' in features.columns:
            feature_cols.append('Grid_Pos')
        else:
             print("Warning: Grid_Pos column not found for Race model training. It's a crucial feature.")
        # Remove Quali_Pos if Grid_Pos is used, unless specific reason to keep
        feature_cols = [f for f in feature_cols if f != 'Quali_Pos']

    # Ensure all selected feature columns actually exist in the dataframe
    feature_cols = [f for f in feature_cols if f in features.columns]
    # Remove the target column itself from the features
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

    # Split data (consider time-based split for F1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, shuffle=True, random_state=config.RANDOM_STATE
    )

    # --- Train LightGBM Model ---
    print("Training LightGBM Regressor...")
    model = lgb.LGBMRegressor(**config.LGBM_PARAMS)

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              eval_metric='mae',
              callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)])

    # Evaluate on test set
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"{model_type.capitalize()} Model MAE on Test Set: {mae:.4f}")

    # --- Optional: Train final model on all available data ---
    # print("Training final model on all data...")
    # final_model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
    # final_model.fit(X, y) # No early stopping needed here usually
    # print(f"{model_type.capitalize()} final model training complete.")
    # return final_model, feature_cols, imputation_values

    print(f"{model_type.capitalize()} model training complete (trained on split).")
    # Return model trained on split, features used, and imputation values
    return model, feature_cols, imputation_values
