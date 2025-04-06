# f1_predictor/prediction.py

import pandas as pd
import numpy as np

# Import from config within the package
from . import config

def prepare_prediction_data(upcoming_info, historical_data, features_used, encoders, imputation_values, actual_quali=None):
    """Prepares the upcoming race data with the necessary features for prediction."""
    print("Preparing data for prediction...")
    if upcoming_info.empty:
        print("No upcoming race info provided.")
        return None

    # Start with the basic info for the drivers/teams in the upcoming race
    predict_df = upcoming_info.copy()

    # --- Re-apply Feature Engineering Logic ---

    # 1. Add Actual Quali/Grid Pos (if predicting Race and data is available)
    if actual_quali is not None and not actual_quali.empty:
        print("Incorporating qualifying data into predictions...")
        # Ensure we're merging on the correct key column
        if 'Driver' not in actual_quali.columns:
            print("Error: 'Driver' column missing from actual_quali data")
            return None
            
        # Debug: Print actual_quali data
        print(f"Qualifying data before merge: {actual_quali[['Driver', 'Quali_Pos']].head()}")
        print(f"Drivers in upcoming_info: {predict_df['Driver'].tolist()}")
        
        # Merge qualifying data to get Grid_Pos
        predict_df = pd.merge(predict_df, actual_quali[['Driver', 'Quali_Pos']], on='Driver', how='left')
        predict_df = predict_df.rename(columns={'Quali_Pos': 'Grid_Pos'})
        
        # Debug: Verify the Grid_Pos data was merged correctly
        print(f"Grid positions after merge: {predict_df[['Driver', 'Grid_Pos']].head()}")
        
        # Ensure Grid_Pos is numeric
        predict_df['Grid_Pos'] = pd.to_numeric(predict_df['Grid_Pos'], errors='coerce')
        
        # Grid_Pos might be NaN if a driver didn't set a time - imputation will handle later
    elif 'Grid_Pos' in features_used:
         # If Grid_Pos is a required feature but actual_quali is missing, add NaN column
         print("Warning: Actual quali results missing but Grid_Pos is a feature. Adding NaN column.")
         predict_df['Grid_Pos'] = np.nan


    # 2. Calculate Rolling & Circuit Features based on Historical Data
    # Combine historical data with the upcoming race structure (targets are NaN)
    temp_predict_df = predict_df.copy()
    # Add columns expected by feature calculations, even if NaN
    if 'Finish_Pos_Clean' not in temp_predict_df.columns: temp_predict_df['Finish_Pos_Clean'] = np.nan
    if 'Quali_Pos' not in temp_predict_df.columns: temp_predict_df['Quali_Pos'] = np.nan
    if 'Quali_Time_Seconds' not in temp_predict_df.columns: temp_predict_df['Quali_Time_Seconds'] = np.nan
    # Grid_Pos was potentially added above

    # Ensure columns align before concatenating
    cols_hist = set(historical_data.columns)
    cols_pred = set(temp_predict_df.columns)
    common_cols = list(cols_hist.intersection(cols_pred))
    missing_in_pred = list(cols_hist - cols_pred)
    for col in missing_in_pred: # Add columns present in historical but not upcoming (e.g., Points) as NaN
         if col not in ['Finish_Pos', 'Status']: # Avoid adding back raw target/status if cleaned version exists
            temp_predict_df[col] = np.nan
    # Ensure order might not matter for concat, but helps debugging
    combined_for_calc = pd.concat([historical_data, temp_predict_df], ignore_index=True, sort=False)
    combined_for_calc = combined_for_calc.sort_values(by=['Year', 'Race_Num']) # Critical for rolling calcs

    # --- Recalculate features on the combined dataframe ---
    # Note: This recalculates for *all* rows, then filters. Can be optimized for large datasets.

    # Rolling Driver Performance
    combined_for_calc['Driver_Avg_Finish_Last_N'] = combined_for_calc.groupby('Driver')['Finish_Pos_Clean'].transform(lambda x: x.shift(1).rolling(config.N_ROLLING, min_periods=1).mean())
    combined_for_calc['Driver_Avg_Quali_Last_N'] = combined_for_calc.groupby('Driver')['Quali_Pos'].transform(lambda x: x.shift(1).rolling(config.N_ROLLING, min_periods=1).mean())
    combined_for_calc['Driver_Avg_Grid_Last_N'] = combined_for_calc.groupby('Driver')['Grid_Pos'].transform(lambda x: x.shift(1).rolling(config.N_ROLLING, min_periods=1).mean())

    # Team Performance
    team_race_avg_finish = combined_for_calc.groupby(['Year', 'Race_Num', 'Team'])['Finish_Pos_Clean'].mean().reset_index()
    team_race_avg_finish['Team_Avg_Finish_Last_N'] = team_race_avg_finish.groupby('Team')['Finish_Pos_Clean'].transform(lambda x: x.shift(1).rolling(config.N_ROLLING, min_periods=1).mean())
    # Merge requires unique keys, drop old column if exists before merge
    if 'Team_Avg_Finish_Last_N' in combined_for_calc.columns: combined_for_calc = combined_for_calc.drop(columns=['Team_Avg_Finish_Last_N'])
    combined_for_calc = pd.merge(combined_for_calc, team_race_avg_finish[['Year', 'Race_Num', 'Team', 'Team_Avg_Finish_Last_N']], on=['Year', 'Race_Num', 'Team'], how='left')

    # Circuit Specific Performance
    combined_for_calc['Driver_Avg_Finish_Circuit'] = combined_for_calc.groupby(['Driver', 'Circuit'])['Finish_Pos_Clean'].transform(lambda x: x.shift(1).expanding().mean())
    combined_for_calc['Driver_Avg_Quali_Circuit'] = combined_for_calc.groupby(['Driver', 'Circuit'])['Quali_Pos'].transform(lambda x: x.shift(1).expanding().mean())


    # 3. Filter to get features for the upcoming race only
    upcoming_race_key = (predict_df['Year'].iloc[0], predict_df['Race_Num'].iloc[0])
    calculated_features = combined_for_calc[
        (combined_for_calc['Year'] == upcoming_race_key[0]) &
        (combined_for_calc['Race_Num'] == upcoming_race_key[1])
    ].reset_index(drop=True) # Reset index after filtering


    # 4. Apply Categorical Encoding (using stored encoders)
    for col, encoder in encoders.items():
        encoded_col_name = col + '_Encoded'
        if col in calculated_features.columns:
            current_col_str = calculated_features[col].astype(str)
            # Handle unseen labels encountered during prediction
            seen_labels = encoder.classes_
            # Use numpy's isin for efficiency
            unseen_mask = ~np.isin(current_col_str, seen_labels)

            # Create temporary array for transformation, handling unseen
            transformed_col = np.empty(len(current_col_str), dtype=int) # Use int type from encoder usually

            # Transform seen labels
            if np.any(~unseen_mask):
                 transformed_col[~unseen_mask] = encoder.transform(current_col_str[~unseen_mask])
            # Assign default value to unseen labels
            if np.any(unseen_mask):
                print(f"Warning: Unseen labels found in '{col}': {current_col_str[unseen_mask].unique()}. Assigning default encoded value (-1).")
                transformed_col[unseen_mask] = -1 # Assign default -1 for unseen

            calculated_features[encoded_col_name] = transformed_col

        elif encoded_col_name in features_used:
             print(f"Warning: Column {col} not found in prediction data for encoding, but {encoded_col_name} is used. Assigning -1.")
             calculated_features[encoded_col_name] = -1 # Assign default


    # 5. Select and Align Final Features
    # Ensure all features the model was trained on are present, in the correct order
    missing_cols = [f for f in features_used if f not in calculated_features.columns]
    for f in missing_cols:
        print(f"Warning: Feature '{f}' used in training is missing in prediction data. Adding as NaN.")
        calculated_features[f] = np.nan

    # Debug: Check if Grid_Pos is in features_used
    if 'Grid_Pos' in features_used:
        print(f"Grid_Pos is used in the model. Values: {calculated_features['Grid_Pos'].tolist()}")
    else:
        print("Grid_Pos is NOT in the features used by the model.")

    # Select only the required features in the correct order
    predict_X = calculated_features[features_used].copy()


    # 6. Impute Remaining Missing Values (using stored imputation values from training)
    predict_X = predict_X.fillna(imputation_values)

    # Final check for any NaNs that might have slipped through
    if predict_X.isnull().any().any():
         print("Warning: NaNs still present in prediction features after imputation. Filling with 0.")
         predict_X = predict_X.fillna(0)


    print("Prediction data preparation complete.")
    # Ensure index aligns with upcoming_info if possible (depends on merges)
    # If calculated_features index was reset, it should align by row order
    # If merges changed order, might need to merge back based on Driver/Team key
    predict_X.index = upcoming_info.index # Assume row order is preserved

    return predict_X


def predict_results(model, features_used, upcoming_info, historical_data, encoders, imputation_values, actual_quali=None, model_type='qualifying'):
    """Predicts results for the upcoming race using the prepared data."""
    print(f"\nPredicting {model_type} results for the upcoming race...")

    if model is None:
        print(f"Cannot predict {model_type}, model not trained.")
        return None
    if upcoming_info.empty:
        print("No upcoming race info provided for prediction.")
        return None

    # Prepare the data using the dedicated function
    predict_X = prepare_prediction_data(
        upcoming_info, historical_data, features_used, encoders, imputation_values,
        actual_quali=(actual_quali if model_type == 'race' else None)
    )

    if predict_X is None or predict_X.empty:
        print(f"Failed to prepare data for {model_type} prediction.")
        return None

    # --- Make Predictions ---
    predictions = model.predict(predict_X)

    # Add predictions back to the original upcoming_info dataframe
    results_df = upcoming_info.copy()
    # Use index from predict_X if it matches upcoming_info's structure
    if results_df.index.equals(predict_X.index):
        results_df[f'Predicted_{model_type.capitalize()}_Pos'] = predictions
    else:
         print("Error: Index mismatch between upcoming_info and predict_X after preparation. Cannot reliably assign predictions.")
         # Attempt merge as fallback - requires 'Driver' key in predict_X
         # predict_X_with_key = predict_X.copy()
         # predict_X_with_key['Driver'] = results_df['Driver'] # Assuming index alignment holds for this step
         # results_df = pd.merge(results_df, predict_X_with_key[['Driver', f'Predicted_{model_type.capitalize()}_Pos']], on='Driver', how='left')
         return None # Fail prediction if indices don't match

    # Rank predictions (lower predicted value = better rank)
    results_df[f'Predicted_{model_type.capitalize()}_Rank'] = results_df[f'Predicted_{model_type.capitalize()}_Pos'].rank(method='first').astype(int)

    # Sort by predicted rank
    results_df = results_df.sort_values(f'Predicted_{model_type.capitalize()}_Rank')

    print(f"\nPredicted {model_type} Ranks:")
    # Select columns to display, handle potential missing Predicted_Pos if merge failed
    display_cols = ['Driver', 'Team', f'Predicted_{model_type.capitalize()}_Rank']
    if f'Predicted_{model_type.capitalize()}_Pos' in results_df.columns:
        display_cols.append(f'Predicted_{model_type.capitalize()}_Pos')
    print(results_df[display_cols])

    return results_df

def predict_with_model(X, models, model_type=config.MODEL_TYPE):
    """
    Make predictions using the trained model(s)
    
    Args:
        X: Features DataFrame
        models: Trained model or dict of models
        model_type: 'lightgbm', 'xgboost', or 'ensemble'
        
    Returns:
        Array of predicted positions
    """
    if model_type == 'ensemble':
        # Combine predictions from multiple models
        predictions = {}
        for model_name, model in models.items():
            predictions[model_name] = model.predict(X)
        
        # Weighted average of predictions
        ensemble_pred = np.zeros_like(predictions[list(predictions.keys())[0]])
        for model_name, pred in predictions.items():
            weight = config.ENSEMBLE_WEIGHTS.get(model_name, 1.0 / len(predictions))
            ensemble_pred += pred * weight
            
        return ensemble_pred
    else:
        # Single model prediction
        return models.predict(X)
