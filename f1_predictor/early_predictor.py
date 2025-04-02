import os
import sys

# Add the project root to the Python path
# This allows Python to find the f1_predictor package
# Adjust the number of os.path.dirname calls if your script is deeper
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np

# Now use absolute imports from the package perspective
from f1_predictor import model_training
from f1_predictor import prediction
from f1_predictor import config
# Import other necessary modules if needed for standalone execution
from f1_predictor import data_loader, feature_engineering


def run_early_predictions(features_df, encoders, upcoming_info):
    """
    Trains models and generates early predictions for Qualifying and Race
    using data only up to the last completed race (no actual quali results).

    Args:
        features_df (pd.DataFrame): The combined historical and current feature data.
        encoders (dict): Dictionary of fitted LabelEncoders.
        upcoming_info (pd.DataFrame): DataFrame containing info about the next race.

    Returns:
        tuple: Contains the trained models, features used, imputation values,
               and the prediction DataFrames:
               (quali_model, quali_features, quali_imputation_values,
                race_model, race_features, race_imputation_values,
                predicted_quali_early, predicted_race_early)
               Returns None for elements if training/prediction fails.
    """
    print("\n--- Early Predictions (Based on data up to last completed race) ---")

    # Initialize return values
    quali_model, quali_features, quali_imputation_values = None, [], None
    race_model, race_features, race_imputation_values = None, [], None
    predicted_quali_early, predicted_race_early = None, None

    # --- Train Qualifying Model ---
    print("\nTraining Qualifying Model...")
    quali_model, quali_features, quali_imputation_values = model_training.train_model(
        features=features_df.copy(),
        target_column_name='Quali_Pos',
        model_type='qualifying'
    )

    # --- Predict Qualifying Results (Early) ---
    if quali_model:
        print("\nPredicting Qualifying Results (Early)...")
        predicted_quali_early = prediction.predict_results(
            model=quali_model,
            features_used=quali_features,
            upcoming_info=upcoming_info.copy(),
            historical_data=features_df.copy(),
            encoders=encoders,
            imputation_values=quali_imputation_values,
            actual_quali=None, # IMPORTANT: No actual quali known yet
            model_type='qualifying'
        )
    else:
        print("Qualifying model training failed. Skipping early qualifying prediction.")

    # --- Train Race Model ---
    print("\nTraining Race Model...")
    race_model, race_features, race_imputation_values = model_training.train_model(
        features=features_df.copy(),
        target_column_name='Finish_Pos_Clean',
        model_type='race'
    )

    # --- Predict Race Results (Early - without actual quali) ---
    if race_model:
        print("\nPredicting Race Results (Early - without actual qualifying)...")
        predicted_race_early = prediction.predict_results(
            model=race_model,
            features_used=race_features,
            upcoming_info=upcoming_info.copy(),
            historical_data=features_df.copy(),
            encoders=encoders,
            imputation_values=race_imputation_values,
            actual_quali=None, # IMPORTANT: No actual quali known yet
            model_type='race'
        )
    else:
        print("Race model training failed. Skipping early race prediction.")

    print("\n--- Early Predictions Complete ---")

    return (
        quali_model, quali_features, quali_imputation_values,
        race_model, race_features, race_imputation_values,
        predicted_quali_early, predicted_race_early 
    )
    

# Add a main execution block if you want to run this file directly
if __name__ == "__main__":
    print("Running early_predictor.py directly...")

    # 1. Load Data
    print("Loading data for direct execution...")
    loaded_data = data_loader.load_data()
    if loaded_data is None:
        print("Failed to load data needed for early predictions. Exiting.")
        sys.exit(1)
    hist_races, hist_quali, curr_races, curr_quali, upcoming_info, _ = loaded_data # Ignore latest_quali

    if upcoming_info.empty:
        print("No upcoming race found. Cannot run early predictions.")
        sys.exit(0)

    # 2. Preprocess & Engineer Features
    print("Running feature engineering...")
    features_df, encoders = feature_engineering.preprocess_and_engineer_features(
        hist_races, hist_quali, curr_races, curr_quali
    )
    if features_df.empty:
         print("Feature engineering failed. Cannot run early predictions.")
         sys.exit(1)

    # 3. Run Early Predictions (Trains models, predicts Quali and Race without actual Quali)
    early_results = run_early_predictions(
        features_df, encoders, upcoming_info.copy() # Pass copy here too
    )

    # 4. Unpack Results
    (
        quali_model, quali_features, quali_imputation_values,
        race_model, race_features, race_imputation_values,
        predicted_quali_early, predicted_race_early # This is the race prediction WITHOUT predicted quali
    ) = early_results

    # 5. Display Initial Early Predictions (Optional but Recommended)
    if predicted_quali_early is not None:
        print("\n--- Early Qualifying Prediction Results (Before using for Race) ---")
        print(predicted_quali_early[['Driver', 'Team', 'Predicted_Qualifying_Rank', 'Predicted_Qualifying_Pos']])

    if predicted_race_early is not None:
        print("\n--- Early Race Prediction Results (Based on Imputed/Historical Grid Pos) ---")
        print(predicted_race_early[['Driver', 'Team', 'Predicted_Race_Rank', 'Predicted_Race_Pos']])


    # --- NEW SECTION: Predict Race using PREDICTED Qualifying ---
    print("\n--- Predicting Race using PREDICTED Qualifying Ranks as Grid Positions ---")
    predicted_race_using_predicted_quali = None

    if race_model is not None and predicted_quali_early is not None:
        # Create a DataFrame mimicking 'latest_quali' but with predicted ranks
        # Use the PREDICTED rank as the 'Quali_Pos' which will become 'Grid_Pos'
        predicted_quali_for_race_input = predicted_quali_early[['Driver', 'Predicted_Qualifying_Rank']].copy()
        # Rename the column to match what predict_results expects for 'actual_quali'
        predicted_quali_for_race_input.rename(columns={'Predicted_Qualifying_Rank': 'Quali_Pos'}, inplace=True)

        # Now, call predict_results again for the RACE model, passing the predicted quali ranks
        predicted_race_using_predicted_quali = prediction.predict_results(
            model=race_model,
            features_used=race_features,
            upcoming_info=upcoming_info.copy(), # Use original upcoming_info
            historical_data=features_df.copy(),
            encoders=encoders,
            imputation_values=race_imputation_values,
            actual_quali=predicted_quali_for_race_input, # Pass PREDICTED quali ranks here
            model_type='race'
        )

        if predicted_race_using_predicted_quali is not None:
            print("\n--- Race Prediction Results (Using Predicted Quali Ranks as Grid) ---")
            print(predicted_race_using_predicted_quali[['Driver', 'Team', 'Predicted_Race_Rank', 'Predicted_Race_Pos']])
        else:
            print("Failed to generate race prediction using predicted qualifying ranks.")

    elif race_model is None:
         print("Race model was not trained successfully. Cannot predict race using predicted qualifying.")
    elif predicted_quali_early is None:
         print("Early qualifying prediction failed. Cannot predict race using predicted qualifying.")
    # --- END NEW SECTION ---


    print("\nDirect execution finished.") 