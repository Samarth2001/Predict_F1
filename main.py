# main.py

import os
import sys

# Add the project root to the Python path to allow imports from f1_predictor
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import necessary functions from the f1_predictor package
from f1_predictor import data_loader, feature_engineering, model_training, prediction, config
# from f1_predictor import utils # Uncomment if using utils

def run_prediction_workflow(force_fetch=False):
    """Runs the full F1 prediction workflow."""

    # 1. Fetch F1 Data (if needed or forced)
    fetch_data_flag = force_fetch
    if not os.path.exists(config.RACES_CSV_PATH) or not os.path.exists(config.QUALI_CSV_PATH):
        print("Local data files not found.")
        fetch_data_flag = True

    if fetch_data_flag:
        print("Fetching data using FastF1...")
        success = data_loader.fetch_f1_data()
        if not success:
            print("Failed to fetch data. Exiting.")
            return # Use return instead of exit() in functions
    else:
        print(f"Using existing data files in '{config.DATA_DIR}'.")

    # 2. Load Data from CSVs
    loaded_data = data_loader.load_data()
    if loaded_data is None:
        print("Failed to load data. Exiting.")
        return
    hist_races, hist_quali, curr_races, curr_quali, upcoming_info, latest_quali = loaded_data

    # Check if there's an upcoming race to predict
    if upcoming_info.empty:
        print("\nNo upcoming race found in the schedule based on loaded data. Nothing to predict.")
        return

    # 3. Preprocess & Engineer Features
    # This combines all historical and current completed data
    print("\n--- Feature Engineering ---")
    features_df, encoders = feature_engineering.preprocess_and_engineer_features(
        hist_races, hist_quali, curr_races, curr_quali
    )

    if features_df.empty:
         print("Feature engineering resulted in empty dataframe. Cannot proceed.")
         return

    # --- Train Models ---
    # Train models once based on all available historical data

    print("\n--- Model Training ---")
    # 4a. Train Qualifying Model
    quali_model, quali_features, quali_imputation_values = model_training.train_model(
        features=features_df.copy(), # Pass copy to avoid modification issues
        target_column_name='Quali_Pos',
        model_type='qualifying'
    )

    # 4b. Train Race Model
    # This model is trained on historical data where Grid_Pos was known.
    # It can still make predictions when Grid_Pos is missing (imputed).
    race_model, race_features, race_imputation_values = model_training.train_model(
        features=features_df.copy(), # Pass copy
        target_column_name='Finish_Pos_Clean',
        model_type='race'
    )

    # --- Pre-Qualifying Predictions ---
    # Predictions made BEFORE actual qualifying happens for the upcoming race

    print("\n--- Pre-Qualifying Predictions ---")

    # 5a. Predict Pre-Quali Qualifying Results
    if quali_model:
        print("\nPredicting Qualifying Results (Pre-Actual Qualifying)...")
        predicted_pre_quali_ranks = prediction.predict_results(
            model=quali_model,
            features_used=quali_features,
            upcoming_info=upcoming_info.copy(), # Pass copy
            historical_data=features_df.copy(), # Pass copy
            encoders=encoders,
            imputation_values=quali_imputation_values,
            actual_quali=None, # Crucially, actual_quali is None here
            model_type='qualifying'
        )
    else:
        print("Qualifying model training failed. Skipping pre-qualifying prediction.")

    # 5b. Predict Pre-Quali Race Results
    if race_model:
        print("\nPredicting Race Results (Pre-Actual Qualifying)...")
        # We use the race_model, but without providing actual_quali.
        # The prepare_prediction_data function will handle the missing Grid_Pos (likely via imputation).
        predicted_pre_quali_race_ranks = prediction.predict_results(
            model=race_model,
            features_used=race_features,
            upcoming_info=upcoming_info.copy(), # Pass copy
            historical_data=features_df.copy(), # Pass copy
            encoders=encoders,
            imputation_values=race_imputation_values,
            actual_quali=None, # Crucially, actual_quali is None here
            model_type='race'
        )
    else:
        print("Race model training failed. Skipping pre-qualifying race prediction.")


    # --- Post-Qualifying Race Prediction (Original Flow) ---
    # Prediction made AFTER actual qualifying results are available

    print("\n--- Post-Qualifying Race Prediction ---")

    if latest_quali.empty:
        print("\nActual qualifying results are not available (or failed to fetch). Cannot make Post-Qualifying Race Prediction.")
    elif not race_model:
         print("\nRace model training failed. Cannot make Post-Qualifying Race Prediction.")
    else:
        # 6. Predict Post-Quali Race Results
        print("\nPredicting Race Results (Using Actual Qualifying Results)...")
        predicted_post_quali_race_ranks = prediction.predict_results(
            model=race_model,
            features_used=race_features,
            upcoming_info=upcoming_info.copy(), # Pass copy
            historical_data=features_df.copy(), # Pass copy
            encoders=encoders,
            imputation_values=race_imputation_values,
            actual_quali=latest_quali.copy(), # Pass actual quali results
            model_type='race'
        )

    print("\n--- Workflow Complete ---")


if __name__ == "__main__":
    # Check for command-line argument to force data fetching
    force_fetch_arg = '--fetch' in sys.argv
    run_prediction_workflow(force_fetch=force_fetch_arg)
