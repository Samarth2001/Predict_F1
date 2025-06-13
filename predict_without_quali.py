import os
import sys
import pandas as pd
import numpy as np

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import necessary modules
from f1_predictor import data_loader, feature_engineering, model_training, prediction

# Load data
loaded_data = data_loader.load_data()
if loaded_data is None:
    print("Failed to load data.")
    sys.exit(1)

hist_races, hist_quali, curr_races, curr_quali, upcoming_info, _ = loaded_data

# Prepare features
feature_engineer = feature_engineering.F1FeatureEngineer()
features_df = feature_engineer.engineer_features(
    hist_races, hist_quali, curr_races, curr_quali, upcoming_info
)

# Train race model
race_model, race_features, race_imputation_values = model_training.train_model(
    features=features_df.copy(),
    target_column_name="Finish_Pos_Clean",
    model_type="race"
)

# Predict race results WITHOUT qualifying data
print("\nPredicting race results WITHOUT qualifying data:")
race_prediction = prediction.predict_results(
    model=race_model,
    features_used=race_features,
    upcoming_info=upcoming_info.copy(),
    historical_data=features_df.copy(),
    encoders={},  # No encoders needed with new approach
    imputation_values=race_imputation_values,
    actual_quali=None,  # No qualifying data
    model_type="race"
)

if race_prediction is not None:
    print("\nFinal Japanese GP 2025 Race Prediction (WITHOUT qualifying data):")
    print(race_prediction[['Driver', 'Team', 'Predicted_Race_Rank']])
    
    # Extract top 5 drivers for easy reading
    top5 = race_prediction.sort_values('Predicted_Race_Rank').head(5)
    print("\nTOP 5 PREDICTION (WITHOUT QUALIFYING DATA):")
    for i, (_, row) in enumerate(top5.iterrows()):
        print(f"{i+1}. {row['Driver']} ({row['Team']})") 