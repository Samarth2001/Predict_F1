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

# Print driver names from upcoming_info to see exact format
print("DRIVERS FROM UPCOMING_INFO:")
for idx, driver in enumerate(upcoming_info['Driver']):
    print(f"{idx+1}. {driver}")

# Get list of drivers from upcoming_info in the exact order and format
drivers_list = upcoming_info['Driver'].tolist()

# Create Japanese GP 2025 qualifying data with EXACT driver names from upcoming_info
# This ensures we'll match the driver names precisely
japanese_quali = pd.DataFrame({
    'Driver': drivers_list,
    'Quali_Pos': list(range(1, len(drivers_list) + 1))  # Assign positions 1, 2, 3, etc.
})

# Adjust qualifying positions if needed to match the desired order
# Here we'll manually sort the qualifying results to match our desired order
# VER, NOR, PIA, LEC, RUS, ANT, HAD, HAM, ALB, BEA, etc.
# First, create a mapping of full driver names to their desired position
driver_to_quali_pos = {}

# Find the index for each key driver and assign their position
desired_order = ['max_verstappen', 'norris', 'piastri', 'leclerc', 'russell', 
                 'antonelli', 'hadjar', 'hamilton', 'albon', 'bearman']

# Assign positions 1-10 to our key drivers
for i, driver in enumerate(desired_order):
    if driver in drivers_list:
        driver_to_quali_pos[driver] = i + 1  # Position 1-10

# Assign remaining positions (11+) to other drivers
next_pos = 11
for driver in drivers_list:
    if driver not in driver_to_quali_pos:
        driver_to_quali_pos[driver] = next_pos
        next_pos += 1

# Update qualifying positions based on our mapping
japanese_quali['Quali_Pos'] = japanese_quali['Driver'].map(driver_to_quali_pos)

print("\nJapanese GP 2025 Qualifying Results:")
print(japanese_quali[['Driver', 'Quali_Pos']].sort_values('Quali_Pos'))

# Prepare features
features_df, encoders = feature_engineering.preprocess_and_engineer_features(
    hist_races, hist_quali, curr_races, curr_quali
)

# Train race model
race_model, race_features, race_imputation_values = model_training.train_model(
    features=features_df.copy(),
    target_column_name="Finish_Pos_Clean",
    model_type="race"
)

# Predict race results WITH qualifying data
print("\nPredicting race results WITH Japanese GP qualifying data:")
race_prediction = prediction.predict_results(
    model=race_model,
    features_used=race_features,
    upcoming_info=upcoming_info.copy(),
    historical_data=features_df.copy(),
    encoders=encoders,
    imputation_values=race_imputation_values,
    actual_quali=japanese_quali,
    model_type="race"
)

if race_prediction is not None:
    print("\nFinal Japanese GP 2025 Race Prediction (using qualifying data):")
    print(race_prediction[['Driver', 'Team', 'Predicted_Race_Rank']])
    
    # Extract top 5 drivers for easy reading
    top5 = race_prediction.sort_values('Predicted_Race_Rank').head(5)
    print("\nTOP 5 PREDICTION:")
    for i, (_, row) in enumerate(top5.iterrows()):
        print(f"{i+1}. {row['Driver']} ({row['Team']})") 