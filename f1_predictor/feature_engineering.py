# f1_predictor/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Import from config within the package
from . import config

def preprocess_and_engineer_features(hist_races, hist_quali, curr_races, curr_quali):
    """Combines, preprocesses data and engineers features."""
    print("\nPreprocessing data and engineering features...")

    if hist_races.empty and curr_races.empty:
        print("No race data to process.")
        return pd.DataFrame(), {} # Return empty dataframe and empty encoders dict
    # Allow proceeding even if quali data is missing, features using it will be NaN
    # if hist_quali.empty and curr_quali.empty:
    #      print("Warning: No quali data to process.")

    # Combine historical and current data
    races = pd.concat([hist_races, curr_races], ignore_index=True)
    quali = pd.concat([hist_quali, curr_quali], ignore_index=True) if not (hist_quali.empty and curr_quali.empty) else pd.DataFrame()


    # --- Basic Preprocessing ---
    for df in [races, quali]:
         if not df.empty:
            for col in df.columns:
                # Identify columns to convert safely
                if any(c in col for c in ['Pos', 'Year', 'Num', 'Laps', 'Points', 'Time']):
                    df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle DNF/Status in race finish position
    max_pos = races['Finish_Pos'].max() + 1 if not races['Finish_Pos'].isnull().all() else config.DEFAULT_IMPUTATION_VALUE + 1
    races['Finish_Pos_Clean'] = races.apply(
        lambda row: row['Finish_Pos'] if row['Status'] in config.FINISHED_STATUSES else max_pos,
        axis=1
    )
    races['Finish_Pos_Clean'] = pd.to_numeric(races['Finish_Pos_Clean'], errors='coerce')

    # --- Feature Engineering ---
    # Sort data chronologically for rolling features
    races = races.sort_values(by=['Year', 'Race_Num'])
    if not quali.empty:
        quali = quali.sort_values(by=['Year', 'Race_Num'])

        # Merge race and quali data
        # Use outer merge to keep all race/quali entries, even if one is missing
        quali_to_merge = quali[['Year', 'Race_Num', 'Driver', 'Quali_Pos', 'Quali_Time_Seconds']].drop_duplicates()
        data = pd.merge(races, quali_to_merge, on=['Year', 'Race_Num', 'Driver'], how='left')
    else:
        # If no quali data, proceed with just race data
        data = races.copy()
        # Add placeholder columns for features that rely on quali, they will remain NaN
        data['Quali_Pos'] = np.nan
        data['Quali_Time_Seconds'] = np.nan


    # Rolling Driver Performance
    data['Driver_Avg_Finish_Last_N'] = data.groupby('Driver')['Finish_Pos_Clean'].transform(lambda x: x.shift(1).rolling(config.N_ROLLING, min_periods=1).mean())
    data['Driver_Avg_Quali_Last_N'] = data.groupby('Driver')['Quali_Pos'].transform(lambda x: x.shift(1).rolling(config.N_ROLLING, min_periods=1).mean())
    data['Driver_Avg_Grid_Last_N'] = data.groupby('Driver')['Grid_Pos'].transform(lambda x: x.shift(1).rolling(config.N_ROLLING, min_periods=1).mean())

    # Team Performance (Average finish pos of *both* drivers for the team in last N races)
    # Calculate mean finish per team per race first
    team_race_avg_finish = data.groupby(['Year', 'Race_Num', 'Team'])['Finish_Pos_Clean'].mean().reset_index()
    # Calculate rolling average of that team average
    team_race_avg_finish['Team_Avg_Finish_Last_N'] = team_race_avg_finish.groupby('Team')['Finish_Pos_Clean'].transform(lambda x: x.shift(1).rolling(config.N_ROLLING, min_periods=1).mean())
    # Merge back into main dataframe
    data = pd.merge(data, team_race_avg_finish[['Year', 'Race_Num', 'Team', 'Team_Avg_Finish_Last_N']], on=['Year', 'Race_Num', 'Team'], how='left')

    # Circuit Specific Performance (Expanding mean - considers all past races at the circuit)
    data['Driver_Avg_Finish_Circuit'] = data.groupby(['Driver', 'Circuit'])['Finish_Pos_Clean'].transform(lambda x: x.shift(1).expanding().mean())
    data['Driver_Avg_Quali_Circuit'] = data.groupby(['Driver', 'Circuit'])['Quali_Pos'].transform(lambda x: x.shift(1).expanding().mean())

    # Encode Categorical Features
    encoders = {}
    for col in ['Driver', 'Team', 'Circuit']:
        le = LabelEncoder()
        # Combine all unique values from both races and quali (if exists) for fitting
        all_values_list = []
        if col in races.columns:
            all_values_list.append(races[col].astype(str))
        if not quali.empty and col in quali.columns:
             all_values_list.append(quali[col].astype(str))

        if all_values_list:
            all_values = pd.concat(all_values_list, ignore_index=True).unique()
            le.fit(all_values)
            # Apply transform only to the final 'data' dataframe
            data[col + '_Encoded'] = le.transform(data[col].astype(str))
            encoders[col] = le
        else:
             print(f"Warning: Column '{col}' not found for encoding.")
             data[col + '_Encoded'] = -1 # Assign default if column missing


    print("Feature engineering complete.")
    return data, encoders

