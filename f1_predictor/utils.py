# f1_predictor/utils.py
# Optional file for utility functions

import pickle
import os
import numpy as np
import pandas as pd

def save_object(obj, filepath):
    """Saves a Python object using pickle."""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        print(f"Object saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving object to {filepath}: {e}")

def load_object(filepath):
    """Loads a Python object using pickle."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        print(f"Object loaded successfully from {filepath}")
        return obj
    except Exception as e:
        print(f"Error loading object from {filepath}: {e}")
        return None

def time_to_seconds(time_str: str) -> float:
    """Convert time string (e.g., '1:23.456') to seconds."""
    if pd.isna(time_str) or time_str == '':
        return np.nan
    
    try:
        time_str = str(time_str).strip()
        if ':' in time_str:
            parts = time_str.split(':')
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            return float(time_str)
    except (ValueError, TypeError):
        return np.nan

# Example usage (would be called from main.py or training/prediction):
# save_object(encoders, os.path.join(config.DATA_DIR, 'encoders.pkl'))
# loaded_encoders = load_object(os.path.join(config.DATA_DIR, 'encoders.pkl'))

