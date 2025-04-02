# f1_predictor/utils.py
# Optional file for utility functions

import pickle
import os

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

# Example usage (would be called from main.py or training/prediction):
# save_object(encoders, os.path.join(config.DATA_DIR, 'encoders.pkl'))
# loaded_encoders = load_object(os.path.join(config.DATA_DIR, 'encoders.pkl'))

