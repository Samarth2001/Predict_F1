# f1_predictor/config.py

import os
from datetime import datetime

# --- Directory Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root
DATA_DIR = os.path.join(BASE_DIR, 'f1_data_real') # Directory to store fetched data
CACHE_DIR = os.path.join(DATA_DIR, 'ff1_cache') # fastf1 cache location
RACES_CSV_PATH = os.path.join(DATA_DIR, 'all_races_data.csv')
QUALI_CSV_PATH = os.path.join(DATA_DIR, 'all_quali_data.csv')

# --- Data Fetching ---
START_YEAR = 2021 # First year of data to fetch
END_YEAR = datetime.now().year # Fetch up to the current year (inclusive)
FETCH_DELAY = 0.5 # Seconds delay between fetching rounds to be polite to APIs

# --- Feature Engineering ---
N_ROLLING = 5 # Window size for rolling average features
FINISHED_STATUSES = ['Finished'] + [f'+{i} Lap' + ('s' if i > 1 else '') for i in range(1, 10)]

# --- Model Training ---
# LightGBM parameters (can be tuned further)
LGBM_PARAMS = {
    'objective': 'regression_l1', # MAE loss, good for ranks/positions
    'metric': 'mae',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}
EARLY_STOPPING_ROUNDS = 100
TEST_SIZE = 0.2 # Proportion of data for test set in training split
RANDOM_STATE = 42

# --- Prediction ---
DEFAULT_IMPUTATION_VALUE = 99 # Value for missing grid positions if needed

