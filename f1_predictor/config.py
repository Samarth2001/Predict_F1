# f1_predictor/config.py

import os
from datetime import datetime

# --- Directory Setup --- 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'f1_data_real')
CACHE_DIR = os.path.join(DATA_DIR, 'ff1_cache')
RACES_CSV_PATH = os.path.join(DATA_DIR, 'all_races_data.csv')
QUALI_CSV_PATH = os.path.join(DATA_DIR, 'all_quali_data.csv')

# --- Model Selection ---
MODEL_TYPE = 'xgboost'  # Options: 'lightgbm', 'xgboost', 'ensemble'

# --- Data Fetching ---
START_YEAR = 2021   
END_YEAR = datetime.now().year
FETCH_DELAY = 1.5  # Keep this to avoid API rate limits

# --- Feature Engineering ---
N_ROLLING = 5  # Good balance between recent performance and stability
FINISHED_STATUSES = ['Finished'] + [f'+{i} Lap' + ('s' if i > 1 else '') for i in range(1, 10)]

# --- LightGBM Model Training ---
LGBM_PARAMS = {
    'objective': 'regression_l1',  # MAE is appropriate for position prediction
    'metric': 'mae',
    'n_estimators': 1500,          
    'learning_rate': 0.03,         
    'max_depth': 7,                
    'num_leaves': 31,              
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,             
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
    'reg_alpha': 0.1,              
    'reg_lambda': 0.3,             
}

# --- XGBoost Model Training ---
XGB_PARAMS = {
    'objective': 'reg:absoluteerror',  # MAE optimization (equivalent to regression_l1)
    'eval_metric': 'mae',
    'n_estimators': 1500,
    'learning_rate': 0.03,
    'max_depth': 7,
    'colsample_bytree': 0.8,      # Equivalent to feature_fraction
    'subsample': 0.8,             # Equivalent to bagging_fraction
    'verbosity': 0,
    'n_jobs': -1,
    'seed': 42,
    'alpha': 0.1,                 # L1 regularization
    'lambda': 0.3,                # L2 regularization
    'tree_method': 'hist',        # For faster training
}

# --- Ensemble Weighting ---
# For ensemble approach, how to weight different model predictions
ENSEMBLE_WEIGHTS = {
    'lightgbm': 0.5,
    'xgboost': 0.5
}

EARLY_STOPPING_ROUNDS = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Prediction ---
DEFAULT_IMPUTATION_VALUE = 99

