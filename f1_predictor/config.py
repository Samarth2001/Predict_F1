# f1_predictor/config.py - Advanced F1 Prediction System Configuration

import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# ============================================================================
# DIRECTORY AND PATH CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'f1_data_real')
CACHE_DIR = os.path.join(DATA_DIR, 'ff1_cache')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
PREDICTIONS_DIR = os.path.join(BASE_DIR, 'predictions')
EVALUATION_DIR = os.path.join(BASE_DIR, 'evaluation')
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, 'visualizations')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Ensure all directories exist
for directory in [DATA_DIR, CACHE_DIR, MODELS_DIR, LOGS_DIR, PREDICTIONS_DIR, 
                  EVALUATION_DIR, VISUALIZATIONS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data file paths
RACES_CSV_PATH = os.path.join(DATA_DIR, 'all_races_data.csv')
QUALI_CSV_PATH = os.path.join(DATA_DIR, 'all_quali_data.csv')
WEATHER_CSV_PATH = os.path.join(DATA_DIR, 'weather_data.csv')
TIRE_STRATEGY_CSV_PATH = os.path.join(DATA_DIR, 'tire_strategy_data.csv')
TELEMETRY_CSV_PATH = os.path.join(DATA_DIR, 'telemetry_data.csv')
CIRCUIT_INFO_CSV_PATH = os.path.join(DATA_DIR, 'circuit_info_data.csv')
DRIVER_STANDINGS_CSV_PATH = os.path.join(DATA_DIR, 'driver_standings_data.csv')
CONSTRUCTOR_STANDINGS_CSV_PATH = os.path.join(DATA_DIR, 'constructor_standings_data.csv')

# ============================================================================
# DATA COLLECTION CONFIGURATION
# ============================================================================

START_YEAR = 2018  # Extended range for comprehensive historical data
END_YEAR = datetime.now().year
CURRENT_SEASON = 2025

# API Interaction Settings
FETCH_DELAY = 1.5  # Optimized delay between individual API calls for a session (e.g., after session.load())
API_RETRY_DELAY = 60  # Base delay in seconds for retrying after a rate limit error (will be increased exponentially)
API_MAX_RETRIES = 5    # Maximum number of retries for API calls when rate limited
TIMEOUT_SECONDS = 45   # Timeout for individual HTTP requests
CACHE_EXPIRY_HOURS = 24 # How long to keep API responses in requests-cache
FUTURE_EVENT_BUFFER_DAYS = 7 # How many days into the future to consider an event for data fetching (prevents fetching too far ahead)

# FastF1 specific settings
FF1_CACHE_ENABLED = True
FF1_CACHE_PATH = CACHE_DIR
FF1_PARALLEL_SESSIONS = 4

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================

# Rolling window configurations for different timeframes
ROLLING_WINDOWS = {
    'short': 3,    # Recent form (last 3 races)
    'medium': 5,   # Medium term form (last 5 races)
    'long': 10,    # Long term form (last 10 races)
    'season': 23   # Full season form
}

# Minimum data requirements
MIN_RACES_FOR_DRIVER_FEATURES = 5
MIN_RACES_FOR_TEAM_FEATURES = 3
MIN_RACES_FOR_CIRCUIT_FEATURES = 2

# Status categorization for race results
FINISHED_STATUSES = ['Finished'] + [f'+{i} Lap{"s" if i > 1 else ""}' for i in range(1, 15)]
DNF_STATUSES = [
    'DNF', 'Accident', 'Collision', 'Engine', 'Gearbox', 'Hydraulics', 
    'Transmission', 'Electrical', 'Brakes', 'Suspension', 'Power Unit',
    'Clutch', 'Wheel', 'Fuel System', 'Oil Leak', 'Throttle'
]

# Circuit categorization for specialized modeling
CIRCUIT_CATEGORIES = {
    'street': ['Monaco', 'Singapore', 'Baku', 'Las Vegas', 'Miami', 'Jeddah'],
    'high_speed': ['Monza', 'Spa-Francorchamps', 'Silverstone', 'Suzuka', 'Bahrain'],
    'technical': ['Monaco', 'Singapore', 'Hungary', 'Spain', 'Abu Dhabi'],
    'power_sensitive': ['Monza', 'Spa-Francorchamps', 'Baku', 'Las Vegas'],
    'aero_sensitive': ['Monaco', 'Singapore', 'Hungary', 'Spain'],
    'tire_degradation_high': ['Spain', 'Hungary', 'Turkey', 'Abu Dhabi'],
    'overtaking_difficult': ['Monaco', 'Singapore', 'Hungary', 'Zandvoort']
}

# Weather impact factors
WEATHER_IMPACT_FACTORS = {
    'rain_threshold': 30,  # % probability threshold for rain impact
    'temp_optimal_range': (20, 30),  # Optimal temperature range in Celsius
    'wind_speed_threshold': 25,  # km/h threshold for significant wind impact
    'humidity_threshold': 80  # % threshold for high humidity impact
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Available model types
MODEL_TYPES = {
    'lightgbm': 'LightGBM Gradient Boosting',
    'xgboost': 'XGBoost Gradient Boosting', 
    'catboost': 'CatBoost Gradient Boosting',
    'random_forest': 'Random Forest',
    'extra_trees': 'Extra Trees',
    'neural_network': 'Multi-layer Perceptron',
    'ensemble': 'Advanced Ensemble Model'
}

DEFAULT_MODEL_TYPE = 'ensemble'

# Enhanced LightGBM parameters optimized for F1 position prediction
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'n_estimators': 3000,
    'learning_rate': 0.015,
    'max_depth': 10,
    'num_leaves': 127,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 7,
    'min_child_samples': 25,
    'min_split_gain': 0.01,
    'reg_alpha': 0.15,
    'reg_lambda': 0.25,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    'early_stopping_rounds': 200,
    'categorical_feature': 'auto'
}

# Enhanced XGBoost parameters
XGB_PARAMS = {
    'objective': 'reg:absoluteerror',
    'eval_metric': 'mae',
    'n_estimators': 3000,
    'learning_rate': 0.015,
    'max_depth': 10,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'min_child_weight': 25,
    'alpha': 0.15,
    'lambda': 0.25,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
    'tree_method': 'hist',
    'early_stopping_rounds': 200
}

# CatBoost parameters (excellent for categorical features)
CATBOOST_PARAMS = {
    'loss_function': 'MAE',
    'iterations': 3000,
    'learning_rate': 0.015,
    'depth': 10,
    'l2_leaf_reg': 3,
    'bootstrap_type': 'Bayesian',
    'random_strength': 1,
    'bagging_temperature': 1,
    'od_type': 'Iter',
    'od_wait': 200,
    'random_seed': 42,
    'allow_writing_files': False,
    'verbose': False
}

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 800,
    'max_depth': 20,
    'min_samples_split': 15,
    'min_samples_leaf': 8,
    'max_features': 'sqrt',
    'bootstrap': True,
    'random_state': 42,
    'n_jobs': -1,
    'oob_score': True
}

# Extra Trees parameters  
ET_PARAMS = {
    'n_estimators': 800,
    'max_depth': 20,
    'min_samples_split': 15,
    'min_samples_leaf': 8,
    'max_features': 'sqrt',
    'bootstrap': False,
    'random_state': 42,
    'n_jobs': -1
}

# Neural Network parameters
NN_PARAMS = {
    'hidden_layer_sizes': (512, 256, 128, 64),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.005,
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.001,
    'max_iter': 2000,
    'random_state': 42,
    'early_stopping': True,
    'validation_fraction': 0.15,
    'n_iter_no_change': 50
}

# Ensemble configuration
ENSEMBLE_CONFIG = {
    'base_models': ['lightgbm', 'xgboost', 'catboost', 'random_forest'],
    'meta_learner': 'lightgbm',
    'cv_folds': 5,
    'optimize_weights': True,
    'blending_method': 'weighted_average'  # 'weighted_average', 'stacking', 'voting'
}

# Dynamic ensemble weights (will be optimized based on validation performance)
ENSEMBLE_WEIGHTS = {
    'lightgbm': 0.30,
    'xgboost': 0.25,
    'catboost': 0.25,
    'random_forest': 0.20
}

# ============================================================================
# TRAINING AND VALIDATION CONFIGURATION
# ============================================================================

# Data splitting strategy
TRAIN_VALIDATION_SPLIT = {
    'test_size': 0.15,
    'validation_size': 0.15,
    'time_aware_split': True,  # Prevent data leakage in time series
    'stratify_by': ['Year', 'Circuit'],  # Ensure balanced representation
    'random_state': 42
}

# Cross-validation configuration
CV_CONFIG = {
    'cv_folds': 7,
    'cv_strategy': 'time_series',  # 'time_series', 'stratified', 'group'
    'cv_group_by': 'Year',
    'shuffle': False,  # Important for time series data
    'random_state': 42
}

# Early stopping and regularization
TRAINING_CONFIG = {
    'early_stopping_rounds': 250,
    'early_stopping_metric': 'mae',
    'early_stopping_tolerance': 1e-6,
    'max_training_time_minutes': 120,
    'enable_pruning': True,
    'pruning_threshold': 0.95
}

# ============================================================================
# HYPERPARAMETER OPTIMIZATION CONFIGURATION
# ============================================================================

HYPEROPT_CONFIG = {
    'enabled': True,
    'n_trials': 150,
    'timeout_hours': 4,
    'n_jobs': -1,
    'cv_folds': 5,
    'optimization_metric': 'mae',
    'optimization_direction': 'minimize',
    'sampler': 'tpe',  # 'tpe', 'random', 'cmaes'
    'pruner': 'median',  # 'median', 'successive_halving', 'hyperband'
    'study_name': f'f1_prediction_study_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
}

# Parameter search spaces for optimization
PARAM_SEARCH_SPACES = {
    'lightgbm': {
        'n_estimators': (1000, 5000),
        'learning_rate': (0.01, 0.05),
        'max_depth': (8, 15),
        'num_leaves': (31, 255),
        'feature_fraction': (0.7, 0.9),
        'bagging_fraction': (0.7, 0.9),
        'reg_alpha': (0.01, 0.3),
        'reg_lambda': (0.01, 0.3)
    },
    'xgboost': {
        'n_estimators': (1000, 5000),
        'learning_rate': (0.01, 0.05),
        'max_depth': (8, 15),
        'colsample_bytree': (0.7, 0.9),
        'subsample': (0.7, 0.9),
        'alpha': (0.01, 0.3),
        'lambda': (0.01, 0.3)
    }
}

# ============================================================================
# PREDICTION CONFIGURATION
# ============================================================================

# Prediction scenarios and their requirements
PREDICTION_SCENARIOS = {
    'pre_weekend': {
        'description': 'Predictions before qualifying and race',
        'required_features': ['driver', 'team', 'circuit', 'weather', 'historical'],
        'confidence_adjustment': 0.85
    },
    'post_qualifying': {
        'description': 'Race predictions using actual qualifying results',
        'required_features': ['qualifying_position', 'qualifying_gap', 'grid_penalties'],
        'confidence_adjustment': 1.0
    },
    'live_race': {
        'description': 'Real-time predictions during race',
        'required_features': ['current_position', 'lap_times', 'tire_strategy'],
        'confidence_adjustment': 1.15
    }
}

# Default imputation values for missing features
DEFAULT_IMPUTATION_VALUES = {
    'Grid_Pos': 15.0,
    'Quali_Pos': 15.0,
    'Driver_Avg_Finish_Last_N': 12.0,
    'Team_Avg_Finish_Last_N': 12.0,
    'Circuit_Avg_Finish': 12.0,
    'Weather_Temp': 25.0,
    'Weather_Humidity': 60.0,
    'Weather_Wind_Speed': 10.0,
    'Safety_Car_Laps': 0.0,
    'Tire_Strategy_Optimal': 1.0,
    'Championship_Pressure': 0.0
}

# Confidence scoring thresholds
CONFIDENCE_THRESHOLDS = {
    'very_high': 0.9,
    'high': 0.8,
    'medium': 0.6,
    'low': 0.4,
    'very_low': 0.2
}

# ============================================================================
# EVALUATION METRICS CONFIGURATION
# ============================================================================

# Primary evaluation metrics for F1 position prediction
EVALUATION_METRICS = {
    'regression': ['mae', 'mse', 'rmse', 'mape', 'r2'],
    'ranking': ['spearman_correlation', 'kendall_tau', 'ndcg'],
    'classification': ['top3_accuracy', 'top5_accuracy', 'podium_precision', 'podium_recall'],
    'f1_specific': ['pole_position_accuracy', 'dnf_prediction_accuracy', 'points_prediction_accuracy']
}

# Position-based weights for evaluation (higher weight for top positions)
POSITION_WEIGHTS = np.array([
    5.0,   # P1 - Most important
    3.0,   # P2
    2.0,   # P3
    1.5,   # P4
    1.2,   # P5
    1.0,   # P6
    0.9,   # P7
    0.8,   # P8
    0.7,   # P9
    0.6,   # P10 - Points positions
    0.4,   # P11-P15
    0.4, 0.4, 0.4, 0.4,
    0.2,   # P16-P20
    0.2, 0.2, 0.2, 0.2
])

# Evaluation configuration
EVAL_CONFIG = {
    'cross_validate': True,
    'holdout_years': [2024],  # Hold out recent year for final testing
    'bootstrap_iterations': 1000,
    'confidence_intervals': [0.05, 0.95],  # 90% confidence intervals
    'statistical_tests': True,
    'generate_plots': True,
    'save_detailed_results': True
}

# ============================================================================
# FEATURE ENGINEERING SPECIFICATIONS
# ============================================================================

# Comprehensive feature categories with detailed specifications
FEATURE_CATEGORIES = {
    'driver_basic': {
        'features': ['Driver_Encoded', 'Driver_Age', 'Driver_Experience', 'Driver_Nationality_Encoded'],
        'description': 'Basic driver demographic and experience features'
    },
    'driver_performance': {
        'features': [
            'Driver_Avg_Finish_Last_3', 'Driver_Avg_Finish_Last_5', 'Driver_Avg_Finish_Last_10',
            'Driver_Avg_Quali_Last_3', 'Driver_Avg_Quali_Last_5', 'Driver_Avg_Quali_Last_10',
            'Driver_Win_Rate', 'Driver_Podium_Rate', 'Driver_Points_Rate', 'Driver_DNF_Rate'
        ],
        'description': 'Driver performance metrics across different timeframes'
    },
    'driver_circuit_specific': {
        'features': [
            'Driver_Circuit_Experience', 'Driver_Circuit_Avg_Finish', 'Driver_Circuit_Avg_Quali',
            'Driver_Circuit_Win_Rate', 'Driver_Circuit_Podium_Rate', 'Driver_Circuit_DNF_Rate'
        ],
        'description': 'Driver performance at specific circuits'
    },
    'driver_contextual': {
        'features': [
            'Driver_Season_Points', 'Driver_Championship_Position', 'Driver_Points_Gap_To_Leader',
            'Driver_Championship_Pressure', 'Driver_Home_Race', 'Driver_Rookie_Season'
        ],
        'description': 'Contextual driver features based on season situation'
    },
    'team_performance': {
        'features': [
            'Team_Encoded', 'Team_Avg_Finish_Last_3', 'Team_Avg_Finish_Last_5', 'Team_Avg_Finish_Last_10',
            'Team_Championship_Position', 'Team_Season_Points', 'Team_Points_Gap_To_Leader'
        ],
        'description': 'Team performance and championship standing features'
    },
    'team_technical': {
        'features': [
            'Team_Reliability_Score', 'Team_Development_Trend', 'Team_Circuit_Performance',
            'Team_Power_Unit_Manufacturer', 'Team_Budget_Tier', 'Team_Experience_Level'
        ],
        'description': 'Technical and organizational team characteristics'
    },
    'circuit_characteristics': {
        'features': [
            'Circuit_Encoded', 'Circuit_Length', 'Circuit_Turns', 'Circuit_Type',
            'Circuit_Avg_Speed', 'Circuit_Difficulty', 'Circuit_Overtaking_Difficulty',
            'Circuit_Tire_Wear_Rate', 'Circuit_DRS_Zones', 'Circuit_Elevation_Change'
        ],
        'description': 'Physical and technical circuit characteristics'
    },
    'weather_conditions': {
        'features': [
            'Weather_Temp', 'Weather_Humidity', 'Weather_Pressure', 'Weather_Wind_Speed',
            'Weather_Rain_Probability', 'Weather_Track_Temp', 'Weather_Grip_Level'
        ],
        'description': 'Weather and track condition features'
    },
    'race_context': {
        'features': [
            'Year', 'Race_Num', 'Season_Progress', 'Grid_Pos', 'Quali_Pos', 'Quali_Gap_To_Pole',
            'Grid_Penalty', 'Safety_Car_Expected', 'Race_Distance', 'Tire_Strategy_Optimal'
        ],
        'description': 'Race-specific contextual information'
    },
    'strategic_features': {
        'features': [
            'Pit_Window_Optimal', 'Tire_Compound_Strategy', 'Fuel_Load_Effect',
            'Aerodynamic_Package', 'Setup_Risk_Level', 'Team_Strategy_Aggressiveness'
        ],
        'description': 'Strategic and setup-related features'
    }
}

# Feature importance tracking
FEATURE_IMPORTANCE_CONFIG = {
    'methods': ['shap', 'permutation', 'gain', 'split'],
    'save_plots': True,
    'top_n_features': 30,
    'correlation_threshold': 0.95  # Remove highly correlated features
}

# ============================================================================
# LOGGING AND MONITORING CONFIGURATION
# ============================================================================

LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    'file_path': os.path.join(LOGS_DIR, f'f1_predictor_{datetime.now().strftime("%Y%m%d")}.log'),
    'max_file_size_mb': 100,
    'backup_count': 5,
    'enable_console_output': True,
    'enable_performance_logging': True
}

# Performance monitoring
PERFORMANCE_CONFIG = {
    'track_training_time': True,
    'track_prediction_time': True,
    'track_memory_usage': True,
    'enable_profiling': False,  # Enable for detailed performance analysis
    'save_performance_metrics': True
}

# ============================================================================
# API AND EXTERNAL DATA CONFIGURATION
# ============================================================================

API_CONFIG = {
    'f1_api_base_url': 'https://api.openf1.org/v1',
    'ergast_api_base_url': 'http://ergast.com/api/f1',
    'weather_api_key': None,  # Set if using external weather API
    'rate_limit_requests_per_minute': 60,
    'request_timeout_seconds': 30,
    'enable_caching': True,
    'cache_ttl_hours': 24
}

# ============================================================================
# 2025 SEASON SPECIFIC CONFIGURATION
# ============================================================================

SEASON_2025_CONFIG = {
    'total_races': 24,
    'calendar_confirmed': True,
    'new_circuits': [],  # Add any new circuits for 2025
    'regulation_changes': {
        'major_changes': False,
        'technical_changes': [],
        'sporting_changes': []
    },
    'prediction_targets': {
        'championship_winner': True,
        'constructors_champion': True,
        'race_winners': True,
        'podium_finishers': True,
        'points_scorers': True
    }
}

# Race weekend prediction schedule
RACE_WEEKEND_SCHEDULE = {
    'pre_weekend_deadline': 'Thursday 18:00 UTC',
    'post_qualifying_deadline': '30 minutes after qualifying',
    'live_updates_interval_minutes': 5,
    'final_prediction_deadline': 'Formation lap'
}

# ============================================================================
# COMPATIBILITY CONSTANTS (for backward compatibility)
# ============================================================================

# Legacy constants referenced in other modules
LOG_LEVEL = LOG_CONFIG['level']
LOG_FORMAT = LOG_CONFIG['format']
LOG_FILE = LOG_CONFIG['file_path']
TEST_SIZE = TRAIN_VALIDATION_SPLIT['test_size']
RANDOM_STATE = 42
CV_FOLDS = CV_CONFIG['cv_folds']
EARLY_STOPPING_ROUNDS = TRAINING_CONFIG['early_stopping_rounds']
MODEL_VERSION = "2.0.0"

# Model optimization settings
ENABLE_HYPERPARAMETER_OPTIMIZATION = HYPEROPT_CONFIG['enabled']
ENABLE_FEATURE_SELECTION = True
ENABLE_MODEL_STACKING = True
OPTUNA_N_TRIALS = HYPEROPT_CONFIG['n_trials']
OPTUNA_TIMEOUT = HYPEROPT_CONFIG['timeout_hours'] * 3600
MAX_FEATURES_SELECTION = 50

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLDS['high']
MEDIUM_CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLDS['medium']

# Feature saving settings
SAVE_PREDICTIONS = True
SAVE_FEATURE_IMPORTANCE = True

# Circuit categorization (simplified)
STREET_CIRCUITS = CIRCUIT_CATEGORIES['street']
HIGH_SPEED_CIRCUITS = CIRCUIT_CATEGORIES['high_speed']
TECHNICAL_CIRCUITS = CIRCUIT_CATEGORIES['technical']

# Rolling window settings (backward compatibility)
N_ROLLING_SHORT = ROLLING_WINDOWS['short']
N_ROLLING_MEDIUM = ROLLING_WINDOWS['medium']
N_ROLLING_LONG = ROLLING_WINDOWS['long']

# ============================================================================
# ADVANCED FEATURES AND EXPERIMENTAL SETTINGS
# ============================================================================

EXPERIMENTAL_FEATURES = {
    'deep_learning_models': False,  # Enable for neural network experiments
    'time_series_forecasting': False,  # Enable for advanced time series models
    'multi_objective_optimization': False,  # Enable for Pareto optimization
    'automated_feature_engineering': False,  # Enable for automated feature discovery
    'reinforcement_learning': False,  # Enable for strategy optimization
    'causal_inference': False,  # Enable for causal relationship analysis
    'uncertainty_quantification': True,  # Enable for prediction uncertainty
    'explainable_ai': True  # Enable for model interpretability
}

# Advanced ensemble methods
ADVANCED_ENSEMBLE_CONFIG = {
    'bayesian_model_averaging': False,
    'dynamic_ensemble_selection': False,
    'meta_learning': False,
    'multi_level_stacking': True,
    'temporal_ensemble': True
}

# ============================================================================
# UTILITY FUNCTIONS FOR CONFIGURATION
# ============================================================================

def get_feature_list(categories: List[str] = None) -> List[str]:
    """Get list of features from specified categories."""
    if categories is None:
        categories = list(FEATURE_CATEGORIES.keys())
    
    features = []
    for category in categories:
        if category in FEATURE_CATEGORIES:
            features.extend(FEATURE_CATEGORIES[category]['features'])
    
    return features

def get_model_params(model_type: str) -> Dict[str, Any]:
    """Get parameters for specified model type."""
    param_map = {
        'lightgbm': LGBM_PARAMS,
        'xgboost': XGB_PARAMS,
        'catboost': CATBOOST_PARAMS,
        'random_forest': RF_PARAMS,
        'extra_trees': ET_PARAMS,
        'neural_network': NN_PARAMS
    }
    return param_map.get(model_type, {})

def get_circuit_category(circuit_name: str) -> List[str]:
    """Get categories for a specific circuit."""
    categories = []
    for category, circuits in CIRCUIT_CATEGORIES.items():
        if circuit_name in circuits:
            categories.append(category)
    return categories

def is_experimental_feature_enabled(feature_name: str) -> bool:
    """Check if an experimental feature is enabled."""
    return EXPERIMENTAL_FEATURES.get(feature_name, False)

# Configuration validation
def validate_config() -> bool:
    """Validate configuration settings."""
    try:
        # Check directory accessibility
        for directory in [MODELS_DIR, LOGS_DIR, PREDICTIONS_DIR]:
            if not os.access(directory, os.W_OK):
                print(f"Warning: Directory {directory} is not writable")
        
        # Check parameter consistency
        if HYPEROPT_CONFIG['n_trials'] < CV_CONFIG['cv_folds']:
            print("Warning: Number of hyperopt trials is less than CV folds")
        
        # Check feature categories
        all_features = get_feature_list()
        if len(all_features) != len(set(all_features)):
            print("Warning: Duplicate features found in feature categories")
        
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

# Initialize configuration
if __name__ == "__main__":
    validate_config()
    print("F1 Prediction System Configuration Loaded Successfully")
    print(f"Total features configured: {len(get_feature_list())}")
    print(f"Model types available: {list(MODEL_TYPES.keys())}")
    print(f"Experimental features enabled: {sum(EXPERIMENTAL_FEATURES.values())}")

