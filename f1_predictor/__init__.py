# f1_predictor/__init__.py - F1 Prediction System v3.0

"""
F1 Prediction System 2.0

A comprehensive Formula 1 prediction system with advanced machine learning,
ensemble modeling, and real-time prediction capabilities.

Modules:
    config: System configuration and constants
    data_loader: F1 data collection and loading
    feature_engineering: Advanced feature engineering for F1 data
    model_training: Ensemble model training and optimization
    prediction: Prediction engine with confidence scoring
    evaluation: Model evaluation and performance analysis
"""

from . import config
from . import data_loader
from . import feature_engineering
from . import model_training
from . import prediction
from . import evaluation
from . import utils
from . import early_predictor

# New v3.0 modules  
from . import config_manager
from . import database
from . import mlflow_integration

__version__ = "3.0.0"
__author__ = "F1 Prediction Team"
__email__ = "contact@f1prediction.ai"

# Export main classes and functions
from .data_loader import F1DataCollector, F1DataLoader, fetch_f1_data, load_data
from .feature_engineering import F1FeatureEngineer
from .model_training import F1ModelTrainer, train_model
from .prediction import F1Predictor, predict_results
from .evaluation import F1ModelEvaluator, compare_predictions

# New v3.0 exports
from .config_manager import F1ConfigManager, get_config_manager, get_config, set_config
from .database import F1Database, get_database
from .mlflow_integration import F1MLflowTracker, get_mlflow_tracker
# Schema imports temporarily disabled

__all__ = [
    # Core modules
    'config',
    'data_loader', 
    'feature_engineering',
    'model_training',
    'prediction',
    'evaluation',
    'utils',
    'early_predictor',
    
    # New v3.0 modules
    'config_manager', 
    'database',
    'mlflow_integration',
    
    # Main classes
    'F1DataCollector',
    'F1DataLoader', 
    'F1FeatureEngineer',
    'F1ModelTrainer',
    'F1Predictor',
    'F1ModelEvaluator',
    
    # New v3.0 classes
    'F1ConfigManager',
    'F1Database',
    'F1MLflowTracker',
    
    # Functions
    'fetch_f1_data',
    'load_data',
    'train_model', 
    'predict_results',
    'compare_predictions',
    
    # New v3.0 functions
    'get_config_manager',
    'get_config',
    'set_config',
    'get_database',
    'get_mlflow_tracker',
    
    # Schemas (temporarily disabled)
]
