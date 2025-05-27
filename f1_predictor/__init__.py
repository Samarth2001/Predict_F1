# f1_predictor/__init__.py - F1 Prediction System Package

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

__version__ = "2.0.0"
__author__ = "F1 Prediction Team"
__description__ = "Advanced F1 Prediction System with Ensemble Learning"

# Import main classes for easy access
from .config import *
from .data_loader import F1DataCollector, F1DataLoader, fetch_f1_data, load_data
from .feature_engineering import F1FeatureEngineer
from .model_training import F1ModelTrainer, train_model
from .prediction import F1Predictor, predict_results
from .evaluation import F1ModelEvaluator, compare_predictions

__all__ = [
    # Configuration
    'config',
    
    # Data handling
    'F1DataCollector',
    'F1DataLoader', 
    'fetch_f1_data',
    'load_data',
    
    # Feature engineering
    'F1FeatureEngineer',
    
    # Model training
    'F1ModelTrainer',
    'train_model',
    
    # Prediction
    'F1Predictor',
    'predict_results',
    
    # Evaluation
    'F1ModelEvaluator',
    'compare_predictions'
]
