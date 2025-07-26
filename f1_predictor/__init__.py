# f1_predictor/__init__.py

"""
F1 Prediction System - Core Package
"""

# Import key components for easy access
from .config import config
from .data_loader import F1DataLoader, F1DataCollector
from .feature_engineering_pipeline import FeatureEngineeringPipeline
from .model_training import F1ModelTrainer
from .prediction import F1Predictor

# Package metadata
__version__ = "4.0.0"
__author__ = "F1 Prediction Team"
