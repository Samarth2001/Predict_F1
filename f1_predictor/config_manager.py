# f1_predictor/config_manager.py - Simple YAML configuration management

import os
import yaml
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class F1ConfigManager:
    """Simple YAML configuration manager."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_dir = config_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
        self.config: Optional[Dict] = None
        self._ensure_config_dir()
        self._load_config()
        
    def _ensure_config_dir(self):
        """Ensure config directory exists."""
        os.makedirs(self.config_dir, exist_ok=True)
        
    def _load_config(self):
        """Load configuration from YAML file."""
        config_file = os.path.join(self.config_dir, "default_config.yaml")
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                self.config = self._get_fallback_config()
        else:
            logger.warning(f"Config file not found: {config_file}, using defaults")
            self.config = self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict:
        """Get fallback configuration with default values."""
        return {
            'model': {
                'default_type': 'ensemble',
                'algorithms': ['lightgbm', 'xgboost', 'random_forest', 'neural_network'],
                'ensemble_weights': {
                    'lightgbm': 0.3,
                    'xgboost': 0.25,
                    'random_forest': 0.25,
                    'neural_network': 0.2
                }
            },
            'features': {
                'rolling_windows': {'short': 3, 'medium': 5, 'long': 10},
                'correlation_threshold': 0.95
            },
            'data': {
                'start_year': 2018,
                'current_season': 2025,
                'database': {'path': 'f1_data_real/f1_prediction.db'}
            },
            'training': {
                'test_size': 0.15,
                'cv_folds': 5,
                'random_state': 42
            },
            'prediction': {
                'confidence_thresholds': {
                    'high': 0.8,
                    'medium': 0.6,
                    'low': 0.4
                }
            },
            'mlflow': {
                'tracking_uri': 'file:./mlruns',
                'experiment_name': 'f1_prediction_v3',
                'log_models': True,
                'log_artifacts': True
            },
            'performance': {
                'use_polars': False,
                'parallel_processing': True,
                'n_jobs': -1
            }
        }
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        if not self.config:
            return default
            
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        if not self.config:
            self.config = {}
            
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
            
        current[keys[-1]] = value
        
    def save(self, config_file: Optional[str] = None):
        """Save configuration to file."""
        if not config_file:
            config_file = os.path.join(self.config_dir, "default_config.yaml")
            
        try:
            with open(config_file, 'w') as f:
                yaml.safe_dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """Get model-specific parameters."""
        return self.get(f'model.{model_type}', {})
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self.get('features', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get('training', {})
    
    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow configuration."""
        return self.get('mlflow', {})

# Global configuration manager instance
_config_manager = None

def get_config_manager() -> F1ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = F1ConfigManager()
    return _config_manager

def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value using global manager."""
    return get_config_manager().get(key, default)

def set_config(key: str, value: Any):
    """Set configuration value using global manager."""
    get_config_manager().set(key, value)

def get_model_params(model_type: str) -> Dict[str, Any]:
    """Get model parameters for backward compatibility."""
    return get_config_manager().get_model_params(model_type)

def get_ensemble_weights() -> Dict[str, float]:
    """Get ensemble weights for backward compatibility."""
    return get_config('model.ensemble_weights', {})

def get_rolling_windows() -> Dict[str, int]:
    """Get rolling windows configuration."""
    return get_config('features.rolling_windows', {'short': 3, 'medium': 5, 'long': 10}) 