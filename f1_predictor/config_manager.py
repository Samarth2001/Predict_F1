# f1_predictor/config_manager.py - Simple YAML configuration management

import os
import yaml
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfigManager:
    """A centralized manager for handling configuration."""

    def __init__(self, config_path: str = 'config/default_config.yaml'):
        """
        Initializes the ConfigManager.

        Args:
            config_path (str): The path to the configuration file.
        """
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self._config_path = os.path.join(self.base_dir, config_path)
        self._config = self._load_config()
        self._resolve_paths()
        self._resolve_dynamic_values()

    def _load_config(self) -> Dict[str, Any]:
        """Loads the YAML configuration file."""
        try:
            with open(self._config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise Exception(f"Configuration file not found at: {self._config_path}")
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing YAML configuration file: {e}")

    def _resolve_paths(self) -> None:
        """Resolves all relative paths in the configuration."""
        paths_config = self.get('paths', {})
        if not paths_config:
            return

        for key, value in paths_config.items():
            if key != 'base_dir' and isinstance(value, str):
                # We assume paths in yaml are relative to base_dir
                paths_config[key] = os.path.join(self.base_dir, value)
        
        # Ensure directories exist
        for key, path in paths_config.items():
            if key.endswith('_dir') and not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

    def _resolve_dynamic_values(self) -> None:
        """Resolves dynamic values in the configuration."""
        data_collection_config = self.get('data_collection', {})
        if data_collection_config and data_collection_config.get('end_year') == 'auto':
            data_collection_config['end_year'] = datetime.now().year
        if data_collection_config and data_collection_config.get('current_season') == 'auto':
            data_collection_config['current_season'] = datetime.now().year

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value.

        Args:
            key (str): The configuration key (e.g., 'data_collection.start_year').
            default (Any): The default value to return if the key is not found.

        Returns:
            Any: The configuration value.
        """
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        """Allows dictionary-style access to configuration."""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Configuration key '{key}' not found.")
        return value

    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """Returns parameters for a specific model."""
        return self.get(f'models.{model_type}_params', {})

    def get_feature_list(self, categories: List[str] = None) -> List[str]:
        """Get list of features from specified categories."""
        feature_specs = self.get('feature_specifications', {})
        if not feature_specs:
            return []
            
        if categories is None:
            categories = list(feature_specs.keys())
        
        features = []
        for category in categories:
            if category in feature_specs:
                features.extend(feature_specs[category].get('features', []))
        
        return list(dict.fromkeys(features)) # Remove duplicates while preserving order

# Create a single instance to be used throughout the application
config_manager = ConfigManager() 