"""
This module provides access to the application's configuration.

It uses a centralized ConfigManager to load settings from a YAML file.
This is the single source of truth for all configuration.

To access configuration values, import the `config` object and use it like a dictionary:

    from f1_predictor.config import config
    
    start_year = config.get('data_collection.start_year')
    api_retries = config['data_collection.api_max_retries']
"""
from .config_manager import config_manager
config = config_manager

