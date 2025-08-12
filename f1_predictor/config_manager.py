"""f1_predictor/config_manager.py - Configuration management with local overrides and env var support"""

import os
import yaml
import json
import hashlib
import logging
import re
from copy import deepcopy
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge override into base and return the merged dict (does not mutate inputs)."""
    if not isinstance(base, dict):
        base = {}
    result = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def _expand_env_vars_in_value(value: Any) -> Any:
    """Expand ${VAR} or ${VAR:-default} occurrences in strings; leave others unchanged."""
    if not isinstance(value, str):
        return value

    pattern = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")

    def repl(match: re.Match) -> str:
        var = match.group(1)
        default = match.group(2) if match.lastindex and match.group(2) is not None else None
        env_val = os.getenv(var)
        if env_val is not None and env_val != "":
            return env_val
        return default if default is not None else ""

                                                                                               
    expanded = pattern.sub(repl, value)
    expanded = os.path.expandvars(expanded)
    return expanded


def _expand_env_vars_in_config(cfg: Any) -> Any:
    if isinstance(cfg, dict):
        return {k: _expand_env_vars_in_config(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_expand_env_vars_in_config(v) for v in cfg]
    return _expand_env_vars_in_value(cfg)


class ConfigManager:
    """Centralized configuration loader with local.yaml overlay and env var interpolation.

    - Loads `config/default_config.yaml` as the single source of truth for tunables
    - Optionally overlays `config/local.yaml` (gitignored) for secrets/overrides
    - Supports environment variable expansion in YAML values via ${VAR} or ${VAR:-default}
    - Provides a stable hash of the merged (pre-resolved) configuration for metadata versioning
    """

    def __init__(self, config_path: str = "config/default_config.yaml", local_config_path: str = "config/local.yaml"):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self._default_config_path = os.path.join(self.base_dir, config_path)
        self._local_config_path = os.path.join(self.base_dir, local_config_path)

                                                                                   
        default_cfg = self._load_yaml(self._default_config_path)
        local_cfg = self._load_yaml(self._local_config_path, required=False)
        merged = _deep_update(default_cfg, local_cfg or {})
        merged = _expand_env_vars_in_config(merged)

                                                                               
        self._raw_config_merged = deepcopy(merged)
        self._config_hash = self._compute_hash(self._raw_config_merged)

                                                                        
        self._config = deepcopy(merged)
        self._resolve_paths()
        self._resolve_dynamic_values()

    def _load_yaml(self, path: str, required: bool = True) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                if not isinstance(data, dict):
                    raise Exception(f"Config at {path} must be a YAML mapping/object.")
                return data
        except FileNotFoundError:
            if required:
                raise Exception(f"Configuration file not found at: {path}")
            logger.info(f"Optional local config not found at: {path}; continuing without it.")
            return {}
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing YAML configuration file {path}: {e}")

    def _compute_hash(self, cfg: Dict[str, Any]) -> str:
                                                                                          
        payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _resolve_paths(self) -> None:
        paths_config = self.get("paths", {})
        if not paths_config:
            return
        for key, value in paths_config.items():
            if key != "base_dir" and isinstance(value, str):
                paths_config[key] = os.path.join(self.base_dir, value)
                                  
        for key, path in paths_config.items():
            if key.endswith("_dir") and isinstance(path, str) and not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

    def _resolve_dynamic_values(self) -> None:
        data_collection_config = self.get("data_collection", {})
        if data_collection_config and data_collection_config.get("end_year") == "auto":
            data_collection_config["end_year"] = datetime.now().year
        if data_collection_config and data_collection_config.get("current_season") == "auto":
            data_collection_config["current_season"] = datetime.now().year

    def get_config_hash(self) -> str:
        """Return the stable hash of the merged configuration before path resolution."""
        return self._config_hash

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value with dotted-path access (e.g., 'a.b.c')."""
        keys = key.split(".")
        value: Any = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        value = self.get(key)
        if value is None:
            raise KeyError(f"Configuration key '{key}' not found.")
        return value

    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        return self.get(f"models.{model_type}_params", {})

    def get_feature_list(self, categories: Optional[List[str]] = None) -> List[str]:
        feature_specs = self.get("feature_specifications", {})
        if not feature_specs:
            return []
        if categories is None:
            categories = list(feature_specs.keys())
        features: List[str] = []
        for category in categories:
            if category in feature_specs:
                features.extend(feature_specs[category].get("features", []))
                                                  
        return list(dict.fromkeys(features))


config_manager = ConfigManager()