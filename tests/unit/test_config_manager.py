from __future__ import annotations

import os

import pytest

from f1_predictor.config_manager import (
    ConfigManager,
    _deep_update,
    _expand_env_vars_in_config,
    _expand_env_vars_in_value,
)


def test_deep_update_merges_nested_dicts_without_mutation() -> None:
    base = {"a": {"b": 1}, "c": 1}
    override = {"a": {"d": 2}, "c": 2}
    merged = _deep_update(base, override)

    assert merged == {"a": {"b": 1, "d": 2}, "c": 2}
    # Ensure originals are not mutated
    assert base == {"a": {"b": 1}, "c": 1}
    assert override == {"a": {"d": 2}, "c": 2}


def test_expand_env_vars_in_value_supports_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CFG_TEST_KEY", "hello")
    assert _expand_env_vars_in_value("${CFG_TEST_KEY}") == "hello"

    monkeypatch.delenv("CFG_TEST_MISSING", raising=False)
    assert _expand_env_vars_in_value("${CFG_TEST_MISSING:-fallback}") == "fallback"


def test_expand_env_vars_in_config_expands_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CFG_TEST_NESTED", "x")
    cfg = {"a": {"b": "${CFG_TEST_NESTED:-y}"}}
    out = _expand_env_vars_in_config(cfg)
    assert out["a"]["b"] == "x"


def test_config_manager_resolves_paths_to_repo_abs() -> None:
    cm = ConfigManager(config_path="configs/default.yaml", local_config_path="configs/does_not_exist.yaml")
    models_dir = cm.get("paths.models_dir")
    assert isinstance(models_dir, str)
    assert os.path.isabs(models_dir)
    assert models_dir.startswith(cm.base_dir)
    assert len(cm.get_config_hash()) == 64

