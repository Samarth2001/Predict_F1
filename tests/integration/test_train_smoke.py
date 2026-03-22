from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

import f1_predictor.model_training as mt


def test_train_model_smoke_random_forest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke test: train a tiny race model without network calls or heavy HPO."""
    original_get = mt.config.get

    def fake_get(key: str, default=None):
        overrides = {
            # Keep training fast
            "training.cv_config.cv_strategy": "simple",
            "training.split_config.test_size": 0.4,
            "evaluation.holdout_years": [],
            "models.race.use_ranking": False,
            "models.race.base_models": ["random_forest"],
            "models.ensemble_config.base_models": ["random_forest"],
            "models.ensemble_config.optimize_weights": False,
            "prediction.calibration.use_quantile_models": False,
            "training.ensemble.dynamic_weights": False,
            "models.race.enable_dnf_model": False,
            # Use temp output dirs
            "paths.models_dir": str(tmp_path / "models"),
            "paths.evaluation_dir": str(tmp_path / "evaluation"),
            # Shrink RF to keep runtime low
            "models.random_forest_params": {
                "n_estimators": 10,
                "max_depth": 5,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
                "n_jobs": 1,
            },
        }
        if key in overrides:
            return overrides[key]
        return original_get(key, default)

    monkeypatch.setattr(mt.config, "get", fake_get)

    # Tiny synthetic dataset with a Date column for time-aware splitting.
    base_date = datetime(2025, 1, 1)
    rows = []
    for i in range(30):
        rows.append(
            {
                "Year": 2025,
                "Race_Num": int(i // 10) + 1,
                "Date": base_date + timedelta(days=i),
                "Quali_Pos": float((i % 10) + 1),
                "Grid": float((i % 10) + 1),
                "Feature_X": float(i % 5),
                # Leaky/outcome columns should be dropped by _prepare_training_data
                "DNF": 0,
                "Points": 0.0,
                "Laps": 0.0,
                # Target
                "Position": float((i % 10) + 1),
            }
        )
    features = pd.DataFrame(rows)

    trainer = mt.F1ModelTrainer()
    model, feature_names, _ = trainer.train_model(
        features, target_column_name="Position", model_type="race"
    )

    assert model is not None
    assert "DNF" not in feature_names
    assert "Points" not in feature_names
    assert "Laps" not in feature_names

