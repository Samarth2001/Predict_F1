from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

import f1_predictor.model_training as mt


def test_train_model_time_series_smoke_random_forest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke test: exercise the time-series CV training path with a tiny dataset."""
    original_get = mt.config.get

    def fake_get(key: str, default=None):
        overrides = {
            "training.cv_config.cv_strategy": "time_series",
            "training.cv_config.cv_folds": 2,
            "training.cv_config.purged": False,
            "training.parallel.n_jobs_per_fold": 1,
            "training.parallel.limit_estimator_threads": True,
            "training.parallel.estimator_n_jobs": 1,
            "evaluation.holdout_years": [],
            "models.race.use_ranking": False,
            "models.race.base_models": ["random_forest"],
            "models.ensemble_config.base_models": ["random_forest"],
            "models.ensemble_config.optimize_weights": False,
            "training.ensemble.random_weight_samples": 10,
            "training.ensemble.dynamic_weights": False,
            "prediction.calibration.use_quantile_models": False,
            "models.race.enable_dnf_model": False,
            "paths.models_dir": str(tmp_path / "models"),
            "paths.evaluation_dir": str(tmp_path / "evaluation"),
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

    base_date = datetime(2025, 1, 1)
    rows = []
    for race_num in range(1, 5):  # 4 event groups
        for drv_idx in range(5):  # 5 drivers per event
            i = (race_num - 1) * 5 + drv_idx
            rows.append(
                {
                    "Year": 2025,
                    "Race_Num": race_num,
                    "Date": base_date + timedelta(days=i),
                    "Quali_Pos": float(drv_idx + 1),
                    "Grid": float(drv_idx + 1),
                    "Feature_X": float(drv_idx),
                    "DNF": 0,
                    "Points": 0.0,
                    "Laps": 0.0,
                    "Position": float(drv_idx + 1),
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

