from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import f1_predictor.feature_engineering_pipeline as fe


def test_prepare_data_merges_quali_and_handles_missing_status(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the pipeline minimal for unit tests.
    original_get = fe.config.get

    def fake_get(key: str, default=None):
        if key == "feature_engineering.pipeline_steps":
            return []
        if key == "feature_engineering.cache.enabled":
            return False
        return original_get(key, default)

    monkeypatch.setattr(fe.config, "get", fake_get)

    races = pd.DataFrame(
        {
            "Year": [2025, 2025, 2025],
            "Race_Num": [1, 1, 2],
            "Driver": ["AAA", "BBB", "AAA"],
            "Team": ["T1", "T2", "T1"],
            "Circuit": ["X Grand Prix", "X Grand Prix", "Y Grand Prix"],
            "Date": [date(2025, 3, 1), date(2025, 3, 1), date(2025, 3, 15)],
            "Grid": [1, 2, None],
            "Position": [1, 2, None],
            "Status": ["Finished", "Finished", None],  # upcoming-like row
        }
    )
    quali = pd.DataFrame(
        {
            "Year": [2025, 2025],
            "Race_Num": [1, 1],
            "Driver": ["AAA", "BBB"],
            "Position": [2, 1],
            "Date": [date(2025, 3, 1), date(2025, 3, 1)],
        }
    )

    pipe = fe.FeatureEngineeringPipeline(races, quali)
    prepared = pipe._prepare_data()
    assert not prepared.empty

    # Quali_Pos should be merged in from quali.Position.
    assert "Quali_Pos" in prepared.columns
    # Missing status should not be treated as DNF=1.
    dnf_last = prepared.loc[(prepared["Year"] == 2025) & (prepared["Race_Num"] == 2), "DNF"].iloc[0]
    assert float(dnf_last) == 0.0

