from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import f1_predictor.feature_engineering_pipeline as fe
from f1_predictor.prediction_features import PredictionFeatureBuilder


class _FakeLoader:
    def __init__(self, hist_races: pd.DataFrame, hist_quali: pd.DataFrame):
        self._hist_races = hist_races
        self._hist_quali = hist_quali

    def load_all_data(self):
        return self._hist_races.copy(), self._hist_quali.copy()

    def _get_event_meta(self, year: int, race_name: str):
        return None


def test_prediction_feature_builder_returns_only_upcoming_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    # Make the feature pipeline deterministic and fast for unit tests.
    original_get = fe.config.get

    def fake_get(key: str, default=None):
        if key == "feature_engineering.pipeline_steps":
            return []
        if key == "feature_engineering.cache.enabled":
            return False
        return original_get(key, default)

    monkeypatch.setattr(fe.config, "get", fake_get)

    hist_races = pd.DataFrame(
        {
            "Year": [2025, 2025],
            "Race_Num": [1, 1],
            "Driver": ["AAA", "BBB"],
            "Team": ["T1", "T2"],
            "Circuit": ["X Grand Prix", "X Grand Prix"],
            "Date": [date(2025, 3, 1), date(2025, 3, 1)],
            "Grid": [1, 2],
            "Position": [1, 2],
            "Status": ["Finished", "Finished"],
        }
    )
    hist_quali = pd.DataFrame(
        {
            "Year": [2025, 2025],
            "Race_Num": [1, 1],
            "Driver": ["AAA", "BBB"],
            "Position": [2, 1],
            "Date": [date(2025, 3, 1), date(2025, 3, 1)],
        }
    )

    upcoming = pd.DataFrame(
        {
            "Year": [2025, 2025],
            "Race_Num": [2, 2],
            "Driver": ["AAA", "BBB"],
            "Team": ["T1", "T2"],
            "Circuit": ["Y Grand Prix", "Y Grand Prix"],
            "Date": [date(2025, 3, 15), date(2025, 3, 15)],
            # Supply predicted/actual qualifying for race inference.
            "Quali_Pos": [3.0, 1.0],
        }
    )

    builder = PredictionFeatureBuilder(data_loader=_FakeLoader(hist_races, hist_quali))
    out = builder.build(upcoming)

    assert len(out) == len(upcoming)
    assert "Position" not in out.columns
    # Quali_Pos should be carried through for the upcoming event.
    assert "Quali_Pos" in out.columns
    assert out["Quali_Pos"].tolist() == [3.0, 1.0]

