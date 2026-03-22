from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_predictor.prediction import F1Predictor


class _DummyModel:
    def __init__(self, scores: list[float]):
        self._scores = np.asarray(scores, dtype=float)

    def predict(self, X) -> np.ndarray:  # noqa: N803 (sklearn-style X)
        n = len(X)
        return self._scores[:n]


class _CapturingFeatureBuilder:
    def __init__(self, features_df: pd.DataFrame):
        self.features_df = features_df
        self.last_input_df: pd.DataFrame | None = None

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        self.last_input_df = df.copy()
        return self.features_df.copy()


def test_predict_qualifying_ranking_outputs_predicted_quali_pos(monkeypatch: pytest.MonkeyPatch) -> None:
    upcoming = pd.DataFrame(
        {
            "Driver": ["AAA", "BBB"],
            "Team": ["T1", "T2"],
            "Year": [2025, 2025],
            "Race_Num": [1, 1],
        }
    )
    feats = pd.DataFrame({"f1": [1.0, 2.0], "f2": [0.0, 0.0]})
    builder = _CapturingFeatureBuilder(feats)

    p = F1Predictor()
    p.qualifying_model = _DummyModel([0.2, 0.9])  # higher is better -> BBB should rank 1
    p.pred_features = builder

    monkeypatch.setattr(p, "_get_upcoming_race_df", lambda y, r: upcoming.copy())
    monkeypatch.setattr(p, "_prepare_features", lambda df: builder.build(df))
    monkeypatch.setattr(p, "_load_and_enforce_metadata", lambda *_: list(feats.columns))
    monkeypatch.setattr(p, "_save_prediction_results", lambda *_args, **_kwargs: None)

    res = p.predict_qualifying(2025, "Some Grand Prix")
    assert res is not None and not res.empty
    assert "Predicted_Quali_Pos" in res.columns
    assert res.iloc[0]["Driver"] == "BBB"
    assert int(res.iloc[0]["Predicted_Quali_Pos"]) == 1


def test_predict_race_post_quali_requires_actual_quali(monkeypatch: pytest.MonkeyPatch) -> None:
    upcoming = pd.DataFrame(
        {
            "Driver": ["AAA", "BBB"],
            "Driver_ID": ["drv:aaa", "drv:bbb"],
            "Team": ["T1", "T2"],
            "Year": [2025, 2025],
            "Race_Num": [1, 1],
        }
    )

    feats = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    builder = _CapturingFeatureBuilder(feats)

    p = F1Predictor()
    p.race_model = _DummyModel([0.1, 0.2])
    p.pred_features = builder

    # No actual qualifying rows for this event -> should fail in post_quali.
    hist_races = pd.DataFrame()
    hist_quali = pd.DataFrame({"Year": [2024], "Race_Num": [99], "Position": [1]})
    monkeypatch.setattr(p, "_get_upcoming_race_df", lambda y, r: upcoming.copy())
    monkeypatch.setattr(p.data_loader, "load_all_data", lambda: (hist_races, hist_quali))

    res = p.predict_race(2025, "Some Grand Prix", mode="post_quali")
    assert res is None


def test_predict_race_post_quali_merges_quali_and_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    upcoming = pd.DataFrame(
        {
            "Driver": ["AAA", "BBB"],
            "Driver_ID": ["drv:aaa", "drv:bbb"],
            "Team": ["T1", "T2"],
            "Year": [2025, 2025],
            "Race_Num": [1, 1],
        }
    )

    hist_quali = pd.DataFrame(
        {
            "Year": [2025, 2025],
            "Race_Num": [1, 1],
            "Driver": ["AAA", "BBB"],
            "Driver_ID": ["drv:aaa", "drv:bbb"],
            "Position": [2, 1],
        }
    )
    hist_races = pd.DataFrame(
        {
            "Year": [2025, 2025],
            "Race_Num": [1, 1],
            "Driver": ["AAA", "BBB"],
            "Driver_ID": ["drv:aaa", "drv:bbb"],
            "Grid": [2, 1],
        }
    )

    feats = pd.DataFrame({"f1": [1.0, 2.0]})
    builder = _CapturingFeatureBuilder(feats)

    p = F1Predictor()
    p.race_model = _DummyModel([0.2, 0.9])  # BBB predicted P1
    p.pred_features = builder

    monkeypatch.setattr(p, "_get_upcoming_race_df", lambda y, r: upcoming.copy())
    monkeypatch.setattr(p.data_loader, "load_all_data", lambda: (hist_races, hist_quali))
    monkeypatch.setattr(p, "_prepare_features", lambda df: builder.build(df))
    monkeypatch.setattr(p, "_load_and_enforce_metadata", lambda *_: list(feats.columns))
    monkeypatch.setattr(p, "_save_prediction_results", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(p, "_persist_quali_used_for_race", lambda *_args, **_kwargs: None)

    res = p.predict_race(2025, "Some Grand Prix", mode="post_quali")
    assert res is not None and not res.empty
    assert "Predicted_Race_Pos" in res.columns
    assert res.iloc[0]["Driver"] == "BBB"
    assert int(res.iloc[0]["Predicted_Race_Pos"]) == 1

    # Verify the upcoming DF handed to feature builder had quali/grid merged in.
    assert builder.last_input_df is not None
    assert "Quali_Pos" in builder.last_input_df.columns
    assert "Grid" in builder.last_input_df.columns
    assert int(builder.last_input_df["Post_Quali"].iloc[0]) == 1

