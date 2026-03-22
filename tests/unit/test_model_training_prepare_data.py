from __future__ import annotations

import pandas as pd

from f1_predictor.model_training import F1ModelTrainer


def test_prepare_training_data_drops_leaky_race_outcomes_for_race_model() -> None:
    trainer = F1ModelTrainer()
    df = pd.DataFrame(
        {
            "Year": [2024, 2024, 2024],
            "Race_Num": [1, 1, 1],
            "Position": [1.0, 2.0, 3.0],
            "Quali_Pos": [1.0, 2.0, 3.0],
            "Grid": [1.0, 2.0, 3.0],
            "DNF": [0, 0, 0],
            "Points": [25.0, 18.0, 15.0],
            "Laps": [57.0, 57.0, 57.0],
            "Feature_X": [0.1, 0.2, 0.3],
        }
    )

    X, y, feature_names, _ = trainer._prepare_training_data(df, target_column="Position")

    assert y.tolist() == [1.0, 2.0, 3.0]
    assert "Position" not in X.columns
    assert "DNF" not in X.columns
    assert "Points" not in X.columns
    assert "Laps" not in X.columns

    # Quali/grid are allowed inputs for race predictions (post-quali or imputed/predicted).
    assert "Quali_Pos" in X.columns
    assert "Grid" in X.columns
    assert "Feature_X" in X.columns
    assert feature_names == list(X.columns)


def test_prepare_training_data_drops_grid_and_outcomes_for_qualifying_model() -> None:
    trainer = F1ModelTrainer()
    df = pd.DataFrame(
        {
            "Year": [2024, 2024, 2024],
            "Race_Num": [1, 1, 1],
            "Quali_Pos": [1.0, 2.0, 3.0],
            "Position": [1.0, 2.0, 3.0],
            "Grid": [1.0, 2.0, 3.0],
            "DNF": [0, 0, 0],
            "Points": [25.0, 18.0, 15.0],
            "Laps": [57.0, 57.0, 57.0],
            "Feature_X": [0.1, 0.2, 0.3],
        }
    )

    X, y, _, _ = trainer._prepare_training_data(df, target_column="Quali_Pos")

    assert y.tolist() == [1.0, 2.0, 3.0]
    assert "Quali_Pos" not in X.columns
    assert "Position" not in X.columns
    assert "Grid" not in X.columns
    assert "DNF" not in X.columns
    assert "Points" not in X.columns
    assert "Laps" not in X.columns
    assert "Feature_X" in X.columns

