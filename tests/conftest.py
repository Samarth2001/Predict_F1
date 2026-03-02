"""Shared pytest fixtures for the F1 predictor test suite."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Drivers / teams used across tests
# ---------------------------------------------------------------------------
DRIVERS = [
    "VER", "PER", "HAM", "RUS", "LEC", "SAI",
    "NOR", "PIA", "ALO", "STR", "ALB", "SAR",
    "OCO", "GAS", "HUL", "MAG", "TSU", "RIC",
    "ZHO", "BOT",
]
TEAMS = (
    ["Red Bull"] * 2 + ["Mercedes"] * 2 + ["Ferrari"] * 2
    + ["McLaren"] * 2 + ["Aston"] * 2 + ["Williams"] * 2
    + ["Alpine"] * 2 + ["Haas"] * 2 + ["AT"] * 2 + ["Sauber"] * 2
)
N = len(DRIVERS)


# ---------------------------------------------------------------------------
# Synthetic race results fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def races_df() -> pd.DataFrame:
    """40 rows: 2 drivers x 20 rounds across 2 seasons."""
    rng = np.random.default_rng(0)
    n_rows = 40
    return pd.DataFrame(
        {
            "Year":     [2023] * 20 + [2024] * 20,
            "Race_Num": list(range(1, 21)) * 2,
            "Circuit":  ["Bahrain Grand Prix"] * n_rows,
            "Date":     pd.date_range("2023-03-05", periods=n_rows, freq="7D"),
            "Driver":   ["VER", "HAM"] * 20,
            "Team":     ["Red Bull", "Mercedes"] * 20,
            "Grid":     rng.integers(1, 20, n_rows).astype(float),
            "Position": rng.integers(1, 20, n_rows).astype(float),
            "Status":   ["Finished"] * n_rows,
            "Laps":     [57] * n_rows,
            "Points":   rng.choice([25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0], n_rows).astype(float),
            "AirTemp":  [28.0] * n_rows,
            "Humidity": [50.0] * n_rows,
            "Pressure": [1013.0] * n_rows,
            "WindSpeed": [5.0] * n_rows,
            "Rainfall": [0.0] * n_rows,
            "Is_Sprint": [0] * n_rows,
        }
    )


@pytest.fixture
def quali_df() -> pd.DataFrame:
    """40-row qualifying results with Q1/Q2/Q3 lap time strings."""
    rng = np.random.default_rng(1)
    n_rows = 40
    return pd.DataFrame(
        {
            "Year":     [2023] * 20 + [2024] * 20,
            "Race_Num": list(range(1, 21)) * 2,
            "Circuit":  ["Bahrain Grand Prix"] * n_rows,
            "Date":     pd.date_range("2023-03-04", periods=n_rows, freq="7D"),
            "Driver":   ["VER", "HAM"] * 20,
            "Team":     ["Red Bull", "Mercedes"] * 20,
            "Position": rng.integers(1, 20, n_rows).astype(float),
            "Q1":       ["0 days 00:01:32.500000"] * n_rows,
            "Q2":       ["0 days 00:01:31.200000"] * n_rows,
            "Q3":       ["0 days 00:01:30.100000", None] * 20,
            "Is_Sprint": [0] * n_rows,
        }
    )


# ---------------------------------------------------------------------------
# Full 20-driver predictions fixture (for simulation tests)
# ---------------------------------------------------------------------------
@pytest.fixture
def predictions_df() -> pd.DataFrame:
    """Standard F1Predictor output DataFrame with 20 drivers."""
    return pd.DataFrame(
        {
            "Driver":             DRIVERS,
            "Team":               TEAMS,
            "Predicted_Race_Pos": np.arange(1, N + 1, dtype=float),
            "Lower_Pos":          np.arange(1, N + 1, dtype=float) - 2.0,
            "Upper_Pos":          np.arange(1, N + 1, dtype=float) + 2.0,
        }
    )


# ---------------------------------------------------------------------------
# Temporary directory fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_db(tmp_path) -> str:
    """Path to a throwaway DuckDB file."""
    return str(tmp_path / "test.duckdb")


@pytest.fixture
def tmp_dir(tmp_path) -> str:
    return str(tmp_path)
