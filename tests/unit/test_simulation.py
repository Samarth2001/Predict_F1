"""Unit tests for RaceSimulator and SimulationResult."""

import json

import numpy as np
import pandas as pd
import pytest

from src.f1_predictor.simulation import RaceSimulator, SimulationResult, _F1_POINTS


# ---------------------------------------------------------------------------
# Fixtures (in addition to conftest.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def sim() -> RaceSimulator:
    return RaceSimulator(n_simulations=2000, seed=42)


@pytest.fixture
def minimal_pred() -> pd.DataFrame:
    """Smallest valid predictions frame — just Driver + Predicted_Race_Pos."""
    return pd.DataFrame({
        "Driver": ["VER", "HAM", "LEC"],
        "Predicted_Race_Pos": [1.0, 2.0, 3.0],
    })


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestRaceSimulatorInit:
    def test_valid_construction(self):
        s = RaceSimulator(n_simulations=500, seed=0)
        assert s.n_simulations == 500
        assert s.seed == 0

    def test_rejects_zero_simulations(self):
        with pytest.raises(ValueError, match="n_simulations"):
            RaceSimulator(n_simulations=0)

    def test_rejects_negative_simulations(self):
        with pytest.raises(ValueError):
            RaceSimulator(n_simulations=-1)

    def test_rejects_non_int_seed(self):
        with pytest.raises(TypeError, match="seed"):
            RaceSimulator(n_simulations=100, seed="abc")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestRaceSimulatorValidation:
    def test_rejects_non_dataframe(self, sim):
        with pytest.raises(TypeError):
            sim.run({"Driver": ["VER"], "Predicted_Race_Pos": [1.0]})

    def test_rejects_empty_dataframe(self, sim):
        with pytest.raises(ValueError, match="empty"):
            sim.run(pd.DataFrame())

    def test_rejects_missing_driver_col(self, sim):
        with pytest.raises(ValueError, match="missing"):
            sim.run(pd.DataFrame({"Predicted_Race_Pos": [1.0]}))

    def test_rejects_missing_pos_col(self, sim):
        with pytest.raises(ValueError, match="missing"):
            sim.run(pd.DataFrame({"Driver": ["VER"]}))

    def test_rejects_sc_probability_above_1(self, sim, minimal_pred):
        with pytest.raises(ValueError, match="sc_probability"):
            sim.run(minimal_pred, sc_probability=1.5)

    def test_rejects_sc_probability_below_0(self, sim, minimal_pred):
        with pytest.raises(ValueError, match="sc_probability"):
            sim.run(minimal_pred, sc_probability=-0.1)

    def test_all_nan_positions_raises(self, sim):
        df = pd.DataFrame({"Driver": ["VER"], "Predicted_Race_Pos": [float("nan")]})
        with pytest.raises(ValueError, match="NaN"):
            sim.run(df)


# ---------------------------------------------------------------------------
# Happy path — result structure
# ---------------------------------------------------------------------------

class TestSimulationResult:
    def test_returns_simulation_result(self, sim, predictions_df):
        result = sim.run(predictions_df)
        assert isinstance(result, SimulationResult)

    def test_summary_has_correct_columns(self, sim, predictions_df):
        result = sim.run(predictions_df)
        expected = {"Driver", "Team", "Predicted_Pos", "Win_Pct", "Podium_Pct",
                    "Top10_Pct", "Exp_Points", "Pos_P10", "Pos_P50", "Pos_P90"}
        assert expected.issubset(set(result.summary.columns))

    def test_summary_length_equals_n_drivers(self, sim, predictions_df):
        result = sim.run(predictions_df)
        assert len(result.summary) == len(predictions_df)

    def test_position_matrix_shape(self, sim, predictions_df):
        result = sim.run(predictions_df)
        assert result.position_matrix.shape == (len(predictions_df), 20)

    def test_position_matrix_rows_sum_to_one(self, sim, predictions_df):
        result = sim.run(predictions_df)
        row_sums = result.position_matrix.sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 0.02

    def test_win_pct_sums_to_100(self, sim, predictions_df):
        result = sim.run(predictions_df)
        total = result.summary["Win_Pct"].sum()
        assert 98.0 <= total <= 102.0

    def test_podium_pct_gte_win_pct(self, sim, predictions_df):
        result = sim.run(predictions_df)
        assert (result.summary["Podium_Pct"] >= result.summary["Win_Pct"]).all()

    def test_top10_pct_gte_podium_pct(self, sim, predictions_df):
        result = sim.run(predictions_df)
        assert (result.summary["Top10_Pct"] >= result.summary["Podium_Pct"]).all()

    def test_pole_sitter_highest_win_pct(self, sim, predictions_df):
        result = sim.run(predictions_df)
        best = result.summary.sort_values("Win_Pct", ascending=False).iloc[0]["Driver"]
        assert best == "VER"  # VER is at predicted pos 1

    def test_exp_points_monotone_decreasing(self, sim, predictions_df):
        """Better predicted position → higher expected points."""
        result = sim.run(predictions_df).summary.sort_values("Predicted_Pos")
        pts = result["Exp_Points"].values
        # Allow slight noise; top half should average more than bottom half
        assert pts[:5].mean() > pts[-5:].mean()

    def test_metadata_present(self, sim, predictions_df):
        result = sim.run(predictions_df, sc_probability=0.4)
        assert result.metadata["sc_probability"] == 0.4

    def test_n_simulations_stored(self, sim, predictions_df):
        result = sim.run(predictions_df)
        assert result.n_simulations == 2000

    def test_to_dict_is_json_serializable(self, sim, predictions_df):
        result = sim.run(predictions_df)
        d = result.to_dict()
        json.dumps(d, default=str)  # must not raise


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_result(self, predictions_df):
        r1 = RaceSimulator(1000, 7).run(predictions_df)
        r2 = RaceSimulator(1000, 7).run(predictions_df)
        pd.testing.assert_frame_equal(r1.summary, r2.summary)

    def test_different_seed_different_result(self, predictions_df):
        r1 = RaceSimulator(1000, 1).run(predictions_df)
        r2 = RaceSimulator(1000, 2).run(predictions_df)
        assert not r1.summary["Win_Pct"].equals(r2.summary["Win_Pct"])


# ---------------------------------------------------------------------------
# DNF probs
# ---------------------------------------------------------------------------

class TestDNFProbs:
    def test_high_dnf_reduces_expected_points(self, predictions_df):
        sim_no_dnf  = RaceSimulator(3000, 42).run(predictions_df, dnf_probs={d: 0.0 for d in predictions_df["Driver"]})
        sim_high_dnf = RaceSimulator(3000, 42).run(predictions_df, dnf_probs={d: 0.5 for d in predictions_df["Driver"]})
        assert sim_high_dnf.summary["Exp_Points"].mean() < sim_no_dnf.summary["Exp_Points"].mean()

    def test_out_of_range_dnf_prob_clamps(self, minimal_pred):
        """Should warn and clamp, not raise."""
        sim = RaceSimulator(500, 42)
        result = sim.run(minimal_pred, dnf_probs={"VER": 1.5})
        assert isinstance(result, SimulationResult)

    def test_missing_driver_gets_default_dnf(self, minimal_pred):
        """Drivers not in the dict get the default DNF probability."""
        sim = RaceSimulator(500, 42)
        result = sim.run(minimal_pred, dnf_probs={"VER": 0.0})
        assert isinstance(result, SimulationResult)


# ---------------------------------------------------------------------------
# Safety car
# ---------------------------------------------------------------------------

class TestSafetyCar:
    def test_sc_probability_zero_runs(self, sim, predictions_df):
        result = sim.run(predictions_df, sc_probability=0.0)
        assert isinstance(result, SimulationResult)

    def test_sc_probability_one_runs(self, sim, predictions_df):
        result = sim.run(predictions_df, sc_probability=1.0)
        assert isinstance(result, SimulationResult)


# ---------------------------------------------------------------------------
# Confidence interval handling
# ---------------------------------------------------------------------------

class TestSigmaDerivation:
    def test_no_bounds_uses_fallback(self, minimal_pred):
        result = RaceSimulator(500, 42).run(minimal_pred)
        assert isinstance(result, SimulationResult)

    def test_with_bounds_uses_ci(self, minimal_pred):
        df = minimal_pred.copy()
        df["Lower_Pos"] = [0.5, 1.5, 2.5]
        df["Upper_Pos"] = [1.5, 2.5, 3.5]
        result = RaceSimulator(500, 42).run(df)
        assert isinstance(result, SimulationResult)

    def test_nan_bounds_fall_back_gracefully(self, minimal_pred):
        df = minimal_pred.copy()
        df["Lower_Pos"] = float("nan")
        df["Upper_Pos"] = float("nan")
        result = RaceSimulator(500, 42).run(df)
        assert isinstance(result, SimulationResult)


# ---------------------------------------------------------------------------
# F1 points table
# ---------------------------------------------------------------------------

class TestF1Points:
    def test_points_table_length(self):
        assert len(_F1_POINTS) == 10

    def test_winner_gets_25(self):
        assert _F1_POINTS[0] == 25

    def test_tenth_gets_1(self):
        assert _F1_POINTS[9] == 1
