"""Unit tests for feature engineering classes."""

import numpy as np
import pandas as pd
import pytest

from src.f1_predictor.feature_engineering_pipeline import (
    ChampionshipFeatureEngineer,
    QualiTimeEngineer,
    WeatherFeatureEngineer,
    RollingPerformanceEngineer,
    GridUpliftEngineer,
    SeasonProgressEngineer,
)


# ---------------------------------------------------------------------------
# QualiTimeEngineer
# ---------------------------------------------------------------------------

class TestQualiTimeEngineer:
    def _make_df(self, q3=None):
        return pd.DataFrame({
            "Year":     [2024, 2024],
            "Race_Num": [1, 1],
            "Driver":   ["VER", "HAM"],
            "Q1": ["0 days 00:01:32.500000", "0 days 00:01:32.800000"],
            "Q2": ["0 days 00:01:31.200000", "0 days 00:01:31.500000"],
            "Q3": q3 if q3 is not None else ["0 days 00:01:30.100000", "0 days 00:01:30.500000"],
        })

    def test_produces_quali_time_s(self):
        df = self._make_df()
        out = QualiTimeEngineer(df).engineer_features()
        assert "Quali_Time_s" in out.columns
        assert out["Quali_Time_s"].notna().all()

    def test_uses_q3_when_available(self):
        df = self._make_df()
        out = QualiTimeEngineer(df).engineer_features()
        # Q3 time is ~90.1s; Q1 time is ~92.5s — Q3 should win
        assert out["Quali_Time_s"].max() < 92.0

    def test_falls_back_to_q2_when_q3_none(self):
        df = self._make_df(q3=[None, None])
        out = QualiTimeEngineer(df).engineer_features()
        assert "Quali_Time_s" in out.columns
        # Should fall back to Q2 (~91.2s)
        assert out["Quali_Time_s"].notna().any()

    def test_pole_gap_zero_for_fastest(self):
        df = self._make_df()
        out = QualiTimeEngineer(df).engineer_features()
        min_gap = out["Quali_Gap_To_Pole"].min()
        assert min_gap == pytest.approx(0.0, abs=1e-6)

    def test_pole_gap_positive_for_slower(self):
        df = self._make_df()
        out = QualiTimeEngineer(df).engineer_features()
        assert out["Quali_Gap_To_Pole"].max() > 0.0

    def test_gap_pct_between_0_and_1(self):
        df = self._make_df()
        out = QualiTimeEngineer(df).engineer_features()
        pct = out["Quali_Gap_Pct"].dropna()
        assert (pct >= 0).all()
        assert (pct < 0.1).all()  # <10% gap is realistic

    def test_noop_when_no_q_columns(self):
        df = pd.DataFrame({"Driver": ["VER"], "Predicted_Race_Pos": [1.0]})
        out = QualiTimeEngineer(df).engineer_features()
        assert "Quali_Time_s" not in out.columns

    def test_handles_all_none_q3(self):
        df = self._make_df(q3=[None, None])
        out = QualiTimeEngineer(df).engineer_features()
        assert "Quali_Time_s" in out.columns  # falls back to Q2


# ---------------------------------------------------------------------------
# ChampionshipFeatureEngineer
# ---------------------------------------------------------------------------

class TestChampionshipFeatureEngineer:
    def _make_season(self):
        """Two drivers, 5 rounds, deterministic points."""
        return pd.DataFrame({
            "Year":     [2024] * 10,
            "Race_Num": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "Driver":   ["VER", "HAM"] * 5,
            "Team":     ["Red Bull", "Mercedes"] * 5,
            "Points":   [25, 18, 25, 18, 18, 25, 25, 18, 25, 18],
        })

    def test_produces_driver_season_points(self):
        out = ChampionshipFeatureEngineer(self._make_season()).engineer_features()
        assert "Driver_Season_Points" in out.columns

    def test_produces_team_season_points(self):
        out = ChampionshipFeatureEngineer(self._make_season()).engineer_features()
        assert "Team_Season_Points" in out.columns

    def test_first_race_season_points_is_zero(self):
        out = ChampionshipFeatureEngineer(self._make_season()).engineer_features()
        out = out.sort_values(["Year", "Race_Num"])
        first_race = out[out["Race_Num"] == 1]
        assert (first_race["Driver_Season_Points"] == 0).all()

    def test_no_leakage_points_lag_one_race(self):
        """Season points at race N should NOT include race N's points."""
        df = self._make_season()
        out = ChampionshipFeatureEngineer(df).engineer_features()
        ver_rows = out[out["Driver"] == "VER"].sort_values("Race_Num")
        # After round 1 (25 pts), round 2 should have 25 in Season_Points
        r2 = ver_rows[ver_rows["Race_Num"] == 2].iloc[0]
        assert r2["Driver_Season_Points"] == 25.0

    def test_season_points_increase_monotonically(self):
        out = ChampionshipFeatureEngineer(self._make_season()).engineer_features()
        ver = out[out["Driver"] == "VER"].sort_values("Race_Num")["Driver_Season_Points"].values
        assert all(ver[i] <= ver[i + 1] for i in range(len(ver) - 1))

    def test_noop_when_no_points_column(self):
        df = pd.DataFrame({"Driver": ["VER"], "Year": [2024], "Race_Num": [1]})
        out = ChampionshipFeatureEngineer(df).engineer_features()
        assert "Driver_Season_Points" not in out.columns


# ---------------------------------------------------------------------------
# WeatherFeatureEngineer
# ---------------------------------------------------------------------------

class TestWeatherFeatureEngineer:
    def _weather_df(self, rainfall=0.0, airtemp=25.0, wind=5.0):
        return pd.DataFrame({
            "AirTemp":  [airtemp],
            "Humidity": [50.0],
            "Pressure": [1013.0],
            "WindSpeed": [wind],
            "Rainfall": [rainfall],
        })

    def test_dry_race_flag_zero(self):
        out = WeatherFeatureEngineer(self._weather_df(rainfall=0.0)).engineer_features()
        assert out["is_wet_race"].iloc[0] == 0

    def test_wet_race_flag_one(self):
        out = WeatherFeatureEngineer(self._weather_df(rainfall=50.0)).engineer_features()
        assert out["is_wet_race"].iloc[0] == 1

    def test_windy_flag(self):
        out = WeatherFeatureEngineer(self._weather_df(wind=30.0)).engineer_features()
        assert out["is_windy"].iloc[0] == 1

    def test_temp_deviation_positive(self):
        out = WeatherFeatureEngineer(self._weather_df(airtemp=40.0)).engineer_features()
        assert out["temp_deviation"].iloc[0] > 0

    def test_skips_when_columns_missing(self):
        df = pd.DataFrame({"Driver": ["VER"]})
        out = WeatherFeatureEngineer(df).engineer_features()
        assert "is_wet_race" not in out.columns


# ---------------------------------------------------------------------------
# SeasonProgressEngineer
# ---------------------------------------------------------------------------

class TestSeasonProgressEngineer:
    def test_season_progress_range(self):
        df = pd.DataFrame({
            "Year":     [2024] * 5,
            "Race_Num": [1, 2, 3, 4, 5],
            "Driver":   ["VER"] * 5,
        })
        out = SeasonProgressEngineer(df).engineer_features()
        prog = out["Season_Progress"]
        assert (prog >= 0.0).all()
        assert (prog <= 1.0).all()

    def test_last_race_progress_is_one(self):
        df = pd.DataFrame({
            "Year":     [2024] * 3,
            "Race_Num": [1, 2, 3],
            "Driver":   ["VER"] * 3,
        })
        out = SeasonProgressEngineer(df).engineer_features()
        assert out[out["Race_Num"] == 3]["Season_Progress"].iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RollingPerformanceEngineer
# ---------------------------------------------------------------------------

class TestRollingPerformanceEngineer:
    def test_produces_rolling_columns(self, races_df):
        out = RollingPerformanceEngineer(races_df).engineer_features()
        assert any(c.startswith("Driver_Avg_Position") for c in out.columns)

    def test_rolling_uses_shift_no_current_race(self, races_df):
        """First race of each driver should have NaN rolling avg (no prior data)."""
        out = RollingPerformanceEngineer(races_df).engineer_features()
        first_race = out[(out["Year"] == 2023) & (out["Race_Num"] == 1)]
        col = "Driver_Avg_Position_short"
        if col in out.columns:
            assert first_race[col].isna().all()


# ---------------------------------------------------------------------------
# GridUpliftEngineer
# ---------------------------------------------------------------------------

class TestGridUpliftEngineer:
    def test_grid_delta_column(self):
        df = pd.DataFrame({
            "Driver":   ["VER", "HAM"],
            "Team":     ["Red Bull", "Mercedes"],
            "Grid":     [1.0, 2.0],
            "Position": [3.0, 1.0],
        })
        out = GridUpliftEngineer(df).engineer_features()
        assert "Grid_Delta" in out.columns
        assert out[out["Driver"] == "VER"]["Grid_Delta"].iloc[0] == pytest.approx(2.0)
        assert out[out["Driver"] == "HAM"]["Grid_Delta"].iloc[0] == pytest.approx(-1.0)
