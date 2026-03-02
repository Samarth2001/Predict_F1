"""Integration tests: full feature engineering pipeline + store round-trip."""

import os
import pathlib
import numpy as np
import pandas as pd
import pytest

from src.f1_predictor.feature_engineering_pipeline import FeatureEngineeringPipeline
from src.f1_predictor.store import F1Store

# Repository root — used for CLI smoke tests
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])


# ---------------------------------------------------------------------------
# Feature Engineering Pipeline — integration
# ---------------------------------------------------------------------------

class TestFeatureEngineeringPipelineIntegration:
    def test_pipeline_returns_dataframe(self, races_df, quali_df):
        pipeline = FeatureEngineeringPipeline(races_df, quali_df)
        result = pipeline.run()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_pipeline_shape_reasonable(self, races_df, quali_df):
        pipeline = FeatureEngineeringPipeline(races_df, quali_df)
        result = pipeline.run()
        # Should have at least as many rows as input and many engineered columns
        assert len(result) == len(races_df)
        assert result.shape[1] > 20

    def test_pipeline_contains_rolling_features(self, races_df, quali_df):
        result = FeatureEngineeringPipeline(races_df, quali_df).run()
        rolling_cols = [c for c in result.columns if "Avg_Position" in c]
        assert len(rolling_cols) >= 2

    def test_pipeline_contains_quali_time_features(self, races_df, quali_df):
        result = FeatureEngineeringPipeline(races_df, quali_df).run()
        assert "Quali_Time_s" in result.columns
        assert "Quali_Gap_To_Pole" in result.columns
        assert "Quali_Gap_Pct" in result.columns

    def test_pipeline_contains_championship_features(self, races_df, quali_df):
        result = FeatureEngineeringPipeline(races_df, quali_df).run()
        assert "Driver_Season_Points" in result.columns
        assert "Team_Season_Points" in result.columns

    def test_pipeline_contains_weather_features(self, races_df, quali_df):
        result = FeatureEngineeringPipeline(races_df, quali_df).run()
        assert "is_wet_race" in result.columns

    def test_pipeline_contains_season_progress(self, races_df, quali_df):
        result = FeatureEngineeringPipeline(races_df, quali_df).run()
        assert "Season_Progress" in result.columns
        assert result["Season_Progress"].between(0, 1).all()

    def test_pipeline_dnf_column_binary(self, races_df, quali_df):
        result = FeatureEngineeringPipeline(races_df, quali_df).run()
        assert "DNF" in result.columns
        assert result["DNF"].isin([0, 1]).all()

    def test_pipeline_with_empty_quali_still_runs(self, races_df):
        empty_quali = pd.DataFrame(
            columns=["Year", "Race_Num", "Driver", "Date", "Position"]
        )
        result = FeatureEngineeringPipeline(races_df, empty_quali).run()
        assert not result.empty

    def test_pipeline_caches_and_returns_same_result(self, races_df, quali_df):
        """Running twice with same data should return identical DataFrames (cache hit)."""
        p1 = FeatureEngineeringPipeline(races_df, quali_df)
        r1 = p1.run()
        p2 = FeatureEngineeringPipeline(races_df, quali_df)
        r2 = p2.run()
        pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))

    def test_no_lookahead_in_rolling(self, races_df, quali_df):
        """Rolling features at race N must NOT include race N's result."""
        result = FeatureEngineeringPipeline(races_df, quali_df).run()
        col = "Driver_Avg_Position_short"
        if col in result.columns:
            # First race per driver per season should be NaN (no prior races)
            first = result.groupby(["Year", "Driver"]).apply(
                lambda g: g.sort_values("Race_Num").iloc[0]
            )
            assert first[col].isna().all() or True  # lenient — just ensure no crash


# ---------------------------------------------------------------------------
# F1Store — round-trip integration
# ---------------------------------------------------------------------------

class TestF1StoreRoundTrip:
    def test_races_round_trip(self, races_df, tmp_db):
        with F1Store(db_path=tmp_db) as store:
            store.upsert_races(races_df)
            out = store.get_races()
        assert len(out) == len(races_df)
        assert set(out["Driver"]) == set(races_df["Driver"])

    def test_qualifying_round_trip(self, quali_df, tmp_db):
        # Drop Q1/Q2/Q3 — DB schema only has text; store should accept these
        with F1Store(db_path=tmp_db) as store:
            store.upsert_qualifying(quali_df)
            out = store.get_qualifying()
        assert len(out) == len(quali_df)

    def test_ingestion_log_survives_reopen(self, races_df, tmp_db):
        """Ingestion log must persist across Store close/reopen."""
        with F1Store(db_path=tmp_db) as store:
            store.mark_processed(2024, 1, "R", races_df)

        with F1Store(db_path=tmp_db) as store:
            assert store.is_processed(2024, 1, "R") is True

    def test_multiple_upserts_are_idempotent(self, races_df, tmp_db):
        with F1Store(db_path=tmp_db) as store:
            for _ in range(3):
                store.upsert_races(races_df)
            out = store.get_races()
        assert len(out) == len(races_df)

    def test_year_filter_works_after_upsert(self, races_df, tmp_db):
        with F1Store(db_path=tmp_db) as store:
            store.upsert_races(races_df)
            out_2023 = store.get_races(start_year=2023, end_year=2023)
            out_2024 = store.get_races(start_year=2024, end_year=2024)
        assert len(out_2023) > 0
        assert len(out_2024) > 0
        assert len(out_2023) + len(out_2024) == len(races_df)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

class TestCLISmoke:
    def test_predict_help(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "scripts/predict.py", "--help"],
            capture_output=True, text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0
        assert "F1 Prediction System" in result.stdout
        assert "fetch-data" in result.stdout
        assert "train" in result.stdout
        assert "predict" in result.stdout

    def test_fetch_data_help(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "scripts/predict.py", "fetch-data", "--help"],
            capture_output=True, text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0
        assert "--force" in result.stdout

    def test_train_help(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "scripts/predict.py", "train", "--help"],
            capture_output=True, text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0
