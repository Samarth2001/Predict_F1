"""Unit tests for F1Store (DuckDB wrapper)."""

import json

import pandas as pd
import pytest

from src.f1_predictor.store import F1Store, _df_checksum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _race_row(**overrides) -> dict:
    base = dict(
        Year=2024, Race_Num=1, Circuit="Bahrain Grand Prix", Date="2024-03-02",
        Driver="VER", Team="Red Bull", Grid=1.0, Position=1.0,
        Status="Finished", Laps=57.0, Points=25.0,
        AirTemp=28.0, Humidity=50.0, Pressure=1013.0,
        WindSpeed=5.0, Rainfall=0.0, Is_Sprint=0,
    )
    base.update(overrides)
    return base


def _quali_row(**overrides) -> dict:
    base = dict(
        Year=2024, Race_Num=1, Circuit="Bahrain Grand Prix", Date="2024-03-01",
        Driver="VER", Team="Red Bull", Position=1.0,
        Q1="0 days 00:01:32.5", Q2="0 days 00:01:31.2", Q3="0 days 00:01:30.1",
        Is_Sprint=0,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestF1StoreContext:
    def test_open_close(self, tmp_db):
        store = F1Store(db_path=tmp_db)
        store.open()
        assert store._con is not None
        store.close()
        assert store._con is None

    def test_context_manager(self, tmp_db):
        with F1Store(db_path=tmp_db) as store:
            assert store._con is not None
        assert store._con is None

    def test_property_raises_when_closed(self, tmp_db):
        store = F1Store(db_path=tmp_db)
        with pytest.raises(RuntimeError, match="not open"):
            _ = store.con


class TestF1StoreRaces:
    def test_upsert_and_get_races(self, tmp_db):
        df = pd.DataFrame([_race_row()])
        with F1Store(db_path=tmp_db) as store:
            store.upsert_races(df)
            out = store.get_races()
        assert len(out) == 1
        assert out.iloc[0]["Driver"] == "VER"

    def test_upsert_idempotent(self, tmp_db):
        df = pd.DataFrame([_race_row()])
        with F1Store(db_path=tmp_db) as store:
            store.upsert_races(df)
            store.upsert_races(df)  # second upsert should not duplicate
            out = store.get_races()
        assert len(out) == 1

    def test_upsert_overwrites_same_pk(self, tmp_db):
        df1 = pd.DataFrame([_race_row(Position=1.0)])
        df2 = pd.DataFrame([_race_row(Position=3.0)])
        with F1Store(db_path=tmp_db) as store:
            store.upsert_races(df1)
            store.upsert_races(df2)
            out = store.get_races()
        assert len(out) == 1
        assert float(out.iloc[0]["Position"]) == 3.0

    def test_upsert_empty_df_is_noop(self, tmp_db):
        with F1Store(db_path=tmp_db) as store:
            store.upsert_races(pd.DataFrame())
            out = store.get_races()
        assert out.empty

    def test_year_filter(self, tmp_db):
        rows = [_race_row(Year=y, Driver="VER") for y in [2022, 2023, 2024]]
        with F1Store(db_path=tmp_db) as store:
            store.upsert_races(pd.DataFrame(rows))
            out = store.get_races(start_year=2023, end_year=2023)
        assert list(out["Year"].unique()) == [2023]

    def test_multiple_drivers(self, tmp_db):
        rows = [
            _race_row(Driver="VER", Position=1.0),
            _race_row(Driver="HAM", Team="Mercedes", Position=2.0),
        ]
        with F1Store(db_path=tmp_db) as store:
            store.upsert_races(pd.DataFrame(rows))
            out = store.get_races()
        assert len(out) == 2
        assert set(out["Driver"]) == {"VER", "HAM"}


class TestF1StoreQualifying:
    def test_upsert_and_get_qualifying(self, tmp_db):
        df = pd.DataFrame([_quali_row()])
        with F1Store(db_path=tmp_db) as store:
            store.upsert_qualifying(df)
            out = store.get_qualifying()
        assert len(out) == 1
        assert out.iloc[0]["Driver"] == "VER"

    def test_upsert_qualifying_idempotent(self, tmp_db):
        df = pd.DataFrame([_quali_row()])
        with F1Store(db_path=tmp_db) as store:
            store.upsert_qualifying(df)
            store.upsert_qualifying(df)
            out = store.get_qualifying()
        assert len(out) == 1


class TestF1StoreWeatherCache:
    def test_set_and_get_weather(self, tmp_db):
        payload = json.dumps({"AirTemp": 28.0, "Humidity": 50.0})
        with F1Store(db_path=tmp_db) as store:
            store.upsert_weather_cache("Bahrain", payload, ttl_hours=12)
            result = store.get_weather_cache("Bahrain")
        assert result == payload

    def test_missing_circuit_returns_none(self, tmp_db):
        with F1Store(db_path=tmp_db) as store:
            result = store.get_weather_cache("NonExistent")
        assert result is None

    def test_weather_overwrite(self, tmp_db):
        with F1Store(db_path=tmp_db) as store:
            store.upsert_weather_cache("Bahrain", '{"AirTemp": 20}', ttl_hours=12)
            store.upsert_weather_cache("Bahrain", '{"AirTemp": 35}', ttl_hours=12)
            result = store.get_weather_cache("Bahrain")
        assert json.loads(result)["AirTemp"] == 35


class TestF1StoreIngestionLog:
    def test_is_processed_false_before_mark(self, tmp_db):
        with F1Store(db_path=tmp_db) as store:
            assert store.is_processed(2024, 1, "R") is False

    def test_mark_and_is_processed(self, tmp_db):
        df = pd.DataFrame([_race_row()])
        with F1Store(db_path=tmp_db) as store:
            store.mark_processed(2024, 1, "R", df)
            assert store.is_processed(2024, 1, "R") is True

    def test_processed_events_returns_set(self, tmp_db):
        df = pd.DataFrame([_race_row()])
        with F1Store(db_path=tmp_db) as store:
            store.mark_processed(2024, 1, "R", df)
            store.mark_processed(2024, 2, "R", df)
            evts = store.processed_events()
        assert (2024, 1) in evts
        assert (2024, 2) in evts

    def test_mark_processed_different_sessions(self, tmp_db):
        df = pd.DataFrame([_race_row()])
        with F1Store(db_path=tmp_db) as store:
            store.mark_processed(2024, 1, "R", df)
            store.mark_processed(2024, 1, "Q", df)
            assert store.is_processed(2024, 1, "R") is True
            assert store.is_processed(2024, 1, "Q") is True
            # only "R" sessions appear in processed_events()
            assert (2024, 1) in store.processed_events()

    def test_checksum_consistency(self, tmp_db):
        df = pd.DataFrame([_race_row()])
        c1 = _df_checksum(df)
        c2 = _df_checksum(df)
        assert c1 == c2

    def test_checksum_changes_on_data_change(self, tmp_db):
        df1 = pd.DataFrame([_race_row(Position=1.0)])
        df2 = pd.DataFrame([_race_row(Position=5.0)])
        assert _df_checksum(df1) != _df_checksum(df2)
