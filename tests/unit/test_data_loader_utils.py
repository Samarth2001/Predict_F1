from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

import pandas as pd
import pytest

import f1_predictor.data_loader as dl


def test_normalize_event_dates_can_return_naive() -> None:
    loader = dl.F1DataLoader()
    df = pd.DataFrame({"EventDate": ["2025-03-01", "2025-03-02T10:00:00Z"]})

    s_naive = loader._normalize_event_dates(df, "EventDate", return_naive=True)
    assert pd.api.types.is_datetime64_any_dtype(s_naive)
    # tz-naive in pandas shows as dtype datetime64[ns]
    assert getattr(s_naive.dtype, "tz", None) is None


def test_weather_forecast_cache_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Use a temp weather cache file.
    weather_csv = tmp_path / "weather.csv"
    original_get = dl.config.get

    def fake_get(key: str, default=None):
        if key == "paths.weather_csv":
            return str(weather_csv)
        return original_get(key, default)

    monkeypatch.setattr(dl.config, "get", fake_get)

    loader = dl.F1DataLoader()
    loader._save_cached_forecast(
        {"Year": 2025, "Race_Num": 1, "AirTemp": 25.0, "Humidity": 50.0, "FetchedAt": "2025-01-01T00:00:00"}
    )
    cached = loader._load_cached_forecast()
    assert not cached.empty
    assert int(cached.iloc[0]["Year"]) == 2025
    assert int(cached.iloc[0]["Race_Num"]) == 1


def test_fetch_openweathermap_forecast_parses_hourly(monkeypatch: pytest.MonkeyPatch) -> None:
    original_get = dl.config.get

    def fake_get(key: str, default=None):
        if key == "data_collection.external_apis.openweathermap_api_key":
            return "test_key"
        return original_get(key, default)

    monkeypatch.setattr(dl.config, "get", fake_get)

    @dataclass
    class _Resp:
        status_code: int = 200

        def json(self):
            return {
                "hourly": [
                    {"temp": 20, "humidity": 40, "pressure": 1000, "wind_speed": 3, "pop": 0.1},
                    {"temp": 22, "humidity": 50, "pressure": 1005, "wind_speed": 5, "pop": 0.3},
                ]
            }

    monkeypatch.setattr(dl.requests, "get", lambda *_args, **_kwargs: _Resp())

    loader = dl.F1DataLoader()
    out = loader._fetch_openweathermap_forecast(lat=25.0, lon=55.0, hours_ahead=2)
    assert out is not None
    assert set(out.keys()) == {"AirTemp", "Humidity", "Pressure", "WindSpeed", "Rainfall"}
    assert out["AirTemp"] > 0

