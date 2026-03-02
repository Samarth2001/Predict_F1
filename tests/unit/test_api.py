"""Unit tests for the FastAPI layer (no trained models required).

Uses FastAPI's TestClient so the tests run entirely in-process without
needing a live server or real model artefacts.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pred_df(session: str = "race") -> pd.DataFrame:
    drivers = ["VER", "HAM", "LEC", "NOR", "ALO"]
    if session == "qualifying":
        return pd.DataFrame(
            {
                "Driver": drivers,
                "Team": ["Red Bull", "Mercedes", "Ferrari", "McLaren", "Aston"],
                "Predicted_Quali_Pos": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
    return pd.DataFrame(
        {
            "Driver": drivers,
            "Team": ["Red Bull", "Mercedes", "Ferrari", "McLaren", "Aston"],
            "Predicted_Race_Pos": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def _make_sim_result() -> MagicMock:
    from src.f1_predictor.simulation import SimulationResult  # noqa: PLC0415

    summary = pd.DataFrame(
        {
            "Driver": ["VER", "HAM"],
            "Team": ["Red Bull", "Mercedes"],
            "Win_Pct": [0.7, 0.3],
            "Podium_Pct": [0.9, 0.6],
            "Top10_Pct": [1.0, 1.0],
            "Exp_Points": [22.0, 15.0],
        }
    )
    pos_matrix = pd.DataFrame(
        [[0.7, 0.3], [0.3, 0.7]], index=["VER", "HAM"], columns=[1, 2]
    )
    return SimulationResult(
        summary=summary,
        position_matrix=pos_matrix,
        n_simulations=200,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """TestClient with F1Predictor mocked out."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")

    from src.f1_predictor import api as api_module  # noqa: PLC0415

    # Reset the cached singleton so mock is used fresh
    api_module._predictor = None

    mock_predictor = MagicMock()
    mock_predictor.predict_qualifying.return_value = _make_pred_df("qualifying")
    mock_predictor.predict_race.return_value = _make_pred_df("race")
    mock_predictor.simulate.return_value = _make_sim_result()

    with patch.object(api_module, "_get_predictor", return_value=mock_predictor):
        yield TestClient(api_module.app)

    # Clean up singleton after test
    api_module._predictor = None


# ---------------------------------------------------------------------------
# Tests — /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "timestamp" in body
        assert "version" in body


# ---------------------------------------------------------------------------
# Tests — /schedule/{year}
# ---------------------------------------------------------------------------

class TestSchedule:
    def test_invalid_year_low(self, client):
        resp = client.get("/schedule/2010")
        assert resp.status_code == 400

    def test_invalid_year_high(self, client):
        resp = client.get("/schedule/2050")
        assert resp.status_code == 400

    def test_schedule_fastf1_error_raises_502(self):
        """If FastF1 raises inside the endpoint, the server returns 500."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi not installed")

        from src.f1_predictor import api as api_module  # noqa: PLC0415

        # raise_server_exceptions=False so TestClient returns a 500 response
        # rather than re-raising the exception in the test process.
        no_raise_client = TestClient(api_module.app, raise_server_exceptions=False)
        with patch.object(api_module, "_get_schedule", side_effect=Exception("network")):
            resp = no_raise_client.get("/schedule/2024")
        assert resp.status_code in (500, 502)


# ---------------------------------------------------------------------------
# Tests — /predict/qualifying
# ---------------------------------------------------------------------------

class TestPredictQualifying:
    def test_success(self, client):
        payload = {"year": 2024, "race": "Italian Grand Prix"}
        resp = client.post("/predict/qualifying", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["session"] == "qualifying"
        assert body["year"] == 2024
        assert body["race"] == "Italian Grand Prix"
        assert len(body["predictions"]) == 5
        assert "Driver" in body["predictions"][0]

    def test_no_predictions_returns_422(self, client):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi not installed")

        from src.f1_predictor import api as api_module  # noqa: PLC0415

        empty_mock = MagicMock()
        empty_mock.predict_qualifying.return_value = None
        with patch.object(api_module, "_get_predictor", return_value=empty_mock):
            resp = client.post(
                "/predict/qualifying", json={"year": 2024, "race": "Italian Grand Prix"}
            )
        assert resp.status_code == 422

    def test_missing_year_returns_422(self, client):
        resp = client.post("/predict/qualifying", json={"race": "Italian Grand Prix"})
        assert resp.status_code == 422

    def test_missing_race_returns_422(self, client):
        resp = client.post("/predict/qualifying", json={"year": 2024})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Tests — /predict/race
# ---------------------------------------------------------------------------

class TestPredictRace:
    def test_success(self, client):
        payload = {"year": 2024, "race": "Italian Grand Prix", "mode": "auto"}
        resp = client.post("/predict/race", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["session"] == "race"
        assert len(body["predictions"]) == 5

    def test_default_mode(self, client):
        payload = {"year": 2024, "race": "Italian Grand Prix"}
        resp = client.post("/predict/race", json=payload)
        assert resp.status_code == 200

    def test_no_predictions_returns_422(self, client):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi not installed")

        from src.f1_predictor import api as api_module  # noqa: PLC0415

        empty_mock = MagicMock()
        empty_mock.predict_race.return_value = pd.DataFrame()
        with patch.object(api_module, "_get_predictor", return_value=empty_mock):
            resp = client.post(
                "/predict/race", json={"year": 2024, "race": "Italian Grand Prix"}
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Tests — /simulate
# ---------------------------------------------------------------------------

class TestSimulate:
    def test_success(self, client):
        payload = {"year": 2024, "race": "Italian Grand Prix", "n_simulations": 200}
        resp = client.post("/simulate", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_simulations"] == 200
        assert "summary" in body
        assert "position_matrix" in body
        assert len(body["summary"]) == 2

    def test_no_result_returns_422(self, client):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi not installed")

        from src.f1_predictor import api as api_module  # noqa: PLC0415

        none_mock = MagicMock()
        none_mock.simulate.return_value = None
        with patch.object(api_module, "_get_predictor", return_value=none_mock):
            resp = client.post(
                "/simulate", json={"year": 2024, "race": "Italian Grand Prix"}
            )
        assert resp.status_code == 422

    def test_n_simulations_too_low(self, client):
        payload = {"year": 2024, "race": "Italian Grand Prix", "n_simulations": 50}
        resp = client.post("/simulate", json=payload)
        assert resp.status_code == 422

    def test_sc_probability_out_of_range(self, client):
        payload = {
            "year": 2024,
            "race": "Italian Grand Prix",
            "sc_probability": 1.5,
        }
        resp = client.post("/simulate", json=payload)
        assert resp.status_code == 422

    def test_openapi_json_served(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["info"]["title"] == "F1 Prediction API"
