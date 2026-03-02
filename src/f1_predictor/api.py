"""FastAPI REST layer for the F1 prediction system.

Start with:
    uvicorn src.f1_predictor.api:app --reload

Endpoints
---------
GET  /health                  → liveness check
GET  /schedule/{year}         → race schedule for a season
POST /predict/qualifying      → qualifying-order predictions
POST /predict/race            → race-order predictions
POST /simulate                → Monte Carlo race simulation
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="F1 Prediction API",
    description="REST interface for the F1 race and qualifying prediction system.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    year: int = Field(..., ge=2018, le=2030, description="Season year")
    race: str = Field(..., description="Official FastF1 EventName, e.g. 'Italian Grand Prix'")
    mode: Optional[str] = Field(
        "auto",
        description="Prediction mode: auto | pre_weekend | pre_quali | post_quali | live",
    )


class SimulateRequest(BaseModel):
    year: int = Field(..., ge=2018, le=2030)
    race: str = Field(..., description="Official FastF1 EventName")
    mode: Optional[str] = Field("auto")
    n_simulations: int = Field(2000, ge=100, le=20000)
    sc_probability: float = Field(0.3, ge=0.0, le=1.0)
    seed: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_predictor: Any = None


def _get_predictor() -> Any:
    """Lazily initialise F1Predictor (expensive due to model loading)."""
    global _predictor
    if _predictor is None:
        from .prediction import F1Predictor  # noqa: PLC0415

        _predictor = F1Predictor()
    return _predictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df_to_records(df: Any) -> List[Dict]:
    """Convert a DataFrame to a list of dicts, handling NaN/NaT safely."""
    import math

    records = df.to_dict(orient="records")
    clean = []
    for row in records:
        clean.append(
            {
                k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in row.items()
            }
        )
    return clean


def _get_schedule(year: int) -> List[Dict]:
    try:
        import fastf1  # noqa: PLC0415

        sched = fastf1.get_event_schedule(year, include_testing=False)
        sched = sched.sort_values("RoundNumber")
        cols = ["RoundNumber", "EventName", "EventDate", "Country", "Location"]
        existing = [c for c in cols if c in sched.columns]
        out = sched[existing].copy()
        out["EventDate"] = out["EventDate"].astype(str)
        return out.to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"FastF1 schedule error: {exc}") from exc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health() -> HealthResponse:
    """Liveness / readiness probe."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version=app.version,
    )


@app.get("/schedule/{year}", tags=["Schedule"])
def schedule(year: int) -> JSONResponse:
    """Return the official race schedule for *year*.

    Uses FastF1's event schedule; results are **not** cached across requests.
    """
    if year < 2018 or year > 2030:
        raise HTTPException(status_code=400, detail="year must be between 2018 and 2030")
    data = _get_schedule(year)
    return JSONResponse({"year": year, "rounds": len(data), "schedule": data})


@app.post("/predict/qualifying", tags=["Predictions"])
def predict_qualifying(req: PredictRequest) -> JSONResponse:
    """Predict qualifying order for the requested event.

    Returns a ranked list of drivers with predicted qualifying positions.
    Requires trained qualifying model artifacts.
    """
    predictor = _get_predictor()
    try:
        result = predictor.predict_qualifying(req.year, req.race, scenario="qualifying")
    except Exception as exc:
        logger.exception("predict_qualifying failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if result is None or result.empty:
        raise HTTPException(
            status_code=422,
            detail=(
                "No predictions returned — check that models are trained and "
                "the race name matches the FastF1 EventName exactly."
            ),
        )
    return JSONResponse(
        {
            "year": req.year,
            "race": req.race,
            "session": "qualifying",
            "predictions": _df_to_records(result),
        }
    )


@app.post("/predict/race", tags=["Predictions"])
def predict_race(req: PredictRequest) -> JSONResponse:
    """Predict race finishing order for the requested event.

    Returns a ranked list of drivers with predicted race positions.
    Requires trained race model artifacts.
    """
    predictor = _get_predictor()
    try:
        result = predictor.predict_race(req.year, req.race, mode=req.mode or "auto")
    except Exception as exc:
        logger.exception("predict_race failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if result is None or result.empty:
        raise HTTPException(
            status_code=422,
            detail=(
                "No predictions returned — check that models are trained and "
                "the race name matches the FastF1 EventName exactly."
            ),
        )
    return JSONResponse(
        {
            "year": req.year,
            "race": req.race,
            "session": "race",
            "mode": req.mode,
            "predictions": _df_to_records(result),
        }
    )


@app.post("/simulate", tags=["Simulation"])
def simulate(req: SimulateRequest) -> JSONResponse:
    """Run a Monte Carlo race simulation.

    Runs *n_simulations* stochastic races and returns per-driver win / podium /
    top-10 probabilities and expected points, plus the full position matrix.
    """
    predictor = _get_predictor()
    try:
        result = predictor.simulate(
            req.year,
            req.race,
            mode=req.mode or "auto",
            n_simulations=req.n_simulations,
            sc_probability=req.sc_probability,
            seed=req.seed,
        )
    except Exception as exc:
        logger.exception("simulate failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if result is None:
        raise HTTPException(
            status_code=422,
            detail="Simulation returned no results — verify models are trained and event name is valid.",
        )

    return JSONResponse(
        {
            "year": req.year,
            "race": req.race,
            "n_simulations": result.n_simulations,
            "seed": result.seed,
            "summary": _df_to_records(result.summary),
            "position_matrix": result.position_matrix.to_dict(orient="split"),
        }
    )
