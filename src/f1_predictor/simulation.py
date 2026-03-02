"""RaceSimulator: Monte Carlo race-outcome simulation.

Takes model predictions (mean position + uncertainty bounds) and returns
win/podium/points probability distributions across N simulated races.

Design constraints
------------------
- No silent failures: validation errors raise; recoverable issues log.warning
- Fully vectorised NumPy hot-path (no Python loops per simulation)
- Deterministic with a fixed seed
- Zero external dependencies beyond numpy/pandas
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Official F1 points for positions 1-10; 0 for 11+
_F1_POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)

# Fallback DNF probability when not supplied by the predictor
_DEFAULT_DNF_PROB = 0.08

# DNF finishing position (penalised to back of grid)
_DNF_POS = 20


@dataclass
class SimulationResult:
    """Output of a single RaceSimulator.run() call.

    Attributes
    ----------
    summary : DataFrame
        One row per driver.  Columns:
          Driver, Team, Win_Pct, Podium_Pct, Top10_Pct,
          Exp_Points, Pos_P10, Pos_P50, Pos_P90
    position_matrix : DataFrame
        Shape (n_drivers, 20).  Cell [i, p] = fraction of simulations
        in which driver i finished in position p.
    n_simulations : int
    seed : int
    """

    summary: pd.DataFrame
    position_matrix: pd.DataFrame
    n_simulations: int
    seed: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary.to_dict(orient="records"),
            "n_simulations": self.n_simulations,
            "seed": self.seed,
            "metadata": self.metadata,
        }


class RaceSimulator:
    """Monte Carlo simulator for F1 race finishing positions.

    Parameters
    ----------
    n_simulations : int
        Number of races to simulate (default 2 000).
    seed : int
        NumPy random seed for reproducibility (default 42).

    Usage
    -----
    >>> sim = RaceSimulator(n_simulations=2000, seed=42)
    >>> result = sim.run(predictions_df)
    >>> print(result.summary)
    """

    # Required columns in the predictions DataFrame
    _REQUIRED_COLS = {"Driver", "Predicted_Race_Pos"}

    def __init__(self, n_simulations: int = 2000, seed: int = 42) -> None:
        if n_simulations < 1:
            raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")
        if not isinstance(seed, int):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")
        self.n_simulations = n_simulations
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        predictions: pd.DataFrame,
        dnf_probs: Optional[dict[str, float]] = None,
        sc_probability: float = 0.3,
    ) -> SimulationResult:
        """Simulate race outcomes from model predictions.

        Parameters
        ----------
        predictions : DataFrame
            Output of F1Predictor.predict_race().  Must contain at least
            ``Driver`` and ``Predicted_Race_Pos``.  Optional columns used:
            ``Team``, ``Lower_Pos``, ``Upper_Pos``.
        dnf_probs : dict[str, float] | None
            Per-driver DNF probability (0–1).  Defaults to
            ``_DEFAULT_DNF_PROB`` for missing drivers.
        sc_probability : float
            Probability of a safety car deployment per race (0–1).

        Returns
        -------
        SimulationResult
        """
        self._validate_inputs(predictions, sc_probability)
        predictions = predictions.reset_index(drop=True).copy()
        n_drivers = len(predictions)
        rng = np.random.default_rng(self.seed)

        # ---- per-driver parameters ------------------------------------
        mu = predictions["Predicted_Race_Pos"].astype(float).to_numpy()
        sigma = self._derive_sigma(predictions, mu)
        p_dnf = self._build_dnf_probs(predictions, dnf_probs, n_drivers)

        # ---- Monte Carlo ----------------------------------------------
        # Shape: (n_simulations, n_drivers)
        raw = rng.normal(loc=mu, scale=sigma, size=(self.n_simulations, n_drivers))
        raw = np.clip(raw, 1.0, float(_DNF_POS))

        # Safety car: on SC laps the field compresses — reduce variance
        sc_mask = rng.random(self.n_simulations) < sc_probability
        if sc_mask.any():
            compression = rng.uniform(0.4, 0.7, size=(sc_mask.sum(), 1))
            raw[sc_mask] = mu + compression * (raw[sc_mask] - mu)

        # DNF: driver does not finish in some simulations
        dnf_mask = rng.random((self.n_simulations, n_drivers)) < p_dnf
        raw[dnf_mask] = _DNF_POS + rng.uniform(0, 1, size=dnf_mask.sum())

        # Resolve ties → actual finishing positions 1..n_drivers
        # argsort of argsort converts raw scores to dense integer ranks
        ranks = np.argsort(np.argsort(raw, axis=1), axis=1) + 1  # 1-based

        # ---- aggregate ------------------------------------------------
        summary = self._aggregate(predictions, ranks)
        pos_matrix = self._position_matrix(predictions, ranks)

        meta = {
            "sc_probability": sc_probability,
            "default_dnf_prob": _DEFAULT_DNF_PROB,
        }
        return SimulationResult(
            summary=summary,
            position_matrix=pos_matrix,
            n_simulations=self.n_simulations,
            seed=self.seed,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(predictions: pd.DataFrame, sc_probability: float) -> None:
        if not isinstance(predictions, pd.DataFrame):
            raise TypeError("predictions must be a pandas DataFrame")
        if predictions.empty:
            raise ValueError("predictions DataFrame is empty")
        missing = RaceSimulator._REQUIRED_COLS - set(predictions.columns)
        if missing:
            raise ValueError(f"predictions is missing required columns: {sorted(missing)}")
        if not (0.0 <= sc_probability <= 1.0):
            raise ValueError(f"sc_probability must be in [0, 1], got {sc_probability}")
        if predictions["Predicted_Race_Pos"].isna().all():
            raise ValueError("All Predicted_Race_Pos values are NaN")

    @staticmethod
    def _derive_sigma(predictions: pd.DataFrame, mu: np.ndarray) -> np.ndarray:
        """Derive per-driver σ from confidence interval when available."""
        if "Lower_Pos" in predictions.columns and "Upper_Pos" in predictions.columns:
            lo = predictions["Lower_Pos"].astype(float).to_numpy()
            hi = predictions["Upper_Pos"].astype(float).to_numpy()
            # 80 % CI  →  z = 1.28
            sigma = np.where(
                np.isfinite(lo) & np.isfinite(hi),
                (hi - lo) / 2.56,
                2.0,  # fallback
            )
        else:
            sigma = np.full(len(mu), 2.0)
        return np.clip(sigma, 0.5, 5.0)

    @staticmethod
    def _build_dnf_probs(
        predictions: pd.DataFrame,
        dnf_probs: Optional[dict],
        n_drivers: int,
    ) -> np.ndarray:
        p = np.full(n_drivers, _DEFAULT_DNF_PROB)
        if dnf_probs:
            for i, driver in enumerate(predictions["Driver"]):
                if driver in dnf_probs:
                    val = float(dnf_probs[driver])
                    if not (0.0 <= val <= 1.0):
                        logger.warning(
                            f"DNF prob for {driver} ({val}) out of range [0,1]; clamping."
                        )
                        val = np.clip(val, 0.0, 1.0)
                    p[i] = val
        return p

    @staticmethod
    def _f1_points(positions: np.ndarray) -> np.ndarray:
        """Convert finishing positions (1-based) to points. Shape: same as input."""
        pts = np.where(positions <= 10, _F1_POINTS[np.clip(positions - 1, 0, 9).astype(int)], 0.0)
        return pts.astype(float)

    def _aggregate(self, predictions: pd.DataFrame, ranks: np.ndarray) -> pd.DataFrame:
        """Compute per-driver summary statistics across simulations."""
        n = len(predictions)
        N = float(self.n_simulations)

        win_pct    = (ranks == 1).sum(axis=0) / N * 100
        podium_pct = (ranks <= 3).sum(axis=0) / N * 100
        top10_pct  = (ranks <= 10).sum(axis=0) / N * 100
        exp_pts    = self._f1_points(ranks).mean(axis=0)
        p10 = np.percentile(ranks, 10, axis=0)
        p50 = np.percentile(ranks, 50, axis=0)
        p90 = np.percentile(ranks, 90, axis=0)

        summary = pd.DataFrame(
            {
                "Driver":     predictions["Driver"].values,
                "Team":       predictions.get("Team", pd.Series([""] * n)).values,
                "Predicted_Pos": predictions["Predicted_Race_Pos"].values,
                "Win_Pct":    np.round(win_pct, 2),
                "Podium_Pct": np.round(podium_pct, 2),
                "Top10_Pct":  np.round(top10_pct, 2),
                "Exp_Points": np.round(exp_pts, 2),
                "Pos_P10":    np.round(p10, 1),
                "Pos_P50":    np.round(p50, 1),
                "Pos_P90":    np.round(p90, 1),
            }
        )
        return summary.sort_values("Predicted_Pos").reset_index(drop=True)

    def _position_matrix(self, predictions: pd.DataFrame, ranks: np.ndarray) -> pd.DataFrame:
        """Return a (n_drivers × 20) probability matrix."""
        n_drivers = len(predictions)
        matrix = np.zeros((n_drivers, 20), dtype=float)
        for pos in range(1, 21):
            matrix[:, pos - 1] = (ranks == pos).sum(axis=0) / self.n_simulations
        return pd.DataFrame(
            matrix,
            index=predictions["Driver"].values,
            columns=[f"P{p}" for p in range(1, 21)],
        )
