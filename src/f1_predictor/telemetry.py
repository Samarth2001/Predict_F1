"""TelemetryFetcher: pull lap-level telemetry from FastF1 (uses FF1 cache, no DB)."""

import logging
from typing import Optional

import fastf1
import pandas as pd

from .config import config

logger = logging.getLogger(__name__)


class TelemetryFetcher:
    """Fetch telemetry for a single session from the FastF1 cache.

    This is read-only and stateless — results are not persisted to the DB
    because telemetry data is large and only needed on demand.

    Usage:
        fetcher = TelemetryFetcher()
        lap = fetcher.get_fastest_lap(2024, 1, "VER")
        sectors = fetcher.get_sector_times(2024, 1)
    """

    def __init__(self):
        try:
            fastf1.Cache.enable_cache(config.get("paths.cache_dir", "data/cache/ff1_cache"))
        except Exception as exc:
            logger.warning(f"FastF1 cache init failed: {exc}")

    def _load_session(self, year: int, race_num: int, session_code: str = "R") -> Optional[fastf1.core.Session]:
        try:
            session = fastf1.get_session(year, race_num, session_code)
            session.load(laps=True, telemetry=True, weather=False, messages=False)
            return session
        except Exception as exc:
            logger.warning(f"Could not load session {session_code} {year} R{race_num}: {exc}")
            return None

    def get_fastest_lap(
        self, year: int, race_num: int, driver: str, session_code: str = "R"
    ) -> Optional[pd.DataFrame]:
        """Return telemetry DataFrame for a driver's fastest lap, or None."""
        session = self._load_session(year, race_num, session_code)
        if session is None:
            return None
        try:
            lap = session.laps.pick_driver(driver).pick_fastest()
            return lap.get_telemetry()
        except Exception as exc:
            logger.warning(f"No fastest lap telemetry for {driver} {year} R{race_num}: {exc}")
            return None

    def get_sector_times(
        self, year: int, race_num: int, session_code: str = "Q"
    ) -> pd.DataFrame:
        """Return a DataFrame with Sector1Time / Sector2Time / Sector3Time per driver."""
        session = self._load_session(year, race_num, session_code)
        if session is None:
            return pd.DataFrame()
        try:
            cols = ["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "Compound"]
            available = [c for c in cols if c in session.laps.columns]
            best = (
                session.laps[available]
                .dropna(subset=["LapTime"])
                .sort_values("LapTime")
                .groupby("Driver", as_index=False)
                .first()
            )
            return best
        except Exception as exc:
            logger.warning(f"Could not extract sector times {year} R{race_num}: {exc}")
            return pd.DataFrame()

    def get_lap_pace(
        self, year: int, race_num: int, driver: str, session_code: str = "R"
    ) -> pd.DataFrame:
        """Return per-lap times for a driver (useful for race pace analysis)."""
        session = self._load_session(year, race_num, session_code)
        if session is None:
            return pd.DataFrame()
        try:
            laps = session.laps.pick_driver(driver)[["LapNumber", "LapTime", "Compound", "TyreLife"]].copy()
            laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()
            return laps.reset_index(drop=True)
        except Exception as exc:
            logger.warning(f"Could not extract lap pace for {driver} {year} R{race_num}: {exc}")
            return pd.DataFrame()
