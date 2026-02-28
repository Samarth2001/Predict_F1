"""F1DataCollector: fetches race and qualifying data from FastF1 and persists via F1Store."""

import logging
import os
import time
from typing import Any, Dict, Optional, Set

import fastf1
import numpy as np
import pandas as pd

from .config import config
from .store import F1Store
from .utils import set_global_seeds

# Initialise FastF1 cache once at import time
try:
    fastf1.Cache.enable_cache(config.get("paths.cache_dir", "data/cache/ff1_cache"))
except Exception as exc:
    logging.getLogger(__name__).warning(f"FastF1 cache init failed: {exc}")

try:
    seed_global = int(config.get("general.random_seed", config.get("general.random_state", 42)))
    set_global_seeds(seed_global)
except Exception:
    pass

# Optional HTTP-level caching for OpenWeather requests
try:
    import requests_cache as _requests_cache

    _ttl_hours = int(config.get("data_collection.external_apis.openweather_cache_ttl_hours", 12))
    _cache_dir = config.get("paths.cache_dir", "data/cache/ff1_cache")
    os.makedirs(_cache_dir, exist_ok=True)
    _requests_cache.install_cache(
        cache_name=os.path.join(_cache_dir, "http_cache"),
        backend="sqlite",
        expire_after=_ttl_hours * 3600,
    )
except ImportError:
    pass
except Exception as exc:
    logging.getLogger(__name__).warning(f"requests_cache setup failed: {exc}")

logger = logging.getLogger(__name__)


class F1DataCollector:
    """Fetches race and qualifying results from FastF1 and stores them via F1Store.

    Supports incremental fetching: already-processed events are skipped unless
    ``force_refresh=True``.  The ingestion log in the DB guards against partial writes.
    """

    def __init__(self, store: Optional[F1Store] = None):
        self._store = store  # injected or opened per-run

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all_f1_data(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        force_refresh: bool = False,
    ) -> bool:
        start_year = start_year or config.get("data_collection.start_year")
        end_year = end_year or config.get("data_collection.end_year")
        logger.info(f"Starting data collection {start_year}–{end_year}. Force refresh: {force_refresh}")

        with F1Store() as store:
            if force_refresh:
                logger.info("Force refresh: clearing existing data.")
                store.con.execute("DELETE FROM races")
                store.con.execute("DELETE FROM qualifying")
                store.con.execute("DELETE FROM ingestion_log")

            existing_events: Set[tuple] = store.processed_events()
            logger.info(f"Resuming — {len(existing_events)} events already processed.")

            try:
                for year in range(start_year, end_year + 1):
                    logger.info(f"Collecting data for {year} season…")
                    self._fetch_season_data(year, store, existing_events, force_refresh)
                    time.sleep(1)
                logger.info("Data collection completed successfully.")
                return True
            except Exception as exc:
                logger.error(f"Data collection failed: {exc}", exc_info=True)
                return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_season_data(
        self,
        year: int,
        store: F1Store,
        existing_events: Set[tuple],
        force_refresh: bool,
    ) -> None:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        if schedule.empty:
            logger.warning(f"No schedule found for {year}.")
            return

        now_utc_naive = pd.Timestamp.utcnow().tz_localize(None)

        for _, event in schedule.sort_values("RoundNumber").iterrows():
            round_num = int(event["RoundNumber"])

            # Skip future events
            event_dt = pd.to_datetime(event["EventDate"], errors="coerce")
            if pd.notna(event_dt):
                try:
                    event_dt = event_dt.tz_convert("UTC")
                except Exception:
                    try:
                        event_dt = (
                            event_dt.tz_localize("UTC")
                            if event_dt.tzinfo is None
                            else event_dt.tz_convert("UTC")
                        )
                    except Exception:
                        pass
                try:
                    event_dt = event_dt.tz_localize(None)
                except Exception:
                    pass

            if (
                isinstance(event_dt, pd.Timestamp)
                and pd.notna(event_dt)
                and event_dt > now_utc_naive
            ):
                logger.info(
                    f"Stopping {year}: '{event['EventName']}' is in the future."
                )
                break

            if round_num == 0 or (not force_refresh and (year, round_num) in existing_events):
                continue

            logger.info(f"Processing {year} R{round_num}: {event['EventName']}")

            race_df = self._fetch_session_with_retry(year, round_num, "R", event)
            if not race_df.empty:
                race_df = self._apply_canonicalization(race_df)
                store.upsert_races(race_df)
                store.mark_processed(year, round_num, "R", race_df)
                existing_events.add((year, round_num))

            quali_df = self._fetch_session_with_retry(year, round_num, "Q", event)
            if not quali_df.empty:
                quali_df = self._apply_canonicalization(quali_df)
                store.upsert_qualifying(quali_df)
                store.mark_processed(year, round_num, "Q", quali_df)

            time.sleep(config.get("data_collection.fastf1.fetch_delay", 1.5))

    def _fetch_session_with_retry(
        self, year: int, round_num: int, session_code: str, event: pd.Series
    ) -> pd.DataFrame:
        max_retries = config.get("data_collection.fastf1.api_max_retries", 3)
        base_delay = config.get("data_collection.fastf1.api_retry_delay", 60)
        for attempt in range(max_retries):
            try:
                session = fastf1.get_session(year, round_num, session_code)
                session.load(laps=True, telemetry=False, weather=True, messages=False)
                if session_code == "R":
                    return self._extract_race_data(session, event)
                elif session_code == "Q":
                    return self._extract_qualifying_data(session, event)
            except Exception as exc:
                logger.error(f"Failed to load {session_code} {year} R{round_num}: {exc}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2**attempt))
        return pd.DataFrame()

    def _extract_race_data(self, session: Any, event: pd.Series) -> pd.DataFrame:
        results = session.results
        if results.empty:
            return pd.DataFrame()

        try:
            is_sprint = 1 if "sprint" in str(event.get("EventFormat", "")).lower() else 0
        except Exception:
            is_sprint = 0

        weather = session.weather_data
        avg_weather: Dict = weather.mean(numeric_only=True).to_dict() if not weather.empty else {}

        rows = []
        for _, result in results.iterrows():
            rows.append(
                {
                    "Year": int(event["EventDate"].year),
                    "Race_Num": int(event["RoundNumber"]),
                    "Circuit": str(event["EventName"]),
                    "Date": event["EventDate"].date(),
                    "Driver": str(result.get("Abbreviation", "Unknown")),
                    "Team": str(result.get("TeamName", "Unknown")),
                    "Grid": result.get("GridPosition", np.nan),
                    "Position": result.get("Position", np.nan),
                    "Status": str(result.get("Status", "Unknown")),
                    "Laps": result.get("Laps", np.nan),
                    "Points": result.get("Points", np.nan),
                    "AirTemp": avg_weather.get("AirTemp", np.nan),
                    "Humidity": avg_weather.get("Humidity", np.nan),
                    "Pressure": avg_weather.get("Pressure", np.nan),
                    "WindSpeed": avg_weather.get("WindSpeed", np.nan),
                    "Rainfall": avg_weather.get("Rainfall", np.nan),
                    "Is_Sprint": is_sprint,
                }
            )
        return pd.DataFrame(rows)

    def _extract_qualifying_data(self, session: Any, event: pd.Series) -> pd.DataFrame:
        results = session.results
        if results.empty:
            return pd.DataFrame()

        try:
            is_sprint = 1 if "sprint" in str(event.get("EventFormat", "")).lower() else 0
        except Exception:
            is_sprint = 0

        rows = []
        for _, result in results.iterrows():
            rows.append(
                {
                    "Year": int(event["EventDate"].year),
                    "Race_Num": int(event["RoundNumber"]),
                    "Circuit": str(event["EventName"]),
                    "Date": event["EventDate"].date(),
                    "Driver": str(result.get("Abbreviation", "Unknown")),
                    "Team": str(result.get("TeamName", "Unknown")),
                    "Position": result.get("Position", np.nan),
                    "Q1": result.get("Q1"),
                    "Q2": result.get("Q2"),
                    "Q3": result.get("Q3"),
                    "Is_Sprint": is_sprint,
                }
            )
        return pd.DataFrame(rows)

    def _apply_canonicalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply driver/team name mappings from config to normalise across seasons."""
        drivers_map: Dict[str, str] = config.get("feature_engineering.canonicalization.drivers", {}) or {}
        teams_map: Dict[str, str] = config.get("feature_engineering.canonicalization.teams", {}) or {}
        out = df.copy()
        if "Driver" in out.columns and drivers_map:
            out["Driver"] = out["Driver"].astype(str).map(lambda x: drivers_map.get(x, x))
        if "Team" in out.columns and teams_map:
            out["Team"] = out["Team"].astype(str).map(lambda x: teams_map.get(x, x))
        return out
