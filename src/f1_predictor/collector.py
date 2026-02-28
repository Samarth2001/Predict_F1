"""F1DataCollector: fetches race and qualifying data from FastF1 and persists to CSV."""

import fastf1
import hashlib
import json
import logging
import os
import time
from contextlib import suppress
from datetime import datetime
from typing import Any, Dict, Optional, Set

import numpy as np
import pandas as pd

from .config import config
from .utils import set_global_seeds

# Initialise FastF1 cache once at import time
try:
    fastf1.Cache.enable_cache(config.get("paths.cache_dir"))
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
    _cache_dir = config.get("paths.cache_dir")
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
    """Fetches race and qualifying results from FastF1 and appends them to CSVs.

    Supports incremental fetching: already-processed events are skipped unless
    ``force_refresh=True``.  An ingestion state JSON (with MD5 checksums) guards
    against partial writes.
    """

    def __init__(self):
        self.existing_events: Set[tuple] = set()
        self.ingestion_state_path: str = config.get("paths.ingestion_state")
        self._load_existing_data_for_resume()

    # ------------------------------------------------------------------
    # Resume / state helpers
    # ------------------------------------------------------------------

    def _load_existing_data_for_resume(self):
        races_csv_path = config.get("paths.races_csv")
        try:
            df = pd.read_csv(races_csv_path, usecols=["Year", "Race_Num"])
            for row in df.itertuples(index=False):
                self.existing_events.add((int(row.Year), int(row.Race_Num)))
            logger.info(f"Loaded {len(self.existing_events)} existing event records.")
        except FileNotFoundError:
            logger.info("No existing race data found. Starting fresh.")
        except Exception as e:
            logger.warning(f"Could not load existing races CSV: {e}")

        try:
            if self.ingestion_state_path and os.path.exists(self.ingestion_state_path):
                with open(self.ingestion_state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                processed = state.get("processed", [])
                try:
                    races_full = pd.read_csv(races_csv_path)
                except Exception:
                    races_full = pd.DataFrame()
                for item in processed:
                    y = int(item.get("Year"))
                    r = int(item.get("Race_Num"))
                    checksum = item.get("checksum")
                    ok = False
                    if not races_full.empty:
                        event_rows = races_full[
                            (races_full["Year"] == y) & (races_full["Race_Num"] == r)
                        ]
                        if not event_rows.empty:
                            subset = event_rows.sort_values(["Year", "Race_Num", "Driver"]).astype(str)
                            payload = subset[
                                [
                                    c
                                    for c in ["Year", "Race_Num", "Driver", "Team", "Position", "Grid"]
                                    if c in subset.columns
                                ]
                            ].to_csv(index=False)
                            hasher = hashlib.md5(payload.encode("utf-8")).hexdigest()
                            ok = hasher == checksum
                    if ok:
                        self.existing_events.add((y, r))
                    else:
                        self.existing_events.discard((y, r))
        except Exception as e:
            logger.warning(f"Failed to load ingestion state: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all_f1_data(
        self, start_year: int = None, end_year: int = None, force_refresh: bool = False
    ) -> bool:
        start_year = start_year or config.get("data_collection.start_year")
        end_year = end_year or config.get("data_collection.end_year")
        logger.info(
            f"Starting data collection from {start_year} to {end_year}. Force refresh: {force_refresh}"
        )

        if force_refresh:
            self.existing_events = set()
            for path in [config.get("paths.races_csv"), config.get("paths.quali_csv")]:
                if os.path.exists(path):
                    os.remove(path)
            if self.ingestion_state_path and os.path.exists(self.ingestion_state_path):
                try:
                    os.remove(self.ingestion_state_path)
                except Exception as e:
                    logger.warning(f"Could not remove ingestion state file: {e}")

        self._load_existing_data_for_resume()

        try:
            for year in range(start_year, end_year + 1):
                logger.info(f"Collecting data for {year} season...")
                self._fetch_season_data(year, force_refresh)
                time.sleep(1)
            logger.info("Data collection completed successfully.")
            return True
        except Exception as e:
            logger.error(f"Data collection failed: {e}", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_season_data(self, year: int, force_refresh: bool = False):
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        if schedule.empty:
            logger.warning(f"No schedule found for {year}.")
            return

        for _, event in schedule.sort_values("RoundNumber").iterrows():
            round_num = event["RoundNumber"]

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
                with suppress(Exception):
                    event_dt = event_dt.tz_localize(None)
            now_utc_naive = pd.Timestamp.utcnow().tz_localize(None)
            if (
                isinstance(event_dt, pd.Timestamp)
                and pd.notna(event_dt)
                and event_dt > now_utc_naive
            ):
                logger.info(
                    f"Stopping data collection for {year}: event '{event['EventName']}' is in the future."
                )
                break

            if round_num == 0 or (
                not force_refresh and (year, round_num) in self.existing_events
            ):
                continue

            logger.info(f"Processing {year} Round {round_num}: {event['EventName']}")
            race_df = self._fetch_session_with_retry(year, round_num, "R", event)
            if not race_df.empty:
                self._append_and_save(race_df, config.get("paths.races_csv"))
                self._update_ingestion_state(year, round_num, race_df)

            quali_df = self._fetch_session_with_retry(year, round_num, "Q", event)
            if not quali_df.empty:
                self._append_and_save(quali_df, config.get("paths.quali_csv"))
                self._update_ingestion_state(year, round_num, quali_df)

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
            except Exception as e:
                logger.error(
                    f"Failed to load {session_code} for {year} Round {round_num}: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
        return pd.DataFrame()

    def _extract_race_data(self, session: Any, event: pd.Series) -> pd.DataFrame:
        results = session.results
        weather = session.weather_data
        if results.empty:
            return pd.DataFrame()

        try:
            event_format = str(event.get("EventFormat", "")).lower()
            is_sprint = 1 if "sprint" in event_format else 0
        except Exception:
            is_sprint = 0

        avg_weather = weather.mean() if not weather.empty else {}
        race_data = []
        for _, result in results.iterrows():
            race_data.append(
                {
                    "Year": event["EventDate"].year,
                    "Race_Num": event["RoundNumber"],
                    "Circuit": str(event["EventName"]),
                    "Date": event["EventDate"].date(),
                    "Driver": result.get("Abbreviation", "Unknown"),
                    "Team": result.get("TeamName", "Unknown"),
                    "Grid": result.get("GridPosition", np.nan),
                    "Position": result.get("Position", np.nan),
                    "Status": result.get("Status", "Unknown"),
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
        return pd.DataFrame(race_data)

    def _extract_qualifying_data(self, session: Any, event: pd.Series) -> pd.DataFrame:
        results = session.results
        if results.empty:
            return pd.DataFrame()

        try:
            event_format = str(event.get("EventFormat", "")).lower()
            is_sprint = 1 if "sprint" in event_format else 0
        except Exception:
            is_sprint = 0

        quali_data = []
        for _, result in results.iterrows():
            quali_data.append(
                {
                    "Year": event["EventDate"].year,
                    "Race_Num": event["RoundNumber"],
                    "Circuit": str(event["EventName"]),
                    "Date": event["EventDate"].date(),
                    "Driver": result.get("Abbreviation", "Unknown"),
                    "Team": result.get("TeamName", "Unknown"),
                    "Position": result.get("Position", np.nan),
                    "Q1": result.get("Q1"),
                    "Q2": result.get("Q2"),
                    "Q3": result.get("Q3"),
                    "Is_Sprint": is_sprint,
                }
            )
        return pd.DataFrame(quali_data)

    def _append_and_save(self, new_data: pd.DataFrame, filepath: str):
        if new_data.empty:
            return

        try:
            existing = pd.read_csv(filepath)
        except FileNotFoundError:
            existing = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error reading existing data from {filepath}: {e}")
            existing = pd.DataFrame()

        required_cols = ["Year", "Race_Num", "Driver", "Team", "Circuit", "Date"]
        missing = [c for c in required_cols if c not in new_data.columns]
        if missing:
            logger.error(f"New data missing required columns: {missing}")
            return

        # Type coercions — log on failure instead of silently swallowing
        for col, dtype_fn in [
            ("Year", lambda s: pd.to_numeric(s, errors="coerce").astype("Int64")),
            ("Race_Num", lambda s: pd.to_numeric(s, errors="coerce").astype("Int64")),
            ("Date", lambda s: pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)),
            ("Driver", lambda s: s.astype(str)),
            ("Team", lambda s: s.astype(str)),
        ]:
            if col in new_data.columns:
                try:
                    new_data[col] = dtype_fn(new_data[col])
                except Exception as e:
                    logger.warning(f"Type coercion failed for column '{col}': {e}")

        new_data = self._apply_canonicalization(new_data)
        combined = pd.concat([existing, new_data])
        combined.drop_duplicates(subset=["Year", "Race_Num", "Driver"], keep="last", inplace=True)
        combined.to_csv(filepath, index=False)

    def _update_ingestion_state(self, year: int, round_num: int, df: pd.DataFrame) -> None:
        """Persist a durable record of processed events with MD5 checksums."""
        if not self.ingestion_state_path:
            return
        try:
            os.makedirs(os.path.dirname(self.ingestion_state_path), exist_ok=True)
            state: Dict[str, Any] = {"processed": []}
            if os.path.exists(self.ingestion_state_path):
                try:
                    with open(self.ingestion_state_path, "r", encoding="utf-8") as f:
                        state = json.load(f) or {"processed": []}
                except Exception as e:
                    logger.warning(f"Could not read existing ingestion state (resetting): {e}")
                    state = {"processed": []}

            subset = df.sort_values(["Year", "Race_Num", "Driver"]).astype(str)
            payload = subset[
                [c for c in subset.columns if c in ["Year", "Race_Num", "Driver", "Team", "Position", "Grid"]]
            ].to_csv(index=False)
            checksum = hashlib.md5(payload.encode("utf-8")).hexdigest()
            entry = {
                "Year": int(year),
                "Race_Num": int(round_num),
                "checksum": checksum,
                "rows": int(df.shape[0]),
            }
            others = [
                e
                for e in state.get("processed", [])
                if not (int(e.get("Year")) == year and int(e.get("Race_Num")) == round_num)
            ]
            state["processed"] = others + [entry]
            with open(self.ingestion_state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update ingestion state: {e}")

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
