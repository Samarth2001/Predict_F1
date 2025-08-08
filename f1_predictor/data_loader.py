# f1_predictor/data_loader.py

import fastf1
import pandas as pd
import numpy as np
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict, Any
import warnings

warnings.filterwarnings("ignore")

from .config import config

fastf1.Cache.enable_cache(config.get("paths.cache_dir"))
logging.basicConfig(level=config.get("general.log_level", "INFO"))
logger = logging.getLogger(__name__)


class F1DataCollector:
    def __init__(self):
        self.existing_events = set()
        self._load_existing_data_for_resume()

    def _load_existing_data_for_resume(self):
        races_csv_path = config.get("paths.races_csv")
        if os.path.exists(races_csv_path):
            try:
                df = pd.read_csv(
                    races_csv_path, usecols=["Year", "Race_Num"], on_bad_lines="skip"
                )
                for row in df.itertuples(index=False):
                    self.existing_events.add((int(row.Year), int(row.Race_Num)))
                logger.info(
                    f"Loaded {len(self.existing_events)} existing event records."
                )
            except Exception as e:
                logger.warning(f"Could not load existing races CSV: {e}")

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

    def _fetch_season_data(self, year: int, force_refresh: bool = False):
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        if schedule.empty:
            logger.warning(f"No schedule found for {year}.")
            return

        for _, event in schedule.sort_values("RoundNumber").iterrows():
            round_num = event["RoundNumber"]
            if round_num == 0 or (
                not force_refresh and (year, round_num) in self.existing_events
            ):
                continue

            logger.info(f"Processing {year} Round {round_num}: {event['EventName']}")
            race_df = self._fetch_session_with_retry(year, round_num, "R", event)
            if not race_df.empty:
                self._append_and_save(race_df, config.get("paths.races_csv"))

            quali_df = self._fetch_session_with_retry(year, round_num, "Q", event)
            if not quali_df.empty:
                self._append_and_save(quali_df, config.get("paths.quali_csv"))

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
                    time.sleep(base_delay * (2**attempt))
        return pd.DataFrame()

    def _extract_race_data(self, session: Any, event: pd.Series) -> pd.DataFrame:
        results = session.results
        if results.empty:
            return pd.DataFrame()

        race_data = []
        for _, result in results.iterrows():
            race_data.append(
                {
                    "Year": event["EventDate"].year,
                    "Race_Num": event["RoundNumber"],
                    "Circuit": event["EventName"].replace(" Grand Prix", ""),
                    "Date": event["EventDate"].date(),
                    "Driver": result["Abbreviation"],
                    "Team": result["TeamName"],
                    "Grid": result["GridPosition"],
                    "Position": result["Position"],
                    "Status": result["Status"],
                    "Laps": result["Laps"],
                    "Points": result["Points"],
                }
            )
        return pd.DataFrame(race_data)

    def _extract_qualifying_data(self, session: Any, event: pd.Series) -> pd.DataFrame:
        results = session.results
        if results.empty:
            return pd.DataFrame()

        quali_data = []
        for _, result in results.iterrows():
            quali_data.append(
                {
                    "Year": event["EventDate"].year,
                    "Race_Num": event["RoundNumber"],
                    "Circuit": event["EventName"].replace(" Grand Prix", ""),
                    "Date": event["EventDate"].date(),
                    "Driver": result["Abbreviation"],
                    "Team": result["TeamName"],
                    "Position": result["Position"],
                    "Q1": result["Q1"],
                    "Q2": result["Q2"],
                    "Q3": result["Q3"],
                }
            )
        return pd.DataFrame(quali_data)

    def _append_and_save(self, new_data: pd.DataFrame, filepath: str):
        if new_data.empty:
            return
        existing = pd.read_csv(filepath) if os.path.exists(filepath) else pd.DataFrame()
        combined = pd.concat([existing, new_data])
        combined.drop_duplicates(
            subset=["Year", "Race_Num", "Driver"], keep="last", inplace=True
        )
        combined.to_csv(filepath, index=False)


class F1DataLoader:
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        hist_races = (
            pd.read_csv(config.get("paths.races_csv"), on_bad_lines="warn")
            if os.path.exists(config.get("paths.races_csv"))
            else pd.DataFrame()
        )
        hist_quali = (
            pd.read_csv(config.get("paths.quali_csv"), on_bad_lines="warn")
            if os.path.exists(config.get("paths.quali_csv"))
            else pd.DataFrame()
        )
        return hist_races, hist_quali

    def _get_upcoming_race_info(self) -> pd.DataFrame:
        try:
            schedule = fastf1.get_event_schedule(datetime.now().year)
            now = pd.Timestamp.now(tz="UTC")
            upcoming = schedule[schedule["EventDate"] > now]
            if upcoming.empty:
                return pd.DataFrame()
            next_event = upcoming.iloc[0]
            drivers = self._get_current_driver_lineup()
            if not drivers:
                return pd.DataFrame()

            upcoming_info = pd.DataFrame(
                [
                    {
                        "Year": next_event["EventDate"].year,
                        "Race_Num": next_event["RoundNumber"],
                        "Circuit": next_event["EventName"].replace(" Grand Prix", ""),
                        "Race_Name": next_event["EventName"],
                        "Date": next_event["EventDate"].date(),
                        "Driver": d["Driver"],
                        "Team": d["Team"],
                    }
                    for d in drivers
                ]
            )
            return upcoming_info
        except Exception as e:
            logger.error(f"Failed to generate upcoming race info: {e}")
            return pd.DataFrame()

    def _get_current_driver_lineup(self) -> List[Dict]:
        hist_races, _ = self.load_all_data()
        if hist_races.empty:
            return []
        latest = hist_races.iloc[-20:]  # Last 20 rows assumed to be last race
        return [
            {"Driver": row["Driver"], "Team": row["Team"]}
            for _, row in latest.iterrows()
        ]

    def _get_latest_qualifying(self) -> Optional[pd.DataFrame]:
        _, hist_quali = self.load_all_data()
        if hist_quali.empty:
            return None
        current_year = datetime.now().year
        current_quali = hist_quali[hist_quali["Year"] == current_year]
        if current_quali.empty:
            return None
        latest = current_quali[
            current_quali["Race_Num"] == current_quali["Race_Num"].max()
        ]
        return latest
