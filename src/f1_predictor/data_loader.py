"""Data loading for training and inference.

F1DataCollector (ingestion) lives in collector.py — re-exported here for
backward compatibility so existing ``from .data_loader import F1DataCollector``
imports continue to work.

F1DataLoader handles loading persisted data via F1Store and building driver
lineup / weather forecast rows for upcoming events.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import fastf1
import numpy as np
import pandas as pd
import requests

from .collector import F1DataCollector  # noqa: F401 – re-exported for compat
from .config import config
from .store import F1Store

logger = logging.getLogger(__name__)


class F1DataLoader:
    """Loads F1 data from the DuckDB store and builds driver-lineup / weather rows for inference."""

    # ------------------------------------------------------------------
    # Core data loading
    # ------------------------------------------------------------------

    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        with F1Store() as store:
            hist_races = store.get_races()
            hist_quali = store.get_qualifying()

        for df in (hist_races, hist_quali):
            if df.empty:
                continue
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            if "Year" in df.columns:
                df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
            if "Race_Num" in df.columns:
                df["Race_Num"] = pd.to_numeric(df["Race_Num"], errors="coerce")
            if "Circuit" in df.columns:
                mask = (
                    df["Circuit"].notna()
                    & ~df["Circuit"].astype(str).str.endswith(" Grand Prix", na=False)
                )
                if mask.any():
                    df.loc[mask, "Circuit"] = df.loc[mask, "Circuit"].astype(str) + " Grand Prix"

        if not hist_races.empty:
            hist_races = self._apply_canonicalization(hist_races)
        if not hist_quali.empty:
            hist_quali = self._apply_canonicalization(hist_quali)
        return hist_races, hist_quali

    # ------------------------------------------------------------------
    # Canonicalization
    # ------------------------------------------------------------------

    def _apply_canonicalization(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            drivers_map: Dict[str, str] = (
                config.get("feature_engineering.canonicalization.drivers", {}) or {}
            )
            teams_map: Dict[str, str] = (
                config.get("feature_engineering.canonicalization.teams", {}) or {}
            )
            out = df.copy()
            if "Driver" in out.columns and drivers_map:
                out["Driver"] = out["Driver"].astype(str).map(lambda x: drivers_map.get(x, x))
            if "Team" in out.columns and teams_map:
                out["Team"] = out["Team"].astype(str).map(lambda x: teams_map.get(x, x))
            return out
        except Exception as exc:
            logger.warning(f"Canonicalization failed, returning original df: {exc}")
            return df

    # ------------------------------------------------------------------
    # Event / schedule helpers
    # ------------------------------------------------------------------

    def _normalize_event_dates(
        self, df: pd.DataFrame, column: str = "EventDate", return_naive: bool = False
    ) -> pd.Series:
        series_utc = pd.to_datetime(df[column], errors="coerce", utc=True)
        if return_naive:
            try:
                return series_utc.dt.tz_convert(None)
            except Exception as exc:
                logger.warning(f"Failed to strip timezone from event dates: {exc}")
        return series_utc

    def _get_event_meta(self, year: int, race_name: str) -> Optional[Dict[str, Any]]:
        try:
            schedule = fastf1.get_event_schedule(year)
            event_rows = schedule[schedule["EventName"] == race_name]
            if event_rows.empty:
                return None
            event = event_rows.iloc[0]
            return {
                "round": int(event["RoundNumber"]) if not pd.isna(event["RoundNumber"]) else None,
                "event_date": pd.to_datetime(event["EventDate"]).to_pydatetime(),
                "circuit_name": str(event["EventName"]).replace(" Grand Prix", ""),
            }
        except Exception as exc:
            logger.warning(f"Could not fetch event meta for {year} '{race_name}': {exc}")
            return None

    # ------------------------------------------------------------------
    # Weather forecast (DB-backed cache)
    # ------------------------------------------------------------------

    def _get_circuit_coords(self, circuit_name: str) -> Optional[Dict[str, float]]:
        locations = config.get("feature_engineering.circuit_locations", {}) or {}
        candidates = [circuit_name]
        if isinstance(circuit_name, str):
            if not circuit_name.endswith(" Grand Prix"):
                candidates.append(f"{circuit_name} Grand Prix")
            else:
                candidates.append(circuit_name.replace(" Grand Prix", ""))
        for cand in candidates:
            coords = locations.get(cand)
            if isinstance(coords, dict) and {"lat", "lon"}.issubset(coords.keys()):
                return {"lat": float(coords["lat"]), "lon": float(coords["lon"])}
        return None

    def _fetch_openweathermap_forecast(
        self, lat: float, lon: float, hours_ahead: int
    ) -> Optional[Dict[str, float]]:
        api_key = config.get("data_collection.external_apis.openweathermap_api_key")
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            return None
        try:
            url = (
                f"https://api.openweathermap.org/data/2.5/onecall"
                f"?lat={lat}&lon={lon}&exclude=minutely,daily,alerts"
                f"&appid={api_key}&units=metric"
            )
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                logger.warning(f"OpenWeatherMap returned status {r.status_code}")
                return None
            data = r.json()
            hourly = data.get("hourly", [])
            if not hourly:
                return None
            subset = hourly[: max(1, int(hours_ahead))]
            temps = [h.get("temp") for h in subset if "temp" in h]
            hums = [h.get("humidity") for h in subset if "humidity" in h]
            press = [h.get("pressure") for h in subset if "pressure" in h]
            winds = [h.get("wind_speed") for h in subset if "wind_speed" in h]
            rain_prob = [h.get("pop", 0) for h in subset]
            return {
                "AirTemp": float(np.nanmean(temps)) if temps else np.nan,
                "Humidity": float(np.nanmean(hums)) if hums else np.nan,
                "Pressure": float(np.nanmean(press)) if press else np.nan,
                "WindSpeed": float(np.nanmean(winds)) if winds else np.nan,
                "Rainfall": float(100.0 * np.nanmean(rain_prob)) if rain_prob else np.nan,
            }
        except Exception as exc:
            logger.warning(f"OpenWeatherMap fetch failed: {exc}")
            return None

    def _get_event_weather_forecast(self, year: int, race_name: str) -> Optional[Dict[str, Any]]:
        meta = self._get_event_meta(year, race_name)
        if meta is None or meta.get("round") is None:
            return None
        circuit = meta["circuit_name"]
        ttl_hours = int(config.get("data_collection.external_apis.openweather_cache_ttl_hours", 12))

        # Check DB cache first
        with F1Store() as store:
            cached_payload = store.get_weather_cache(circuit)
            if cached_payload:
                try:
                    return json.loads(cached_payload)
                except Exception as exc:
                    logger.warning(f"Weather cache parse error for {circuit}: {exc}")

        coords = self._get_circuit_coords(circuit)
        if coords is None:
            return None
        hours_ahead = int(config.get("feature_engineering.weather_forecast.hours_ahead", 24))
        fore = self._fetch_openweathermap_forecast(coords["lat"], coords["lon"], hours_ahead)
        if fore is None:
            return None

        # Persist to DB
        with F1Store() as store:
            store.upsert_weather_cache(circuit, json.dumps(fore), ttl_hours)

        return fore

    # ------------------------------------------------------------------
    # Driver lineup helpers
    # ------------------------------------------------------------------

    def _get_upcoming_race_info(self, year: int) -> pd.DataFrame:
        try:
            schedule = fastf1.get_event_schedule(year).copy()
            schedule["EventDate"] = self._normalize_event_dates(schedule, "EventDate", return_naive=False)
            now = pd.Timestamp.utcnow().tz_localize("UTC")
            upcoming = schedule[schedule["EventDate"] > now]
            if upcoming.empty:
                return pd.DataFrame()
            next_event = upcoming.sort_values("EventDate").iloc[0]
            drivers = self._get_driver_lineup(year, up_to_round=int(next_event["RoundNumber"]))
            if not drivers:
                return pd.DataFrame()

            rows = [
                {
                    "Year": int(pd.Timestamp(next_event["EventDate"]).year),
                    "Race_Num": int(next_event["RoundNumber"]),
                    "Circuit": str(next_event["EventName"]),
                    "Race_Name": str(next_event["EventName"]),
                    "Date": pd.Timestamp(next_event["EventDate"]).tz_localize(None).date(),
                    "Driver": d["Driver"],
                    "Team": d["Team"],
                    "Is_Sprint": 1 if "sprint" in str(next_event.get("EventFormat", "")).lower() else 0,
                }
                for d in drivers
            ]
            return self._apply_canonicalization(pd.DataFrame(rows))
        except Exception as exc:
            logger.error(f"Failed to generate upcoming race info for {year}: {exc}")
            return pd.DataFrame()

    def _get_race_info(self, year: int, race_name: str) -> pd.DataFrame:
        try:
            schedule = fastf1.get_event_schedule(year).copy()
            schedule["EventDate"] = self._normalize_event_dates(schedule, "EventDate", return_naive=False)
            event_rows = schedule[schedule["EventName"] == race_name]
            if event_rows.empty:
                logger.error(f"Race '{race_name}' not found in {year} schedule.")
                return pd.DataFrame()
            event = event_rows.iloc[0]
            round_num = int(event["RoundNumber"]) if not pd.isna(event["RoundNumber"]) else None
            drivers = self._get_driver_lineup(year, up_to_round=round_num)
            if not drivers:
                return pd.DataFrame()
            rows = [
                {
                    "Year": int(pd.Timestamp(event["EventDate"]).year),
                    "Race_Num": int(round_num) if round_num is not None else int(event["RoundNumber"]),
                    "Circuit": str(event["EventName"]),
                    "Race_Name": str(event["EventName"]),
                    "Date": pd.Timestamp(event["EventDate"]).tz_localize(None).date(),
                    "Driver": d["Driver"],
                    "Team": d["Team"],
                    "Is_Sprint": 1 if "sprint" in str(event.get("EventFormat", "")).lower() else 0,
                }
                for d in drivers
            ]
            info = pd.DataFrame(rows)
            fore = self._get_event_weather_forecast(int(event["EventDate"].year), str(event["EventName"]))
            if fore is not None:
                for k, v in fore.items():
                    info[k] = v
            return self._apply_canonicalization(info)
        except Exception as exc:
            logger.error(f"Failed to generate race info for {year} {race_name}: {exc}")
            return pd.DataFrame()

    def _get_current_driver_lineup(self) -> List[Dict]:
        try:
            latest_season = fastf1.get_event_schedule(datetime.now().year).copy()
            latest_season["EventDate"] = self._normalize_event_dates(
                latest_season, "EventDate", return_naive=False
            )
            completed = latest_season[
                latest_season["EventDate"] < pd.Timestamp.utcnow().tz_localize("UTC")
            ]
            if completed.empty:
                return []
            latest_race = completed.sort_values("EventDate").iloc[-1]
            session = fastf1.get_session(
                int(pd.Timestamp(latest_race.EventDate).year),
                int(latest_race.RoundNumber),
                "R",
            )
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            return [
                {"Driver": r.get("Abbreviation", "UNK"), "Team": r.get("TeamName", "Unknown")}
                for _, r in session.results.iterrows()
            ]
        except Exception as exc:
            logger.error(f"Could not get current driver lineup: {exc}")
            return []

    def _get_driver_lineup(self, year: int, up_to_round: Optional[int] = None) -> List[Dict]:
        """Driver lineup for a season up to a given round.

        Strategy:
        1. Prefer lineup from historical races in DB (stable, fast).
        2. Fall back to FastF1 latest completed race of the season.
        3. Fall back to current-year latest race if season hasn't started.
        """
        # Strategy 1: DB
        try:
            hist_races, _ = self.load_all_data()
            if not hist_races.empty:
                df = hist_races.copy()
                for col in ("Year", "Race_Num"):
                    try:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    except Exception as exc:
                        logger.warning(f"Could not coerce '{col}' to numeric: {exc}")
                try:
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                except Exception as exc:
                    logger.warning(f"Could not parse 'Date' column: {exc}")

                season_df = df[df["Year"] == float(year)]
                if up_to_round is not None and "Race_Num" in df.columns:
                    season_df = season_df[season_df["Race_Num"] <= float(up_to_round)]
                if season_df.empty:
                    season_df = df[df["Year"] < float(year)]
                if not season_df.empty:
                    sort_cols = [c for c in ["Year", "Race_Num", "Date"] if c in season_df.columns]
                    season_df = season_df.sort_values(sort_cols)
                    last_year = int(season_df.iloc[-1]["Year"])
                    last_round = int(season_df.iloc[-1]["Race_Num"])
                    event_rows = season_df[
                        (season_df["Year"] == last_year) & (season_df["Race_Num"] == last_round)
                    ]
                    if not event_rows.empty:
                        lineup = (
                            event_rows.groupby("Driver", as_index=False)
                            .agg({"Team": "last"})
                            .reset_index(drop=True)
                        )
                        return [
                            {"Driver": str(r.Driver), "Team": str(r.Team)}
                            for r in lineup.itertuples(index=False)
                        ]
        except Exception as exc:
            logger.warning(f"Historical lineup lookup failed for {year}: {exc}")

        # Strategy 2: FastF1 schedule
        try:
            schedule = fastf1.get_event_schedule(year).copy()
            schedule["EventDate"] = self._normalize_event_dates(
                schedule, "EventDate", return_naive=False
            )
            now = pd.Timestamp.utcnow().tz_localize("UTC")
            completed = schedule[schedule["EventDate"] < now]
            if up_to_round is not None:
                completed = completed[completed["RoundNumber"] <= up_to_round]
            if completed.empty:
                return self._get_current_driver_lineup()
            latest = completed.sort_values("EventDate").iloc[-1]
            session = fastf1.get_session(
                int(pd.Timestamp(latest.EventDate).year), int(latest.RoundNumber), "R"
            )
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            return [
                {"Driver": r.get("Abbreviation", "UNK"), "Team": r.get("TeamName", "Unknown")}
                for _, r in session.results.iterrows()
            ]
        except Exception as exc:
            logger.error(f"Failed to build driver lineup for {year}: {exc}")
            return []

    def _get_latest_qualifying(self) -> Optional[pd.DataFrame]:
        _, hist_quali = self.load_all_data()
        if hist_quali.empty:
            return None
        current_year = datetime.now().year
        current_quali = hist_quali[hist_quali["Year"] == current_year]
        if current_quali.empty:
            return None
        latest = current_quali[current_quali["Race_Num"] == current_quali["Race_Num"].max()]
        return latest
