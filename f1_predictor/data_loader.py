                             

import fastf1
import pandas as pd
import numpy as np
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict, Any
import warnings
import requests
import json
import hashlib
from contextlib import suppress

try:
    import requests_cache
except Exception:                                          
    requests_cache = None

warnings.filterwarnings("ignore")

from .config import config
from .utils import set_global_seeds

                                                 
fastf1.Cache.enable_cache(config.get("paths.cache_dir"))
try:
    seed_global = int(config.get("general.random_seed", config.get("general.random_state", 42)))
    set_global_seeds(seed_global)
except Exception:
    pass

                                                                       
with suppress(Exception):
    if requests_cache is not None:
        ttl_hours = int(
            config.get(
                "data_collection.external_apis.openweather_cache_ttl_hours", 12
            )
        )
        expire = ttl_hours * 3600
        os.makedirs(config.get("paths.cache_dir"), exist_ok=True)
        requests_cache.install_cache(
            cache_name=os.path.join(config.get("paths.cache_dir"), "http_cache"),
            backend="sqlite",
            expire_after=expire,
        )
logger = logging.getLogger(__name__)


class F1DataCollector:
    def __init__(self):
        self.existing_events = set()
        self.ingestion_state_path = config.get("paths.ingestion_state")
        self._load_existing_data_for_resume()

    def _load_existing_data_for_resume(self):
        races_csv_path = config.get("paths.races_csv")
        try:
            df = pd.read_csv(races_csv_path, usecols=["Year", "Race_Num"])
            for row in df.itertuples(index=False):
                self.existing_events.add((int(row.Year), int(row.Race_Num)))
            logger.info(
                f"Loaded {len(self.existing_events)} existing event records."
            )
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
                        event_rows = races_full[(races_full["Year"] == y) & (races_full["Race_Num"] == r)]
                        if not event_rows.empty:
                            subset = (
                                event_rows.sort_values(["Year", "Race_Num", "Driver"]).astype(str)
                            )
                            payload = subset[
                                [
                                    c
                                    for c in [
                                        "Year",
                                        "Race_Num",
                                        "Driver",
                                        "Team",
                                        "Position",
                                        "Grid",
                                    ]
                                    if c in subset.columns
                                ]
                            ].to_csv(index=False)
                            hasher = hashlib.md5(payload.encode("utf-8")).hexdigest()
                            ok = hasher == checksum
                    if ok:
                        self.existing_events.add((y, r))
                    else:
                                                                                
                        with suppress(Exception):
                            self.existing_events.discard((y, r))
        except Exception as e:
            logger.warning(f"Failed to load ingestion state: {e}")

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
                with suppress(Exception):
                    os.remove(self.ingestion_state_path)

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

                                                
                                                                      
            event_dt = pd.to_datetime(event['EventDate'], errors='coerce')
            if pd.notna(event_dt):
                                                            
                try:
                    event_dt = event_dt.tz_convert("UTC")
                except Exception:
                    try:
                        event_dt = event_dt.tz_localize("UTC") if event_dt.tzinfo is None else event_dt.tz_convert("UTC")
                    except Exception:
                        pass
                with suppress(Exception):
                    event_dt = event_dt.tz_localize(None)
            now_utc_naive = pd.Timestamp.utcnow().tz_localize(None)
            if isinstance(event_dt, pd.Timestamp) and pd.notna(event_dt) and event_dt > now_utc_naive:
                logger.info(f"Stopping data collection for {year} as event {event['EventName']} is in the future.")
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
                    time.sleep(base_delay * (2**attempt))
        return pd.DataFrame()

    def _extract_race_data(self, session: Any, event: pd.Series) -> pd.DataFrame:
        results = session.results
        weather = session.weather_data
        if results.empty:
            return pd.DataFrame()

        race_data = []
                                              
        try:
            event_format = str(event.get("EventFormat", "")).lower()
            is_sprint = 1 if ("sprint" in event_format) else 0
        except Exception:
            is_sprint = 0
        for _, result in results.iterrows():
            avg_weather = weather.mean()
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

        quali_data = []
        try:
            event_format = str(event.get("EventFormat", "")).lower()
            is_sprint = 1 if ("sprint" in event_format) else 0
        except Exception:
            is_sprint = 0
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
                                            
        with suppress(Exception):
            new_data["Year"] = pd.to_numeric(new_data["Year"], errors="coerce").astype("Int64")
        with suppress(Exception):
            new_data["Race_Num"] = pd.to_numeric(new_data["Race_Num"], errors="coerce").astype("Int64")
        with suppress(Exception):
            new_data["Date"] = pd.to_datetime(new_data["Date"], errors="coerce", utc=True).dt.tz_localize(None)
        with suppress(Exception):
            new_data["Driver"] = new_data["Driver"].astype(str)
        with suppress(Exception):
            new_data["Team"] = new_data["Team"].astype(str)
                                                           
        new_data = self._apply_canonicalization(new_data)

        combined = pd.concat([existing, new_data])
        combined.drop_duplicates(
            subset=["Year", "Race_Num", "Driver"], keep="last", inplace=True
        )
        combined.to_csv(filepath, index=False)

    def _update_ingestion_state(self, year: int, round_num: int, df: pd.DataFrame) -> None:
        """Persist a durable record of processed events and a simple checksum to guard against partial writes."""
        if not self.ingestion_state_path:
            return
        try:
            os.makedirs(os.path.dirname(self.ingestion_state_path), exist_ok=True)
            state: Dict[str, Any] = {"processed": []}
            if os.path.exists(self.ingestion_state_path):
                with open(self.ingestion_state_path, "r", encoding="utf-8") as f:
                    with suppress(Exception):
                        state = json.load(f) or {"processed": []}
                                                 
            subset = df.sort_values(["Year", "Race_Num", "Driver"]).astype(str)
            payload = subset[[c for c in subset.columns if c in ["Year", "Race_Num", "Driver", "Team", "Position", "Grid"]]].to_csv(index=False)
            checksum = hashlib.md5(payload.encode("utf-8")).hexdigest()
            entry = {"Year": int(year), "Race_Num": int(round_num), "checksum": checksum, "rows": int(df.shape[0])}
                                                
            others = [e for e in state.get("processed", []) if not (int(e.get("Year")) == year and int(e.get("Race_Num")) == round_num)]
            state["processed"] = others + [entry]
            with open(self.ingestion_state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update ingestion state: {e}")

    def _apply_canonicalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply driver/team canonicalization mappings from config if present."""
        drivers_map: Dict[str, str] = config.get("feature_engineering.canonicalization.drivers", {}) or {}
        teams_map: Dict[str, str] = config.get("feature_engineering.canonicalization.teams", {}) or {}
        out = df.copy()
        if "Driver" in out.columns and drivers_map:
            out["Driver"] = out["Driver"].astype(str).map(lambda x: drivers_map.get(x, x))
        if "Team" in out.columns and teams_map:
            out["Team"] = out["Team"].astype(str).map(lambda x: teams_map.get(x, x))
        return out


class F1DataLoader:
    def _normalize_event_dates(self, df: pd.DataFrame, column: str = "EventDate", return_naive: bool = False) -> pd.Series:
        """Return a Series of UTC-normalized datetimes from a schedule column.

        This uses `pd.to_datetime(..., utc=True)` to safely coerce mixed tz-naive and tz-aware
        inputs into a consistent tz-aware UTC dtype, then optionally drops tz info.
        """
        series_utc = pd.to_datetime(df[column], errors="coerce", utc=True)
        if return_naive:
            with suppress(Exception):
                return series_utc.dt.tz_convert(None)
        return series_utc
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            hist_races = pd.read_csv(config.get("paths.races_csv"))
        except FileNotFoundError:
            hist_races = pd.DataFrame()

        try:
            hist_quali = pd.read_csv(config.get("paths.quali_csv"))
        except FileNotFoundError:
            hist_quali = pd.DataFrame()

                                                                                      
        for df in (hist_races, hist_quali):
            if not df.empty and 'Date' in df.columns:
                                                                             
                dt = pd.to_datetime(df['Date'], errors='coerce', utc=True)
                df['Date'] = dt.dt.tz_localize(None)
            if not df.empty and 'Year' in df.columns:
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            if not df.empty and 'Race_Num' in df.columns:
                df['Race_Num'] = pd.to_numeric(df['Race_Num'], errors='coerce')
            if not df.empty and 'Circuit' in df.columns:
                                                                                         
                mask = df['Circuit'].notna() & ~df['Circuit'].astype(str).str.endswith(' Grand Prix', na=False)
                if mask.any():
                    df.loc[mask, 'Circuit'] = df.loc[mask, 'Circuit'].astype(str) + ' Grand Prix'
                                                                        
        if not hist_races.empty:
            hist_races = self._apply_canonicalization(hist_races)
        if not hist_quali.empty:
            hist_quali = self._apply_canonicalization(hist_quali)
        return hist_races, hist_quali

    def _apply_canonicalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply driver/team canonicalization mappings from config if present.

        This mirrors the logic used during ingestion to ensure consistent identifiers
        across all loader outputs as well.
        """
        try:
            drivers_map: Dict[str, str] = config.get("feature_engineering.canonicalization.drivers", {}) or {}
            teams_map: Dict[str, str] = config.get("feature_engineering.canonicalization.teams", {}) or {}
            out = df.copy()
            if "Driver" in out.columns and drivers_map:
                out["Driver"] = out["Driver"].astype(str).map(lambda x: drivers_map.get(x, x))
            if "Team" in out.columns and teams_map:
                out["Team"] = out["Team"].astype(str).map(lambda x: teams_map.get(x, x))
            return out
        except Exception:
            return df

    def _get_circuit_coords(self, circuit_name: str) -> Optional[Dict[str, float]]:
        locations = config.get('feature_engineering.circuit_locations', {}) or {}
                                                    
        candidates = [circuit_name]
        if isinstance(circuit_name, str):
            if not circuit_name.endswith(' Grand Prix'):
                candidates.append(f"{circuit_name} Grand Prix")
            else:
                candidates.append(circuit_name.replace(' Grand Prix', ''))
        for cand in candidates:
            coords = locations.get(cand)
            if isinstance(coords, dict) and {'lat', 'lon'}.issubset(coords.keys()):
                return {'lat': float(coords['lat']), 'lon': float(coords['lon'])}
        return None

    def _get_event_meta(self, year: int, race_name: str) -> Optional[Dict[str, Any]]:
        try:
            schedule = fastf1.get_event_schedule(year)
            event_rows = schedule[schedule["EventName"] == race_name]
            if event_rows.empty:
                return None
            event = event_rows.iloc[0]
            return {
                'round': int(event["RoundNumber"]) if not pd.isna(event["RoundNumber"]) else None,
                'event_date': pd.to_datetime(event["EventDate"]).to_pydatetime(),
                'circuit_name': str(event["EventName"]).replace(" Grand Prix", "")
            }
        except Exception:
            return None

    def _load_cached_forecast(self) -> pd.DataFrame:
        path = config.get('paths.weather_csv')
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def _save_cached_forecast(self, df_row: Dict[str, Any]):
        path = config.get('paths.weather_csv')
        try:
            existing = pd.read_csv(path)
        except FileNotFoundError:
            existing = pd.DataFrame()
        except Exception:
            existing = pd.DataFrame()
        new_df = pd.DataFrame([df_row])
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["Year", "Race_Num"], keep="last", inplace=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        combined.to_csv(path, index=False)

    def _fetch_openweathermap_forecast(self, lat: float, lon: float, hours_ahead: int) -> Optional[Dict[str, float]]:
        api_key = config.get('data_collection.external_apis.openweathermap_api_key')
        if not api_key or api_key == 'YOUR_API_KEY_HERE':
            return None
        try:
            url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=minutely,daily,alerts&appid={api_key}&units=metric"
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                return None
            data = r.json()
            hourly = data.get('hourly', [])
            if not hourly:
                return None
            horizon = max(1, int(hours_ahead))
            subset = hourly[:horizon]
            temps = [h.get('temp') for h in subset if 'temp' in h]
            hums = [h.get('humidity') for h in subset if 'humidity' in h]
            press = [h.get('pressure') for h in subset if 'pressure' in h]
            winds = [h.get('wind_speed') for h in subset if 'wind_speed' in h]
            rain_prob = [h.get('pop', 0) for h in subset]
            return {
                'AirTemp': float(np.nanmean(temps)) if temps else np.nan,
                'Humidity': float(np.nanmean(hums)) if hums else np.nan,
                'Pressure': float(np.nanmean(press)) if press else np.nan,
                'WindSpeed': float(np.nanmean(winds)) if winds else np.nan,
                'Rainfall': float(100.0 * np.nanmean(rain_prob)) if rain_prob else np.nan,
            }
        except Exception:
            return None

    def _get_event_weather_forecast(self, year: int, race_name: str) -> Optional[Dict[str, Any]]:
        meta = self._get_event_meta(year, race_name)
        if meta is None or meta.get('round') is None:
            return None
        round_num = int(meta['round'])
        circuit = meta['circuit_name']
        cached = self._load_cached_forecast()
        if not cached.empty:
            row = cached[(cached['Year'] == year) & (cached['Race_Num'] == round_num)]
            if not row.empty:
                r = row.iloc[0]
                                                              
                ttl_hours = int(config.get('data_collection.external_apis.openweather_cache_ttl_hours', 12))
                fetched_at = pd.to_datetime(r.get('FetchedAt'), errors='coerce', utc=True)
                now_utc = pd.Timestamp.utcnow().tz_localize('UTC')
                if pd.notna(fetched_at) and (now_utc - fetched_at).total_seconds() < ttl_hours * 3600:
                    return {
                        'AirTemp': r.get('AirTemp', np.nan),
                        'Humidity': r.get('Humidity', np.nan),
                        'Pressure': r.get('Pressure', np.nan),
                        'WindSpeed': r.get('WindSpeed', np.nan),
                        'Rainfall': r.get('Rainfall', np.nan),
                    }
        coords = self._get_circuit_coords(circuit)
        if coords is None:
            return None
        hours_ahead = int(config.get('feature_engineering.weather_forecast.hours_ahead', 24))
        fore = self._fetch_openweathermap_forecast(coords['lat'], coords['lon'], hours_ahead)
        if fore is None:
            return None
        self._save_cached_forecast({
            'Year': year,
            'Race_Num': round_num,
            **fore,
            'FetchedAt': datetime.utcnow().isoformat()
        })
        return fore

    def _get_upcoming_race_info(self, year: int) -> pd.DataFrame:
        """Return driver lineup rows for the next upcoming event in the specified year."""
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

            upcoming_info = pd.DataFrame(
                [
                    {
                        "Year": int(pd.Timestamp(next_event["EventDate"]).year),
                        "Race_Num": int(next_event["RoundNumber"]),
                        "Circuit": str(next_event["EventName"]),
                        "Race_Name": str(next_event["EventName"]),
                        "Date": pd.Timestamp(next_event["EventDate"]).tz_localize(None).date(),
                        "Driver": d["Driver"],
                        "Team": d["Team"],
                        "Is_Sprint": 1 if str(next_event.get("EventFormat", "")).lower().find("sprint") != -1 else 0,
                    }
                    for d in drivers
                ]
            )
                                                  
            upcoming_info = self._apply_canonicalization(upcoming_info)
            return upcoming_info
        except Exception as e:
            logger.error(f"Failed to generate upcoming race info for {year}: {e}")
            return pd.DataFrame()

    def _get_race_info(self, year: int, race_name: str) -> pd.DataFrame:
        """Return driver lineup rows for the specified race in the specified year."""
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
            info = pd.DataFrame(
                [
                    {
                        "Year": int(pd.Timestamp(event["EventDate"]).year),
                        "Race_Num": int(round_num) if round_num is not None else int(event["RoundNumber"]),
                        "Circuit": str(event["EventName"]),
                        "Race_Name": str(event["EventName"]),
                        "Date": pd.Timestamp(event["EventDate"]).tz_localize(None).date(),
                        "Driver": d["Driver"],
                        "Team": d["Team"],
                        "Is_Sprint": 1 if str(event.get("EventFormat", "")).lower().find("sprint") != -1 else 0,
                    }
                    for d in drivers
                ]
            )
            fore = self._get_event_weather_forecast(int(event["EventDate"].year), str(event["EventName"]))
            if fore is not None:
                for k, v in fore.items():
                    info[k] = v
                                                  
            info = self._apply_canonicalization(info)
            return info
        except Exception as e:
            logger.error(f"Failed to generate race info for {year} {race_name}: {e}")
            return pd.DataFrame()

    def _get_current_driver_lineup(self) -> List[Dict]:
        """Fallback: driver lineup from latest completed race of the current year."""
        try:
            latest_season = fastf1.get_event_schedule(datetime.now().year).copy()
            latest_season["EventDate"] = self._normalize_event_dates(latest_season, "EventDate", return_naive=False)
            completed = latest_season[latest_season["EventDate"] < pd.Timestamp.utcnow().tz_localize("UTC")]
            if completed.empty:
                return []
            latest_race = completed.sort_values("EventDate").iloc[-1]
                                                                                  
            session = fastf1.get_session(int(pd.Timestamp(latest_race.EventDate).year), int(latest_race.RoundNumber), 'R')
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            results = session.results
            return [{'Driver': r.get('Abbreviation', 'UNK'), 'Team': r.get('TeamName', 'Unknown')} for _, r in results.iterrows()]
        except Exception as e:
            logger.error(f"Could not get current driver lineup: {e}")
            return []

    def _get_driver_lineup(self, year: int, up_to_round: Optional[int] = None) -> List[Dict]:
        """Driver lineup for a season up to a certain round.

        Strategy:
        1) Prefer lineup from cached historical races (stable and fast).
        2) Fallback to FastF1 latest completed race of the season.
        3) Fallback to current-year latest race if season hasn't started.
        """
                                            
        try:
            hist_races, _ = self.load_all_data()
            if not hist_races.empty:
                df = hist_races.copy()
                                      
                with suppress(Exception):
                    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                with suppress(Exception):
                    df['Race_Num'] = pd.to_numeric(df['Race_Num'], errors='coerce')
                with suppress(Exception):
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

                                                                                   
                season_mask = df['Year'] == float(year)
                if up_to_round is not None and 'Race_Num' in df.columns:
                    season_df = df[season_mask & (df['Race_Num'] <= float(up_to_round))]
                else:
                    season_df = df[season_mask]
                if season_df.empty:
                    prior_df = df[df['Year'] < float(year)]
                    season_df = prior_df
                if not season_df.empty:
                                                                 
                    sort_cols = [c for c in ['Year', 'Race_Num', 'Date'] if c in season_df.columns]
                    season_df = season_df.sort_values(sort_cols)
                    last_year = int(season_df.iloc[-1]['Year']) if 'Year' in season_df.columns else year
                    last_round = int(season_df.iloc[-1]['Race_Num']) if 'Race_Num' in season_df.columns else int(season_df.shape[0])
                    event_rows = season_df[(season_df['Year'] == last_year) & (season_df['Race_Num'] == last_round)]
                    if not event_rows.empty:
                                                                      
                        event_rows = event_rows.sort_values(['Driver', 'Team'])
                        lineup = (
                            event_rows.groupby('Driver', as_index=False)
                            .agg({'Team': 'last'})
                            .reset_index(drop=True)
                        )
                        return [{'Driver': str(r.Driver), 'Team': str(r.Team)} for r in lineup.itertuples(index=False)]
        except Exception:
            pass

                                                                    
        try:
            schedule = fastf1.get_event_schedule(year).copy()
            schedule["EventDate"] = self._normalize_event_dates(schedule, "EventDate", return_naive=False)
            now = pd.Timestamp.utcnow().tz_localize("UTC")
            completed = schedule[schedule["EventDate"] < now]
            if up_to_round is not None:
                completed = completed[completed["RoundNumber"] <= up_to_round]
            if completed.empty:
                return self._get_current_driver_lineup()
            latest = completed.sort_values("EventDate").iloc[-1]
            session = fastf1.get_session(int(pd.Timestamp(latest.EventDate).year), int(latest.RoundNumber), 'R')
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            results = session.results
            return [{'Driver': r.get('Abbreviation', 'UNK'), 'Team': r.get('TeamName', 'Unknown')} for _, r in results.iterrows()]
        except Exception as e:
            logger.error(f"Failed to build driver lineup for {year}: {e}")
            return []

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
