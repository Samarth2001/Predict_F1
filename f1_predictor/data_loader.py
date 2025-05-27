# f1_predictor/data_loader.py - Advanced F1 Data Collection and Loading System

import fastf1
import pandas as pd
import numpy as np
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from . import config

# Configure FastF1
fastf1.Cache.enable_cache(config.FF1_CACHE_PATH)

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_CONFIG['level']))
logger = logging.getLogger(__name__)

class F1DataCollector:
    """
    Advanced F1 data collection system using FastF1 API.
    
    Handles comprehensive data fetching for historical and current seasons
    with intelligent caching, error handling, and data validation.
    """
    
    def __init__(self):
        self.session_cache = {}
        self.failed_sessions = set()
        self.data_quality_metrics = {}
        self._load_existing_data_for_resume()
        
    def _load_existing_data_for_resume(self):
        """Load existing race and quali data to assist in resuming downloads."""
        self.existing_races_df = pd.DataFrame()
        self.existing_quali_df = pd.DataFrame()
        if os.path.exists(config.RACES_CSV_PATH):
            try:
                self.existing_races_df = pd.read_csv(config.RACES_CSV_PATH)
                logger.info(f"Loaded {len(self.existing_races_df)} existing race records for download resumption.")
            except Exception as e:
                logger.warning(f"Could not load existing races CSV for resume: {e}")
        if os.path.exists(config.QUALI_CSV_PATH):
            try:
                self.existing_quali_df = pd.read_csv(config.QUALI_CSV_PATH)
                logger.info(f"Loaded {len(self.existing_quali_df)} existing quali records for download resumption.")
            except Exception as e:
                logger.warning(f"Could not load existing quali CSV for resume: {e}")

    def _is_event_data_present(self, year: int, event_identifier: Any, data_type: str) -> bool:
        """Check if data for a given event (by round number or name) already exists in the loaded CSVs."""
        df_to_check = self.existing_races_df if data_type == 'race' else self.existing_quali_df
        if df_to_check.empty:
            return False
        
        # Ensure key columns are present for matching
        # For round number matching, we need 'Year' and 'Race_Num'
        # For event name matching, we also need 'Circuit'
        required_cols_for_round_match = ['Year', 'Race_Num']
        required_cols_for_name_match = ['Year', 'Race_Num', 'Circuit']

        # Determine the actual identifier type
        is_round_number_identifier = isinstance(event_identifier, (int, float, np.number))
        is_name_identifier = isinstance(event_identifier, str)

        if is_round_number_identifier:
            if not all(col in df_to_check.columns for col in required_cols_for_round_match):
                logger.debug(f"_is_event_data_present: Key columns Year/Race_Num missing in {data_type} data for round matching. Assuming not present.")
                return False
            return not df_to_check[(df_to_check['Year'] == year) & (df_to_check['Race_Num'] == event_identifier)].empty
        
        elif is_name_identifier:
            if not all(col in df_to_check.columns for col in required_cols_for_name_match):
                logger.debug(f"_is_event_data_present: Key columns (Year, Race_Num, Circuit) missing in {data_type} data for name matching. Assuming not present.")
                return False
            normalized_event_name = event_identifier.replace(' Grand Prix', '')
            return not df_to_check[(df_to_check['Year'] == year) & (df_to_check['Circuit'] == normalized_event_name)].empty
        
        logger.debug(f"_is_event_data_present: Could not determine match type for {event_identifier}. Assuming not present.")
        return False
    
    def fetch_all_f1_data(self, start_year: int = None, end_year: int = None, 
                          force_refresh: bool = False) -> bool:
        """
        Fetch comprehensive F1 data for specified year range.
        
        Args:
            start_year: Starting year for data collection
            end_year: Ending year for data collection
            force_refresh: Force refresh of existing cached data
            
        Returns:
            True if successful, False otherwise
        """
        start_year = start_year or config.START_YEAR
        end_year = end_year or config.END_YEAR
        
        logger.info(f"Starting F1 data collection from {start_year} to {end_year}. Force refresh: {force_refresh}")
        
        # Reload existing data at the start of a full fetch operation to ensure it's current
        if not force_refresh:
            self._load_existing_data_for_resume()

        try:
            all_races_list = []
            all_quali_list = []
            
            # If not forcing a refresh, and existing data was loaded, add it to our lists
            if not force_refresh and not self.existing_races_df.empty:
                all_races_list.append(self.existing_races_df)
                logger.info(f"Initialized with {len(self.existing_races_df)} pre-existing race records.")
            
            if not force_refresh and not self.existing_quali_df.empty:
                all_quali_list.append(self.existing_quali_df)
                logger.info(f"Initialized with {len(self.existing_quali_df)} pre-existing quali records.")

            for year in range(start_year, end_year + 1):
                logger.info(f"Collecting data for {year} season...")
                
                # Pass force_refresh to _fetch_season_data
                year_races_df, year_quali_df = self._fetch_season_data(year, force_refresh)
                
                if year_races_df is not None and not year_races_df.empty:
                    all_races_list.append(year_races_df)
                if year_quali_df is not None and not year_quali_df.empty:
                    all_quali_list.append(year_quali_df)
            
            # Combine all data
            final_races_df = pd.DataFrame()
            if all_races_list:
                final_races_df = pd.concat(all_races_list, ignore_index=True)
                # Drop duplicates before saving, keeping the last entry (potentially updated)
                final_races_df.drop_duplicates(subset=['Year', 'Race_Num', 'Driver'], keep='last', inplace=True)
                self._save_data(final_races_df, config.RACES_CSV_PATH) # append=False by default, overwrites
                logger.info(f"Saved {len(final_races_df)} unique race records to {config.RACES_CSV_PATH}")
            
            final_quali_df = pd.DataFrame()
            if all_quali_list:
                final_quali_df = pd.concat(all_quali_list, ignore_index=True)
                # Drop duplicates before saving
                final_quali_df.drop_duplicates(subset=['Year', 'Race_Num', 'Driver'], keep='last', inplace=True)
                self._save_data(final_quali_df, config.QUALI_CSV_PATH) # append=False by default, overwrites
                logger.info(f"Saved {len(final_quali_df)} unique qualifying records to {config.QUALI_CSV_PATH}")
            
            # Generate data quality report based on the final collected data
            self._generate_data_quality_report(final_races_df, final_quali_df)
            
            logger.info("F1 data collection completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"F1 data collection failed: {e}", exc_info=True)
            return False
    
    def _fetch_season_data(self, year: int, force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch data for a complete season, attempting to resume if possible."""
        season_races_list = []
        season_quali_list = []
        try:
            season_schedule = fastf1.get_event_schedule(year)
            if season_schedule.empty:
                logger.warning(f"No events found for {year}. Skipping season.")
                return pd.DataFrame(), pd.DataFrame()
            
            # Sort by round number to process in order
            season_schedule = season_schedule.sort_values(by='RoundNumber').reset_index(drop=True)

            for _, event in season_schedule.iterrows():
                event_name = event.get('EventName', f'Round {event.get("RoundNumber")}') # Use RoundNum if EventName is missing
                round_num = event.get('RoundNumber')
                event_date_obj = pd.to_datetime(event.get('EventDate', 'today'))
                # Only process events up to the current date + a small buffer (e.g., 7 days for race weekend completion)
                if event_date_obj > datetime.now() + timedelta(days=config.FUTURE_EVENT_BUFFER_DAYS):
                    logger.info(f"  Skipping future event: {year} Round {round_num} ({event_name}) scheduled for {event_date_obj.date()}")
                    continue

                logger.info(f"  Processing {year} Round {round_num}: {event_name} on {event_date_obj.date()}")

                # Determine identifier for checking existing data: prefer RoundNumber if available
                identifier_for_check = round_num if pd.notna(round_num) else event_name

                # Fetch Race Data
                if not force_refresh and self._is_event_data_present(year, identifier_for_check, 'race'):
                    logger.info(f"    Race data for {year} {event_name} (ID: {identifier_for_check}) already exists. Skipping race download.")
                else:
                    logger.info(f"    Fetching race data for {year} {event_name}...")
                    race_data_df = self._fetch_session_with_retry(year, round_num, 'R', event)
                    if race_data_df is not None and not race_data_df.empty:
                        season_races_list.append(race_data_df)
                    elif not force_refresh:
                         logger.warning(f"    Race data fetch attempt for {year} {event_name} returned no data but was not skipped by resume logic.")

                # Fetch Qualifying Data
                if not force_refresh and self._is_event_data_present(year, identifier_for_check, 'quali'):
                    logger.info(f"    Qualifying data for {year} {event_name} (ID: {identifier_for_check}) already exists. Skipping quali download.")
                else:
                    logger.info(f"    Fetching qualifying data for {year} {event_name}...")
                    quali_data_df = self._fetch_session_with_retry(year, round_num, 'Q', event)
                    if quali_data_df is not None and not quali_data_df.empty:
                        season_quali_list.append(quali_data_df)
                    elif not force_refresh:
                        logger.warning(f"    Quali data fetch attempt for {year} {event_name} returned no data but was not skipped by resume logic.")
            
            final_season_races_df = pd.concat(season_races_list, ignore_index=True) if season_races_list else pd.DataFrame()
            final_season_quali_df = pd.concat(season_quali_list, ignore_index=True) if season_quali_list else pd.DataFrame()
            
            return final_season_races_df, final_season_quali_df
            
        except Exception as e:
            logger.error(f"Failed to fetch {year} season data: {e}", exc_info=True)
            return pd.DataFrame(), pd.DataFrame()
    
    def _fetch_session_with_retry(self, year: int, round_or_event_name: Any, session_code: str, event_details: pd.Series) -> Optional[pd.DataFrame]:
        """Fetch data for a single session (Race or Qualifying) with retry logic for rate limits."""
        max_retries = config.API_MAX_RETRIES
        base_delay = config.API_RETRY_DELAY
        # Use event_details for more robust naming if available
        event_display_name = event_details.get('EventName', str(round_or_event_name))
        actual_round_num = event_details.get('RoundNumber') # Use this for get_session

        if pd.isna(actual_round_num):
            logger.error(f"    Cannot fetch session {session_code} for {year} {event_display_name}: Missing RoundNumber in event details.")
            return None
        
        # Convert actual_round_num to int if it's not already (e.g. if it was float from Series)
        try:
            actual_round_num = int(actual_round_num)
        except ValueError:
            logger.error(f"    Cannot fetch session {session_code} for {year} {event_display_name}: Invalid RoundNumber {actual_round_num}.")
            return None

        for attempt in range(max_retries):
            try:
                logger.debug(f"    Attempt {attempt + 1}/{max_retries} to fetch {session_code} for {year} {event_display_name} (Round {actual_round_num})")
                session = fastf1.get_session(year, actual_round_num, session_code) # Use actual_round_num
                session.load(laps=True, telemetry=False, weather=True, messages=False, livedata=None) # Explicitly load what is needed
                
                if session_code == 'R':
                    data = self._extract_race_data(session, event_details)
                elif session_code == 'Q':
                    data = self._extract_qualifying_data(session, event_details)
                else:
                    logger.warning(f"    Unknown session code {session_code} requested.")
                    return None
                
                time.sleep(config.FETCH_DELAY) # Brief delay after successful fetch
                logger.info(f"    Successfully fetched {session_code} for {year} {event_display_name}.")
                return data

            except fastf1.ergast.ErgastMissingDataError as e: # Lower frequency data source
                logger.warning(f"    No data for {session_code} at {year} {event_display_name} on Ergast: {e}")
                self.failed_sessions.add(f"{year}-{actual_round_num}-{session_code}-ErgastMissingData")
                return None # No data available, don't retry for this specific error
            except fastf1.api.SessionNotAvailableError as e: # API says session doesn't exist
                logger.warning(f"    Session {session_code} for {year} {event_display_name} (Round {actual_round_num}) not available via API: {e}")
                self.failed_sessions.add(f"{year}-{actual_round_num}-{session_code}-NotAvailable")
                return None # Session doesn't exist or not loadable, don't retry
            except fastf1.req.RateLimitExceededError as e:
                logger.warning(f"    Rate limit hit fetching {session_code} for {year} {event_display_name}. Attempt {attempt + 1}/{max_retries}. {e}")
                self.failed_sessions.add(f"{year}-{actual_round_num}-{session_code}-RateLimited")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) # Exponential backoff
                    logger.info(f"    Waiting {delay} seconds before next attempt for {year} {event_display_name}...")
                    time.sleep(delay)
                else:
                    logger.error(f"    Max retries ({max_retries}) reached for {session_code} at {year} {event_display_name} due to rate limiting.")
                    return None
            except Exception as e:
                # Catching a broader range of fastf1 specific errors or general issues during load
                if "SessionNotAvailableError" in str(e) or "EventNotAvailable" in str(e) or "No data for this session" in str(e):
                     logger.warning(f"    Session data appears unavailable for {session_code} at {year} {event_display_name}: {e}")
                     self.failed_sessions.add(f"{year}-{actual_round_num}-{session_code}-LoadErrorNotAvailable")
                     return None # Treat as not available
                logger.error(f"    Failed to load/process {session_code} for {year} {event_display_name} (Round {actual_round_num}), attempt {attempt + 1}: {e}", exc_info=True)
                self.failed_sessions.add(f"{year}-{actual_round_num}-{session_code}-LoadProcessError")
                # For general errors, also retry with backoff
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"    Waiting {delay} seconds before next attempt for {year} {event_display_name} due to load/process error...")
                    time.sleep(delay)
                else:
                    logger.error(f"    Max retries ({max_retries}) reached for {session_code} at {year} {event_display_name} after load/process error.")
                    return None
        
        logger.error(f"    Exhausted all retries for {session_code} at {year} {event_display_name}.")
        return None
    
    def _extract_race_data(self, session: Any, event: pd.Series) -> pd.DataFrame:
        """Extract race results data from FastF1 session."""
        try:
            session.load()
            results = session.results
            
            if results.empty:
                return pd.DataFrame()
            
            # Extract key race data
            race_data = []
            
            for idx, result in results.iterrows():
                driver_data = {
                    'Year': event.get('EventDate').year if pd.notna(event.get('EventDate')) else session.date.year,
                    'Race_Num': event.get('RoundNumber', 0),
                    'Circuit': event.get('EventName', 'Unknown').replace(' Grand Prix', ''),
                    'Date': event.get('EventDate', session.date),
                    'Driver': result.get('Abbreviation', result.get('DriverNumber', 'UNK')),
                    'Full_Name': result.get('FullName', 'Unknown Driver'),
                    'Team': result.get('TeamName', 'Unknown Team'),
                    'Grid': result.get('GridPosition', np.nan),
                    'Position': result.get('Position', np.nan),
                    'Finish_Pos_Clean': self._clean_position(result.get('Position')),
                    'Status': result.get('Status', 'Unknown'),
                    'Time': result.get('Time', None),
                    'Laps': result.get('Laps', 0),
                    'Points': result.get('Points', 0)
                }
                
                # Add additional telemetry if available
                try:
                    driver_laps = session.laps.pick_driver(result.get('Abbreviation'))
                    if not driver_laps.empty:
                        driver_data['Fastest_Lap'] = driver_laps['LapTime'].min()
                        driver_data['Avg_Lap_Time'] = driver_laps['LapTime'].mean()
                except:
                    driver_data['Fastest_Lap'] = None
                    driver_data['Avg_Lap_Time'] = None
                
                race_data.append(driver_data)
            
            return pd.DataFrame(race_data)
            
        except Exception as e:
            logger.warning(f"Failed to extract race data: {e}")
            return pd.DataFrame()
    
    def _extract_qualifying_data(self, session: Any, event: pd.Series) -> pd.DataFrame:
        """Extract qualifying results data from FastF1 session."""
        try:
            session.load()
            results = session.results
            
            if results.empty:
                return pd.DataFrame()
            
            # Extract qualifying data
            quali_data = []
            
            for idx, result in results.iterrows():
                driver_data = {
                    'Year': event.get('EventDate').year if pd.notna(event.get('EventDate')) else session.date.year,
                    'Race_Num': event.get('RoundNumber', 0),
                    'Circuit': event.get('EventName', 'Unknown').replace(' Grand Prix', ''),
                    'Date': event.get('EventDate', session.date),
                    'Driver': result.get('Abbreviation', result.get('DriverNumber', 'UNK')),
                    'Full_Name': result.get('FullName', 'Unknown Driver'),
                    'Team': result.get('TeamName', 'Unknown Team'),
                    'Position': result.get('Position', np.nan),
                    'Q1': result.get('Q1', None),
                    'Q2': result.get('Q2', None),
                    'Q3': result.get('Q3', None)
                }
                
                quali_data.append(driver_data)
            
            return pd.DataFrame(quali_data)
            
        except Exception as e:
            logger.warning(f"Failed to extract qualifying data: {e}")
            return pd.DataFrame()
    
    def _clean_position(self, position: Any) -> Optional[int]:
        """Clean and convert position to integer."""
        if pd.isna(position):
            return None
        
        try:
            # Handle various position formats
            if isinstance(position, str):
                if position.upper() in ['DNF', 'DNS', 'DSQ']:
                    return None
                # Extract numeric part
                import re
                numbers = re.findall(r'\d+', position)
                if numbers:
                    return int(numbers[0])
            else:
                return int(position)
        except:
            return None
        
        return None
    
    def _save_data(self, data: pd.DataFrame, filepath: str, append: bool = False):
        """Save data to CSV with backup and append capability (now default is overwrite via fetch_all_f1_data)."""
        if data is None or data.empty:
            logger.info(f"No data provided to save to {filepath}. Skipping save.")
            return
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            if append and os.path.exists(filepath):
                try:
                    existing_df = pd.read_csv(filepath)
                    if not existing_df.empty:
                        # Ensure columns are compatible for concatenation
                        # Align columns, fill missing with NaN, then concat
                        # This is a simplified alignment; more robust might be needed for highly disparate schemas
                        cols_existing = set(existing_df.columns)
                        cols_new = set(data.columns)
                        all_cols = list(cols_existing.union(cols_new))
                        
                        existing_df = existing_df.reindex(columns=all_cols)
                        data_reindexed = data.reindex(columns=all_cols)

                        combined_df = pd.concat([existing_df, data_reindexed], ignore_index=True)
                    else:
                        combined_df = data.copy() # existing_df was empty
                except pd.errors.EmptyDataError:
                    logger.warning(f"Existing file {filepath} was empty. Will overwrite with new data.")
                    combined_df = data.copy()
                except Exception as e:
                    logger.error(f"Error reading existing data from {filepath} for append: {e}. Overwriting with new data as a fallback.")
                    combined_df = data.copy() # Fallback to just using new data if read fails catastrophically
                
                # Define subset for duplicate removal based on data type
                # These should be the unique identifiers for a race/quali entry for a driver
                subset_cols = ['Year', 'Race_Num', 'Driver'] 
                # Ensure these columns actually exist before trying to drop duplicates
                if all(col in combined_df.columns for col in subset_cols):
                    combined_df.drop_duplicates(subset=subset_cols, keep='last', inplace=True)
                else:
                    logger.warning(f"Cannot drop duplicates for {filepath} as one or more key columns {subset_cols} are missing.")
                
                combined_df.to_csv(filepath, index=False, encoding='utf-8')
                logger.info(f"Saved/Appended {len(data)} new records to {filepath}. Total unique records now: {len(combined_df)}")
            else: # Overwrite mode (or file doesn't exist yet for append)
                # Create backup if file exists and we are overwriting
                if os.path.exists(filepath):
                    backup_filepath = filepath + ".bak"
                    if os.path.exists(backup_filepath):
                        try:
                            os.remove(backup_filepath)
                        except OSError as e:
                            logger.warning(f"Could not remove old backup {backup_filepath}: {e}")
                    try:
                        os.rename(filepath, backup_filepath)
                        logger.info(f"Created backup: {backup_filepath}")
                    except OSError as e:
                        logger.warning(f"Could not create backup {backup_filepath} from {filepath}: {e}")

                data.to_csv(filepath, index=False, encoding='utf-8')
                logger.info(f"Saved {len(data)} records to {filepath} (overwrite mode).")

        except Exception as e:
            logger.error(f"Failed to save data to {filepath}: {e}", exc_info=True)
    
    def _generate_data_quality_report(self, final_races_df: Optional[pd.DataFrame] = None, 
                                      final_quali_df: Optional[pd.DataFrame] = None):
        """Generate data quality report based on the provided or loaded dataframes."""
        logger.info("Generating data quality report...")
        report = {
            "timestamp": datetime.now().isoformat(),
            "failed_sessions_count": len(self.failed_sessions),
            "failed_sessions_details": list(self.failed_sessions),
            "data_metrics": {}
        }

        data_sources = {
            "races": final_races_df if final_races_df is not None else (self.existing_races_df if hasattr(self, 'existing_races_df') else pd.DataFrame()),
            "qualifying": final_quali_df if final_quali_df is not None else (self.existing_quali_df if hasattr(self, 'existing_quali_df') else pd.DataFrame())
        }

        for name, df in data_sources.items():
            if df is not None and not df.empty:
                metrics = {
                    "total_records": len(df),
                    "columns": list(df.columns),
                    "missing_values_per_column": df.isnull().sum().to_dict(),
                    "unique_drivers": df['Driver'].nunique() if 'Driver' in df.columns else 0,
                    "unique_circuits": df['Circuit'].nunique() if 'Circuit' in df.columns else 0,
                    "min_year": int(df['Year'].min()) if 'Year' in df.columns and not df['Year'].empty else None,
                    "max_year": int(df['Year'].max()) if 'Year' in df.columns and not df['Year'].empty else None,
                }
                report["data_metrics"][name] = metrics
            else:
                report["data_metrics"][name] = {"status": "No data loaded or provided."}

        self.data_quality_metrics = report
        # Optionally save to a file or log extensively
        logger.info(f"Data Quality Report: {report}") 

class F1DataLoader:
    """
    Advanced F1 data loading system with intelligent preprocessing and validation.
    """
    
    def __init__(self):
        self.data_cache = {}
        self.validation_errors = []
        
    def load_all_data(self, validate_data: bool = True) -> Optional[Tuple]:
        """
        Load all F1 data for model training and prediction.
        
        Args:
            validate_data: Whether to perform data validation
            
        Returns:
            Tuple of (hist_races, hist_quali, curr_races, curr_quali, upcoming_info, latest_quali)
        """
        logger.info("Loading F1 data...")
        
        try:
            # Load historical race data
            hist_races = self._load_historical_races()
            if hist_races is None or hist_races.empty:
                logger.warning("No historical race data available")
                hist_races = pd.DataFrame()
            
            # Load historical qualifying data
            hist_quali = self._load_historical_qualifying()
            if hist_quali is None or hist_quali.empty:
                logger.warning("No historical qualifying data available")
                hist_quali = pd.DataFrame()
            
            # Load current season data
            curr_races, curr_quali = self._load_current_season_data()
            
            # Generate upcoming race information
            upcoming_info = self._generate_upcoming_race_info()
            
            # Get latest qualifying results
            latest_quali = self._get_latest_qualifying()
            
            # Data validation
            if validate_data:
                self._validate_loaded_data(hist_races, hist_quali, curr_races, curr_quali)
            
            logger.info("Data loading completed successfully")
            return hist_races, hist_quali, curr_races, curr_quali, upcoming_info, latest_quali
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return None
    
    def _load_historical_races(self) -> pd.DataFrame:
        """Load historical race data."""
        try:
            if os.path.exists(config.RACES_CSV_PATH):
                races_df = pd.read_csv(config.RACES_CSV_PATH)
                races_df['Date'] = pd.to_datetime(races_df['Date'])
                
                # Filter to valid years
                current_year = datetime.now().year
                races_df = races_df[races_df['Year'] < current_year]
                
                logger.info(f"Loaded {len(races_df)} historical race records")
                return races_df
            else:
                logger.warning(f"Historical race data file not found: {config.RACES_CSV_PATH}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to load historical race data: {e}")
            return pd.DataFrame()
    
    def _load_historical_qualifying(self) -> pd.DataFrame:
        """Load historical qualifying data."""
        try:
            if os.path.exists(config.QUALI_CSV_PATH):
                quali_df = pd.read_csv(config.QUALI_CSV_PATH)
                quali_df['Date'] = pd.to_datetime(quali_df['Date'])
                
                # Filter to valid years
                current_year = datetime.now().year
                quali_df = quali_df[quali_df['Year'] < current_year]
                
                logger.info(f"Loaded {len(quali_df)} historical qualifying records")
                return quali_df
            else:
                logger.warning(f"Historical qualifying data file not found: {config.QUALI_CSV_PATH}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to load historical qualifying data: {e}")
            return pd.DataFrame()
    
    def _load_current_season_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load current season data."""
        try:
            current_year = config.CURRENT_SEASON
            
            # Try to load current season data from cache or fetch fresh
            curr_races = pd.DataFrame()
            curr_quali = pd.DataFrame()
            
            # Check if current season data exists in main files
            if os.path.exists(config.RACES_CSV_PATH):
                all_races = pd.read_csv(config.RACES_CSV_PATH)
                curr_races = all_races[all_races['Year'] == current_year]
            
            if os.path.exists(config.QUALI_CSV_PATH):
                all_quali = pd.read_csv(config.QUALI_CSV_PATH)
                curr_quali = all_quali[all_quali['Year'] == current_year]
            
            logger.info(f"Loaded {len(curr_races)} current season race records")
            logger.info(f"Loaded {len(curr_quali)} current season qualifying records")
            
            return curr_races, curr_quali
            
        except Exception as e:
            logger.error(f"Failed to load current season data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def _generate_upcoming_race_info(self) -> pd.DataFrame:
        """Generate upcoming race information."""
        try:
            current_year = config.CURRENT_SEASON
            
            # Get current season schedule
            schedule = fastf1.get_event_schedule(current_year)
            
            if schedule.empty:
                logger.warning(f"No schedule found for {current_year}")
                return pd.DataFrame()
            
            # Find upcoming races
            now = datetime.now()
            upcoming_events = schedule[schedule['EventDate'] > now]
            
            if upcoming_events.empty:
                logger.info("No upcoming races found")
                return pd.DataFrame()
            
            # Get the next upcoming race
            next_event = upcoming_events.iloc[0]
            
            # Generate driver list for upcoming race (based on most recent participation)
            driver_list = self._get_current_driver_lineup()
            
            upcoming_info = []
            for driver_info in driver_list:
                upcoming_info.append({
                    'Year': current_year,
                    'Race_Num': next_event.get('RoundNumber', 0),
                    'Circuit': next_event.get('EventName', 'Unknown').replace(' Grand Prix', ''),
                    'Race_Name': next_event.get('EventName', 'Unknown'),
                    'Date': next_event.get('EventDate'),
                    'Driver': driver_info['Driver'],
                    'Full_Name': driver_info.get('Full_Name', 'Unknown'),
                    'Team': driver_info.get('Team', 'Unknown'),
                    'Country': next_event.get('Country', 'Unknown')
                })
            
            upcoming_df = pd.DataFrame(upcoming_info)
            logger.info(f"Generated upcoming race info for {len(upcoming_df)} drivers")
            
            return upcoming_df
            
        except Exception as e:
            logger.error(f"Failed to generate upcoming race info: {e}")
            return pd.DataFrame()
    
    def _get_current_driver_lineup(self) -> List[Dict]:
        """Get current season driver lineup."""
        try:
            # Try to get from most recent race data
            if os.path.exists(config.RACES_CSV_PATH):
                all_races = pd.read_csv(config.RACES_CSV_PATH)
                current_races = all_races[all_races['Year'] == config.CURRENT_SEASON]
                
                if not current_races.empty:
                    # Get latest race
                    latest_race = current_races[current_races['Race_Num'] == current_races['Race_Num'].max()]
                    
                    drivers = []
                    for _, row in latest_race.iterrows():
                        drivers.append({
                            'Driver': row['Driver'],
                            'Full_Name': row.get('Full_Name', 'Unknown'),
                            'Team': row['Team']
                        })
                    
                    return drivers
            
            # Fallback: Generate standard grid
            standard_drivers = [
                {'Driver': 'VER', 'Full_Name': 'Max Verstappen', 'Team': 'Red Bull Racing Honda RBPT'},
                {'Driver': 'PER', 'Full_Name': 'Sergio Perez', 'Team': 'Red Bull Racing Honda RBPT'},
                {'Driver': 'LEC', 'Full_Name': 'Charles Leclerc', 'Team': 'Ferrari'},
                {'Driver': 'SAI', 'Full_Name': 'Carlos Sainz', 'Team': 'Ferrari'},
                {'Driver': 'HAM', 'Full_Name': 'Lewis Hamilton', 'Team': 'Mercedes'},
                {'Driver': 'RUS', 'Full_Name': 'George Russell', 'Team': 'Mercedes'},
                {'Driver': 'NOR', 'Full_Name': 'Lando Norris', 'Team': 'McLaren Mercedes'},
                {'Driver': 'PIA', 'Full_Name': 'Oscar Piastri', 'Team': 'McLaren Mercedes'},
                {'Driver': 'ALO', 'Full_Name': 'Fernando Alonso', 'Team': 'Aston Martin Aramco Mercedes'},
                {'Driver': 'STR', 'Full_Name': 'Lance Stroll', 'Team': 'Aston Martin Aramco Mercedes'},
                {'Driver': 'OCO', 'Full_Name': 'Esteban Ocon', 'Team': 'Alpine Renault'},
                {'Driver': 'GAS', 'Full_Name': 'Pierre Gasly', 'Team': 'Alpine Renault'},
                {'Driver': 'ALB', 'Full_Name': 'Alexander Albon', 'Team': 'Williams Mercedes'},
                {'Driver': 'SAR', 'Full_Name': 'Logan Sargeant', 'Team': 'Williams Mercedes'},
                {'Driver': 'TSU', 'Full_Name': 'Yuki Tsunoda', 'Team': 'AlphaTauri Honda RBPT'},
                {'Driver': 'RIC', 'Full_Name': 'Daniel Ricciardo', 'Team': 'AlphaTauri Honda RBPT'},
                {'Driver': 'BOT', 'Full_Name': 'Valtteri Bottas', 'Team': 'Alfa Romeo Ferrari'},
                {'Driver': 'ZHO', 'Full_Name': 'Zhou Guanyu', 'Team': 'Alfa Romeo Ferrari'},
                {'Driver': 'MAG', 'Full_Name': 'Kevin Magnussen', 'Team': 'Haas Ferrari'},
                {'Driver': 'HUL', 'Full_Name': 'Nico Hulkenberg', 'Team': 'Haas Ferrari'}
            ]
            
            logger.info("Using standard driver lineup")
            return standard_drivers
            
        except Exception as e:
            logger.warning(f"Failed to get current driver lineup: {e}")
            return []
    
    def _get_latest_qualifying(self) -> Optional[pd.DataFrame]:
        """Get latest qualifying results if available."""
        try:
            if os.path.exists(config.QUALI_CSV_PATH):
                all_quali = pd.read_csv(config.QUALI_CSV_PATH)
                current_quali = all_quali[all_quali['Year'] == config.CURRENT_SEASON]
                
                if not current_quali.empty:
                    latest_quali = current_quali[current_quali['Race_Num'] == current_quali['Race_Num'].max()]
                    logger.info(f"Found latest qualifying with {len(latest_quali)} drivers")
                    return latest_quali
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get latest qualifying: {e}")
            return None
    
    def _validate_loaded_data(self, hist_races: pd.DataFrame, hist_quali: pd.DataFrame,
                            curr_races: pd.DataFrame, curr_quali: pd.DataFrame):
        """Validate loaded data quality."""
        self.validation_errors = []
        
        # Check historical data completeness
        if hist_races.empty:
            self.validation_errors.append("No historical race data available")
        
        if hist_quali.empty:
            self.validation_errors.append("No historical qualifying data available")
        
        # Check data consistency
        if not hist_races.empty and not hist_quali.empty:
            # Check if race and qualifying data align
            race_events = set(zip(hist_races['Year'], hist_races['Race_Num']))
            quali_events = set(zip(hist_quali['Year'], hist_quali['Race_Num']))
            
            missing_quali = race_events - quali_events
            if missing_quali:
                self.validation_errors.append(f"Missing qualifying data for {len(missing_quali)} race events")
        
        # Check for required columns
        required_race_cols = ['Year', 'Race_Num', 'Circuit', 'Driver', 'Team', 'Position']
        required_quali_cols = ['Year', 'Race_Num', 'Circuit', 'Driver', 'Team', 'Position']
        
        if not hist_races.empty:
            missing_race_cols = set(required_race_cols) - set(hist_races.columns)
            if missing_race_cols:
                self.validation_errors.append(f"Missing race columns: {missing_race_cols}")
        
        if not hist_quali.empty:
            missing_quali_cols = set(required_quali_cols) - set(hist_quali.columns)
            if missing_quali_cols:
                self.validation_errors.append(f"Missing qualifying columns: {missing_quali_cols}")
        
        # Log validation results
        if self.validation_errors:
            logger.warning("Data validation issues found:")
            for error in self.validation_errors:
                logger.warning(f"  - {error}")
        else:
            logger.info("Data validation passed")

def fetch_f1_data(start_year: int = None, end_year: int = None, force_refresh: bool = False) -> bool:
    """Fetch F1 data using the data collector."""
    collector = F1DataCollector()
    return collector.fetch_all_f1_data(start_year, end_year, force_refresh)

def load_data() -> Optional[Tuple]:
    """Load all F1 data for the prediction system."""
    loader = F1DataLoader()
    return loader.load_all_data()

# Initialize FastF1 cache on import
logger.info(f"FastF1 cache enabled at: {config.FF1_CACHE_PATH}")
