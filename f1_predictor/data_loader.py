# f1_predictor/data_loader.py

import pandas as pd
import numpy as np
import fastf1 as ff1
from fastf1.ergast import Ergast
import os
import time
import warnings

# Import from config within the package
from . import config

warnings.filterwarnings('ignore')

# --- FastF1 Setup ---
try:
    ff1.Cache.enable_cache(config.CACHE_DIR)
    print(f"FastF1 cache enabled at: {config.CACHE_DIR}")
except Exception as e:
    print(f"Error enabling FastF1 cache: {e}. Check permissions or path.")

# --- Data Fetching ---

def fetch_f1_data(start_year=config.START_YEAR, end_year=config.END_YEAR):
    """Fetches historical F1 qualifying and race data using FastF1 and Ergast."""
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)

    ergast = Ergast()
    all_races_data = []
    all_quali_data = []

    print(f"Fetching F1 data from {start_year} to {end_year}...")

    for year in range(start_year, end_year + 1):
        print(f"\nFetching data for {year} season...")
        try:
            schedule = ff1.get_event_schedule(year, include_testing=False)
            print(f"Found {len(schedule)} events for {year}.")

            for round_num in schedule['RoundNumber']:
                try:
                    print(f"  Fetching Round {round_num}/{len(schedule)}...")
                    event = ff1.get_event(year, round_num)
                    event_name = event['EventName']
                    event_location = event['Location']
                    print(f"    Event: {event_name}")

                    # --- Fetch Qualifying Results ---
                    try:
                        quali_session = ff1.get_session(year, round_num, 'Q')
                        # Load only necessary data
                        quali_session.load(laps=True, telemetry=False, weather=False, messages=False)

                        if quali_session.laps is not None and not quali_session.laps.empty:
                            quali_results_df = quali_session.laps.loc[quali_session.laps.groupby('Driver')['LapTime'].idxmin()]
                            quali_results_df = quali_results_df[['Driver', 'Team', 'LapTime']]
                            # Handle potential NaT LapTimes before calculating seconds
                            quali_results_df = quali_results_df.dropna(subset=['LapTime'])
                            if not quali_results_df.empty:
                                quali_results_df['Quali_Time_Seconds'] = quali_results_df['LapTime'].dt.total_seconds()
                                quali_results_df = quali_results_df.sort_values('Quali_Time_Seconds').reset_index(drop=True) # Use drop=True
                                quali_results_df['Quali_Pos'] = quali_results_df.index + 1

                                quali_results_df['Year'] = year
                                quali_results_df['Race_Num'] = round_num
                                quali_results_df['Circuit'] = event_location
                                quali_results_df['RaceName'] = event_name

                                quali_to_append = quali_results_df[['Year', 'Race_Num', 'RaceName', 'Circuit', 'Driver', 'Team', 'Quali_Pos', 'Quali_Time_Seconds']]
                                all_quali_data.append(quali_to_append)
                                print(f"    Qualifying data fetched for {event_name}.")
                            else:
                                print(f"    Qualifying lap data found but empty after dropping invalid times for {year} Round {round_num} ({event_name}).")
                        else:
                             print(f"    No qualifying lap data found for {year} Round {round_num} ({event_name}).")

                    except Exception as q_err:
                         print(f"    WARNING: Could not fetch/process qualifying data for {year} Round {round_num} ({event_name}): {q_err}")

                    # --- Fetch Race Results (using Ergast) ---
                    try:
                        # Ergast results are usually more stable for final positions
                        race_results_ergast = ergast.get_race_results(season=year, round=round_num).content[0]
                        race_results_df = pd.DataFrame({
                            'Year': year, 'Race_Num': round_num, 'RaceName': event_name,
                            'Circuit': event_location, 'Driver': race_results_ergast['driverId'],
                            'Team': race_results_ergast['constructorId'], 'Grid_Pos': race_results_ergast['grid'],
                            'Finish_Pos': race_results_ergast['position'], 'Status': race_results_ergast['status'],
                            'Points': race_results_ergast['points'], 'Laps': race_results_ergast['laps'],
                        })

                        race_results_df['Grid_Pos'] = pd.to_numeric(race_results_df['Grid_Pos'], errors='coerce').fillna(config.DEFAULT_IMPUTATION_VALUE)
                        race_results_df['Finish_Pos'] = pd.to_numeric(race_results_df['Finish_Pos'], errors='coerce').fillna(config.DEFAULT_IMPUTATION_VALUE)
                        all_races_data.append(race_results_df)
                        print(f"    Race data fetched for {event_name}.")

                    except IndexError:
                         print(f"    WARNING: No race results content found via Ergast for {year} Round {round_num} ({event_name}). Race might not have happened or data unavailable.")
                    except Exception as r_err:
                        print(f"    WARNING: Could not fetch race data for {year} Round {round_num} ({event_name}): {r_err}")

                    time.sleep(config.FETCH_DELAY)

                except ff1.exceptions.FastF1Error as ff1_err:
                     # Handle errors like SessionNotAvailableError gracefully
                     print(f"  FastF1Error processing event for {year} Round {round_num}: {ff1_err}. Skipping.")
                     continue
                except Exception as event_err:
                    print(f"  ERROR: Could not process event for {year} Round {round_num}: {event_err}")
                    continue

        except Exception as year_err:
            print(f"ERROR: Could not fetch schedule or process year {year}: {year_err}")
            continue

    if not all_races_data: # Only require race data to proceed, quali is optional
        print("\nERROR: No race data fetched. Check FastF1 setup, network connection, and date ranges.")
        return False

    final_races_df = pd.concat(all_races_data, ignore_index=True)
    final_races_df.to_csv(config.RACES_CSV_PATH, index=False)
    print(f"\nFetched race data saved to: {config.RACES_CSV_PATH}")

    if all_quali_data:
        final_quali_df = pd.concat(all_quali_data, ignore_index=True)
        final_quali_df.to_csv(config.QUALI_CSV_PATH, index=False)
        print(f"Fetched qualifying data saved to: {config.QUALI_CSV_PATH}")
    else:
        print("No qualifying data was successfully fetched or processed.")
        # Create empty file if it doesn't exist, or handle absence in load_data
        if not os.path.exists(config.QUALI_CSV_PATH):
             pd.DataFrame().to_csv(config.QUALI_CSV_PATH, index=False)


    return True

# --- Data Loading (from saved CSVs) ---

def load_data():
    """Loads fetched F1 data from CSV files and identifies upcoming race."""
    print("\nLoading data from CSV files...")
    # Require race data, quali data is optional
    if not os.path.exists(config.RACES_CSV_PATH):
         print(f"Error: Race data file not found ({config.RACES_CSV_PATH}). Please fetch data first.")
         return None

    try:
        races_df = pd.read_csv(config.RACES_CSV_PATH)
        # Load quali data if file exists, otherwise create empty DataFrame
        if os.path.exists(config.QUALI_CSV_PATH):
             quali_df = pd.read_csv(config.QUALI_CSV_PATH)
        else:
             print(f"Warning: Qualifying data file not found ({config.QUALI_CSV_PATH}). Proceeding without it.")
             quali_df = pd.DataFrame()


        current_year = 0
        if not races_df.empty:
             current_year = races_df['Year'].max()
        else:
             print("Error: Race data file is empty.")
             return None # Cannot proceed without race data
        print(f"Identifying current year as: {current_year}")


        last_race_num = 0
        current_year_races = races_df[races_df['Year'] == current_year]
        if not current_year_races.empty:
            last_race_num = current_year_races['Race_Num'].max()
        print(f"Last completed race number in loaded data for {current_year}: {last_race_num}")

        curr_races = current_year_races[current_year_races['Race_Num'] <= last_race_num].copy()
        hist_races = races_df[races_df['Year'] < current_year].copy()

        curr_quali = pd.DataFrame()
        hist_quali = pd.DataFrame()
        if not quali_df.empty:
             curr_quali = quali_df[(quali_df['Year'] == current_year) & (quali_df['Race_Num'] <= last_race_num)].copy()
             hist_quali = quali_df[quali_df['Year'] < current_year].copy()


        # --- Identify Upcoming Race Info ---
        ergast = Ergast()
        upcoming_info = pd.DataFrame()
        latest_quali = pd.DataFrame(columns=['Driver', 'Quali_Pos']) # Default empty

        try:
            full_schedule_current_year = ergast.get_race_schedule(season=current_year)
            # Convert round column to numeric for comparison if needed, handle potential errors
            full_schedule_current_year['round'] = pd.to_numeric(full_schedule_current_year['round'], errors='coerce')
            full_schedule_current_year = full_schedule_current_year.dropna(subset=['round']) # Drop rows where round couldn't be converted
            full_schedule_current_year['round'] = full_schedule_current_year['round'].astype(int)

            next_race_num = last_race_num + 1

            # Check if next race number exceeds the maximum round number in the schedule
            max_round_in_schedule = full_schedule_current_year['round'].max() if not full_schedule_current_year.empty else 0

            if next_race_num > max_round_in_schedule:
                 print(f"Calculated next race number ({next_race_num}) exceeds max round in schedule ({max_round_in_schedule}). Season might be complete or schedule mismatch.")
            else:
                # Find the next race details using integer comparison
                next_race_filter = full_schedule_current_year[full_schedule_current_year['round'] == next_race_num]

                # ******** ERROR FIX START ********
                if next_race_filter.empty:
                    # Handle the case where no details are found for the next round
                    print(f"Error: Could not find details for Round {next_race_num} in the fetched schedule for {current_year}.")
                    print("This might happen if the race was cancelled, postponed, or Ergast data is lagging.")
                    print("Schedule returned by Ergast (showing rounds):")
                    print(full_schedule_current_year[['round', 'raceName', 'circuitId']].to_string()) # Print relevant columns for debugging
                    # upcoming_info remains empty
                else:
                    # Proceed only if details were found
                    next_race_details = next_race_filter.iloc[0]
                    upcoming_circuit = next_race_details['circuitId'] # Or circuitName
                    upcoming_race_name = next_race_details['raceName']
                    print(f"\nNext race identified: Round {next_race_num} - {upcoming_race_name} at {upcoming_circuit}")

                    # Get likely drivers/teams (use latest available race participants)
                    latest_drivers_teams = pd.DataFrame(columns=['Driver', 'Team']) # Default empty
                    if not curr_races.empty and last_race_num > 0:
                         latest_drivers_teams = curr_races[curr_races['Race_Num'] == last_race_num][['Driver', 'Team']].drop_duplicates()
                    elif not hist_races.empty: # Fallback if no current season data yet
                         last_hist_year = hist_races['Year'].max()
                         last_hist_race_num = hist_races[hist_races['Year']==last_hist_year]['Race_Num'].max()
                         latest_drivers_teams = hist_races[(hist_races['Year'] == last_hist_year) & (hist_races['Race_Num'] == last_hist_race_num)][['Driver', 'Team']].drop_duplicates()

                    if not latest_drivers_teams.empty:
                        upcoming_info = pd.DataFrame({
                            'Year': [current_year] * len(latest_drivers_teams),
                            'Race_Num': [next_race_num] * len(latest_drivers_teams),
                            'Circuit': [upcoming_circuit] * len(latest_drivers_teams),
                            'Driver': latest_drivers_teams['Driver'],
                            'Team': latest_drivers_teams['Team']
                        })

                        # --- Attempt to Fetch Latest Qualifying Results ---
                        print(f"\nAttempting to fetch actual qualifying results for Round {next_race_num}...")
                        try:
                            latest_quali_session = ff1.get_session(current_year, next_race_num, 'Q')
                            latest_quali_session.load(laps=True, telemetry=False, weather=False, messages=False)

                            if latest_quali_session.laps is not None and not latest_quali_session.laps.empty:
                                latest_quali_laps = latest_quali_session.laps.loc[latest_quali_session.laps.groupby('Driver')['LapTime'].idxmin()]
                                latest_quali_laps = latest_quali_laps.dropna(subset=['LapTime']) # Drop NaT times
                                if not latest_quali_laps.empty:
                                    latest_quali_laps = latest_quali_laps[['Driver', 'LapTime']].sort_values('LapTime').reset_index(drop=True)
                                    latest_quali_laps['Quali_Pos'] = latest_quali_laps.index + 1
                                    latest_quali = latest_quali_laps[['Driver', 'Quali_Pos']].copy()
                                    print("Successfully fetched actual qualifying results for the upcoming race.")
                                else:
                                     print("Actual qualifying session found, but lap data empty after cleaning.")
                            else:
                                print("Actual qualifying session found, but no lap data available yet.")

                        except ff1.exceptions.FastF1Error as ff1_err:
                             print(f"Could not fetch actual qualifying session for Round {next_race_num} (FastF1 Error): {ff1_err}")
                        except Exception as e:
                            print(f"Could not fetch actual qualifying results for Round {next_race_num} (Other Error): {e}")
                            print("Race prediction will proceed without actual qualifying data.")
                    else:
                         print("Warning: Could not determine driver/team list for upcoming race. Cannot create upcoming_info.")
                         # upcoming_info remains empty

                # ******** ERROR FIX END ********

        except Exception as schedule_err:
             print(f"An error occurred while fetching or processing the race schedule for {current_year}: {schedule_err}")
             # Allow function to return empty upcoming_info

        print("Data loading process finished.")
        return hist_races, hist_quali, curr_races, curr_quali, upcoming_info, latest_quali

    except FileNotFoundError:
        print(f"Error: Race data file not found ({config.RACES_CSV_PATH}). Please fetch data first.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        # Optionally re-raise the exception if debugging is needed: raise e
        return None
