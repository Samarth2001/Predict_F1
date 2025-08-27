import argparse
import logging
import sys
import os
from datetime import datetime
from typing import List, Optional
import pandas as pd

from f1_predictor.config import config
from f1_predictor.utils import set_global_seeds

def setup_logging():
    """Set up logging for the application."""
    log_level = config.get('general.log_level', 'INFO').upper()
    log_file = os.path.join(config.get('paths.logs_dir'), 'f1_predictions.log')
    os.makedirs(config.get('paths.logs_dir'), exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    return logging.getLogger(__name__)

logger = setup_logging()

try:
    seed_value = int(config.get('general.random_seed', config.get('general.random_state', 42)))
    set_global_seeds(seed_value)
    logger.info(f"Seeds initialized: {seed_value}")
except Exception:
    logger.warning("Failed to initialize global seeds.")


def setup_arg_parser():
    """Set up the command-line argument parser."""
    parser = argparse.ArgumentParser(description="F1 Prediction System - Unified Workflow")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

                                
    fetch_parser = subparsers.add_parser('fetch-data', help='Fetch data from APIs.')
    fetch_parser.add_argument('--force', action='store_true', help='Force refetching all data, ignoring cache.')
    
                           
    train_parser = subparsers.add_parser('train', help='Train prediction models.')
    train_parser.add_argument('--model-type', type=str, default='all', choices=['qualifying', 'race', 'all'], help='Which model to train.')

                                                
    predict_parser = subparsers.add_parser('predict', help='Make predictions for a race weekend or run utilities.')
    predict_subparsers = predict_parser.add_subparsers(dest='predict_command', required=False)

    predict_parser.add_argument('--year', type=int, required=False, help='The year of the race. Defaults to current year if --season-current is set or if omitted with --race next.')
    predict_parser.add_argument('--season-current', action='store_true', help='Use the current calendar year if --year is not provided.')
    predict_parser.add_argument('--race', type=str, required=False, help='Race name (e.g., "Italian Grand Prix") or "next" for upcoming event.')
    predict_parser.add_argument('--session', type=str, choices=['qualifying', 'race'], help='Session to predict. Defaults to race.', default='race')
    predict_parser.add_argument('--mode', type=str, choices=['pre-weekend', 'pre-quali', 'post-quali', 'live', 'auto'], default='auto', help='Race prediction mode.')
    predict_parser.add_argument('--season', type=str, choices=['single', 'all'], default='single', help='Run for a single event or the entire season.')
    predict_parser.add_argument('--rounds', type=str, default=None, help='Rounds to run (e.g., "1-10" or "1,3,5-7"). Overrides --season when provided.')
    predict_parser.add_argument('--interval', type=int, default=300, help='Refresh interval (seconds) for live/watch mode.')

    watch_parser = predict_subparsers.add_parser('watch', help='Watch mode: auto-refresh race predictions until actual qualifying is available.')
    watch_parser.add_argument('--year', type=int, required=False, help='Year. Defaults to current with --season-current or when omitted.')
    watch_parser.add_argument('--season-current', action='store_true', help='Use the current calendar year if --year is not provided.')
    watch_parser.add_argument('--race', type=str, required=False, default='next', help='Race name or "next" (default).')
    watch_parser.add_argument('--interval', type=int, default=300, help='Refresh interval (seconds). Default 300.')

    sim_parser = predict_subparsers.add_parser('simulate', help='Simulate predictions for a future season or custom lineup.')
    sim_parser.add_argument('--year', type=int, required=True, help='Target season year (can be future).')
    sim_parser.add_argument('--race', type=str, required=True, help='Race name (e.g., "Italian Grand Prix").')
    sim_parser.add_argument('--session', type=str, choices=['qualifying', 'race'], default='race', help='Session to simulate.')
    sim_parser.add_argument('--lineup', type=str, required=False, help='Path to CSV with custom lineup (columns at least Driver, Team).')

    list_parser = predict_subparsers.add_parser('list-schedule', help='List official FastF1 event names for a season (useful to copy exact strings).')
    list_parser.add_argument('--year', type=int, required=False, help='Season year to list. Defaults to current year when omitted or with --season-current.')
    list_parser.add_argument('--season-current', action='store_true', help='Use the current calendar year if --year is not provided.')

    return parser


def handle_fetch_data(args):
    """Handle the data fetching process."""
    logger.info("Starting data fetching process...")
    from f1_predictor.data_loader import F1DataCollector
    collector = F1DataCollector()
    if not collector.fetch_all_f1_data(force_refresh=args.force):
        logger.error("Data fetching failed.")
        sys.exit(1)
    logger.info("Data fetching completed successfully.")


def handle_train(args):
    """Handle the model training process."""
    logger.info(f"Starting model training for: {args.model_type} model(s)...")
    from f1_predictor.data_loader import F1DataLoader
    from f1_predictor.feature_engineering_pipeline import FeatureEngineeringPipeline
    from f1_predictor.model_training import F1ModelTrainer
    loader = F1DataLoader()
    data = loader.load_all_data()
    if data is None or data[0].empty:
        logger.error("Could not load data for training. Run 'fetch-data' first.")
        sys.exit(1)
    
    hist_races, hist_quali = data
    pipeline = FeatureEngineeringPipeline(hist_races, hist_quali)
    features_df = pipeline.run()

    if features_df.empty:
        logger.error("Feature engineering resulted in no data. Aborting training.")
        sys.exit(1)

    trainer = F1ModelTrainer()
    if args.model_type in ['qualifying', 'all']:
        trainer.train_model(features_df, target_column_name="Quali_Pos", model_type="qualifying")
    
    if args.model_type in ['race', 'all']:
        trainer.train_model(features_df, target_column_name="Position", model_type="race")
    
    logger.info("Model training finished.")


def handle_predict(args):
    """Handle the prediction process with subcommands and batch support."""
    from f1_predictor.prediction import F1Predictor
    predictor = F1Predictor()

    def _resolve_year(year_val: Optional[int], season_current: bool) -> int:
        if year_val is not None:
            return int(year_val)
                                                                                    
        return int(datetime.now().year)

    def _resolve_race(race_val: Optional[str]) -> str:
        if not race_val or str(race_val).strip() == "":
            return "next"
        return str(race_val)

    def _parse_rounds(spec: Optional[str]) -> Optional[List[int]]:
        if not spec:
            return None
        parts = [p.strip() for p in str(spec).split(',') if p.strip()]
        out: List[int] = []
        for part in parts:
            if '-' in part:
                start_s, end_s = part.split('-', 1)
                try:
                    start_i, end_i = int(start_s), int(end_s)
                except Exception:
                    continue
                out.extend(list(range(min(start_i, end_i), max(start_i, end_i) + 1)))
            else:
                try:
                    out.append(int(part))
                except Exception:
                    continue
                               
        return sorted(list(dict.fromkeys(out)))

    def _iter_events_for_year(year: int, rounds: Optional[List[int]] = None) -> List[str]:
        try:
            import fastf1
            schedule = fastf1.get_event_schedule(year, include_testing=False).copy()
            schedule.sort_values("RoundNumber", inplace=True)
            if rounds:
                schedule = schedule[schedule["RoundNumber"].isin(rounds)]
            return [str(x) for x in schedule["EventName"].tolist()]
        except Exception as e:
            logger.error(f"Failed to load schedule for {year}: {e}")
            return []

                          
    if getattr(args, 'predict_command', None) == 'simulate':
        year = int(args.year)
        race = str(args.race)
        session = str(args.session)
        lineup = getattr(args, 'lineup', None)
        logger.info(f"Simulating {session} for {year} {race}...")
        res = predictor.simulate(year, race, session=session, lineup_csv=lineup)
        if res is None or res.empty:
            logger.error("Simulation failed or returned no results.")
            return
        print(res.head(20))
        return

                               
    if getattr(args, 'predict_command', None) == 'list-schedule':
        year = _resolve_year(getattr(args, 'year', None), getattr(args, 'season_current', False))
        try:
            import fastf1
            schedule = fastf1.get_event_schedule(year, include_testing=False).copy()
            if schedule.empty:
                logger.error(f"No schedule found for {year}.")
                return
            df = schedule[['RoundNumber', 'EventName', 'EventDate']].copy()
            df['EventDate'] = pd.to_datetime(df['EventDate'], errors='coerce', utc=True).dt.tz_convert(None)
            df = df.sort_values('RoundNumber')
            print(df.to_string(index=False))
        except Exception as e:
            logger.error(f"Failed to load schedule for {year}: {e}")
        return

                       
    if getattr(args, 'predict_command', None) == 'watch':
        year = _resolve_year(getattr(args, 'year', None), getattr(args, 'season_current', False))
        race = _resolve_race(getattr(args, 'race', 'next'))
        interval = int(getattr(args, 'interval', 300))
        logger.info(f"Watch mode: {year} {race} every {interval}s until actual quali is available.")
        predictor.live_update_race_predictions(year, race, refresh_seconds=interval, stop_on_quali=True)
        return

                                             
    year = _resolve_year(getattr(args, 'year', None), getattr(args, 'season_current', False))
    race = _resolve_race(getattr(args, 'race', None))
    session = str(getattr(args, 'session', 'race'))
    mode = str(getattr(args, 'mode', 'auto'))
    mode_internal = mode.replace('-', '_')
    rounds = _parse_rounds(getattr(args, 'rounds', None))
    season = str(getattr(args, 'season', 'single'))

                     
    races_to_run: List[str] = []
    if rounds is not None:
        races_to_run = _iter_events_for_year(year, rounds)
    elif season == 'all':
        races_to_run = _iter_events_for_year(year, None)
    else:
        races_to_run = [race]

    def _run_single(y: int, rname: str) -> Optional['pd.DataFrame']:
        if session == 'qualifying':
            logger.info(f"Predicting qualifying for {y} {rname}...")
            return predictor.predict_qualifying(y, rname, scenario='qualifying')
        if mode_internal == 'live':
            interval = int(getattr(args, 'interval', 300))
            logger.info(f"Live mode: refreshing race predictions for {y} {rname} every {interval}s.")
            predictor.live_update_race_predictions(y, rname, refresh_seconds=interval, stop_on_quali=True)
            return None
        logger.info(f"Predicting race for {y} {rname} with mode={mode}...")
        return predictor.predict_race(y, rname, mode=mode_internal)

    any_success = False
    for rname in races_to_run:
        res = _run_single(year, rname)
        if res is not None and hasattr(res, 'empty') and not res.empty:
            print(res.head(20))
            any_success = True
    if not any_success and mode_internal != 'live':
        logger.error("No predictions produced.")


def main():
    """Main function to orchestrate the workflow."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    if args.command == 'fetch-data':
        handle_fetch_data(args)
    elif args.command == 'train':
        handle_train(args)
    elif args.command == 'predict':
        handle_predict(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main() 