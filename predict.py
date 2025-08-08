# predict.py - New Unified Prediction Workflow

import argparse
import logging
import pandas as pd
import sys
import os

from f1_predictor.config import config
from f1_predictor.data_loader import F1DataCollector, F1DataLoader
from f1_predictor.feature_engineering_pipeline import FeatureEngineeringPipeline
from f1_predictor.model_training import F1ModelTrainer
from f1_predictor.prediction import F1Predictor

# --- Basic Setup ---
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
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


def setup_arg_parser():
    """Set up the command-line argument parser."""
    parser = argparse.ArgumentParser(description="F1 Prediction System - Unified Workflow")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- Fetch Data Command ---
    fetch_parser = subparsers.add_parser('fetch-data', help='Fetch data from APIs.')
    fetch_parser.add_argument('--force', action='store_true', help='Force refetching all data, ignoring cache.')
    
    # --- Train Command ---
    train_parser = subparsers.add_parser('train', help='Train prediction models.')
    train_parser.add_argument('--model-type', type=str, default='all', choices=['qualifying', 'race', 'all'], help='Which model to train.')

    # --- Predict Command ---
    predict_parser = subparsers.add_parser('predict', help='Make predictions for a race weekend.')
    predict_parser.add_argument('--year', type=int, required=True, help='The year of the race.')
    predict_parser.add_argument('--race', type=str, required=True, help='The name of the race (e.g., "Italian Grand Prix").')
    predict_parser.add_argument('--session', type=str, required=True, choices=['qualifying', 'race'], help='The session to predict.')

    return parser


def handle_fetch_data(args):
    """Handle the data fetching process."""
    logger.info("Starting data fetching process...")
    collector = F1DataCollector()
    if not collector.fetch_all_f1_data(force_refresh=args.force):
        logger.error("Data fetching failed.")
        sys.exit(1)
    logger.info("Data fetching completed successfully.")


def handle_train(args):
    """Handle the model training process."""
    logger.info(f"Starting model training for: {args.model_type} model(s)...")
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
    """Handle the prediction process."""
    logger.info(f"Making prediction for {args.year} {args.race}, session: {args.session}.")
    
    predictor = F1Predictor()
    
    if args.session == 'qualifying':
        predictions = predictor.predict_qualifying(args.year, args.race)
    elif args.session == 'race':
        predictions = predictor.predict_race(args.year, args.race)
    else:
        logger.error(f"Unknown session type: {args.session}")
        predictions = None

    if predictions is not None and not predictions.empty:
        logger.info("Prediction successful. Results:")
        print(predictions.head(10))
        # Save results
        output_dir = config.get('paths.predictions_dir')
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{args.year}_{args.race.replace(' ', '_')}_{args.session}_predictions.csv"
        predictions.to_csv(os.path.join(output_dir, filename), index=False)
        logger.info(f"Results saved to {filename} in {output_dir}")
    else:
        logger.error("Prediction failed.")


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