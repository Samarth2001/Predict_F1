"""
Complete workflow for validating predictions and improving the model.
Run this after making predictions for a race that has now completed.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure 'src' and 'scripts' are importable when running from repo root
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SCRIPTS = _ROOT / "scripts"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from validate_predictions import PredictionValidator
from improve_model import ModelImprover

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_complete_workflow(
    year: int,
    race_name: str,
    prediction_file: str,
    retrain: bool = False
):
    """
    Complete workflow: validate predictions, analyze errors, suggest improvements,
    and optionally retrain the model.
    """
    
    print("\n" + "=" * 80)
    print("F1 PREDICTION VALIDATION & IMPROVEMENT WORKFLOW")
    print("=" * 80)
    print(f"Race: {year} {race_name}")
    print(f"Prediction File: {prediction_file}")
    print("=" * 80 + "\n")
    
    os.makedirs("validation", exist_ok=True)
    
    validator = PredictionValidator()
    
    logger.info("STEP 1: Loading predictions...")
    predictions = validator.load_predictions(prediction_file)
    if predictions is None:
        logger.error("Failed to load predictions. Exiting.")
        return False
    
    logger.info("STEP 2: Fetching actual race results...")
    actual_results = validator.fetch_actual_race_results(year, race_name)
    if actual_results is None:
        logger.error("Failed to fetch actual results. Exiting.")
        return False
    
    logger.info("STEP 3: Comparing predictions with actual results...")
    merged, metrics = validator.compare_predictions(predictions, actual_results)
    
    logger.info("STEP 4: Generating validation report...")
    validator.generate_report(merged, metrics, race_name, year)
    
    logger.info("STEP 5: Generating improvement suggestions...")
    improvements = validator.suggest_improvements(metrics, merged)
    try:
        print(improvements)
    except UnicodeEncodeError:
        print(improvements.encode('ascii', 'replace').decode('ascii'))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    improvement_file = os.path.join(
        "validation",
        f"improvement_suggestions_{year}_{race_name.replace(' ', '_')}_{timestamp}.txt"
    )
    with open(improvement_file, 'w', encoding='utf-8') as f:
        f.write(improvements)
    logger.info(f"Suggestions saved to: {improvement_file}")
    
    logger.info("STEP 6: Analyzing model weak areas...")
    improver = ModelImprover()
    weak_areas = improver.identify_weak_areas(merged)
    feature_suggestions = improver.suggest_feature_improvements(weak_areas)
    
    analysis = improver.analyze_validation_results()
    improver.generate_improvement_report(
        analysis,
        weak_areas,
        feature_suggestions
    )
    
    if retrain:
        logger.info("STEP 7: Retraining model with improvements...")
        logger.info("NOTE: This step requires loading full historical data...")
        
        try:
            from f1_predictor.data_loader import F1DataCollector
            import pandas as pd
            
            collector = F1DataCollector()
            
            races_file = os.path.join("data", "raw", "all_races_data.csv")
            quali_file = os.path.join("data", "raw", "all_quali_data.csv")
            
            if not os.path.exists(races_file):
                logger.error(f"Races data not found at {races_file}")
                return False
            
            races_df = pd.read_csv(races_file)
            quali_df = pd.read_csv(quali_file) if os.path.exists(quali_file) else pd.DataFrame()
            
            logger.info(f"Loaded {len(races_df)} race records")
            
            new_model, new_metrics = improver.retrain_with_improvements(
                races_df, 
                quali_df,
                model_type="race"
            )
            
            old_metrics_file = os.path.join("artifacts", "models", "race_model.metadata.json")
            if os.path.exists(old_metrics_file):
                import json
                with open(old_metrics_file, 'r') as f:
                    old_metrics = json.load(f)
                
                comparison = improver.compare_model_versions(old_metrics, new_metrics)
                try:
                    print(comparison)
                except UnicodeEncodeError:
                    print(comparison.encode('ascii', 'replace').decode('ascii'))
                
                comparison_file = os.path.join(
                    "validation",
                    f"model_comparison_{timestamp}.txt"
                )
                with open(comparison_file, 'w', encoding='utf-8') as f:
                    f.write(comparison)
                logger.info(f"Model comparison saved to: {comparison_file}")
            
            logger.info("Model retraining complete!")
            
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
            logger.info("Skipping retraining step.")
    
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  - Validation report in validation/")
    print("  - Visualization plots in validation/")
    print("  - Improvement suggestions in validation/")
    print("  - Detailed comparison CSV in validation/")
    
    if retrain:
        print("  - Model comparison report in validation/")
    
    print("\nNext Steps:")
    print("  1. Review validation report and visualizations")
    print("  2. Analyze improvement suggestions")
    print("  3. Implement recommended features in feature_engineering_pipeline.py")
    print("  4. Retrain model with new features")
    print("  5. Re-run this workflow to validate improvements")
    print("=" * 80 + "\n")
    
    return True


def main():
    """Main entry point with command-line argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="Validate F1 predictions and generate improvement suggestions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate predictions for Singapore GP 2025
  python scripts/workflow_validate_improve.py --year 2025 --race "Singapore Grand Prix" --file artifacts/predictions/race_predictions_2025_Singapore_Grand_Prix_20251009_195213.csv
  
  # Validate and retrain model
  python scripts/workflow_validate_improve.py --year 2025 --race "Singapore Grand Prix" --file artifacts/predictions/race_predictions_2025_Singapore_Grand_Prix_20251009_195213.csv --retrain
        """
    )
    
    parser.add_argument(
        '--year',
        type=int,
        required=True,
        help='Race year (e.g., 2025)'
    )
    
    parser.add_argument(
        '--race',
        type=str,
        required=True,
        help='Race name (e.g., "Singapore Grand Prix")'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        required=True,
        help='Path to prediction CSV file'
    )
    
    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Retrain model with improvements after validation'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        logger.error(f"Prediction file not found: {args.file}")
        sys.exit(1)
    
    success = run_complete_workflow(
        year=args.year,
        race_name=args.race,
        prediction_file=args.file,
        retrain=args.retrain
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

