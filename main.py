# main.py - F1 Prediction System 2.0

import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import F1 predictor modules
from f1_predictor import (
    data_loader,
    feature_engineering,
    model_training,
    prediction,
    config,
)

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class F1PredictionSystem:
    """Comprehensive F1 Prediction System with full weekend prediction capabilities."""
    
    def __init__(self):
        self.feature_engineer = feature_engineering.F1FeatureEngineer()
        self.model_trainer = model_training.F1ModelTrainer()
        self.predictor = prediction.F1Predictor()
        self.models = {}
        self.features = {}
        self.imputation_values = {}
        self.encoders = {}
        
    def run_complete_workflow(self, force_fetch: bool = False, race_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete F1 prediction workflow.
        
        Args:
            force_fetch: Force data fetching from FastF1
            race_name: Specific race to predict (optional)
            
        Returns:
            Dictionary with all prediction results
        """
        logger.info("=" * 60)
        logger.info("F1 PREDICTION SYSTEM 2.0 - COMPLETE WORKFLOW")
        logger.info("=" * 60)
        
        try:
            # 1. Data Collection and Loading
            logger.info("STEP 1: Data Collection and Loading")
            if not self._ensure_data_available(force_fetch):
                logger.error("Failed to ensure data availability. Aborting.")
                return {}
            
            data_dict = self._load_all_data()
            if not data_dict:
                logger.error("Failed to load data. Aborting.")
                return {}
            
            # 2. Feature Engineering
            logger.info("STEP 2: Comprehensive Feature Engineering")
            features_df = self._perform_feature_engineering(data_dict)
            if features_df.empty:
                logger.error("Feature engineering failed. Aborting.")
                return {}
            
            # 3. Model Training
            logger.info("STEP 3: Advanced Model Training")
            if not self._train_all_models(features_df):
                logger.error("Model training failed. Aborting.")
                return {}
            
            # 4. Weekend Predictions
            logger.info("STEP 4: Complete Weekend Prediction")
            weekend_results = self._predict_race_weekend(data_dict, race_name)
            
            # 5. Generate Reports
            logger.info("STEP 5: Generate Prediction Reports")
            self._generate_comprehensive_report(weekend_results)
            
            logger.info("COMPLETE WORKFLOW FINISHED SUCCESSFULLY!")
            return weekend_results
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return {}
    
    def run_live_update_workflow(self, actual_quali_results: Optional[str] = None) -> Dict[str, Any]:
        """
        Run live update workflow with actual qualifying results.
        
        Args:
            actual_quali_results: Path to CSV with actual qualifying results
            
        Returns:
            Updated race predictions
        """
        logger.info("LIVE UPDATE: Processing actual qualifying results...")
        
        try:
            # Load actual qualifying results
            if actual_quali_results and os.path.exists(actual_quali_results):
                actual_quali = pd.read_csv(actual_quali_results)
                logger.info(f"Loaded actual qualifying results: {len(actual_quali)} drivers")
            else:
                logger.error("No valid qualifying results provided.")
                return {}
            
            # Load data for context
            data_dict = self._load_all_data()
            if not data_dict:
                logger.error("Failed to load data for live update.")
                return {}
            
            # Update race predictions
            updated_race_predictions = self.predictor.update_predictions_with_actual_quali(
                race_model=self.models.get('race'),
                race_features=self.features.get('race', []),
                upcoming_info=data_dict['upcoming_info'],
                historical_data=data_dict['features_df'],
                encoders=self.encoders,
                race_imputation=self.imputation_values.get('race', {}),
                actual_quali=actual_quali
            )
            
            if updated_race_predictions is not None:
                logger.info("Live update completed successfully!")
                self._save_live_update_results(updated_race_predictions)
                return {'live_race_predictions': updated_race_predictions}
            else:
                logger.error("Live update failed.")
                return {}
                
        except Exception as e:
            logger.error(f"Live update failed: {e}")
            return {}
    
    def _ensure_data_available(self, force_fetch: bool) -> bool:
        """Ensure data is available, fetch if necessary."""
        need_fetch = force_fetch
        
        # Check if data files exist
        if not os.path.exists(config.RACES_CSV_PATH) or not os.path.exists(config.QUALI_CSV_PATH):
            logger.info("Local data files not found.")
            need_fetch = True
        
        if need_fetch:
            logger.info("Fetching F1 data using FastF1...")
            success = data_loader.fetch_f1_data()
            if not success:
                logger.error("Data fetching failed.")
                return False
            logger.info("Data fetching completed.")
        else:
            logger.info("Using existing local data files.")
        
        return True
    
    def _load_all_data(self) -> Dict[str, Any]:
        """Load all required data."""
        try:
            loaded_data = data_loader.load_data()
            if loaded_data is None:
                return {}
            
            hist_races, hist_quali, curr_races, curr_quali, upcoming_info, latest_quali = loaded_data
            
            logger.info(f"Data loaded successfully:")
            logger.info(f"  Historical races: {len(hist_races)}")
            logger.info(f"  Historical qualifying: {len(hist_quali)}")
            logger.info(f"  Current races: {len(curr_races)}")
            logger.info(f"  Current qualifying: {len(curr_quali)}")
            logger.info(f"  Upcoming race info: {len(upcoming_info)}")
            
            return {
                'hist_races': hist_races,
                'hist_quali': hist_quali,
                'curr_races': curr_races,
                'curr_quali': curr_quali,
                'upcoming_info': upcoming_info,
                'latest_quali': latest_quali
            }
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return {}
    
    def _perform_feature_engineering(self, data_dict: Dict[str, Any]) -> pd.DataFrame:
        """Perform comprehensive feature engineering."""
        try:
            features_df, encoders = self.feature_engineer.preprocess_and_engineer_features(
                data_dict['hist_races'],
                data_dict['hist_quali'],
                data_dict['curr_races'],
                data_dict['curr_quali']
            )
            
            self.encoders = encoders
            data_dict['features_df'] = features_df  # Store for later use
            
            logger.info(f"Feature engineering completed: {features_df.shape}")
            return features_df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return pd.DataFrame()
    
    def _train_all_models(self, features_df: pd.DataFrame) -> bool:
        """Train both qualifying and race models."""
        try:
            # Train qualifying model
            logger.info("Training qualifying model...")
            quali_model, quali_features, quali_imputation = self.model_trainer.train_model(
                features=features_df.copy(),
                target_column_name="Quali_Pos",
                model_type="qualifying"
            )
            
            if quali_model is None:
                logger.warning("Qualifying model training failed.")
            else:
                self.models['qualifying'] = quali_model
                self.features['qualifying'] = quali_features
                self.imputation_values['qualifying'] = quali_imputation
                logger.info("Qualifying model trained successfully.")
            
            # Train race model
            logger.info("Training race model...")
            race_model, race_features, race_imputation = self.model_trainer.train_model(
                features=features_df.copy(),
                target_column_name="Finish_Pos_Clean",
                model_type="race"
            )
            
            if race_model is None:
                logger.error("Race model training failed.")
                return False
            else:
                self.models['race'] = race_model
                self.features['race'] = race_features
                self.imputation_values['race'] = race_imputation
                logger.info("Race model trained successfully.")
            
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def _predict_race_weekend(self, data_dict: Dict[str, Any], race_name: Optional[str] = None) -> Dict[str, Any]:
        """Predict complete race weekend."""
        try:
            upcoming_info = data_dict['upcoming_info']
            
            if upcoming_info.empty:
                logger.warning("No upcoming race found for prediction.")
                return {}
            
            # Filter for specific race if provided
            if race_name:
                upcoming_info = upcoming_info[upcoming_info['Race_Name'].str.contains(race_name, case=False, na=False)]
                if upcoming_info.empty:
                    logger.warning(f"No upcoming race found matching '{race_name}'.")
                    return {}
            
            # Get the current upcoming race
            current_race = upcoming_info.iloc[0] if not upcoming_info.empty else None
            if current_race is not None:
                logger.info(f"Predicting for: {current_race.get('Race_Name', 'Unknown Race')} ({current_race.get('Year', 'Unknown Year')})")
            
            # Predict complete weekend
            weekend_results = self.predictor.predict_race_weekend(
                quali_model=self.models.get('qualifying'),
                race_model=self.models.get('race'),
                quali_features=self.features.get('qualifying', []),
                race_features=self.features.get('race', []),
                upcoming_info=upcoming_info,
                historical_data=data_dict['features_df'],
                encoders=self.encoders,
                quali_imputation=self.imputation_values.get('qualifying', {}),
                race_imputation=self.imputation_values.get('race', {})
            )
            
            return weekend_results
            
        except Exception as e:
            logger.error(f"Weekend prediction failed: {e}")
            return {}
    
    def _generate_comprehensive_report(self, weekend_results: Dict[str, Any]):
        """Generate comprehensive prediction report."""
        try:
            logger.info("\n" + "=" * 60)
            logger.info("COMPREHENSIVE F1 WEEKEND PREDICTION REPORT")
            logger.info("=" * 60)
            
            # Qualifying predictions
            if 'quali_predictions' in weekend_results:
                logger.info("\n🏁 QUALIFYING PREDICTIONS:")
                quali_results = weekend_results['quali_predictions']
                for i, (_, row) in enumerate(quali_results.head(10).iterrows()):
                    confidence = row.get('Quali_Confidence', 0)
                    confidence_emoji = "🔥" if confidence > 0.8 else "✅" if confidence > 0.6 else "⚠️"
                    logger.info(f"  {i+1:2d}. {row['Driver']:20s} ({row.get('Team', 'Unknown'):15s}) {confidence_emoji}")
            
            # Pre-qualifying race predictions
            if 'pre_quali_race_predictions' in weekend_results:
                logger.info("\n🏆 RACE PREDICTIONS (Pre-Qualifying):")
                race_results = weekend_results['pre_quali_race_predictions']
                for i, (_, row) in enumerate(race_results.head(10).iterrows()):
                    confidence = row.get('Race_Confidence', 0)
                    confidence_emoji = "🔥" if confidence > 0.8 else "✅" if confidence > 0.6 else "⚠️"
                    logger.info(f"  {i+1:2d}. {row['Driver']:20s} ({row.get('Team', 'Unknown'):15s}) {confidence_emoji}")
            
            # Post-qualifying race predictions
            if 'post_quali_race_predictions' in weekend_results:
                logger.info("\n🏆 RACE PREDICTIONS (Post-Qualifying):")
                race_results = weekend_results['post_quali_race_predictions']
                for i, (_, row) in enumerate(race_results.head(10).iterrows()):
                    confidence = row.get('Race_Confidence', 0)
                    confidence_emoji = "🔥" if confidence > 0.8 else "✅" if confidence > 0.6 else "⚠️"
                    logger.info(f"  {i+1:2d}. {row['Driver']:20s} ({row.get('Team', 'Unknown'):15s}) {confidence_emoji}")
            
            # Weekend summary
            if 'weekend_summary' in weekend_results:
                summary = weekend_results['weekend_summary']
                logger.info("\n📊 WEEKEND SUMMARY:")
                
                if summary.get('predicted_pole_sitter'):
                    pole = summary['predicted_pole_sitter']
                    logger.info(f"  🥇 Predicted Pole: {pole['driver']} ({pole['team']})")
                
                if summary.get('predicted_race_winner'):
                    winner = summary['predicted_race_winner']
                    logger.info(f"  🏆 Predicted Winner: {winner['driver']} ({winner['team']})")
                
                if summary.get('predicted_podium'):
                    logger.info("  🏅 Predicted Podium:")
                    for podium_spot in summary['predicted_podium']:
                        pos = podium_spot['position']
                        driver = podium_spot['driver']
                        team = podium_spot['team']
                        emoji = "🥇" if pos == 1 else "🥈" if pos == 2 else "🥉"
                        logger.info(f"     {emoji} P{pos}: {driver} ({team})")
            
            logger.info("\n" + "=" * 60)
            logger.info("Report generated successfully!")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
    
    def _save_live_update_results(self, results: pd.DataFrame):
        """Save live update results."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{config.PREDICTIONS_DIR}/live_update_{timestamp}.csv"
            results.to_csv(filename, index=False)
            logger.info(f"Live update results saved: {filename}")
        except Exception as e:
            logger.warning(f"Failed to save live update results: {e}")

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description='F1 Prediction System 2.0')
    parser.add_argument('--fetch', action='store_true', help='Force data fetching')
    parser.add_argument('--race', type=str, help='Specific race to predict')
    parser.add_argument('--live-update', type=str, help='Path to CSV with actual qualifying results')
    parser.add_argument('--mode', choices=['complete', 'live'], default='complete', 
                       help='Run mode: complete workflow or live update')
    
    args = parser.parse_args()
    
    # Initialize the prediction system
    f1_system = F1PredictionSystem()
    
    if args.mode == 'live' or args.live_update:
        # Live update mode
        results = f1_system.run_live_update_workflow(args.live_update)
    else:
        # Complete workflow mode
        results = f1_system.run_complete_workflow(
            force_fetch=args.fetch,
            race_name=args.race
        )
    
    if results:
        print("\n✅ F1 Prediction System completed successfully!")
        print(f"📁 Check {config.PREDICTIONS_DIR} for detailed results")
        print(f"📝 Check {config.LOG_FILE} for detailed logs")
    else:
        print("\n❌ F1 Prediction System failed!")
        print(f"📝 Check {config.LOG_FILE} for error details")
        sys.exit(1)

if __name__ == "__main__":
    main()
