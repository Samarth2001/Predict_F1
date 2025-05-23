# example_usage.py - F1 Prediction System Usage Examples

"""
Example usage scripts for the F1 Prediction System 2.0

This file demonstrates various ways to use the F1 prediction system
including basic usage, advanced features, and custom configurations.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Import F1 predictor modules
from f1_predictor import (
    F1DataCollector,
    F1DataLoader, 
    F1FeatureEngineer,
    F1ModelTrainer,
    F1Predictor,
    F1ModelEvaluator,
    config
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_1_basic_prediction():
    """Example 1: Basic race prediction workflow."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Race Prediction Workflow")
    print("="*60)
    
    try:
        # 1. Load data
        print("Step 1: Loading F1 data...")
        loader = F1DataLoader()
        data = loader.load_all_data()
        
        if data is None:
            print("❌ No data available. Please run data collection first.")
            return
        
        hist_races, hist_quali, curr_races, curr_quali, upcoming_info, latest_quali = data
        
        # 2. Feature engineering
        print("Step 2: Engineering features...")
        engineer = F1FeatureEngineer()
        features_df = engineer.engineer_features(hist_races, hist_quali, curr_races, curr_quali)
        
        if features_df.empty:
            print("❌ Feature engineering failed.")
            return
        
        # 3. Train model
        print("Step 3: Training race prediction model...")
        trainer = F1ModelTrainer()
        model, feature_names, imputation_values = trainer.train_model(
            features=features_df,
            target_column_name="Position",  # or "Finish_Pos_Clean"
            model_type="race"
        )
        
        if model is None:
            print("❌ Model training failed.")
            return
        
        # 4. Make predictions
        print("Step 4: Making race predictions...")
        predictor = F1Predictor()
        
        if not upcoming_info.empty:
            predictions = predictor.predict_results(
                model=model,
                features_used=feature_names,
                upcoming_info=upcoming_info,
                historical_data=features_df,
                encoders=engineer.label_encoders,
                imputation_values=imputation_values,
                model_type="race"
            )
            
            if predictions is not None:
                print("\n🏁 RACE PREDICTIONS:")
                print(predictions[['Driver', 'Team', 'Predicted_Race_Pos', 'Race_Confidence']].head(10))
                print("✅ Basic prediction workflow completed!")
            else:
                print("❌ Prediction failed.")
        else:
            print("ℹ️ No upcoming races to predict.")
            
    except Exception as e:
        print(f"❌ Basic prediction workflow failed: {e}")

def example_2_complete_weekend_prediction():
    """Example 2: Complete weekend prediction (qualifying + race)."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Complete Weekend Prediction")
    print("="*60)
    
    try:
        # Load data
        loader = F1DataLoader()
        data = loader.load_all_data()
        
        if data is None:
            print("❌ No data available.")
            return
        
        hist_races, hist_quali, curr_races, curr_quali, upcoming_info, latest_quali = data
        
        # Feature engineering
        engineer = F1FeatureEngineer()
        features_df = engineer.engineer_features(hist_races, hist_quali, curr_races, curr_quali)
        
        # Train both models
        trainer = F1ModelTrainer()
        
        # Train qualifying model
        print("Training qualifying model...")
        quali_model, quali_features, quali_imputation = trainer.train_model(
            features=features_df.copy(),
            target_column_name="Quali_Pos",
            model_type="qualifying"
        )
        
        # Train race model
        print("Training race model...")
        race_model, race_features, race_imputation = trainer.train_model(
            features=features_df.copy(),
            target_column_name="Position",
            model_type="race"
        )
        
        if quali_model is None or race_model is None:
            print("❌ Model training failed.")
            return
        
        # Complete weekend prediction
        print("Making complete weekend predictions...")
        predictor = F1Predictor()
        
        if not upcoming_info.empty:
            weekend_results = predictor.predict_race_weekend(
                quali_model=quali_model,
                race_model=race_model,
                quali_features=quali_features,
                race_features=race_features,
                upcoming_info=upcoming_info,
                historical_data=features_df,
                encoders=engineer.label_encoders,
                quali_imputation=quali_imputation,
                race_imputation=race_imputation
            )
            
            # Display results
            if 'quali_predictions' in weekend_results:
                print("\n🏁 QUALIFYING PREDICTIONS:")
                quali_df = weekend_results['quali_predictions']
                print(quali_df[['Driver', 'Team', 'Predicted_Quali_Pos', 'Quali_Confidence']].head(10))
            
            if 'pre_quali_race_predictions' in weekend_results:
                print("\n🏆 RACE PREDICTIONS (Pre-Qualifying):")
                race_df = weekend_results['pre_quali_race_predictions']
                print(race_df[['Driver', 'Team', 'Predicted_Race_Pos', 'Race_Confidence']].head(10))
            
            if 'weekend_summary' in weekend_results:
                summary = weekend_results['weekend_summary']
                print("\n📊 WEEKEND SUMMARY:")
                if summary.get('predicted_pole_sitter'):
                    pole = summary['predicted_pole_sitter']
                    print(f"  🥇 Predicted Pole: {pole['driver']} ({pole['team']})")
                if summary.get('predicted_race_winner'):
                    winner = summary['predicted_race_winner']
                    print(f"  🏆 Predicted Winner: {winner['driver']} ({winner['team']})")
            
            print("✅ Complete weekend prediction completed!")
        else:
            print("ℹ️ No upcoming races to predict.")
            
    except Exception as e:
        print(f"❌ Weekend prediction failed: {e}")

def example_3_model_evaluation():
    """Example 3: Model evaluation and comparison."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Model Evaluation and Analysis")
    print("="*60)
    
    try:
        # Load and prepare data
        loader = F1DataLoader()
        data = loader.load_all_data()
        
        if data is None:
            print("❌ No data available.")
            return
        
        hist_races, hist_quali, curr_races, curr_quali, upcoming_info, latest_quali = data
        
        # Feature engineering
        engineer = F1FeatureEngineer()
        features_df = engineer.engineer_features(hist_races, hist_quali, curr_races, curr_quali)
        
        # Train multiple models for comparison
        trainer = F1ModelTrainer()
        
        # Train with different model types
        print("Training LightGBM model...")
        config.DEFAULT_MODEL_TYPE = 'lightgbm'
        lgbm_model, features_used, imputation = trainer.train_model(
            features=features_df.copy(),
            target_column_name="Position",
            model_type="race_lgbm"
        )
        
        print("Training Ensemble model...")
        config.DEFAULT_MODEL_TYPE = 'ensemble'
        ensemble_model, _, _ = trainer.train_model(
            features=features_df.copy(),
            target_column_name="Position", 
            model_type="race_ensemble"
        )
        
        # Evaluate models
        evaluator = F1ModelEvaluator()
        
        # Prepare test data
        test_data = features_df.dropna(subset=["Position"]).copy()
        if len(test_data) > 100:
            test_sample = test_data.sample(100, random_state=42)
            X_test = test_sample[features_used].fillna(0)
            y_test = test_sample["Position"]
            
            # Evaluate both models
            if lgbm_model is not None:
                lgbm_metrics = evaluator.evaluate_model_performance(
                    lgbm_model, X_test.values, y_test.values, "LightGBM"
                )
                print(f"\n📊 LightGBM Performance:")
                print(f"  MAE: {lgbm_metrics.get('mae', 0):.3f}")
                print(f"  Top-3 Accuracy: {lgbm_metrics.get('top3_accuracy', 0):.3f}")
                print(f"  Podium Precision: {lgbm_metrics.get('podium_precision', 0):.3f}")
            
            if ensemble_model is not None:
                ensemble_metrics = evaluator.evaluate_model_performance(
                    ensemble_model, X_test.values, y_test.values, "Ensemble"
                )
                print(f"\n📊 Ensemble Performance:")
                print(f"  MAE: {ensemble_metrics.get('mae', 0):.3f}")
                print(f"  Top-3 Accuracy: {ensemble_metrics.get('top3_accuracy', 0):.3f}")
                print(f"  Podium Precision: {ensemble_metrics.get('podium_precision', 0):.3f}")
            
            # Generate evaluation report
            print("\nGenerating evaluation report...")
            report = evaluator.generate_evaluation_report(save_plots=True)
            print("✅ Model evaluation completed!")
        else:
            print("❌ Insufficient test data for evaluation.")
            
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")

def example_4_custom_configuration():
    """Example 4: Using custom configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Configuration Example")
    print("="*60)
    
    try:
        # Backup original config
        original_lgbm_params = config.LGBM_PARAMS.copy()
        original_ensemble_weights = config.ENSEMBLE_WEIGHTS.copy()
        
        # Modify configuration for demonstration
        print("Applying custom configuration...")
        
        # Custom LightGBM parameters for faster training
        config.LGBM_PARAMS.update({
            'n_estimators': 100,  # Reduced for faster training
            'learning_rate': 0.1,  # Higher for faster convergence
            'max_depth': 6,  # Reduced complexity
            'verbose': -1
        })
        
        # Custom ensemble weights
        config.ENSEMBLE_WEIGHTS = {
            'lightgbm': 0.5,
            'xgboost': 0.3,
            'random_forest': 0.2
        }
        
        # Custom feature selection
        config.MAX_FEATURES_SELECTION = 20  # Use only top 20 features
        
        print("Custom configuration applied:")
        print(f"  LightGBM estimators: {config.LGBM_PARAMS['n_estimators']}")
        print(f"  Ensemble weights: {config.ENSEMBLE_WEIGHTS}")
        print(f"  Max features: {config.MAX_FEATURES_SELECTION}")
        
        # Run training with custom config
        loader = F1DataLoader()
        data = loader.load_all_data()
        
        if data is not None:
            hist_races, hist_quali, curr_races, curr_quali, upcoming_info, latest_quali = data
            
            engineer = F1FeatureEngineer()
            features_df = engineer.engineer_features(hist_races, hist_quali, curr_races, curr_quali)
            
            trainer = F1ModelTrainer()
            print("Training with custom configuration...")
            model, features_used, imputation = trainer.train_model(
                features=features_df,
                target_column_name="Position",
                model_type="custom_race"
            )
            
            if model is not None:
                print("✅ Custom configuration training completed!")
            else:
                print("❌ Custom training failed.")
        
        # Restore original configuration
        config.LGBM_PARAMS = original_lgbm_params
        config.ENSEMBLE_WEIGHTS = original_ensemble_weights
        print("Original configuration restored.")
        
    except Exception as e:
        print(f"❌ Custom configuration example failed: {e}")

def example_5_data_collection():
    """Example 5: Data collection and updating."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Data Collection Example")
    print("="*60)
    
    try:
        # Initialize data collector
        collector = F1DataCollector()
        
        # Collect data for specific years (small range for demo)
        print("Collecting F1 data for recent years...")
        print("⚠️ This may take several minutes...")
        
        # Collect just the last 2 years for demonstration
        current_year = datetime.now().year
        success = collector.fetch_all_f1_data(
            start_year=current_year - 2,
            end_year=current_year,
            force_refresh=False
        )
        
        if success:
            print("✅ Data collection completed!")
            
            # Verify data was saved
            import os
            if os.path.exists(config.RACES_CSV_PATH):
                races_df = pd.read_csv(config.RACES_CSV_PATH)
                print(f"📊 Race records collected: {len(races_df)}")
            
            if os.path.exists(config.QUALI_CSV_PATH):
                quali_df = pd.read_csv(config.QUALI_CSV_PATH)
                print(f"📊 Qualifying records collected: {len(quali_df)}")
        else:
            print("❌ Data collection failed.")
            
    except Exception as e:
        print(f"❌ Data collection example failed: {e}")

def main():
    """Run all examples."""
    print("🏎️ F1 PREDICTION SYSTEM 2.0 - USAGE EXAMPLES")
    print("=" * 60)
    
    # Check if data exists
    import os
    has_data = (os.path.exists(config.RACES_CSV_PATH) and 
                os.path.exists(config.QUALI_CSV_PATH))
    
    if not has_data:
        print("⚠️ No F1 data found. Running data collection example first...")
        example_5_data_collection()
        print("\n" + "ℹ️ For full functionality, please run: python main.py --fetch")
        print("ℹ️ The following examples will run with limited/dummy data.\n")
    
    # Run examples
    try:
        example_1_basic_prediction()
        example_2_complete_weekend_prediction()
        example_3_model_evaluation()
        example_4_custom_configuration()
        
        if not has_data:
            example_5_data_collection()
        
    except KeyboardInterrupt:
        print("\n⏹️ Examples interrupted by user.")
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")
    
    print("\n" + "="*60)
    print("🎉 F1 Prediction System Examples Completed!")
    print("For full system usage, run: python main.py --fetch")
    print("="*60)

if __name__ == "__main__":
    main() 