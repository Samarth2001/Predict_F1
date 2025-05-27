# f1_predictor/prediction.py

import pandas as pd
import numpy as np
import logging
import joblib
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Feature engineering imports
from .feature_engineering import F1FeatureEngineer
from . import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class F1Predictor:
    """Comprehensive F1 prediction system with confidence scoring and scenario handling."""
    
    def __init__(self):
        self.feature_engineer = F1FeatureEngineer()
        self.confidence_scores = {}
        self.prediction_metadata = {}
        
    def predict_results(self,
                       model: Any,
                       features_used: List[str],
                       upcoming_info: pd.DataFrame,
                       historical_data: pd.DataFrame,
                       encoders: Dict,
                       imputation_values: Dict,
                       actual_quali: Optional[pd.DataFrame] = None,
                       model_type: str = "race") -> Optional[pd.DataFrame]:
        """
        Comprehensive prediction system for F1 results.
        
        Args:
            model: Trained model
            features_used: List of feature names used in training
            upcoming_info: DataFrame with upcoming race driver information
            historical_data: Historical race data for context
            encoders: Feature encoders from training
            imputation_values: Imputation values from training
            actual_quali: Actual qualifying results (if available)
            model_type: Type of prediction ('qualifying', 'race')
            
        Returns:
            DataFrame with predictions and confidence scores
        """
        logger.info(f"Starting {model_type} prediction...")
        
        if upcoming_info.empty:
            logger.error("No upcoming race information provided.")
            return None
            
        try:
            # Prepare prediction features
            prediction_features = self._prepare_prediction_features(
                upcoming_info, historical_data, encoders, imputation_values, actual_quali
            )
            
            if prediction_features.empty:
                logger.error("Failed to prepare prediction features.")
                return None
            
            # Make predictions
            predictions = self._make_predictions(model, prediction_features, features_used, model_type)
            
            if predictions is None:
                logger.error("Prediction failed.")
                return None
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                model, prediction_features, features_used, predictions, model_type
            )
            
            # Create results dataframe
            results = self._create_results_dataframe(
                upcoming_info, predictions, confidence_scores, model_type
            )
            
            # Add prediction metadata
            self._add_prediction_metadata(results, model_type, actual_quali is not None)
            
            # Save predictions if enabled
            if config.SAVE_PREDICTIONS:
                self._save_predictions(results, model_type)
            
            logger.info(f"{model_type} prediction completed successfully.")
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def predict_race_weekend(self,
                           quali_model: Any,
                           race_model: Any,
                           quali_features: List[str],
                           race_features: List[str],
                           upcoming_info: pd.DataFrame,
                           historical_data: pd.DataFrame,
                           encoders: Dict,
                           quali_imputation: Dict,
                           race_imputation: Dict) -> Dict[str, pd.DataFrame]:
        """
        Predict complete race weekend (qualifying + race scenarios).
        
        Returns:
            Dictionary with 'pre_quali_predictions', 'quali_predictions', 'post_quali_predictions'
        """
        logger.info("Starting complete race weekend prediction...")
        
        results = {}
        
        # 1. Pre-qualifying predictions (both qualifying and race)
        logger.info("Making pre-qualifying predictions...")
        
        # Predict qualifying results
        if quali_model is not None:
            quali_predictions = self.predict_results(
                model=quali_model,
                features_used=quali_features,
                upcoming_info=upcoming_info.copy(),
                historical_data=historical_data.copy(),
                encoders=encoders,
                imputation_values=quali_imputation,
                actual_quali=None,
                model_type="qualifying"
            )
            results['quali_predictions'] = quali_predictions
        
        # Predict race results without qualifying data
        if race_model is not None:
            pre_quali_race_predictions = self.predict_results(
                model=race_model,
                features_used=race_features,
                upcoming_info=upcoming_info.copy(),
                historical_data=historical_data.copy(),
                encoders=encoders,
                imputation_values=race_imputation,
                actual_quali=None,
                model_type="race_pre_quali"
            )
            results['pre_quali_race_predictions'] = pre_quali_race_predictions
        
        # 2. Post-qualifying predictions (if qualifying predictions available)
        if 'quali_predictions' in results and race_model is not None:
            logger.info("Making post-qualifying race predictions...")
            
            # Use predicted qualifying as input for race prediction
            predicted_quali = self._convert_predictions_to_quali_format(
                results['quali_predictions'], upcoming_info
            )
            
            post_quali_race_predictions = self.predict_results(
                model=race_model,
                features_used=race_features,
                upcoming_info=upcoming_info.copy(),
                historical_data=historical_data.copy(),
                encoders=encoders,
                imputation_values=race_imputation,
                actual_quali=predicted_quali,
                model_type="race_post_quali"
            )
            results['post_quali_race_predictions'] = post_quali_race_predictions
        
        # 3. Generate weekend summary
        results['weekend_summary'] = self._generate_weekend_summary(results)
        
        logger.info("Complete race weekend prediction completed.")
        return results
    
    def _prepare_prediction_features(self,
                                   upcoming_info: pd.DataFrame,
                                   historical_data: pd.DataFrame,
                                   encoders: Dict,
                                   imputation_values: Dict,
                                   actual_quali: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Prepare features for prediction."""
        logger.info("Preparing prediction features...")
        
        try:
            # Create prediction dataset
            prediction_data = upcoming_info.copy()
            
            # Add historical context features
            prediction_data = self._add_historical_context(prediction_data, historical_data)
            
            # Add qualifying information if available
            if actual_quali is not None:
                prediction_data = self._merge_qualifying_data(prediction_data, actual_quali)
            
            # Apply feature engineering
            prediction_features = self._engineer_prediction_features(prediction_data, historical_data, encoders)
            
            # Handle missing values
            prediction_features = self._handle_missing_values_prediction(prediction_features, imputation_values)
            
            return prediction_features
            
        except Exception as e:
            logger.error(f"Failed to prepare prediction features: {e}")
            return pd.DataFrame()
    
    def _add_historical_context(self, prediction_data: pd.DataFrame, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Add historical performance context to prediction data."""
        if historical_data.empty:
            return prediction_data
        
        # Add latest performance metrics for each driver
        latest_performance = {}
        
        for driver in prediction_data['Driver'].unique():
            driver_history = historical_data[historical_data['Driver'] == driver].sort_values(['Year', 'Race_Num'])
            
            if not driver_history.empty:
                latest = driver_history.iloc[-1]
                latest_performance[driver] = {
                    'Last_Finish_Pos': latest.get('Finish_Pos_Clean', config.DEFAULT_IMPUTATION_VALUES['Driver_Avg_Finish_Last_N']),
                    'Last_Quali_Pos': latest.get('Quali_Pos', config.DEFAULT_IMPUTATION_VALUES['Quali_Pos']),
                    'Driver_Avg_Finish_Last_5': latest.get('Driver_Avg_Finish_Last_5', config.DEFAULT_IMPUTATION_VALUES['Driver_Avg_Finish_Last_N']),
                    'Driver_Win_Rate': latest.get('Driver_Win_Rate', 0),
                    'Driver_Podium_Rate': latest.get('Driver_Podium_Rate', 0),
                    'Driver_DNF_Rate': latest.get('Driver_DNF_Rate', 0.1)
                }
        
        # Apply to prediction data
        for col, default_val in config.DEFAULT_IMPUTATION_VALUES.items():
            if col.startswith('Driver_'):
                prediction_data[col] = prediction_data['Driver'].map(
                    lambda x: latest_performance.get(x, {}).get(col, default_val)
                )
        
        return prediction_data
    
    def _merge_qualifying_data(self, prediction_data: pd.DataFrame, actual_quali: pd.DataFrame) -> pd.DataFrame:
        """Merge actual qualifying results into prediction data."""
        quali_merge = actual_quali[['Driver', 'Quali_Pos']].copy()
        quali_merge['Grid_Pos'] = quali_merge['Quali_Pos']  # Assume same unless penalties
        
        # Merge qualifying data
        prediction_data = pd.merge(prediction_data, quali_merge, on='Driver', how='left')
        
        return prediction_data
    
    def _engineer_prediction_features(self, prediction_data: pd.DataFrame, historical_data: pd.DataFrame, encoders: Dict) -> pd.DataFrame:
        """Apply feature engineering for predictions."""
        # Apply categorical encoders
        for feature, encoder in encoders.items():
            if feature in prediction_data.columns:
                try:
                    # Handle unseen categories
                    prediction_data[f'{feature}_Encoded'] = prediction_data[feature].apply(
                        lambda x: encoder.transform([str(x)])[0] if str(x) in encoder.classes_ 
                        else -1  # Default for unseen categories
                    )
                except Exception as e:
                    logger.warning(f"Failed to encode {feature}: {e}")
                    prediction_data[f'{feature}_Encoded'] = -1
        
        # Add circuit-specific features
        if 'Circuit' in prediction_data.columns:
            prediction_data['Circuit_Type'] = prediction_data['Circuit'].apply(
                lambda x: 1 if x in config.STREET_CIRCUITS else
                         2 if x in config.HIGH_SPEED_CIRCUITS else
                         3 if x in config.TECHNICAL_CIRCUITS else 0
            )
        
        # Add weather features (dummy for now)
        prediction_data['Weather_Temp'] = 25
        prediction_data['Weather_Humidity'] = 60
        prediction_data['Weather_Rain_Probability'] = 0.1
        prediction_data['Weather_Extreme'] = 0
        
        # Add season progress
        if 'Race_Num' in prediction_data.columns:
            prediction_data['Season_Progress'] = prediction_data['Race_Num'] / 24  # Typical season length
        
        return prediction_data
    
    def _handle_missing_values_prediction(self, prediction_features: pd.DataFrame, imputation_values: Dict) -> pd.DataFrame:
        """Handle missing values in prediction features."""
        # Apply training imputation values
        for col, value in imputation_values.items():
            if col in prediction_features.columns:
                prediction_features[col] = prediction_features[col].fillna(value)
        
        # Apply config default values
        for col, value in config.DEFAULT_IMPUTATION_VALUES.items():
            if col in prediction_features.columns:
                prediction_features[col] = prediction_features[col].fillna(value)
        
        # Fill remaining numeric columns with median or 0
        numeric_cols = prediction_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if prediction_features[col].isnull().sum() > 0:
                median_val = prediction_features[col].median()
                prediction_features[col] = prediction_features[col].fillna(median_val if pd.notna(median_val) else 0)
        
        return prediction_features
    
    def _make_predictions(self, model: Any, prediction_features: pd.DataFrame, features_used: List[str], model_type: str) -> Optional[np.ndarray]:
        """Make predictions using the trained model."""
        try:
            # Select only features used in training
            available_features = [f for f in features_used if f in prediction_features.columns]
            missing_features = set(features_used) - set(available_features)
            
            if missing_features:
                logger.warning(f"Missing features for prediction: {missing_features}")
                # Create missing features with default values
                for feature in missing_features:
                    prediction_features[feature] = config.DEFAULT_IMPUTATION_VALUES.get('Driver_Avg_Finish_Last_N', 0)
            
            # Prepare feature matrix
            X_pred = prediction_features[features_used].fillna(0)
            
            # Make predictions
            predictions = model.predict(X_pred)
            
            # Post-process predictions (round to reasonable position range)
            predictions = np.clip(predictions, 1, 20)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def _calculate_confidence_scores(self, 
                                   model: Any, 
                                   prediction_features: pd.DataFrame,
                                   features_used: List[str],
                                   predictions: np.ndarray,
                                   model_type: str) -> np.ndarray:
        """Calculate confidence scores for predictions."""
        try:
            confidence_scores = np.ones(len(predictions))  # Default confidence
            
            # Method 1: Use prediction variance for ensemble models
            if hasattr(model, 'estimators_'):
                individual_predictions = []
                for estimator in model.estimators_:
                    try:
                        X_pred = prediction_features[features_used].fillna(0)
                        pred = estimator.predict(X_pred)
                        individual_predictions.append(pred)
                    except:
                        continue
                
                if individual_predictions:
                    pred_array = np.array(individual_predictions)
                    pred_std = np.std(pred_array, axis=0)
                    # Convert standard deviation to confidence (lower std = higher confidence)
                    confidence_scores = 1 / (1 + pred_std)
            
            # Method 2: Feature completeness confidence
            feature_completeness = []
            for _, row in prediction_features.iterrows():
                available_features = sum(1 for f in features_used if pd.notna(row.get(f, np.nan)))
                completeness = available_features / len(features_used)
                feature_completeness.append(completeness)
            
            feature_confidence = np.array(feature_completeness)
            
            # Combine confidence measures
            final_confidence = (confidence_scores + feature_confidence) / 2
            
            return np.clip(final_confidence, 0.1, 1.0)
            
        except Exception as e:
            logger.warning(f"Failed to calculate confidence scores: {e}")
            return np.ones(len(predictions)) * 0.5  # Default medium confidence
    
    def _create_results_dataframe(self,
                                upcoming_info: pd.DataFrame,
                                predictions: np.ndarray,
                                confidence_scores: np.ndarray,
                                model_type: str) -> pd.DataFrame:
        """Create formatted results dataframe."""
        results = upcoming_info.copy()
        
        # Add predictions
        if model_type == "qualifying":
            results['Predicted_Quali_Pos'] = np.round(predictions).astype(int)
            results['Quali_Confidence'] = confidence_scores
        else:
            results['Predicted_Race_Pos'] = np.round(predictions).astype(int)
            results['Race_Confidence'] = confidence_scores
        
        # Sort by predictions
        sort_col = 'Predicted_Quali_Pos' if model_type == "qualifying" else 'Predicted_Race_Pos'
        results = results.sort_values(sort_col)
        
        # Add ranking
        results['Predicted_Rank'] = range(1, len(results) + 1)
        
        # Add confidence categories
        confidence_col = 'Quali_Confidence' if model_type == "qualifying" else 'Race_Confidence'
        results['Confidence_Category'] = results[confidence_col].apply(
            lambda x: 'High' if x >= config.HIGH_CONFIDENCE_THRESHOLD else
                     'Medium' if x >= config.MEDIUM_CONFIDENCE_THRESHOLD else 'Low'
        )
        
        return results
    
    def _add_prediction_metadata(self, results: pd.DataFrame, model_type: str, has_quali_data: bool):
        """Add metadata to predictions."""
        metadata = {
            'prediction_timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'has_qualifying_data': has_quali_data,
            'prediction_count': len(results),
            'high_confidence_count': len(results[results['Confidence_Category'] == 'High']),
            'medium_confidence_count': len(results[results['Confidence_Category'] == 'Medium']),
            'low_confidence_count': len(results[results['Confidence_Category'] == 'Low'])
        }
        
        self.prediction_metadata[model_type] = metadata
    
    def _convert_predictions_to_quali_format(self, quali_predictions: pd.DataFrame, upcoming_info: pd.DataFrame) -> pd.DataFrame:
        """Convert qualifying predictions to format usable for race prediction."""
        quali_format = upcoming_info[['Driver']].copy()
        quali_format['Quali_Pos'] = quali_predictions['Predicted_Quali_Pos']
        return quali_format
    
    def _generate_weekend_summary(self, results: Dict[str, pd.DataFrame]) -> Dict:
        """Generate comprehensive weekend prediction summary."""
        summary = {
            'predicted_pole_sitter': None,
            'predicted_race_winner': None,
            'predicted_podium': [],
            'confidence_analysis': {},
            'key_predictions': []
        }
        
        # Pole sitter
        if 'quali_predictions' in results:
            quali_results = results['quali_predictions']
            if not quali_results.empty:
                pole_sitter = quali_results.iloc[0]
                summary['predicted_pole_sitter'] = {
                    'driver': pole_sitter['Driver'],
                    'team': pole_sitter.get('Team', 'Unknown'),
                    'confidence': pole_sitter.get('Quali_Confidence', 0)
                }
        
        # Race winner and podium
        race_key = 'post_quali_race_predictions' if 'post_quali_race_predictions' in results else 'pre_quali_race_predictions'
        if race_key in results:
            race_results = results[race_key]
            if not race_results.empty:
                winner = race_results.iloc[0]
                summary['predicted_race_winner'] = {
                    'driver': winner['Driver'],
                    'team': winner.get('Team', 'Unknown'),
                    'confidence': winner.get('Race_Confidence', 0)
                }
                
                # Podium
                podium = race_results.head(3)
                summary['predicted_podium'] = [
                    {
                        'position': i + 1,
                        'driver': row['Driver'],
                        'team': row.get('Team', 'Unknown'),
                        'confidence': row.get('Race_Confidence', 0)
                    }
                    for i, (_, row) in enumerate(podium.iterrows())
                ]
        
        return summary
    
    def _save_predictions(self, results: pd.DataFrame, model_type: str):
        """Save predictions to file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{config.PREDICTIONS_DIR}/predictions_{model_type}_{timestamp}.csv"
            results.to_csv(filename, index=False)
            
            # Save metadata
            metadata_filename = f"{config.PREDICTIONS_DIR}/metadata_{model_type}_{timestamp}.json"
            with open(metadata_filename, 'w') as f:
                json.dump(self.prediction_metadata.get(model_type, {}), f, indent=2)
            
            logger.info(f"Predictions saved: {filename}")
            
        except Exception as e:
            logger.warning(f"Failed to save predictions: {e}")

# Backward compatibility function
def predict_results(model: Any,
                   features_used: List[str],
                   upcoming_info: pd.DataFrame,
                   historical_data: pd.DataFrame,
                   encoders: Dict,
                   imputation_values: Dict,
                   actual_quali: Optional[pd.DataFrame] = None,
                   model_type: str = "race") -> Optional[pd.DataFrame]:
    """Predict results function for backward compatibility."""
    predictor = F1Predictor()
    return predictor.predict_results(
        model, features_used, upcoming_info, historical_data, 
        encoders, imputation_values, actual_quali, model_type
    ) 