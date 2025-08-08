# f1_predictor/prediction.py - Unified Prediction Module

import pandas as pd
import logging
import joblib
import os
from typing import Optional
import numpy as np

from .config import config
from .data_loader import F1DataLoader
from .feature_engineering_pipeline import FeatureEngineeringPipeline

logger = logging.getLogger(__name__)

class F1Predictor:
    """Handles the end-to-end prediction process for qualifying and race sessions."""

    def __init__(self):
        self.models_dir = config.get('paths.models_dir')
        self.qualifying_model = self._load_model('qualifying')
        self.race_model = self._load_model('race')
        self.data_loader = F1DataLoader()

    def _load_model(self, model_type: str) -> Optional[any]:
        model_path = os.path.join(self.models_dir, f'{model_type}_model.pkl')
        if not os.path.exists(model_path):
            logger.error(f"{model_type.capitalize()} model not found at {model_path}. Please train it first.")
            return None
        try:
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            return None

    def predict_qualifying(self, year: int, race_name: str) -> Optional[pd.DataFrame]:
        if not self.qualifying_model:
            return None

        upcoming_df = self._get_upcoming_race_df(year, race_name)
        if upcoming_df.empty:
            return None
        
        features_df = self._prepare_features(upcoming_df)
        
        features_used = self.qualifying_model.feature_names_in_
        predictions = self.qualifying_model.predict(features_df[features_used])
        
        results = self._format_results(upcoming_df, predictions, 'Quali_Pos')
        return results

    def predict_race(self, year: int, race_name: str) -> Optional[pd.DataFrame]:
        if not self.race_model:
            return None
        
        upcoming_df = self._get_upcoming_race_df(year, race_name)
        if upcoming_df.empty:
            return None

        # Check for and merge actual qualifying results
        latest_quali = self.data_loader._get_latest_qualifying()
        if latest_quali is not None and latest_quali.iloc[0]['Race_Name'] == race_name:
            logger.info("Actual qualifying results found. Incorporating into race prediction.")
            quali_results = latest_quali[['Driver', 'Position']].rename(columns={'Position': 'Quali_Pos'})
            upcoming_df = pd.merge(upcoming_df, quali_results, on='Driver', how='left')
        else:
            logger.info("No actual qualifying results found. Making pre-qualifying prediction.")
            quali_predictions = self.predict_qualifying(year, race_name)
            if quali_predictions is not None:
                upcoming_df = pd.merge(upcoming_df, quali_predictions[['Driver', 'Predicted_Quali_Pos']], on='Driver', how='left')
                upcoming_df.rename(columns={'Predicted_Quali_Pos': 'Quali_Pos'}, inplace=True)

        features_df = self._prepare_features(upcoming_df)

        features_used = self.race_model.feature_names_in_
        predictions = self.race_model.predict(features_df[features_used])

        results = self._format_results(upcoming_df, predictions, 'Race_Pos')
        return results

    def _simulate_future_prediction(self, year: int, race_name: str, session: str) -> Optional[pd.DataFrame]:
        """Simulates predictions for future years using historical data aggregation."""
        hist_races, hist_quali, _ = self.data_loader.load_all_data()
        if hist_races is None or hist_races.empty:
            logger.error("No historical data available for simulation.")
            return None
        
        # Aggregate historical performance for each driver
        hist_agg = hist_races.groupby('Driver').agg(
            Avg_Pos=('Position', 'mean'),
            Avg_Quali_Pos=('Quali_Pos', 'mean'),
            Win_Rate=('Position', lambda x: (x == 1).mean()),
            Podium_Rate=('Position', lambda x: (x <= 3).mean()),
            DNF_Rate=('DNF', 'mean')
        ).reset_index()
        
        # Get driver lineup for the future race
        upcoming_df = self._get_upcoming_race_df(year, race_name)
        if upcoming_df.empty:
            return None
        
        # Merge aggregated stats with upcoming lineup
        simulated_df = pd.merge(upcoming_df, hist_agg, on='Driver', how='left')
        
        # Fill any missing stats with reasonable defaults
        simulated_df.fillna({
            'Avg_Pos': 10.5,
            'Avg_Quali_Pos': 10.5,
            'Win_Rate': 0.0,
            'Podium_Rate': 0.0,
            'DNF_Rate': 0.2
        }, inplace=True)
        
        # Simulate positions based on aggregated stats (simple ranking for now)
        if session == 'qualifying':
            simulated_df = simulated_df.sort_values('Avg_Quali_Pos')
        else:
            simulated_df = simulated_df.sort_values('Avg_Pos')
        
        simulated_df['Predicted_Pos'] = range(1, len(simulated_df) + 1)
        simulated_df['Prediction_Score'] = simulated_df['Win_Rate'] + simulated_df['Podium_Rate'] - simulated_df['DNF_Rate']
        
        # Format results
        results = simulated_df[['Driver', 'Team', 'Predicted_Pos', 'Prediction_Score']]
        results = results.sort_values('Predicted_Pos').reset_index(drop=True)
        return results

    def _get_upcoming_race_df(self, year: int, race_name: str) -> pd.DataFrame:
        """Gets the DataFrame for the specified upcoming race."""
        # Pass the year to get the correct schedule
        upcoming_races = self.data_loader._get_upcoming_race_info(year)
        if upcoming_races.empty:
            logger.error(f"No upcoming race information could be generated for {year}.")
            return pd.DataFrame()
        
        race_df = upcoming_races[(upcoming_races['Year'] == year) & (upcoming_races['Race_Name'] == race_name)]
        if race_df.empty:
            logger.error(f"Could not find upcoming race for {year} {race_name}.")
            return pd.DataFrame()
        return race_df
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares the full feature set for a given DataFrame."""
        # This method is now DEPRECATED and will be removed in a future refactoring.
        # The logic has been moved to _prepare_prediction_data.
        
        # Load historical data for feature engineering
        hist_races, hist_quali, _ = self.data_loader.load_all_data()
        if hist_races is None or hist_quali is None:
            logger.error("Could not load historical data for feature engineering.")
            return pd.DataFrame()

        combined_races = pd.concat([hist_races, df], ignore_index=True) if hist_races is not None else df
        pipeline = FeatureEngineeringPipeline(combined_races, hist_quali)
        all_features = pipeline.run()

        # Return only the rows corresponding to the upcoming race
        return all_features.tail(len(df))

    def _format_results(self, upcoming_df: pd.DataFrame, predictions: np.ndarray, prediction_col_name: str) -> pd.DataFrame:
        """Formats the raw predictions into a readable DataFrame."""
        results = upcoming_df[['Driver', 'Team']].copy()
        results['Prediction_Score'] = np.round(predictions, 4)

        results = results.sort_values('Prediction_Score').reset_index(drop=True)
        results['Predicted_Pos'] = results.index + 1
        
        # Reorder for clarity
        return results[['Predicted_Pos', 'Driver', 'Team', 'Prediction_Score']] 