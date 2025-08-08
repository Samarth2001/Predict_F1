# f1_predictor/feature_engineering_pipeline.py

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any

from .config import config

logger = logging.getLogger(__name__)

class BaseFeatureEngineer(ABC):
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @abstractmethod
    def engineer_features(self) -> pd.DataFrame:
        pass

class DataPreparer(BaseFeatureEngineer):
    def __init__(self, hist_races: pd.DataFrame, hist_quali: pd.DataFrame):
        self.hist_races = hist_races
        self.hist_quali = hist_quali
        super().__init__(pd.DataFrame())

    def engineer_features(self) -> pd.DataFrame:
        logger.info("Preparing initial data...")
        if self.hist_races is None or self.hist_quali is None:
            logger.error("Historical race or qualifying data is missing.")
            return pd.DataFrame()
            
        cleaned_races = self._clean_race_data(self.hist_races)
        cleaned_quali = self._clean_quali_data(self.hist_quali)
        
        quali_to_merge = cleaned_quali[['Year', 'Race_Num', 'Driver', 'Position']].rename(columns={'Position': 'Quali_Pos'})
        self.df = pd.merge(cleaned_races, quali_to_merge, on=['Year', 'Race_Num', 'Driver'], how='left')
        return self.df

    def _clean_race_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
        df['Grid'] = pd.to_numeric(df['Grid'], errors='coerce')
        df['DNF'] = df['Status'].apply(lambda x: 1 if x not in ['Finished', '+1 Lap', '+2 Laps', '+3 Laps'] else 0)
        return df

    def _clean_quali_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df

class RollingPerformanceEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        logger.info("Engineering rolling performance features...")
        self.df.sort_values(by=['Year', 'Race_Num'], inplace=True)
        self._calculate_rolling_metrics_for_entity('Driver')
        self._calculate_rolling_metrics_for_entity('Team')
        return self.df.sort_index()

    def _calculate_rolling_metrics_for_entity(self, entity: str):
        windows = config.get('feature_engineering.rolling_windows', {'short': 3, 'medium': 5})
        for metric in ['Position', 'Quali_Pos', 'DNF']:
            if metric in self.df.columns:
                for name, size in windows.items():
                    col = f'{entity}_Avg_{metric}_{name}'
                    shifted = self.df.groupby(entity)[metric].shift(1)
                    self.df[col] = shifted.rolling(window=size, min_periods=1).mean()

class WeatherFeatureEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        logger.info("Engineering weather features...")
        weather_cols = ['AirTemp', 'Humidity', 'Pressure', 'WindSpeed', 'Rainfall']
        if not all(col in self.df.columns for col in weather_cols):
            logger.warning("Weather data columns not found. Skipping weather features.")
            return self.df

        self.df['is_wet_race'] = (self.df['Rainfall'] > 0).astype(int)
        
        optimal_temp = np.mean(config.get('feature_engineering.weather_impact.optimal_temp_range', [20, 30]))
        self.df['temp_deviation'] = (self.df['AirTemp'] - optimal_temp).abs()
        
        wind_threshold = config.get('feature_engineering.weather_impact.wind_speed_threshold', 25)
        self.df['is_windy'] = (self.df['WindSpeed'] > wind_threshold).astype(int)
        return self.df

class FeatureOptimizer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        logger.info("Optimizing final feature set...")
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if self.df[col].isnull().any():
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        self.df.fillna(0, inplace=True)
        return self.df

class FeatureEngineeringPipeline:
    def __init__(self, hist_races: pd.DataFrame, hist_quali: pd.DataFrame):
        self.hist_races = hist_races
        self.hist_quali = hist_quali
        self.engineers = [
            RollingPerformanceEngineer,
            WeatherFeatureEngineer,
            FeatureOptimizer
        ]

    def run(self) -> pd.DataFrame:
        logger.info("Running feature engineering pipeline...")
        preparer = DataPreparer(self.hist_races, self.hist_quali)
        df = preparer.engineer_features()
        if df.empty:
            return pd.DataFrame()
            
        for engineer_class in self.engineers:
            df = engineer_class(df).engineer_features()
            
        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        return df 