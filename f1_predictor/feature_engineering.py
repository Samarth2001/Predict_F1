# f1_predictor/feature_engineering.py - Advanced F1 Feature Engineering System

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from category_encoders import TargetEncoder, BinaryEncoder, CatBoostEncoder
import warnings
warnings.filterwarnings('ignore')

from . import config

logger = logging.getLogger(__name__)

class F1FeatureEngineer:
    """
    Advanced F1 Feature Engineering System
    
    Comprehensive feature engineering for Formula 1 race predictions including:
    - Driver performance metrics across multiple timeframes
    - Team reliability and development trends
    - Circuit-specific characteristics and historical performance
    - Weather impact modeling
    - Strategic and contextual features
    - Advanced encoding and transformation techniques
    """
    
    def __init__(self):
        """Initialize the F1 Feature Engineer."""
        self.label_encoders = {}
        self.target_encoders = {}
        self.scalers = {}
        self.imputers = {}
        self.feature_importance = {}
        self.feature_correlations = {}
        
        # Circuit information database
        self.circuit_info = self._initialize_circuit_database()
        
        # Driver and team mappings
        self.driver_mappings = {}
        self.team_mappings = {}
        
        logger.info("F1 Feature Engineer initialized successfully")
    
    def engineer_features(self, 
                         hist_races: pd.DataFrame, 
                         hist_quali: pd.DataFrame,
                         curr_races: pd.DataFrame = None,
                         curr_quali: pd.DataFrame = None,
                         upcoming_info: pd.DataFrame = None) -> pd.DataFrame:
        """
        Engineer comprehensive features for F1 race prediction.
        
        Args:
            hist_races: Historical race results
            hist_quali: Historical qualifying results  
            curr_races: Current season race results
            curr_quali: Current season qualifying results
            upcoming_info: Information about upcoming races
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting comprehensive F1 feature engineering...")
        
        try:
            # Combine historical and current data
            all_races = self._combine_race_data(hist_races, curr_races)
            all_quali = self._combine_quali_data(hist_quali, curr_quali)
            
            # Data cleaning and preprocessing
            all_races = self._clean_race_data(all_races)
            all_quali = self._clean_quali_data(all_quali)
            
            # Merge race and qualifying data
            merged_data = self._merge_race_quali_data(all_races, all_quali)
            
            # Engineer all feature categories
            features_df = self._engineer_all_features(merged_data)
            
            # Add upcoming race features if provided
            if upcoming_info is not None and not upcoming_info.empty:
                upcoming_features = self._engineer_upcoming_features(upcoming_info, merged_data)
                features_df = pd.concat([features_df, upcoming_features], ignore_index=True)
            
            # Feature selection and optimization
            features_df = self._optimize_features(features_df)
            
            # Final validation and cleaning
            features_df = self._validate_features(features_df)
            
            logger.info(f"Feature engineering completed. Final shape: {features_df.shape}")
            return features_df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
    
    def _initialize_circuit_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive circuit characteristic database."""
        circuit_db = {
            'Monaco': {
                'length': 3.337, 'turns': 19, 'type': 'street',
                'avg_speed': 160, 'difficulty': 10, 'overtaking_difficulty': 10,
                'tire_wear_rate': 6, 'drs_zones': 1, 'elevation_change': 42,
                'categories': ['street', 'technical', 'aero_sensitive', 'overtaking_difficult']
            },
            'Silverstone': {
                'length': 5.891, 'turns': 18, 'type': 'permanent',
                'avg_speed': 240, 'difficulty': 7, 'overtaking_difficulty': 4,
                'tire_wear_rate': 8, 'drs_zones': 2, 'elevation_change': 20,
                'categories': ['high_speed', 'power_sensitive']
            },
            'Monza': {
                'length': 5.793, 'turns': 11, 'type': 'permanent',
                'avg_speed': 250, 'difficulty': 5, 'overtaking_difficulty': 2,
                'tire_wear_rate': 5, 'drs_zones': 3, 'elevation_change': 15,
                'categories': ['high_speed', 'power_sensitive']
            },
            'Singapore': {
                'length': 5.063, 'turns': 23, 'type': 'street',
                'avg_speed': 170, 'difficulty': 9, 'overtaking_difficulty': 8,
                'tire_wear_rate': 7, 'drs_zones': 3, 'elevation_change': 8,
                'categories': ['street', 'technical', 'aero_sensitive', 'overtaking_difficult']
            },
            'Spa-Francorchamps': {
                'length': 7.004, 'turns': 19, 'type': 'permanent',
                'avg_speed': 230, 'difficulty': 8, 'overtaking_difficulty': 3,
                'tire_wear_rate': 6, 'drs_zones': 2, 'elevation_change': 100,
                'categories': ['high_speed', 'power_sensitive']
            },
            # Add more circuits as needed
        }
        
        # Fill in default values for any missing circuits
        default_circuit = {
            'length': 5.0, 'turns': 16, 'type': 'permanent',
            'avg_speed': 200, 'difficulty': 6, 'overtaking_difficulty': 5,
            'tire_wear_rate': 6, 'drs_zones': 2, 'elevation_change': 30,
            'categories': []
        }
        
        return circuit_db
    
    def _combine_race_data(self, hist_races: pd.DataFrame, curr_races: pd.DataFrame = None) -> pd.DataFrame:
        """Combine historical and current race data."""
        if curr_races is not None and not curr_races.empty:
            combined = pd.concat([hist_races, curr_races], ignore_index=True)
        else:
            combined = hist_races.copy()
        
        combined = combined.sort_values(['Year', 'Race_Num', 'Driver']).reset_index(drop=True)
        logger.info(f"Combined race data shape: {combined.shape}")
        return combined
    
    def _combine_quali_data(self, hist_quali: pd.DataFrame, curr_quali: pd.DataFrame = None) -> pd.DataFrame:
        """Combine historical and current qualifying data."""
        if curr_quali is not None and not curr_quali.empty:
            combined = pd.concat([hist_quali, curr_quali], ignore_index=True)
        else:
            combined = hist_quali.copy()
        
        combined = combined.sort_values(['Year', 'Race_Num', 'Driver']).reset_index(drop=True)
        logger.info(f"Combined qualifying data shape: {combined.shape}")
        return combined
    
    def _clean_race_data(self, races_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize race data."""
        df = races_df.copy()
        
        # Standardize status categories
        df['Status_Cleaned'] = df['Status'].apply(self._categorize_race_status)
        df['Finished'] = df['Status_Cleaned'].apply(lambda x: 1 if x == 'Finished' else 0)
        df['DNF'] = df['Status_Cleaned'].apply(lambda x: 1 if x == 'DNF' else 0)
        
        # Handle position data - ensure proper numeric conversion
        df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
        df['Grid'] = pd.to_numeric(df['Grid'], errors='coerce')
        
        # Clean and convert Finish_Pos_Clean
        if 'Finish_Pos_Clean' in df.columns:
            df['Finish_Pos_Clean'] = pd.to_numeric(df['Finish_Pos_Clean'], errors='coerce')
        else:
            df['Finish_Pos_Clean'] = self._clean_position(df['Position'])
        
        # Points calculation - only for finished races
        points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        df['Points_Calculated'] = df['Position'].map(points_system).fillna(0)
        
        # Time-based features
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            df['Year'] = df['Date'].dt.year.fillna(df.get('Year', 2024))
            df['Month'] = df['Date'].dt.month.fillna(6)  # Default to mid-season
            df['Day_of_Year'] = df['Date'].dt.dayofyear.fillna(150)
        else:
            # Use existing Year column or default
            df['Year'] = df.get('Year', 2024)
            df['Month'] = 6
            df['Day_of_Year'] = 150
        
        # Ensure critical columns exist and have proper defaults
        if 'Race_Num' not in df.columns:
            df['Race_Num'] = 1
        if 'Circuit' not in df.columns:
            df['Circuit'] = 'Unknown'
        if 'Driver' not in df.columns:
            df['Driver'] = 'UNK'
        if 'Team' not in df.columns:
            df['Team'] = 'Unknown Team'
            
        # Convert Race_Num to numeric
        df['Race_Num'] = pd.to_numeric(df['Race_Num'], errors='coerce').fillna(1)
        
        return df
    
    def _clean_quali_data(self, quali_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize qualifying data."""
        df = quali_df.copy()
        
        # Handle qualifying positions
        df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
        
        # Create Quali_Pos if it doesn't exist
        if 'Quali_Pos' not in df.columns:
            df['Quali_Pos'] = df['Position']
        else:
            df['Quali_Pos'] = pd.to_numeric(df['Quali_Pos'], errors='coerce')
        
        # Convert qualifying times to seconds for analysis
        df['Q1_Seconds'] = df['Q1'].apply(self._time_to_seconds) if 'Q1' in df.columns else np.nan
        df['Q2_Seconds'] = df['Q2'].apply(self._time_to_seconds) if 'Q2' in df.columns else np.nan
        df['Q3_Seconds'] = df['Q3'].apply(self._time_to_seconds) if 'Q3' in df.columns else np.nan
        
        # Best qualifying time
        time_cols = [col for col in ['Q1_Seconds', 'Q2_Seconds', 'Q3_Seconds'] if col in df.columns]
        if time_cols:
            df['Best_Quali_Time'] = df[time_cols].min(axis=1)
        else:
            df['Best_Quali_Time'] = np.nan
        
        # Handle qualifying time in seconds if available
        if 'Quali_Time_Seconds' in df.columns:
            df['Quali_Time_Seconds'] = pd.to_numeric(df['Quali_Time_Seconds'], errors='coerce')
            # Use this as Quali_Time if Best_Quali_Time is not available
            if df['Best_Quali_Time'].isna().all():
                df['Best_Quali_Time'] = df['Quali_Time_Seconds']
        
        # Time-based features
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            df['Year'] = df['Date'].dt.year.fillna(df.get('Year', 2024))
        else:
            df['Year'] = df.get('Year', 2024)
        
        # Ensure critical columns exist
        if 'Race_Num' not in df.columns:
            df['Race_Num'] = 1
        if 'Circuit' not in df.columns:
            df['Circuit'] = 'Unknown'
        if 'Driver' not in df.columns:
            df['Driver'] = 'UNK'
        if 'Team' not in df.columns:
            df['Team'] = 'Unknown Team'
            
        # Convert Race_Num to numeric
        df['Race_Num'] = pd.to_numeric(df['Race_Num'], errors='coerce').fillna(1)
        
        return df
    
    def _merge_race_quali_data(self, races_df: pd.DataFrame, quali_df: pd.DataFrame) -> pd.DataFrame:
        """Merge race and qualifying data intelligently."""
        # Prepare qualifying data for merge
        quali_merge = quali_df[['Year', 'Circuit', 'Driver', 'Position', 'Best_Quali_Time']].copy()
        quali_merge.columns = ['Year', 'Circuit', 'Driver', 'Quali_Pos', 'Quali_Time']
        
        # Merge on Year, Circuit, Driver
        merged = races_df.merge(
            quali_merge, 
            on=['Year', 'Circuit', 'Driver'], 
            how='left'
        )
        
        # Fill missing qualifying positions with grid positions
        merged['Quali_Pos'] = merged['Quali_Pos'].fillna(merged['Grid'])
        
        logger.info(f"Merged data shape: {merged.shape}")
        return merged
    
    def _engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer all feature categories."""
        logger.info("Engineering all feature categories...")
        
        # Initialize features dataframe
        features_df = df.copy()
        
        # 1. Basic Features
        features_df = self._add_basic_features(features_df)
        
        # 2. Driver Performance Features
        features_df = self._add_driver_performance_features(features_df)
        
        # 3. Team Performance Features  
        features_df = self._add_team_performance_features(features_df)
        
        # 4. Circuit-Specific Features
        features_df = self._add_circuit_features(features_df)
        
        # 5. Weather and Environmental Features
        features_df = self._add_weather_features(features_df)
        
        # 6. Strategic Features
        features_df = self._add_strategic_features(features_df)
        
        # 7. Contextual Features
        features_df = self._add_contextual_features(features_df)
        
        # 8. Advanced Statistical Features
        features_df = self._add_statistical_features(features_df)
        
        return features_df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic driver, team, and race features."""
        logger.info("Adding basic features...")
        
        # Encode categorical variables
        df['Driver_Encoded'] = self._encode_categorical(df, 'Driver', 'target')
        df['Team_Encoded'] = self._encode_categorical(df, 'Team', 'target')
        df['Circuit_Encoded'] = self._encode_categorical(df, 'Circuit', 'target')
        
        # Driver experience and age (synthetic if not available)
        driver_first_race = df.groupby('Driver')['Year'].min().to_dict()
        df['Driver_Experience'] = df.apply(lambda x: x['Year'] - driver_first_race[x['Driver']] + 1, axis=1)
        df['Driver_Age'] = df['Driver_Experience'] + 22  # Synthetic age based on experience
        
        # Race context
        df['Season_Progress'] = df['Race_Num'] / 23.0  # Normalized season progress
        df['Grid_Pos'] = df['Grid'].fillna(20)
        df['Quali_Gap_To_Pole'] = self._calculate_quali_gap_to_pole(df)
        
        return df
    
    def _add_driver_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive driver performance features."""
        logger.info("Adding driver performance features...")
        
        # Sort data for rolling calculations
        df = df.sort_values(['Driver', 'Year', 'Race_Num']).reset_index(drop=True)
        
        # Rolling averages for different windows
        for window in config.ROLLING_WINDOWS.values():
            if window <= len(df):
                # Finishing position rolling averages
                df[f'Driver_Avg_Finish_Last_{window}'] = (
                    df.groupby('Driver')['Position']
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                
                # Qualifying position rolling averages
                df[f'Driver_Avg_Quali_Last_{window}'] = (
                    df.groupby('Driver')['Quali_Pos']
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                
                # Points rolling averages
                df[f'Driver_Avg_Points_Last_{window}'] = (
                    df.groupby('Driver')['Points_Calculated']
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
        
        # Career statistics
        df = self._add_driver_career_stats(df)
        
        # Circuit-specific driver performance
        df = self._add_driver_circuit_performance(df)
        
        # Recent form indicators
        df = self._add_driver_form_indicators(df)
        
        # Championship pressure features (moved here to ensure they're created early)
        df = self._add_championship_pressure_features(df)
        
        return df
    
    def _add_team_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive team performance features."""
        logger.info("Adding team performance features...")
        
        # Sort data for rolling calculations
        df = df.sort_values(['Team', 'Year', 'Race_Num']).reset_index(drop=True)
        
        # Team rolling averages
        for window in config.ROLLING_WINDOWS.values():
            if window <= len(df):
                df[f'Team_Avg_Finish_Last_{window}'] = (
                    df.groupby('Team')['Position']
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                
                df[f'Team_Avg_Points_Last_{window}'] = (
                    df.groupby('Team')['Points_Calculated']
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
        
        # Team reliability and development
        df = self._add_team_reliability_features(df)
        df = self._add_team_development_features(df)
        
        # Championship standings
        df = self._add_team_championship_features(df)
        
        return df
    
    def _add_circuit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add circuit-specific features."""
        logger.info("Adding circuit features...")
        
        # Circuit characteristics from database
        for feature in ['length', 'turns', 'avg_speed', 'difficulty', 'overtaking_difficulty', 
                       'tire_wear_rate', 'drs_zones', 'elevation_change']:
            df[f'Circuit_{feature.title()}'] = df['Circuit'].map(
                {circuit: info.get(feature, 0) for circuit, info in self.circuit_info.items()}
            ).fillna(self.circuit_info.get('default', {}).get(feature, 0))
        
        # Circuit type encoding
        circuit_types = {circuit: info.get('type', 'permanent') for circuit, info in self.circuit_info.items()}
        df['Circuit_Type'] = df['Circuit'].map(circuit_types).fillna('permanent')
        df['Circuit_Type_Encoded'] = self._encode_categorical(df, 'Circuit_Type', 'label')
        
        # Circuit categories
        df = self._add_circuit_category_features(df)
        
        # Historical circuit performance
        df = self._add_circuit_historical_performance(df)
        
        return df
    
    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weather and environmental features."""
        logger.info("Adding weather features...")
        
        # Synthetic weather data based on circuit and season
        df['Weather_Temp'] = self._generate_weather_temperature(df)
        df['Weather_Humidity'] = self._generate_weather_humidity(df)
        df['Weather_Wind_Speed'] = self._generate_weather_wind_speed(df)
        df['Weather_Rain_Probability'] = self._generate_rain_probability(df)
        
        # Weather impact features
        df['Weather_Impact_Score'] = self._calculate_weather_impact(df)
        df['Weather_Favorable'] = (df['Weather_Impact_Score'] > 0).astype(int)
        
        return df
    
    def _add_strategic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add strategic and tactical features."""
        logger.info("Adding strategic features...")
        
        # Tire strategy features
        df['Tire_Strategy_Optimal'] = self._calculate_optimal_tire_strategy(df)
        df['Pit_Window_Optimal'] = self._calculate_optimal_pit_window(df)
        
        # Safety car probability
        df['Safety_Car_Expected'] = self._calculate_safety_car_probability(df)
        
        # Strategic risk assessment
        df['Setup_Risk_Level'] = self._calculate_setup_risk(df)
        
        return df
    
    def _add_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add contextual features based on championship situation."""
        logger.info("Adding contextual features...")
        
        # Home race indicators
        df = self._add_home_race_features(df)
        
        # Season phase indicators
        df['Season_Phase'] = pd.cut(df['Race_Num'], bins=[0, 6, 12, 18, 23], 
                                  labels=['Early', 'Mid_Early', 'Mid_Late', 'Late'])
        df['Season_Phase_Encoded'] = self._encode_categorical(df, 'Season_Phase', 'label')
        
        # Rookie season indicators
        df['Driver_Rookie_Season'] = (df['Driver_Experience'] == 1).astype(int)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced statistical features."""
        logger.info("Adding statistical features...")
        
        # Position variance and consistency
        df = self._add_consistency_features(df)
        
        # Performance trends
        df = self._add_trend_features(df)
        
        # Interaction features
        df = self._add_interaction_features(df)
        
        return df
    
    # ========================================================================
    # HELPER METHODS FOR SPECIFIC FEATURE CALCULATIONS
    # ========================================================================
    
    def _categorize_race_status(self, status: str) -> str:
        """Categorize race status into standardized categories."""
        if pd.isna(status):
            return 'Unknown'
        
        status_str = str(status).strip()
        
        if status_str in config.FINISHED_STATUSES or status_str == 'Finished':
            return 'Finished'
        elif any(dnf in status_str for dnf in config.DNF_STATUSES):
            return 'DNF'
        elif '+' in status_str and 'Lap' in status_str:
            return 'Finished'
        else:
            return 'Other'
    
    def _time_to_seconds(self, time_str: str) -> float:
        """Convert time string to seconds."""
        if pd.isna(time_str) or time_str == '':
            return np.nan
        
        try:
            time_str = str(time_str).strip()
            if ':' in time_str:
                parts = time_str.split(':')
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                return float(time_str)
        except:
            return np.nan
    
    def _encode_categorical(self, df: pd.DataFrame, column: str, method: str = 'target') -> pd.Series:
        """Encode categorical variables using specified method."""
        if column not in df.columns:
            return pd.Series(0, index=df.index)
        
        if method == 'label':
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                encoded = self.label_encoders[column].fit_transform(df[column].astype(str))
            else:
                encoded = self.label_encoders[column].transform(df[column].astype(str))
        
        elif method == 'target':
            if column not in self.target_encoders:
                self.target_encoders[column] = TargetEncoder()
                if 'Position' in df.columns:
                    # Filter out rows with missing target values for encoding
                    valid_mask = df['Position'].notna()
                    if valid_mask.sum() > 0:  # Ensure we have valid data
                        valid_df = df[valid_mask]
                        # Fit the encoder on valid data only
                        self.target_encoders[column].fit(valid_df[column], valid_df['Position'])
                        # Transform all data (encoder will handle unseen categories)
                        encoded = self.target_encoders[column].transform(df[column])
                    else:
                        # No valid target data, fallback to label encoding
                        logger.warning(f"No valid target data for encoding {column}, using label encoding instead")
                        return self._encode_categorical(df, column, 'label')
                else:
                    # Fallback to label encoding if no target available
                    return self._encode_categorical(df, column, 'label')
            else:
                encoded = self.target_encoders[column].transform(df[column])
        
        else:
            # Default to label encoding
            return self._encode_categorical(df, column, 'label')
        
        # Handle 2D arrays/DataFrames from TargetEncoder - extract 1D values
        if isinstance(encoded, pd.DataFrame):
            # Get the values from the first (and only) column of the DataFrame
            encoded = encoded.iloc[:, 0].values
        elif isinstance(encoded, np.ndarray) and encoded.ndim == 2:
            encoded = encoded.flatten()
        
        return pd.Series(encoded, index=df.index)
    
    def _calculate_quali_gap_to_pole(self, df: pd.DataFrame) -> pd.Series:
        """Calculate qualifying gap to pole position."""
        if 'Quali_Time' not in df.columns:
            return pd.Series(0, index=df.index)
        
        gaps = []
        for _, group in df.groupby(['Year', 'Circuit']):
            pole_time = group['Quali_Time'].min()
            group_gaps = group['Quali_Time'] - pole_time
            gaps.extend(group_gaps.tolist())
        
        return pd.Series(gaps, index=df.index)
    
    def _add_driver_career_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add driver career statistics."""
        # Calculate cumulative stats up to each race
        df_sorted = df.sort_values(['Driver', 'Year', 'Race_Num'])
        
        # Wins, podiums, points
        df_sorted['Driver_Wins_Career'] = (df_sorted['Position'] == 1).astype(int).groupby(df_sorted['Driver']).cumsum()
        df_sorted['Driver_Podiums_Career'] = (df_sorted['Position'] <= 3).astype(int).groupby(df_sorted['Driver']).cumsum()
        df_sorted['Driver_Points_Career'] = df_sorted['Points_Calculated'].groupby(df_sorted['Driver']).cumsum()
        
        # Rates (avoid division by zero)
        df_sorted['Driver_Races_Career'] = df_sorted.groupby('Driver').cumcount() + 1
        df_sorted['Driver_Win_Rate'] = df_sorted['Driver_Wins_Career'] / df_sorted['Driver_Races_Career']
        df_sorted['Driver_Podium_Rate'] = df_sorted['Driver_Podiums_Career'] / df_sorted['Driver_Races_Career']
        df_sorted['Driver_Points_Rate'] = df_sorted['Driver_Points_Career'] / df_sorted['Driver_Races_Career']
        
        # DNF rate
        df_sorted['Driver_DNFs_Career'] = df_sorted['DNF'].groupby(df_sorted['Driver']).cumsum()
        df_sorted['Driver_DNF_Rate'] = df_sorted['Driver_DNFs_Career'] / df_sorted['Driver_Races_Career']
        
        return df_sorted.sort_index()
    
    def _add_driver_circuit_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add driver performance at specific circuits."""
        # Circuit experience
        df['Driver_Circuit_Experience'] = df.groupby(['Driver', 'Circuit']).cumcount()
        
        # Circuit-specific averages (calculated progressively)
        circuit_stats = ['Position', 'Quali_Pos', 'Points_Calculated']
        
        for stat in circuit_stats:
            if stat in df.columns:
                # Calculate expanding mean for each driver-circuit combination
                df[f'Driver_Circuit_Avg_{stat.split("_")[0]}'] = (
                    df.groupby(['Driver', 'Circuit'])[stat]
                    .expanding()
                    .mean()
                    .reset_index(level=[0, 1], drop=True)
                )
        
        return df
    
    def _add_driver_form_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add driver form and momentum indicators."""
        # Recent performance trend (last 3 races)
        df['Driver_Position_Trend_3'] = (
            df.groupby('Driver')['Position']
            .rolling(window=3, min_periods=1)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            .reset_index(level=0, drop=True)
        )
        
        # Consistency (standard deviation of recent positions)
        df['Driver_Consistency_5'] = (
            df.groupby('Driver')['Position']
            .rolling(window=5, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(5.0)
        )
        
        # Hot streak indicator
        df['Driver_Recent_Podiums'] = (
            (df['Position'] <= 3).astype(int)
            .groupby(df['Driver'])
            .rolling(window=3, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        
        return df
    
    def _add_team_reliability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team reliability metrics."""
        # Calculate team reliability score based on DNF rate
        team_dnf_rate = df.groupby(['Team', 'Year'])['DNF'].mean().reset_index()
        team_dnf_rate['Team_Reliability_Score'] = 1 - team_dnf_rate['DNF']
        
        # Merge back to main dataframe
        df = df.merge(team_dnf_rate[['Team', 'Year', 'Team_Reliability_Score']], 
                     on=['Team', 'Year'], how='left')
        
        df['Team_Reliability_Score'] = df['Team_Reliability_Score'].fillna(0.85)
        
        return df
    
    def _add_team_development_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team development and improvement trends."""
        # Calculate team performance trend within season
        df_sorted = df.sort_values(['Team', 'Year', 'Race_Num'])
        
        df_sorted['Team_Development_Trend'] = (
            df_sorted.groupby(['Team', 'Year'])['Position']
            .rolling(window=5, min_periods=2)
            .apply(lambda x: -np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            .reset_index(level=[0, 1], drop=True)
        )
        
        return df_sorted.sort_index()
    
    def _add_team_championship_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team championship standing features."""
        # Calculate team points and standings
        team_standings = (
            df.groupby(['Team', 'Year', 'Race_Num'])['Points_Calculated']
            .sum()
            .groupby(['Team', 'Year'])
            .cumsum()
            .reset_index()
        )
        team_standings.columns = ['Team', 'Year', 'Race_Num', 'Team_Season_Points']
        
        # Merge back
        df = df.merge(team_standings, on=['Team', 'Year', 'Race_Num'], how='left')
        df['Team_Season_Points'] = df['Team_Season_Points'].fillna(0)
        
        # Calculate championship position
        df['Team_Championship_Position'] = (
            df.groupby(['Year', 'Race_Num'])['Team_Season_Points']
            .rank(method='dense', ascending=False)
        )
        
        return df
    
    def _add_circuit_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add circuit category features."""
        for category in config.CIRCUIT_CATEGORIES.keys():
            df[f'Circuit_Is_{category.title()}'] = df['Circuit'].apply(
                lambda x: 1 if x in config.CIRCUIT_CATEGORIES[category] else 0
            )
        
        return df
    
    def _add_circuit_historical_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add historical performance at each circuit."""
        # Average finishing position at circuit
        circuit_avg_pos = df.groupby('Circuit')['Position'].mean().to_dict()
        df['Circuit_Avg_Finish'] = df['Circuit'].map(circuit_avg_pos)
        
        # Average qualifying position at circuit  
        circuit_avg_quali = df.groupby('Circuit')['Quali_Pos'].mean().to_dict()
        df['Circuit_Avg_Quali'] = df['Circuit'].map(circuit_avg_quali)
        
        return df
    
    def _generate_weather_temperature(self, df: pd.DataFrame) -> pd.Series:
        """Generate realistic weather temperature based on circuit and season."""
        # Base temperatures by circuit (simplified)
        base_temps = {
            'Monaco': 22, 'Singapore': 30, 'Abu Dhabi': 28, 'Bahrain': 26,
            'Miami': 28, 'Las Vegas': 20, 'Mexico City': 18, 'Brazil': 25
        }
        
        # Seasonal adjustment
        seasonal_adjustment = np.sin((df['Race_Num'] - 1) * 2 * np.pi / 23) * 8
        
        base_temp = df['Circuit'].map(base_temps).fillna(24)
        return base_temp + seasonal_adjustment + np.random.normal(0, 2, len(df))
    
    def _generate_weather_humidity(self, df: pd.DataFrame) -> pd.Series:
        """Generate realistic humidity based on circuit characteristics."""
        base_humidity = {
            'Singapore': 80, 'Malaysia': 85, 'Brazil': 75, 'Miami': 70,
            'Monaco': 65, 'Abu Dhabi': 45, 'Bahrain': 50
        }
        
        humidity = df['Circuit'].map(base_humidity).fillna(60)
        return humidity + np.random.normal(0, 5, len(df))
    
    def _generate_weather_wind_speed(self, df: pd.DataFrame) -> pd.Series:
        """Generate wind speed based on circuit location."""
        return np.random.exponential(8, len(df)) + 2
    
    def _generate_rain_probability(self, df: pd.DataFrame) -> pd.Series:
        """Generate rain probability based on circuit and season."""
        rain_prone_circuits = ['Silverstone', 'Spa-Francorchamps', 'Brazil', 'Malaysia']
        
        base_prob = df['Circuit'].apply(
            lambda x: 25 if x in rain_prone_circuits else 10
        )
        
        return base_prob + np.random.normal(0, 5, len(df))
    
    def _calculate_weather_impact(self, df: pd.DataFrame) -> pd.Series:
        """Calculate overall weather impact score."""
        temp_impact = np.abs(df['Weather_Temp'] - 25) / 15  # Optimal around 25°C
        humidity_impact = np.abs(df['Weather_Humidity'] - 60) / 40
        wind_impact = df['Weather_Wind_Speed'] / 30
        rain_impact = df['Weather_Rain_Probability'] / 100
        
        return -(temp_impact + humidity_impact + wind_impact + rain_impact)
    
    def _calculate_optimal_tire_strategy(self, df: pd.DataFrame) -> pd.Series:
        """Calculate optimal tire strategy indicator."""
        # Simplified: based on circuit tire wear and weather
        base_strategy = df['Circuit_Tire_Wear_Rate'] / 10
        weather_adjustment = df['Weather_Rain_Probability'] / 100
        
        return base_strategy + weather_adjustment
    
    def _calculate_optimal_pit_window(self, df: pd.DataFrame) -> pd.Series:
        """Calculate optimal pit window timing."""
        # Based on race distance and tire wear
        race_distance = df['Circuit_Length'] * 50  # Approximate race distance
        return race_distance / 3  # Simplified pit window calculation
    
    def _calculate_safety_car_probability(self, df: pd.DataFrame) -> pd.Series:
        """Calculate safety car probability based on circuit characteristics."""
        base_prob = {
            'Monaco': 0.8, 'Singapore': 0.7, 'Baku': 0.6, 'Miami': 0.5,
            'Spa-Francorchamps': 0.3, 'Monza': 0.2
        }
        
        return df['Circuit'].map(base_prob).fillna(0.4)
    
    def _calculate_setup_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate setup risk level."""
        # Based on championship position and circuit difficulty
        championship_pressure = 1 / (df['Driver_Championship_Position'] + 1)
        circuit_difficulty = df['Circuit_Difficulty'] / 10
        
        return championship_pressure * circuit_difficulty
    
    def _add_championship_pressure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add championship pressure indicators."""
        # Calculate driver season points first (if not already present)
        if 'Driver_Season_Points' not in df.columns:
            # Calculate cumulative driver points for each season
            df_sorted = df.sort_values(['Driver', 'Year', 'Race_Num'])
            df_sorted['Driver_Season_Points'] = df_sorted['Points_Calculated'].groupby([df_sorted['Driver'], df_sorted['Year']]).cumsum()
            df = df_sorted.sort_index()
        
        # Points gap to leader
        leader_points = df.groupby(['Year', 'Race_Num'])['Driver_Season_Points'].max()
        df = df.merge(leader_points.rename('Leader_Points'), left_on=['Year', 'Race_Num'], right_index=True)
        
        df['Driver_Points_Gap_To_Leader'] = df['Leader_Points'] - df['Driver_Season_Points']
        df['Driver_Championship_Pressure'] = df['Driver_Points_Gap_To_Leader'] / (df['Race_Num'] * 25)
        
        # Calculate driver championship position
        df['Driver_Championship_Position'] = (
            df.groupby(['Year', 'Race_Num'])['Driver_Season_Points']
            .rank(method='dense', ascending=False)
        )
        
        return df
    
    def _add_home_race_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add home race indicators."""
        # Simplified home race mapping
        home_races = {
            'Hamilton': ['Silverstone'], 'Russell': ['Silverstone'],
            'Verstappen': ['Zandvoort'], 'Leclerc': ['Monaco'],
            'Sainz': ['Spain'], 'Alonso': ['Spain']
        }
        
        df['Driver_Home_Race'] = 0
        for driver, circuits in home_races.items():
            mask = (df['Driver'] == driver) & (df['Circuit'].isin(circuits))
            df.loc[mask, 'Driver_Home_Race'] = 1
        
        return df
    
    def _add_consistency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add performance consistency features."""
        # Position variance over last N races
        for window in [5, 10]:
            df[f'Driver_Position_Std_{window}'] = (
                df.groupby('Driver')['Position']
                .rolling(window=window, min_periods=2)
                .std()
                .reset_index(level=0, drop=True)
                .fillna(5.0)
            )
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add performance trend features."""
        # Calculate trends for different metrics
        metrics = ['Position', 'Points_Calculated']
        
        for metric in metrics:
            if metric in df.columns:
                df[f'{metric}_Trend_5'] = (
                    df.groupby('Driver')[metric]
                    .rolling(window=5, min_periods=3)
                    .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0)
                    .reset_index(level=0, drop=True)
                )
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between key variables."""
        # Driver experience × Circuit difficulty
        df['Driver_Experience_Circuit_Difficulty'] = df['Driver_Experience'] * df['Circuit_Difficulty']
        
        # Team reliability × Circuit difficulty
        df['Team_Reliability_Circuit_Type'] = df['Team_Reliability_Score'] * df['Circuit_Type_Encoded']
        
        # Weather impact × Circuit type
        df['Weather_Circuit_Interaction'] = df['Weather_Impact_Score'] * df['Circuit_Is_Street']
        
        return df
    
    def _engineer_upcoming_features(self, upcoming_info: pd.DataFrame, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for upcoming races."""
        logger.info("Engineering features for upcoming races...")
        
        # Use historical data to create baseline features
        upcoming_features = upcoming_info.copy()
        
        # Add circuit information
        for circuit in upcoming_features['Circuit'].unique():
            circuit_data = historical_data[historical_data['Circuit'] == circuit]
            if not circuit_data.empty:
                # Use most recent data for this circuit
                recent_circuit_data = circuit_data.iloc[-20:]  # Last 20 races at this circuit
                
                # Calculate expected features
                upcoming_features.loc[upcoming_features['Circuit'] == circuit, 'Expected_Avg_Position'] = (
                    recent_circuit_data['Position'].mean()
                )
        
        return upcoming_features
    
    def _optimize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize features through selection and transformation."""
        logger.info("Optimizing features...")
        
        # Remove highly correlated features
        df = self._remove_correlated_features(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Feature scaling for numerical features
        df = self._scale_features(df)
        
        return df
    
    def _remove_correlated_features(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
            df = df.drop(columns=to_drop)
            
            if to_drop:
                logger.info(f"Removed {len(to_drop)} highly correlated features")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently."""
        try:
            # Separate numeric and categorical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Handle numeric missing values
            if len(numeric_cols) > 0:
                # Check for any columns with all NaN values and handle them separately
                all_nan_cols = [col for col in numeric_cols if df[col].isna().all()]
                valid_numeric_cols = [col for col in numeric_cols if col not in all_nan_cols]
                
                # Fill all-NaN columns with 0
                for col in all_nan_cols:
                    df[col] = 0
                
                # Use KNNImputer only on columns with some valid data
                if len(valid_numeric_cols) > 0:
                    numeric_imputer = KNNImputer(n_neighbors=min(5, len(df)))
                    imputed_values = numeric_imputer.fit_transform(df[valid_numeric_cols])
                    
                    # Ensure the imputed values have the correct shape
                    if imputed_values.shape[1] == len(valid_numeric_cols):
                        df[valid_numeric_cols] = imputed_values
                    else:
                        logger.warning("KNNImputer shape mismatch, using simple imputation instead")
                        df[valid_numeric_cols] = df[valid_numeric_cols].fillna(df[valid_numeric_cols].median())
            
            # Handle categorical missing values
            if len(categorical_cols) > 0:
                categorical_imputer = SimpleImputer(strategy='most_frequent')
                try:
                    imputed_categorical = categorical_imputer.fit_transform(df[categorical_cols])
                    if imputed_categorical.shape[1] == len(categorical_cols):
                        df[categorical_cols] = imputed_categorical
                    else:
                        logger.warning("Categorical imputer shape mismatch, using forward fill instead")
                        df[categorical_cols] = df[categorical_cols].fillna(method='ffill').fillna('Unknown')
                except Exception as e:
                    logger.warning(f"Categorical imputation failed: {e}, using forward fill instead")
                    df[categorical_cols] = df[categorical_cols].fillna(method='ffill').fillna('Unknown')
            
            # Final cleanup - replace any remaining NaN values
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Missing value handling failed: {e}")
            # Fallback: simple fillna
            return df.fillna(0)
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Exclude target and ID columns from scaling
        exclude_cols = ['Position', 'Driver_Encoded', 'Team_Encoded', 'Circuit_Encoded', 'Year', 'Race_Num']
        scale_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(scale_cols) > 0:
            scaler = RobustScaler()
            df[scale_cols] = scaler.fit_transform(df[scale_cols])
            self.scalers['main'] = scaler
        
        return df
    
    def _validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean final features."""
        # Remove any remaining infinite or NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Ensure required columns exist
        required_cols = ['Position', 'Driver', 'Team', 'Circuit', 'Year']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Feature validation completed. Final shape: {df.shape}")
        return df
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: str = 'Position') -> Dict[str, float]:
        """Calculate feature importance scores."""
        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} not found")
            return {}
        
        # Select numeric features only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        if len(feature_cols) == 0:
            return {}
        
        try:
            # Use mutual information for feature importance
            X = df[feature_cols].fillna(0)
            y = df[target_col].fillna(df[target_col].median())
            
            importance_scores = mutual_info_regression(X, y, random_state=42)
            
            importance_dict = dict(zip(feature_cols, importance_scores))
            
            # Sort by importance
            self.feature_importance = dict(sorted(importance_dict.items(), 
                                                key=lambda x: x[1], reverse=True))
            
            logger.info(f"Calculated importance for {len(self.feature_importance)} features")
            return self.feature_importance
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            return {}
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of engineered features."""
        summary = {
            'total_features': len(self.feature_importance),
            'encoders_fitted': len(self.label_encoders) + len(self.target_encoders),
            'top_10_features': list(self.feature_importance.keys())[:10] if self.feature_importance else [],
            'circuit_categories': list(config.CIRCUIT_CATEGORIES.keys()),
            'rolling_windows': config.ROLLING_WINDOWS
        }
        
        return summary
    
    def _clean_position(self, position: Any) -> Optional[int]:
        """Clean and convert position to integer."""
        if pd.isna(position):
            return None
        
        try:
            # Handle various position formats
            if isinstance(position, str):
                position_upper = position.upper().strip()
                if position_upper in ['DNF', 'DNS', 'DSQ', 'NC', 'EX']:
                    return None
                # Extract numeric part
                import re
                numbers = re.findall(r'\d+', position)
                if numbers:
                    pos_int = int(numbers[0])
                    # Validate reasonable position range
                    return pos_int if 1 <= pos_int <= 30 else None
            else:
                pos_int = int(float(position))
                # Validate reasonable position range
                return pos_int if 1 <= pos_int <= 30 else None
        except (ValueError, TypeError):
            return None
        
        return None

