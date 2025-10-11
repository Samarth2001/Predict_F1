import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple

from .data_loader import F1DataLoader
from .feature_engineering_pipeline import FeatureEngineeringPipeline

logger = logging.getLogger(__name__)


class PredictionFeatureBuilder:
    """Builds inference-time features for upcoming race rows using historical data.

    This builder avoids re-merging target labels for upcoming events and returns only
    the rows corresponding to the provided upcoming dataframe with engineered features.
    """

    def __init__(self, data_loader: Optional[F1DataLoader] = None):
        self.data_loader = data_loader or F1DataLoader()

    def build(self, upcoming_df: pd.DataFrame) -> pd.DataFrame:
        """Return engineered features for the upcoming rows.

        Args:
            upcoming_df: DataFrame with upcoming race lineup and minimal columns
                         such as ['Year', 'Race_Num', 'Driver', 'Team', 'Date', ...]

        Returns:
            DataFrame of features aligned to upcoming_df rows (same order/length).
        """
        try:
            hist_races, hist_quali = self.data_loader.load_all_data()
            if hist_races is None or hist_quali is None:
                logger.error("Could not load historical data for feature engineering.")
                return pd.DataFrame()

                                   
            upcoming_df = upcoming_df.copy()
                                                                                      
            if 'Date' not in upcoming_df.columns or upcoming_df['Date'].isna().all():
                try:
                    year_val = int(upcoming_df.iloc[0]['Year']) if 'Year' in upcoming_df.columns else None
                    race_name = str(upcoming_df.iloc[0].get('Race_Name', '')) if 'Race_Name' in upcoming_df.columns else ''
                    if year_val is not None and race_name:
                        meta = self.data_loader._get_event_meta(year_val, race_name)
                        if meta and meta.get('event_date') is not None:
                            event_dt = pd.to_datetime(meta['event_date'], errors='coerce')
                            upcoming_df['Date'] = event_dt
                except Exception:
                                                                                      
                    try:
                        if 'Year' in upcoming_df.columns:
                            fallback_year = int(upcoming_df.iloc[0]['Year'])
                            upcoming_df['Date'] = pd.Timestamp(fallback_year, 1, 1)
                    except Exception:
                        pass
            upcoming_df['Date'] = pd.to_datetime(upcoming_df.get('Date'), errors='coerce')
            hist_races['Date'] = pd.to_datetime(hist_races['Date'], errors='coerce')
            hist_quali['Date'] = pd.to_datetime(hist_quali['Date'], errors='coerce')

                                                                                                                    
            if 'Position' in hist_quali.columns and 'Quali_Pos' not in hist_quali.columns:
                cols = [c for c in ['Year', 'Race_Num', 'Driver', 'Date', 'Position'] if c in hist_quali.columns]
                historical_quali = hist_quali[cols].rename(columns={'Position': 'Quali_Pos'})
            else:
                keep_cols = [c for c in ['Year', 'Race_Num', 'Driver', 'Date', 'Quali_Pos'] if c in hist_quali.columns]
                historical_quali = hist_quali[keep_cols].copy()
                                                                                                   
            if 'Date' not in historical_quali.columns:
                historical_quali['Date'] = pd.NaT

                                                                                                      
            if 'Quali_Pos' in upcoming_df.columns:
                quali_override = upcoming_df[['Year', 'Race_Num', 'Driver', 'Quali_Pos']].copy()
                historical_quali = pd.concat([historical_quali, quali_override], ignore_index=True)
                historical_quali.sort_values(['Year', 'Race_Num'], inplace=True)
                historical_quali.drop_duplicates(['Year', 'Race_Num', 'Driver'], keep='last', inplace=True)

                                                                                            
            combined_races = pd.concat([hist_races, upcoming_df], ignore_index=True) if not hist_races.empty else upcoming_df

                                                                         
            pipeline = FeatureEngineeringPipeline(combined_races, historical_quali)
            all_features = pipeline.run()

                                                                  
            upcoming_features = all_features.tail(len(upcoming_df)).copy()

                                                                                                            
            upcoming_features.drop(columns=[c for c in ['Position'] if c in upcoming_features.columns], inplace=True, errors='ignore')

            return upcoming_features
        except Exception as e:
            logger.error(f"Prediction feature building failed: {e}")
            return pd.DataFrame()




