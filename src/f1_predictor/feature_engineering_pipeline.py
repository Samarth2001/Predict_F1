                                              

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import os
import json
import hashlib
import joblib
from sklearn.cluster import KMeans

from .config import config
from .utils import downcast_dataframe

logger = logging.getLogger(__name__)


class BaseFeatureEngineer(ABC):
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @abstractmethod
    def engineer_features(self) -> pd.DataFrame:
        pass


class RollingPerformanceEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        logger.info("Engineering rolling performance features...")
        self.df.sort_values(by=["Year", "Race_Num"], inplace=True)
        self._calculate_rolling_metrics_for_entity("Driver")
        self._calculate_rolling_metrics_for_entity("Team")
        return self.df.sort_index()

    def _calculate_rolling_metrics_for_entity(self, entity: str):
        windows = config.get(
            "feature_engineering.rolling_windows",
            [{"name": "short", "size": 3}, {"name": "medium", "size": 5}],
        )
        metrics = config.get(
            "feature_engineering.rolling_metrics", ["Position", "Quali_Pos", "DNF"]
        )

        for metric in metrics:
            if metric in self.df.columns:
                for window in windows:
                    col_name = f'{entity}_Avg_{metric}_{window["name"]}'
                    self.df[col_name] = (
                        self.df.groupby(entity)[metric]
                        .shift(1)
                        .rolling(window=window["size"], min_periods=1)
                        .mean()
                    )


class TimeDecayEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        if not config.get("feature_engineering.time_decay.enabled", True):
            return self.df
        logger.info("Engineering time-decay features (EWM)...")
        apply_to: List[str] = config.get(
            "feature_engineering.time_decay.apply_to", ["Position", "Quali_Pos", "DNF"]
        )
        spans: Dict[str, int] = config.get(
            "feature_engineering.time_decay.spans",
            {"short": 3, "medium": 5, "long": 10},
        )
        for metric in apply_to:
            if metric not in self.df.columns:
                continue
            for name, span in spans.items():
                                  
                self.df[f"Driver_EWM_{metric}_{name}"] = (
                    self.df.groupby("Driver")[metric]
                    .apply(
                        lambda s: s.shift(1)
                        .ewm(span=span, min_periods=1, adjust=False)
                        .mean()
                    )
                    .reset_index(level=0, drop=True)
                )
                                
                self.df[f"Team_EWM_{metric}_{name}"] = (
                    self.df.groupby("Team")[metric]
                    .apply(
                        lambda s: s.shift(1)
                        .ewm(span=span, min_periods=1, adjust=False)
                        .mean()
                    )
                    .reset_index(level=0, drop=True)
                )
        return self.df


class CircuitTypeEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        logger.info("Engineering circuit type and cluster features...")
        categories: Dict[str, List[str]] = config.get(
            "feature_engineering.circuit_categories", {}
        )
                                                                                                          
        norm_df_circuit = self.df["Circuit"].astype(str)
        missing_suffix_mask = ~norm_df_circuit.str.endswith(" Grand Prix", na=False)
        norm_df_circuit = norm_df_circuit.where(
            ~missing_suffix_mask, norm_df_circuit + " Grand Prix"
        )
        for cat_name, circuits in categories.items():
            circuits_norm = [
                (
                    c
                    if isinstance(c, str) and c.endswith(" Grand Prix")
                    else f"{c} Grand Prix"
                )
                for c in (circuits or [])
            ]
            self.df[f"CircuitType_{cat_name}"] = norm_df_circuit.isin(
                circuits_norm
            ).astype(int)

        pit_loss_by_circuit: Dict[str, float] = config.get(
            "feature_engineering.track_meta.pit_loss_by_circuit", {}
        )
        overtake_by_circuit: Dict[str, float] = config.get(
            "feature_engineering.track_meta.overtake_difficulty_by_circuit", {}
        )
        expected_stops_by_circuit: Dict[str, float] = config.get(
            "feature_engineering.track_meta.expected_stops_by_circuit", {}
        )
        sc_probability_by_circuit: Dict[str, float] = config.get(
            "feature_engineering.track_meta.sc_probability_by_circuit", {}
        )
        track_position_importance_by_circuit: Dict[str, float] = config.get(
            "feature_engineering.track_meta.track_position_importance_by_circuit", {}
        )
        tyre_degradation_by_circuit: Dict[str, float] = config.get(
            "feature_engineering.track_meta.tyre_degradation_by_circuit", {}
        )

                                                                                                
        def _with_suffix_keys(d: Dict[str, float]) -> Dict[str, float]:
            out: Dict[str, float] = {}
            for k, v in (d or {}).items():
                if isinstance(k, str):
                    out[k] = v
                    if not k.endswith(" Grand Prix"):
                        out[f"{k} Grand Prix"] = v
            return out

        pit_loss_by_circuit = _with_suffix_keys(pit_loss_by_circuit)
        overtake_by_circuit = _with_suffix_keys(overtake_by_circuit)
        expected_stops_by_circuit = _with_suffix_keys(expected_stops_by_circuit)
        sc_probability_by_circuit = _with_suffix_keys(sc_probability_by_circuit)
        track_position_importance_by_circuit = _with_suffix_keys(
            track_position_importance_by_circuit
        )
        tyre_degradation_by_circuit = _with_suffix_keys(tyre_degradation_by_circuit)

        self.df["Pit_Loss_Time"] = norm_df_circuit.map(pit_loss_by_circuit)
        self.df["Overtake_Difficulty"] = norm_df_circuit.map(overtake_by_circuit)
        self.df["Expected_Pit_Stops"] = norm_df_circuit.map(expected_stops_by_circuit)
        self.df["SC_Probability"] = norm_df_circuit.map(sc_probability_by_circuit)
        self.df["Track_Pos_Importance"] = norm_df_circuit.map(
            track_position_importance_by_circuit
        )
        self.df["Tyre_Degradation"] = norm_df_circuit.map(tyre_degradation_by_circuit)
                                  
        for col in [
            "Pit_Loss_Time",
            "Overtake_Difficulty",
            "Expected_Pit_Stops",
            "SC_Probability",
            "Track_Pos_Importance",
            "Tyre_Degradation",
        ]:
            if col in self.df.columns:
                med = self.df[col].median() if not self.df[col].dropna().empty else 0.0
                self.df[col] = self.df[col].fillna(med)

                                                                     
        try:
            cluster_k = int(
                config.get("feature_engineering.track_meta.num_circuit_clusters", 4)
            )
            cluster_features = [
                c for c in self.df.columns if c.startswith("CircuitType_")
            ] + ["Pit_Loss_Time", "Overtake_Difficulty"]
            X = self.df[cluster_features].fillna(0.0).drop_duplicates().values
            if X.shape[0] >= cluster_k and cluster_k > 1:
                kmeans = KMeans(n_clusters=cluster_k, random_state=42, n_init=10)
                kmeans.fit(self.df[cluster_features].fillna(0.0))
                self.df["Circuit_Cluster"] = kmeans.labels_
            else:
                self.df["Circuit_Cluster"] = 0
        except Exception as e:
            logger.warning(f"Circuit clustering failed: {e}")
            self.df["Circuit_Cluster"] = 0
        return self.df


class CircuitAggregatesEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        if "Circuit_Cluster" not in self.df.columns:
            return self.df
        agg = (
            self.df.groupby(["Circuit_Cluster"])
            .agg(
                Cluster_Avg_SC_Prob=("SC_Probability", "mean"),
                Cluster_Avg_Overtake=("Overtake_Difficulty", "mean"),
                Cluster_Avg_PitLoss=("Pit_Loss_Time", "mean"),
            )
            .reset_index()
        )
        self.df = self.df.merge(agg, on="Circuit_Cluster", how="left")
        return self.df


class TeammateFeaturesEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        if not config.get("feature_engineering.teammate_features.enabled", True):
            return self.df
        logger.info("Engineering teammate gap features...")

                                            
        def _maybe(col: str) -> bool:
            return col in self.df.columns

                                              
        if _maybe("Driver_Avg_Quali_Pos_short") and _maybe("Team_Avg_Quali_Pos_short"):
            self.df["Teammate_Gap_Quali_short"] = (
                self.df["Driver_Avg_Quali_Pos_short"]
                - self.df["Team_Avg_Quali_Pos_short"]
            )
        if _maybe("Driver_Avg_Position_short") and _maybe("Team_Avg_Position_short"):
            self.df["Teammate_Gap_Race_short"] = (
                self.df["Driver_Avg_Position_short"]
                - self.df["Team_Avg_Position_short"]
            )
        return self.df


class SprintFeatureEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        if not config.get("feature_engineering.sprint_handling.enabled", True):
            return self.df
        logger.info("Engineering sprint-related features...")
        if "Is_Sprint" not in self.df.columns:
            self.df["Is_Sprint"] = 0
        self.df["Is_Sprint"] = self.df["Is_Sprint"].fillna(0).astype(int)
                                             
        self.df.sort_values(["Year", "Race_Num"], inplace=True)
        self.df["Driver_Sprint_Count_Prev"] = (
            self.df.groupby("Driver")["Is_Sprint"]
            .apply(lambda s: s.shift(1).cumsum())
            .reset_index(level=0, drop=True)
        )
        self.df["Team_Sprint_Count_Prev"] = (
            self.df.groupby("Team")["Is_Sprint"]
            .apply(lambda s: s.shift(1).cumsum())
            .reset_index(level=0, drop=True)
        )
        return self.df


class WeatherFeatureEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        logger.info("Engineering weather features...")
        weather_cols = ["AirTemp", "Humidity", "Pressure", "WindSpeed", "Rainfall"]
        if not all(col in self.df.columns for col in weather_cols):
            logger.warning("Weather data columns not found. Skipping weather features.")
            return self.df

        rain_threshold = config.get(
            "feature_engineering.weather_impact.rain_probability_threshold", 30
        )
        self.df["is_wet_race"] = (self.df["Rainfall"] > rain_threshold).astype(int)

        optimal_temp_range = config.get(
            "feature_engineering.weather_impact.optimal_temp_range", [20, 30]
        )
        optimal_temp = np.mean(optimal_temp_range)
        self.df["temp_deviation"] = (self.df["AirTemp"] - optimal_temp).abs()

        wind_threshold = config.get(
            "feature_engineering.weather_impact.wind_speed_threshold", 25
        )
        self.df["is_windy"] = (self.df["WindSpeed"] > wind_threshold).astype(int)
        return self.df


class SeasonProgressEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        logger.info("Engineering season progress features...")
        if not {"Year", "Race_Num"}.issubset(self.df.columns):
            return self.df
        max_rounds = self.df.groupby("Year")["Race_Num"].transform("max")
        self.df["Season_Progress"] = self.df["Race_Num"] / max_rounds.replace(0, np.nan)
        self.df["Season_Progress"] = self.df["Season_Progress"].fillna(0.0)
        return self.df


class ClusterPerformanceEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        logger.info("Engineering circuit cluster performance features...")
        if "Circuit_Cluster" not in self.df.columns:
            return self.df
        self.df.sort_values(["Year", "Race_Num"], inplace=True)
        for entity in ["Driver", "Team"]:
            for target, name in [("Quali_Pos", "Quali"), ("Position", "Race")]:
                if target not in self.df.columns:
                    continue
                key_cols = [entity, "Circuit_Cluster"]
                shifted = self.df.groupby(key_cols)[target].shift(1)
                self.df[f"{entity}_Cluster_Avg_{name}"] = shifted.groupby(
                    [self.df[entity], self.df["Circuit_Cluster"]]
                ).transform(lambda s: s.rolling(window=5, min_periods=1).mean())
        return self.df


class QualiDeltaEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        logger.info("Engineering qualifying delta features...")
        if not {"Team", "Quali_Pos"}.issubset(self.df.columns):
            return self.df
        team_best = self.df.groupby(["Year", "Race_Num", "Team"])[
            "Quali_Pos"
        ].transform("min")
        self.df["Quali_Delta_Teammate"] = self.df["Quali_Pos"] - team_best
        return self.df


class GridUpliftEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        logger.info("Engineering grid uplift/drawdown features...")
        if not {"Grid", "Position"}.issubset(self.df.columns):
            return self.df
        self.df["Grid_Delta"] = self.df["Position"] - self.df["Grid"]
        if "Circuit_Cluster" in self.df.columns:
            for entity in ["Driver", "Team"]:
                key = [entity, "Circuit_Cluster"]
                self.df[f"{entity}_Cluster_Uplift"] = (
                    self.df.groupby(key)["Grid_Delta"]
                    .shift(1)
                    .groupby([self.df[entity], self.df["Circuit_Cluster"]])
                    .transform(lambda s: s.rolling(window=6, min_periods=1).mean())
                )
        if "Overtake_Difficulty" in self.df.columns:
            self.df["Adj_Grid_Delta"] = self.df["Grid_Delta"] * (
                1.0 - self.df["Overtake_Difficulty"].rank(pct=True)
            )
        return self.df


class InteractionFeaturesEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        df = self.df
        if (
            "Driver_Avg_Position_short" in df.columns
            and "Overtake_Difficulty" in df.columns
        ):
            df["Driver_FormShort_x_Overtake"] = (
                df["Driver_Avg_Position_short"] * df["Overtake_Difficulty"]
            )
        if (
            "Team_Avg_Position_short" in df.columns
            and "Overtake_Difficulty" in df.columns
        ):
            df["Team_FormShort_x_Overtake"] = (
                df["Team_Avg_Position_short"] * df["Overtake_Difficulty"]
            )
        if "Grid" in df.columns and "Track_Pos_Importance" in df.columns:
            df["Grid_x_TrackImportance"] = (
                df["Grid"].fillna(0) * df["Track_Pos_Importance"].fillna(0)
            )
        self.df = df
        return self.df


class QualiTimeEngineer(BaseFeatureEngineer):
    """Convert Q1/Q2/Q3 lap-time strings to seconds and compute pole-gap features."""

    @staticmethod
    def _to_seconds(series: pd.Series) -> pd.Series:
        """Parse timedelta-like strings to float seconds; returns NaN on failure."""
        def _parse(val):
            if pd.isna(val):
                return np.nan
            try:
                td = pd.to_timedelta(str(val), errors="coerce")
                return td.total_seconds() if not pd.isna(td) else np.nan
            except Exception:
                return np.nan
        return series.map(_parse)

    def engineer_features(self) -> pd.DataFrame:
        logger.info("Engineering qualifying time features...")
        q_cols = [c for c in ["Q1", "Q2", "Q3"] if c in self.df.columns]
        if not q_cols:
            return self.df

        for col in q_cols:
            self.df[f"{col}_s"] = self._to_seconds(self.df[col])

        # Best lap time available (Q3 > Q2 > Q1 priority)
        time_cols = [f"{c}_s" for c in ["Q3", "Q2", "Q1"] if f"{c}_s" in self.df.columns]
        self.df["Quali_Time_s"] = self.df[time_cols].bfill(axis=1).iloc[:, 0]

        # Gap to pole (fastest in that event)
        if "Quali_Time_s" in self.df.columns and {"Year", "Race_Num"}.issubset(self.df.columns):
            event_best = self.df.groupby(["Year", "Race_Num"])["Quali_Time_s"].transform("min")
            self.df["Quali_Gap_To_Pole"] = self.df["Quali_Time_s"] - event_best
            # Percentage gap (avoid division by zero)
            self.df["Quali_Gap_Pct"] = self.df["Quali_Gap_To_Pole"] / event_best.replace(0, np.nan)

        return self.df


class ChampionshipFeatureEngineer(BaseFeatureEngineer):
    """Cumulative championship points up to (but not including) the current race."""

    def engineer_features(self) -> pd.DataFrame:
        logger.info("Engineering championship standing features...")
        if "Points" not in self.df.columns:
            return self.df

        self.df.sort_values(["Year", "Race_Num"], inplace=True)

        for entity in ["Driver", "Team"]:
            if entity not in self.df.columns:
                continue
            # Season-cumulative points shifted by 1 to avoid leakage
            self.df[f"{entity}_Season_Points"] = (
                self.df.groupby(["Year", entity])["Points"]
                .apply(lambda s: s.shift(1).cumsum())
                .reset_index(level=[0, 1], drop=True)
                .fillna(0)
            )
        return self.df


class FeatureOptimizer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        logger.info(
            "Skipping global imputation to prevent leakage; deferring to train-time."
        )
        return self.df


class FeatureEngineeringPipeline:
    def __init__(self, hist_races: pd.DataFrame, hist_quali: pd.DataFrame):
        self.hist_races = hist_races
        self.hist_quali = hist_quali

        pipeline_steps = config.get("feature_engineering.pipeline_steps", [])
        self.engineers = self._load_engineers(pipeline_steps)
        self._cache_enabled: bool = bool(config.get("feature_engineering.cache.enabled", True))
        self._cache_dir: str = config.get("paths.features_cache_dir", os.path.join(config.get("paths.cache_dir"), "features_cache"))
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
        except Exception:
            pass

    def _load_engineers(self, steps: List[str]) -> List[BaseFeatureEngineer]:
        engineer_map = {
            "RollingPerformanceEngineer": RollingPerformanceEngineer,
            "TimeDecayEngineer": TimeDecayEngineer,
            "CircuitTypeEngineer": CircuitTypeEngineer,
            "CircuitAggregatesEngineer": CircuitAggregatesEngineer,
            "SeasonProgressEngineer": SeasonProgressEngineer,
            "ClusterPerformanceEngineer": ClusterPerformanceEngineer,
            "TeammateFeaturesEngineer": TeammateFeaturesEngineer,
            "SprintFeatureEngineer": SprintFeatureEngineer,
            "QualiDeltaEngineer": QualiDeltaEngineer,
            "QualiTimeEngineer": QualiTimeEngineer,
            "GridUpliftEngineer": GridUpliftEngineer,
            "InteractionFeaturesEngineer": InteractionFeaturesEngineer,
            "WeatherFeatureEngineer": WeatherFeatureEngineer,
            "ChampionshipFeatureEngineer": ChampionshipFeatureEngineer,
            "FeatureOptimizer": FeatureOptimizer,
        }

        loaded_engineers = []
        for step_name in steps:
            engineer_class = engineer_map.get(step_name)
            if engineer_class:
                loaded_engineers.append(engineer_class)
            else:
                logger.warning(
                    f"Feature engineering step '{step_name}' not found. Skipping."
                )
        return loaded_engineers

    def _prepare_data(self) -> pd.DataFrame:
        logger.info("Preparing initial data...")
        if self.hist_races is None or self.hist_quali is None:
            logger.error("Historical race or qualifying data is missing.")
            return pd.DataFrame()

        races_df = self.hist_races.copy()
        quali_df = self.hist_quali.copy()

        races_df["Date"] = pd.to_datetime(races_df["Date"], errors="coerce")
        races_df["Position"] = pd.to_numeric(races_df["Position"], errors="coerce")
        races_df["Grid"] = pd.to_numeric(races_df["Grid"], errors="coerce")

        finished_statuses = config.get(
            "feature_engineering.finished_statuses",
            ["Finished", "+1 Lap", "+2 Laps", "+3 Laps"],
        )
        races_df["DNF"] = races_df["Status"].apply(
            lambda x: 1 if x not in finished_statuses else 0
        )

        quali_df["Date"] = pd.to_datetime(quali_df["Date"], errors="coerce")
        if "Quali_Pos" not in quali_df.columns:
            if "Position" in quali_df.columns:
                quali_df = quali_df.rename(columns={"Position": "Quali_Pos"})
            else:
                quali_df["Quali_Pos"] = np.nan
                                                      
        q_time_cols = [c for c in ["Q1", "Q2", "Q3"] if c in quali_df.columns]
        quali_to_merge = quali_df[["Year", "Race_Num", "Driver", "Quali_Pos"] + q_time_cols].copy()
        quali_to_merge.sort_values(["Year", "Race_Num"], inplace=True)
        quali_to_merge.drop_duplicates(
            ["Year", "Race_Num", "Driver"], keep="last", inplace=True
        )
        df = pd.merge(
            races_df, quali_to_merge, on=["Year", "Race_Num", "Driver"], how="left"
        )
                                                                                                                 
        if "Quali_Pos_x" in df.columns and "Quali_Pos_y" in df.columns:
                                                                                         
            df["Quali_Pos"] = df["Quali_Pos_x"].where(
                ~df["Quali_Pos_x"].isna(), df["Quali_Pos_y"]
            )
            df.drop(columns=["Quali_Pos_x", "Quali_Pos_y"], inplace=True)
                                                    
        if "Grid" in df.columns and "Quali_Pos" in df.columns:
            df["Grid"] = df["Grid"].fillna(df["Quali_Pos"])
        return df

    def run(self) -> pd.DataFrame:
        logger.info("Running feature engineering pipeline...")
        df = self._prepare_data()
        if df.empty:
            return pd.DataFrame()

                                                                 
        cache_key = None
        cache_path = None
        if self._cache_enabled:
            try:
                races_hash = pd.util.hash_pandas_object(self.hist_races, index=True)
                quali_hash = pd.util.hash_pandas_object(self.hist_quali, index=True)
                races_sig = hashlib.sha256(races_hash.values.tobytes()).hexdigest()
                quali_sig = hashlib.sha256(quali_hash.values.tobytes()).hexdigest()
                steps_sig = hashlib.sha256(json.dumps(config.get("feature_engineering.pipeline_steps", []), sort_keys=True).encode("utf-8")).hexdigest()
                cfg_sig = getattr(config, "get_config_hash", lambda: "")( )
                key_raw = json.dumps({
                    "races": races_sig,
                    "quali": quali_sig,
                    "steps": steps_sig,
                    "cfg": cfg_sig,
                }, sort_keys=True)
                cache_key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
                cache_path = os.path.join(self._cache_dir, f"features_{cache_key}.pkl")
                if os.path.exists(cache_path):
                    logger.info("Loading engineered features from cache...")
                    cached = joblib.load(cache_path)
                    if isinstance(cached, pd.DataFrame) and not cached.empty:
                        return cached
            except Exception as e:
                logger.warning(f"Feature cache unavailable: {e}")

        for engineer_class in self.engineers:
            df = engineer_class(df).engineer_features()
                                                                        
            try:
                df = downcast_dataframe(df)
            except Exception:
                pass

        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
                             
        try:
            df = downcast_dataframe(df)
        except Exception:
            pass
                       
        if self._cache_enabled and cache_path:
            try:
                joblib.dump(df, cache_path)
                logger.info(f"Cached engineered features to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache engineered features: {e}")
        return df
