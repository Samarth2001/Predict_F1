                                              

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
import inspect
import sys

from .config import config
from .utils import downcast_dataframe, pick_group_col, safe_merge

logger = logging.getLogger(__name__)


class BaseFeatureEngineer(ABC):
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @abstractmethod
    def engineer_features(self) -> pd.DataFrame:
        pass


class RegulationEraEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        """Add simple regulation-era categorical features.

        - Era_2022plus: 1 if Year >= 2022 else 0
        - Era_Pre2022: 1 if Year < 2022 else 0

        Keeps features numeric and low-cardinality to stabilize across rule changes.
        """
        if "Year" not in self.df.columns:
            return self.df
        try:
            y = pd.to_numeric(self.df["Year"], errors="coerce")
        except Exception:
            return self.df
        self.df["Era_2022plus"] = (y >= 2022).astype(int)
        self.df["Era_Pre2022"] = (y < 2022).astype(int)
        return self.df

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
        separate_by_sprint: bool = bool(
            config.get("feature_engineering.sprint_handling.separate_windows", True)
        )
        if "Is_Sprint" not in self.df.columns:
            self.df["Is_Sprint"] = 0
        drop_partial = bool(
            config.get("feature_engineering.rolling_windows_hygiene.drop_partial_windows", True)
        )

        for metric in metrics:
            if metric in self.df.columns:
                for window in windows:
                    col_name = f'{entity}_Avg_{metric}_{window["name"]}'
                    group_col = pick_group_col(self.df, entity)
                    minp = int(window["size"]) if drop_partial else 1
                    if separate_by_sprint:
                        group_cols = [group_col, "Is_Sprint"]
                        self.df[col_name] = self.df.groupby(group_cols)[metric].transform(
                            lambda s: s.shift(1).rolling(window=window["size"], min_periods=minp).mean()
                        )
                    else:
                        self.df[col_name] = self.df.groupby(group_col)[metric].transform(
                            lambda s: s.shift(1).rolling(window=window["size"], min_periods=minp).mean()
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
        separate_by_sprint: bool = bool(
            config.get("feature_engineering.sprint_handling.separate_windows", True)
        )
        if "Is_Sprint" not in self.df.columns:
            self.df["Is_Sprint"] = 0
        for metric in apply_to:
            if metric not in self.df.columns:
                continue
            for name, span in spans.items():
                drv_group = [pick_group_col(self.df, "Driver")] + (["Is_Sprint"] if separate_by_sprint else [])
                team_group = [pick_group_col(self.df, "Team")] + (["Is_Sprint"] if separate_by_sprint else [])

                self.df[f"Driver_EWM_{metric}_{name}"] = self.df.groupby(drv_group)[metric].transform(
                    lambda s: s.shift(1).ewm(span=span, min_periods=1, adjust=False).mean()
                )

                self.df[f"Team_EWM_{metric}_{name}"] = self.df.groupby(team_group)[metric].transform(
                    lambda s: s.shift(1).ewm(span=span, min_periods=1, adjust=False).mean()
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
        self.df = safe_merge(self.df, agg, on=["Circuit_Cluster"], how="left", join_name="circuit_agg")
        return self.df


class CircuitHistoryEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        """Per-circuit historical form for drivers and teams.

        Adds prior-at-this-circuit aggregates using only past information:
        - Driver_Circuit_Avg_Quali, Driver_Circuit_Avg_Race
        - Team_Circuit_Avg_Quali, Team_Circuit_Avg_Race
        - Driver_Circuit_Starts_Prev, Team_Circuit_Starts_Prev
        - Driver_Circuit_WinRate_Prev, Driver_Circuit_PodiumRate_Prev
        - Team_Circuit_WinRate_Prev, Team_Circuit_PodiumRate_Prev
        """
        required = {"Year", "Race_Num", "Circuit"}
        if not required.issubset(self.df.columns):
            return self.df

        df = self.df
        df.sort_values(["Year", "Race_Num"], inplace=True)

        # Numeric helpers for race outcomes
        pos_num = pd.to_numeric(df.get("Position"), errors="coerce")
        is_win = (pos_num == 1).astype(float) if pos_num is not None else pd.Series(0.0, index=df.index)
        is_podium = (pos_num <= 3).astype(float) if pos_num is not None else pd.Series(0.0, index=df.index)

        for entity in ["Driver", "Team"]:
            group_entity = pick_group_col(df, entity)
            if group_entity not in df.columns:
                continue
            key = [group_entity, "Circuit"]

            # Prior starts count at this circuit
            df[f"{entity}_Circuit_Starts_Prev"] = df.groupby(key).cumcount()

            # Averages of prior results at this circuit (qualifying and race)
            if "Quali_Pos" in df.columns:
                df[f"{entity}_Circuit_Avg_Quali"] = (
                    df.groupby(key, group_keys=False)["Quali_Pos"].apply(
                        lambda s: pd.to_numeric(s, errors="coerce").shift(1).expanding().mean()
                    )
                )
            if "Position" in df.columns:
                df[f"{entity}_Circuit_Avg_Race"] = (
                    df.groupby(key, group_keys=False)["Position"].apply(
                        lambda s: pd.to_numeric(s, errors="coerce").shift(1).expanding().mean()
                    )
                )

            # Win/podium rates using only prior events at this circuit
            wins_prev_cum = (
                df.groupby(key, group_keys=False)
                .apply(lambda g: is_win.loc[g.index].shift(1).cumsum())
                .values
            )
            pods_prev_cum = (
                df.groupby(key, group_keys=False)
                .apply(lambda g: is_podium.loc[g.index].shift(1).cumsum())
                .values
            )
            starts_prev = df[f"{entity}_Circuit_Starts_Prev"].replace(0, np.nan)
            df[f"{entity}_Circuit_WinRate_Prev"] = (wins_prev_cum / starts_prev).fillna(0.0)
            df[f"{entity}_Circuit_PodiumRate_Prev"] = (pods_prev_cum / starts_prev).fillna(0.0)

        self.df = df
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
            self.df.groupby(pick_group_col(self.df, "Driver"))["Is_Sprint"]
            .apply(lambda s: s.shift(1).cumsum())
            .reset_index(level=0, drop=True)
        )
        self.df["Team_Sprint_Count_Prev"] = (
            self.df.groupby(pick_group_col(self.df, "Team"))["Is_Sprint"]
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


class WeatherNormalizationEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        """Standardize weather columns to numeric ranges and handle units implicitly.

        We avoid changing units to prevent breaking existing models. We coerce types and clip
        to reasonable bounds to ensure stability across sources (session vs forecast).
        """
        cols = ["AirTemp", "Humidity", "Pressure", "WindSpeed", "Rainfall"]
        present = [c for c in cols if c in self.df.columns]
        if not present:
            return self.df
        df = self.df
        try:
            for c in present:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            if "AirTemp" in present:
                df["AirTemp"] = df["AirTemp"].clip(lower=-20, upper=60)
            if "Humidity" in present:
                df["Humidity"] = df["Humidity"].clip(lower=0, upper=100)
            if "Pressure" in present:
                df["Pressure"] = df["Pressure"].clip(lower=850, upper=1100)
            if "WindSpeed" in present:
                df["WindSpeed"] = df["WindSpeed"].clip(lower=0, upper=200)
            if "Rainfall" in present:
                df["Rainfall"] = df["Rainfall"].clip(lower=0, upper=100)
        except Exception:
            pass
        self.df = df
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
        drop_partial = bool(
            config.get("feature_engineering.rolling_windows_hygiene.drop_partial_windows", True)
        )
        for entity in ["Driver", "Team"]:
            for target, name in [("Quali_Pos", "Quali"), ("Position", "Race")]:
                if target not in self.df.columns:
                    continue
                group_entity = pick_group_col(self.df, entity)
                key_cols = [group_entity, "Circuit_Cluster"]
                shifted = self.df.groupby(key_cols)[target].shift(1)
                self.df[f"{entity}_Cluster_Avg_{name}"] = shifted.groupby(
                    [self.df[group_entity], self.df["Circuit_Cluster"]]
                ).transform(lambda s: s.rolling(window=5, min_periods=(5 if drop_partial else 1)).mean())
        return self.df


class QualiDeltaEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        logger.info("Engineering qualifying delta features...")
        if not {"Team", "Quali_Pos"}.issubset(self.df.columns):
            return self.df
        team_col = pick_group_col(self.df, "Team")
        team_best = self.df.groupby(["Year", "Race_Num", team_col])[
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
                group_entity = pick_group_col(self.df, entity)
                key = [group_entity, "Circuit_Cluster"]
                drop_partial = bool(
                    config.get("feature_engineering.rolling_windows_hygiene.drop_partial_windows", True)
                )
                self.df[f"{entity}_Cluster_Uplift"] = (
                    self.df.groupby(key)["Grid_Delta"]
                    .shift(1)
                    .groupby([self.df[group_entity], self.df["Circuit_Cluster"]])
                    .transform(lambda s: s.rolling(window=6, min_periods=(6 if drop_partial else 1)).mean())
                )
        if "Overtake_Difficulty" in self.df.columns:
            self.df["Adj_Grid_Delta"] = self.df["Grid_Delta"] * (
                1.0 - self.df["Overtake_Difficulty"].rank(pct=True)
            )
        return self.df


class TypeNormalizerEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        """Normalize dtypes for consistency and performance.

        - Cast entity columns and small-int flags to category
        - Downcast numeric columns to minimal types
        """
        df = self.df
        for col in [
            "Driver",
            "Team",
            "Circuit",
            "Driver_ID",
            "Team_ID",
            "Circuit_ID",
            "Era_2022plus",
            "Era_Pre2022",
            "Is_Sprint",
        ]:
            if col in df.columns:
                try:
                    df[col] = df[col].astype("category")
                except Exception:
                    pass
        try:
            df = downcast_dataframe(df)
        except Exception:
            pass
        self.df = df
        return self.df


class ScenarioGateEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        """Drop columns that are disallowed for pre-quali scenarios to avoid leakage.

        If column 'Post_Quali' exists and contains any 0 values, drop columns listed in
        config under feature_engineering.scenario_gate.pre_quali_drop.
        """
        try:
            gate_cfg = config.get("feature_engineering.scenario_gate", {}) or {}
            if not bool(gate_cfg.get("enabled", True)):
                return self.df
            if "Post_Quali" not in self.df.columns:
                return self.df
            if int(pd.to_numeric(self.df["Post_Quali"], errors="coerce").fillna(0).max()) == 1:
                # All ones => post-quali only; nothing to drop
                return self.df
            drop_list = gate_cfg.get("pre_quali_drop", []) or []
            existing = [c for c in drop_list if c in self.df.columns]
            if existing:
                self.df.drop(columns=existing, inplace=True, errors="ignore")
        except Exception:
            pass
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



class AdvancedStatsEngineer(BaseFeatureEngineer):
    def engineer_features(self) -> pd.DataFrame:
        logger.info("Engineering advanced stats (DNF rate, overtaking, consistency)...")
        df = self.df
        
        # 1. DNF Rate
        if "Status" in df.columns and "Driver" in df.columns:
            dnf_keywords = config.get("feature_engineering.dnf_statuses", ["DNF", "Retired", "Accident", "Collision", "Mechanical"])
            # Ensure we match against string representation
            is_dnf = df["Status"].astype(str).apply(
                lambda x: 1 if any(kw in x for kw in dnf_keywords) else 0
            )
            # We need to respect time order
            df.sort_values(["Year", "Race_Num"], inplace=True)
            group_col = pick_group_col(df, "Driver")
            
            # Rolling mean of DNF over last 10 races
            # We shift by 1 to avoid leakage (using current race status for current prediction)
            df["Driver_DNF_Rate_Last10"] = (
                is_dnf.groupby(df[group_col])
                .transform(lambda s: s.shift(1).rolling(window=10, min_periods=3).mean())
            )
            df["Driver_DNF_Rate_Last10"] = df["Driver_DNF_Rate_Last10"].fillna(0.0)

        # 2. Overtaking (Positions Gained)
        if {"Position", "Grid"}.issubset(df.columns) and "Driver" in df.columns:
            # Calculate positions gained for previous races
            # Note: This uses current race outcome, so we must shift it for prediction features
            pos_gained = df["Grid"] - df["Position"]
            group_col = pick_group_col(df, "Driver")
            
            df["Driver_Overtaking_Avg_Last5"] = (
                pos_gained.groupby(df[group_col])
                .transform(lambda s: s.shift(1).rolling(window=5, min_periods=2).mean())
            )
            df["Driver_Overtaking_Avg_Last5"] = df["Driver_Overtaking_Avg_Last5"].fillna(0.0)

        # 3. Top 5 Conversion
        if {"Position", "Grid"}.issubset(df.columns):
            started_top5 = (df["Grid"] <= 5).astype(int)
            finished_top5 = (df["Position"] <= 5).astype(int)
            conversion = (started_top5 & finished_top5).astype(int)
            
            group_col = pick_group_col(df, "Driver")
            df["Driver_Top5_Conversion_Rate"] = (
                conversion.groupby(df[group_col])
                .transform(lambda s: s.shift(1).rolling(window=10, min_periods=2).mean())
            )
            df["Driver_Top5_Conversion_Rate"] = df["Driver_Top5_Conversion_Rate"].fillna(0.0)

        # 4. Team Consistency
        if "Team" in df.columns and "Position" in df.columns:
            group_col = pick_group_col(df, "Team")
            df["Team_Consistency_Std_Last5"] = (
                df.groupby(group_col)["Position"]
                .transform(lambda s: s.shift(1).rolling(window=5, min_periods=2).std())
            )
            df["Team_Consistency_Std_Last5"] = df["Team_Consistency_Std_Last5"].fillna(0.0)

        self.df = df
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
            "CircuitHistoryEngineer": CircuitHistoryEngineer,
            "TeammateFeaturesEngineer": TeammateFeaturesEngineer,
            "SprintFeatureEngineer": SprintFeatureEngineer,
            "WeatherNormalizationEngineer": WeatherNormalizationEngineer,
            "WeatherFeatureEngineer": WeatherFeatureEngineer,
            "SeasonProgressEngineer": SeasonProgressEngineer,
            "ClusterPerformanceEngineer": ClusterPerformanceEngineer,
            "QualiDeltaEngineer": QualiDeltaEngineer,
            "GridUpliftEngineer": GridUpliftEngineer,
            "RegulationEraEngineer": RegulationEraEngineer,
            "TypeNormalizerEngineer": TypeNormalizerEngineer,
            "ScenarioGateEngineer": ScenarioGateEngineer,
            "InteractionFeaturesEngineer": InteractionFeaturesEngineer,
            "AdvancedStatsEngineer": AdvancedStatsEngineer,
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
        if "Status" in races_df.columns:
            # Missing Status for upcoming/inference rows should not be treated as DNF.
            races_df["DNF"] = races_df["Status"].apply(
                lambda x: 0 if pd.isna(x) else (1 if x not in finished_statuses else 0)
            )
        else:
            races_df["DNF"] = np.nan

        quali_df["Date"] = pd.to_datetime(quali_df["Date"], errors="coerce")
        if "Quali_Pos" not in quali_df.columns:
            if "Position" in quali_df.columns:
                quali_df = quali_df.rename(columns={"Position": "Quali_Pos"})
            else:
                quali_df["Quali_Pos"] = np.nan
                                                      
        quali_to_merge = quali_df[["Year", "Race_Num", "Driver", "Driver_ID", "Quali_Pos"]].copy()
        quali_to_merge.sort_values(["Year", "Race_Num"], inplace=True)
        quali_to_merge.drop_duplicates(
            ["Year", "Race_Num", "Driver_ID"], keep="last", inplace=True
        )
        # Prefer merging on IDs to avoid aliasing issues
        if "Driver_ID" in races_df.columns:
            df = safe_merge(
                races_df,
                quali_to_merge,
                on=["Year", "Race_Num", "Driver_ID"],
                how="left",
                join_name="races_x_quali_by_driver_id",
            )
        else:
            df = safe_merge(
                races_df,
                quali_to_merge.drop(columns=["Driver_ID"], errors="ignore"),
                on=["Year", "Race_Num", "Driver"],
                how="left",
                join_name="races_x_quali_by_driver",
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
                try:
                    module_src = inspect.getsource(sys.modules[__name__])
                except Exception:
                    module_src = ""
                code_sig = hashlib.sha256(module_src.encode("utf-8")).hexdigest()
                try:
                    if {"Year", "Race_Num"}.issubset(self.hist_races.columns):
                        last_year = int(pd.to_numeric(self.hist_races["Year"], errors="coerce").max())
                        sub = self.hist_races[pd.to_numeric(self.hist_races["Year"], errors="coerce") == last_year]
                        last_round = int(pd.to_numeric(sub.get("Race_Num"), errors="coerce").max()) if not sub.empty else -1
                    else:
                        last_year, last_round = -1, -1
                except Exception:
                    last_year, last_round = -1, -1
                scenario_flag = "post" if ("Post_Quali" in df.columns and pd.to_numeric(df["Post_Quali"], errors="coerce").fillna(0).max() >= 1) else "pre"
                sprint_flag = int(df.get("Is_Sprint", pd.Series([0])).max()) if "Is_Sprint" in df.columns else 0
                key_raw = json.dumps({
                    "races": races_sig,
                    "quali": quali_sig,
                    "steps": steps_sig,
                    "cfg": cfg_sig,
                    "code": code_sig,
                    "event": {"year": last_year, "round": last_round, "scenario": scenario_flag, "sprint": sprint_flag},
                }, sort_keys=True)
                cache_key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
                cache_path = os.path.join(self._cache_dir, f"features_{cache_key}.pkl")
                try:
                    sig_path = os.path.join(self._cache_dir, "cache_code_sig.txt")
                    prev_sig = None
                    if os.path.exists(sig_path):
                        with open(sig_path, "r", encoding="utf-8") as f:
                            prev_sig = f.read().strip()
                    if prev_sig != code_sig:
                        for fname in os.listdir(self._cache_dir):
                            if fname.startswith("features_") and fname.endswith(".pkl"):
                                try:
                                    os.remove(os.path.join(self._cache_dir, fname))
                                except Exception:
                                    pass
                        with open(sig_path, "w", encoding="utf-8") as f:
                            f.write(code_sig)
                except Exception:
                    pass
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
            # Freeze canonical ordered feature list for downstream training/prediction
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            # Keep grouping/meta columns at front if present
            front = [c for c in ["Year", "Race_Num", "Date", "Driver_ID", "Team_ID", "Circuit_ID", "Event_ID"] if c in df.columns]
            ordered = front + [c for c in numeric_cols if c not in front]
            df = df[[c for c in ordered if c in df.columns]]
        except Exception:
            pass
                             
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
