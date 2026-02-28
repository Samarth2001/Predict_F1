
import pandas as pd
import logging
import joblib
import os
from typing import Optional, List, Dict, Any
from datetime import datetime
import numpy as np
from scipy.stats import norm

from .config import config
from .data_loader import F1DataLoader
from .feature_engineering_pipeline import FeatureEngineeringPipeline
from .prediction_features import PredictionFeatureBuilder
from .simulation import RaceSimulator, SimulationResult

logger = logging.getLogger(__name__)


class F1Predictor:
    """Handles the end-to-end prediction process for qualifying and race sessions."""

    def __init__(self):
        self.models_dir = config.get("paths.models_dir")
        self.qualifying_model = self._load_model("qualifying")
        self.race_model = self._load_model("race")
        self.data_loader = F1DataLoader()
        self.pred_features = PredictionFeatureBuilder(self.data_loader)
                                          
        self.dnf_model = self._load_optional_model("race_dnf_model.pkl")

    def _load_model(self, model_type: str) -> Optional[any]:
        model_path = os.path.join(self.models_dir, f"{model_type}_model.pkl")
        if not os.path.exists(model_path):
            logger.error(
                f"{model_type.capitalize()} model not found at {model_path}. Please train it first."
            )
            return None
        try:
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            return None

    def _load_optional_model(self, filename: str) -> Optional[any]:
        try:
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path):
                return joblib.load(path)
        except Exception as e:
            logger.warning(f"Optional model '{filename}' failed to load: {e}")
        return None

    def predict_qualifying(
        self, year: int, race_name: str, scenario: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        if not self.qualifying_model:
            return None

        upcoming_df = self._get_upcoming_race_df(year, race_name)
        if upcoming_df.empty:
            return None

        features_df = self._prepare_features(upcoming_df)

        features_used = self._load_and_enforce_metadata("qualifying", features_df)
        if not features_used:
            logger.error(
                "Model metadata missing or no matching features found for qualifying. Aborting."
            )
            return None
                                                                          
        raw_scores = self.qualifying_model.predict(features_df[features_used])
        if bool(config.get("models.qualifying.use_ranking", False)):
            order = np.argsort(-raw_scores)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(raw_scores) + 1)
            positions = ranks.astype(float)
            results = self._format_results(
                upcoming_df,
                positions,
                "Quali_Pos",
                raw_scores=raw_scores,
                score_type="rank_score",
            )
        else:
            positions = raw_scores.astype(float)
            results = self._format_results(
                upcoming_df,
                positions,
                "Quali_Pos",
                raw_scores=None,
                score_type="position_value",
            )
        if "Predicted_Pos" in results.columns:
            results = results.rename(columns={"Predicted_Pos": "Predicted_Quali_Pos"})
                                        
        try:
            self._save_prediction_results(
                "qualifying",
                year,
                race_name,
                results,
                features_used=features_used,
                scenario=scenario or "qualifying",
            )
        except Exception:
            pass
        return results

    def predict_race(
        self, year: int, race_name: str, mode: str = "auto"
    ) -> Optional[pd.DataFrame]:
        if not self.race_model:
            return None

        upcoming_df = self._get_upcoming_race_df(year, race_name)
        if upcoming_df.empty:
            return None

                                                                   
                                                   
                                                                    
                                                                     
                                                                     
        hist_races, hist_quali = self.data_loader.load_all_data()
        if not upcoming_df.empty:
            yr = int(upcoming_df.iloc[0]["Year"])
            rnd = int(upcoming_df.iloc[0]["Race_Num"])
            actual_quali = pd.DataFrame()
            if not hist_quali.empty:
                actual_quali = hist_quali[
                    (hist_quali["Year"] == yr) & (hist_quali["Race_Num"] == rnd)
                ]

            if mode == "post_quali":
                if actual_quali.empty:
                    logger.error(
                        "Post-quali mode requested, but no actual qualifying results are available yet."
                    )
                    return None
                logger.info("Post-quali mode: using actual qualifying results.")
                quali_results = actual_quali[["Driver", "Position"]].rename(
                    columns={"Position": "Quali_Pos"}
                )
                upcoming_df = pd.merge(
                    upcoming_df, quali_results, on="Driver", how="left"
                )
                upcoming_df["Post_Quali"] = 1
                try:
                    self._persist_quali_used_for_race(
                        upcoming_df[["Driver", "Quali_Pos", "Year", "Race_Num"]].copy(),
                        yr,
                        race_name,
                        source="actual",
                    )
                except Exception:
                    pass
            elif mode == "pre_weekend":
                logger.info(
                    "Pre-weekend mode: not incorporating any qualifying (actual or predicted)."
                )
                upcoming_df["Post_Quali"] = 0
            elif mode == "pre_quali":
                logger.info("Pre-quali mode: using predicted qualifying only.")
                quali_predictions = self.predict_qualifying(
                    year, race_name, scenario="pre_quali"
                )
                if quali_predictions is not None:
                    upcoming_df = pd.merge(
                        upcoming_df,
                        quali_predictions[["Driver", "Predicted_Quali_Pos"]],
                        on="Driver",
                        how="left",
                    )
                    upcoming_df.rename(
                        columns={"Predicted_Quali_Pos": "Quali_Pos"}, inplace=True
                    )
                    upcoming_df["Post_Quali"] = 0
                    try:
                        tmp = upcoming_df[
                            ["Driver", "Quali_Pos", "Year", "Race_Num"]
                        ].copy()
                        self._persist_quali_used_for_race(
                            tmp, yr, race_name, source="predicted"
                        )
                    except Exception:
                        pass
            else:        
                if not actual_quali.empty:
                    logger.info(
                        "Auto mode: actual qualifying results found. Incorporating into race prediction."
                    )
                    quali_results = actual_quali[["Driver", "Position"]].rename(
                        columns={"Position": "Quali_Pos"}
                    )
                    upcoming_df = pd.merge(
                        upcoming_df, quali_results, on="Driver", how="left"
                    )
                    upcoming_df["Post_Quali"] = 1
                    try:
                        self._persist_quali_used_for_race(
                            upcoming_df[
                                ["Driver", "Quali_Pos", "Year", "Race_Num"]
                            ].copy(),
                            yr,
                            race_name,
                            source="actual",
                        )
                    except Exception:
                        pass
                else:
                    logger.info(
                        "Auto mode: no actual qualifying results found. Using predicted qualifying."
                    )
                    quali_predictions = self.predict_qualifying(
                        year, race_name, scenario="pre_quali"
                    )
                    if quali_predictions is not None:
                        upcoming_df = pd.merge(
                            upcoming_df,
                            quali_predictions[["Driver", "Predicted_Quali_Pos"]],
                            on="Driver",
                            how="left",
                        )
                        upcoming_df.rename(
                            columns={"Predicted_Quali_Pos": "Quali_Pos"}, inplace=True
                        )
                        try:
                            tmp = upcoming_df[
                                ["Driver", "Quali_Pos", "Year", "Race_Num"]
                            ].copy()
                            self._persist_quali_used_for_race(
                                tmp, yr, race_name, source="predicted"
                            )
                        except Exception:
                            pass
                    upcoming_df["Post_Quali"] = (
                        0
                        if "Post_Quali" not in upcoming_df.columns
                        else upcoming_df["Post_Quali"]
                    )
        if "Post_Quali" not in upcoming_df.columns:
            upcoming_df["Post_Quali"] = 0
        if not hist_races.empty and not upcoming_df.empty:
            yr = int(upcoming_df.iloc[0]["Year"])
            rnd = int(upcoming_df.iloc[0]["Race_Num"])
            grid_df = hist_races[
                (hist_races["Year"] == yr) & (hist_races["Race_Num"] == rnd)
            ][["Driver", "Grid"]]
            if not grid_df.empty:
                upcoming_df = pd.merge(upcoming_df, grid_df, on="Driver", how="left")
        if "Grid" not in upcoming_df.columns and "Quali_Pos" in upcoming_df.columns:
            upcoming_df["Grid"] = upcoming_df["Quali_Pos"]
        upcoming_df["Parc_Ferme"] = upcoming_df["Post_Quali"]

        features_df = self._prepare_features(upcoming_df)

        features_used = self._load_and_enforce_metadata("race", features_df)
        if not features_used:
            logger.error(
                "Model metadata missing or no matching features found for race. Aborting."
            )
            return None
        raw_scores = self.race_model.predict(features_df[features_used])
        if bool(config.get("models.race.use_ranking", False)):
            order = np.argsort(-raw_scores)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(raw_scores) + 1)
            predictions = ranks.astype(float)
        else:
            predictions = raw_scores.astype(float)

                                                                     
        if self.dnf_model is not None and config.get("reliability.enable", True):
            try:
                                                                              
                model_feature_names = getattr(
                    self.dnf_model, "feature_name_", None
                ) or getattr(self.dnf_model, "feature_names_in_", None)
                if model_feature_names is not None and len(model_feature_names) > 0:
                    aligned = {}
                    for name in model_feature_names:
                        if name in features_df.columns:
                            aligned[name] = features_df[name]
                        else:
                            aligned[name] = 0.0
                    X_dnf = pd.DataFrame(aligned, index=features_df.index)
                    X_dnf = X_dnf.select_dtypes(include=np.number)
                else:
                    X_dnf = features_df.select_dtypes(include=np.number)
                p_dnf = self.dnf_model.predict_proba(X_dnf)[:, 1]
                                                                                          
                circuit_map = (
                    config.get("reliability.dnf_position_by_circuit", {}) or {}
                )
                default_dnf_pos = float(
                    config.get(
                        "models.race.default_dnf_position",
                        config.get("reliability.dnf_position", 20),
                    )
                )
                                                   
                if "Circuit_Cluster" in features_df.columns:
                    dnf_pos_series = features_df["Circuit_Cluster"].map(
                        lambda c: circuit_map.get(str(int(c)), default_dnf_pos)
                    )
                else:
                    dnf_pos_series = pd.Series(default_dnf_pos, index=features_df.index)
                dnf_pos = dnf_pos_series.astype(float).values
                expected_pos = (1 - p_dnf) * predictions + p_dnf * dnf_pos
                predictions = expected_pos
            except Exception as e:
                logger.warning(f"Failed to compose DNF risk into race predictions: {e}")

        results = self._format_results(
            upcoming_df,
            predictions,
            "Race_Pos",
            raw_scores=(
                raw_scores
                if bool(config.get("models.race.use_ranking", False))
                else None
            ),
            score_type=(
                "rank_score"
                if bool(config.get("models.race.use_ranking", False))
                else "position_value"
            ),
        )
        if "Predicted_Pos" in results.columns:
            results = results.rename(columns={"Predicted_Pos": "Predicted_Race_Pos"})
                                                                                                    
        try:
            q_lo = getattr(self.race_model, "quantile_lower_model_", None)
            q_hi = getattr(self.race_model, "quantile_upper_model_", None)
            if q_lo is not None and q_hi is not None:
                Xq = features_df[features_used]
                ql = q_lo.predict(Xq)
                qu = q_hi.predict(Xq)
                results["Lower_Pos"] = np.clip(ql, 1, 20)
                results["Upper_Pos"] = np.clip(qu, 1, 20)
            elif hasattr(self.race_model, "calibration_"):
                std = float(self.race_model.calibration_.get("oof_residual_std", 1.5))
                band = float(
                    config.get("prediction.calibration.confidence_band_delta", 1.5)
                )
                results["Lower_Pos"] = np.clip(
                    results["Predicted_Race_Pos"] - band * std, 1, 20
                )
                results["Upper_Pos"] = np.clip(
                    results["Predicted_Race_Pos"] + band * std, 1, 20
                )
                                                                        
                mu = results["Predicted_Race_Pos"].astype(float).values
                sigma = max(std, 1e-6)
                results["Prob_Podium"] = norm.cdf((3.5 - mu) / sigma)
                results["Prob_Top10"] = norm.cdf((10.5 - mu) / sigma)
        except Exception:
            pass
                                  
        try:
            scenario = {
                "pre_weekend": "pre-weekend",
                "pre_quali": "pre-quali",
                "post_quali": "post-quali",
                "auto": "auto",
            }.get(mode, mode)
            self._save_prediction_results(
                "race",
                year,
                race_name,
                results,
                features_used=features_used,
                scenario=scenario,
            )
        except Exception:
            pass
        return results

    def _simulate_future_prediction(
        self, year: int, race_name: str, session: str, lineup_csv: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Simulates predictions for future years using historical data aggregation."""
        hist_races, hist_quali = self.data_loader.load_all_data()
        if hist_races is None or hist_races.empty:
            logger.error("No historical data available for simulation.")
            return None

                                                          
        hist_agg = (
            hist_races.groupby("Driver")
            .agg(
                Avg_Pos=("Position", "mean"),
                Avg_Quali_Pos=("Quali_Pos", "mean"),
                Win_Rate=("Position", lambda x: (x == 1).mean()),
                Podium_Rate=("Position", lambda x: (x <= 3).mean()),
                DNF_Rate=("DNF", "mean"),
            )
            .reset_index()
        )

                                               
        if lineup_csv is not None and os.path.exists(lineup_csv):
            try:
                custom = pd.read_csv(lineup_csv)
                                            
                if "Driver" not in custom.columns or "Team" not in custom.columns:
                    logger.error(
                        "Lineup CSV must contain at least 'Driver' and 'Team' columns."
                    )
                    return None
                custom = custom.copy()
                if "Year" not in custom.columns:
                    custom["Year"] = int(year)
                if "Race_Name" not in custom.columns:
                    custom["Race_Name"] = str(race_name)
                if "Race_Num" not in custom.columns:
                    meta = self.data_loader._get_event_meta(int(year), str(race_name))
                    custom["Race_Num"] = (
                        int(meta.get("round", 0))
                        if meta and meta.get("round") is not None
                        else 0
                    )
                if "Date" not in custom.columns:
                    custom["Date"] = pd.Timestamp(int(year), 1, 1)
                upcoming_df = custom[
                    ["Year", "Race_Num", "Race_Name", "Driver", "Team", "Date"]
                ].copy()
            except Exception as e:
                logger.error(f"Failed to read lineup CSV: {e}")
                return None
        else:
            upcoming_df = self._get_upcoming_race_df(year, race_name)
        if upcoming_df.empty:
            return None

                                                     
        simulated_df = pd.merge(upcoming_df, hist_agg, on="Driver", how="left")

                                                         
        simulated_df.fillna(
            {
                "Avg_Pos": 10.5,
                "Avg_Quali_Pos": 10.5,
                "Win_Rate": 0.0,
                "Podium_Rate": 0.0,
                "DNF_Rate": 0.2,
            },
            inplace=True,
        )

                                                                               
        if session == "qualifying":
            simulated_df = simulated_df.sort_values("Avg_Quali_Pos")
        else:
            simulated_df = simulated_df.sort_values("Avg_Pos")

        simulated_df["Predicted_Pos"] = range(1, len(simulated_df) + 1)
        simulated_df["Prediction_Score"] = (
            simulated_df["Win_Rate"]
            + simulated_df["Podium_Rate"]
            - simulated_df["DNF_Rate"]
        )

                        
        results = simulated_df[["Driver", "Team", "Predicted_Pos", "Prediction_Score"]]
        results = results.sort_values("Predicted_Pos").reset_index(drop=True)
        return results

    def simulate(
        self,
        year: int,
        race_name: str,
        mode: str = "auto",
        n_simulations: int = 2000,
        sc_probability: float = 0.3,
        seed: Optional[int] = None,
        lineup_csv: Optional[str] = None,
    ) -> Optional[SimulationResult]:
        """Run Monte Carlo race simulation using the trained model predictions.

        Parameters
        ----------
        year, race_name : int, str
            Target event.
        mode : str
            Prediction mode passed to predict_race() ('auto', 'post_quali', etc.).
        n_simulations : int
            Number of race simulations to run (default 2 000).
        sc_probability : float
            Probability of a safety car in any given simulation (default 0.3).
        seed : int | None
            RNG seed; falls back to config general.random_seed.
        lineup_csv : str | None
            Optional path to a custom lineup CSV (Driver, Team columns).

        Returns
        -------
        SimulationResult | None
        """
        if seed is None:
            seed = int(config.get("general.random_seed", 42))

        # Get point predictions from the race model
        predictions = self.predict_race(year, race_name, mode=mode)
        if predictions is None or predictions.empty:
            logger.error(f"Cannot simulate: predict_race returned no predictions for {year} {race_name}.")
            return None

        # Rename to standard column expected by RaceSimulator
        if "Predicted_Race_Pos" not in predictions.columns and "Predicted_Pos" in predictions.columns:
            predictions = predictions.rename(columns={"Predicted_Pos": "Predicted_Race_Pos"})

        # Build per-driver DNF probs from model if available
        dnf_probs: Optional[Dict[str, float]] = None
        if self.dnf_model is not None:
            try:
                upcoming_df = self._get_upcoming_race_df(year, race_name)
                if not upcoming_df.empty:
                    features_df = self._prepare_features(upcoming_df)
                    features_used = self._load_and_enforce_metadata("race", features_df)
                    if features_used:
                        X_dnf = features_df[features_used].select_dtypes(include=np.number)
                        p_dnf_arr = self.dnf_model.predict_proba(X_dnf)[:, 1]
                        dnf_probs = dict(zip(upcoming_df["Driver"].values, p_dnf_arr.tolist()))
            except Exception as exc:
                logger.warning(f"Could not compute per-driver DNF probs for simulation: {exc}")

        simulator = RaceSimulator(n_simulations=n_simulations, seed=seed)
        result = simulator.run(predictions, dnf_probs=dnf_probs, sc_probability=sc_probability)

        # Persist summary
        try:
            self._save_prediction_results(
                "simulate_race",
                year,
                race_name,
                result.summary,
                features_used=None,
                scenario=f"simulate_{mode}",
                extra_meta={"n_simulations": n_simulations, "sc_probability": sc_probability},
            )
        except Exception as exc:
            logger.warning(f"Could not save simulation results: {exc}")

        return result

    def _get_upcoming_race_df(self, year: int, race_name: str) -> pd.DataFrame:
        """Gets the DataFrame for the specified upcoming race."""
        try:
                                                 
            if race_name is None or str(race_name).strip().lower() in {
                "upcoming",
                "next",
            }:
                race_df = self.data_loader._get_upcoming_race_info(year)
            else:
                race_df = self.data_loader._get_race_info(year, race_name)
            if race_df.empty:
                logger.error(f"Could not build race info for {year} {race_name}.")
                return pd.DataFrame()
            return race_df
        except Exception as e:
            logger.error(
                f"Failed to get upcoming race dataframe for {year} {race_name}: {e}"
            )
            return pd.DataFrame()

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build inference-time features for upcoming rows only."""
        return self.pred_features.build(df)

    def _format_results(
        self,
        upcoming_df: pd.DataFrame,
        positions: np.ndarray,
        prediction_col_name: str,
        raw_scores: Optional[np.ndarray] = None,
        score_type: str = "position_value",
    ) -> pd.DataFrame:
        """Format results with clear score vs position semantics.

        - positions: final predicted positions (1=best)
        - raw_scores: optional model scores (e.g., ranker scores, higher better)
        - score_type: 'rank_score' when raw_scores are ranking scores; 'position_value' otherwise
        """
        results = upcoming_df[["Driver", "Team"]].copy()
                                                                          
        pos = np.asarray(positions, dtype=float)
        results["Predicted_Pos"] = pos
        if raw_scores is not None:
            results["Model_Score"] = np.asarray(raw_scores, dtype=float)
            results["Prediction_Score"] = results["Model_Score"]
            results["Score_Type"] = score_type
        else:
            results["Prediction_Score"] = results["Predicted_Pos"]
            results["Score_Type"] = score_type
        results = results.sort_values("Predicted_Pos").reset_index(drop=True)
        return results[
            ["Predicted_Pos", "Driver", "Team", "Prediction_Score", "Score_Type"]
            + (["Model_Score"] if "Model_Score" in results.columns else [])
        ]

    def _save_prediction_results(
        self,
        model_type: str,
        year: int,
        race_name: str,
        df: pd.DataFrame,
        *,
        features_used: Optional[List[str]] = None,
        scenario: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        out_dir = config.get("paths.predictions_dir")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname_base = f"{model_type}_predictions_{year}_{race_name}_{ts}"
        csv_path = os.path.join(out_dir, f"{fname_base}.csv")
        meta_path = os.path.join(out_dir, f"{fname_base}.meta.json")
        df.to_csv(csv_path, index=False)
                                          
        try:
            from .config_manager import config_manager

            config_hash = config_manager.get_config_hash()
        except Exception:
            config_hash = ""
        meta: Dict[str, Any] = {
            "model_type": model_type,
            "year": int(year),
            "race_name": str(race_name),
            "generated_at": datetime.now().isoformat(),
            "columns": list(df.columns),
            "model_version": str(config.get("general.model_version", "")),
            "scenario": scenario or "auto",
            "config_hash": config_hash,
        }
        if features_used is not None:
            meta["features_used"] = list(features_used)
        if isinstance(extra_meta, dict) and extra_meta:
            meta.update(extra_meta)
        try:
            import json

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

    def _persist_quali_used_for_race(
        self, df: pd.DataFrame, year: int, race_name: str, source: str
    ) -> None:
        try:
            out_dir = config.get("paths.predictions_dir")
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"quali_used_for_race_{year}_{race_name}_{ts}.csv"
            cols = ["Driver", "Quali_Pos"]
            if "Year" in df.columns:
                cols.append("Year")
            if "Race_Num" in df.columns:
                cols.append("Race_Num")
            out = df.copy()
            out["Quali_Source"] = source
            out[cols + ["Quali_Source"]].to_csv(
                os.path.join(out_dir, fname), index=False
            )
        except Exception:
            pass

    def live_update_race_predictions(
        self,
        year: int,
        race_name: str,
        refresh_seconds: int = 300,
        stop_on_quali: bool = True,
        max_cycles: Optional[int] = None,
    ) -> None:
        """Refresh race predictions periodically; if stop_on_quali is True, re-run once actual quali is available and stop."""
        import time

        cycles = 0
        while True:
            try:
                res = self.predict_race(year, race_name)
                if res is not None:
                    logger.info(
                        f"Race predictions refreshed at {datetime.now().isoformat()}"
                    )
                                                                                  
                hist_races, hist_quali = self.data_loader.load_all_data()
                yr, rn = int(year), None
                if (
                    res is not None
                    and "Year" in res.columns
                    and "Race_Num" in res.columns
                ):
                    yr = int(res.iloc[0].get("Year", year))
                    rn = int(res.iloc[0].get("Race_Num", 0))
                if stop_on_quali and not hist_quali.empty:
                    if rn is None:
                                                               
                        upcoming_df = self._get_upcoming_race_df(year, race_name)
                        if not upcoming_df.empty:
                            rn = int(upcoming_df.iloc[0]["Race_Num"])
                    if rn is not None:
                        actual_quali = hist_quali[
                            (hist_quali["Year"] == yr) & (hist_quali["Race_Num"] == rn)
                        ]
                        if not actual_quali.empty:
                            logger.info(
                                "Actual qualifying detected during live refresh. Finalizing prediction."
                            )
                            break
            except Exception as e:
                logger.warning(f"Live update loop encountered an error: {e}")
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            time.sleep(max(10, int(refresh_seconds)))

    def _load_and_enforce_metadata(self, model_type: str, features_df: pd.DataFrame):
        """Load model metadata if available and enforce feature list/imputation.

        Strict mode: relies on persisted metadata feature list only for stability. Missing
        required features are created using saved imputation values (or 0.0 fallback) and
        columns are ordered exactly as in training. Extra columns are ignored.
        """
        try:
            strict = bool(config.get("general.strict_feature_enforcement", True))
            meta_path = os.path.join(
                self.models_dir, f"{model_type}_model.metadata.json"
            )
            feature_names = None
            imputation_values = None
            if os.path.exists(meta_path):
                import json

                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                feature_names = meta.get("feature_names", None)
                imputation_values = meta.get("imputation_values", None)
                ensemble = meta.get("ensemble", {})
                if ensemble and hasattr(
                    getattr(self, f"{model_type}_model"), "set_weights"
                ):
                    weights = ensemble.get("weights", None)
                    if weights is not None:
                        try:
                            getattr(self, f"{model_type}_model").set_weights(weights)
                        except Exception:
                            pass
                                                                                    
            if not feature_names or not isinstance(feature_names, list):
                if strict:
                    logger.error(
                        f"Strict metadata enforcement failed for {model_type}: missing feature_names in metadata."
                    )
                    return []
                                                                               
                model = getattr(self, f"{model_type}_model")
                feature_names = getattr(model, "feature_names_in_", None)
                if feature_names is None:
                    feature_names = getattr(model, "selected_feature_names_", None)
                if feature_names is None and hasattr(model, "feature_names_"):
                    feature_names = model.feature_names_
                if not feature_names:
                    return []

                                                                               
            critical_cols = set(config.get("general.critical_features", []))
            missing_required = [
                col
                for col in feature_names
                if col not in features_df.columns
                and (not critical_cols or col in critical_cols)
            ]
            if missing_required and bool(
                config.get("general.strict_feature_enforcement", True)
            ):
                logger.error(
                    f"Missing critical features for {model_type}: {missing_required}"
                )
                raise RuntimeError(
                    f"Missing critical features for {model_type}: {missing_required}"
                )
            for col in feature_names:
                if col not in features_df.columns:
                    default_val = None
                    if isinstance(imputation_values, dict):
                        default_val = imputation_values.get(col, None)
                    if default_val is None:
                        default_val = 0.0
                    features_df[col] = default_val

            if isinstance(imputation_values, dict) and not features_df.empty:
                for col, val in imputation_values.items():
                    if col in features_df.columns:
                        try:
                            fill_value = val
                            if fill_value is None:
                                fill_value = 0.0
                            else:
                                if isinstance(fill_value, str):
                                    try:
                                        fill_value = float(fill_value)
                                    except Exception:
                                        fill_value = 0.0
                                if isinstance(fill_value, (int, float)):
                                    if not np.isfinite(float(fill_value)):
                                        fill_value = 0.0
                            features_df[col] = features_df[col].fillna(fill_value)
                        except Exception:
                            features_df[col] = features_df[col].fillna(0.0)

            for col in feature_names:
                if col in features_df.columns:
                    try:
                        features_df[col] = pd.to_numeric(
                            features_df[col], errors="coerce"
                        )
                    except Exception:
                        pass
                    if features_df[col].isna().any():
                        fill_val = 0.0
                        if isinstance(imputation_values, dict):
                            fill_val = imputation_values.get(col, 0.0)
                        features_df[col] = features_df[col].fillna(fill_val)

            try:
                incoming_cols = set(features_df.columns)
                required_cols = set(feature_names)
                extras = sorted(list(incoming_cols - required_cols))
                miss = sorted(list(required_cols - incoming_cols))
                if extras:
                    logger.info(
                        f"Ignoring extra features not in training for {model_type}: {extras[:20]}{'...' if len(extras)>20 else ''}"
                    )
                if miss:
                    logger.info(
                        f"Missing features for {model_type} (imputed or failed): {miss[:20]}{'...' if len(miss)>20 else ''}"
                    )
            except Exception:
                pass
            return list(feature_names)
        except Exception as e:
            logger.warning(f"Failed to enforce metadata for {model_type}: {e}")
            return []
