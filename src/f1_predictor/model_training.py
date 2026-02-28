import pandas as pd
import numpy as np
import logging
import joblib
import os
import json
from typing import Tuple, Dict, Optional, List, Any
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from lightgbm import LGBMRanker, LGBMClassifier
import xgboost as xgb
from xgboost import XGBRanker, XGBClassifier
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from scipy.optimize import linprog

from .config import config
from .utils import set_global_seeds
from .evaluation import F1ModelEvaluator
from .ensemble import EnsembleModel  # extracted to ensemble.py

try:
    import mlflow
except Exception:
    mlflow = None

logger = logging.getLogger(__name__)


class F1ModelTrainer:
    """Trains and saves F1 prediction models."""

    def train_model(
        self, features: pd.DataFrame, target_column_name: str, model_type: str
    ) -> Tuple[Any, List[str], Dict[str, float]]:
        """
        Train an F1 prediction model.
        """
        logger.info(f"Starting {model_type} model training...")

        try:
                                 
            try:
                seed_global = int(config.get("general.random_seed", config.get("general.random_state", 42)))
                set_global_seeds(seed_global)
                logger.info(f"Global random seeds set to {seed_global}")
            except Exception:
                logger.warning("Failed to set global seeds; continuing without explicit determinism.")

            X, y, feature_names, _ = self._prepare_training_data(
                features, target_column_name
            )

            if len(X) == 0:
                logger.error("No training data available.")
                return None, [], {}

                                 
            mlflow_run = None
            if bool(config.get("general.mlflow.enabled", False)) and mlflow is not None:
                try:
                    tracking_uri = config.get("general.mlflow.tracking_uri", None)
                    if tracking_uri:
                        mlflow.set_tracking_uri(tracking_uri)
                    exp_name = config.get("general.mlflow.experiment_name", "f1_prediction")
                    mlflow.set_experiment(exp_name)
                    mlflow_run = mlflow.start_run(run_name=f"train_{model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
                    mlflow.log_params({
                        "model_type": model_type,
                        "cv_strategy": config.get("training.cv_config.cv_strategy"),
                        "cv_folds": int(config.get("training.cv_config.cv_folds", 5)),
                        "use_ranking": bool(config.get(f"models.{model_type}.use_ranking", False)),
                    })
                except Exception:
                    mlflow_run = None
            
                              
            if config.get("training.cv_config.cv_strategy") == "time_series":
                n_splits = config.get("training.cv_config.cv_folds", 5)

                if bool(config.get("training.cv_config.purged", True)):
                    folds = self._purged_group_time_series_splits(
                        features,
                        n_splits=n_splits,
                        purge_events=int(config.get("training.cv_config.purge_events", 1)),
                        embargo_events=int(config.get("training.cv_config.embargo_events", 1)),
                    )
                else:
                    folds = self._grouped_time_series_splits(features, n_splits=n_splits)

                base_model_names: List[str] = config.get(
                    f"models.{model_type}.base_models", None
                ) or config.get(
                    "models.ensemble_config.base_models", ["lightgbm", "xgboost"]
                )
                num_models = len(base_model_names)
                oof_matrix = np.zeros((len(X), num_models))
                per_algo_fold_mae: Dict[str, List[Tuple[Dict[str, Any], float]]] = {
                    name: [] for name in base_model_names
                }
                per_algo_best_iterations: Dict[str, List[int]] = {name: [] for name in base_model_names}
                use_ranking: bool = bool(
                    config.get(f"models.{model_type}.use_ranking", False)
                )

                                                   
                from joblib import Parallel, delayed
                n_jobs_per_fold = int(config.get("training.parallel.n_jobs_per_fold", -1))
                limit_estimator_threads = bool(config.get("training.parallel.limit_estimator_threads", True))
                estimator_n_jobs = int(config.get("training.parallel.estimator_n_jobs", 1))

                def _process_one_fold(args_tuple):
                    fold_id, train_index, val_index = args_tuple
                    logger.info(f"--- Fold {fold_id+1}/{len(folds)} ---")
                                                                                                  
                    train_labels = pd.Index(train_index)
                    val_labels = pd.Index(val_index)
                    train_idx = X.index.intersection(train_labels)
                    val_idx = X.index.intersection(val_labels)

                    X_train, X_val = (
                        X.loc[train_idx].copy(),
                        X.loc[val_idx].copy(),
                    )
                    y_train, y_val = y.loc[train_idx], y.loc[val_idx]

                    X_train, X_val = self._apply_feature_governance(
                        X_train, X_val, model_type
                    )
                    if bool(
                        config.get(
                            "feature_engineering.governance.per_event_zscore.enabled",
                            False,
                        )
                    ):
                        X_train = self._per_event_zscore(X_train, features.loc[train_idx])
                        X_val = self._per_event_zscore(X_val, features.loc[val_idx])

                                                            
                    train_impute = {
                        col: X_train[col].median()
                        for col in X_train.columns
                        if X_train[col].isnull().any()
                    }
                    X_train = X_train.fillna(train_impute).fillna(0)
                    X_val = X_val.fillna(train_impute).fillna(0)

                    trained_fold_models: List[Tuple[str, Any]] = []

                    def fit_one_algorithm(m_idx: int, algo: str):
                        """Train a single base algorithm for this fold and return results."""
                                                          
                        local_use_ranking = use_ranking and ("ranker" in algo)
                        local_params: Dict[str, Any]
                        best_iter: Optional[int] = None
                        if local_use_ranking:
                            local_params = config.get_model_params(algo)
                            local_params = {
                                **local_params,
                                **(config.get(f"models.{model_type}.overrides.{algo}", {}) or {}),
                            }
                            if limit_estimator_threads and "n_jobs" in local_params:
                                local_params["n_jobs"] = estimator_n_jobs
                        else:
                            if config.get("hyperparameter_optimization.enabled", False):
                                try:
                                    import optuna               

                                    sampler = optuna.samplers.TPESampler(
                                        seed=int(config.get("hyperparameter_optimization.seed", 42))
                                    )
                                    storage = config.get("hyperparameter_optimization.storage", None)
                                    study = optuna.create_study(
                                        direction=config.get(
                                            "hyperparameter_optimization.optimization_direction",
                                            "minimize",
                                        ),
                                        sampler=sampler,
                                        storage=storage if storage else None,
                                        study_name=f"{algo}_tuning_{model_type}_{datetime.utcnow().isoformat()}",
                                        load_if_exists=True if storage else False,
                                    )
                                    timeout_seconds = int(float(config.get("hyperparameter_optimization.timeout_hours", 2)) * 3600)
                                    n_trials = int(config.get("hyperparameter_optimization.n_trials", 100))
                                    n_jobs_hpo = int(config.get("hyperparameter_optimization.n_jobs", 1))
                                    study.optimize(
                                        lambda trial: self._objective(
                                            trial,
                                            X_train,
                                            y_train,
                                            X_val,
                                            y_val,
                                            algorithm=algo,
                                        ),
                                        n_trials=n_trials,
                                        timeout=timeout_seconds,
                                        n_jobs=n_jobs_hpo,
                                    )
                                    base_params = config.get_model_params(algo)
                                    tuned = study.best_trial.params
                                    local_params = {**base_params, **tuned}
                                except Exception as e:
                                    logger.warning(
                                        f"Hyperparameter optimization skipped due to: {e}. Falling back to default params for {algo}."
                                    )
                                    local_params = config.get_model_params(algo)
                            else:
                                local_params = config.get_model_params(algo)
                            local_params = {
                                **local_params,
                                **(config.get(f"models.{model_type}.overrides.{algo}", {}) or {}),
                            }
                            if limit_estimator_threads and "n_jobs" in local_params:
                                local_params["n_jobs"] = estimator_n_jobs

                                         
                        if local_use_ranking:
                            group_train = self._build_groups_for_indices(features, train_idx.values)
                            group_val = self._build_groups_for_indices(features, val_idx.values)
                            if algo == "lightgbm_ranker":
                                estimator = LGBMRanker(**local_params)
                                estimator.fit(
                                    X_train,
                                    y_train,
                                    group=group_train,
                                    eval_set=[(X_val, y_val)],
                                    eval_group=[group_val],
                                    callbacks=[
                                        lgb.early_stopping(
                                            int(
                                                config.get(
                                                    "training.training_config.early_stopping_rounds",
                                                    100,
                                                )
                                            ),
                                            verbose=False,
                                        )
                                    ],
                                )
                                try:
                                    best_iter = int(getattr(estimator, "best_iteration_", 0) or 0)
                                except Exception:
                                    best_iter = None
                                preds_scores = estimator.predict(X_val)
                                preds_val_loc = self._scores_to_group_ranks(
                                    preds_scores, features.loc[val_idx]
                                )
                            elif algo == "xgboost_ranker":
                                params_tmp = {**local_params}
                                if params_tmp.get("eval_metric", "ndcg") == "map":
                                    params_tmp["eval_metric"] = "ndcg"
                                estimator = XGBRanker(**params_tmp)
                                max_pos = float(np.max(y_train)) if len(y_train) else 0.0
                                y_train_rel = (max_pos + 1.0) - y_train
                                y_val_rel = (float(np.max(y_val)) + 1.0) - y_val
                                try:
                                    estimator.fit(
                                        X_train,
                                        y_train_rel,
                                        group=group_train,
                                        eval_set=[(X_val, y_val_rel)],
                                        eval_group=[group_val],
                                        verbose=False,
                                    )
                                except TypeError:
                                    estimator.fit(
                                        X_train,
                                        y_train_rel,
                                        group=group_train,
                                    )
                                try:
                                    best_iter = int(getattr(estimator, "best_iteration", 0) or 0)
                                except Exception:
                                    best_iter = None
                                preds_scores = estimator.predict(X_val)
                                preds_val_loc = self._scores_to_group_ranks(
                                    preds_scores, features.loc[val_idx]
                                )
                            else:
                                raise ValueError(f"Unsupported base model: {algo}")
                        else:
                            if algo == "lightgbm":
                                estimator = lgb.LGBMRegressor(**local_params)
                                estimator.fit(
                                    X_train,
                                    y_train,
                                    eval_set=[(X_val, y_val)],
                                    callbacks=[
                                        lgb.early_stopping(
                                            int(
                                                config.get(
                                                    "training.training_config.early_stopping_rounds",
                                                    100,
                                                )
                                            ),
                                            verbose=False,
                                        )
                                    ],
                                )
                                try:
                                    best_iter = int(getattr(estimator, "best_iteration_", 0) or 0)
                                except Exception:
                                    best_iter = None
                                preds_val_loc = estimator.predict(X_val)
                            elif algo == "xgboost":
                                estimator = xgb.XGBRegressor(**local_params)
                                estimator.fit(
                                    X_train,
                                    y_train,
                                    eval_set=[(X_val, y_val)],
                                    early_stopping_rounds=int(
                                        config.get(
                                            "training.training_config.early_stopping_rounds",
                                            100,
                                        )
                                    ),
                                    verbose=False,
                                )
                                try:
                                    best_iter = int(getattr(estimator, "best_iteration", 0) or 0)
                                except Exception:
                                    best_iter = None
                                preds_val_loc = estimator.predict(X_val)
                            elif algo == "random_forest":
                                estimator = RandomForestRegressor(**local_params)
                                estimator.fit(X_train, y_train)
                                preds_val_loc = estimator.predict(X_val)
                            else:
                                raise ValueError(f"Unsupported base model: {algo}")

                                                                                                    
                        mae_val_loc = float(
                            np.mean(
                                np.abs(
                                    np.asarray(y_val, dtype=float) - np.asarray(preds_val_loc, dtype=float)
                                )
                            )
                        )
                        return (m_idx, algo, preds_val_loc, float(mae_val_loc), estimator, local_params, best_iter)

                                                                    
                                                                           
                    n_jobs_folds = int(config.get("training.parallel.n_jobs_folds", 1))
                    effective_n_jobs_per_fold = n_jobs_per_fold if n_jobs_folds == 1 else 1

                    results = Parallel(n_jobs=effective_n_jobs_per_fold)(
                        delayed(fit_one_algorithm)(m_idx, algo)
                        for m_idx, algo in enumerate(base_model_names)
                    )

                                        
                    val_pos = X.index.get_indexer(val_idx)
                    val_pos = val_pos[val_pos >= 0]
                    fold_oof = np.zeros((len(val_pos), num_models))
                    fold_mae_entries: List[Tuple[str, Dict[str, Any], float]] = []
                    fold_best_iters: Dict[str, List[int]] = {name: [] for name in base_model_names}

                    for m_idx, algo, preds_val_loc, mae_val_loc, estimator_loc, params_loc, best_iter_loc in results:
                                                              
                        fold_oof[:, m_idx] = np.asarray(preds_val_loc, dtype=float)
                        fold_mae_entries.append((algo, params_loc, mae_val_loc))
                        trained_fold_models.append((algo, estimator_loc))
                        if best_iter_loc and best_iter_loc > 0:
                            fold_best_iters[algo].append(int(best_iter_loc))

                    return (val_pos, fold_oof, fold_mae_entries, fold_best_iters)

                                                            
                n_jobs_folds = int(config.get("training.parallel.n_jobs_folds", 1))
                fold_args = [(i, tr, va) for i, (tr, va) in enumerate(folds)]
                if n_jobs_folds == 1:
                    fold_results = [_process_one_fold(args_t) for args_t in fold_args]
                else:
                    fold_results = Parallel(n_jobs=n_jobs_folds)(
                        delayed(_process_one_fold)(args_t) for args_t in fold_args
                    )

                                        
                for val_pos, fold_oof, fold_mae_entries, fold_best_iters in fold_results:
                    oof_matrix[val_pos, :] = fold_oof
                    for algo, params_loc, mae_val_loc in fold_mae_entries:
                        per_algo_fold_mae[algo].append((params_loc, float(mae_val_loc)))
                    for algo, iters_list in fold_best_iters.items():
                        if iters_list:
                            per_algo_best_iterations[algo].extend([int(v) for v in iters_list])

                weights = self._optimize_ensemble_weights_metric(
                    y_true=y.values if hasattr(y, "values") else y,
                    oof_preds=oof_matrix,
                    features=features,
                    model_type=model_type,
                )
                logger.info(f"Optimized ensemble weights: {weights}")

                                                                  
                best_params_per_algo: Dict[str, Dict[str, Any]] = {}
                for algo, entries in per_algo_fold_mae.items():
                    if not entries:
                        best_params_per_algo[algo] = config.get_model_params(algo)
                        continue
                                                                   
                    agg: Dict[str, Tuple[Dict[str, Any], List[float]]] = {}
                    for params, mae in entries:
                        key = str(sorted(params.items()))
                        if key not in agg:
                            agg[key] = (params, [mae])
                        else:
                            agg[key][1].append(mae)
                                           
                    best = min(agg.values(), key=lambda pr: float(np.mean(pr[1])))
                    best_params_per_algo[algo] = best[0]

                                                                                 
                final_models: Dict[str, Any] = {}
                for algo in base_model_names:
                    params = {
                        **best_params_per_algo.get(algo, config.get_model_params(algo)),
                        **(
                            config.get(f"models.{model_type}.overrides.{algo}", {})
                            or {}
                        ),
                    }
                    if per_algo_best_iterations.get(algo):
                        try:
                            avg_best_iter = int(np.median(np.asarray(per_algo_best_iterations[algo], dtype=float)))
                            if avg_best_iter > 0:
                                params = {**params, "n_estimators": avg_best_iter}
                        except Exception:
                            pass
                    if use_ranking and "ranker" in algo:
                                                                         
                        holdout_years = set(config.get("evaluation.holdout_years", []))
                        if "Year" in features.columns and holdout_years:
                            non_holdout_idx = features.index[
                                ~features["Year"].isin(holdout_years)
                            ].values
                        else:
                            non_holdout_idx = features.index.values
                                                                                                            
                        nh_idx = X.index.intersection(pd.Index(non_holdout_idx))
                        group_full = self._build_groups_for_indices(
                            features, nh_idx.values
                        )
                                                                     
                        X_full = X.loc[nh_idx].copy()
                        y_full = y.loc[nh_idx]
                        final_imputation_values = {
                            col: X_full[col].median()
                            for col in X_full.columns
                            if X_full[col].isnull().any()
                        }
                        X_full = X_full.fillna(final_imputation_values).fillna(0)
                        if algo == "lightgbm_ranker":
                            model_final = LGBMRanker(**params)
                            model_final.fit(
                                X_full,
                                y_full,
                                group=group_full,
                            )
                            final_models[algo] = model_final
                        elif algo == "xgboost_ranker":
                            params = {**params}
                            if params.get("eval_metric", "ndcg") == "map":
                                params["eval_metric"] = "ndcg"
                            model_final = XGBRanker(**params)
                            max_pos = float(np.max(y_full)) if len(y_full) else 0.0
                            y_full_rel = (max_pos + 1.0) - y_full
                            try:
                                model_final.fit(
                                    X_full,
                                    y_full_rel,
                                    group=group_full,
                                    verbose=False,
                                )
                            except TypeError:
                                model_final.fit(
                                    X_full,
                                    y_full_rel,
                                    group=group_full,
                                )
                            final_models[algo] = model_final
                    else:
                                                                
                        if "Year" in features.columns and set(
                            config.get("evaluation.holdout_years", [])
                        ):
                            mask_non_holdout = ~features["Year"].isin(
                                set(config.get("evaluation.holdout_years", []))
                            )
                        else:
                            mask_non_holdout = np.ones(len(X), dtype=bool)
                        X_full = X.loc[mask_non_holdout].copy()
                        y_full = y.loc[mask_non_holdout]
                        final_imputation_values = {
                            col: X_full[col].median()
                            for col in X_full.columns
                            if X_full[col].isnull().any()
                        }
                        X_full = X_full.fillna(final_imputation_values).fillna(0)
                        if algo == "lightgbm":
                            est = lgb.LGBMRegressor(**params)
                            try:
                                split_point = int(len(X_full) * 0.9)
                                X_f_tr, X_f_va = X_full.iloc[:split_point], X_full.iloc[split_point:]
                                y_f_tr, y_f_va = y_full.iloc[:split_point], y_full.iloc[split_point:]
                                est.fit(
                                    X_f_tr,
                                    y_f_tr,
                                    eval_set=[(X_f_va, y_f_va)],
                                    callbacks=[
                                        lgb.early_stopping(
                                            int(config.get("training.training_config.early_stopping_rounds", 100)),
                                            verbose=False,
                                        )
                                    ],
                                )
                            except Exception:
                                est.fit(X_full, y_full)
                            final_models[algo] = est
                        elif algo == "xgboost":
                            if limit_estimator_threads and "n_jobs" in params:
                                params = {**params, "n_jobs": estimator_n_jobs}
                                                                                   
                            if limit_estimator_threads and "nthread" in params:
                                params = {**params, "nthread": estimator_n_jobs}
                            est = xgb.XGBRegressor(**params)
                            try:
                                split_point = int(len(X_full) * 0.9)
                                X_f_tr, X_f_va = X_full.iloc[:split_point], X_full.iloc[split_point:]
                                y_f_tr, y_f_va = y_full.iloc[:split_point], y_full.iloc[split_point:]
                                est.fit(
                                    X_f_tr,
                                    y_f_tr,
                                    eval_set=[(X_f_va, y_f_va)],
                                    early_stopping_rounds=int(config.get("training.training_config.early_stopping_rounds", 100)),
                                    verbose=False,
                                )
                            except Exception:
                                est.fit(X_full, y_full)
                            final_models[algo] = est
                        elif algo == "random_forest":
                                                                       
                            est = RandomForestRegressor(**params)
                            est.fit(X_full, y_full)
                            final_models[algo] = est
                model = EnsembleModel(final_models)
                                                                                
                if weights is None or (
                    hasattr(weights, "size") and int(weights.size) == 0
                ):
                    weights = np.ones(len(final_models)) / max(1, len(final_models))
                model.set_weights(list(weights))
                                              
                if "Year" in features.columns and set(
                    config.get("evaluation.holdout_years", [])
                ):
                    mask_non_holdout = ~features["Year"].isin(
                        set(config.get("evaluation.holdout_years", []))
                    )
                                                                                          
                    nh_idx = X.index.intersection(features.index[mask_non_holdout])
                    model.fit(
                        X.loc[nh_idx].fillna(final_imputation_values).fillna(0),
                        y.loc[nh_idx],
                    )
                else:
                    model.fit(X.fillna(final_imputation_values).fillna(0), y)
                                                      
                ensemble_oof = oof_matrix.dot(weights)
                residuals = (y.values if hasattr(y, "values") else y) - ensemble_oof
                model.calibration_ = {
                    "oof_residual_std": float(np.std(residuals)),
                    "base_model_names": base_model_names,
                }
                try:
                    if bool(config.get("prediction.calibration.use_quantile_models", True)):
                        lower_q = float(config.get("prediction.calibration.lower_quantile", 0.1))
                        upper_q = float(config.get("prediction.calibration.upper_quantile", 0.9))
                        q_params = {
                            k: v
                            for k, v in config.get_model_params("lightgbm").items()
                            if k not in {"objective", "alpha"}
                        }
                        q_lo = lgb.LGBMRegressor(objective="quantile", alpha=lower_q, **q_params)
                        q_hi = lgb.LGBMRegressor(objective="quantile", alpha=upper_q, **q_params)
                                                                                        
                        if "Year" in features.columns and set(config.get("evaluation.holdout_years", [])):
                            mask_non_holdout = ~features["Year"].isin(set(config.get("evaluation.holdout_years", [])))
                            q_X = X.loc[mask_non_holdout].fillna(final_imputation_values).fillna(0)
                            q_y = y.loc[mask_non_holdout]
                        else:
                            q_X = X.fillna(final_imputation_values).fillna(0)
                            q_y = y
                        q_lo.fit(q_X, q_y)
                        q_hi.fit(q_X, q_y)
                        setattr(model, "quantile_lower_model_", q_lo)
                        setattr(model, "quantile_upper_model_", q_hi)
                        model.calibration_["quantiles"] = {"lower": lower_q, "upper": upper_q}
                except Exception as e:
                    logger.warning(f"Quantile interval training failed: {e}")

                                                                                  
                try:
                    if bool(config.get("training.ensemble.dynamic_weights", True)):
                        seg_cols = config.get("training.ensemble.segment_keys", ["Circuit_Cluster", "Is_Sprint"])
                        if all(c in features.columns for c in seg_cols):
                            weights_by_segment: Dict[str, List[float]] = {}
                            seg_keys = features[seg_cols].astype(float).fillna(-1).astype(int)
                            seg_tuple = list(map(tuple, seg_keys.values.tolist()))
                            for seg in sorted(set(seg_tuple)):
                                seg_mask = [t == seg for t in seg_tuple]
                                if int(np.sum(seg_mask)) < 10:
                                    continue
                                seg_weights = self._optimize_ensemble_weights_metric(
                                    y_true=(y.values if hasattr(y, "values") else y)[seg_mask],
                                    oof_preds=oof_matrix[seg_mask, :],
                                    features=features.iloc[np.where(seg_mask)[0]],
                                    model_type=model_type,
                                )
                                weights_by_segment[str(seg)] = list(map(float, seg_weights))
                            if weights_by_segment:
                                setattr(model, "weights_by_segment_", weights_by_segment)
                                setattr(model, "segment_keys_", seg_cols)
                except Exception as e:
                    logger.warning(f"Dynamic weight segmentation failed: {e}")
                                                               
                imputation_values = final_imputation_values
            else:
                                                 
                X_train, X_test, y_train, y_test = self._time_aware_split(
                    X, y, features
                )

                base_model_names: List[str] = config.get(
                    f"models.{model_type}.base_models", None
                ) or config.get(
                    "models.ensemble_config.base_models", ["lightgbm", "xgboost"]
                )
                                                                             
                train_impute = {
                    col: X_train[col].median()
                    for col in X_train.columns
                    if X_train[col].isnull().any()
                }
                X_train_f = X_train.fillna(train_impute).fillna(0)
                X_test_f = X_test.fillna(train_impute).fillna(0)

                models_dict: Dict[str, Any] = {}
                use_ranking: bool = bool(
                    config.get(f"models.{model_type}.use_ranking", False)
                )
                for algo in base_model_names:
                    if use_ranking and "ranker" in algo:
                        params = {
                            **config.get_model_params(algo),
                            **(
                                config.get(f"models.{model_type}.overrides.{algo}", {})
                                or {}
                            ),
                        }
                        group_train = self._build_groups_for_indices(
                            features, X_train.index.values
                        )
                        if algo == "lightgbm_ranker":
                            ranker = LGBMRanker(**params)
                            ranker.fit(
                                X_train_f,
                                y_train,
                                group=group_train,
                                eval_set=[(X_test_f, y_test)],
                                eval_group=[
                                    self._build_groups_for_indices(
                                        features, X_test.index.values
                                    )
                                ],
                                callbacks=[
                                    lgb.early_stopping(
                                        int(
                                            config.get(
                                                "training.training_config.early_stopping_rounds",
                                                100,
                                            )
                                        ),
                                        verbose=False,
                                    )
                                ],
                            )
                            models_dict[algo] = ranker
                        elif algo == "xgboost_ranker":
                            params = {**params}
                            if params.get("eval_metric", "ndcg") == "map":
                                params["eval_metric"] = "ndcg"
                            ranker = XGBRanker(**params)
                            max_pos = float(np.max(y_train)) if len(y_train) else 0.0
                            y_train_rel = (max_pos + 1.0) - y_train
                            try:
                                ranker.fit(
                                    X_train_f,
                                    y_train_rel,
                                    group=group_train,
                                    eval_set=[
                                        (X_test_f, (float(np.max(y_test)) + 1.0) - y_test)
                                    ],
                                    eval_group=[
                                        self._build_groups_for_indices(
                                            features, X_test.index.values
                                        )
                                    ],
                                    verbose=False,
                                )
                            except TypeError:
                                ranker.fit(
                                    X_train_f,
                                    y_train_rel,
                                    group=group_train,
                                )
                            models_dict[algo] = ranker
                    else:
                        if config.get("hyperparameter_optimization.enabled", False):
                            try:
                                import optuna               

                                sampler = optuna.samplers.TPESampler(
                                    seed=int(config.get("hyperparameter_optimization.seed", 42))
                                )
                                storage = config.get("hyperparameter_optimization.storage", None)
                                study = optuna.create_study(
                                    direction=config.get(
                                        "hyperparameter_optimization.optimization_direction",
                                        "minimize",
                                    ),
                                    sampler=sampler,
                                    storage=storage if storage else None,
                                    study_name=f"{algo}_tuning_{model_type}_{datetime.utcnow().isoformat()}",
                                    load_if_exists=True if storage else False,
                                )
                                timeout_seconds = int(float(config.get("hyperparameter_optimization.timeout_hours", 2)) * 3600)
                                study.optimize(
                                    lambda trial: self._objective(
                                        trial,
                                        X_train,
                                        y_train,
                                        X_test,
                                        y_test,
                                        algorithm=algo,
                                    ),
                                    n_trials=config.get(
                                        "hyperparameter_optimization.n_trials", 100
                                    ),
                                    timeout=timeout_seconds,
                                )
                                                                            
                                base_params = config.get_model_params(algo)
                                tuned = study.best_trial.params
                                params = {**base_params, **tuned}
                            except Exception as e:
                                logger.warning(
                                    f"Hyperparameter optimization skipped due to: {e}. Falling back to default params for {algo}."
                                )
                                params = config.get_model_params(algo)
                        else:
                            params = config.get_model_params(algo)
                        params = {
                            **params,
                            **(
                                config.get(f"models.{model_type}.overrides.{algo}", {})
                                or {}
                            ),
                        }
                        if algo == "lightgbm":
                            if limit_estimator_threads and "n_jobs" in params:
                                params = {**params, "n_jobs": estimator_n_jobs}
                            est = lgb.LGBMRegressor(**params)
                            est.fit(
                                X_train_f,
                                y_train,
                                eval_set=[(X_test_f, y_test)],
                                callbacks=[
                                    lgb.early_stopping(
                                        int(
                                            config.get(
                                                "training.training_config.early_stopping_rounds",
                                                100,
                                            )
                                        ),
                                        verbose=False,
                                    )
                                ],
                            )
                            models_dict[algo] = est
                        elif algo == "xgboost":
                            est = xgb.XGBRegressor(**params)
                            est.fit(
                                X_train_f,
                                y_train,
                                eval_set=[(X_test_f, y_test)],
                                early_stopping_rounds=int(
                                    config.get(
                                        "training.training_config.early_stopping_rounds",
                                        100,
                                    )
                                ),
                                verbose=False,
                            )
                            models_dict[algo] = est
                        elif algo == "random_forest":
                                                                              
                            est = RandomForestRegressor(**params)
                            est.fit(X_train_f, y_train)
                            models_dict[algo] = est

                X_train_f, X_test_f = self._apply_feature_governance(
                    X_train_f, X_test_f, model_type
                )
                if bool(
                    config.get(
                        "feature_engineering.governance.per_event_zscore.enabled",
                        False,
                    )
                ):
                    X_train_f = self._per_event_zscore(X_train_f, features.loc[X_train.index])
                    X_test_f = self._per_event_zscore(X_test_f, features.loc[X_test.index])

                model = EnsembleModel(models_dict)
                                                                                   
                model.fit(X_train_f, y_train)
                                                      
                                                            
                if use_ranking:
                    val_components = []
                    for est in model.models.values():
                        scores = est.predict(X_test)
                        ranks = self._scores_to_group_ranks(
                            scores, features.loc[X_test.index]
                        )
                        val_components.append(ranks)
                    val_matrix = np.column_stack(val_components)
                else:
                    val_matrix = np.column_stack(
                        [est.predict(X_test_f) for est in model.models.values()]
                    )
                weights = self._optimize_ensemble_weights(
                    y_test.values if hasattr(y_test, "values") else y_test, val_matrix
                )
                model.set_weights(list(weights))
                self._evaluate_model(model, X_test_f, y_test, model_type)
                                                         
                if mlflow_run is not None:
                    try:
                        from sklearn.metrics import mean_absolute_error, r2_score
                        preds_tmp = model.predict(X_test_f)
                        mlflow.log_metrics({
                            "mae_val": float(mean_absolute_error(y_test, preds_tmp)),
                            "r2_val": float(r2_score(y_test, preds_tmp)),
                        })
                    except Exception:
                        pass
                                                            
                imputation_values = train_impute

            self._save_model(model, model_type)
                                                           
            try:
                self._save_model_metadata(
                    model=model,
                    model_type=model_type,
                    feature_names=feature_names,
                    imputation_values=imputation_values,
                )
            except Exception as e:
                logger.warning(f"Failed to save model metadata for {model_type}: {e}")

                                                                  
            try:
                summary_path: Optional[str] = None
                summary = {
                    "model_type": model_type,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "base_models": list(getattr(model, "models", {}).keys()),
                    "weights": list(map(float, getattr(model, "weights_", []))) if hasattr(model, "weights_") else [],
                    "dynamic_weights_segments": list(getattr(model, "weights_by_segment_", {}).keys()) if hasattr(model, "weights_by_segment_") else [],
                    "calibration": getattr(model, "calibration_", {}),
                    "cv": {
                        "strategy": config.get("training.cv_config.cv_strategy"),
                        "purged": bool(config.get("training.cv_config.purged", True)),
                        "folds": int(config.get("training.cv_config.cv_folds", 5)),
                    },
                }
                out_dir = config.get("paths.evaluation_dir")
                os.makedirs(out_dir, exist_ok=True)
                summary_path = os.path.join(out_dir, f"training_summary_{model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"Training summary saved: {summary_path}")
                                                   
                if mlflow_run is not None and summary_path and os.path.exists(summary_path):
                    try:
                        mlflow.log_artifact(summary_path, artifact_path="reports")
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Failed to save training summary: {e}")

                                                 
            if model_type == "race" and bool(
                config.get("models.race.enable_dnf_model", True)
            ):
                try:
                    self._train_and_save_dnf_classifier(features)
                except Exception as e:
                    logger.warning(f"DNF classifier training failed: {e}")

                                                                                                            
            try:
                holdout_years = set(config.get("evaluation.holdout_years", []))
                if "Year" in features.columns and holdout_years:
                    test_mask = features["Year"].isin(holdout_years)
                    if test_mask.any():
                                                                                           
                        X_holdout = X.loc[test_mask]
                        impute_vals = (
                            imputation_values
                            if isinstance(imputation_values, dict)
                            else {}
                        )
                        X_holdout = X_holdout.fillna(impute_vals).fillna(0)
                        self._evaluate_model(
                            model, X_holdout, y.loc[test_mask], model_type
                        )
            except Exception as e:
                logger.warning(f"Holdout evaluation failed: {e}")

                                                     
            if mlflow_run is not None:
                try:
                    models_dir = config.get("paths.models_dir")
                    model_path = os.path.join(models_dir, f"{model_type}_model.pkl")
                    if os.path.exists(model_path):
                        mlflow.log_artifact(model_path, artifact_path="models")
                    mlflow.end_run()
                except Exception:
                    pass
            logger.info(f"{model_type} model training completed successfully.")
            return (
                model,
                feature_names,
                imputation_values,
            )

        except Exception as e:
            logger.error(f"Model training failed: {e}", exc_info=True)
            return None, [], {}

    def _time_aware_split(
        self, X: pd.DataFrame, y: pd.Series, features: pd.DataFrame
    ) -> Tuple:
        """Hold out configured years; otherwise split by chronological order."""
        logger.info("Performing time-aware split...")
        holdout_years = set(config.get("evaluation.holdout_years", []))
        if "Year" in features.columns and holdout_years:
            test_mask = features["Year"].isin(holdout_years)
            if test_mask.any() and (~test_mask).any():
                X_train, X_test = X.loc[~test_mask], X.loc[test_mask]
                y_train, y_test = y.loc[~test_mask], y.loc[test_mask]
                logger.info(
                    f"Train size: {len(X_train)}, Test size (holdout years): {len(X_test)}"
                )
                return X_train, X_test, y_train, y_test

        split_config = config.get("training.split_config", {})
        test_size = split_config.get("test_size", 0.2)
        if "Date" not in features.columns:
            logger.warning("Date column missing; falling back to random split.")
            return train_test_split(X, y, test_size=test_size, random_state=42)
        sorted_indices = features["Date"].sort_values().index
        split_point = int(len(sorted_indices) * (1 - test_size))
        train_indices = sorted_indices[:split_point]
        test_indices = sorted_indices[split_point:]
        X_train, X_test = X.loc[train_indices], X.loc[test_indices]
        y_train, y_test = y.loc[train_indices], y.loc[test_indices]
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def _objective(
        self, trial, X_train, y_train, X_val, y_val, algorithm: str
    ) -> float:
        """Objective function for Optuna per-algorithm search spaces."""
        param_search_spaces = config.get(
            f"hyperparameter_optimization.param_search_spaces.{algorithm}", {}
        )
        params = {}
        if "n_estimators" in param_search_spaces:
            params["n_estimators"] = trial.suggest_int(
                "n_estimators", *param_search_spaces.get("n_estimators", [1000, 5000])
            )
        if "learning_rate" in param_search_spaces:
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", *param_search_spaces.get("learning_rate", [0.01, 0.05])
            )
        if "max_depth" in param_search_spaces:
            params["max_depth"] = trial.suggest_int(
                "max_depth", *param_search_spaces.get("max_depth", [7, 15])
            )
                                                              
        train_impute = {
            col: X_train[col].median()
            for col in X_train.columns
            if X_train[col].isnull().any()
        }
        X_tr = X_train.fillna(train_impute).fillna(0)
        X_va = X_val.fillna(train_impute).fillna(0)

        if algorithm == "lightgbm":
            model = lgb.LGBMRegressor(
                **{**config.get_model_params("lightgbm"), **params}
            )
            model.fit(
                X_tr,
                y_train,
                eval_set=[(X_va, y_val)],
                callbacks=[
                    lgb.early_stopping(
                        int(
                            config.get(
                                "training.training_config.early_stopping_rounds", 100
                            )
                        ),
                        verbose=False,
                    )
                ],
            )
        elif algorithm == "xgboost":
            model = xgb.XGBRegressor(**{**config.get_model_params("xgboost"), **params})
            model.fit(
                X_tr,
                y_train,
                eval_set=[(X_va, y_val)],
                early_stopping_rounds=int(
                    config.get("training.training_config.early_stopping_rounds", 100)
                ),
                verbose=False,
            )
        elif algorithm == "random_forest":
            model = RandomForestRegressor(
                **{**config.get_model_params("random_forest"), **params}
            )
            model.fit(X_tr, y_train)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        preds = model.predict(X_va)
        objective_metric = str(
            config.get("hyperparameter_optimization.objective_metric", "mae")
        ).lower()
        if objective_metric == "spearman":
            try:
                from scipy.stats import spearmanr
                corr = float(spearmanr(y_val, preds)[0])
                return 1.0 - (0.5 * (corr + 1.0))
            except Exception:
                return mean_absolute_error(y_val, preds)
        return mean_absolute_error(y_val, preds)

    def _evaluate_model_oof(self, y_true, oof_preds, model_type):
        """Evaluates the model using out-of-fold predictions."""
        logger.info(f"Evaluating {model_type} model with OOF predictions...")
        mae = mean_absolute_error(y_true, oof_preds)
        r2 = r2_score(y_true, oof_preds)

        logger.info(f"OOF Evaluation metrics for {model_type}:")
        logger.info(f"  Mean Absolute Error (MAE): {mae:.4f}")
        logger.info(f"  R^2 Score: {r2:.4f}")

    def _evaluate_model(self, model, X_test, y_test, model_type):
        """Evaluate with F1-specific metrics and persist reports to evaluation/."""
        try:
            logger.info(f"Evaluating {model_type} model on test/holdout set...")
            evaluator = F1ModelEvaluator()
            _ = evaluator.evaluate_model_performance(
                model, X_test, y_test, model_name=model_type
            )
                                                
            evaluator.generate_evaluation_report(save_plots=True)
            evaluator.save_evaluation_data()
        except Exception as e:
            logger.warning(f"Evaluation/report generation failed for {model_type}: {e}")

    def _train_and_save_dnf_classifier(self, features: pd.DataFrame) -> None:
        """Train and persist a calibrated DNF probability classifier for race reliability modeling."""
        try:
            if "DNF" not in features.columns:
                logger.warning("DNF column missing; skipping DNF classifier training.")
                return
            df = features.dropna(subset=["DNF"]).copy()
            if df.empty:
                logger.warning("No rows available for DNF classifier training.")
                return
            y = df["DNF"].astype(int)
            X = df.drop(
                columns=["DNF", "Position", "Quali_Pos"], errors="ignore"
            ).select_dtypes(include=np.number)
            if X.empty:
                logger.warning("No numeric features for DNF classifier; skipping.")
                return
            X_train, X_test, y_train, y_test = self._time_aware_split(X, y, df)
                                                             
            try:
                clf_base = LGBMClassifier(**config.get_model_params("lightgbm_classifier"))
            except Exception:
                clf_base = XGBClassifier(**config.get_model_params("xgboost_classifier"))
            clf_base.fit(X_train, y_train)
                                                                    
            method = str(config.get("reliability.calibration.method", "isotonic")).lower()
            if method not in {"isotonic", "sigmoid"}:
                method = "sigmoid"
            calibrator = CalibratedClassifierCV(base_estimator=clf_base, method=method, cv="prefit")
            calibrator.fit(X_test, y_test)
            models_dir = config.get("paths.models_dir")
            os.makedirs(models_dir, exist_ok=True)
            joblib.dump(calibrator, os.path.join(models_dir, "race_dnf_model.pkl"))
            logger.info("Calibrated DNF classifier trained and saved: race_dnf_model.pkl")
        except Exception as e:
            logger.warning(f"Failed DNF classifier training: {e}")

    def _prepare_training_data(
        self, features: pd.DataFrame, target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict]:
        """Prepare data for training without global imputation to avoid leakage.

        Returns numeric-only feature matrix X with NaNs preserved; per-split imputation is
        performed later in the training routine. Also returns an empty imputation mapping
        placeholder to be filled with train-only statistics and persisted in metadata.
        """
        logger.info("Preparing training data...")

        data_clean = features.dropna(subset=[target_column]).copy()

        y = data_clean[target_column]
        drop_cols: List[str] = [target_column]
        if target_column == "Quali_Pos":
            drop_cols.append("Position")
        if target_column == "Position":
            pass
        X = data_clean.drop(columns=drop_cols, errors="ignore")

                                      
        X = X.select_dtypes(include=np.number)
        feature_names = list(X.columns)

                                                                  
        imputation_values: Dict[str, float] = {}

        logger.info(
            f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features"
        )
        return X, y, feature_names, imputation_values

    def _save_model(self, model: Any, model_type: str):
        """Save trained model."""
        try:
            models_dir = config.get("paths.models_dir")
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f"{model_type}_model.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Model saved to: {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def _save_model_metadata(
        self,
        model: Any,
        model_type: str,
        feature_names: List[str],
        imputation_values: Dict[str, float],
    ) -> None:
        """Persist model metadata alongside the model artifact for reproducible inference."""
        models_dir = config.get("paths.models_dir")
        os.makedirs(models_dir, exist_ok=True)
        metadata_path = os.path.join(models_dir, f"{model_type}_model.metadata.json")

        ensemble_info: Dict[str, Any] = {}
        try:
            weights = getattr(model, "weights_", None)
            base_models = (
                list(getattr(model, "models", {}).keys())
                if hasattr(model, "models")
                else None
            )
            if weights is not None:
                ensemble_info["weights"] = list(map(float, weights))
            if base_models is not None:
                ensemble_info["base_models"] = base_models
        except Exception:
            pass

                                   
        git_commit: Optional[str] = None
        try:
            import subprocess

            git_commit = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
        except Exception:
            git_commit = None

                             
        try:
            from importlib.metadata import version as _pkg_version
        except Exception:                    
            _pkg_version = lambda _: ""

        try:
            from . import __version__ as package_version                
        except Exception:
            package_version = ""

        config_hash_val = getattr(_config_obj, "get_config_hash", lambda: None)()
        model_version_val = str(config.get("general.model_version", ""))

        metadata: Dict[str, Any] = {
            "model_type": model_type,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "model_version": model_version_val,
            "config_hash": config_hash_val,
            "git_commit": git_commit,
            "feature_names": feature_names,
            "imputation_values": {
                k: (None if pd.isna(v) else float(v))
                for k, v in (imputation_values or {}).items()
            },
            "ensemble": ensemble_info,
            "calibration": getattr(model, "calibration_", {}),
            "holdout_years": config.get("evaluation.holdout_years", []),
            "config": {
                "base_models": config.get(f"models.{model_type}.base_models", None)
                or config.get("models.ensemble_config.base_models", []),
                "use_ranking": bool(
                    config.get(f"models.{model_type}.use_ranking", False)
                ),
                "config_hash": config_hash_val,
                "model_version": model_version_val,
            },
            "git": {"commit": git_commit},
            "dependencies": {
                "lightgbm": getattr(lgb, "__version__", ""),
                "xgboost": getattr(xgb, "__version__", ""),
                "scikit_learn": __import__("sklearn").__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
            "package": {"version": package_version},
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved model metadata to: {metadata_path}")

    def _grouped_time_series_splits(
        self, features: pd.DataFrame, n_splits: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create time-ordered splits grouped by (Year, Race_Num), excluding holdout years."""
        if "Year" not in features.columns or "Race_Num" not in features.columns:
            logger.warning(
                "Missing Year/Race_Num for grouped split; falling back to ungrouped TimeSeriesSplit."
            )
            tscv = TimeSeriesSplit(n_splits=n_splits)
            X_dummy = np.arange(len(features))
            return [(train_idx, val_idx) for train_idx, val_idx in tscv.split(X_dummy)]

        holdout_years = set(config.get("evaluation.holdout_years", []))
        mask = ~features["Year"].isin(holdout_years)
        idx = features.index[mask]
        events = features.loc[idx, ["Year", "Race_Num", "Date"]].copy()
        if "Date" in events.columns:
            events.sort_values(["Year", "Race_Num", "Date"], inplace=True)
        else:
            events.sort_values(["Year", "Race_Num"], inplace=True)
        groups = events[["Year", "Race_Num"]].drop_duplicates().values.tolist()
        m = len(groups)
        if m < n_splits + 1:
            n_splits = max(1, m - 1)
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(n_splits):
            train_end = int(m * (i + 1) / (n_splits + 1))
            val_end = int(m * (i + 2) / (n_splits + 1))
            train_groups = set(tuple(g) for g in groups[:train_end])
            val_groups = set(tuple(g) for g in groups[train_end:val_end])
            train_idx = features.index[
                features[["Year", "Race_Num"]].apply(tuple, axis=1).isin(train_groups)
            ]
            val_idx = features.index[
                features[["Year", "Race_Num"]].apply(tuple, axis=1).isin(val_groups)
            ]
            splits.append((train_idx.values, val_idx.values))
        return splits

    def _purged_group_time_series_splits(
        self,
        features: pd.DataFrame,
        n_splits: int,
        purge_events: int = 1,
        embargo_events: int = 1,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Purged and embargoed grouped time-series splits at event level.

        Purge removes the last `purge_events` event groups from the training window that
        immediately precede the validation window to reduce label proximity leakage.
        Embargo introduces a gap after the validation window for subsequent folds.
        """
        if "Year" not in features.columns or "Race_Num" not in features.columns:
            logger.warning(
                "Missing Year/Race_Num for purged grouped split; falling back to grouped split."
            )
            return self._grouped_time_series_splits(features, n_splits=n_splits)

        holdout_years = set(config.get("evaluation.holdout_years", []))
        mask = ~features["Year"].isin(holdout_years)
        idx = features.index[mask]
        events = features.loc[idx, ["Year", "Race_Num", "Date"]].copy()
        if "Date" in events.columns:
            events.sort_values(["Year", "Race_Num", "Date"], inplace=True)
        else:
            events.sort_values(["Year", "Race_Num"], inplace=True)
        groups = events[["Year", "Race_Num"]].drop_duplicates().values.tolist()
        m = len(groups)
        if m < n_splits + 1:
            n_splits = max(1, m - 1)
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(n_splits):
            train_end = int(m * (i + 1) / (n_splits + 1))
            val_end = int(m * (i + 2) / (n_splits + 1))
            purged_train_end = max(0, train_end - int(max(0, purge_events)))
            train_groups = set(tuple(g) for g in groups[:purged_train_end])
            val_groups = set(tuple(g) for g in groups[train_end:val_end])
            train_idx = features.index[
                features[["Year", "Race_Num"]].apply(tuple, axis=1).isin(train_groups)
            ]
            val_idx = features.index[
                features[["Year", "Race_Num"]].apply(tuple, axis=1).isin(val_groups)
            ]
            splits.append((train_idx.values, val_idx.values))
                                                                                                      
        return splits

    def _optimize_ensemble_weights_metric(
        self,
        y_true: np.ndarray,
        oof_preds: np.ndarray,
        features: Optional[pd.DataFrame] = None,
        model_type: str = "race",
    ) -> np.ndarray:
        """Optimize ensemble weights with MAE or event-level ranking metrics."""
        n_samples, n_models = oof_preds.shape
        if n_models == 1:
            return np.array([1.0])
        metric = str(config.get("training.ensemble.target_metric", "mae")).lower()
        if metric == "mae":
            c = np.concatenate([np.zeros(n_models), np.ones(n_samples)])
            A1 = np.hstack([oof_preds, -np.eye(n_samples)])
            b1 = y_true
            A2 = np.hstack([-oof_preds, -np.eye(n_samples)])
            b2 = -y_true
            A_ub = np.vstack([A1, A2])
            b_ub = np.concatenate([b1, b2])
            A_eq = np.hstack([np.ones((1, n_models)), np.zeros((1, n_samples))])
            b_eq = np.array([1.0])
            bounds = [(0, None)] * (n_models + n_samples)
            res = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
                options={"presolve": True},
            )
            if res.success:
                w = res.x[:n_models]
                if w.sum() <= 0:
                    return np.ones(n_models) / n_models
                return w / w.sum()
            logger.warning("Weight optimization (MAE) failed; using equal weights.")
            return np.ones(n_models) / n_models
                                                  
        if features is None or not {"Year", "Race_Num"}.issubset(features.columns):
            logger.warning("Ranking metric requested but grouping columns missing; falling back to MAE.")
            return self._optimize_ensemble_weights_metric(y_true, oof_preds, None, model_type)
        groups = features[["Year", "Race_Num"]].apply(tuple, axis=1).values
        unique_groups = list(dict.fromkeys(groups))
        idx_by_group: Dict[Any, np.ndarray] = {g: np.where(groups == g)[0] for g in unique_groups}
        def eval_weights(w: np.ndarray) -> float:
            preds = oof_preds.dot(w)
            if metric == "spearman":
                try:
                    from scipy.stats import spearmanr
                    vals = []
                    for g, idxs in idx_by_group.items():
                        if idxs.size < 3:
                            continue
                        corr = spearmanr(y_true[idxs], preds[idxs])[0]
                        if np.isnan(corr):
                            continue
                        vals.append(corr)
                    if not vals:
                        return 1e3
                    return -float(np.mean(vals))
                except Exception:
                    return float(mean_absolute_error(y_true, preds))
            elif metric == "ndcg":
                def dcg(relevances: np.ndarray) -> float:
                    order = np.argsort(-relevances)
                    ranks = np.arange(1, len(relevances) + 1)
                    return float(np.sum((2 ** relevances[order] - 1) / np.log2(ranks + 1)))
                vals = []
                for g, idxs in idx_by_group.items():
                    if idxs.size < 3:
                        continue
                    rel_true = 1.0 / np.maximum(1.0, y_true[idxs].astype(float))
                    rel_pred = 1.0 / np.maximum(1.0, preds[idxs].astype(float))
                    dcg_pred = dcg(rel_pred)
                    dcg_ideal = dcg(rel_true)
                    if dcg_ideal <= 0:
                        continue
                    vals.append(dcg_pred / dcg_ideal)
                if not vals:
                    return 1e3
                return -float(np.mean(vals))
            return float(mean_absolute_error(y_true, preds))
        rng = np.random.default_rng(int(config.get("general.random_seed", 42)))
        samples = int(config.get("training.ensemble.random_weight_samples", 1000))
        best_w = np.ones(n_models) / n_models
        best_score = eval_weights(best_w)
        for i in range(n_models):
            w = np.zeros(n_models)
            w[i] = 1.0
            sc = eval_weights(w)
            if sc < best_score:
                best_score, best_w = sc, w
        for _ in range(samples):
            w = rng.dirichlet(np.ones(n_models))
            sc = eval_weights(w)
            if sc < best_score:
                best_score, best_w = sc, w
        return best_w

    def _build_groups_for_indices(
        self, features: pd.DataFrame, indices: np.ndarray
    ) -> List[int]:
        """Build ranking group sizes (per event) preserving the order of the provided indices."""
        if not {"Year", "Race_Num"}.issubset(features.columns):
            return [len(indices)]
                                        
        sub = features.loc[indices, ["Year", "Race_Num"]]
        keys = list(sub.apply(tuple, axis=1))
        groups: List[int] = []
        last_key = None
        for key in keys:
            if key != last_key:
                groups.append(1)
                last_key = key
            else:
                groups[-1] += 1
        return groups

    def _scores_to_group_ranks(
        self, scores: np.ndarray, features_subset: Optional[pd.DataFrame]
    ) -> np.ndarray:
        """Convert raw ranking scores into position ranks (1=best) within each event group using features_subset grouping."""
        if features_subset is None or not {"Year", "Race_Num"}.issubset(
            features_subset.columns
        ):
            order = np.argsort(-scores)                              
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(scores) + 1)
            return ranks.astype(float)

        ranks = np.zeros_like(scores, dtype=float)
        grouped = features_subset.groupby(["Year", "Race_Num"]).indices
        idx = features_subset.index
        for _, locs in grouped.items():
                                                                                        
            positions = idx.get_indexer(locs)
                                                                             
            if (positions < 0).all():
                try:
                    locs_arr = np.asarray(locs, dtype=int)
                    if locs_arr.ndim == 1 and np.all(
                        (locs_arr >= 0) & (locs_arr < len(idx))
                    ):
                        positions = locs_arr
                    else:
                                                           
                        continue
                except Exception:
                    continue
            else:
                positions = positions[positions >= 0]
                if positions.size == 0:
                    continue

            local_scores = scores[positions]
            order = np.argsort(-local_scores)
            local_ranks = np.empty_like(order)
            local_ranks[order] = np.arange(1, len(local_scores) + 1)
            ranks[positions] = local_ranks.astype(float)
        return ranks

    def _apply_feature_governance(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, model_type: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        allow = config.get(
            f"feature_engineering.governance.allowlists.{model_type}", []
        )
        if isinstance(allow, list) and len(allow) > 0:
            cols = [c for c in X_train.columns if c in allow]
            X_train = X_train[cols]
            X_val = X_val[cols]
        return X_train, X_val

    def _per_event_zscore(
        self, X: pd.DataFrame, features_ref: pd.DataFrame
    ) -> pd.DataFrame:
        if not {"Year", "Race_Num"}.issubset(features_ref.columns):
            return X
        Xn = X.copy()
        keys = features_ref[["Year", "Race_Num"]].apply(tuple, axis=1)
        df = Xn.copy()
        df["__grp__"] = list(keys)
        def _z(g):
            m = g.mean()
            s = g.std(ddof=0).replace(0, np.nan)
            return (g - m) / s
        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = df.groupby("__grp__")[num_cols].transform(_z)
        df.drop(columns=["__grp__"], inplace=True)
        return df
