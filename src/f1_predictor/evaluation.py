 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from .config import config

logger = logging.getLogger(__name__)

class F1ModelEvaluator:
    """Comprehensive evaluation system for F1 prediction models."""
    
    def __init__(self):
        self.evaluation_results = {}
        self.comparison_results = {}
        self.per_event_metrics_ = None                        
        self.reliability_ = {}
        self.drift_report_ = {}
        
    def evaluate_model_performance(self, 
                                  model: Any,
                                  X_test: np.ndarray,
                                  y_test: np.ndarray,
                                  model_name: str = "model") -> Dict[str, float]:
        """
        Comprehensive model performance evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {model_name} performance...")
        
        try:
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            metrics = self._calculate_f1_specific_metrics(y_test, y_pred)

            try:
                if isinstance(X_test, (pd.DataFrame,)) and {'Year', 'Race_Num'}.issubset(X_test.columns):
                    ev_agg, per_event_df = self._calculate_event_level_metrics(X_test, y_test, y_pred)
                    metrics.update(ev_agg)
                    self.per_event_metrics_ = per_event_df
            except Exception:
                pass

            try:
                rel = self._calibration_diagnostics(model, X_test, y_test, y_pred)
                if rel:
                    summary = rel.get('summary', {})
                    for k, v in summary.items():
                        metrics[f'calibration_{k}'] = float(v)
                    self.reliability_ = rel
            except Exception:
                pass

            try:
                if getattr(self, 'per_event_metrics_', None) is not None and not self.per_event_metrics_.empty:
                    drift = self._drift_tracking(self.per_event_metrics_)
                    if drift:
                        summary = drift.get('summary', {})
                        for k, v in summary.items():
                            metrics[f'drift_{k}'] = float(v)
                        self.drift_report_ = drift
            except Exception:
                pass
            
            all_metrics = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                **metrics
            }
            
            self.evaluation_results[model_name] = all_metrics
            
            logger.info(f"{model_name} evaluation completed.")
            return all_metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {}
    
    def _calculate_f1_specific_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate F1-specific evaluation metrics."""
        metrics = {}
        
        position_errors = np.abs(y_pred - y_true)
        metrics['top3_accuracy'] = np.mean(position_errors <= 3)
        metrics['top5_accuracy'] = np.mean(position_errors <= 5)
        metrics['exact_accuracy'] = np.mean(position_errors == 0)
        true_podium = (y_true <= 3).astype(int)
        pred_podium = (y_pred <= 3).astype(int)
        if np.sum(true_podium) > 0:
            metrics['podium_precision'] = np.sum((true_podium == 1) & (pred_podium == 1)) / np.sum(pred_podium == 1) if np.sum(pred_podium == 1) > 0 else 0
            metrics['podium_recall'] = np.sum((true_podium == 1) & (pred_podium == 1)) / np.sum(true_podium == 1)
            metrics['podium_f1'] = 2 * metrics['podium_precision'] * metrics['podium_recall'] / (metrics['podium_precision'] + metrics['podium_recall']) if (metrics['podium_precision'] + metrics['podium_recall']) > 0 else 0
        true_points = (y_true <= 10).astype(int)
        pred_points = (y_pred <= 10).astype(int)
        if np.sum(true_points) > 0:
            metrics['points_precision'] = np.sum((true_points == 1) & (pred_points == 1)) / np.sum(pred_points == 1) if np.sum(pred_points == 1) > 0 else 0
            metrics['points_recall'] = np.sum((true_points == 1) & (pred_points == 1)) / np.sum(true_points == 1)
        weights = np.ones(len(y_true))
        weighted_errors = position_errors * weights
        metrics['weighted_mae'] = np.mean(weighted_errors)
        try:
            from scipy.stats import spearmanr
            metrics['rank_correlation'] = spearmanr(y_true, y_pred)[0]
        except ImportError:
            logger.warning("scipy not available for rank correlation calculation")
        
        return metrics

    def _calculate_event_level_metrics(self, X: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Compute per-event metrics then aggregate, including NDCG@k and MAP@k."""
        try:
            from scipy.stats import spearmanr
        except Exception:
            spearmanr = None                
        groups = X[['Year', 'Race_Num']].apply(tuple, axis=1).values
        unique_groups = list(dict.fromkeys(groups))
        k_list: List[int] = list(map(int, config.get('evaluation.rank_metrics.k_list', config.get('prediction.top_k_thresholds', [3, 10]))))
        rows: List[Dict[str, Any]] = []
        for g in unique_groups:
            idxs = np.where(groups == g)[0]
            if idxs.size < 3:
                continue
            yt = y_true[idxs].astype(float)
            yp = y_pred[idxs].astype(float)
            ev: Dict[str, Any] = {
                'Year': int(X.iloc[idxs[0]]['Year']) if 'Year' in X.columns else None,
                'Race_Num': int(X.iloc[idxs[0]]['Race_Num']) if 'Race_Num' in X.columns else None,
                'event_size': int(idxs.size),
            }
            ev['event_mae'] = float(mean_absolute_error(yt, yp))
            if spearmanr is not None:
                try:
                    ev['event_spearman'] = float(spearmanr(yt, yp)[0])
                except Exception:
                    ev['event_spearman'] = np.nan
            rel_true_full = 1.0 / np.maximum(1.0, yt)
            rel_pred_full = 1.0 / np.maximum(1.0, yp)
            ev['event_ndcg'] = self._ndcg_at_k(rel_pred_full, rel_true_full, k=None)
            pred_order = np.argsort(yp)
            true_order = np.argsort(yt)
            true_top_set_cache = {}
            for k in k_list:
                rel_true_k = np.zeros_like(yt)
                rel_true_k[true_order[:k]] = 1.0
                rel_pred_k = np.zeros_like(yt)
                rel_pred_k[pred_order[:k]] = 1.0
                ev[f'ndcg@{k}'] = self._ndcg_at_k(rel_pred_k, rel_true_k, k=k)
                if k not in true_top_set_cache:
                    true_top_set_cache[k] = set(map(int, true_order[:k]))
                ev[f'map@{k}'] = self._map_at_k(pred_order, true_top_set_cache[k], k)
            rows.append(ev)
        per_event_df = pd.DataFrame.from_records(rows)
        agg: Dict[str, float] = {}
        if not per_event_df.empty:
            cols = ['event_mae', 'event_spearman', 'event_ndcg'] + [c for c in per_event_df.columns if c.startswith(('ndcg@', 'map@'))]
            for c in cols:
                if c in per_event_df.columns and per_event_df[c].notna().any():
                    agg[c] = float(per_event_df[c].dropna().mean())
        return agg, per_event_df

    @staticmethod
    def _dcg(relevances: np.ndarray, k: Optional[int] = None) -> float:
        rel = np.asarray(relevances, dtype=float)
        if k is not None:
            rel = rel[:k]
        if rel.size == 0:
            return 0.0
        gains = (2.0 ** rel - 1.0)
        discounts = np.log2(np.arange(2, rel.size + 2))
        return float(np.sum(gains / discounts))

    def _ndcg_at_k(self, rel_pred: np.ndarray, rel_true: np.ndarray, k: Optional[int] = None) -> float:
        order = np.argsort(-np.asarray(rel_pred, dtype=float))
        ordered_true = np.asarray(rel_true, dtype=float)[order]
        dcg_pred = self._dcg(ordered_true, k=k)
        ideal_true = np.sort(np.asarray(rel_true, dtype=float))[::-1]
        dcg_ideal = self._dcg(ideal_true, k=k)
        if dcg_ideal <= 0:
            return 0.0
        return float(dcg_pred / dcg_ideal)

    @staticmethod
    def _map_at_k(pred_order: np.ndarray, true_top_indices: set, k: int) -> float:
        hits = 0
        sum_precisions = 0.0
        limit = min(k, len(pred_order))
        for i in range(limit):
            idx = int(pred_order[i])
            if idx in true_top_indices:
                hits += 1
                sum_precisions += hits / float(i + 1)
        if hits == 0:
            return 0.0
        denom = float(min(k, max(1, len(true_top_indices))))
        return float(sum_precisions / denom)

    def _calibration_diagnostics(self, model: Any, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Build reliability curves and summarize calibration."""
                        
        sigma: float
        try:
            sigma = float(getattr(model, 'calibration_', {}).get('oof_residual_std', np.nan))
        except Exception:
            sigma = np.nan
        if not np.isfinite(sigma) or sigma <= 0:
            try:
                sigma = float(np.std(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))
            except Exception:
                sigma = 1.5
        sigma = max(1e-6, sigma)

        ks: List[int] = list(map(int, config.get('evaluation.rank_metrics.k_list', config.get('prediction.top_k_thresholds', [3, 10]))))
        mu = np.asarray(y_pred, dtype=float)
        curves: Dict[str, Any] = {}
        summary: Dict[str, float] = {}

        try:
            from sklearn.calibration import calibration_curve                
        except Exception:
            calibration_curve = None                

        for k in ks:
            from scipy.stats import norm as _norm               
            probs = _norm.cdf((float(k) + 0.5 - mu) / sigma)
            events = (np.asarray(y_true, dtype=float) <= float(k)).astype(int)
            n_bins = int(config.get('evaluation.calibration.n_bins', 10))
            if calibration_curve is not None:
                frac_pos, mean_pred = calibration_curve(events, probs, n_bins=n_bins, strategy='quantile')
            else:
                bins = np.linspace(0.0, 1.0, n_bins + 1)
                inds = np.digitize(probs, bins) - 1
                mean_pred_list: List[float] = []
                frac_pos_list: List[float] = []
                for b in range(n_bins):
                    mask = inds == b
                    if np.any(mask):
                        mean_pred_list.append(float(np.mean(probs[mask])))
                        frac_pos_list.append(float(np.mean(events[mask])))
                mean_pred = np.asarray(mean_pred_list)
                frac_pos = np.asarray(frac_pos_list)
            try:
                bins = np.linspace(0.0, 1.0, n_bins + 1)
                inds = np.digitize(np.asarray(probs), bins) - 1
                weights = np.array([np.sum(inds == b) for b in range(n_bins)], dtype=float)
                weights = weights[: len(mean_pred)]
                if weights.sum() > 0:
                    ece = float(np.sum(weights * np.abs(frac_pos - mean_pred)) / np.sum(weights))
                else:
                    ece = float(np.mean(np.abs(frac_pos - mean_pred)))
            except Exception:
                ece = float(np.mean(np.abs(frac_pos - mean_pred)))
            curves[str(k)] = {'mean_pred': mean_pred.tolist(), 'frac_pos': frac_pos.tolist()}
            summary[f'ece_top{k}'] = ece

        return {'curves': curves, 'summary': summary}

    def _drift_tracking(self, per_event_df: pd.DataFrame) -> Dict[str, Any]:
        """Track rolling performance across seasons and emit drift summary."""
        if per_event_df is None or per_event_df.empty:
            return {}
        df = per_event_df.copy()
        sort_cols = [c for c in ['Year', 'Race_Num'] if c in df.columns]
        if sort_cols:
            df.sort_values(sort_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)

        window = int(config.get('evaluation.drift.rolling_window_events', 6))
        thr_mae = float(config.get('evaluation.drift.threshold_mae', 0.5))
        thr_spear = float(config.get('evaluation.drift.threshold_spearman', 0.15))

        df['mae_roll'] = df['event_mae'].rolling(window, min_periods=max(2, window // 2)).mean()
        if 'event_spearman' in df.columns:
            df['spearman_roll'] = df['event_spearman'].rolling(window, min_periods=max(2, window // 2)).mean()

        hist_mean_mae = float(df['event_mae'].mean())
        hist_mean_s = float(df['event_spearman'].mean()) if 'event_spearman' in df.columns else np.nan
        recent_mae = float(df['mae_roll'].iloc[-1]) if df['mae_roll'].notna().any() else np.nan
        recent_s = float(df['spearman_roll'].iloc[-1]) if 'spearman_roll' in df.columns and df['spearman_roll'].notna().any() else np.nan

        drift_mae = float(recent_mae - hist_mean_mae) if np.isfinite(recent_mae) else np.nan
        drift_s = float(hist_mean_s - recent_s) if np.isfinite(recent_s) and np.isfinite(hist_mean_s) else np.nan
        alert_mae = bool(np.isfinite(drift_mae) and abs(drift_mae) >= thr_mae)
        alert_s = bool(np.isfinite(drift_s) and abs(drift_s) >= thr_spear)

        try:
            out_dir = config.get('paths.evaluation_dir')
            os.makedirs(out_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df.index, df['event_mae'], label='Event MAE', alpha=0.5)
            ax.plot(df.index, df['mae_roll'], label=f'Rolling MAE (w={window})', linewidth=2)
            ax.axhline(hist_mean_mae, color='k', linestyle='--', alpha=0.5, label='Historical mean MAE')
            ax.set_title('Rolling MAE over events')
            ax.set_xlabel('Event index (chronological)')
            ax.set_ylabel('MAE')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'rolling_mae_{datetime.now().strftime("%Y%m%d")}.png'), dpi=300)
            plt.close()
            if 'spearman_roll' in df.columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df.index, df['event_spearman'], label='Event Spearman', alpha=0.5)
                ax.plot(df.index, df['spearman_roll'], label=f'Rolling Spearman (w={window})', linewidth=2)
                ax.axhline(hist_mean_s, color='k', linestyle='--', alpha=0.5, label='Historical mean Spearman')
                ax.set_title('Rolling Spearman over events')
                ax.set_xlabel('Event index (chronological)')
                ax.set_ylabel('Spearman correlation')
                ax.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f'rolling_spearman_{datetime.now().strftime("%Y%m%d")}.png'), dpi=300)
                plt.close()
        except Exception:
            pass

        summary = {
            'mae_recent_minus_hist': float(drift_mae) if np.isfinite(drift_mae) else 0.0,
            'spearman_hist_minus_recent': float(drift_s) if np.isfinite(drift_s) else 0.0,
            'alert_mae': 1.0 if alert_mae else 0.0,
            'alert_spearman': 1.0 if alert_s else 0.0,
        }
        return {'summary': summary, 'per_event_with_roll': df.to_dict(orient='list')}
    
    def compare_predictions(self, 
                          pred1: pd.DataFrame, 
                          pred2: pd.DataFrame,
                          label1: str = "Prediction 1",
                          label2: str = "Prediction 2") -> pd.DataFrame:
        """
        Compare two prediction sets.
        
        Args:
            pred1: First prediction dataframe
            pred2: Second prediction dataframe
            label1: Label for first prediction
            label2: Label for second prediction
            
        Returns:
            Comparison dataframe
        """
        logger.info(f"Comparing {label1} vs {label2}")
        
        try:
            merged = pd.merge(pred1[['Driver', 'Team']], pred2[['Driver']], on='Driver', how='inner')
            if 'Predicted_Quali_Pos' in pred1.columns:
                pos_col = 'Predicted_Quali_Pos'
                rank_col = 'Quali_Rank'
            else:
                pos_col = 'Predicted_Race_Pos'
                rank_col = 'Race_Rank'
            merged[f'{label1}_Pos'] = pred1.set_index('Driver')[pos_col].reindex(merged['Driver']).values
            merged[f'{label2}_Pos'] = pred2.set_index('Driver')[pos_col].reindex(merged['Driver']).values
            merged[f'{label1}_Rank'] = merged[f'{label1}_Pos'].rank()
            merged[f'{label2}_Rank'] = merged[f'{label2}_Pos'].rank()
            merged['Rank_Change'] = merged[f'{label2}_Rank'] - merged[f'{label1}_Rank']
            comparison = merged.sort_values('Rank_Change', ascending=False)
            
            self.comparison_results[f"{label1}_vs_{label2}"] = comparison
            
            return comparison
            
        except Exception as e:
            logger.error(f"Prediction comparison failed: {e}")
            return pd.DataFrame()
    
    def evaluate_historical_performance(self, 
                                      model: Any,
                                      historical_data: pd.DataFrame,
                                      features_used: List[str],
                                      target_column: str) -> Dict[str, Any]:
        """
        Evaluate model performance on historical data.
        
        Args:
            model: Trained model
            historical_data: Historical data for evaluation
            features_used: List of feature names
            target_column: Target column name
            
        Returns:
            Historical performance metrics
        """
        logger.info("Evaluating historical performance...")
        
        try:
            hist_data = historical_data.dropna(subset=[target_column]).copy()
            
            if len(hist_data) == 0:
                logger.warning("No historical data available for evaluation.")
                return {}
            
            available_features = [f for f in features_used if f in hist_data.columns]
            missing_features = set(features_used) - set(available_features)
            
            if missing_features:
                logger.warning(f"Missing features in historical data: {missing_features}")
                for feature in missing_features:
                    hist_data[feature] = 0
            
            X_hist = hist_data[features_used].fillna(0)
            y_hist = hist_data[target_column]
            
            y_pred = model.predict(X_hist)
            
            performance = self._calculate_f1_specific_metrics(y_hist.values, y_pred)
            
            if 'Year' in hist_data.columns:
                yearly_performance = self._analyze_yearly_performance(hist_data, y_hist.values, y_pred)
                performance['yearly_analysis'] = yearly_performance
            
            if 'Driver' in hist_data.columns:
                driver_performance = self._analyze_driver_performance(hist_data, y_hist.values, y_pred)
                performance['driver_analysis'] = driver_performance
            
            if 'Circuit' in hist_data.columns:
                circuit_performance = self._analyze_circuit_performance(hist_data, y_hist.values, y_pred)
                performance['circuit_analysis'] = circuit_performance
            
            return performance
            
        except Exception as e:
            logger.error(f"Historical performance evaluation failed: {e}")
            return {}
    
    def _analyze_yearly_performance(self, data: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Analyze performance by year."""
        yearly_metrics = {}
        
        for year in data['Year'].unique():
            year_mask = data['Year'] == year
            if np.sum(year_mask) > 0:
                year_true = y_true[year_mask]
                year_pred = y_pred[year_mask]
                
                mae = mean_absolute_error(year_true, year_pred)
                top3_acc = np.mean(np.abs(year_pred - year_true) <= 3)
                
                yearly_metrics[str(year)] = {
                    'mae': mae,
                    'top3_accuracy': top3_acc,
                    'sample_count': len(year_true)
                }
        
        return yearly_metrics
    
    def _analyze_driver_performance(self, data: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Analyze performance by driver."""
        driver_metrics = {}
        
        for driver in data['Driver'].unique():
            driver_mask = data['Driver'] == driver
            if np.sum(driver_mask) > 5:                                           
                driver_true = y_true[driver_mask]
                driver_pred = y_pred[driver_mask]
                
                mae = mean_absolute_error(driver_true, driver_pred)
                bias = np.mean(driver_pred - driver_true)                                      
                
                driver_metrics[driver] = {
                    'mae': mae,
                    'bias': bias,
                    'sample_count': len(driver_true)
                }
        
        return driver_metrics
    
    def _analyze_circuit_performance(self, data: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Analyze performance by circuit."""
        circuit_metrics = {}
        
        for circuit in data['Circuit'].unique():
            circuit_mask = data['Circuit'] == circuit
            if np.sum(circuit_mask) > 3:                   
                circuit_true = y_true[circuit_mask]
                circuit_pred = y_pred[circuit_mask]
                
                mae = mean_absolute_error(circuit_true, circuit_pred)
                std_error = np.std(np.abs(circuit_pred - circuit_true))
                
                circuit_metrics[circuit] = {
                    'mae': mae,
                    'std_error': std_error,
                    'sample_count': len(circuit_true)
                }
        
        return circuit_metrics
    
    def generate_evaluation_report(self, save_plots: bool = True) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            save_plots: Whether to save evaluation plots
            
        Returns:
            Report text
        """
        logger.info("Generating evaluation report...")
        
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("F1 PREDICTION MODEL EVALUATION REPORT")
        report_lines.append("="*60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
                                   
        if self.evaluation_results:
            report_lines.append("MODEL PERFORMANCE SUMMARY")
            report_lines.append("-" * 30)
            
            for model_name, metrics in self.evaluation_results.items():
                report_lines.append(f"\n{model_name.upper()}:")
                report_lines.append(f"  MAE: {metrics.get('mae', 0):.3f}")
                report_lines.append(f"  RMSE: {metrics.get('rmse', 0):.3f}")
                report_lines.append(f"  R²: {metrics.get('r2', 0):.3f}")
                report_lines.append(f"  Top-3 Accuracy: {metrics.get('top3_accuracy', 0):.3f}")
                report_lines.append(f"  Top-5 Accuracy: {metrics.get('top5_accuracy', 0):.3f}")
                report_lines.append(f"  Podium Precision: {metrics.get('podium_precision', 0):.3f}")
                report_lines.append(f"  Rank Correlation: {metrics.get('rank_correlation', 0):.3f}")
                                                       
                if any(k.startswith(('event_', 'ndcg@', 'map@')) for k in metrics.keys()):
                    report_lines.append("  Event-level aggregates:")
                    if 'event_mae' in metrics:
                        report_lines.append(f"    Event MAE (mean): {metrics.get('event_mae', 0):.3f}")
                    if 'event_spearman' in metrics:
                        report_lines.append(f"    Event Spearman (mean): {metrics.get('event_spearman', 0):.3f}")
                    if 'event_ndcg' in metrics:
                        report_lines.append(f"    Event NDCG (mean): {metrics.get('event_ndcg', 0):.3f}")
                    for k in config.get('evaluation.rank_metrics.k_list', config.get('prediction.top_k_thresholds', [3, 10])):
                        k = int(k)
                        if f'ndcg@{k}' in metrics:
                            report_lines.append(f"    NDCG@{k}: {metrics.get(f'ndcg@{k}', 0):.3f}")
                        if f'map@{k}' in metrics:
                            report_lines.append(f"    MAP@{k}: {metrics.get(f'map@{k}', 0):.3f}")
                             
                cal_entries = {k: v for k, v in metrics.items() if k.startswith('calibration_')}
                if cal_entries:
                    report_lines.append("  Calibration diagnostics:")
                    for k, v in cal_entries.items():
                        label = k.replace('calibration_', '').upper()
                        report_lines.append(f"    {label}: {v:.3f}")
                       
                drift_entries = {k: v for k, v in metrics.items() if k.startswith('drift_')}
                if drift_entries:
                    report_lines.append("  Drift tracking:")
                    report_lines.append(f"    MAE recent - hist: {metrics.get('drift_mae_recent_minus_hist', 0.0):.3f}")
                    report_lines.append(f"    Spearman hist - recent: {metrics.get('drift_spearman_hist_minus_recent', 0.0):.3f}")
        
                            
        if self.comparison_results:
            report_lines.append("\n\nPREDICTION COMPARISONS")
            report_lines.append("-" * 30)
            
            for comparison_name, comparison_df in self.comparison_results.items():
                report_lines.append(f"\n{comparison_name}:")
                
                significant_changes = len(comparison_df[comparison_df['Rank_Change'].abs() >= 2])
                avg_change = comparison_df['Rank_Change'].abs().mean()
                max_change = comparison_df['Rank_Change'].abs().max()
                
                report_lines.append(f"  Significant changes (≥2 positions): {significant_changes}")
                report_lines.append(f"  Average position change: {avg_change:.2f}")
                report_lines.append(f"  Maximum position change: {max_change:.0f}")
                
                                       
                if not comparison_df.empty:
                    top_positive = comparison_df.nlargest(3, 'Rank_Change')
                    top_negative = comparison_df.nsmallest(3, 'Rank_Change')
                    
                    report_lines.append(f"  Biggest gainers:")
                    for _, row in top_positive.iterrows():
                        if row['Rank_Change'] > 0:
                            report_lines.append(f"    {row['Driver']}: +{row['Rank_Change']:.0f} positions")
                    
                    report_lines.append(f"  Biggest losers:")
                    for _, row in top_negative.iterrows():
                        if row['Rank_Change'] < 0:
                            report_lines.append(f"    {row['Driver']}: {row['Rank_Change']:.0f} positions")
        
                                     
        if save_plots:
            self._generate_evaluation_plots()
        
                     
        report_text = "\n".join(report_lines)
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_dir = config.get('paths.evaluation_dir')
            os.makedirs(report_dir, exist_ok=True)
            report_filename = f"{report_dir}/evaluation_report_{timestamp}.txt"
            with open(report_filename, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved: {report_filename}")
        except Exception as e:
            logger.warning(f"Failed to save evaluation report: {e}")
        
        return report_text
    
    def _generate_evaluation_plots(self):
        """Generate evaluation plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('default')
            sns.set_palette("husl")
            
                                                  
            if len(self.evaluation_results) > 1:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle('Model Performance Comparison', fontsize=16)
                
                models = list(self.evaluation_results.keys())
                metrics = ['mae', 'top3_accuracy', 'podium_precision', 'rank_correlation']
                metric_labels = ['MAE', 'Top-3 Accuracy', 'Podium Precision', 'Rank Correlation']
                
                for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                    ax = axes[i//2, i%2]
                    values = [self.evaluation_results[model].get(metric, 0) for model in models]
                    
                    bars = ax.bar(models, values)
                    ax.set_title(label)
                    ax.set_ylabel(label)
                    
                                              
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                out_dir = config.get('paths.evaluation_dir')
                os.makedirs(out_dir, exist_ok=True)
                plt.savefig(f"{out_dir}/model_comparison_{datetime.now().strftime('%Y%m%d')}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
                                                         
            for comparison_name, comparison_df in self.comparison_results.items():
                if not comparison_df.empty:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    col1 = [col for col in comparison_df.columns if col.endswith('_Pos')][0]
                    col2 = [col for col in comparison_df.columns if col.endswith('_Pos')][1]
                    
                    scatter = ax.scatter(comparison_df[col1], comparison_df[col2], 
                                       c=comparison_df['Rank_Change'].abs(), 
                                       cmap='Reds', alpha=0.7, s=60)
                    
                    ax.plot([1, 20], [1, 20], 'k--', alpha=0.5, label='Perfect Agreement')
                    ax.set_xlabel(col1.replace('_Pos', ''))
                    ax.set_ylabel(col2.replace('_Pos', ''))
                    ax.set_title(f'Prediction Comparison: {comparison_name}')
                    ax.legend()
                    
                    plt.colorbar(scatter, label='Position Change')
                    plt.tight_layout()
                    out_dir = config.get('paths.evaluation_dir')
                    os.makedirs(out_dir, exist_ok=True)
                    plt.savefig(f"{out_dir}/comparison_{comparison_name}_{datetime.now().strftime('%Y%m%d')}.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
            
                                                     
            if self.reliability_ and 'curves' in self.reliability_:
                for k, data in self.reliability_.get('curves', {}).items():
                    try:
                        fig, ax = plt.subplots(figsize=(6, 6))
                        mp = np.asarray(data.get('mean_pred', []), dtype=float)
                        fp = np.asarray(data.get('frac_pos', []), dtype=float)
                        if mp.size == 0 or fp.size == 0:
                            continue
                        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
                        ax.plot(mp, fp, marker='o')
                        ax.set_xlabel('Predicted probability')
                        ax.set_ylabel('Empirical frequency')
                        ax.set_title(f'Reliability curve - Top {k}')
                        ax.legend()
                        plt.tight_layout()
                        out_dir = config.get('paths.evaluation_dir')
                        os.makedirs(out_dir, exist_ok=True)
                        plt.savefig(os.path.join(out_dir, f'reliability_top{k}_{datetime.now().strftime("%Y%m%d")}.png'), dpi=300)
                        plt.close()
                    except Exception:
                        pass

                                                                                     
            
            logger.info("Evaluation plots generated successfully.")
            
        except Exception as e:
            logger.warning(f"Failed to generate evaluation plots: {e}")
    
    def save_evaluation_data(self):
        """Save evaluation data to JSON."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
                                     
            out_dir = config.get('paths.evaluation_dir')
            os.makedirs(out_dir, exist_ok=True)
            eval_filename = f"{out_dir}/evaluation_data_{timestamp}.json"
            with open(eval_filename, 'w') as f:
                json.dump(self.evaluation_results, f, indent=2, default=str)
            
                                     
            if self.comparison_results:
                for name, df in self.comparison_results.items():
                    comp_filename = f"{out_dir}/comparison_{name}_{timestamp}.csv"
                    df.to_csv(comp_filename, index=False)
                                    
            if self.per_event_metrics_ is not None and not self.per_event_metrics_.empty:
                per_event_path = f"{out_dir}/per_event_metrics_{timestamp}.csv"
                self.per_event_metrics_.to_csv(per_event_path, index=False)
                                      
            if self.reliability_:
                rel_path = f"{out_dir}/reliability_{timestamp}.json"
                with open(rel_path, 'w') as f:
                    json.dump(self.reliability_, f, indent=2)
                               
            if self.drift_report_:
                drift_path = f"{out_dir}/drift_{timestamp}.json"
                with open(drift_path, 'w') as f:
                    json.dump(self.drift_report_, f, indent=2)
            
            logger.info("Evaluation data saved successfully.")
            
        except Exception as e:
            logger.warning(f"Failed to save evaluation data: {e}")

                                                  
def compare_predictions(pred1: pd.DataFrame, pred2: pd.DataFrame) -> pd.DataFrame:
    """Compare two prediction sets."""
    evaluator = F1ModelEvaluator()
    return evaluator.compare_predictions(pred1, pred2) 