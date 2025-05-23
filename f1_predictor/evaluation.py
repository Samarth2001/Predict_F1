# f1_predictor/evaluation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from . import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class F1ModelEvaluator:
    """Comprehensive evaluation system for F1 prediction models."""
    
    def __init__(self):
        self.evaluation_results = {}
        self.comparison_results = {}
        
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
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate standard regression metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # F1-specific metrics
            metrics = self._calculate_f1_specific_metrics(y_test, y_pred)
            
            # Combine all metrics
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
        
        # Position accuracy metrics
        position_errors = np.abs(y_pred - y_true)
        
        metrics['top3_accuracy'] = np.mean(position_errors <= 3)
        metrics['top5_accuracy'] = np.mean(position_errors <= 5)
        metrics['exact_accuracy'] = np.mean(position_errors == 0)
        
        # Podium prediction accuracy
        true_podium = (y_true <= 3).astype(int)
        pred_podium = (y_pred <= 3).astype(int)
        
        if np.sum(true_podium) > 0:
            metrics['podium_precision'] = np.sum((true_podium == 1) & (pred_podium == 1)) / np.sum(pred_podium == 1) if np.sum(pred_podium == 1) > 0 else 0
            metrics['podium_recall'] = np.sum((true_podium == 1) & (pred_podium == 1)) / np.sum(true_podium == 1)
            metrics['podium_f1'] = 2 * metrics['podium_precision'] * metrics['podium_recall'] / (metrics['podium_precision'] + metrics['podium_recall']) if (metrics['podium_precision'] + metrics['podium_recall']) > 0 else 0
        
        # Top 10 (points) prediction accuracy
        true_points = (y_true <= 10).astype(int)
        pred_points = (y_pred <= 10).astype(int)
        
        if np.sum(true_points) > 0:
            metrics['points_precision'] = np.sum((true_points == 1) & (pred_points == 1)) / np.sum(pred_points == 1) if np.sum(pred_points == 1) > 0 else 0
            metrics['points_recall'] = np.sum((true_points == 1) & (pred_points == 1)) / np.sum(true_points == 1)
        
        # Weighted accuracy (higher weight for better positions)
        weights = config.POSITION_WEIGHTS[:len(y_true)]
        weighted_errors = position_errors * weights
        metrics['weighted_mae'] = np.mean(weighted_errors)
        
        # Spearman rank correlation
        try:
            from scipy.stats import spearmanr
            metrics['rank_correlation'] = spearmanr(y_true, y_pred)[0]
        except ImportError:
            logger.warning("scipy not available for rank correlation calculation")
        
        return metrics
    
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
            # Merge predictions on driver
            merged = pd.merge(pred1[['Driver', 'Team']], pred2[['Driver']], on='Driver', how='inner')
            
            # Add prediction columns
            if 'Predicted_Quali_Pos' in pred1.columns:
                pos_col = 'Predicted_Quali_Pos'
                rank_col = 'Quali_Rank'
            else:
                pos_col = 'Predicted_Race_Pos'
                rank_col = 'Race_Rank'
            
            merged[f'{label1}_Pos'] = pred1.set_index('Driver')[pos_col].reindex(merged['Driver']).values
            merged[f'{label2}_Pos'] = pred2.set_index('Driver')[pos_col].reindex(merged['Driver']).values
            
            # Calculate rank changes
            merged[f'{label1}_Rank'] = merged[f'{label1}_Pos'].rank()
            merged[f'{label2}_Rank'] = merged[f'{label2}_Pos'].rank()
            merged['Rank_Change'] = merged[f'{label2}_Rank'] - merged[f'{label1}_Rank']
            
            # Sort by rank change
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
            # Prepare data
            hist_data = historical_data.dropna(subset=[target_column]).copy()
            
            if len(hist_data) == 0:
                logger.warning("No historical data available for evaluation.")
                return {}
            
            # Select features
            available_features = [f for f in features_used if f in hist_data.columns]
            missing_features = set(features_used) - set(available_features)
            
            if missing_features:
                logger.warning(f"Missing features in historical data: {missing_features}")
                for feature in missing_features:
                    hist_data[feature] = 0
            
            X_hist = hist_data[features_used].fillna(0)
            y_hist = hist_data[target_column]
            
            # Make predictions
            y_pred = model.predict(X_hist)
            
            # Calculate metrics
            performance = self._calculate_f1_specific_metrics(y_hist.values, y_pred)
            
            # Add time-based analysis
            if 'Year' in hist_data.columns:
                yearly_performance = self._analyze_yearly_performance(hist_data, y_hist.values, y_pred)
                performance['yearly_analysis'] = yearly_performance
            
            # Driver-specific analysis
            if 'Driver' in hist_data.columns:
                driver_performance = self._analyze_driver_performance(hist_data, y_hist.values, y_pred)
                performance['driver_analysis'] = driver_performance
            
            # Circuit-specific analysis
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
            if np.sum(driver_mask) > 5:  # Minimum samples for meaningful analysis
                driver_true = y_true[driver_mask]
                driver_pred = y_pred[driver_mask]
                
                mae = mean_absolute_error(driver_true, driver_pred)
                bias = np.mean(driver_pred - driver_true)  # Positive = overestimated positions
                
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
            if np.sum(circuit_mask) > 3:  # Minimum samples
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
        
        # Model performance summary
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
        
        # Comparison results
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
                
                # Most affected drivers
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
        
        # Generate plots if requested
        if save_plots:
            self._generate_evaluation_plots()
        
        # Save report
        report_text = "\n".join(report_lines)
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"{config.EVALUATION_DIR}/evaluation_report_{timestamp}.txt"
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
            
            # Plot 1: Model performance comparison
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
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(f"{config.EVALUATION_DIR}/model_comparison_{datetime.now().strftime('%Y%m%d')}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            # Plot 2: Prediction comparison scatter plots
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
                    plt.savefig(f"{config.EVALUATION_DIR}/comparison_{comparison_name}_{datetime.now().strftime('%Y%m%d')}.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
            
            logger.info("Evaluation plots generated successfully.")
            
        except Exception as e:
            logger.warning(f"Failed to generate evaluation plots: {e}")
    
    def save_evaluation_data(self):
        """Save evaluation data to JSON."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save evaluation results
            eval_filename = f"{config.EVALUATION_DIR}/evaluation_data_{timestamp}.json"
            with open(eval_filename, 'w') as f:
                json.dump(self.evaluation_results, f, indent=2, default=str)
            
            # Save comparison results
            if self.comparison_results:
                for name, df in self.comparison_results.items():
                    comp_filename = f"{config.EVALUATION_DIR}/comparison_{name}_{timestamp}.csv"
                    df.to_csv(comp_filename, index=False)
            
            logger.info("Evaluation data saved successfully.")
            
        except Exception as e:
            logger.warning(f"Failed to save evaluation data: {e}")

# Convenience functions for backward compatibility
def compare_predictions(pred1: pd.DataFrame, pred2: pd.DataFrame) -> pd.DataFrame:
    """Compare two prediction sets."""
    evaluator = F1ModelEvaluator()
    return evaluator.compare_predictions(pred1, pred2) 