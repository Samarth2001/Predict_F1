"""
Validation script for comparing F1 predictions against actual race results.
Fetches actual results, compares with predictions, and provides insights for model improvement.
"""

import pandas as pd
import numpy as np
import fastf1
import logging
import json
import os
from datetime import datetime
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from pathlib import Path

# Ensure 'src' is importable when running from repo root
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from f1_predictor.config import config
from f1_predictor.data_loader import F1DataCollector
from f1_predictor.evaluation import F1ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

fastf1.Cache.enable_cache(config.get("paths.cache_dir"))


class PredictionValidator:
    """Validates predictions against actual race results."""
    
    def __init__(self):
        self.evaluator = F1ModelEvaluator()
        self.data_collector = F1DataCollector()
        
    def fetch_actual_race_results(self, year: int, race_name: str) -> Optional[pd.DataFrame]:
        """Fetch actual race results from FastF1."""
        logger.info(f"Fetching actual results for {year} {race_name}...")
        
        try:
            schedule = fastf1.get_event_schedule(year)
            event = schedule[schedule['EventName'].str.contains(race_name.replace(' Grand Prix', ''), case=False)]
            
            if event.empty:
                logger.error(f"Race '{race_name}' not found in {year} schedule")
                return None
            
            round_num = event.iloc[0]['RoundNumber']
            session = fastf1.get_session(year, round_num, 'Race')
            session.load()
            
            results = session.results
            actual_results = pd.DataFrame({
                'Driver': results['Abbreviation'].values,
                'Team': results['TeamName'].values,
                'Actual_Race_Pos': results['Position'].values,
                'Grid_Pos': results['GridPosition'].values,
                'Status': results['Status'].values,
                'Points': results['Points'].values
            })
            
            actual_results['Actual_Race_Pos'] = pd.to_numeric(
                actual_results['Actual_Race_Pos'], errors='coerce'
            )
            actual_results = actual_results.dropna(subset=['Actual_Race_Pos'])
            
            logger.info(f"Successfully fetched results for {len(actual_results)} drivers")
            return actual_results
            
        except Exception as e:
            logger.error(f"Error fetching actual results: {e}")
            return None
    
    def load_predictions(self, prediction_file: str) -> Optional[pd.DataFrame]:
        """Load prediction CSV file."""
        try:
            predictions = pd.read_csv(prediction_file)
            logger.info(f"Loaded predictions for {len(predictions)} drivers")
            return predictions
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            return None
    
    def compare_predictions(
        self, 
        predictions: pd.DataFrame, 
        actual_results: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """Compare predictions with actual results and calculate metrics."""
        
        merged = predictions.merge(
            actual_results[['Driver', 'Actual_Race_Pos', 'Grid_Pos', 'Status', 'Points']], 
            on='Driver', 
            how='inner'
        )
        
        merged['Position_Error'] = merged['Predicted_Race_Pos'] - merged['Actual_Race_Pos']
        merged['Absolute_Error'] = merged['Position_Error'].abs()
        merged['Grid_to_Finish'] = merged['Actual_Race_Pos'] - merged['Grid_Pos']
        
        metrics = self._calculate_metrics(merged)
        error_analysis = self._analyze_errors(merged)
        metrics.update(error_analysis)
        
        return merged, metrics
    
    def _calculate_metrics(self, merged: pd.DataFrame) -> Dict:
        """Calculate comprehensive F1-specific metrics."""
        
        metrics = {
            'mae': merged['Absolute_Error'].mean(),
            'rmse': np.sqrt((merged['Position_Error'] ** 2).mean()),
            'median_error': merged['Absolute_Error'].median(),
            'max_error': merged['Absolute_Error'].max(),
            'std_error': merged['Position_Error'].std(),
        }
        
        if len(merged) >= 3:
            top3_predicted = set(merged.nsmallest(3, 'Predicted_Race_Pos')['Driver'])
            top3_actual = set(merged.nsmallest(3, 'Actual_Race_Pos')['Driver'])
            metrics['podium_accuracy'] = len(top3_predicted & top3_actual) / 3.0
            
            top5_predicted = set(merged.nsmallest(5, 'Predicted_Race_Pos')['Driver'])
            top5_actual = set(merged.nsmallest(5, 'Actual_Race_Pos')['Driver'])
            metrics['top5_accuracy'] = len(top5_predicted & top5_actual) / 5.0
            
            top10_predicted = set(merged.nsmallest(10, 'Predicted_Race_Pos')['Driver'])
            top10_actual = set(merged.nsmallest(10, 'Actual_Race_Pos')['Driver'])
            metrics['top10_accuracy'] = len(top10_predicted & top10_actual) / 10.0
        
        within_3_positions = (merged['Absolute_Error'] <= 3).sum() / len(merged)
        within_5_positions = (merged['Absolute_Error'] <= 5).sum() / len(merged)
        metrics['accuracy_within_3'] = within_3_positions
        metrics['accuracy_within_5'] = within_5_positions
        
        from scipy.stats import spearmanr
        correlation, _ = spearmanr(merged['Predicted_Race_Pos'], merged['Actual_Race_Pos'])
        metrics['spearman_correlation'] = correlation
        
        return metrics
    
    def _analyze_errors(self, merged: pd.DataFrame) -> Dict:
        """Analyze prediction errors by category."""
        
        analysis = {}
        
        top_teams = ['Mercedes', 'Red Bull Racing', 'Ferrari', 'McLaren']
        top_team_mask = merged['Team'].isin(top_teams)
        if top_team_mask.sum() > 0:
            analysis['top_team_mae'] = merged[top_team_mask]['Absolute_Error'].mean()
            analysis['midfield_mae'] = merged[~top_team_mask]['Absolute_Error'].mean()
        
        front_runners = merged[merged['Grid_Pos'] <= 5]
        if len(front_runners) > 0:
            analysis['front_runners_mae'] = front_runners['Absolute_Error'].mean()
        
        back_markers = merged[merged['Grid_Pos'] >= 15]
        if len(back_markers) > 0:
            analysis['back_markers_mae'] = back_markers['Absolute_Error'].mean()
        
        biggest_errors = merged.nlargest(5, 'Absolute_Error')[
            ['Driver', 'Team', 'Predicted_Race_Pos', 'Actual_Race_Pos', 'Absolute_Error', 'Status']
        ]
        analysis['biggest_errors'] = biggest_errors.to_dict('records')
        
        merged_with_abs = merged.copy()
        merged_with_abs['Grid_to_Finish_Abs'] = merged_with_abs['Grid_to_Finish'].abs()
        biggest_surprises = merged_with_abs.nlargest(5, 'Grid_to_Finish_Abs')[
            ['Driver', 'Team', 'Grid_Pos', 'Actual_Race_Pos', 'Grid_to_Finish', 'Status']
        ]
        analysis['biggest_surprises'] = biggest_surprises.to_dict('records')
        
        return analysis
    
    def generate_report(
        self, 
        merged: pd.DataFrame, 
        metrics: Dict, 
        race_name: str, 
        year: int,
        output_dir: str = "validation"
    ):
        """Generate comprehensive validation report."""
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_lines = [
            f"=" * 80,
            f"PREDICTION VALIDATION REPORT",
            f"Race: {year} {race_name}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"=" * 80,
            "",
            "OVERALL METRICS:",
            "-" * 80,
            f"Mean Absolute Error (MAE):        {metrics['mae']:.2f} positions",
            f"Root Mean Squared Error (RMSE):   {metrics['rmse']:.2f} positions",
            f"Median Error:                      {metrics['median_error']:.2f} positions",
            f"Max Error:                         {metrics['max_error']:.2f} positions",
            f"Standard Deviation:                {metrics['std_error']:.2f} positions",
            "",
            f"Spearman Correlation:              {metrics.get('spearman_correlation', 0):.3f}",
            "",
            "ACCURACY METRICS:",
            "-" * 80,
            f"Podium Accuracy (Top 3):           {metrics.get('podium_accuracy', 0)*100:.1f}%",
            f"Top 5 Accuracy:                    {metrics.get('top5_accuracy', 0)*100:.1f}%",
            f"Top 10 Accuracy:                   {metrics.get('top10_accuracy', 0)*100:.1f}%",
            f"Within ±3 Positions:               {metrics.get('accuracy_within_3', 0)*100:.1f}%",
            f"Within ±5 Positions:               {metrics.get('accuracy_within_5', 0)*100:.1f}%",
            "",
        ]
        
        if 'top_team_mae' in metrics:
            report_lines.extend([
                "ERROR BY TEAM CATEGORY:",
                "-" * 80,
                f"Top Teams MAE:                     {metrics['top_team_mae']:.2f} positions",
                f"Midfield Teams MAE:                {metrics['midfield_mae']:.2f} positions",
                "",
            ])
        
        if 'front_runners_mae' in metrics:
            report_lines.extend([
                "ERROR BY GRID POSITION:",
                "-" * 80,
                f"Front Runners (Grid ≤5) MAE:      {metrics['front_runners_mae']:.2f} positions",
                f"Back Markers (Grid ≥15) MAE:      {metrics.get('back_markers_mae', 0):.2f} positions",
                "",
            ])
        
        report_lines.extend([
            "BIGGEST PREDICTION ERRORS:",
            "-" * 80,
        ])
        
        for i, error in enumerate(metrics.get('biggest_errors', []), 1):
            report_lines.append(
                f"{i}. {error['Driver']} ({error['Team']}): "
                f"Predicted {error['Predicted_Race_Pos']:.1f}, "
                f"Actual {error['Actual_Race_Pos']:.0f}, "
                f"Error {error['Absolute_Error']:.1f} - {error['Status']}"
            )
        
        report_lines.extend([
            "",
            "BIGGEST RACE SURPRISES (Grid vs Finish):",
            "-" * 80,
        ])
        
        for i, surprise in enumerate(metrics.get('biggest_surprises', []), 1):
            report_lines.append(
                f"{i}. {surprise['Driver']} ({surprise['Team']}): "
                f"Grid P{surprise['Grid_Pos']:.0f} → Finish P{surprise['Actual_Race_Pos']:.0f} "
                f"({surprise['Grid_to_Finish']:+.0f}) - {surprise['Status']}"
            )
        
        report_lines.extend([
            "",
            "=" * 80,
            "DETAILED COMPARISON:",
            "-" * 80,
        ])
        
        comparison_df = merged.sort_values('Actual_Race_Pos')[
            ['Driver', 'Team', 'Predicted_Race_Pos', 'Actual_Race_Pos', 
             'Position_Error', 'Grid_Pos', 'Status']
        ].copy()
        
        report_lines.append(comparison_df.to_string(index=False))
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        report_path = os.path.join(
            output_dir, 
            f"validation_{year}_{race_name.replace(' ', '_')}_{timestamp}.txt"
        )
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Report saved to: {report_path}")
        
        try:
            print(report_text)
        except UnicodeEncodeError:
            print(report_text.encode('ascii', 'replace').decode('ascii'))
        
        self._generate_visualizations(merged, metrics, race_name, year, output_dir, timestamp)
        
        metrics_path = os.path.join(
            output_dir, 
            f"validation_metrics_{year}_{race_name.replace(' ', '_')}_{timestamp}.json"
        )
        with open(metrics_path, 'w', encoding='utf-8') as f:
            metrics_serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                   for k, v in metrics.items() 
                                   if not isinstance(v, (list, dict))}
            json.dump(metrics_serializable, f, indent=2)
        
        comparison_path = os.path.join(
            output_dir, 
            f"validation_comparison_{year}_{race_name.replace(' ', '_')}_{timestamp}.csv"
        )
        merged.to_csv(comparison_path, index=False)
        logger.info(f"Detailed comparison saved to: {comparison_path}")
    
    def _generate_visualizations(
        self, 
        merged: pd.DataFrame, 
        metrics: Dict, 
        race_name: str, 
        year: int,
        output_dir: str,
        timestamp: str
    ):
        """Generate visualization plots for validation."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{year} {race_name} - Prediction Validation', fontsize=16, fontweight='bold')
        
        ax = axes[0, 0]
        merged_sorted = merged.sort_values('Actual_Race_Pos')
        x = range(len(merged_sorted))
        ax.plot(x, merged_sorted['Predicted_Race_Pos'], 'o-', label='Predicted', linewidth=2, markersize=8)
        ax.plot(x, merged_sorted['Actual_Race_Pos'], 's-', label='Actual', linewidth=2, markersize=8)
        ax.set_xlabel('Driver (sorted by actual finish)', fontsize=11)
        ax.set_ylabel('Position', fontsize=11)
        ax.set_title('Predicted vs Actual Positions', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        
        ax = axes[0, 1]
        error_sorted = merged.sort_values('Absolute_Error', ascending=False)
        colors = ['red' if e > 5 else 'orange' if e > 3 else 'green' 
                 for e in error_sorted['Absolute_Error']]
        ax.barh(range(len(error_sorted)), error_sorted['Absolute_Error'], color=colors)
        ax.set_yticks(range(len(error_sorted)))
        ax.set_yticklabels(error_sorted['Driver'], fontsize=9)
        ax.set_xlabel('Absolute Error (positions)', fontsize=11)
        ax.set_title('Prediction Error by Driver', fontsize=12, fontweight='bold')
        ax.axvline(x=3, color='orange', linestyle='--', alpha=0.5, label='±3 positions')
        ax.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='±5 positions')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        ax = axes[1, 0]
        ax.scatter(merged['Predicted_Race_Pos'], merged['Actual_Race_Pos'], 
                  s=100, alpha=0.6, edgecolors='black')
        min_pos = min(merged['Predicted_Race_Pos'].min(), merged['Actual_Race_Pos'].min())
        max_pos = max(merged['Predicted_Race_Pos'].max(), merged['Actual_Race_Pos'].max())
        ax.plot([min_pos, max_pos], [min_pos, max_pos], 'r--', label='Perfect prediction', linewidth=2)
        ax.set_xlabel('Predicted Position', fontsize=11)
        ax.set_ylabel('Actual Position', fontsize=11)
        ax.set_title(f'Prediction Accuracy (r={metrics.get("spearman_correlation", 0):.3f})', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        error_counts = pd.cut(merged['Absolute_Error'], 
                             bins=[0, 1, 3, 5, 10, float('inf')],
                             labels=['≤1', '1-3', '3-5', '5-10', '>10']).value_counts().sort_index()
        colors_bar = ['green', 'lightgreen', 'orange', 'red', 'darkred']
        ax.bar(range(len(error_counts)), error_counts.values, color=colors_bar)
        ax.set_xticks(range(len(error_counts)))
        ax.set_xticklabels(error_counts.index)
        ax.set_xlabel('Error Range (positions)', fontsize=11)
        ax.set_ylabel('Number of Drivers', fontsize=11)
        ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(error_counts.values):
            ax.text(i, v + 0.3, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        plot_path = os.path.join(
            output_dir, 
            f"validation_plots_{year}_{race_name.replace(' ', '_')}_{timestamp}.png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to: {plot_path}")
        plt.close()
    
    def suggest_improvements(self, metrics: Dict, merged: pd.DataFrame) -> str:
        """Generate suggestions for model improvement based on validation results."""
        
        suggestions = [
            "\n" + "=" * 80,
            "MODEL IMPROVEMENT SUGGESTIONS",
            "=" * 80,
        ]
        
        mae = metrics.get('mae', 0)
        if mae > 5:
            suggestions.append(
                "\n⚠ HIGH MAE (>5 positions):\n"
                "  • Review feature engineering - may need more predictive features\n"
                "  • Consider adding more historical race data\n"
                "  • Check for data quality issues or outliers\n"
                "  • Increase model complexity or try different algorithms"
            )
        elif mae > 3:
            suggestions.append(
                "\n⚡ MODERATE MAE (3-5 positions):\n"
                "  • Fine-tune hyperparameters\n"
                "  • Add circuit-specific features\n"
                "  • Consider ensemble methods with different model types"
            )
        else:
            suggestions.append(
                "\n✓ GOOD MAE (<3 positions):\n"
                "  • Model performing well overall\n"
                "  • Focus on edge cases and outliers"
            )
        
        podium_acc = metrics.get('podium_accuracy', 0)
        if podium_acc < 0.5:
            suggestions.append(
                "\n⚠ LOW PODIUM ACCURACY:\n"
                "  • Add more features for top-tier driver/team performance\n"
                "  • Consider separate model for top 3 predictions\n"
                "  • Include qualifying pace and sector times\n"
                "  • Add race-specific strategy features"
            )
        
        if 'top_team_mae' in metrics and 'midfield_mae' in metrics:
            if metrics['top_team_mae'] > metrics['midfield_mae']:
                suggestions.append(
                    "\n⚠ TOP TEAMS HARDER TO PREDICT:\n"
                    "  • Add features capturing team development rate\n"
                    "  • Include upgrade timing and effectiveness\n"
                    "  • Consider driver-team synergy metrics"
                )
            else:
                suggestions.append(
                    "\n⚠ MIDFIELD HARDER TO PREDICT:\n"
                    "  • Midfield is inherently more volatile\n"
                    "  • Add reliability/consistency features\n"
                    "  • Include weather impact features\n"
                    "  • Consider incidents and safety car likelihood"
                )
        
        if 'biggest_errors' in metrics:
            dnf_errors = [e for e in metrics['biggest_errors'] 
                         if 'DNF' in str(e.get('Status', '')) or 'Retired' in str(e.get('Status', ''))]
            if dnf_errors:
                suggestions.append(
                    "\n⚠ DNF/RETIREMENT PREDICTION ISSUES:\n"
                    "  • Implement separate DNF prediction model\n"
                    "  • Add reliability features (team/driver history)\n"
                    "  • Include mechanical failure likelihood\n"
                    "  • Consider circuit-specific reliability factors"
                )
        
        correlation = metrics.get('spearman_correlation', 0)
        if correlation < 0.7:
            suggestions.append(
                "\n⚠ LOW RANK CORRELATION (<0.7):\n"
                "  • Model not capturing relative driver performance well\n"
                "  • Review feature importance and remove noise\n"
                "  • Use ranking loss functions instead of regression\n"
                "  • Add pairwise comparison features"
            )
        
        accuracy_3 = metrics.get('accuracy_within_3', 0)
        if accuracy_3 < 0.6:
            suggestions.append(
                "\n⚡ IMPROVE ±3 POSITION ACCURACY:\n"
                "  • Add more granular performance metrics\n"
                "  • Include recent form (last 3-5 races)\n"
                "  • Add track-specific driver performance\n"
                "  • Consider tire strategy factors"
            )
        
        suggestions.extend([
            "\nGENERAL RECOMMENDATIONS:",
            "  • Continuously update with latest race data",
            "  • Perform feature importance analysis",
            "  • Cross-validate across different seasons",
            "  • Monitor prediction confidence intervals",
            "  • A/B test new features before full deployment",
            "=" * 80
        ])
        
        return "\n".join(suggestions)


def main():
    """Main validation workflow."""
    
    validator = PredictionValidator()
    
    year = 2025
    race_name = "Singapore Grand Prix"
    prediction_file = "artifacts/predictions/race_predictions_2025_Singapore Grand Prix_20251009_195213.csv"
    
    print(f"\n{'='*80}")
    print(f"VALIDATING PREDICTIONS FOR {year} {race_name}")
    print(f"{'='*80}\n")
    
    predictions = validator.load_predictions(prediction_file)
    if predictions is None:
        return
    
    actual_results = validator.fetch_actual_race_results(year, race_name)
    if actual_results is None:
        return
    
    merged, metrics = validator.compare_predictions(predictions, actual_results)
    
    validator.generate_report(merged, metrics, race_name, year)
    
    improvements = validator.suggest_improvements(metrics, merged)
    print(improvements)
    
    improvement_file = f"validation/improvement_suggestions_{year}_{race_name.replace(' ', '_')}.txt"
    with open(improvement_file, 'w') as f:
        f.write(improvements)
    logger.info(f"Improvement suggestions saved to: {improvement_file}")


if __name__ == "__main__":
    main()

