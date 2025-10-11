"""
Model improvement script that analyzes validation results and implements improvements.
"""

import pandas as pd
import numpy as np
import json
import logging
import os
from typing import Dict, List, Tuple
from datetime import datetime

import sys
from pathlib import Path

# Ensure 'src' is importable when running from repo root
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from f1_predictor.config import config
from f1_predictor.data_loader import F1DataCollector
from f1_predictor.feature_engineering_pipeline import FeatureEngineeringPipeline
from f1_predictor.model_training import F1ModelTrainer
from f1_predictor.evaluation import F1ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelImprover:
    """Analyzes validation results and implements model improvements."""
    
    def __init__(self):
        self.data_collector = F1DataCollector()
        self.evaluator = F1ModelEvaluator()
        
    def analyze_validation_results(self, validation_dir: str = "validation") -> Dict:
        """Analyze all validation results to identify patterns."""
        
        logger.info("Analyzing validation results...")
        
        all_metrics = []
        for filename in os.listdir(validation_dir):
            if filename.startswith("validation_metrics_") and filename.endswith(".json"):
                with open(os.path.join(validation_dir, filename), 'r') as f:
                    metrics = json.load(f)
                    all_metrics.append(metrics)
        
        if not all_metrics:
            logger.warning("No validation results found")
            return {}
        
        analysis = {
            'total_validations': len(all_metrics),
            'avg_mae': np.mean([m.get('mae', 0) for m in all_metrics]),
            'avg_podium_accuracy': np.mean([m.get('podium_accuracy', 0) for m in all_metrics]),
            'avg_top5_accuracy': np.mean([m.get('top5_accuracy', 0) for m in all_metrics]),
            'avg_correlation': np.mean([m.get('spearman_correlation', 0) for m in all_metrics]),
            'worst_mae': max([m.get('mae', 0) for m in all_metrics]),
            'best_mae': min([m.get('mae', 0) for m in all_metrics]),
        }
        
        logger.info(f"Analysis complete: {analysis['total_validations']} races analyzed")
        logger.info(f"Average MAE: {analysis['avg_mae']:.2f} positions")
        logger.info(f"Average Podium Accuracy: {analysis['avg_podium_accuracy']*100:.1f}%")
        
        return analysis
    
    def identify_weak_areas(self, validation_comparison: pd.DataFrame) -> Dict:
        """Identify specific areas where model performs poorly."""
        
        weak_areas = {
            'high_error_drivers': {},
            'high_error_teams': {},
            'dnf_issues': {},
            'top_team_issues': {},
            'position_specific_issues': {}
        }
        
        high_error = validation_comparison[validation_comparison['Absolute_Error'] > 5]
        if not high_error.empty:
            driver_errors = high_error.groupby('Driver')['Absolute_Error'].agg(['mean', 'count'])
            driver_errors_filtered = driver_errors[driver_errors['count'] >= 2].sort_values('mean', ascending=False)
            if not driver_errors_filtered.empty:
                weak_areas['high_error_drivers'] = driver_errors_filtered.head(5).to_dict('index')
            
            team_errors = high_error.groupby('Team')['Absolute_Error'].agg(['mean', 'count'])
            team_errors_filtered = team_errors[team_errors['count'] >= 2].sort_values('mean', ascending=False)
            if not team_errors_filtered.empty:
                weak_areas['high_error_teams'] = team_errors_filtered.head(5).to_dict('index')
        
        dnf_mask = validation_comparison['Status'].str.contains('DNF|Retired', case=False, na=False)
        if dnf_mask.sum() > 0:
            weak_areas['dnf_issues'] = {
                'count': int(dnf_mask.sum()),
                'avg_error': float(validation_comparison[dnf_mask]['Absolute_Error'].mean())
            }
        
        top_teams = ['Mercedes', 'Red Bull Racing', 'Ferrari', 'McLaren']
        top_team_mask = validation_comparison['Team'].isin(top_teams)
        if top_team_mask.sum() > 0:
            weak_areas['top_team_issues'] = {
                'mae': float(validation_comparison[top_team_mask]['Absolute_Error'].mean()),
                'count': int(top_team_mask.sum())
            }
        
        for pos_range in [(1, 5), (6, 10), (11, 15), (16, 20)]:
            mask = (validation_comparison['Actual_Race_Pos'] >= pos_range[0]) & \
                   (validation_comparison['Actual_Race_Pos'] <= pos_range[1])
            if mask.sum() > 0:
                weak_areas['position_specific_issues'][f'P{pos_range[0]}-{pos_range[1]}'] = {
                    'mae': float(validation_comparison[mask]['Absolute_Error'].mean()),
                    'count': int(mask.sum())
                }
        
        return weak_areas
    
    def suggest_feature_improvements(self, weak_areas: Dict) -> List[str]:
        """Suggest specific feature engineering improvements."""
        
        suggestions = []
        
        if weak_areas.get('dnf_issues', {}).get('count', 0) > 0:
            suggestions.append(
                "DNF Prediction Features:\n"
                "  - Add driver_reliability_last_10_races\n"
                "  - Add team_dnf_rate_this_season\n"
                "  - Add circuit_specific_reliability\n"
                "  - Add power_unit_age_features"
            )
        
        if weak_areas.get('top_team_issues', {}).get('mae', 0) > 3:
            suggestions.append(
                "Top Team Performance Features:\n"
                "  - Add quali_race_pace_delta\n"
                "  - Add recent_upgrade_impact\n"
                "  - Add driver_teammate_gap\n"
                "  - Add championship_pressure_index"
            )
        
        if weak_areas.get('high_error_drivers'):
            suggestions.append(
                "Driver-Specific Features:\n"
                "  - Add driver_circuit_history_detailed\n"
                "  - Add driver_weather_performance\n"
                "  - Add driver_overtaking_skill\n"
                "  - Add driver_race_start_performance"
            )
        
        pos_issues = weak_areas.get('position_specific_issues', {})
        if pos_issues.get('P1-5', {}).get('mae', 0) > 2:
            suggestions.append(
                "Front-Runner Features:\n"
                "  - Add quali_to_race_conversion_rate\n"
                "  - Add pole_position_advantage\n"
                "  - Add clean_air_pace_advantage\n"
                "  - Add safety_car_strategy_impact"
            )
        
        if pos_issues.get('P11-20', {}).get('mae', 0) > 5:
            suggestions.append(
                "Midfield/Back-Marker Features:\n"
                "  - Add midfield_volatility_index\n"
                "  - Add traffic_management_skill\n"
                "  - Add alternative_strategy_likelihood\n"
                "  - Add incident_involvement_probability"
            )
        
        return suggestions
    
    def implement_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Implement additional feature engineering based on analysis."""
        
        logger.info("Implementing additional features...")
        
        df_enhanced = df.copy()
        
        if 'Status' in df_enhanced.columns:
            dnf_keywords = ['DNF', 'Retired', 'Accident', 'Collision', 'Mechanical']
            df_enhanced['is_dnf'] = df_enhanced['Status'].apply(
                lambda x: 1 if any(kw in str(x) for kw in dnf_keywords) else 0
            )
            
            driver_dnf_rate = df_enhanced.groupby('Driver')['is_dnf'].rolling(
                window=10, min_periods=3
            ).mean().reset_index(level=0, drop=True)
            df_enhanced['driver_dnf_rate_last_10'] = driver_dnf_rate
        
        if {'Driver', 'Position', 'GridPosition'}.issubset(df_enhanced.columns):
            df_enhanced['positions_gained'] = df_enhanced['GridPosition'] - df_enhanced['Position']
            
            driver_overtaking = df_enhanced.groupby('Driver')['positions_gained'].rolling(
                window=5, min_periods=2
            ).mean().reset_index(level=0, drop=True)
            df_enhanced['driver_overtaking_avg'] = driver_overtaking
        
        if {'Position', 'GridPosition'}.issubset(df_enhanced.columns):
            df_enhanced['started_top5'] = (df_enhanced['GridPosition'] <= 5).astype(int)
            df_enhanced['finished_top5'] = (df_enhanced['Position'] <= 5).astype(int)
            df_enhanced['top5_conversion'] = (
                df_enhanced['started_top5'] & df_enhanced['finished_top5']
            ).astype(int)
        
        if 'Team' in df_enhanced.columns and 'Position' in df_enhanced.columns:
            team_consistency = df_enhanced.groupby('Team')['Position'].rolling(
                window=5, min_periods=2
            ).std().reset_index(level=0, drop=True)
            df_enhanced['team_consistency_std'] = team_consistency
        
        logger.info(f"Added {len(df_enhanced.columns) - len(df.columns)} new features")
        
        return df_enhanced
    
    def retrain_with_improvements(
        self, 
        races_df: pd.DataFrame,
        quali_df: pd.DataFrame,
        model_type: str = "race"
    ) -> Tuple[object, Dict]:
        """Retrain model with improvements."""
        
        logger.info(f"Retraining {model_type} model with improvements...")
        
        enhanced_races = self.implement_feature_engineering(races_df)
        
        feature_engineer = FeatureEngineeringPipeline(
            hist_races=enhanced_races,
            hist_quali=quali_df if model_type == "race" else pd.DataFrame()
        )
        
        features_df = feature_engineer.engineer_features(
            enhanced_races, 
            quali_df if model_type == "race" else None
        )
        
        trainer = F1ModelTrainer()
        
        target_col = 'Position' if model_type == "race" else 'grid_position'
        
        model, metrics = trainer.train_model(
            features_df,
            target_column=target_col,
            model_type=model_type
        )
        
        logger.info(f"Retraining complete. MAE: {metrics.get('mae', 0):.2f}")
        
        return model, metrics
    
    def compare_model_versions(
        self, 
        old_metrics: Dict, 
        new_metrics: Dict
    ) -> str:
        """Compare old and new model performance."""
        
        comparison = [
            "\n" + "=" * 80,
            "MODEL COMPARISON",
            "=" * 80,
        ]
        
        metrics_to_compare = ['mae', 'rmse', 'r2', 'top3_accuracy', 'top5_accuracy']
        
        for metric in metrics_to_compare:
            old_val = old_metrics.get(metric, 0)
            new_val = new_metrics.get(metric, 0)
            
            if metric in ['mae', 'rmse']:
                improvement = old_val - new_val
                pct = (improvement / old_val * 100) if old_val != 0 else 0
                direction = "↓ IMPROVED" if improvement > 0 else "↑ WORSE"
                comparison.append(
                    f"{metric.upper():20} {old_val:8.3f} → {new_val:8.3f}  "
                    f"({improvement:+.3f}, {pct:+.1f}%) {direction}"
                )
            else:
                improvement = new_val - old_val
                pct = (improvement / old_val * 100) if old_val != 0 else 0
                direction = "↑ IMPROVED" if improvement > 0 else "↓ WORSE"
                comparison.append(
                    f"{metric.upper():20} {old_val:8.3f} → {new_val:8.3f}  "
                    f"({improvement:+.3f}, {pct:+.1f}%) {direction}"
                )
        
        comparison.append("=" * 80)
        
        return "\n".join(comparison)
    
    def generate_improvement_report(
        self, 
        analysis: Dict,
        weak_areas: Dict,
        feature_suggestions: List[str],
        output_path: str = "validation/improvement_plan.txt"
    ):
        """Generate comprehensive improvement plan."""
        
        report = [
            "=" * 80,
            "MODEL IMPROVEMENT PLAN",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
            "CURRENT PERFORMANCE SUMMARY:",
            "-" * 80,
            f"Total Races Analyzed:     {analysis.get('total_validations', 0)}",
            f"Average MAE:               {analysis.get('avg_mae', 0):.2f} positions",
            f"Average Podium Accuracy:   {analysis.get('avg_podium_accuracy', 0)*100:.1f}%",
            f"Average Top 5 Accuracy:    {analysis.get('avg_top5_accuracy', 0)*100:.1f}%",
            f"Average Correlation:       {analysis.get('avg_correlation', 0):.3f}",
            f"Best MAE:                  {analysis.get('best_mae', 0):.2f}",
            f"Worst MAE:                 {analysis.get('worst_mae', 0):.2f}",
            "",
            "IDENTIFIED WEAK AREAS:",
            "-" * 80,
        ]
        
        if weak_areas.get('dnf_issues'):
            report.append(f"\nDNF Prediction Issues:")
            report.append(f"  - Count: {weak_areas['dnf_issues']['count']}")
            report.append(f"  - Avg Error: {weak_areas['dnf_issues']['avg_error']:.2f} positions")
        
        if weak_areas.get('high_error_drivers'):
            report.append(f"\nHigh Error Drivers:")
            for driver, stats in list(weak_areas['high_error_drivers'].items())[:5]:
                report.append(f"  - {driver}: {stats['mean']:.2f} avg error ({stats['count']} races)")
        
        if weak_areas.get('high_error_teams'):
            report.append(f"\nHigh Error Teams:")
            for team, stats in list(weak_areas['high_error_teams'].items())[:5]:
                report.append(f"  - {team}: {stats['mean']:.2f} avg error ({stats['count']} races)")
        
        report.extend([
            "",
            "RECOMMENDED FEATURE IMPROVEMENTS:",
            "-" * 80,
        ])
        
        for suggestion in feature_suggestions:
            report.append(f"\n{suggestion}")
        
        report.extend([
            "",
            "IMPLEMENTATION PRIORITY:",
            "-" * 80,
            "1. HIGH: Address DNF prediction issues",
            "2. HIGH: Improve top team performance prediction",
            "3. MEDIUM: Add driver-specific historical features",
            "4. MEDIUM: Enhance circuit-specific features",
            "5. LOW: Fine-tune hyperparameters",
            "",
            "=" * 80
        ])
        
        report_text = "\n".join(report)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Improvement plan saved to: {output_path}")
        try:
            print(report_text)
        except UnicodeEncodeError:
            print(report_text.encode('ascii', 'replace').decode('ascii'))


def main():
    """Main improvement workflow."""
    
    improver = ModelImprover()
    
    print("\n" + "="*80)
    print("F1 MODEL IMPROVEMENT ANALYSIS")
    print("="*80 + "\n")
    
    analysis = improver.analyze_validation_results()
    
    if not analysis:
        logger.error("No validation results found. Run validate_predictions.py first.")
        return
    
    validation_files = [f for f in os.listdir("validation") 
                       if f.startswith("validation_comparison_") and f.endswith(".csv")]
    
    if not validation_files:
        logger.error("No validation comparison files found.")
        return
    
    latest_validation = sorted(validation_files)[-1]
    validation_df = pd.read_csv(os.path.join("validation", latest_validation))
    
    weak_areas = improver.identify_weak_areas(validation_df)
    
    feature_suggestions = improver.suggest_feature_improvements(weak_areas)
    
    improver.generate_improvement_report(
        analysis, 
        weak_areas, 
        feature_suggestions
    )
    
    print("\n" + "="*80)
    print("WEAK AREAS IDENTIFIED:")
    print("="*80)
    print(json.dumps(weak_areas, indent=2, default=str))


if __name__ == "__main__":
    main()

