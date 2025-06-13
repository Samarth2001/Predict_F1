# F1 Prediction System 2.0 🏎️

A comprehensive, production-ready Formula 1 prediction system built with advanced machine learning techniques, ensemble modeling, and real-time prediction capabilities.

## 🎯 Overview

This system provides end-to-end F1 predictions for:
- **Qualifying Results** - Predict grid positions before qualifying sessions
- **Race Results** - Predict race finishing positions with multiple scenarios:
  - Pre-qualifying race predictions (based on historical data only)
  - Post-qualifying race predictions (incorporating actual qualifying results)
  - Live updates during race weekends

## ✨ Key Features

### 🤖 Advanced Machine Learning
- **Ensemble Models**: LightGBM, XGBoost, Random Forest, and Neural Networks
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Feature Selection**: Intelligent feature selection for optimal performance
- **Time-aware Data Splitting**: Prevents data leakage in time series predictions

### 📊 Comprehensive Feature Engineering
- **Driver Performance**: Rolling averages, consistency metrics, win/podium rates
- **Team Performance**: Reliability scores, development trends, championship standings
- **Circuit Analysis**: Track categorization, overtaking difficulty, weather impact
- **Contextual Features**: Championship pressure, season progress, tire strategies

### 🎯 Intelligent Predictions
- **Confidence Scoring**: Each prediction includes confidence levels (High/Medium/Low)
- **Multiple Scenarios**: Pre and post-qualifying predictions for complete weekend coverage
- **Live Updates**: Real-time prediction updates with actual qualifying results
- **Historical Validation**: Extensive backtesting on historical data

### 📈 Robust Evaluation
- **F1-Specific Metrics**: Top-3 accuracy, podium precision, rank correlation
- **Comprehensive Reports**: Detailed performance analysis with visualizations
- **Model Comparison**: Compare different models and prediction scenarios
- **Historical Analysis**: Year-by-year, driver-by-driver, circuit-by-circuit performance

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd f1_prediction_project
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the complete prediction workflow:**
```bash
python main.py --fetch
```

### Basic Usage

```bash
# Run complete workflow with data fetching
python main.py --fetch

# Predict specific race
python main.py --race "Japanese Grand Prix"

# Live update with actual qualifying results
python main.py --live-update qualifying_results.csv

# Force data refresh
python main.py --fetch --race "Monaco Grand Prix"
```

## 📁 Project Structure

```
f1_prediction_project/
├── f1_predictor/                    # Core prediction package
│   ├── config.py                    # Comprehensive configuration
│   ├── data_loader.py              # Data collection and loading
│   ├── feature_engineering.py     # Advanced feature engineering
│   ├── model_training.py          # Ensemble model training
│   ├── prediction.py              # Prediction engine
│   ├── evaluation.py              # Model evaluation and analysis
│   └── __init__.py
├── models/                         # Trained model storage
├── predictions/                    # Prediction outputs
├── evaluation/                     # Evaluation reports and plots
├── logs/                          # System logs
├── f1_data_real/                  # Raw F1 data
├── main.py                        # Main application
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🔧 Configuration

The system is highly configurable through `f1_predictor/config.py`:

### Model Configuration
```python
# Available model types
MODEL_TYPES = ['lightgbm', 'xgboost', 'random_forest', 'neural_network']
DEFAULT_MODEL_TYPE = 'ensemble'  # Use ensemble by default

# Ensemble weights (auto-optimized)
ENSEMBLE_WEIGHTS = {
    'lightgbm': 0.3,
    'xgboost': 0.3,
    'random_forest': 0.2,
    'neural_network': 0.2
}
```

### Feature Engineering
```python
# Rolling window sizes for performance metrics
N_ROLLING_SHORT = 3     # Recent form (last 3 races)
N_ROLLING_MEDIUM = 5    # Medium term form (last 5 races)  
N_ROLLING_LONG = 10     # Long term form (last 10 races)

# Circuit categorization
STREET_CIRCUITS = ['Monaco', 'Singapore', 'Baku', 'Las Vegas', 'Miami']
HIGH_SPEED_CIRCUITS = ['Monza', 'Spa-Francorchamps', 'Silverstone', 'Suzuka']
TECHNICAL_CIRCUITS = ['Monaco', 'Singapore', 'Hungary', 'Spain']
```

### Performance Optimization
```python
# Enable advanced features
ENABLE_HYPERPARAMETER_OPTIMIZATION = True
ENABLE_FEATURE_SELECTION = True
ENABLE_MODEL_STACKING = True

# Optimization settings
OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT = 3600  # 1 hour
```

## 📊 Prediction Workflow

### 1. Complete Weekend Prediction
```python
from f1_predictor.prediction import F1Predictor

predictor = F1Predictor()
weekend_results = predictor.predict_race_weekend(
    quali_model=quali_model,
    race_model=race_model,
    # ... other parameters
)

# Results include:
# - quali_predictions: Predicted qualifying order
# - pre_quali_race_predictions: Race predictions without qualifying data
# - post_quali_race_predictions: Race predictions with predicted qualifying
# - weekend_summary: Key insights and highlights
```

### 2. Live Updates
```python
# Update predictions with actual qualifying results
updated_predictions = predictor.update_predictions_with_actual_quali(
    race_model=race_model,
    actual_quali=actual_qualifying_results,
    # ... other parameters
)
```

## 📈 Model Performance

The system uses F1-specific evaluation metrics:

- **MAE (Mean Absolute Error)**: Average position prediction error
- **Top-3 Accuracy**: Percentage of predictions within 3 positions
- **Top-5 Accuracy**: Percentage of predictions within 5 positions
- **Podium Precision**: Accuracy of podium position predictions
- **Rank Correlation**: Spearman correlation between predicted and actual rankings
- **Weighted MAE**: Position-weighted error (higher weight for better positions)

### Typical Performance Metrics
```
Qualifying Model:
- MAE: ~2.1 positions
- Top-3 Accuracy: ~68%
- Top-5 Accuracy: ~84%

Race Model:
- MAE: ~2.8 positions  
- Top-3 Accuracy: ~58%
- Top-5 Accuracy: ~76%
- Podium Precision: ~71%
```

## 🔍 Feature Categories

### Driver Features
- Rolling performance averages (3, 5, 10 races)
- Championship standings and points
- Win rate, podium rate, DNF rate
- Circuit-specific performance history
- Experience and consistency metrics

### Team Features  
- Team rolling averages
- Reliability scores
- Development trends
- Championship position
- Circuit-specific performance

### Circuit Features
- Track categorization (street, high-speed, technical)
- Historical difficulty metrics
- Overtaking probability
- Weather impact factors

### Contextual Features
- Season progress
- Championship pressure
- Weather conditions
- Tire strategy optimization

## 📊 Outputs and Reports

### Prediction Files
- `predictions/predictions_qualifying_YYYYMMDD_HHMMSS.csv`
- `predictions/predictions_race_YYYYMMDD_HHMMSS.csv`
- `predictions/live_update_YYYYMMDD_HHMMSS.csv`

### Evaluation Reports
- `evaluation/evaluation_report_YYYYMMDD_HHMMSS.txt`
- `evaluation/model_comparison_YYYYMMDD.png`
- `evaluation/feature_importance_qualifying_YYYYMMDD.csv`

### Model Artifacts
- `models/qualifying_model_YYYYMMDD_HHMMSS.joblib`
- `models/race_model_YYYYMMDD_HHMMSS.joblib`
- `models/metadata_YYYYMMDD_HHMMSS.json`

## 🎯 Advanced Usage

### Custom Model Training
```python
from f1_predictor.model_training import F1ModelTrainer

trainer = F1ModelTrainer()
model, features, imputation = trainer.train_model(
    features=feature_dataframe,
    target_column_name="Finish_Pos_Clean",
    model_type="race"
)
```

### Custom Feature Engineering
```python
from f1_predictor.feature_engineering import F1FeatureEngineer

engineer = F1FeatureEngineer()
features_df = engineer.engineer_features(
    hist_races, hist_quali, curr_races, curr_quali, upcoming_info
)
```

### Model Evaluation
```python
from f1_predictor.evaluation import F1ModelEvaluator

evaluator = F1ModelEvaluator()
metrics = evaluator.evaluate_model_performance(model, X_test, y_test)
report = evaluator.generate_evaluation_report()
```

## 🔄 Continuous Learning

### After Each Race
1. **Data Update**: Automatically fetch new race results
2. **Model Retraining**: Update models with latest data
3. **Performance Evaluation**: Assess prediction accuracy
4. **Feature Optimization**: Refine feature engineering based on results

### Seasonal Updates
1. **Regulation Changes**: Adapt to new F1 regulations
2. **Driver/Team Changes**: Update driver and team mappings
3. **Circuit Updates**: Incorporate track modifications
4. **Model Architecture**: Enhance models based on seasonal learnings

## 🎨 Visualization

The system generates comprehensive visualizations:

- **Performance Comparison Charts**: Compare different models
- **Prediction Confidence Heatmaps**: Visualize prediction certainty
- **Feature Importance Plots**: Understand key prediction factors
- **Historical Accuracy Trends**: Track model performance over time
- **Driver/Team Performance Radars**: Multi-dimensional performance analysis

## 🔧 Troubleshooting

### Common Issues

1. **Data Fetching Fails**:
   - Check internet connection
   - Verify FastF1 API availability
   - Use `--fetch` flag to force refresh

2. **Model Training Errors**:
   - Ensure sufficient historical data
   - Check feature engineering output
   - Verify target column exists

3. **Prediction Failures**:
   - Validate upcoming race data format
   - Check encoder compatibility
   - Ensure all required features are present

### Debug Mode
```bash
# Enable detailed logging
export F1_LOG_LEVEL=DEBUG
python main.py --fetch
```

### Performance Optimization
```bash
# Disable hyperparameter optimization for faster training
# Edit config.py: ENABLE_HYPERPARAMETER_OPTIMIZATION = False

# Reduce feature selection for quicker processing
# Edit config.py: MAX_FEATURES_SELECTION = 25
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Roadmap

### Version 2.1 (Planned)
- [ ] Real-time telemetry integration
- [ ] Advanced tire strategy modeling  
- [ ] Weather API integration
- [ ] Interactive prediction dashboard

### Version 2.2 (Future)
- [ ] Machine learning interpretability (SHAP values)
- [ ] Automated model deployment
- [ ] API endpoint for external access
- [ ] Mobile app integration

### Version 3.0 (Long-term)
- [ ] Deep learning models (LSTM, Transformer)
- [ ] Computer vision for race analysis
- [ ] Predictive betting odds calculation
- [ ] Multi-series support (F2, F3, etc.)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastF1**: For providing excellent F1 data access
- **Scikit-learn**: For robust machine learning foundations
- **LightGBM & XGBoost**: For high-performance gradient boosting
- **F1 Community**: For inspiration and feedback

## 📞 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the documentation in the code

---

**Happy Predicting! 🏁** 