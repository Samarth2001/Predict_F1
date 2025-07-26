# F1 Prediction System

A production-grade Formula 1 prediction system designed for accuracy and ease of use, featuring a modular architecture, advanced machine learning, and a streamlined workflow.

## Usage

Here is the complete workflow for fetching data, training models, and making predictions.

### Step 1: Fetch Data

Download the latest F1 data, including race results, qualifying, and weather information. The system is smart and will only fetch new or missing data.

```bash
python predict.py fetch-data
```
To force a complete refresh of all data (recommended for the first run), use the `--force` flag:
```bash
python predict.py fetch-data --force
```

### Step 2: Train Models

Train the qualifying and race prediction models using the latest data. This step only needs to be run after you have fetched new historical data.

```bash
python predict.py train
```

### Step 3: Make Predictions

Generate predictions for any race weekend session. The system will automatically use the latest qualifying data if it's available for race predictions.

```bash
# Example: Predict qualifying results for the Italian Grand Prix
python predict.py predict --year 2024 --race "Italian Grand Prix" --session qualifying

# Example: Predict race results for the Italian Grand Prix
python predict.py predict --year 2024 --race "Italian Grand Prix" --session race
```

---

## Key Features

- **Modular Workflow** - Separate, independent commands for fetching data, training, and predicting.
- **Intelligent Predictions** - Automatically uses post-qualifying data for race predictions when available.
- **Advanced Feature Engineering** - Creates over 100 F1-specific features, including weather and strategy factors.
- **Ensemble Models** - Combines LightGBM and XGBoost for robust and accurate predictions.
- **Configuration-Driven** - All settings are managed in a central `config/default_config.yaml` file.

## Project Structure

```
f1_prediction_project/
├── predict.py                    #  Main entry point for all commands
├── config/
│   └── default_config.yaml       # Central configuration file
├── f1_predictor/                 # Core prediction package
│   ├── data_loader.py            # Data collection and loading
│   ├── feature_engineering_pipeline.py # Feature engineering
│   ├── model_training.py         # Model training logic
│   ├── prediction.py             # Prediction engine
│   └── ...
├── models/                       # Trained model artifacts (.pkl)
├── predictions/                  # Prediction outputs (.csv)
├── logs/                         # System logs
└── f1_data_real/                 # Raw F1 data and cache
```

## Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd f1_prediction_project

# 2. Create a virtual environment and install dependencies
python -m venv .venv
# On Windows: .\.venv\Scripts\Activate.ps1
# On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

The entire system is controlled via `config/default_config.yaml`. Here you can adjust:
- Data collection start year
- Model hyperparameters
- Feature engineering settings (e.g., rolling window sizes)
- And much more. 