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

The CLI is designed for ergonomics and reproducibility.

```bash
# Qualifying or Race (single event)
python predict.py predict --year 2024 --race "Italian Grand Prix" --session qualifying
python predict.py predict --year 2024 --race "Italian Grand Prix" --session race

# Scenarios for race predictions via --mode
#   - pre-weekend: no qualifying used
#   - pre-quali  : use predicted qualifying only
#   - post-quali : require actual qualifying
#   - live       : auto-refresh until actual qualifying appears (then exit)
#   - auto       : use actual when available, else predicted (default)
python predict.py predict --year 2025 --race "Bahrain Grand Prix" --session race --mode auto
python predict.py predict --year 2025 --race "Bahrain Grand Prix" --session race --mode pre-weekend
python predict.py predict --year 2025 --race "Bahrain Grand Prix" --session race --mode post-quali

# Next event shortcut and current season default
python predict.py predict --race next --session race --mode auto --season-current

# Batch runs (entire season or specific rounds)
python predict.py predict --year 2024 --season all --session race
python predict.py predict --year 2024 --session race --rounds 1-10
python predict.py predict --year 2024 --session race --rounds 1,3,5-7

# Watch mode (auto-refresh every N seconds until quali is published)
python predict.py predict watch --year 2025 --race "Italian Grand Prix" --interval 300

# Live mode via --mode (single event)
python predict.py predict --year 2025 --race "Italian Grand Prix" --session race --mode live --interval 300

# Simulations (future seasons / what-ifs)
# Provide an optional custom lineup CSV with at least columns: Driver, Team
python predict.py predict simulate --year 2026 --race "Italian Grand Prix" --session race --lineup path/to/lineup.csv
```

---

## Training strategies and scenarios

### Train separately vs train all

- Qualifying only
  - Use when you are tuning qualifying rankers (faster iteration) or you changed quali-specific config.
  - Command:

    ```bash
    python predict.py train --model-type qualifying
    ```

- Race only
  - Use when you are iterating on race model logic (e.g., DNF integration, race features) or changed race config.
  - Command:

    ```bash
    python predict.py train --model-type race
    ```

- All (qualifying + race)
  - Use after significant changes to features/config, after fetching a lot of new data, or before a new season.
  - Command:

    ```bash
    python predict.py train  # same as --model-type all
    ```

When to retrain:

- After `fetch-data` pulls new seasons/rounds you want reflected in training
- After changing feature engineering or config defaults (e.g., ensemble metrics, ranker overrides)
- After upgrading dependencies or model versions

Optional hyperparameter tuning and ensembles:

- In `config/default_config.yaml`:
  - `hyperparameter_optimization.enabled: true`
  - `hyperparameter_optimization.objective_metric: spearman`  # good for ranking
  - `hyperparameter_optimization.n_trials: 100` (increase if time allows)
  - `models.ensemble_config.target_metric: spearman` (or `ndcg`)

Notes:

- HPO runs for non-ranker models by default. To benefit during qualifying, consider adding non-ranker base models or manually override ranker params under `models.qualifying.overrides`.

### Prediction scenarios (when to use which)

- `pre-weekend` (race)
  - No actual quali available. Use to get early race outlook from historical/context.

    ```bash
    python predict.py predict --year 2025 --race "Bahrain Grand Prix" --session race --mode pre-weekend
    ```

- `pre-quali` (race)
  - Right before qualifying; still no actual quali. Similar to pre-weekend but closer in time.

    ```bash
    python predict.py predict --year 2025 --race "Bahrain Grand Prix" --session race --mode pre-quali
    ```

- `post-quali` (race)
  - After qualifying; uses actual quali if available to refine race positions.

    ```bash
    python predict.py predict --year 2025 --race "Bahrain Grand Prix" --session race --mode post-quali
    ```

- `live`
  - Auto-refreshes predictions periodically until actual qualifying is detected, then exits.

    ```bash
    python predict.py predict --year 2025 --race "Bahrain Grand Prix" --session race --mode live --interval 300
    ```

- `auto` (default)
  - Uses actual qualifying when present; otherwise falls back to predicted quali.

    ```bash
    python predict.py predict --year 2025 --race "Bahrain Grand Prix" --session race --mode auto
    ```

### Recommended command sequences (chronological)

- Fresh setup (or start of a season):

  ```bash
  python predict.py fetch-data --force
  python predict.py train               # trains qualifying + race
  python predict.py predict list-schedule --year 2025  # copy exact EventName
  python predict.py predict --year 2025 --race "Bahrain Grand Prix" --session race --mode auto
  ```

- Race week – pre-quali planning:

  ```bash
  python predict.py fetch-data
  # retrain only if you changed features/config or fetched a lot of new historical data
  # python predict.py train --model-type qualifying
  python predict.py predict --year 2025 --race "Bahrain Grand Prix" --session qualifying
  python predict.py predict --year 2025 --race "Bahrain Grand Prix" --session race --mode pre-quali
  ```

- After qualifying – finalize race outlook:

  ```bash
  python predict.py predict --year 2025 --race "Bahrain Grand Prix" --session race --mode post-quali
  ```

- Model tuning loop (faster iterations):

  ```bash
  # Edit config/default_config.yaml
  #  - Set hyperparameter_optimization.enabled: true (optional)
  #  - Set objective_metric / ensemble target_metric to spearman or ndcg
  #  - Override ranker params under models.<target>.overrides as needed
  python predict.py train --model-type qualifying
  python predict.py predict --year 2025 --race "Bahrain Grand Prix" --session qualifying
  # Repeat for race when satisfied
  python predict.py train --model-type race
  python predict.py predict --year 2025 --race "Bahrain Grand Prix" --session race --mode auto
  ```

Utilities:

- List official event names for a season:

  ```bash
  python predict.py predict list-schedule --year 2025
  ```

- Simulate future seasons or custom lineups:

  ```bash
  python predict.py predict simulate --year 2026 --race "Italian Grand Prix" --session race --lineup path/to/lineup.csv
  ```

## Key Features

- **Modular Workflow** - Separate, independent commands for fetching data, training, and predicting.
- **Prediction Scenarios** - `--mode` supports pre-weekend, pre-quali, post-quali, live, and auto.
- **Next Event Shortcuts** - Use `--race next` and `--season-current` to target the upcoming race this year.
- **Batch Runs** - Run entire seasons or ranges of rounds in a single command.
- **Watch Mode** - `predict watch` auto-refreshes until actual qualifying appears.
- **Simulations** - Future-season what-ifs with `predict simulate` and custom lineups.
- **Intelligent Predictions** - Automatically uses post-qualifying data (auto mode) when available.
- **Advanced Feature Engineering** - Creates over 100 F1-specific features, including weather and strategy factors.
- **Ensemble Models** - Combines LightGBM and XGBoost (including ranking variants) for robust predictions.
- **Configuration-Driven** - All settings are managed in a central `config/default_config.yaml` file.

## Project Structure

```text
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
git clone https://github.com/yourname/f1_prediction_project.git
cd f1_prediction_project

# 2. Create a virtual environment and install dependencies
python -m venv .venv
# On Windows: .\.venv\Scripts\Activate.ps1
# On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Configuration

The entire system is controlled via `config/default_config.yaml` (single source of truth). You can provide a gitignored `config/local.yaml` for environment-specific overrides and secrets. Values support environment variable interpolation using `${VAR}` or `${VAR:-default}`.

Examples:

```yaml
data_collection:
  external_apis:
    openweathermap_api_key: "${OPENWEATHER_API_KEY:-}"
```

Create `config/local.yaml` to override any setting locally (do not commit):

```yaml
data_collection:
  external_apis:
    openweathermap_api_key: "${OPENWEATHER_API_KEY}"
```

Strict feature enforcement at inference is enabled by default via `general.strict_feature_enforcement: true`. Models persist their training feature list, imputation values, config hash, `model_version`, and git commit in `models/*_model.metadata.json`.

### Prediction Artifacts & Metadata

Every prediction writes two files into `predictions/`:

- `<type>_predictions_<year>_<race>_<timestamp>.csv`
- `<type>_predictions_<year>_<race>_<timestamp>.meta.json`

The `.meta.json` includes:

- `model_type`, `year`, `race_name`, `generated_at`
- `model_version` (from `config.default_config.yaml`)
- `scenario` (e.g., `pre-weekend`, `pre-quali`, `post-quali`, `live`, `auto`, or `simulate`)
- `features_used` (training-time feature list, when known)
- `config_hash` (stable hash of the merged configuration for full traceability)
