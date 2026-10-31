# Project Architecture

## Purpose

Build a reproducible daily BTC-USD next-close forecasting pipeline for ARIMA, XGBoost, and LSTM. Keep exploratory notebooks available, while moving reusable data and modeling logic into importable Python modules. The target layout below is a proposal; the current implementation snapshot and file mapping distinguish completed work from remaining work.

## Current implementation snapshot

- The FastAPI backend serves persistence and ARIMA forecasts from the configured local OHLCV CSV. It exposes process health, dataset readiness, and model status endpoints. It does not fetch current prices or serve XGBoost/LSTM forecasts.
- `models/arima.py` provides a fixed-order ARIMA forecast adapter. The API reads its order from the existing `arima/arima_metrics.json` file and fits it to the configured close-price history for each request.
- `models/xgboost_model.py` provides a fit/predict adapter with target-column and feature-schema checks; the experiment runner uses it, but the API does not serve saved XGBoost models yet.
- `evaluation.py` compares named prediction series with shared metrics and requires identical chronological dates. Generic rolling-origin orchestration is not implemented; the current script handles the ARIMA rolling state explicitly.
- `scripts/run_experiment.py` runs an initial ARIMA/XGBoost/persistence comparison. It selects XGBoost rounds on validation data, selects ARIMA order by training-period AIC, and evaluates both models on target-date-aligned test rows.
- `artifacts.py` writes predictions, metrics, validation scores, and run metadata to a new run directory. `models/lstm.py` remains unimplemented, and the XGBoost and LSTM notebooks remain empty.

## Target layout

```text
Crypto/
├── README.md
├── ARCHITECTURE.md
├── pyproject.toml                 # package metadata and bounded runtime dependencies
├── configs/
│   └── default.yaml               # dates, target, feature groups, model settings
├── data/
│   ├── raw/                       # immutable downloaded OHLCV snapshots
│   └── processed/                 # canonical feature/target dataset
├── notebooks/
│   ├── preprocessing_exploration.ipynb
│   └── model_analysis.ipynb       # optional reports using saved run artifacts
├── src/
│   └── btc_forecasting/
│       ├── __init__.py
│       ├── config.py              # load and validate configuration
│       ├── data.py                # download/load, schema and date-gap checks
│       ├── features.py            # causal indicators, lags and next-day target
│       ├── splitting.py            # chronological train/validation/test windows
│       ├── metrics.py             # shared regression and direction metrics
│       ├── evaluation.py          # common rolling-origin evaluation interface
│       ├── artifacts.py           # save predictions, metrics and run metadata
│       └── models/
│           ├── __init__.py
│           ├── arima.py
│           ├── xgboost.py
│           └── lstm.py
├── scripts/
│   ├── prepare_data.py
│   └── run_experiment.py
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_splitting.py
│   └── test_no_leakage.py
├── artifacts/                     # generated run outputs; exclude large files from Git
│   ├── metrics/
│   ├── predictions/
│   └── figures/
└── legacy/                         # optional holding place during a later migration only
```

Do not move the existing files as part of this architecture decision. Later migration can copy or refactor them in small, reviewable steps; retain the existing notebooks and outputs until replacements are checked.

## Responsibilities and design decisions

- **Keep notebooks for exploration and explanation.** The existing preprocessing and ARIMA notebooks contain useful work. Core feature and model logic should eventually live in `src/` so notebooks and command-line runs share one implementation.
- **Separate raw data from derived data.** Preserve the downloaded OHLCV snapshot under `data/raw/`; store the engineered table under `data/processed/`. This resolves the current ambiguity where both CSV names contain the same processed data and there is no raw snapshot in the exports.
- **Use one configuration and one run entry point.** Dates, target, feature choices, and model settings belong in `configs/default.yaml`. The script should record the effective configuration with each run so results can be reproduced.
- **Share time handling and evaluation.** All models should use the same chronological splits, prediction dates, forecast horizon, metrics, and baseline. Model-specific sequence construction or ARIMA state updates stay inside their model adapters.
- **Keep learned preprocessing inside the split boundary.** Any scaler or other fitted transform must be fit using training data only. Feature columns must exclude every target column. Unit tests should cover these boundaries before relying on scores.
- **Save predictions as well as summary scores.** A per-date prediction file allows direct comparison and audit of missing dates or model differences. Store run date, source data range, split cutoffs, package versions, configuration, and random seeds alongside scores.
- **Delay the app and deployment.** Build an application only after a documented command can recreate the data, models, predictions, and metrics from a clean environment.

## Mapping of current files

| Current item | Current state | Eventual role |
|---|---|---|
| `preprocessing.ipynb` | Implemented download, features, targets, plots and CSV exports | Keep as exploration; move reusable logic into `data.py` and `features.py` |
| `arima/arima.ipynb` | Implemented order selection, backtest, diagnostics and forecast | Keep as analysis; move reusable model/evaluation code into `models/arima.py` and shared modules |
| `scratch/train_arima.py` | Duplicates much of ARIMA workflow | Refactor into the shared run entry point, then retire duplicate logic after parity is checked |
| `scratch/build_arima_notebook.py` | Generates the ARIMA notebook | Keep only if notebook generation remains a deliberate workflow; otherwise archive after confirming it is not needed |
| `src/btc_forecasting/api/` | Local FastAPI health, readiness, model-status, and persistence/ARIMA forecast endpoints | Keep as the backend interface; serve trained models after artifact and inference contracts exist |
| `src/btc_forecasting/evaluation.py` | Shared metric comparison on exactly aligned prediction dates | Extend with rolling-origin orchestration and prediction records |
| `src/btc_forecasting/models/xgboost_model.py` | Fit/predict adapter with target-column and feature-schema checks | Integrate with chronological tuning, shared evaluation, and artifact persistence |
| `src/btc_forecasting/models/arima.py` | Fixed-order forecast with prediction intervals | Reproduce notebook evaluation using the shared protocol |
| `src/btc_forecasting/models/lstm.py` | Documentation-only stub | Implement after sequence and training-only scaling contracts are defined |
| `scripts/run_experiment.py` | Initial ARIMA/XGBoost/persistence comparison with validation selection and test artifacts | Extend to additional model adapters and richer experiment reporting |
| `src/btc_forecasting/artifacts.py` | Writes predictions, scores, and run metadata to unique run directories | Extend as the experiment contract stabilizes |
| `xgboost/xgboost.ipynb` | Empty (0 bytes) | Optional analysis notebook; model implementation belongs in `models/xgboost.py` |
| `lstm/lstm.ipynb` | Empty (0 bytes) | Optional analysis notebook; model implementation belongs in `models/lstm.py` |
| `bitcoin_data.csv`, `bitcoin_features.csv` at root and in `data/` | Four identical processed CSV copies | Migrate to one canonical processed path; preserve old copies until the new pipeline is verified |
| `arima/arima_metrics.json` | Existing ARIMA metrics | Retain as historical output; new runs should use a consistent artifact format |
| `plots/` | Existing preprocessing and ARIMA plots | Preserve; later runs save plots with their metrics and predictions under `artifacts/` |
| `.venv/` | Local Python 3.14 environment, currently without project dependencies | Keep local and out of Git; use a separate Python 3.12 environment for the declared model stack |
| `README.md` | Project overview, proposed protocol, environment setup, and implementation sequence | Keep as quick start and user-facing project guide |

## Missing pieces before model comparison

The package, dependency bounds, and default experiment configuration exist. Data validation, feature engineering, chronological date splits, shared metrics, persistence/ARIMA API serving, XGBoost and rolling ARIMA adapters, aligned-date comparison, an initial ARIMA/XGBoost experiment runner, and run artifact persistence are implemented. LSTM, automated checks, model artifact loading for API serving, and broader experiment reporting are still missing.

1. Install the declared dependencies in a compatible Python 3.12 environment and record a resolved lockfile.
2. Preserve a raw OHLCV snapshot and define one canonical processed-data output.
3. Add automated checks for data validation, feature generation, split boundaries, target leakage, and rolling ARIMA updates.
4. Review and run the initial ARIMA/XGBoost experiment in a compatible environment; verify its saved prediction records and metrics.
5. Implement the LSTM adapter with sequence construction and training-only scaling, then evaluate it under the same protocol.
## Evaluation contract

The current proposed comparison is a one-day-ahead next-close forecast made after the daily candle closes. Train through 2023-12-31, select settings using 2024, and reserve 2025 for final testing. At test time use rolling one-step predictions with fixed learned parameters; update ARIMA's state only as actual closes become available. Compare all models on the same dates against the persistence baseline (`next close = current close`). Report MAE, RMSE, and direction accuracy using each forecast versus the prior observed close. Confirm split boundaries and available dates against the raw source before execution.

## Recommended build order

1. Install dependencies into a Python 3.12 environment and lock the resolved versions.
2. Implement data validation, raw/processed paths, and feature generation; compare new outputs with the legacy CSVs before changing the existing workflow.
3. Add automated checks for split boundaries, target leakage, and equal model evaluation windows.
4. Review and validate the current ARIMA/XGBoost comparison in a compatible environment.
5. Implement and evaluate LSTM with sequence construction and training-only scaling.
6. Add saved-model loading to the API after defining model artifact versioning and compatibility.
7. Consider UI and deployment after clean-environment reproducibility is established.

Each step should be reviewed before retiring or moving its corresponding existing work.

