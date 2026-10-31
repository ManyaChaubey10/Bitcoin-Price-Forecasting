<<<<<<< HEAD

=======
# Bitcoin (BTC-USD) Price Forecasting

This repository is a research project for daily Bitcoin price forecasting. The current implementation contains a yfinance preprocessing notebook and an ARIMA workflow. XGBoost and LSTM notebooks exist as empty placeholders. The project is being organized into a reproducible pipeline before those models are implemented.

## Current files

- `preprocessing.ipynb` downloads BTC-USD daily OHLCV data, engineers technical and calendar features, creates next-day targets, and exports CSVs and plots.
- `arima/arima.ipynb` performs ARIMA order selection, a 2025 one-step backtest, diagnostics, and a 90-day forecast.
- `scratch/train_arima.py` contains a script version of much of the ARIMA notebook workflow.
- `scratch/build_arima_notebook.py` builds the ARIMA notebook.
- `xgboost/xgboost.ipynb` and `lstm/lstm.ipynb` are currently empty.
- `data/` and the repository root contain identical copies of the processed feature CSVs. Both `bitcoin_data.csv` and `bitcoin_features.csv` currently contain the same engineered dataset, including target columns; neither is a raw-only OHLCV snapshot.
- `plots/` contains preprocessing and ARIMA figures; `arima/arima_metrics.json` stores ARIMA results.

The preprocessing notebook requests data from 2018-01-01 through 2025-12-31 (yfinance's end date is exclusive). After rolling-window rows are removed, the checked-in CSVs cover 2018-07-19 through 2025-12-30. The original raw OHLCV data is not currently preserved in those exports.

## Proposed model comparison protocol

Use this protocol for ARIMA, XGBoost, LSTM, and the naïve baseline so that scores answer the same forecasting question:

- **Decision time:** after each UTC daily candle has closed and its OHLCV values are available. A prediction made at date `t` forecasts the next daily close at `t+1`.
- **Common target:** `Target_Next_Close`, the next-day close. All reported model metrics use the same target and dates.
- **Chronological periods:** train through 2023-12-31; use 2024 for model and hyperparameter selection; keep 2025 as the final test period. Do not use 2025 to choose features, transformations, model orders, or hyperparameters. Confirm these cutoffs against the actual source data before running the pipeline.
- **Test procedure:** evaluate one day ahead in rolling-origin fashion. At each date, use only observations available by that prediction time. Models keep their learned parameters fixed through the test period; ARIMA may update its state with each newly observed close. This reflects daily forecast use and avoids comparing a 90-day fixed-origin forecast with a one-step forecast.
- **Inputs and transforms:** construct features using current and past values only. Explicitly remove all target columns from model inputs. Fit scalers and other learned preprocessing on training data only, then apply them unchanged to validation and test data. LSTM sequences must not cross split boundaries using future values.
- **Baseline:** predict that the next close equals the latest observed close (`Close[t]`). A complex model should be compared with this persistence forecast.
- **Metrics:** report MAE and RMSE in USD, plus directional accuracy computed by comparing `sign(predicted_close[t+1] - actual_close[t])` with `sign(actual_close[t+1] - actual_close[t])`. Use the exact same test dates for every model; document any unavailable predictions instead of silently changing the comparison window.

This is a proposed protocol, not a claim that existing ARIMA results already follow it. The current ARIMA notebook selects its order using data through 2025 before evaluating on 2025, so its reported 2025 metrics are not a clean held-out estimate. Its directional accuracy also compares consecutive predicted values rather than each prediction with the prior observed close. Recompute results after the protocol is implemented.

## Reproducibility and execution

Current workflow:

1. Run `preprocessing.ipynb` to download data, build features and targets, and export the processed dataset.
2. Run `arima/arima.ipynb` for the existing ARIMA analysis.

The project now declares base and optional dependencies in `pyproject.toml`. Use Python 3.12 on Windows for the planned model stack. The existing local `.venv` uses Python 3.14 and is left untouched; install Python 3.12 before creating a compatible environment:

```powershell
py -3.12 -m venv .venv-312
.\.venv-312\Scripts\python -m pip install --upgrade pip
.\.venv-312\Scripts\python -m pip install -e ".[notebooks,arima,xgboost,lstm,dev]"
```

The dependency entries currently specify supported version ranges, not an exact lockfile. Exact resolved versions should be locked once the dependencies are installed and the environment is verified. A reproducible run should also record date range, source and retrieval time, split boundaries, feature/target definitions, random seeds where applicable, and model settings. Preserve downloaded raw OHLCV separately from processed features and use one canonical processed-data path.

## Recommended implementation sequence

1. Establish the shared forecast protocol and document the project (done).
2. Add package/dependency/configuration skeletons (started; see `pyproject.toml`, `src/`, and `configs/default.yaml`).
3. Implement data validation, feature generation, chronological splitting, and shared metrics; preserve raw data and migrate to one canonical processed output only after comparing it with the existing CSVs.
4. Refactor ARIMA to select settings without using the final test period, then reproduce the rolling one-step evaluation and baseline.
5. Implement XGBoost and LSTM against the same target, feature availability, split dates, rolling test procedure, and metrics.
6. Add leakage and split-boundary checks, then save comparable metrics, predictions, configuration, and plots as reproducible run artifacts.
7. Build a forecasting application only after the pipeline can reproduce model artifacts and forecasts from documented inputs.

## Notes

- Technical indicators that include the current day's close are available only under the after-close decision-time assumption above.
- Missing calendar dates should be investigated before filling; forward-filling a close creates artificial zero-change observations.
- `README.md` is the project entry point. Update it as implementation decisions and supported commands become concrete.
>>>>>>> de6c385 (the arch)
