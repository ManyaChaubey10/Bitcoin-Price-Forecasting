# BTC-USD Forecasting Lab

A date-based Bitcoin price forecasting platform. A user selects a target date and receives a closing-price forecast, trajectory, routing explanation, and model comparison. ARIMA, XGBoost, and LSTM remain independent implementations, with persistence as a baseline.

## Research protocol

| Use | Target dates | Rule |
|---|---|---|
| Model training | 2018-01-01 through 2023-12-31 | Chronological; no shuffling |
| Validation and selection | 2024-01-01 through 2024-12-31 | Choose configurations and router decisions here |
| Final model fit | Through 2025-12-31 | Refit selected settings after validation selection |
| Final evaluation | Available 2026 targets only | Report once; never use these outcomes for tuning or routing |

The user-facing history is fixed at December 31, 2025, even though saved final-evaluation artifacts cover available 2026 dates through November 14. The router registry contains one-day validation scores only. For other horizons, automatic routing falls back to persistence and labels the result experimental. Manual ARIMA, XGBoost, and LSTM forecasts are available as exploratory projections; recursive XGBoost/LSTM projections assume synthetic future candles and carry the last known volume forward.

The existing models were developed with a shared engineered table, so model-specific feature ablation has not yet been demonstrated. The registry does not contain 7-, 14-, or 30-day validation records. No ensemble is enabled because aligned per-date 2024 forecasts for all models are not saved. These are open research tasks, not assumed model strengths.

## Repository map

```text
configs/
  default.yaml                 split, final-fit, and model settings
  model_selection.json        validation-only routing evidence and provenance
data/
  raw/                         source OHLCV snapshots
  processed/                   causal features used by saved ML artifacts
notebooks/                     preserved educational and experiment notebooks
src/btc_forecasting/
  data.py, features.py         input validation and causal feature engineering
  splitting.py, metrics.py     chronological partitions and shared scores
  models/                       independent ARIMA, XGBoost, and LSTM adapters
  selection.py                  validation-only model router
  api/                          FastAPI services and response contracts
scripts/
  prepare_data.py, run_experiment.py
frontend/                       React/Vite forecasting dashboard
artifacts/runs/                 ignored, unique saved model and evaluation runs
```

## Run locally

Use Python 3.12 (the package supports Python 3.11–3.12) and Node.js 20 or newer. Keep the compatible environment separate from any existing `.venv`. From the repository root in PowerShell:

```powershell
if (-not (Test-Path .venv312)) { uv venv .venv312 --python 3.12 }
. .\.venv312\Scripts\Activate.ps1
uv sync --active --locked --all-extras
```

The application uses the checked-in/local data and saved run artifacts. It does not train models on startup or download new data on startup. Start the backend in one terminal:

```powershell
python -m uvicorn btc_forecasting.api.main:app --reload --app-dir src --port 8001
```

Start the React dashboard in another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the frontend terminal (normally `http://127.0.0.1:5173`). Use `npm.cmd run dev -- --host 127.0.0.1` in PowerShell when the `npm` shim is unavailable. FastAPI documentation is at `http://127.0.0.1:8001/docs`. For a production bundle, run `npm run build` in `frontend/`; serve the generated `frontend/dist` with a static host configured to proxy `/api` to FastAPI.

If using different local data, set `BTC_FORECASTING_DATA` to a daily OHLCV CSV that includes the 2025-12-31 cutoff. Features are regenerated through the shared causal feature builder from cutoff-limited OHLCV. `BTC_FORECASTING_MODEL_RUN` can point to an explicit artifact run directory. Model artifacts are matched to the registry's recorded artifact run by default; do not replace those files in place.

## API overview

- `GET /health`, `GET /ready`: service and daily-data readiness.
- `GET /api/v1/models`: implementation, artifact readiness, and validated horizons.
- `GET /api/v1/performance`: stored validation records only.
- `GET /api/v1/history`: historical market series truncated to the December 31, 2025 forecast cutoff.
- `GET /api/v1/evaluation`: saved final 2026 evaluation for display only; response states it is not used for selection.
- `GET /api/v1/explainability`: saved model settings and available diagnostics.
- `POST /api/v1/forecast`: automatic or manual forecasting.
- `POST /api/v1/forecast/compare`: same-origin forecasts from available models and persistence.

Automatic selection uses exact objective/horizon validation records. Only one-day scores are currently recorded; longer horizons are explicitly experimental. No ensemble is enabled because it has not demonstrated a validation improvement.

## Demonstration guide

See [DEMO_GUIDE.md](DEMO_GUIDE.md) for a presentation flow and scientific caveats. For the model/data protocol and current gaps, see [ARCHITECTURE.md](ARCHITECTURE.md).
