# BTC-USD Forecasting Lab

A research platform for daily Bitcoin price forecasting. ARIMA, XGBoost, and LSTM remain independent model implementations. A separate validation-only router supports objective-specific automatic selection and explicit manual selection. Persistence is always included as a baseline.

## Research protocol

| Use | Target dates | Rule |
|---|---|---|
| Model training | 2018-01-01 through 2023-12-31 | Chronological; no shuffling |
| Validation and selection | 2024-01-01 through 2024-12-31 | Choose configurations and router decisions here |
| Final model fit | Through 2025-12-31 | Refit selected settings after validation selection |
| Final evaluation | Available 2026 targets only | Report once; never use these outcomes for tuning or routing |

The current saved 2026 evaluation covers January 1 through November 14, 2026. The routed registry presently contains one-day validation scores only. The interface shows a validation-based automatic route only where that evidence exists. Manual selection can be used for other forecast horizons, but it is labeled as unvalidated. Automatic routing currently falls back to persistence because no learned-model gain has been established as reliable using paired validation uncertainty.

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
python -m uvicorn btc_forecasting.api.main:app --reload --app-dir src
```

Start the React dashboard in another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the frontend terminal (normally `http://127.0.0.1:5173`). FastAPI documentation is at `http://127.0.0.1:8000/docs`. For a production bundle, run `npm run build` in `frontend/`; serve the generated `frontend/dist` with a static host configured to proxy `/api` to FastAPI.

If using different local data, set `BTC_FORECASTING_DATA` and `BTC_FORECASTING_FEATURES` to compatible CSVs. `BTC_FORECASTING_MODEL_RUN` can point to an explicit artifact run directory. Model artifacts are matched to the registry's recorded artifact run by default; do not replace those files in place.

## API overview

- `GET /health`, `GET /ready`: service and daily-data readiness.
- `GET /api/v1/models`: implementation, artifact readiness, and validated horizons.
- `GET /api/v1/performance`: stored validation records only.
- `GET /api/v1/history`: actual historical market series for the dashboard.
- `GET /api/v1/evaluation`: saved final 2026 evaluation for display only; response states it is not used for selection.
- `GET /api/v1/explainability`: saved model settings and available diagnostics.
- `POST /api/v1/forecast`: automatic or manual forecasting.
- `POST /api/v1/forecast/compare`: same-origin forecasts from available models and persistence.

Automatic selection requires a matching objective/horizon validation record. The frontend does not extrapolate a one-day validation winner to longer horizons. XGBoost and LSTM currently have a validated inference feature contract for one-day forecasts. ARIMA can produce multi-step forecasts, but no multi-step validation record is currently available. Ensemble output remains disabled.

## Demonstration guide

See [DEMO_GUIDE.md](DEMO_GUIDE.md) for a presentation flow and scientific caveats. For the model/data protocol and current gaps, see [ARCHITECTURE.md](ARCHITECTURE.md).
