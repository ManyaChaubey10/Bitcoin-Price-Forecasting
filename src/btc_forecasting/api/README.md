# Forecasting API

FastAPI serving layer for the local BTC-USD forecasting project. It loads the existing final-fit model bundles and data; it does not fetch market data or train models on application startup.

## Start

From the repository root in PowerShell, create and activate a separate Python 3.12 environment, install locked dependencies, and start the service:

```powershell
if (-not (Test-Path .venv312)) { uv venv .venv312 --python 3.12 }
. .\.venv312\Scripts\Activate.ps1
uv sync --active --locked --all-extras
python -m uvicorn btc_forecasting.api.main:app --reload --app-dir src
```

Interactive API docs are at `http://127.0.0.1:8000/docs`. The React/Vite client is in `frontend/`; see the root README and `DEMO_GUIDE.md` for launch instructions.

## Endpoints

- `GET /health`, `GET /ready`: process and configured data readiness.
- `GET /api/v1/models`: model implementation, artifact readiness, and validation-backed horizons.
- `GET /api/v1/performance`: scores from the stored validation-only registry.
- `GET /api/v1/history`: actual historical market series for the dashboard.
- `GET /api/v1/evaluation`: saved 2026 final-evaluation results for display only.
- `GET /api/v1/explainability`: actual saved model configurations and available diagnostics.
- `POST /api/v1/forecast`: automatic or manual model selection.
- `POST /api/v1/forecast/compare`: forecasts from all ready models on the same origin.

Example:

```json
{
  "symbol": "BTC-USD",
  "model": "auto",
  "objective": "price_rmse",
  "horizon_days": 1
}
```

The router reads `configs/model_selection.json`; it never reads final test metrics. Currently only one-day validation records exist. For an unsupported horizon, automatic selection is unavailable and the response explains that no validation record exists. Manual ARIMA/persistence multi-step forecasts are possible, but remain unvalidated. XGBoost/LSTM use the saved feature pipeline only for one-day predictions.

The configured protocol is training target dates through 2023-12-31, validation in 2024, final fit through 2025-12-31, and final evaluation on available 2026 targets (currently through 2026-11-14). The 2026 endpoint is display-only. `BTC_FORECASTING_DATA`, `BTC_FORECASTING_FEATURES`, and `BTC_FORECASTING_MODEL_RUN` can point to compatible local inputs or a specific model-run directory.

For gaps in feature optimization, horizon-specific validation, reliable paired comparisons, ensemble evaluation, and deployment, see `ARCHITECTURE.md`.
