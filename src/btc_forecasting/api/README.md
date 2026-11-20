# Forecasting API

FastAPI serving layer for the local BTC-USD forecasting project. It loads the existing final-fit model bundles and data; it does not fetch market data or train models on application startup.

## Start

From the repository root in PowerShell, create and activate a separate Python 3.12 environment, install locked dependencies, and start the service:

```powershell
if (-not (Test-Path .venv312)) { uv venv .venv312 --python 3.12 }
. .\.venv312\Scripts\Activate.ps1
uv sync --active --locked --all-extras
python -m uvicorn btc_forecasting.api.main:app --reload --app-dir src --port 8001
```

Interactive API docs are at `http://127.0.0.1:8001/docs`. The React/Vite client is in `frontend/`; see the root README and `DEMO_GUIDE.md` for launch instructions.

## Endpoints

- `GET /health`, `GET /ready`: process and configured data readiness.
- `GET /api/v1/models`: model implementation, artifact readiness, and validation-backed horizons.
- `GET /api/v1/performance`: scores from the stored validation-only registry.
- `GET /api/v1/history`: historical market series through the fixed forecast cutoff.
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
  "target_date": "2026-01-01"
}
```

The router reads `configs/model_selection.json`; it never reads final test metrics. Currently only one-day validation records exist. For other dates, automatic routing falls back to persistence and marks the forecast experimental. Manual ARIMA, XGBoost, and LSTM are also available; recursive ML projections synthesize future candles and carry forward the latest observed volume, so they remain experimental until multi-horizon validation is completed.

The forecast origin is fixed at 2025-12-31. Source snapshots may contain later actuals, but user-facing history and all forecast inputs are truncated before feature generation. Saved 2026 evaluation results (currently through 2026-11-14) are display-only. `BTC_FORECASTING_DATA` and `BTC_FORECASTING_MODEL_RUN` can point to a compatible OHLCV snapshot or a specific model run.

For gaps in feature optimization, horizon-specific validation, reliable paired comparisons, ensemble evaluation, and deployment, see `ARCHITECTURE.md`.
