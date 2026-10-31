# Forecasting API

This is a local development API for the BTC-USD project. It serves forecasts from the repository's current historical CSV; it does not download fresh market data. The configured default is `data/bitcoin_data.csv`. Set `BTC_FORECASTING_DATA` to use another CSV with a `Date` column and valid daily OHLCV columns. Relative values are resolved from the repository root; absolute paths are used as given.

## Install and run

Use a Python version allowed by the project (`>=3.11,<3.13`). From the repository root:

```powershell
python -m pip install -e ".[api,arima]"
python -m uvicorn btc_forecasting.api.main:app --reload --app-dir src
```

The interactive API description is available at `http://127.0.0.1:8000/docs` while the local server is running.

## Endpoints

- `GET /health` reports process health and whether the configured data file exists.
- `GET /api/v1/models` lists model implementation and dependency readiness.
- `POST /api/v1/forecast` accepts a BTC-USD model name and a one-to-90-day horizon.

Example request:

```json
{
  "symbol": "BTC-USD",
  "model": "persistence",
  "horizon_days": 7
}
```

`persistence` repeats the latest observed close and serves as a baseline. `arima` fits the order recorded in `arima/arima_metrics.json` to the local close history. Both use the last date in the CSV as the forecast origin. The API marks XGBoost and LSTM unavailable until their serving implementations exist; it does not return fabricated values for them.

This API is a development foundation. It has no authentication, rate limiting, CORS policy, live data refresh, trained XGBoost/LSTM artifacts, or deployment configuration. Set those policies when the consuming application and hosting environment are defined.
