"""FastAPI application for the BTC-USD forecasting backend."""

from fastapi import FastAPI, HTTPException

from .schemas import ForecastRequest, ForecastResponse, HealthResponse, ModelListResponse
from .service import (
    DataUnavailableError,
    ForecastUnavailableError,
    ModelUnavailableError,
    create_forecast,
    data_path,
    list_models,
)

app = FastAPI(
    title="BTC-USD Forecasting API",
    version="0.1.0",
    description="Daily BTC-USD close forecasts from the configured historical snapshot.",
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Report API process health and whether the configured data file exists."""
    return HealthResponse(status="ok", data_available=data_path().is_file())


@app.get("/api/v1/models", response_model=ModelListResponse, tags=["forecasting"])
def models() -> ModelListResponse:
    """List implemented models and whether their dependencies are available."""
    return ModelListResponse(models=list_models())


@app.post("/api/v1/forecast", response_model=ForecastResponse, tags=["forecasting"])
def forecast(request: ForecastRequest) -> ForecastResponse:
    """Create a one-to-90-day forecast using the local BTC-USD history."""
    try:
        return create_forecast(request)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except (DataUnavailableError, ForecastUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
