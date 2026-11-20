"""FastAPI application for the BTC-USD forecasting backend."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from .schemas import (
    ForecastComparisonResponse,
    ForecastRequest,
    ForecastObjective,
    ForecastResponse,
    HealthResponse,
    ModelListResponse,
    ReadinessResponse,
)
from .service import (
    DataUnavailableError,
    ForecastUnavailableError,
    ModelUnavailableError,
    check_readiness,
    compare_forecasts,
    create_forecast,
    data_path,
    diagnostic_image_path,
    final_evaluation_report,
    historical_market_data,
    list_models,
    model_explainability,
    performance_records,
)
from .web import web_router

app = FastAPI(
    title="BTC-USD Forecasting API",
    version="0.1.0",
    description="Daily BTC-USD close forecasts from the configured historical snapshot.",
)
app.include_router(web_router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Report API process health and whether the configured data file exists."""
    return HealthResponse(status="ok", data_available=data_path().is_file())


@app.get("/ready", response_model=ReadinessResponse, tags=["system"])
def readiness() -> ReadinessResponse:
    """Report whether valid historical data is available for forecasting."""
    try:
        return check_readiness()
    except DataUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v1/models", response_model=ModelListResponse, tags=["forecasting"])
def models() -> ModelListResponse:
    """List implemented models and whether their dependencies are available."""
    return ModelListResponse(models=list_models())


@app.get("/api/v1/performance", tags=["forecasting"])
def performance(
    objective: ForecastObjective | None = Query(default=None),
    horizon_days: int | None = Query(default=None, ge=1, le=90),
) -> dict:
    """Return model scores from the saved validation-only registry."""
    try:
        return performance_records(objective=objective, horizon_days=horizon_days)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v1/history", tags=["market data"])
def history(days: int = Query(default=365, ge=1, le=3650)) -> dict:
    """Serve the latest price, volume, return, and volatility history."""
    try:
        return historical_market_data(days=days)
    except (ValueError, DataUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v1/evaluation", tags=["evaluation"])
def evaluation() -> dict:
    """Display frozen 2026 final-evaluation artifacts; routing never reads them."""
    return final_evaluation_report()


@app.get("/api/v1/explainability", tags=["models"])
def explainability() -> dict:
    """Return model-specific configurations and saved diagnostics."""
    try:
        return model_explainability()
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v1/diagnostics/{model_name}/{image_name}", tags=["models"])
def diagnostic_image(model_name: str, image_name: str) -> FileResponse:
    """Serve an explicitly whitelisted diagnostic image from a saved model run."""
    try:
        return FileResponse(diagnostic_image_path(model_name, image_name), media_type="image/png")
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/forecast", response_model=ForecastResponse, tags=["forecasting"])
def forecast(request: ForecastRequest) -> ForecastResponse:
    """Create a manually selected or validation-routed BTC-USD forecast."""
    try:
        return create_forecast(request)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except (DataUnavailableError, ForecastUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/v1/forecast/compare", response_model=ForecastComparisonResponse, tags=["forecasting"])
def forecast_comparison(request: ForecastRequest) -> ForecastComparisonResponse:
    """Compare each ready model on the same forecast origin."""
    try:
        return compare_forecasts(request)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except (DataUnavailableError, ForecastUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
