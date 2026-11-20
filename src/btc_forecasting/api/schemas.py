"""Typed API request and response contracts."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModelName = Literal["persistence", "arima", "xgboost", "lstm"]
RequestedModel = Literal["auto", "persistence", "arima", "xgboost", "lstm"]
ForecastObjective = Literal["price_mae", "price_rmse", "directional_accuracy"]


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: Literal["BTC-USD"] = "BTC-USD"
    model: RequestedModel = "auto"
    objective: ForecastObjective = "price_rmse"
    horizon_days: int = Field(default=1, ge=1, le=90)


class ForecastPoint(BaseModel):
    date: date
    predicted_close_usd: float
    lower_95_usd: float | None = None
    upper_95_usd: float | None = None


class ForecastResponse(BaseModel):
    symbol: Literal["BTC-USD"]
    model: ModelName
    horizon_days: int
    generated_at_utc: datetime
    history_start: date
    history_end: date
    last_observed_close_usd: float
    data_source: str
    forecasts: list[ForecastPoint]
    selection: dict | None = None


class ModelForecastResult(BaseModel):
    model: ModelName
    ready: bool
    supported: bool
    detail: str
    forecasts: list[ForecastPoint] = Field(default_factory=list)
    validation_metrics: dict | None = None


class ForecastComparisonResponse(BaseModel):
    symbol: Literal["BTC-USD"]
    objective: ForecastObjective
    horizon_days: int
    history_end: date
    selected_model: ModelName | None = None
    selection: dict | None = None
    models: list[ModelForecastResult]
    combined_forecast: list[ForecastPoint] | None = None
    combined_forecast_reason: str


class ModelStatus(BaseModel):
    name: ModelName
    implemented: bool
    ready: bool
    detail: str
    validated_horizons: list[int] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    models: list[ModelStatus]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    data_available: bool


class ReadinessResponse(BaseModel):
    status: Literal["ready"]
    data_source: str
    history_start: date
    history_end: date
