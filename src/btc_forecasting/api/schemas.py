"""Typed API request and response contracts."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ModelName = Literal["persistence", "arima", "xgboost", "lstm"]
RequestedModel = Literal["auto", "persistence", "arima", "xgboost", "lstm"]
ForecastObjective = Literal["price_mae", "price_rmse", "directional_accuracy"]


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: Literal["BTC-USD"] = "BTC-USD"
    model: RequestedModel = "auto"
    objective: ForecastObjective = "price_rmse"
    # `target_date` powers the date-first product flow. `horizon_days` remains
    # for API clients that already use the original horizon-based contract.
    target_date: date | None = None
    horizon_days: int | None = Field(default=None, ge=1, le=3650)

    @model_validator(mode="after")
    def validate_requested_date(self) -> ForecastRequest:
        if self.target_date is not None:
            cutoff = date(2025, 12, 31)
            if self.target_date <= cutoff:
                raise ValueError(f"target_date must be after the fixed history cutoff {cutoff}.")
            days = (self.target_date - cutoff).days
            if days > 3650:
                raise ValueError("target_date may be at most 10 years after the history cutoff.")
            if self.horizon_days is not None and self.horizon_days != days:
                raise ValueError("target_date and horizon_days must identify the same date.")
        return self


class ForecastPoint(BaseModel):
    date: date
    predicted_close_usd: float
    lower_95_usd: float | None = None
    upper_95_usd: float | None = None


class ForecastResponse(BaseModel):
    symbol: Literal["BTC-USD"]
    model: ModelName
    horizon_days: int
    requested_date: date
    experimental_horizon: bool = False
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
    experimental_horizon: bool = False


class ForecastComparisonResponse(BaseModel):
    symbol: Literal["BTC-USD"]
    objective: ForecastObjective
    horizon_days: int
    requested_date: date
    experimental_horizon: bool = False
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
