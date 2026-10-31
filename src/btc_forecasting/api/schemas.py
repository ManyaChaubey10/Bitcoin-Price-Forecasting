"""Typed API request and response contracts."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModelName = Literal["persistence", "arima", "xgboost", "lstm"]


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: Literal["BTC-USD"] = "BTC-USD"
    model: ModelName = "persistence"
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


class ModelStatus(BaseModel):
    name: ModelName
    implemented: bool
    ready: bool
    detail: str


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
