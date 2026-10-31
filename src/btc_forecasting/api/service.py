"""Application services for health, model readiness, and forecasts."""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ..data import load_ohlcv_csv
from ..models.arima import forecast_arima
from .schemas import ForecastPoint, ForecastRequest, ForecastResponse, ModelStatus

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "bitcoin_data.csv"
ARIMA_ORDER_PATH = PROJECT_ROOT / "arima" / "arima_metrics.json"
DATA_ENV_VAR = "BTC_FORECASTING_DATA"


class DataUnavailableError(RuntimeError):
    """The configured historical data cannot support a forecast."""


class ModelUnavailableError(RuntimeError):
    """The requested model is not implemented or its dependency is missing."""


def data_path() -> Path:
    """Return the configured dataset path or the repository's current CSV.

    Relative paths in ``BTC_FORECASTING_DATA`` are anchored at the project
    root so the selected dataset does not depend on the server's working dir.
    """
    configured = os.environ.get(DATA_ENV_VAR)
    if not configured:
        return DEFAULT_DATA_PATH

    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def list_models() -> list[ModelStatus]:
    """Report implemented models separately from optional runtime readiness."""
    arima_installed = (
        importlib.util.find_spec("statsmodels") is not None and ARIMA_ORDER_PATH.is_file()
    )
    return [
        ModelStatus(
            name="persistence",
            implemented=True,
            ready=True,
            detail="Repeats the latest observed close; available as a baseline.",
        ),
        ModelStatus(
            name="arima",
            implemented=True,
            ready=arima_installed,
            detail=(
                "Fits the saved ARIMA order to the available close history."
                if arima_installed
                else "ARIMA needs statsmodels and the saved model-order file to be available."
            ),
        ),
        ModelStatus(
            name="xgboost",
            implemented=False,
            ready=False,
            detail="No trained or serving implementation is available yet.",
        ),
        ModelStatus(
            name="lstm",
            implemented=False,
            ready=False,
            detail="No trained or serving implementation is available yet.",
        ),
    ]


def _saved_arima_order() -> tuple[int, int, int]:
    try:
        metrics = json.loads(ARIMA_ORDER_PATH.read_text(encoding="utf-8-sig"))
        order = tuple(int(value) for value in metrics["model_order"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ModelUnavailableError(
            "The saved ARIMA model order is missing or invalid."
        ) from exc
    if len(order) != 3 or any(value < 0 for value in order):
        raise ModelUnavailableError("The saved ARIMA model order must contain three non-negative integers.")
    return order  # type: ignore[return-value]


def create_forecast(request: ForecastRequest) -> ForecastResponse:
    """Build a response from the local historical snapshot and chosen model."""
    path = data_path()
    if not path.is_file():
        raise DataUnavailableError(
            f"Historical BTC data is unavailable. Set {DATA_ENV_VAR} to a CSV path."
        )
    try:
        history = load_ohlcv_csv(path)
    except (OSError, ValueError, KeyError) as exc:
        raise DataUnavailableError(f"Historical BTC data could not be loaded: {exc}") from exc

    close = history["Close"].astype(float)
    if close.empty:
        raise DataUnavailableError("Historical BTC data has no close prices.")

    lower = upper = None
    if request.model == "persistence":
        predicted_values = np.repeat(float(close.iloc[-1]), request.horizon_days)
    elif request.model == "arima":
        if importlib.util.find_spec("statsmodels") is None:
            raise ModelUnavailableError("Install the project's arima dependency group to enable ARIMA.")
        order = _saved_arima_order()
        try:
            predicted, lower, upper = forecast_arima(close, request.horizon_days, order=order)
        except Exception as exc:
            raise DataUnavailableError(f"ARIMA forecast failed: {exc}") from exc
        predicted_values = predicted.to_numpy(dtype=float)
    else:
        raise ModelUnavailableError(f"The {request.model} model is not implemented for serving yet.")

    predicted_values = np.asarray(predicted_values, dtype=float)
    if predicted_values.ndim != 1 or len(predicted_values) != request.horizon_days:
        raise DataUnavailableError(
            "The selected model returned a forecast with an unexpected length."
        )
    if not np.isfinite(predicted_values).all():
        raise DataUnavailableError("The selected model returned a non-finite forecast.")

    lower_values = upper_values = None
    if lower is not None or upper is not None:
        if lower is None or upper is None:
            raise DataUnavailableError(
                "The selected model returned an incomplete confidence interval."
            )
        lower_values = np.asarray(lower, dtype=float)
        upper_values = np.asarray(upper, dtype=float)
        if (
            lower_values.ndim != 1
            or upper_values.ndim != 1
            or len(lower_values) != request.horizon_days
            or len(upper_values) != request.horizon_days
        ):
            raise DataUnavailableError(
                "The selected model returned confidence intervals with an unexpected length."
            )
        if not np.isfinite(lower_values).all() or not np.isfinite(upper_values).all():
            raise DataUnavailableError(
                "The selected model returned non-finite confidence intervals."
            )
        if np.any(lower_values > upper_values):
            raise DataUnavailableError(
                "The selected model returned invalid confidence interval bounds."
            )

    start = history.index[0]
    last_date = history.index[-1]
    future_dates = [last_date.date() + timedelta(days=step) for step in range(1, request.horizon_days + 1)]
    points = [
        ForecastPoint(
            date=forecast_date,
            predicted_close_usd=float(predicted_values[index]),
            lower_95_usd=float(lower_values[index]) if lower_values is not None else None,
            upper_95_usd=float(upper_values[index]) if upper_values is not None else None,
        )
        for index, forecast_date in enumerate(future_dates)
    ]
    return ForecastResponse(
        symbol=request.symbol,
        model=request.model,
        horizon_days=request.horizon_days,
        generated_at_utc=datetime.now(timezone.utc),
        history_start=start.date(),
        history_end=last_date.date(),
        last_observed_close_usd=float(close.iloc[-1]),
        data_source=path.name,
        forecasts=points,
    )
