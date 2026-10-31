"""ARIMA forecasting for the backend's daily close-price series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _validate_close_history(close_history: pd.Series) -> pd.Series:
    if not isinstance(close_history, pd.Series) or close_history.empty:
        raise ValueError("close_history must be a non-empty pandas Series")
    if not isinstance(close_history.index, pd.DatetimeIndex):
        raise ValueError("close_history must use a DatetimeIndex")

    values = pd.to_numeric(close_history, errors="raise").astype(float)
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("close_history must contain finite, non-missing values")
    if (values <= 0).any():
        raise ValueError("close_history prices must be greater than zero")
    if not values.index.is_monotonic_increasing or values.index.has_duplicates:
        raise ValueError("close_history dates must be unique and chronological")
    if len(values) > 1 and not (
        values.index.to_series().diff().dropna() == pd.Timedelta(days=1)
    ).all():
        raise ValueError("close_history must have one observation per calendar day")
    return values


class RollingARIMAForecaster:
    """One-step ARIMA forecasts with fixed parameters and observed-state updates.

    After each call to ``forecast_next``, the caller must provide that date's
    actual close through ``observe`` before requesting another prediction.
    Parameters are fitted once in ``fit``; appending observations never refits.
    """

    def __init__(self, order: tuple[int, int, int]) -> None:
        if len(order) != 3 or any(type(value) is not int or value < 0 for value in order):
            raise ValueError("order must contain three non-negative integers")
        self.order = order
        self._result: Any | None = None
        self._last_date: pd.Timestamp | None = None
        self._awaiting_observation: pd.Timestamp | None = None
        self._index_name = "Date"

    def fit(self, close_history: pd.Series) -> RollingARIMAForecaster:
        """Fit fixed ARIMA parameters using the supplied development history."""
        values = _validate_close_history(close_history)

        # Imported lazily so other model adapters do not require statsmodels.
        from statsmodels.tsa.arima.model import ARIMA

        self._result = ARIMA(values.asfreq("D"), order=self.order).fit()
        self._last_date = values.index[-1]
        self._awaiting_observation = None
        self._index_name = values.index.name or "Date"
        return self

    def forecast_next(self) -> float:
        """Forecast the next calendar day's close without updating model state."""
        if self._result is None or self._last_date is None:
            raise RuntimeError("fit must be called before forecast_next")
        if self._awaiting_observation is not None:
            raise RuntimeError("observe the previous forecast's actual close first")

        forecast = self._result.get_forecast(steps=1)
        value = float(np.asarray(forecast.predicted_mean, dtype=float).reshape(-1)[0])
        if not np.isfinite(value):
            raise ValueError("ARIMA returned a non-finite forecast")
        self._awaiting_observation = self._last_date + pd.Timedelta(days=1)
        return value

    def observe(self, date: str | pd.Timestamp, close: float) -> None:
        """Append an observed close for the forecast date without refitting."""
        if self._result is None or self._last_date is None:
            raise RuntimeError("fit must be called before observe")
        if self._awaiting_observation is None:
            raise RuntimeError("forecast_next must be called before observe")

        observed_date = pd.Timestamp(date)
        if observed_date.tzinfo is not None:
            observed_date = observed_date.tz_convert(None)
        observed_date = observed_date.normalize()
        if observed_date != self._awaiting_observation:
            raise ValueError(
                f"expected observation for {self._awaiting_observation:%Y-%m-%d}, "
                f"got {observed_date:%Y-%m-%d}"
            )
        observed_close = float(close)
        if not np.isfinite(observed_close):
            raise ValueError("observed close must be finite")

        observation = pd.Series(
            [observed_close],
            index=pd.DatetimeIndex([observed_date], name=self._index_name),
        )
        self._result = self._result.append(observation, refit=False)
        self._last_date = observed_date
        self._awaiting_observation = None


def forecast_arima(
    close_history: pd.Series,
    horizon_days: int,
    *,
    order: tuple[int, int, int],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Fit a fixed ARIMA order and return mean and 95% interval by future date."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least one")
    if len(order) != 3 or any(type(value) is not int or value < 0 for value in order):
        raise ValueError("order must contain three non-negative integers")
    values = _validate_close_history(close_history)

    # Imported here so persistence-only API users do not need statsmodels.
    from statsmodels.tsa.arima.model import ARIMA

    fitted = ARIMA(values, order=order).fit()
    result = fitted.get_forecast(steps=horizon_days)
    index = pd.date_range(
        start=values.index[-1] + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
        name=values.index.name or "Date",
    )
    interval = result.conf_int(alpha=0.05).to_numpy(dtype=float)
    mean = pd.Series(result.predicted_mean.to_numpy(dtype=float), index=index, name="prediction")
    lower = pd.Series(interval[:, 0], index=index, name="lower_95")
    upper = pd.Series(interval[:, 1], index=index, name="upper_95")
    return mean, lower, upper
