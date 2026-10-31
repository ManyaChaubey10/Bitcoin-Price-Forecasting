"""ARIMA forecasting for the backend's daily close-price series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_arima(
    close_history: pd.Series,
    horizon_days: int,
    *,
    order: tuple[int, int, int],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Fit a fixed ARIMA order and return mean and 95% interval by future date."""
    if not isinstance(close_history, pd.Series) or close_history.empty:
        raise ValueError("close_history must be a non-empty pandas Series")
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least one")
    if len(order) != 3 or any(not isinstance(value, int) or value < 0 for value in order):
        raise ValueError("order must contain three non-negative integers")
    if not isinstance(close_history.index, pd.DatetimeIndex):
        raise ValueError("close_history must use a DatetimeIndex")

    values = pd.to_numeric(close_history, errors="raise").astype(float)
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("close_history must contain finite, non-missing values")
    if not values.index.is_monotonic_increasing or values.index.has_duplicates:
        raise ValueError("close_history dates must be unique and chronological")
    if len(values) > 1 and not (values.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("close_history must have one observation per calendar day")

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
