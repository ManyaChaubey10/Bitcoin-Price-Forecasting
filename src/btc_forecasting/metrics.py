"""Shared next-close metrics and persistence-baseline forecasts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def persistence_predictions(current_close: pd.Series) -> pd.Series:
    """Forecast that the next close equals the close known at decision time."""
    if not isinstance(current_close, pd.Series):
        raise TypeError("current_close must be a pandas Series indexed by forecast date")
    values = pd.to_numeric(current_close, errors="raise")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("current_close must contain finite, non-missing values")
    return values.rename("prediction")


def evaluate_predictions(
    actual_next_close: pd.Series | Sequence[float],
    predicted_next_close: pd.Series | Sequence[float],
    previous_close: pd.Series | Sequence[float],
) -> dict[str, float]:
    """Return USD MAE/RMSE and direction accuracy on exactly aligned rows.

    Direction is measured relative to each row’s observed close at decision
    time, not relative to the previous model prediction.
    """
    if isinstance(actual_next_close, pd.Series) and isinstance(predicted_next_close, pd.Series):
        if not actual_next_close.index.equals(predicted_next_close.index):
            raise ValueError("Actual and predicted Series must have identical indexes")
    if isinstance(actual_next_close, pd.Series) and isinstance(previous_close, pd.Series):
        if not actual_next_close.index.equals(previous_close.index):
            raise ValueError("Actual and previous-close Series must have identical indexes")

    actual = np.asarray(actual_next_close, dtype=float).reshape(-1)
    predicted = np.asarray(predicted_next_close, dtype=float).reshape(-1)
    prior = np.asarray(previous_close, dtype=float).reshape(-1)
    if not len(actual) or len(actual) != len(predicted) or len(actual) != len(prior):
        raise ValueError("Actual, predicted, and previous-close inputs must have equal non-zero lengths")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all() and np.isfinite(prior).all()):
        raise ValueError("Metric inputs must contain only finite values")

    errors = predicted - actual
    actual_direction = np.sign(actual - prior)
    predicted_direction = np.sign(predicted - prior)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "directional_accuracy": float(np.mean(actual_direction == predicted_direction)),
    }
