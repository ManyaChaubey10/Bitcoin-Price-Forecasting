"""Shared comparison tools for one-step-ahead model forecasts."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .metrics import evaluate_predictions


def compare_predictions(
    actual_next_close: pd.Series,
    previous_close: pd.Series,
    predictions: Mapping[str, pd.Series],
) -> pd.DataFrame:
    """Score named forecasts on exactly the same chronological dates.

    Forecasts must be indexed by their decision date (the date on which the
    prior close is known). The function rejects mismatched indexes instead of
    silently intersecting them, so model scores cannot use different test
    windows by accident. Include the persistence baseline in ``predictions``
    to compare it under the same contract.
    """
    if not isinstance(actual_next_close, pd.Series) or actual_next_close.empty:
        raise ValueError("actual_next_close must be a non-empty pandas Series")
    if not isinstance(previous_close, pd.Series) or previous_close.empty:
        raise ValueError("previous_close must be a non-empty pandas Series")
    if not isinstance(actual_next_close.index, pd.DatetimeIndex):
        raise ValueError("evaluation inputs must use a DatetimeIndex")
    if (
        actual_next_close.index.has_duplicates
        or not actual_next_close.index.is_monotonic_increasing
    ):
        raise ValueError("evaluation dates must be unique and chronological")
    if not actual_next_close.index.equals(previous_close.index):
        raise ValueError("actual and previous-close indexes must match exactly")
    if not predictions:
        raise ValueError("provide at least one named model prediction")

    scores: dict[str, dict[str, float]] = {}
    for model_name, forecast in predictions.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model names must be non-empty strings")
        if not isinstance(forecast, pd.Series):
            raise ValueError(f"predictions for {model_name!r} must be a pandas Series")
        if not forecast.index.equals(actual_next_close.index):
            raise ValueError(
                f"prediction dates for {model_name!r} do not match the evaluation dates"
            )
        scores[model_name] = evaluate_predictions(
            actual_next_close=actual_next_close,
            predicted_next_close=forecast,
            previous_close=previous_close,
        )

    result = pd.DataFrame.from_dict(scores, orient="index")
    result.index.name = "model"
    return result
