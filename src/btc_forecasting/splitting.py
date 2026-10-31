"""Chronological train, validation, and final-test partitions."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def split_by_date(
    frame: pd.DataFrame,
    *,
    train_end: str,
    validation_start: str,
    validation_end: str,
    test_start: str,
    test_end: str,
    date_column: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Split rows into non-overlapping, inclusive calendar-date ranges.

    ``date_column=None`` uses the DataFrame index. No shuffling or imputation
    occurs. Missing dates remain missing and should be investigated upstream.
    """
    boundaries = {
        "train_end": pd.Timestamp(train_end),
        "validation_start": pd.Timestamp(validation_start),
        "validation_end": pd.Timestamp(validation_end),
        "test_start": pd.Timestamp(test_start),
        "test_end": pd.Timestamp(test_end),
    }
    if not (
        boundaries["train_end"] < boundaries["validation_start"]
        <= boundaries["validation_end"]
        < boundaries["test_start"]
        <= boundaries["test_end"]
    ):
        raise ValueError("Train, validation, and test date boundaries overlap or are out of order")

    data = frame.copy()
    if date_column is None:
        dates = pd.to_datetime(data.index, errors="raise", utc=True).tz_convert(None).normalize()
    else:
        if date_column not in data.columns:
            raise ValueError(f"Date column {date_column!r} is missing")
        dates = pd.to_datetime(data[date_column], errors="raise", utc=True).dt.tz_convert(None).dt.normalize()

    if dates.hasnans:
        raise ValueError("Dates cannot contain missing values")
    if not dates.is_monotonic_increasing:
        raise ValueError("Rows must be chronological before splitting")
    if dates.has_duplicates:
        raise ValueError("Rows contain duplicate dates")

    train_mask = dates <= boundaries["train_end"]
    validation_mask = dates.between(
        boundaries["validation_start"], boundaries["validation_end"], inclusive="both"
    )
    test_mask = dates.between(boundaries["test_start"], boundaries["test_end"], inclusive="both")
    partitions = {
        "train": data.loc[train_mask].copy(),
        "validation": data.loc[validation_mask].copy(),
        "test": data.loc[test_mask].copy(),
    }
    empty = [name for name, part in partitions.items() if part.empty]
    if empty:
        raise ValueError(f"Date split produced empty partition(s): {', '.join(empty)}")
    return partitions


def split_from_config(
    frame: pd.DataFrame,
    config: Mapping,
    *,
    date_column: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Apply the ``split`` block from the project YAML configuration."""
    try:
        boundaries = config["split"]
        return split_by_date(frame, date_column=date_column, **boundaries)
    except (KeyError, TypeError) as exc:
        raise ValueError("Config must contain valid split date boundaries") from exc
