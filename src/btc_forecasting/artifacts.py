"""Persist experiment predictions, metrics, and run metadata."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _json_default(value: object) -> object:
    """Convert common path, NumPy, and date values for JSON metadata."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def save_experiment_artifacts(
    output_root: str | Path,
    *,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    metadata: Mapping[str, Any],
    validation_scores: pd.DataFrame | None = None,
) -> Path:
    """Write one complete run to a new, timestamped directory.

    Files are written in a temporary directory and renamed into place only
    after all outputs have been serialized, avoiding a visible half-written
    run if a write fails.
    """
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise ValueError("predictions must be a non-empty pandas DataFrame")
    if not isinstance(predictions.index, pd.DatetimeIndex):
        raise ValueError("predictions must use a DatetimeIndex")
    if predictions.index.has_duplicates or not predictions.index.is_monotonic_increasing:
        raise ValueError("prediction dates must be unique and chronological")
    if not isinstance(metrics, pd.DataFrame) or metrics.empty:
        raise ValueError("metrics must be a non-empty pandas DataFrame")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    if validation_scores is not None and (
        not isinstance(validation_scores, pd.DataFrame) or validation_scores.empty
    ):
        raise ValueError("validation_scores must be a non-empty DataFrame when provided")

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = root / run_id
    run_metadata = {
        **metadata,
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with tempfile.TemporaryDirectory(prefix=f".{run_id}-", dir=root) as temporary:
        temporary_path = Path(temporary)
        staging_path = temporary_path / "run"
        staging_path.mkdir()
        predictions.to_csv(
            staging_path / "predictions.csv",
            index_label="Date",
            date_format="%Y-%m-%d",
        )
        metrics.to_csv(staging_path / "metrics.csv", index_label="model")
        if validation_scores is not None:
            validation_scores.to_csv(
                staging_path / "validation_scores.csv", index_label="candidate"
            )
        (staging_path / "metadata.json").write_text(
            json.dumps(
                run_metadata,
                indent=2,
                sort_keys=True,
                default=_json_default,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        staging_path.rename(destination)

    return destination
