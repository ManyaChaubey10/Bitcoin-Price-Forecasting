"""Validation-only model routing for objective and horizon-specific forecasts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

OBJECTIVE_METRICS = {
    "price_mae": ("mae", "min"),
    "price_rmse": ("rmse", "min"),
    "directional_accuracy": ("directional_accuracy", "max"),
}
MODEL_NAMES = {"persistence", "arima", "xgboost", "lstm"}


@dataclass(frozen=True)
class ModelSelection:
    objective: str
    horizon_days: int
    selected_model: str
    selection_mode: str
    metric: str | None
    score: float | None
    reason: str
    candidates: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_performance_registry(path: str | Path) -> dict[str, Any]:
    """Load the explicit, auditable record of validation-only scores."""
    try:
        registry = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load validation performance registry: {exc}") from exc
    if registry.get("schema_version") != 1 or not isinstance(registry.get("records"), list):
        raise ValueError("Validation performance registry has an unsupported schema")
    return registry


def performance_for(
    registry: dict[str, Any], *, objective: str, horizon_days: int
) -> list[dict[str, Any]]:
    """Return only records matching the requested objective and horizon."""
    if objective not in OBJECTIVE_METRICS:
        raise ValueError(f"Unsupported forecasting objective: {objective}")
    return [
        record.copy()
        for record in registry["records"]
        if record.get("objective") == objective
        and record.get("horizon_days") == horizon_days
    ]


def select_model(
    registry: dict[str, Any],
    *,
    objective: str,
    horizon_days: int,
    manual_model: str | None = None,
    available_models: set[str] | None = None,
) -> ModelSelection:
    """Choose a model from 2024 validation records, or honor manual choice.

    A manual choice is permitted without validation evidence, but the response
    states that the requested objective/horizon has not been validated.
    """
    if objective not in OBJECTIVE_METRICS:
        raise ValueError(f"Unsupported forecasting objective: {objective}")
    if type(horizon_days) is not int or horizon_days < 1:
        raise ValueError("horizon_days must be a positive integer")
    if manual_model is not None and manual_model not in MODEL_NAMES:
        raise ValueError(f"Unsupported model: {manual_model}")

    metric, direction = OBJECTIVE_METRICS[objective]
    records = performance_for(registry, objective=objective, horizon_days=horizon_days)
    records.sort(key=lambda row: row["model"])

    if manual_model is not None:
        match = next((row for row in records if row["model"] == manual_model), None)
        if match is None:
            reason = (
                f"Manual selection: {manual_model} was requested. No 2024 validation "
                f"record exists for {objective} at {horizon_days} day(s), so this choice "
                "is not validation-ranked for this request."
            )
            return ModelSelection(
                objective, horizon_days, manual_model, "manual", None, None, reason, records
            )
        return ModelSelection(
            objective,
            horizon_days,
            manual_model,
            "manual",
            metric,
            float(match[metric]),
            f"Manual selection: {manual_model} was requested. Its 2024 validation "
            f"{metric} for {objective} at {horizon_days} day(s) was {match[metric]:.6g}.",
            records,
        )

    if available_models is not None:
        records = [row for row in records if row["model"] in available_models]
    if not records:
        raise ValueError(
            f"No available 2024 validation record for {objective} at {horizon_days} day(s)."
        )

    ranked = sorted(records, key=lambda row: row[metric], reverse=(direction == "max"))
    winner = ranked[0]
    baseline = next((row for row in records if row["model"] == "persistence"), None)
    if (
        winner["model"] != "persistence"
        and baseline is not None
        and not winner.get("reliable_vs_persistence", False)
    ):
        return ModelSelection(
            objective,
            horizon_days,
            "persistence",
            "automatic",
            metric,
            float(baseline[metric]),
            f"Selected persistence because no learned model has validation evidence that its "
            f"{metric} improvement over persistence is reliable. {winner['model']} had the "
            f"best point estimate ({winner[metric]:.6g} versus {baseline[metric]:.6g}), "
            "but paired validation uncertainty has not been established.",
            records,
        )
    if baseline is not None and winner["model"] != "persistence":
        baseline_score = float(baseline[metric])
        win_score = float(winner[metric])
        if direction == "min":
            change = 100 * (baseline_score - win_score) / baseline_score if baseline_score else 0.0
            comparison = f"{change:.3f}% lower than persistence"
        else:
            change = 100 * (win_score - baseline_score)
            comparison = f"{change:.3f} percentage points above persistence"
    else:
        comparison = "the top score among the available validation records"

    reason = (
        f"Selected {winner['model']} because it had the best 2024 validation "
        f"{metric} ({winner[metric]:.6g}) for {objective} at {horizon_days} day(s); "
        f"this was {comparison}. Only validation records are used for routing."
    )
    return ModelSelection(
        objective,
        horizon_days,
        winner["model"],
        "automatic",
        metric,
        float(winner[metric]),
        reason,
        records,
    )
