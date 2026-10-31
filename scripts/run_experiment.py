"""Tune XGBoost on validation data and compare it with persistence on test data."""

from __future__ import annotations

import argparse
import hashlib
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from btc_forecasting.artifacts import save_experiment_artifacts
from btc_forecasting.evaluation import compare_predictions
from btc_forecasting.metrics import evaluate_predictions, persistence_predictions
from btc_forecasting.models.arima import RollingARIMAForecaster
from btc_forecasting.models.xgboost_model import XGBoostForecaster
from btc_forecasting.splitting import split_from_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Experiment configuration must be a YAML mapping: {path}")
    return config


def _load_feature_table(path: Path, *, horizon_days: int) -> pd.DataFrame:
    if horizon_days != 1:
        raise ValueError("This experiment runner currently supports a one-day-ahead target only")
    if not path.is_file():
        raise FileNotFoundError(f"Feature dataset does not exist: {path}")

    frame = pd.read_csv(path)
    if "Date" not in frame.columns:
        raise ValueError("Feature dataset must contain a Date column")
    dates = pd.to_datetime(frame.pop("Date"), errors="raise", utc=True)
    frame.index = dates.dt.tz_convert(None).dt.normalize()
    frame.index.name = "Date"
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Feature dates must be unique and chronological")
    if len(frame) > 1 and not (
        frame.index.to_series().diff().dropna() == pd.Timedelta(days=1)
    ).all():
        raise ValueError("Feature data must contain one row per calendar day")
    frame["Target_Date"] = frame.index + pd.Timedelta(days=horizon_days)
    return frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _select_arima_order(
    training_close: pd.Series,
    settings: dict[str, Any],
) -> tuple[tuple[int, int, int], float]:
    """Select ARIMA order by AIC using training-period closes only."""
    from pmdarima import auto_arima

    selected = auto_arima(
        training_close.asfreq("D"),
        start_p=int(settings["start_p"]),
        start_q=int(settings["start_q"]),
        max_p=int(settings["max_p"]),
        max_q=int(settings["max_q"]),
        d=int(settings["d"]),
        seasonal=bool(settings["seasonal"]),
        stepwise=bool(settings["stepwise"]),
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )
    order = tuple(int(value) for value in selected.order)
    return order, float(selected.aic())


def _rolling_arima_predictions(
    history: pd.Series,
    rows: pd.DataFrame,
    *,
    order: tuple[int, int, int],
    target_column: str,
) -> pd.Series:
    """Predict each target in order, then update state with its actual close."""
    model = RollingARIMAForecaster(order).fit(history)
    values: list[float] = []
    for _, row in rows.iterrows():
        values.append(model.forecast_next())
        model.observe(row["Target_Date"], float(row[target_column]))
    return pd.Series(values, index=rows.index, name="arima_prediction")


def run_experiment(config_path: Path, data_override: Path | None, output_root: Path) -> Path:
    """Run validation selection and a final held-out XGBoost comparison."""
    config = _load_config(config_path)
    try:
        horizon_days = int(config["target"]["horizon_days"])
        target_column = str(config["target"]["column"])
        data_path_value = data_override or _project_path(config["data"]["features_file"])
        xgb_config = config["xgboost"]
        candidate_estimators = xgb_config["n_estimators_candidates"]
        selection_metric = xgb_config["selection_metric"]
        arima_config = config["arima"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Config must define data.features_file, target, XGBoost, and ARIMA settings"
        ) from exc

    if not isinstance(candidate_estimators, list) or not candidate_estimators:
        raise ValueError("xgboost.n_estimators_candidates must be a non-empty list")
    if any(type(value) is not int or value < 1 for value in candidate_estimators):
        raise ValueError("XGBoost estimator candidates must be positive integers")
    if len(set(candidate_estimators)) != len(candidate_estimators):
        raise ValueError("XGBoost estimator candidates must be unique")
    if selection_metric not in {"mae", "rmse"}:
        raise ValueError("xgboost.selection_metric must be 'mae' or 'rmse'")

    frame = _load_feature_table(data_path_value, horizon_days=horizon_days)
    if target_column not in frame.columns or "Close" not in frame.columns:
        raise ValueError("Feature dataset must include the configured target and Close columns")
    partitions = split_from_config(frame, config, date_column="Target_Date")

    feature_columns = [
        column
        for column in frame.columns
        if not str(column).startswith("Target_")
    ]
    if not feature_columns:
        raise ValueError("No feature columns remain after excluding target columns")

    train = partitions["train"]
    validation = partitions["validation"]
    test = partitions["test"]

    # Order selection sees training-period closes only. Each target-date
    # partition is forecast sequentially with actual closes appended afterward.
    train_end = pd.Timestamp(config["split"]["train_end"])
    validation_end = pd.Timestamp(config["split"]["validation_end"])
    training_close = pd.to_numeric(
        frame.loc[frame.index <= train_end, "Close"], errors="raise"
    )
    arima_order, arima_training_aic = _select_arima_order(training_close, arima_config)
    arima_validation_prediction = _rolling_arima_predictions(
        training_close,
        validation,
        order=arima_order,
        target_column=target_column,
    )
    arima_validation_metrics = evaluate_predictions(
        actual_next_close=validation[target_column],
        predicted_next_close=arima_validation_prediction,
        previous_close=validation["Close"],
    )

    common_params = {
        key: xgb_config[key]
        for key in (
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "random_state",
        )
        if key in xgb_config
    }

    validation_rows: list[dict[str, float | int]] = []
    best_score: float | None = None
    best_estimators: int | None = None
    for n_estimators in candidate_estimators:
        model = XGBoostForecaster(n_estimators=n_estimators, **common_params)
        model.fit(train[feature_columns], train[target_column])
        prediction = model.predict(validation[feature_columns])
        scores = evaluate_predictions(
            actual_next_close=validation[target_column],
            predicted_next_close=prediction,
            previous_close=validation["Close"],
        )
        validation_rows.append({"n_estimators": n_estimators, **scores})
        score = scores[selection_metric]
        if best_score is None or score < best_score:
            best_score = score
            best_estimators = n_estimators

    if best_estimators is None:
        raise RuntimeError("Validation did not select an XGBoost candidate")

    development = pd.concat([train, validation]).sort_index()
    final_model = XGBoostForecaster(n_estimators=best_estimators, **common_params)
    final_model.fit(development[feature_columns], development[target_column])
    xgb_prediction = final_model.predict(test[feature_columns])

    # The first test forecast is made after the 2024-12-31 close is known.
    # That close is included in ARIMA history, but its 2025-01-01 target is not.
    development_close = pd.to_numeric(
        frame.loc[frame.index <= validation_end, "Close"], errors="raise"
    )
    arima_test_prediction = _rolling_arima_predictions(
        development_close,
        test,
        order=arima_order,
        target_column=target_column,
    )
    baseline_prediction = persistence_predictions(test["Close"])
    metrics = compare_predictions(
        actual_next_close=test[target_column],
        previous_close=test["Close"],
        predictions={
            "persistence": baseline_prediction,
            "arima": arima_test_prediction,
            "xgboost": xgb_prediction,
        },
    )

    predictions = pd.DataFrame(
        {
            "actual_next_close_usd": test[target_column],
            "previous_close_usd": test["Close"],
            "persistence_prediction_usd": baseline_prediction,
            "arima_prediction_usd": arima_test_prediction,
            "xgboost_prediction_usd": xgb_prediction,
        },
        index=test.index,
    )
    predictions.index.name = "Date"
    validation_scores = pd.DataFrame(validation_rows).set_index("n_estimators")

    split_config = config["split"]
    metadata = {
        "config": config,
        "config_file": str(config_path),
        "data_file": str(data_path_value),
        "data_sha256": _sha256(data_path_value),
        "target_column": target_column,
        "horizon_days": horizon_days,
        "prediction_index": "decision date; target date is next calendar day",
        "split_ranges_by_target_date": split_config,
        "rows": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "feature_columns": feature_columns,
        "validation_metrics": {"arima": arima_validation_metrics},
        "model": {
            "xgboost": {
                "selected_n_estimators": best_estimators,
                "selection_metric": selection_metric,
                "fit_dates": {
                    "first_decision_date": development.index[0],
                    "last_decision_date": development.index[-1],
                },
                "refit_during_test": False,
                "parameters": {**common_params, "n_estimators": best_estimators},
            },
            "arima": {
                "order": arima_order,
                "order_selection": "AIC on training closes only",
                "training_aic": arima_training_aic,
                "refit_during_test": False,
                "state_updates": "append observed close after each one-step forecast",
                "settings": arima_config,
            },
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "PyYAML": _package_version("PyYAML"),
            "xgboost": _package_version("xgboost"),
            "statsmodels": _package_version("statsmodels"),
            "pmdarima": _package_version("pmdarima"),
        },
    }
    return save_experiment_artifacts(
        output_root,
        predictions=predictions,
        metrics=metrics,
        metadata=metadata,
        validation_scores=validation_scores,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, help="Override the configured feature CSV path")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "runs",
    )
    args = parser.parse_args()
    config_path = _project_path(args.config)
    data_override = _project_path(args.data) if args.data is not None else None
    output_root = _project_path(args.output_root)

    run_path = run_experiment(config_path, data_override, output_root)
    print(f"Experiment artifacts saved to: {run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
