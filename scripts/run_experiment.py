"""Select ARIMA, XGBoost, and LSTM settings and compare on a held-out test year."""

from __future__ import annotations

import argparse
import hashlib
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from btc_forecasting.artifacts import save_experiment_artifacts
from btc_forecasting.evaluation import compare_predictions
from btc_forecasting.metrics import evaluate_predictions, persistence_predictions
from btc_forecasting.models.arima import RollingARIMAForecaster
from btc_forecasting.models.lstm import LSTMForecaster
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


def _build_lstm_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_column: str,
    lookback: int,
) -> tuple[Any, pd.Index, pd.Series]:
    """Build causal lookback windows ending on each row's decision date."""
    import numpy as np

    if type(lookback) is not int or lookback < 1:
        raise ValueError("lstm.lookback must be a positive integer")
    if len(frame) < lookback:
        raise ValueError("Feature history is shorter than the configured LSTM lookback")

    feature_values = frame[feature_columns].to_numpy(dtype=np.float32)
    close = pd.to_numeric(frame["Close"], errors="raise").to_numpy(dtype=float)
    next_close = pd.to_numeric(frame[target_column], errors="raise").to_numpy(dtype=float)
    if (close <= 0).any():
        raise ValueError("LSTM log-return targets require positive close prices")
    valid_positions = np.flatnonzero(
        np.isfinite(next_close) & (np.arange(len(frame)) >= lookback - 1)
    )
    if not len(valid_positions):
        raise ValueError("No complete LSTM target sequences are available")
    if (next_close[valid_positions] <= 0).any():
        raise ValueError("LSTM log-return targets require positive next-close prices")

    sequences = np.stack(
        [feature_values[end - lookback + 1 : end + 1] for end in valid_positions]
    )
    index = frame.index[valid_positions]
    log_return = np.log(next_close[valid_positions] / close[valid_positions])
    target = pd.Series(log_return, index=index, name="Target_Log_Return")
    return sequences, index, target


def run_experiment(config_path: Path, data_override: Path | None, output_root: Path) -> Path:
    """Select settings on validation and compare all models on held-out test data."""
    config = _load_config(config_path)
    try:
        horizon_days = int(config["target"]["horizon_days"])
        target_column = str(config["target"]["column"])
        data_path_value = data_override or _project_path(config["data"]["features_file"])
        xgb_config = config["xgboost"]
        candidate_estimators = xgb_config["n_estimators_candidates"]
        selection_metric = xgb_config["selection_metric"]
        arima_config = config["arima"]
        lstm_config = config["lstm"]
        candidate_epochs = lstm_config["epochs_candidates"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Config must define data.features_file, target, XGBoost, ARIMA, and LSTM settings"
        ) from exc

    if not isinstance(candidate_estimators, list) or not candidate_estimators:
        raise ValueError("xgboost.n_estimators_candidates must be a non-empty list")
    if any(type(value) is not int or value < 1 for value in candidate_estimators):
        raise ValueError("XGBoost estimator candidates must be positive integers")
    if len(set(candidate_estimators)) != len(candidate_estimators):
        raise ValueError("XGBoost estimator candidates must be unique")
    if selection_metric not in {"mae", "rmse"}:
        raise ValueError("xgboost.selection_metric must be 'mae' or 'rmse'")
    if not isinstance(candidate_epochs, list) or not candidate_epochs:
        raise ValueError("lstm.epochs_candidates must be a non-empty list")
    if any(type(value) is not int or value < 1 for value in candidate_epochs):
        raise ValueError("LSTM epoch candidates must be positive integers")
    if len(set(candidate_epochs)) != len(candidate_epochs):
        raise ValueError("LSTM epoch candidates must be unique")
    lstm_selection_metric = lstm_config.get("selection_metric", selection_metric)
    if lstm_selection_metric not in {"mae", "rmse"}:
        raise ValueError("lstm.selection_metric must be 'mae' or 'rmse'")

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

    validation_rows: list[dict[str, Any]] = []
    best_score: float | None = None
    best_estimators: int | None = None
    best_xgb_validation_metrics: dict[str, float] | None = None
    for n_estimators in candidate_estimators:
        model = XGBoostForecaster(n_estimators=n_estimators, **common_params)
        model.fit(train[feature_columns], train[target_column])
        prediction = model.predict(validation[feature_columns])
        scores = evaluate_predictions(
            actual_next_close=validation[target_column],
            predicted_next_close=prediction,
            previous_close=validation["Close"],
        )
        validation_rows.append(
            {
                "candidate": f"xgboost_n_estimators_{n_estimators}",
                "model": "xgboost",
                "candidate_value": n_estimators,
                **scores,
            }
        )
        score = scores[selection_metric]
        if best_score is None or score < best_score:
            best_score = score
            best_estimators = n_estimators
            best_xgb_validation_metrics = scores

    if best_estimators is None or best_xgb_validation_metrics is None:
        raise RuntimeError("Validation did not select an XGBoost candidate")

    final_fit_end = pd.Timestamp(config["final_fit"]["target_end"])
    final_fit = frame.loc[
        frame["Target_Date"].le(final_fit_end) & frame[target_column].notna()
    ].copy()
    if final_fit.empty or final_fit["Target_Date"].max() != final_fit_end:
        raise ValueError(
            f"Final-fit labels must be available through {final_fit_end.date()}"
        )
    final_model = XGBoostForecaster(n_estimators=best_estimators, **common_params)
    final_model.fit(final_fit[feature_columns], final_fit[target_column])
    xgb_prediction = final_model.predict(test[feature_columns])

    # Sequence windows end on their decision date. Validation and test windows
    # may use earlier observed feature rows as context, while their labels stay
    # in their respective target-date partitions.
    lookback = int(lstm_config["lookback"])
    lstm_sequences, sequence_index, lstm_targets = _build_lstm_sequences(
        frame,
        feature_columns,
        target_column=target_column,
        lookback=lookback,
    )
    sequence_positions = pd.Series(
        range(len(sequence_index)), index=sequence_index, dtype="int64"
    )
    train_sequence_index = train.index.intersection(sequence_index, sort=False)
    validation_sequence_index = validation.index.intersection(sequence_index, sort=False)
    test_sequence_index = test.index.intersection(sequence_index, sort=False)
    if not len(train_sequence_index) or not len(validation_sequence_index) or not len(test_sequence_index):
        raise ValueError("LSTM sequence construction left an empty date partition")
    if not test_sequence_index.equals(test.index):
        raise ValueError("LSTM sequence dates do not cover the complete shared test window")
    train_sequence_positions = sequence_positions.loc[train_sequence_index].to_numpy()
    validation_sequence_positions = sequence_positions.loc[validation_sequence_index].to_numpy()
    test_sequence_positions = sequence_positions.loc[test_sequence_index].to_numpy()
    train_log_returns = lstm_targets.loc[train_sequence_index]
    validation_log_returns = lstm_targets.loc[validation_sequence_index]

    lstm_common_parameters = {
        key: lstm_config[key]
        for key in (
            "hidden_size",
            "num_layers",
            "dropout",
            "learning_rate",
            "batch_size",
            "random_state",
        )
        if key in lstm_config
    }
    best_lstm_score: float | None = None
    best_epochs: int | None = None
    best_lstm_validation_metrics: dict[str, float] | None = None
    for epochs in candidate_epochs:
        model = LSTMForecaster(epochs=epochs, **lstm_common_parameters)
        model.fit(lstm_sequences[train_sequence_positions], train_log_returns)
        predicted_log_return = model.predict(
            lstm_sequences[validation_sequence_positions],
            index=validation_sequence_index,
        )
        predicted_close = validation.loc[validation_sequence_index, "Close"] * np.exp(
            predicted_log_return
        )
        scores = evaluate_predictions(
            actual_next_close=validation.loc[validation_sequence_index, target_column],
            predicted_next_close=predicted_close,
            previous_close=validation.loc[validation_sequence_index, "Close"],
        )
        validation_rows.append(
            {
                "candidate": f"lstm_epochs_{epochs}",
                "model": "lstm",
                "candidate_value": epochs,
                **scores,
            }
        )
        score = scores[lstm_selection_metric]
        if best_lstm_score is None or score < best_lstm_score:
            best_lstm_score = score
            best_epochs = epochs
            best_lstm_validation_metrics = scores

    if best_epochs is None or best_lstm_validation_metrics is None:
        raise RuntimeError("Validation did not select an LSTM candidate")

    final_fit_sequence_index = final_fit.index.intersection(sequence_index, sort=False)
    final_fit_sequence_positions = sequence_positions.loc[final_fit_sequence_index].to_numpy()
    final_fit_log_returns = lstm_targets.loc[final_fit_sequence_index]
    final_lstm = LSTMForecaster(epochs=best_epochs, **lstm_common_parameters)
    final_lstm.fit(
        lstm_sequences[final_fit_sequence_positions],
        final_fit_log_returns,
    )
    lstm_test_prediction = final_lstm.predict(
        lstm_sequences[test_sequence_positions],
        index=test_sequence_index,
    )
    lstm_test_prediction = test.loc[test_sequence_index, "Close"] * np.exp(
        lstm_test_prediction
    )

    # Final-fit parameters use labels through 2025-12-31. The first 2026
    # one-day forecast is made after the 2025-12-31 close is observed.
    final_fit_close = pd.to_numeric(
        frame.loc[frame.index <= final_fit_end, "Close"], errors="raise"
    )
    arima_test_prediction = _rolling_arima_predictions(
        final_fit_close,
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
            "lstm": lstm_test_prediction,
        },
    )

    predictions = pd.DataFrame(
        {
            "actual_next_close_usd": test[target_column],
            "previous_close_usd": test["Close"],
            "persistence_prediction_usd": baseline_prediction,
            "arima_prediction_usd": arima_test_prediction,
            "xgboost_prediction_usd": xgb_prediction,
            "lstm_prediction_usd": lstm_test_prediction,
        },
        index=test.index,
    )
    predictions.index.name = "Date"
    validation_scores = pd.DataFrame(validation_rows).set_index("candidate")

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
        "validation_metrics": {
            "arima": arima_validation_metrics,
            "xgboost_selected": best_xgb_validation_metrics,
            "lstm_selected": best_lstm_validation_metrics,
        },
        "model": {
            "xgboost": {
                "selected_n_estimators": best_estimators,
                "selection_metric": selection_metric,
                "fit_dates": {
                    "first_decision_date": final_fit.index[0],
                    "last_decision_date": final_fit.index[-1],
                },
                "fit_target_end": final_fit_end,
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
            "lstm": {
                "selected_epochs": best_epochs,
                "selection_metric": lstm_selection_metric,
                "sequence_length": lookback,
                "fit_target_end": final_fit_end,
                "target_transform": "predict log(next_close / current_close), then convert to USD close",
                "scaling": "feature and target scalers fit on final-fit sequences dated through the configured final-fit target end",
                "refit_during_test": False,
                "parameters": {**lstm_common_parameters, "epochs": best_epochs},
            },
        },
        "final_fit": {
            "fit_after_validation_selection": True,
            "target_date_range": {
                "first": final_fit["Target_Date"].min(),
                "last": final_fit["Target_Date"].max(),
            },
            "target_end": final_fit_end,
            "excludes_final_evaluation_rows": True,
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "PyYAML": _package_version("PyYAML"),
            "xgboost": _package_version("xgboost"),
            "statsmodels": _package_version("statsmodels"),
            "pmdarima": _package_version("pmdarima"),
            "scikit-learn": _package_version("scikit-learn"),
            "torch": _package_version("torch"),
        },
    }
    return save_experiment_artifacts(
        output_root,
        predictions=predictions,
        metrics=metrics,
        metadata=metadata,
        validation_scores=validation_scores,
        models={"xgboost": final_model, "lstm": final_lstm},
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
