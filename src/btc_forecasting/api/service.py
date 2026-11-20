"""Application services for health, model readiness, and forecasts."""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import load_ohlcv_csv_through
from ..features import make_features
from ..metrics import evaluate_predictions
from ..models.lstm import LSTMForecaster
from ..models.xgboost_model import XGBoostForecaster
from ..selection import load_performance_registry, performance_for, select_model
from .schemas import (
    ForecastComparisonResponse,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    ModelForecastResult,
    ModelStatus,
    ReadinessResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "bitcoin_data.csv"
DEFAULT_RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw"
DEFAULT_RUNS_PATH = PROJECT_ROOT / "artifacts" / "runs"
SELECTION_REGISTRY_PATH = PROJECT_ROOT / "configs" / "model_selection.json"
FORECAST_CUTOFF = pd.Timestamp("2025-12-31")
DATA_ENV_VAR = "BTC_FORECASTING_DATA"
MODEL_RUN_ENV_VAR = "BTC_FORECASTING_MODEL_RUN"
_MODEL_CACHE: dict[tuple[str, int], object] = {}


class DataUnavailableError(RuntimeError):
    """The configured historical data cannot support a forecast."""


class ModelUnavailableError(RuntimeError):
    """The requested model is not implemented or its dependency is missing."""


class ForecastUnavailableError(RuntimeError):
    """An implemented model cannot produce a forecast for the current input."""


def data_path() -> Path:
    """Return the configured dataset path or the repository's current CSV.

    Relative paths in ``BTC_FORECASTING_DATA`` are anchored at the project
    root so the selected dataset does not depend on the server's working dir.
    """
    configured = os.environ.get(DATA_ENV_VAR)
    if not configured:
        snapshots = sorted(DEFAULT_RAW_DIRECTORY.glob("*.csv")) if DEFAULT_RAW_DIRECTORY.is_dir() else []
        return snapshots[-1] if snapshots else DEFAULT_DATA_PATH

    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _model_run_path(model_name: str) -> Path | None:
    configured = os.environ.get(MODEL_RUN_ENV_VAR)
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path
    if not DEFAULT_RUNS_PATH.is_dir():
        return None
    if SELECTION_REGISTRY_PATH.is_file():
        try:
            registry = json.loads(SELECTION_REGISTRY_PATH.read_text(encoding="utf-8"))
            source_runs = sorted(
                {
                    row.get("artifact_run", row.get("source_run"))
                    for row in registry.get("records", [])
                    if row.get("model") == model_name
                    and row.get("artifact_run", row.get("source_run"))
                }
            )
        except (OSError, json.JSONDecodeError, TypeError, KeyError):
            source_runs = []
        if source_runs:
            # Bind deployed weights to the exact runs whose validation scores
            # drive routing; never silently pair scores with a newer artifact.
            for run_id in source_runs:
                candidate = (DEFAULT_RUNS_PATH / run_id).resolve()
                try:
                    candidate.relative_to(DEFAULT_RUNS_PATH.resolve())
                    metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
                    relative_path = metadata["model_artifacts"][model_name]
                    artifact = (candidate / relative_path).resolve()
                    artifact.relative_to(candidate)
                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                if artifact.is_file():
                    return candidate
            return None
    candidates = sorted(
        (path for path in DEFAULT_RUNS_PATH.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in candidates:
        try:
            metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
            relative_path = metadata["model_artifacts"][model_name]
            model_path = (candidate / relative_path).resolve()
            model_path.relative_to(candidate.resolve())
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if model_path.is_file():
            return candidate
    return None


def _model_bundle(name: str) -> tuple[Path, dict]:
    run_path = _model_run_path(name)
    if run_path is None or not run_path.is_dir():
        raise ModelUnavailableError(
            f"No experiment run is available. Run the experiment and set {MODEL_RUN_ENV_VAR} if needed."
        )
    try:
        metadata = json.loads((run_path / "metadata.json").read_text(encoding="utf-8"))
        relative_path = metadata["model_artifacts"][name]
        model_path = (run_path / relative_path).resolve()
        model_path.relative_to(run_path.resolve())
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ModelUnavailableError(
            f"The selected experiment run has no valid {name} model bundle."
        ) from exc
    if not model_path.is_file():
        raise ModelUnavailableError(f"The saved {name} model bundle is missing: {model_path.name}")
    return model_path, metadata


def _cached_model(name: str, path: Path, metadata: dict):
    cache_key = (str(path), path.stat().st_mtime_ns)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    try:
        if name == "arima":
            with path.open("rb") as stream:
                model = pickle.load(stream)
        elif name == "xgboost":
            settings = metadata["model"]["xgboost"]
            model = XGBoostForecaster.load(
                path,
                feature_columns=metadata["feature_columns"],
                parameters=settings["parameters"],
            )
        else:
            model = LSTMForecaster.load(path)
    except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ModelUnavailableError(f"The saved {name} model could not be loaded: {exc}") from exc
    _MODEL_CACHE.clear()
    _MODEL_CACHE[cache_key] = model
    return model


def list_models() -> list[ModelStatus]:
    """Report implemented models separately from optional runtime readiness."""
    try:
        registry = _selection_registry()
        validated = {
            name: sorted(
                {
                    int(row["horizon_days"])
                    for row in registry["records"]
                    if row.get("model") == name
                }
            )
            for name in ("persistence", "arima", "xgboost", "lstm")
        }
    except (ModelUnavailableError, KeyError, TypeError, ValueError):
        validated = {name: [] for name in ("persistence", "arima", "xgboost", "lstm")}
    arima_ready = False
    if importlib.util.find_spec("statsmodels") is None:
        arima_detail = "Install the project's arima dependency group to enable ARIMA."
    else:
        try:
            _model_bundle("arima")
        except ModelUnavailableError as exc:
            arima_detail = str(exc)
        else:
            arima_ready = True
            arima_detail = (
                "Loads the saved final-fit ARIMA artifact ending 2025-12-31. "
                "Multi-step forecasts are available, but only one-day validation evidence is recorded."
            )

    ml_status: dict[str, tuple[bool, str]] = {}
    for name, dependencies in {
        "xgboost": ("xgboost", "sklearn"),
        "lstm": ("torch", "joblib", "sklearn"),
    }.items():
        missing_dependencies = [
            dependency
            for dependency in dependencies
            if importlib.util.find_spec(dependency) is None
        ]
        if missing_dependencies:
            ml_status[name] = (
                False,
                f"Install the project's {name} dependency group to enable serving.",
            )
            continue
        try:
            _model_bundle(name)
        except ModelUnavailableError as exc:
            ml_status[name] = (False, str(exc))
        else:
            ml_status[name] = (
                True,
                "Loads the saved final-fit bundle. One-day inference uses the production feature contract; "
                "recursive multi-day forecasts are experimental.",
            )

    return [
        ModelStatus(
            name="persistence",
            implemented=True,
            ready=True,
            detail="Repeats the latest observed close; available as a baseline.",
            validated_horizons=validated["persistence"],
        ),
        ModelStatus(
            name="arima",
            implemented=True,
            ready=arima_ready,
            detail=arima_detail,
            validated_horizons=validated["arima"],
        ),
        ModelStatus(
            name="xgboost",
            implemented=True,
            ready=ml_status["xgboost"][0],
            detail=ml_status["xgboost"][1],
            validated_horizons=validated["xgboost"],
        ),
        ModelStatus(
            name="lstm",
            implemented=True,
            ready=ml_status["lstm"][0],
            detail=ml_status["lstm"][1],
            validated_horizons=validated["lstm"],
        ),
    ]


def _saved_arima_order() -> tuple[int, int, int]:
    try:
        _, metadata = _model_bundle("arima")
        raw_order = metadata["model"]["arima"]["order"]
    except (ModelUnavailableError, OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ModelUnavailableError(
            "A final-fit ARIMA artifact with a valid order is unavailable."
        ) from exc

    if (
        not isinstance(raw_order, list)
        or len(raw_order) != 3
        or any(type(value) is not int or value < 0 for value in raw_order)
    ):
        raise ModelUnavailableError("The saved ARIMA model order must contain three non-negative integers.")
    return tuple(raw_order)  # type: ignore[return-value]


def _load_history() -> tuple[Path, pd.DataFrame]:
    """Load OHLCV and enforce the fixed user-forecast information boundary.

    The source snapshot may contain 2026 evaluation rows. Those rows are
    deliberately excluded from every user-facing forecast and market summary.
    """
    path = data_path()
    if not path.is_file():
        raise DataUnavailableError(
            f"Historical BTC data is unavailable. Set {DATA_ENV_VAR} to a CSV path."
        )
    try:
        history = load_ohlcv_csv_through(path, FORECAST_CUTOFF)
    except (OSError, ValueError, KeyError) as exc:
        raise DataUnavailableError(f"Historical BTC data could not be loaded: {exc}") from exc
    if history.empty or history.index[-1] != FORECAST_CUTOFF:
        raise DataUnavailableError(
            "Historical data must contain observations through the fixed forecast cutoff "
            f"{FORECAST_CUTOFF.date()}."
        )
    return path, history


def check_readiness() -> ReadinessResponse:
    """Confirm that a valid daily dataset is available to forecast from."""
    path, history = _load_history()
    return ReadinessResponse(
        status="ready",
        data_source=path.name,
        history_start=history.index[0].date(),
        history_end=history.index[-1].date(),
    )


def _selection_registry() -> dict:
    try:
        return load_performance_registry(SELECTION_REGISTRY_PATH)
    except ValueError as exc:
        raise ModelUnavailableError(str(exc)) from exc


def _feature_history_for(name: str, metadata: dict, history: pd.DataFrame) -> pd.DataFrame | None:
    if name not in {"xgboost", "lstm"}:
        return None
    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ModelUnavailableError("The saved run does not contain a valid model feature schema.")
    # Recreate features from the already cutoff-limited OHLCV history. This
    # prevents a processed feature file containing 2026 rows from entering an
    # inference path and keeps production features in sync with features.py.
    features = make_features(history, drop_incomplete=False)
    missing = [column for column in feature_columns if column not in features.columns]
    if missing:
        raise ModelUnavailableError(
            f"The production feature builder is missing saved model columns: {', '.join(missing)}"
        )
    selected = features.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna()
    if selected.empty or selected.index[-1] != history.index[-1]:
        raise DataUnavailableError("Causal model features do not reach the forecast cutoff date.")
    return selected


def _model_forecast(
    name: str,
    horizon_days: int,
    history: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Predict using a saved final-fit bundle, with no 2026 refitting."""
    close = history["Close"].astype(float)
    if name == "persistence":
        return np.repeat(float(close.iloc[-1]), horizon_days), None, None

    model_path, metadata = _model_bundle(name)
    model = _cached_model(name, model_path, metadata)
    if name == "arima":
        final_fit_end = pd.Timestamp(metadata["model"]["arima"]["final_fit_close_end"])
        if history.index[-1] < final_fit_end:
            raise DataUnavailableError(
                f"ARIMA artifact requires history through {final_fit_end.date()}."
            )
        row_labels = getattr(model.model.data, "row_labels", None)
        if row_labels is None or len(row_labels) == 0:
            raise ModelUnavailableError("Saved ARIMA artifact has no dated training history.")
        artifact_end = pd.Timestamp(row_labels[-1]).tz_localize(None).normalize()
        if artifact_end != final_fit_end:
            raise ModelUnavailableError(
                "Saved ARIMA artifact end date does not match its recorded final-fit boundary."
            )
        additions = close.loc[close.index > artifact_end]
        if not additions.empty:
            expected = pd.date_range(artifact_end + pd.Timedelta(days=1), additions.index[-1], freq="D")
            if not additions.index.equals(expected):
                raise DataUnavailableError("ARIMA state update history has missing calendar dates.")
            model = model.append(additions, refit=False)
        forecast = model.get_forecast(steps=horizon_days)
        values = np.asarray(forecast.predicted_mean, dtype=float).reshape(-1)
        interval = np.asarray(forecast.conf_int(alpha=0.05), dtype=float)
        return values, interval[:, 0], interval[:, 1]

    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ModelUnavailableError("The saved run does not contain a valid model feature schema.")
    sequence_length = (
        int(metadata["model"]["lstm"]["sequence_length"]) if name == "lstm" else 1
    )
    simulated_history = history.copy()
    values: list[float] = []
    try:
        for _ in range(horizon_days):
            feature_history = _feature_history_for(name, metadata, simulated_history)
            assert feature_history is not None
            if name == "xgboost":
                predicted_close = float(model.predict(feature_history.tail(1)).iloc[0])
            elif name == "lstm":
                if len(feature_history) < sequence_length:
                    raise ValueError("Feature history is shorter than the saved LSTM sequence length")
                sequence = feature_history.tail(sequence_length).to_numpy(dtype=np.float32)[None, :, :]
                predicted_log_return = float(model.predict(sequence).iloc[0])
                predicted_close = float(simulated_history["Close"].iloc[-1]) * np.exp(predicted_log_return)
            else:
                raise ModelUnavailableError(f"{name} is not an implemented forecasting model.")
            if not np.isfinite(predicted_close) or predicted_close <= 0:
                raise ValueError("model returned an invalid future close")
            values.append(predicted_close)
            previous_close = float(simulated_history["Close"].iloc[-1])
            next_date = simulated_history.index[-1] + pd.Timedelta(days=1)
            # Recursive ML horizons require unknown future OHLCV inputs. We
            # create a synthetic candle from the predicted close and carry the
            # latest known volume forward. This is explicitly experimental.
            synthetic = pd.DataFrame(
                {
                    "Open": [previous_close],
                    "High": [max(previous_close, predicted_close)],
                    "Low": [min(previous_close, predicted_close)],
                    "Close": [predicted_close],
                    "Volume": [float(simulated_history["Volume"].iloc[-1])],
                },
                index=pd.DatetimeIndex([next_date], name="Date"),
            )
            simulated_history = pd.concat([simulated_history, synthetic])
    except (ValueError, RuntimeError, KeyError, OverflowError, FloatingPointError) as exc:
        raise ForecastUnavailableError(f"{name} forecast failed: {exc}") from exc
    return np.asarray(values, dtype=float).reshape(-1), None, None


def _request_horizon(request: ForecastRequest) -> tuple[int, date]:
    """Resolve a date-first request against the fixed 2025-12-31 origin."""
    if request.target_date is not None:
        target = pd.Timestamp(request.target_date)
        if target <= FORECAST_CUTOFF:
            raise ValueError(
                f"Prediction date must be after the historical cutoff {FORECAST_CUTOFF.date()}."
            )
        horizon = int((target - FORECAST_CUTOFF).days)
        if request.horizon_days is not None and request.horizon_days != horizon:
            raise ValueError("target_date and horizon_days do not describe the same forecast date.")
        if horizon > 3650:
            raise ValueError("Prediction dates may be at most 10 years after the historical cutoff.")
        return horizon, target.date()
    horizon = request.horizon_days or 1
    if horizon > 3650:
        raise ValueError("Forecast horizon may not exceed 3650 days.")
    return horizon, (FORECAST_CUTOFF + pd.Timedelta(days=horizon)).date()


def _route_or_experimental_persistence(
    registry: dict, *, objective: str, horizon_days: int,
    available_models: set[str], manual_model: str | None = None,
):
    """Route from exact horizon evidence; use a disclosed baseline otherwise."""
    try:
        return select_model(
            registry, objective=objective, horizon_days=horizon_days,
            manual_model=manual_model, available_models=available_models,
        )
    except ValueError:
        if manual_model is not None:
            return select_model(
                registry, objective=objective, horizon_days=horizon_days,
                manual_model=manual_model,
            )
        from ..selection import ModelSelection
        return ModelSelection(
            objective=objective, horizon_days=horizon_days,
            selected_model="persistence", selection_mode="automatic_experimental_fallback",
            metric=None, score=None,
            reason=(f"No 2024 validation evidence exists for {objective} at {horizon_days} day(s). "
                    "Using the persistence baseline as an explicitly experimental fallback; "
                    "no learned model is claimed reliable for this horizon."),
            candidates=performance_for(registry, objective=objective, horizon_days=horizon_days),
        )


def _forecast_points(
    history_end: pd.Timestamp,
    values: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> list[ForecastPoint]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ForecastUnavailableError("A model returned empty, non-finite, or non-positive price forecasts.")
    if (lower is None) != (upper is None):
        raise ForecastUnavailableError("A model returned incomplete confidence intervals.")
    lower_values = None if lower is None else np.asarray(lower, dtype=float).reshape(-1)
    upper_values = None if upper is None else np.asarray(upper, dtype=float).reshape(-1)
    if lower_values is not None and (
        len(lower_values) != len(values)
        or len(upper_values) != len(values)
        or not np.isfinite(lower_values).all()
        or not np.isfinite(upper_values).all()
        or np.any(lower_values > upper_values)
    ):
        raise ForecastUnavailableError("A model returned invalid confidence intervals.")
    dates = pd.date_range(history_end + pd.Timedelta(days=1), periods=len(values), freq="D")
    return [
        ForecastPoint(
            date=dates[i].date(),
            predicted_close_usd=float(values[i]),
            lower_95_usd=float(lower_values[i]) if lower_values is not None else None,
            upper_95_usd=float(upper_values[i]) if upper_values is not None else None,
        )
        for i in range(len(values))
    ]


def performance_records(*, objective: str | None = None, horizon_days: int | None = None) -> dict:
    """Expose only the stored 2024 validation records and their provenance."""
    registry = _selection_registry()
    records = registry["records"]
    if objective is not None:
        records = [row for row in records if row["objective"] == objective]
    if horizon_days is not None:
        records = [row for row in records if row["horizon_days"] == horizon_days]
    return {
        "data_protocol": registry["data_protocol"],
        "records": records,
        "validated_horizons": sorted({row["horizon_days"] for row in records}),
        "ensemble": registry.get("ensemble"),
    }


def historical_market_data(*, days: int = 365) -> dict:
    """Return actual OHLCV-derived dashboard series from the configured CSV."""
    if not 1 <= days <= 3650:
        raise ValueError("days must be between 1 and 3650")
    _, history = _load_history()
    returns = history["Close"].pct_change()
    volatility = returns.rolling(30, min_periods=30).std() * np.sqrt(365)
    recent = history.tail(days)
    return {
        "source": data_path().name,
        "latest_date": recent.index[-1].date().isoformat(),
        "latest_close_usd": float(recent["Close"].iloc[-1]),
        "latest_volume": float(recent["Volume"].iloc[-1]),
        "latest_return": float(returns.loc[recent.index[-1]]) if pd.notna(returns.loc[recent.index[-1]]) else None,
        "latest_annualized_volatility": float(volatility.loc[recent.index[-1]]) if pd.notna(volatility.loc[recent.index[-1]]) else None,
        "points": [
            {
                "date": date.date().isoformat(),
                "close_usd": float(row.Close),
                "volume": float(row.Volume),
                "return_1d": float(returns.loc[date]) if pd.notna(returns.loc[date]) else None,
                "annualized_volatility_30d": float(volatility.loc[date]) if pd.notna(volatility.loc[date]) else None,
            }
            for date, row in recent.iterrows()
        ],
    }


def final_evaluation_report() -> dict:
    """Read saved 2026 artifacts for display only; never used by the router."""
    registry = _selection_registry()
    sources = registry.get("final_evaluation_artifacts", {}).get("sources", {})
    if not sources:
        return {"available": False, "reason": "No saved final evaluation artifact sources are configured."}
    frames: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    for name in ("arima", "xgboost", "lstm"):
        run_id = sources.get(name)
        run_path = (DEFAULT_RUNS_PATH / str(run_id)).resolve()
        try:
            run_path.relative_to(DEFAULT_RUNS_PATH.resolve())
            prediction = pd.read_csv(run_path / "predictions.csv")
            metrics = pd.read_csv(run_path / "metrics.csv")
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            return {"available": False, "reason": f"Could not load saved {name} final-evaluation records: {exc}"}
        required = {"Target_Date", "actual_next_close_usd", "previous_close_usd", "prediction_usd"}
        if not required.issubset(prediction.columns) or "model" not in metrics.columns:
            return {"available": False, "reason": f"Saved {name} run has an unexpected evaluation schema."}
        part = prediction.loc[:, ["Target_Date", "actual_next_close_usd", "previous_close_usd", "prediction_usd"]].copy()
        part = part.rename(
            columns={
                "prediction_usd": f"{name}_prediction_usd",
                "actual_next_close_usd": f"{name}_actual_usd",
                "previous_close_usd": f"{name}_previous_close_usd",
            }
        )
        frames.append(part)
        row = metrics.loc[metrics["model"] == name]
        if not row.empty:
            metric_rows.append({"model": name, **row.iloc[0].drop(labels="model").to_dict(), "source_run": run_id})
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="Target_Date", how="inner", validate="one_to_one")
    if merged.empty:
        return {"available": False, "reason": "Saved model runs have no common final-evaluation dates."}
    actual_columns = [column for column in merged if column.endswith("_actual_usd")]
    if any(not merged[actual_columns[0]].equals(merged[column]) for column in actual_columns[1:]):
        return {"available": False, "reason": "Saved model runs disagree on actual final-evaluation prices."}
    previous_columns = [column for column in merged if column.endswith("_previous_close_usd")]
    if any(not merged[previous_columns[0]].equals(merged[column]) for column in previous_columns[1:]):
        return {"available": False, "reason": "Saved model runs disagree on the final-evaluation forecast origins."}
    merged["actual_close_usd"] = merged[actual_columns[0]]
    baseline_metrics = evaluate_predictions(
        actual_next_close=merged["actual_close_usd"],
        predicted_next_close=merged[previous_columns[0]],
        previous_close=merged[previous_columns[0]],
    )
    metric_rows.append({"model": "persistence", **baseline_metrics, "source_run": "computed from saved final-evaluation rows"})
    prediction_columns = ["Target_Date", "actual_close_usd", "arima_prediction_usd", "xgboost_prediction_usd", "lstm_prediction_usd"]
    merged = merged.sort_values("Target_Date").tail(365)
    return {
        "available": True,
        "period": [str(merged["Target_Date"].iloc[0]), str(merged["Target_Date"].iloc[-1])],
        "rows": merged.loc[:, prediction_columns].to_dict(orient="records"),
        "metrics": metric_rows,
        "selection_uses_this_data": False,
    }


def model_explainability() -> dict:
    """Return saved model settings and diagnostics from training run artifacts."""
    registry = _selection_registry()
    models: dict[str, dict] = {}
    for name in ("arima", "xgboost", "lstm"):
        try:
            artifact_path, metadata = _model_bundle(name)
        except ModelUnavailableError as exc:
            models[name] = {"available": False, "reason": str(exc)}
            continue
        run_path = artifact_path.parent.parent
        model_metadata = metadata.get("model", {}).get(name, {})
        details: dict = {
            "available": True,
            "run_id": run_path.name,
            "parameters": model_metadata.get("parameters", model_metadata.get("order")),
            "feature_columns": metadata.get("feature_columns", []) if name != "arima" else [],
        }
        diagnostic_files = {
            "arima": ("stationarity_tests.csv", "fitted_coefficients.csv"),
            "xgboost": ("feature_importance.csv",),
            "lstm": ("training_loss_per_epoch.csv",),
        }
        for filename in diagnostic_files[name]:
            path = run_path / filename
            if path.is_file():
                try:
                    details[filename.removesuffix(".csv")] = pd.read_csv(path).to_dict(orient="records")
                except (OSError, pd.errors.ParserError):
                    details[filename.removesuffix(".csv")] = []
        if name == "arima":
            residual_plot = run_path / "2026_residual_diagnostics.png"
            if residual_plot.is_file():
                details["residual_diagnostics_image"] = "/api/v1/diagnostics/arima/residuals"
        models[name] = details
    return {
        "validation_period": registry["data_protocol"]["validation_targets"],
        "models": models,
    }


def diagnostic_image_path(model_name: str, image_name: str) -> Path:
    """Resolve a whitelisted saved diagnostic image from a selected artifact run."""
    allowed = {("arima", "residuals"): "2026_residual_diagnostics.png"}
    filename = allowed.get((model_name, image_name))
    if filename is None:
        raise ModelUnavailableError("No saved image is registered for that model diagnostic.")
    artifact_path, _ = _model_bundle(model_name)
    run_path = artifact_path.parent.parent
    image_path = (run_path / filename).resolve()
    try:
        image_path.relative_to(run_path.resolve())
    except ValueError as exc:
        raise ModelUnavailableError("The diagnostic image path is invalid.") from exc
    if not image_path.is_file():
        raise ModelUnavailableError("The requested saved diagnostic image is unavailable.")
    return image_path


def create_forecast(request: ForecastRequest) -> ForecastResponse:
    """Forecast with manual choice or validation-only automatic routing."""
    horizon_days, requested_date = _request_horizon(request)
    path, history = _load_history()
    registry = _selection_registry()
    statuses = {status.name: status for status in list_models()}
    if request.model == "auto":
        try:
            choice = select_model(
                registry,
                objective=request.objective,
                horizon_days=horizon_days,
                available_models={name for name, status in statuses.items() if status.ready},
            )
        except ValueError:
            choice = _route_or_experimental_persistence(
                registry, objective=request.objective, horizon_days=horizon_days,
                available_models={name for name, status in statuses.items() if status.ready},
            )
    else:
        if not statuses[request.model].ready:
            raise ModelUnavailableError(statuses[request.model].detail)
        choice = select_model(
            registry,
            objective=request.objective,
            horizon_days=horizon_days,
            manual_model=request.model,
        )

    values, lower, upper = _model_forecast(choice.selected_model, horizon_days, history)
    points = _forecast_points(history.index[-1], values, lower, upper)
    return ForecastResponse(
        symbol=request.symbol,
        model=choice.selected_model,
        horizon_days=horizon_days,
        requested_date=requested_date,
        experimental_horizon=horizon_days not in {
            int(row["horizon_days"]) for row in registry["records"]
            if row.get("model") == choice.selected_model and row.get("objective") == request.objective
        },
        generated_at_utc=datetime.now(timezone.utc),
        history_start=history.index[0].date(),
        history_end=history.index[-1].date(),
        last_observed_close_usd=float(history["Close"].iloc[-1]),
        data_source=path.name,
        forecasts=points,
        selection=choice.to_dict(),
    )


def compare_forecasts(request: ForecastRequest) -> ForecastComparisonResponse:
    """Return same-origin forecasts for every ready model and the baseline."""
    horizon_days, requested_date = _request_horizon(request)
    _, history = _load_history()
    registry = _selection_registry()
    objective_records = performance_for(
        registry, objective=request.objective, horizon_days=horizon_days
    )
    metric_by_model = {record["model"]: record for record in objective_records}
    statuses = {status.name: status for status in list_models()}
    result_rows: list[ModelForecastResult] = []
    forecastable: set[str] = set()

    for name in ("persistence", "arima", "xgboost", "lstm"):
        status = statuses[name]
        validation = metric_by_model.get(name)
        if not status.ready:
            result_rows.append(
                ModelForecastResult(
                    model=name,
                    ready=False,
                    supported=False,
                    detail=status.detail,
                    validation_metrics=validation,
                )
            )
            continue
        try:
            values, lower, upper = _model_forecast(name, horizon_days, history)
            points = _forecast_points(history.index[-1], values, lower, upper)
        except (ModelUnavailableError, DataUnavailableError, ForecastUnavailableError) as exc:
            result_rows.append(
                ModelForecastResult(
                    model=name,
                    ready=status.ready,
                    supported=False,
                    detail=str(exc),
                    validation_metrics=validation,
                    experimental_horizon=horizon_days not in status.validated_horizons,
                )
            )
            continue
        forecastable.add(name)
        result_rows.append(
            ModelForecastResult(
                model=name,
                ready=True,
                supported=True,
                detail=(
                    "Recursive ML forecast uses synthetic future candles and carries the last known "
                    "volume forward; this horizon is experimental."
                    if name in {"xgboost", "lstm"} and horizon_days > 1
                    else "Forecast generated from the fixed 2025-12-31 history cutoff."
                ),
                forecasts=points,
                validation_metrics=validation,
                experimental_horizon=horizon_days not in status.validated_horizons,
            )
        )

    choice = None
    if request.model == "auto":
        try:
            choice = select_model(
                registry,
                objective=request.objective,
                horizon_days=horizon_days,
                available_models=forecastable,
            )
        except ValueError:
            choice = _route_or_experimental_persistence(
                registry, objective=request.objective, horizon_days=horizon_days,
                available_models=forecastable,
            )
    else:
        if request.model in forecastable:
            choice = select_model(
                registry,
                objective=request.objective,
                horizon_days=horizon_days,
                manual_model=request.model,
            )
        else:
            choice = None

    ensemble = registry.get("ensemble", {})
    return ForecastComparisonResponse(
        symbol=request.symbol,
        objective=request.objective,
        horizon_days=horizon_days,
        requested_date=requested_date,
        experimental_horizon=(
            True if choice is None else horizon_days not in {
                int(row["horizon_days"]) for row in registry["records"]
                if row.get("model") == choice.selected_model and row.get("objective") == request.objective
            }
        ),
        history_end=history.index[-1].date(),
        selected_model=choice.selected_model if choice else None,
        selection=choice.to_dict() if choice else None,
        models=result_rows,
        combined_forecast=None,
        combined_forecast_reason=ensemble.get(
            "reason", "No ensemble has passed validation-only evaluation."
        ),
    )
