"""Leakage-conscious XGBoost adapter for next-close regression."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import TARGET_COLUMNS


class XGBoostForecaster:
    """Fit an XGBoost regressor on dated, numeric, causal feature columns.

    The adapter deliberately does not select columns from a full feature table:
    callers must pass the feature frame and target series separately. Any
    ``Target_*`` column in the feature frame is rejected to guard against
    accidental target leakage.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 500,
        max_depth: int = 5,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        if n_estimators < 1 or max_depth < 1:
            raise ValueError("n_estimators and max_depth must be positive")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 < subsample <= 1 or not 0 < colsample_bytree <= 1:
            raise ValueError("subsample and colsample_bytree must be in (0, 1]")

        self.parameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
        }
        self._model = None
        self._feature_columns: tuple[str, ...] | None = None

    @staticmethod
    def _validate_features(features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame) or features.empty:
            raise ValueError("features must be a non-empty pandas DataFrame")
        if not features.columns.is_unique:
            raise ValueError("feature names must be unique")
        if not all(isinstance(column, str) for column in features.columns):
            raise ValueError("feature names must be strings")
        leaked = [
            str(column)
            for column in features.columns
            if column in TARGET_COLUMNS or str(column).startswith("Target_")
        ]
        if leaked:
            raise ValueError(f"Target columns cannot be model features: {', '.join(leaked)}")
        if not isinstance(features.index, pd.DatetimeIndex):
            raise ValueError("features must use a DatetimeIndex")
        if features.index.has_duplicates or not features.index.is_monotonic_increasing:
            raise ValueError("feature dates must be unique and chronological")
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
            raise ValueError("all model feature columns must be numeric")

        values = features.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("features must contain finite, non-missing values")
        return features

    def fit(self, features: pd.DataFrame, target: pd.Series) -> XGBoostForecaster:
        """Fit the model; feature and target indexes must match exactly."""
        x = self._validate_features(features)
        if not isinstance(target, pd.Series) or target.empty:
            raise ValueError("target must be a non-empty pandas Series")
        if not target.index.equals(x.index):
            raise ValueError("feature and target indexes must match exactly")

        y = pd.to_numeric(target, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("target must contain finite, non-missing values")

        from xgboost import XGBRegressor

        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=-1,
            **self.parameters,
        )
        model.fit(x, y)
        self._model = model
        self._feature_columns = tuple(str(column) for column in x.columns)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict next close for each row, retaining the row's decision date."""
        if self._model is None or self._feature_columns is None:
            raise RuntimeError("fit must be called before predict")

        x = self._validate_features(features)
        expected = set(self._feature_columns)
        provided = set(str(column) for column in x.columns)
        if provided != expected:
            missing = sorted(expected - provided)
            extra = sorted(provided - expected)
            raise ValueError(f"prediction features differ from training; missing={missing}, extra={extra}")

        ordered = x.loc[:, list(self._feature_columns)]
        values = np.asarray(self._model.predict(ordered), dtype=float).reshape(-1)
        if len(values) != len(ordered) or not np.isfinite(values).all():
            raise ValueError("XGBoost returned invalid predictions")
        return pd.Series(values, index=ordered.index, name="prediction")
