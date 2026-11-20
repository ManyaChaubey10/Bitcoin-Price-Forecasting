from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from btc_forecasting.api.main import app
from btc_forecasting.api.schemas import ForecastRequest
from btc_forecasting.api.service import (
    FORECAST_CUTOFF,
    _load_history,
    _request_horizon,
    compare_forecasts,
    historical_market_data,
)
from btc_forecasting.data import load_ohlcv_csv_through
from btc_forecasting.features import make_features
from btc_forecasting.selection import load_performance_registry

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


def test_target_date_uses_fixed_cutoff_and_rejects_past_cutoff():
    request = ForecastRequest(target_date=date(2026, 1, 7))
    assert _request_horizon(request) == (7, date(2026, 1, 7))
    with pytest.raises(ValueError, match="after the fixed history cutoff"):
        ForecastRequest(target_date=date(2025, 12, 31))


def test_inference_history_and_dashboard_exclude_2026_actuals():
    _, history = _load_history()
    assert history.index[-1] == FORECAST_CUTOFF
    assert historical_market_data(days=3)["latest_date"] == "2025-12-31"


def test_features_at_cutoff_are_invariant_to_later_rows():
    cutoff = load_ohlcv_csv_through(
        ROOT / "data/raw/BTC-USD_20180101_20261114_20261115T172122Z.csv", FORECAST_CUTOFF
    )
    synthetic_future = pd.DataFrame(
        [{"Open": 90000.0, "High": 95000.0, "Low": 89000.0, "Close": 94000.0, "Volume": 1.0}],
        index=pd.DatetimeIndex([FORECAST_CUTOFF + pd.Timedelta(days=1)], name="Date"),
    )
    full_features = make_features(pd.concat([cutoff, synthetic_future]), drop_incomplete=False)
    prefix_features = make_features(cutoff, drop_incomplete=False)
    pd.testing.assert_series_equal(
        full_features.loc[FORECAST_CUTOFF].drop(labels=[
            "Target_Next_Close", "Target_Next_Return", "Target_Direction"
        ]),
        prefix_features.loc[FORECAST_CUTOFF].drop(labels=[
            "Target_Next_Close", "Target_Next_Return", "Target_Direction"
        ]),
        check_names=False,
    )


def test_cutoff_loader_stops_before_parsing_later_price_values(tmp_path):
    source = tmp_path / "prices.csv"
    source.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2025-12-31,10,11,9,10,100\n"
        "2026-01-01,not-a-price,not-a-price,not-a-price,not-a-price,not-a-volume\n",
        encoding="utf-8",
    )
    result = load_ohlcv_csv_through(source, FORECAST_CUTOFF)
    assert result.index[-1] == FORECAST_CUTOFF
    assert result["Close"].iloc[-1] == 10


def test_saved_artifacts_produce_cutoff_origin_forecast():
    result = compare_forecasts(ForecastRequest(target_date=date(2026, 1, 1)))
    assert result.history_end == date(2025, 12, 31)
    assert result.requested_date == date(2026, 1, 1)
    assert result.horizon_days == 1
    assert {row.model for row in result.models if row.supported} == {
        "persistence", "arima", "xgboost", "lstm"
    }
    assert all(row.forecasts[-1].date == date(2026, 1, 1) for row in result.models if row.supported)


def test_unvalidated_horizon_is_labeled_experimental_and_uses_baseline_router():
    registry = load_performance_registry(ROOT / "configs/model_selection.json")
    result = compare_forecasts(ForecastRequest(target_date=date(2026, 1, 7)))
    assert result.experimental_horizon is True
    assert result.selected_model == "persistence"
    assert "experimental fallback" in result.selection["reason"]
    assert all(row.experimental_horizon for row in result.models if row.supported)
    assert registry["data_protocol"]["routing_uses_final_evaluation"] is False


def test_http_api_serves_history_and_date_forecast():
    history = client.get("/api/v1/history?days=2")
    assert history.status_code == 200
    assert history.json()["latest_date"] == "2025-12-31"
    response = client.post(
        "/api/v1/forecast/compare",
        json={"target_date": "2026-01-01", "model": "auto", "objective": "price_rmse"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["history_end"] == "2025-12-31"
    assert body["requested_date"] == "2026-01-01"
    assert body["horizon_days"] == 1
    assert body["selected_model"] is not None
