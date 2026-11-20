"""Selection policy tests use synthetic records without fitting models."""

import unittest

from btc_forecasting.selection import select_model


def registry():
    records = []
    for model, rmse, direction, reliability in (
        ("persistence", 10.0, 0.0, True),
        ("arima", 9.0, 0.50, False),
        ("xgboost", 11.0, 0.48, False),
    ):
        records.append(
            {
                "model": model,
                "objective": "price_rmse",
                "horizon_days": 1,
                "rmse": rmse,
                "directional_accuracy": direction,
                "reliable_vs_persistence": reliability,
            }
        )
    records.extend(
        {**row, "objective": "directional_accuracy"}
        for row in list(records)
    )
    return {"schema_version": 1, "records": records}


class SelectionTests(unittest.TestCase):
    def test_auto_uses_persistence_without_reliable_validation_win(self):
        choice = select_model(registry(), objective="price_rmse", horizon_days=1)
        self.assertEqual(choice.selected_model, "persistence")
        self.assertIn("reliable", choice.reason)

    def test_direction_router_does_not_claim_an_unverified_win(self):
        choice = select_model(
            registry(), objective="directional_accuracy", horizon_days=1
        )
        self.assertEqual(choice.selected_model, "persistence")
        self.assertIn("paired validation uncertainty", choice.reason)

    def test_manual_choice_is_honored_and_explained(self):
        choice = select_model(
            registry(), objective="price_rmse", horizon_days=1, manual_model="xgboost"
        )
        self.assertEqual(choice.selected_model, "xgboost")
        self.assertEqual(choice.selection_mode, "manual")
        self.assertIn("Manual selection", choice.reason)

    def test_auto_rejects_horizon_without_validation_record(self):
        with self.assertRaisesRegex(ValueError, "No available 2024 validation record"):
            select_model(registry(), objective="price_rmse", horizon_days=7)

    def test_auto_filters_unavailable_models(self):
        choice = select_model(
            registry(),
            objective="price_rmse",
            horizon_days=1,
            available_models={"arima", "xgboost"},
        )
        self.assertEqual(choice.selected_model, "arima")


if __name__ == "__main__":
    unittest.main()
