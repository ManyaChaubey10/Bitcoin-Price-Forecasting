# Experiment scripts

## ARIMA and XGBoost versus persistence

Use Python `>=3.11,<3.13`, as declared in `pyproject.toml`. From the repository
root, install the optional ARIMA and XGBoost dependencies and run:

```powershell
python -m pip install -e ".[arima,xgboost]"
python scripts/run_experiment.py
```

The runner reads `data/bitcoin_features.csv` by default. Use `--data` to supply
another feature CSV, `--config` to select a YAML configuration, or
`--output-root` to change where run artifacts are stored. Relative paths are
resolved from the repository root.

The validation period selects the number of boosting rounds using the
configured MAE or RMSE metric. ARIMA order is selected by AIC using training
closes only. Both models are evaluated on the same held-out target dates against
the persistence baseline. The feature row at date *t* predicts the target close
at *t+1*; partitioning uses the target date so a boundary label cannot cross
into the next split. Predictions remain indexed by their decision date. XGBoost
is refit on the training and validation target dates; ARIMA is fit through the
last validation close. During test, parameters stay fixed and ARIMA appends
each newly observed close after forecasting it.

Each run writes a new directory under `artifacts/runs/` containing predictions,
test metrics, validation candidate scores, the effective configuration, the
input CSV SHA-256, and software versions. The generated artifacts are ignored
by Git.

This runner compares ARIMA and XGBoost with the persistence baseline. LSTM will
be added after its sequence and training-only scaling behavior follows the same
split and prediction contract.
