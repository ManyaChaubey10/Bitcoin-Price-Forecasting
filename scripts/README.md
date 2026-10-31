# Experiment scripts

## XGBoost versus persistence

Use Python `>=3.11,<3.13`, as declared in `pyproject.toml`. From the repository
root, install the optional XGBoost dependencies and run:

```powershell
python -m pip install -e ".[xgboost]"
python scripts/run_experiment.py
```

The runner reads `data/bitcoin_features.csv` by default. Use `--data` to supply
another feature CSV, `--config` to select a YAML configuration, or
`--output-root` to change where run artifacts are stored. Relative paths are
resolved from the repository root.

The validation period selects the number of boosting rounds using the
configured MAE or RMSE metric. The selected model is refit on the training and
validation target dates, then evaluated once on the final test target dates.
The feature row at date *t* predicts the target close at *t+1*; partitioning
uses the target date so a boundary label cannot cross into the next split.
Predictions remain indexed by their decision date. Test-time model parameters
are fixed, and each row uses only features available after that day's close.

Each run writes a new directory under `artifacts/runs/` containing predictions,
test metrics, validation candidate scores, the effective configuration, the
input CSV SHA-256, and software versions. The generated artifacts are ignored
by Git.

This initial runner compares XGBoost with the persistence baseline. ARIMA and
LSTM will be added after their experiment adapters follow the same split and
prediction contract.
