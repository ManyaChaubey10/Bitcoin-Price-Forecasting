# Architecture and implementation status

## Design

The project separates ingestion, feature generation, model training, evaluation, routing, serving, and presentation. Existing notebooks, model outputs, and source datasets remain in place. The API loads saved model bundles; application startup does not retrain them.

```text
OHLCV CSV -> data.py -> features.py -> chronological experiment
                                      |-> ARIMA adapter
                                      |-> XGBoost adapter
                                      |-> LSTM adapter
                                      |-> persistence baseline
                    validation artifacts -> selection.py -> FastAPI -> React/Vite
                    final 2026 artifacts ---------------------> read-only display
```

### Components

- `src/btc_forecasting/data.py` validates dated OHLCV inputs.
- `src/btc_forecasting/features.py` builds lagged/rolling/technical inputs and the next-day supervised targets.
- `src/btc_forecasting/splitting.py` creates chronological date partitions; `metrics.py` and `evaluation.py` score aligned predictions.
- `src/btc_forecasting/models/` contains independent ARIMA, XGBoost, and LSTM implementations.
- `scripts/run_experiment.py` selects settings on validation data, refits selected models through the configured 2025 boundary, and saves a unique run.
- `configs/model_selection.json` records the router's validation evidence and source run provenance.
- `src/btc_forecasting/selection.py` ranks only matching validation records by objective and horizon, honors manual choices, and falls back to persistence when a learned-model gain has not been established as reliable.
- `src/btc_forecasting/api/` serves forecasts, performance, history, diagnostics, and saved final-evaluation results.
- `frontend/` is the React/Vite dashboard and communicates with FastAPI through the Vite `/api` proxy.
- `artifacts/runs/` contains ignored generated run outputs and existing fitted model bundles. The registry pins model serving to specific artifact runs.

## Data and evaluation contract

| Partition | Target-date window | Use |
|---|---|---|
| Training | 2018-01-01–2023-12-31 | Fit candidate configurations |
| Validation | 2024-01-01–2024-12-31 | Hyperparameter/model selection and routing evidence |
| Final fit | Through 2025-12-31 | Refit the selected configuration |
| Final evaluation | 2026 actuals available in the saved dataset | Out-of-sample reporting only |

The configured 2026 evaluation window ends November 14, 2026. No 2026 result is included in the router registry. Rolling one-day evaluation means each target is forecast from the immediately preceding observed close; after that close becomes available, it may update forecast state for the next origin. A fixed 365-day forecast instead uses one origin and predicts the entire year without observing intervening closes. These strategies answer different questions.

The saved serving artifacts are final-fit models through 2025. ARIMA updates its stored state with observed closes after its fit boundary without refitting parameters. XGBoost and LSTM use the saved feature contract for one-day predictions. Their future covariates are not synthesized for longer horizons.

## Current evidence

- The registry has validation scores for one-day persistence, ARIMA, XGBoost, and LSTM only.
- Existing XGBoost and LSTM runs use the existing shared feature table; model-specific ablation has not been run. The application exposes the actual saved features and diagnostics but does not claim they are optimized.
- No validation results exist for 7-, 14-, or 30-day objectives. The router refuses automatic selection for those horizons. ARIMA and persistence can still produce manual comparison forecasts; the UI marks the absence of validation evidence.
- For every one-day objective, the current registry does not establish learned-model gains as reliable against persistence. Automatic routing therefore uses the persistence fallback; this does not imply persistence forecasts direction well.
- No 2024 per-date prediction set for all models is saved. Simple/weighted ensembles have not been validated and remain disabled.
- Existing 2026 results are available for display only. The API labels them as final evaluation and the routing code does not read them.

## Implementation plan and open work

1. **Feature specialization:** create separate causal feature groups and controlled train/2024-validation ablations for ARIMA (price vs log-return, candidate orders, justified exogenous inputs), XGBoost (price/return, volume, volatility, momentum/technical groups), and LSTM (input groups, lookback, training-only scaling, regularization, early stopping). Store per-date validation predictions and selected settings in unique run folders.
2. **Horizon study:** evaluate direct, date-aligned 1/7/14/30-day targets with the same training and validation boundaries; define each model's forecast-origin strategy and score every model against persistence. Do not extrapolate scores between horizons.
3. **Router refresh:** after reviewing those actual 2024 results, replace/add registry records with feature/config IDs and reliability evidence. Refit selected configurations through 2025 into new unique artifact bundles; never overwrite the existing bundles.
4. **Ensemble check:** calculate average and validation-weighted combinations only from aligned 2024 per-date predictions; include persistence and compare on identical dates. Enable only if it improves validation under a predeclared criterion.
5. **Verification:** automated API/frontend integration checks, repeatability checks on immutable inputs, and a final untouched 2026 report. Avoid tuning from that final report.

This document distinguishes implemented plumbing from the still-required model-specific research. The current app is usable for saved one-day model comparisons, but the full model-specialized, multi-horizon research objective is not yet complete.
