import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import json

# Ensure stdout flushes immediately
sys.stdout.reconfigure(line_buffering=True)

# Ensure directories exist
os.makedirs('plots', exist_ok=True)
os.makedirs('arima', exist_ok=True)

# 1. Load Data
data_path = 'bitcoin_data.csv' if os.path.exists('bitcoin_data.csv') else 'data/bitcoin_data.csv'
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Filter data till 31st Dec 2025
df_filtered = df[df['Date'] <= '2025-12-31'].copy()
df_filtered.set_index('Date', inplace=True)
price_series = df_filtered['Close'].asfreq('D')

# Fill any missing calendar dates if any (forward fill)
price_series = price_series.ffill()

print(f"Data range for ARIMA training: {price_series.index.min().strftime('%Y-%m-%d')} to {price_series.index.max().strftime('%Y-%m-%d')}", flush=True)
print(f"Total daily observations: {len(price_series)}", flush=True)

# 2. Stationarity Tests
adf_raw = adfuller(price_series.dropna())
diff_series = price_series.diff().dropna()
adf_diff = adfuller(diff_series)

print("\n--- Stationarity Test Results ---", flush=True)
print(f"ADF Statistic (Raw Close): {adf_raw[0]:.4f}, p-value: {adf_raw[1]:.4f}", flush=True)
print(f"ADF Statistic (1st Difference d=1): {adf_diff[0]:.4f}, p-value: {adf_diff[1]:.4f}", flush=True)

# Plot ACF & PACF
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes[0, 0].plot(price_series, color='#1f77b4', linewidth=1.5)
axes[0, 0].set_title("Bitcoin Daily Close Price (Till Dec 31, 2025)", fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel("USD ($)")

plot_acf(price_series.dropna(), ax=axes[0, 1], lags=40, color='#1f77b4', title="ACF - Raw Price")

axes[1, 0].plot(diff_series, color='#ff7f0e', linewidth=1)
axes[1, 0].set_title("1st Difference (d=1)", fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel("Daily Change ($)")

plot_pacf(diff_series.dropna(), ax=axes[1, 1], lags=40, color='#ff7f0e', title="PACF - 1st Difference (d=1)")

plt.tight_layout()
plt.savefig('plots/arima_acf_pacf.png', dpi=300)
plt.close()
print("Saved plots/arima_acf_pacf.png", flush=True)

# 3. Auto ARIMA Model Selection
print("\n--- Running Auto-ARIMA Model Selection ---", flush=True)
auto_model = pm.auto_arima(
    price_series,
    start_p=0, start_q=0,
    max_p=5, max_q=5, d=1,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore',
    trace=True
)

order = auto_model.order
print(f"\nBest ARIMA Order Selected: {order}", flush=True)
print(auto_model.summary(), flush=True)

# 4. Train / Test Out-of-Sample Backtesting (Train: up to 2024-12-31, Test: 2025-01-01 to 2025-12-31)
train_series = price_series[:'2024-12-31']
test_series = price_series['2025-01-01':'2025-12-31']

print(f"\nBacktesting Split - Train: {len(train_series)} days, Test (2025): {len(test_series)} days", flush=True)

# Fast 1-step rolling prediction using statsmodels append
fitted_train = ARIMA(train_series, order=order).fit()
res_test = fitted_train.append(test_series, refit=False)
predictions_series = res_test.fittedvalues[test_series.index]

# Evaluation Metrics
rmse = np.sqrt(mean_squared_error(test_series, predictions_series))
mae = mean_absolute_error(test_series, predictions_series)
mape = mean_absolute_percentage_error(test_series, predictions_series) * 100
r2 = r2_score(test_series, predictions_series)

# Directional Accuracy
act_dir = np.sign(test_series.diff().dropna())
pred_dir = np.sign(predictions_series.diff().dropna())
dir_acc = np.mean(act_dir == pred_dir) * 100

print("\n--- 2025 Out-of-Sample Backtesting Metrics ---", flush=True)
print(f"RMSE: ${rmse:,.2f}", flush=True)
print(f"MAE:  ${mae:,.2f}", flush=True)
print(f"MAPE: {mape:.2f}%", flush=True)
print(f"R^2 Score: {r2:.4f}", flush=True)
print(f"Directional Accuracy: {dir_acc:.2f}%", flush=True)

# Plot 2025 Test Predictions vs Actuals
plt.figure(figsize=(14, 6))
plt.plot(train_series['2024-06-01':].index, train_series['2024-06-01':], label='Train Price (H2 2024)', color='#2ca02c', linewidth=1.5)
plt.plot(test_series.index, test_series, label='Actual Price (2025)', color='#1f77b4', linewidth=2)
plt.plot(test_series.index, predictions_series, label=f'ARIMA{order} 1-Step Forecast (2025)', color='#d62728', linestyle='--', linewidth=1.8)
plt.title(f"Bitcoin ARIMA{order} Model Evaluation - 2025 Test Set", fontsize=14, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/arima_test_evaluation.png', dpi=300)
plt.close()
print("Saved plots/arima_test_evaluation.png", flush=True)

# 5. Fit Final Model on Full Dataset (Till 31st Dec 2025)
print("\n--- Fitting Final ARIMA Model on Data Till 31st Dec 2025 ---", flush=True)
final_model = ARIMA(price_series, order=order).fit()
print(final_model.summary(), flush=True)

# Multi-step future forecast (90 days into 2026)
forecast_steps = 90
forecast_res = final_model.get_forecast(steps=forecast_steps)
forecast_mean = forecast_res.predicted_mean
forecast_ci = forecast_res.conf_int(alpha=0.05)

future_dates = pd.date_range(start=price_series.index.max() + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
forecast_mean.index = future_dates
forecast_ci.index = future_dates

# Save Residual Diagnostics Plot
fig = final_model.plot_diagnostics(figsize=(14, 8))
plt.tight_layout()
plt.savefig('plots/arima_residual_diagnostics.png', dpi=300)
plt.close()
print("Saved plots/arima_residual_diagnostics.png", flush=True)

# Plot Final Historical Data + Future Forecast
plt.figure(figsize=(14, 7))
plot_start = '2024-01-01'
plt.plot(price_series[plot_start:].index, price_series[plot_start:], label='Historical BTC Close Price (Till Dec 31, 2025)', color='#1f77b4', linewidth=2)
plt.plot(final_model.fittedvalues[plot_start:].index, final_model.fittedvalues[plot_start:], label=f'Fitted ARIMA{order} Model', color='#ff7f0e', alpha=0.7, linestyle=':')
plt.plot(forecast_mean.index, forecast_mean, label=f'90-Day Out-of-Sample Forecast (2026)', color='#d62728', linewidth=2.5, linestyle='--')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='#d62728', alpha=0.2, label='95% Confidence Interval')

plt.title(f"Bitcoin (BTC-USD) ARIMA{order} Model Forecast (Trained Till Dec 31, 2025)", fontsize=14, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/arima_forecast_2025_dec.png', dpi=300)
plt.close()
print("Saved plots/arima_forecast_2025_dec.png", flush=True)

# Save summary metrics and parameters to JSON artifact
metrics_summary = {
    "model_order": list(order),
    "train_end_date": "2025-12-31",
    "total_observations": int(len(price_series)),
    "adf_p_value_raw": float(adf_raw[1]),
    "adf_p_value_diff": float(adf_diff[1]),
    "aic": float(final_model.aic),
    "bic": float(final_model.bic),
    "test_2025_metrics": {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
        "directional_accuracy_pct": float(dir_acc)
    },
    "forecast_90d_end_date": future_dates[-1].strftime('%Y-%m-%d'),
    "forecast_90d_end_price": float(forecast_mean.iloc[-1]),
    "forecast_90d_lower_ci": float(forecast_ci.iloc[-1, 0]),
    "forecast_90d_upper_ci": float(forecast_ci.iloc[-1, 1])
}

with open('arima/arima_metrics.json', 'w') as f:
    json.dump(metrics_summary, f, indent=4)
print("Saved arima/arima_metrics.json", flush=True)
