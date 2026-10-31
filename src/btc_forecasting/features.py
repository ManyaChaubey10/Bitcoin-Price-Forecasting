"""Create causal technical features and next-day supervised targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import validate_ohlcv

TARGET_COLUMNS = ("Target_Next_Close", "Target_Next_Return", "Target_Direction")


def make_features(
    raw_ohlcv: pd.DataFrame,
    *,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Build features from current/past candles and targets from the next candle.

    The returned row dated ``t`` contains only information available after
    candle ``t`` closes. Its target columns refer to candle ``t+1``. The last
    row has no next-day target and is removed when ``drop_incomplete`` is true.
    """
    source = validate_ohlcv(raw_ohlcv)
    df = source.copy()

    # Returns and lagged values use no observations later than the row date.
    df["Return_1d"] = df["Close"].pct_change()
    for horizon in (3, 7, 14, 30):
        df[f"Return_{horizon}d"] = df["Close"].pct_change(horizon)
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    for lag in (1, 2, 3, 5, 7, 14, 30):
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
        df[f"Return_Lag_{lag}"] = df["Return_1d"].shift(lag)
        df[f"Volume_Lag_{lag}"] = df["Volume"].shift(lag)

    for window in (7, 14, 30, 50, 200):
        df[f"SMA_{window}"] = df["Close"].rolling(window=window).mean()
        df[f"EMA_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
    df["SMA_7_50_Ratio"] = df["SMA_7"] / df["SMA_50"]
    df["SMA_50_200_Ratio"] = df["SMA_50"] / df["SMA_200"]
    df["Close_EMA_14_Ratio"] = df["Close"] / df["EMA_14"]

    for window in (7, 14, 30):
        df[f"Vol_{window}d"] = df["Return_1d"].rolling(window).std()
    df["Annual_Vol_30d"] = df["Vol_30d"] * np.sqrt(365)

    previous_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR_14"] = true_range.rolling(14).mean()

    df["BB_Middle"] = df["Close"].rolling(20).mean()
    df["BB_Std"] = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Middle"] - 2 * df["BB_Std"]
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    band_width = df["BB_Upper"] - df["BB_Lower"]
    df["BB_PctB"] = (df["Close"] - df["BB_Lower"]) / band_width

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df.loc[(loss == 0) & (gain > 0), "RSI_14"] = 100
    df.loc[(loss == 0) & (gain == 0), "RSI_14"] = 50

    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    low_14 = df["Low"].rolling(14).min()
    high_14 = df["High"].rolling(14).max()
    stochastic_range = high_14 - low_14
    df["Stoch_K"] = 100 * (df["Close"] - low_14) / stochastic_range
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    df["Volume_SMA_7"] = df["Volume"].rolling(7).mean()
    df["Volume_SMA_30"] = df["Volume"].rolling(30).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_7"]
    df["OBV"] = (np.sign(delta) * df["Volume"]).fillna(0).cumsum()

    df["High_Low_Pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["Upper_Shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Close"]
    df["Lower_Shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Close"]
    df["Body_Pct"] = (df["Close"] - df["Open"]).abs() / df["Close"]

    day_of_week = df.index.dayofweek
    month = df.index.month
    df["Day_of_Week"] = day_of_week
    df["Day_of_Month"] = df.index.day
    df["Month"] = month
    df["Quarter"] = df.index.quarter
    df["Is_Weekend"] = (day_of_week >= 5).astype(int)
    df["Sin_DayOfWeek"] = np.sin(2 * np.pi * day_of_week / 7.0)
    df["Cos_DayOfWeek"] = np.cos(2 * np.pi * day_of_week / 7.0)
    df["Sin_Month"] = np.sin(2 * np.pi * month / 12.0)
    df["Cos_Month"] = np.cos(2 * np.pi * month / 12.0)

    df["Target_Next_Close"] = df["Close"].shift(-1)
    df["Target_Next_Return"] = df["Close"].pct_change().shift(-1)
    df["Target_Direction"] = (df["Target_Next_Return"] > 0).astype("int8")

    # Undefined ratios (for example a flat stochastic window) are missing data,
    # not infinity; remove those rows by default along with indicator warm-up.
    df = df.replace([np.inf, -np.inf], np.nan)
    if drop_incomplete:
        df = df.dropna().copy()
    return df
