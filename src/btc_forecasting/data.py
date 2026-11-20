"""Download, load, validate, and preserve daily OHLCV data."""

from __future__ import annotations

import csv
from pathlib import Path
import os

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def validate_ohlcv(
    frame: pd.DataFrame,
    *,
    require_daily: bool = True,
) -> pd.DataFrame:
    """Return canonical OHLCV indexed by naive UTC dates, raising on bad input.

    Crypto trades every calendar day, so daily continuity is required by
    default. Set ``require_daily=False`` only when a caller will handle gaps.
    The function never fills or silently sorts missing or unordered rows.
    """
    if frame.empty:
        raise ValueError("OHLCV data is empty")

    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Date" in data.columns:
        dates = pd.to_datetime(data.pop("Date"), errors="raise", utc=True)
        data.index = dates
    elif not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("OHLCV data must have a Date column or DatetimeIndex")

    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")
    data.index = data.index.normalize().tz_localize(None)
    data.index.name = "Date"

    missing = [column for column in OHLCV_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {', '.join(missing)}")
    data = data.loc[:, list(OHLCV_COLUMNS)].copy()

    if data.index.has_duplicates:
        duplicates = data.index[data.index.duplicated()].unique().strftime("%Y-%m-%d")
        raise ValueError(f"Duplicate dates found: {', '.join(duplicates[:5])}")
    if not data.index.is_monotonic_increasing:
        raise ValueError("Dates must already be in ascending order")

    for column in OHLCV_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="raise")
    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("OHLCV values must be finite and contain no missing values")
    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be greater than zero")
    if (data["Volume"] < 0).any():
        raise ValueError("Volume cannot be negative")
    if (data["High"] < data[["Open", "Close", "Low"]].max(axis=1)).any():
        raise ValueError("High must be at least the Open, Close, and Low")
    if (data["Low"] > data[["Open", "Close", "High"]].min(axis=1)).any():
        raise ValueError("Low must be at most the Open, Close, and High")

    if require_daily and len(data) > 1:
        gaps = data.index.to_series().diff().dropna()
        if (gaps != pd.Timedelta(days=1)).any():
            first_gap = gaps[gaps != pd.Timedelta(days=1)].index[0]
            previous = first_gap - gaps.loc[first_gap]
            raise ValueError(
                "Daily OHLCV data has a calendar gap between "
                f"{previous:%Y-%m-%d} and {first_gap:%Y-%m-%d}; inspect the source "
                "instead of forward-filling prices."
            )
    return data


def load_ohlcv_csv(path: str | Path, *, require_daily: bool = True) -> pd.DataFrame:
    """Load an OHLCV CSV containing a ``Date`` column and validate it."""
    raw = pd.read_csv(path)
    return validate_ohlcv(raw, require_daily=require_daily)


def load_ohlcv_csv_through(
    path: str | Path,
    end_date: str | pd.Timestamp,
    *,
    require_daily: bool = True,
) -> pd.DataFrame:
    """Read an ascending OHLCV CSV only through the requested calendar date.

    Reading stops at the first later date. Price, volume, and other values in
    later rows are not parsed or passed into validation or feature generation.
    The CSV must be date-ordered, as produced by the project's data pipeline.
    """
    cutoff = pd.Timestamp(end_date).normalize()
    rows: list[dict[str, object]] = []
    input_path = Path(path)
    with input_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "Date" not in reader.fieldnames:
            raise ValueError("OHLCV CSV must contain a Date column")
        missing = [column for column in OHLCV_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {', '.join(missing)}")
        for row in reader:
            date_value = pd.to_datetime(row["Date"], errors="raise", utc=True)
            day = date_value.normalize().tz_localize(None)
            if day > cutoff:
                break
            rows.append({"Date": day, **{column: row[column] for column in OHLCV_COLUMNS}})
    if not rows:
        raise ValueError("OHLCV CSV has no rows on or before the requested cutoff")
    return validate_ohlcv(pd.DataFrame(rows), require_daily=require_daily)


def fetch_ohlcv(
    symbol: str = "BTC-USD",
    start: str = "2018-01-01",
    end_exclusive: str = "2026-01-01",
) -> pd.DataFrame:
    """Fetch daily OHLCV from yfinance; ``end_exclusive`` is not included."""
    import yfinance as yf

    # yfinance stores cookies and timezone mappings in a user cache directory.
    # Point it at a project-local ignored directory so runs work in restricted
    # environments and remain independent of machine-specific profile paths.
    cache_dir = Path(
        os.environ.get(
            "BTC_FORECASTING_CACHE_DIR",
            Path(__file__).resolve().parents[2] / "data" / "raw" / ".yfinance-cache",
        )
    ).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))

    downloaded = yf.download(
        symbol,
        start=start,
        end=end_exclusive,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if downloaded.empty:
        raise ValueError(f"yfinance returned no daily data for {symbol}")
    if isinstance(downloaded.columns, pd.MultiIndex):
        downloaded.columns = downloaded.columns.get_level_values(0)
    downloaded.index.name = "Date"
    return validate_ohlcv(downloaded)


def save_raw_snapshot(frame: pd.DataFrame, path: str | Path) -> Path:
    """Save validated source OHLCV without replacing an existing snapshot."""
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing raw snapshot: {output}")
    clean = validate_ohlcv(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output, date_format="%Y-%m-%d", index_label="Date")
    return output
