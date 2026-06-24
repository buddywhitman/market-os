"""yfinance OHLCV fetcher — the zero-cost default source.

Raw CSV bytes land in the lake immutably; a normalized DataFrame is returned for use.
If yfinance/network is unavailable, `synthetic_ohlcv` generates a deterministic
geometric-Brownian-motion series so the whole pipeline (and `make demo`) runs offline.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from marketos.data.lake import DataLake


def fetch_ohlcv(
    symbol: str,
    *,
    lake: DataLake,
    period: str = "5y",
    interval: str = "1d",
    code_version: str = "0.0.0",
) -> pd.DataFrame:
    try:
        import yfinance as yf

        raw = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError("empty frame")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        csv_bytes = df.to_csv().encode()
        lake.put_raw("ohlcv", csv_bytes, source="yfinance", ext="csv",
                     code_version=code_version, extra={"symbol": symbol, "interval": interval})
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        df = synthetic_ohlcv(symbol)
        lake.put_raw("ohlcv", df.to_csv().encode(), source="synthetic-gbm", ext="csv",
                     code_version=code_version, extra={"symbol": symbol, "interval": interval})
        return df


def synthetic_ohlcv(symbol: str, n: int = 1260, seed: int | None = None) -> pd.DataFrame:
    """Deterministic GBM OHLCV so the pipeline runs with no network. Seeded per symbol."""
    seed = seed if seed is not None else (abs(hash(symbol)) % (2**31))
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    mu, sigma = 0.0004, 0.018
    rets = rng.normal(mu, sigma, n)
    close = 100 * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(1_000_000, 8_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
