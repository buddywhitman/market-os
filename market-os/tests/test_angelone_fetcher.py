"""Unit tests for data/fetchers/angelone_fetcher.py — focused on the contract
`fetch_ohlcv_history` must satisfy: same column shape as `yfinance_fetcher.fetch_ohlcv`,
since `build_technical_features()` depends on it exactly. Column-shape mismatches caused
real bugs repeatedly elsewhere in this session — this guards against the same class here.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from marketos.data.fetchers.angelone_fetcher import fetch_ohlcv_history


def test_ohlcv_history_has_the_same_columns_as_yfinance_fetcher():
    fake_candles = [
        {"date": "2026-01-01T00:00:00+05:30", "open": 100.0, "high": 105.0,
         "low": 98.0, "close": 102.0, "volume": 50000},
        {"date": "2026-01-02T00:00:00+05:30", "open": 102.0, "high": 108.0,
         "low": 101.0, "close": 106.0, "volume": 60000},
    ]
    with patch("marketos.data.fetchers.angelone_fetcher.get_historical_candles",
              return_value=fake_candles):
        df = fetch_ohlcv_history("BEL", "383", jwt_token="fake", api_key="fake", years=1)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    # Deliberately unnamed (matches yfinance's behavior closely enough, and avoids the
    # real index/column-name collision bug — see test_ohlcv_history_index_has_no_name).
    assert df.index.name is None
    assert len(df) == 2


def test_ohlcv_history_sorted_ascending_and_deduplicated():
    """Real AngelOne responses come oldest-first already, but don't assume — and a
    duplicate date (e.g. a retried/overlapping fetch) must not produce two rows for one
    day, which would corrupt every rolling-window technical feature."""
    fake_candles = [
        {"date": "2026-01-02T00:00:00+05:30", "open": 102.0, "high": 108.0,
         "low": 101.0, "close": 106.0, "volume": 60000},
        {"date": "2026-01-01T00:00:00+05:30", "open": 100.0, "high": 105.0,
         "low": 98.0, "close": 102.0, "volume": 50000},
        {"date": "2026-01-01T00:00:00+05:30", "open": 100.0, "high": 105.0,
         "low": 98.0, "close": 102.0, "volume": 50000},  # duplicate
    ]
    with patch("marketos.data.fetchers.angelone_fetcher.get_historical_candles",
              return_value=fake_candles):
        df = fetch_ohlcv_history("BEL", "383", jwt_token="fake", api_key="fake", years=1)
    assert len(df) == 2  # duplicate collapsed
    assert df.index.is_monotonic_increasing


def test_empty_response_returns_correctly_shaped_empty_frame():
    """No candles (e.g. delisted symbol, bad token) must return the SAME column shape
    empty, not a frame missing columns — a caller that does df["close"] must not KeyError
    just because there was no data, only because of a genuinely different bug."""
    with patch("marketos.data.fetchers.angelone_fetcher.get_historical_candles",
              return_value=[]):
        df = fetch_ohlcv_history("FAKESYM", "0", jwt_token="fake", api_key="fake")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 0


def test_ohlcv_history_index_has_no_name():
    """Regression guard for a real bug caught running fit_supervised_latent against real
    India data: an index literally named "date" collides with a "date" COLUMN that
    latent_supervised.py's _cross_sectional_ic constructs internally (`pd.DataFrame({...,
    "date": dates})`, where `dates` carries its own index name through) — pandas raises
    "'date' is both an index level and a column label, which is ambiguous." yfinance's
    OHLCV index is named "Date" (capital D) and never collided; this fetcher must produce
    an UNNAMED index so it can never collide with any column name, not just "date"."""
    fake_candles = [{"date": "2026-01-01T00:00:00+05:30", "open": 100.0, "high": 105.0,
                     "low": 98.0, "close": 102.0, "volume": 50000}]
    with patch("marketos.data.fetchers.angelone_fetcher.get_historical_candles",
              return_value=fake_candles):
        df = fetch_ohlcv_history("BEL", "383", jwt_token="fake", api_key="fake", years=1)
    assert df.index.name is None


def test_cross_sectional_ic_would_raise_on_a_named_date_index():
    """Proves the bug is real, not a misreading of the traceback: feed
    `_cross_sectional_ic` a pred/actual Series whose index is named "date" (the exact
    shape the unfixed fetcher produced) and confirm it raises the same ambiguity error."""
    import pandas as pd
    import pytest as pt
    from marketos.features.latent_supervised import _cross_sectional_ic

    idx = pd.date_range("2026-01-01", periods=10, name="date")
    pred = pd.Series(range(10), index=idx, dtype=float)
    actual = pd.Series(range(10), index=idx, dtype=float)
    dates = pd.Series(idx, index=idx)
    with pt.raises(ValueError, match="ambiguous"):
        _cross_sectional_ic(pred, actual, dates)
