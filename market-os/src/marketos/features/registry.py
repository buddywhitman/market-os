"""Feature registry — assembles the full feature vector per instrument.

This is the single entry point the scheduler calls. It runs each feature-family builder,
takes the latest causal row from each, broadcasts panel-level families (macro), and merges
everything into one wide vector per symbol. It also reports the realized feature count so we
can track progress toward the 1,100–1,500 target.

Design principle: families are independent and degrade gracefully. If the factor panel is
unavailable, cross-asset features are simply absent — the rest still computes. Nothing here
raises on missing inputs; it returns what it can and logs the gap.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from marketos.features.technical import build_technical_features
from marketos.features.cross_asset import build_cross_asset_features, build_factor_return_panel, FACTOR_SERIES
from marketos.features.seasonality import build_seasonality_features

logger = logging.getLogger(__name__)

# Maps a FACTOR_SERIES key → the symbol/series we fetch it from.
# Equity/ETF factors come from yfinance; macro factors from the FRED panel.
FACTOR_EQUITY_SYMBOLS = {
    # Core market
    "SPY": "SPY", "QQQ": "QQQ",
    # Crypto
    "BTC": "BTC-USD",
    # Commodities
    "GOLD": "GLD",
    # Sector ETFs — add sensitivity to sector rotation
    "XLK": "XLK",   # Technology
    "XLF": "XLF",   # Financials
    "XLE": "XLE",   # Energy
    "XLRE": "XLRE", # Real Estate (rate-sensitive proxy)
    "XLV": "XLV",   # Health Care (defensive)
    "XLU": "XLU",   # Utilities (rate-sensitive defensive)
    "XLI": "XLI",   # Industrials (cyclical)
}
FACTOR_MACRO_COLUMNS = {"VIX": "VIX", "DXY": "DXY", "US10Y": "US10Y", "OIL": "OIL_WTI"}


def _latest_row(frame: pd.DataFrame) -> dict:
    """Take the last fully-formed row of a feature frame as a flat dict."""
    if frame is None or frame.empty:
        return {}
    row = frame.iloc[-1].to_dict()
    # Drop the point-in-time bookkeeping columns; they're set by the caller
    for k in ("asof_ts", "knowledge_ts"):
        row.pop(k, None)
    # Replace non-finite with None so JSON/Postgres accept them
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
            for k, v in row.items()}


def build_factor_panel(price_frames: dict[str, pd.DataFrame],
                       macro_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assemble a daily factor-return panel from equity prices + macro series.

    price_frames: {FACTOR_KEY: ohlcv_frame} for equity/ETF/crypto factors.
    macro_frame : the raw FRED macro panel (daily, columns like 'VIX','DXY','US10Y','OIL_WTI').
    """
    panel = build_factor_return_panel(price_frames)
    if macro_frame is not None and not macro_frame.empty:
        macro_returns = {}
        for fkey, mcol in FACTOR_MACRO_COLUMNS.items():
            if mcol in macro_frame.columns:
                # For VIX/rates we use level-change; for DXY/oil pct-change
                if fkey in ("US10Y",):
                    macro_returns[fkey] = macro_frame[mcol].diff()
                else:
                    macro_returns[fkey] = macro_frame[mcol].pct_change()
        if macro_returns:
            mret = pd.DataFrame(macro_returns)
            mret.index = pd.to_datetime(mret.index)
            if mret.index.tz is None:
                mret.index = mret.index.tz_localize("UTC")
            else:
                mret.index = mret.index.tz_convert("UTC")
            if not panel.empty:
                panel.index = pd.to_datetime(panel.index)
                if panel.index.tz is None:
                    panel.index = panel.index.tz_localize("UTC")
                else:
                    panel.index = panel.index.tz_convert("UTC")
                panel = panel.join(mret, how="outer")
            else:
                panel = mret
    return panel.sort_index() if not panel.empty else panel


def compute_symbol_features(
    symbol: str,
    ohlcv: pd.DataFrame,
    *,
    factor_panel: pd.DataFrame | None = None,
    macro_broadcast: dict | None = None,
    factor_series: list[str] | None = None,
) -> dict:
    """Compute the full feature vector for one symbol's latest bar.

    Returns a flat dict {feature_name: value} including asof_ts/knowledge_ts.
    Families: technical, cross_asset, seasonality, + broadcast macro.

    `factor_series` — passed through to `build_cross_asset_features`; defaults to the US
    FACTOR_SERIES names if omitted. MUST be set to match `factor_panel`'s actual column
    names for a non-US panel (e.g. India's NIFTY50/BANKNIFTY/INDIA_VIX) — see that
    function's docstring for why this silently produces zero cross-asset features
    otherwise, not an error.
    """
    vector: dict = {}
    if ohlcv is None or ohlcv.empty:
        return vector

    # 1. Technical (≈180 features)
    tech = build_technical_features(ohlcv)
    vector.update(_latest_row(tech))

    # 2. Cross-asset (≈60 features) — needs the factor panel + this asset's returns
    if factor_panel is not None and not factor_panel.empty:
        asset_ret = ohlcv["close"].pct_change()
        if asset_ret.index.tz is None:
            asset_ret.index = pd.to_datetime(asset_ret.index).tz_localize("UTC")
        try:
            xa = build_cross_asset_features(asset_ret, factor_panel, factor_series=factor_series)
            vector.update({f"xa_{k}": v for k, v in _latest_row(xa).items()})
        except Exception as e:
            logger.warning(f"cross-asset features failed for {symbol}: {e}")

    # 3. Seasonality (≈22 features) — deterministic on the latest timestamp
    seas = build_seasonality_features(ohlcv.index[-1:])
    vector.update(_latest_row(seas))

    # 4. Macro broadcast — attach current macro regime to every symbol
    if macro_broadcast:
        vector.update({f"macro_{k}": v for k, v in macro_broadcast.items()
                       if k not in ("asof_ts", "knowledge_ts", "feature_family")})

    # Point-in-time stamps
    now = datetime.now(timezone.utc)
    asof = ohlcv.index[-1]
    if hasattr(asof, "to_pydatetime"):
        asof = asof.to_pydatetime()
    if getattr(asof, "tzinfo", None) is None:
        asof = pd.Timestamp(asof).tz_localize("UTC").to_pydatetime()
    vector["asof_ts"] = asof
    vector["knowledge_ts"] = now
    return vector


def feature_count(vector: dict) -> int:
    """Count actual feature columns (excludes bookkeeping)."""
    return len([k for k in vector if k not in ("asof_ts", "knowledge_ts")])
