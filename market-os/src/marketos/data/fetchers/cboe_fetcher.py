"""CBOE volatility surface fetcher — free via yfinance.

Captures:
  - VIX term structure: VIX9D / VIX / VIX3M / VIX6M (contango vs backwardation)
  - VVIX: volatility-of-VIX (tail-risk hedging activity of dealers)
  - SKEW: CBOE Skew index (left-tail fear in options market)
  - Sector VIX: ^GVZ (gold), ^OVX (crude oil), ^VXEEM (EM equities)
  - Move index proxy: ^MOVE isn't on yfinance, but TLT options give bond vol context

Term structure conventions:
  slope_front = VIX9D - VIX      (positive → near-term fear spike, inverted)
  slope_back  = VIX - VIX3M      (positive → backwardation, stressed market)
  contango    = VIX3M - VIX       (positive → normal, calm market)

SKEW interpretation:
  Normal SKEW ~115-125. SKEW > 140 = expensive downside puts (tail-risk hedging).
  SKEW < 100 = complacency (rare). SKEW rising with VIX rising = genuine crash fear.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# (yfinance_ticker, feature_prefix)
_VIX_TERM_STRUCTURE = [
    ("^VIX9D",  "vix9d"),
    ("^VIX",    "vix"),
    ("^VIX3M",  "vix3m"),
    ("^VIX6M",  "vix6m"),
]

_CBOE_EXTRAS = [
    ("^VVIX",  "vvix"),
    ("^SKEW",  "skew"),
    ("^GVZ",   "gvz"),
    ("^OVX",   "ovx"),
    ("^VXEEM", "vxeem"),
]


def _fetch_last(ticker: str, period: str = "5d") -> float | None:
    """Fetch the most recent closing value for a CBOE index via yfinance."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return None
        return float(hist["Close"].dropna().iloc[-1])
    except Exception as e:
        logger.debug(f"cboe {ticker}: {e}")
        return None


def _fetch_history(ticker: str, period: str = "1y") -> pd.Series | None:
    """Fetch daily closing history for rolling z-scores."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return None
        return hist["Close"].dropna()
    except Exception as e:
        logger.debug(f"cboe hist {ticker}: {e}")
        return None


def compute_cboe_features() -> dict:
    """Fetch CBOE volatility surface and compute derived features.

    Returns a single flat dict suitable for _store_features under the
    symbol '_cboe' and family 'cboe_vol'.
    """
    now = datetime.now(timezone.utc)
    row: dict = {"asof_ts": now, "knowledge_ts": now}

    # 1. VIX term structure — latest levels
    ts_values: dict[str, float] = {}
    for ticker, prefix in _VIX_TERM_STRUCTURE:
        v = _fetch_last(ticker)
        if v is not None:
            row[prefix] = v
            ts_values[prefix] = v

    # 2. Term structure slopes (regime signals)
    vix9d = ts_values.get("vix9d")
    vix   = ts_values.get("vix")
    vix3m = ts_values.get("vix3m")
    vix6m = ts_values.get("vix6m")

    if vix9d is not None and vix is not None:
        row["vix_slope_front"] = vix9d - vix        # + → near-term fear spike
    if vix is not None and vix3m is not None:
        row["vix_slope_back"] = vix - vix3m          # + → backwardation (stressed)
        row["vix_contango"]   = vix3m - vix          # + → normal/calm
        row["vix_backwardation"] = int(vix > vix3m)  # binary flag
    if vix3m is not None and vix6m is not None:
        row["vix_long_slope"] = vix6m - vix3m        # + → long-end steepening
    if vix9d is not None and vix3m is not None:
        row["vix_ts_spread_total"] = vix3m - vix9d   # full term spread

    # 3. VIX z-score vs 1y history
    vix_hist = _fetch_history("^VIX", period="1y")
    if vix_hist is not None and vix is not None and len(vix_hist) >= 20:
        mu, sd = float(vix_hist.mean()), float(vix_hist.std())
        row["vix_zscore_1y"] = (vix - mu) / sd if sd > 0 else 0.0
        row["vix_pct_rank_1y"] = float((vix_hist <= vix).mean())  # percentile
        # Percentile ≥ 0.8 → elevated stress regime
        row["vix_elevated"] = int(row.get("vix_pct_rank_1y", 0) >= 0.8)

    # 4. Extras: VVIX, SKEW, sector vols
    for ticker, prefix in _CBOE_EXTRAS:
        v = _fetch_last(ticker)
        if v is not None:
            row[prefix] = v

    vvix = row.get("vvix")
    skew = row.get("skew")

    # VVIX z-score (measures dealer hedging activity on VIX options)
    vvix_hist = _fetch_history("^VVIX", period="1y")
    if vvix_hist is not None and vvix is not None and len(vvix_hist) >= 20:
        mu, sd = float(vvix_hist.mean()), float(vvix_hist.std())
        row["vvix_zscore_1y"] = (vvix - mu) / sd if sd > 0 else 0.0

    # SKEW z-score and tail-risk flag (SKEW > 140 → expensive downside puts)
    if skew is not None:
        row["skew_tail_risk"] = int(skew > 140)
        row["skew_complacency"] = int(skew < 100)
        skew_hist = _fetch_history("^SKEW", period="1y")
        if skew_hist is not None and len(skew_hist) >= 20:
            mu, sd = float(skew_hist.mean()), float(skew_hist.std())
            row["skew_zscore_1y"] = (skew - mu) / sd if sd > 0 else 0.0

    # Cross-asset vol spread: OVX - VIX (oil stress vs equity stress)
    ovx = row.get("ovx")
    if ovx is not None and vix is not None:
        row["vol_oil_equity_spread"] = ovx - vix

    # GVZ - VIX (gold vol vs equity vol — flight to safety signal)
    gvz = row.get("gvz")
    if gvz is not None and vix is not None:
        row["vol_gold_equity_spread"] = gvz - vix

    # EM vol spread: VXEEM - VIX (EM risk premium vs US risk)
    vxeem = row.get("vxeem")
    if vxeem is not None and vix is not None:
        row["vol_em_us_spread"] = vxeem - vix

    # Composite regime signal: 0=calm, 1=elevated, 2=backwardation+elevated
    regime_score = 0
    if row.get("vix_elevated"):
        regime_score += 1
    if row.get("vix_backwardation"):
        regime_score += 1
    row["cboe_vol_regime_score"] = regime_score

    return row
