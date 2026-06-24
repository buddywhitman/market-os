"""Polygon.io fetcher — free tier: 5 req/min, no key expiry.

Register at: https://polygon.io/dashboard/signup
Set env var: POLYGON_API_KEY

Key free-tier endpoints:
  - /v2/aggs/ticker/{ticker}/range/{mult}/{timespan}/{from}/{to} — OHLCV aggregates
  - /v3/reference/options/{underlying_asset} — options contracts
  - /v2/snapshot/locale/us/markets/options/tickers/{option_ticker} — option snapshot
  - /v2/snapshot/locale/us/markets/stocks/tickers/{ticker} — stock snapshot

Options-derived signals (high alpha):
  - IV term structure slope (30d vs 60d vs 90d vs 180d IV) → forward vol expectations
  - Put/call IV skew → tail risk pricing
  - IV rank (IVR) = (current IV - 1yr low) / (1yr high - 1yr low) → mean-reversion setup
  - Options volume vs open interest → options flow direction
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone, date, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
_BASE = "https://api.polygon.io"
_REQUEST_GAP = 13  # 5 req/min free tier → 12s between calls + buffer


def _get(path: str, params: dict | None = None) -> dict | None:
    if not POLYGON_API_KEY:
        return None
    p = {"apiKey": POLYGON_API_KEY, **(params or {})}
    try:
        r = requests.get(f"{_BASE}{path}", params=p, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"Polygon {path}: {e}")
        return None


def fetch_stock_snapshot(symbol: str) -> dict:
    """Previous-day OHLCV. /v2/snapshot/.../tickers/{symbol} (intraday/real-time) is gated
    on the free tier (verified 2026-06-22: 403 NOT_AUTHORIZED) — /v2/aggs/.../prev is not,
    and gives the same fields one day lagged, which is fine for a daily-cadence feature."""
    data = _get(f"/v2/aggs/ticker/{symbol}/prev")
    time.sleep(_REQUEST_GAP)
    if not data or not data.get("results"):
        return {}
    bar = data["results"][0]
    return {
        "poly_open": float(bar.get("o", 0) or 0),
        "poly_high": float(bar.get("h", 0) or 0),
        "poly_low": float(bar.get("l", 0) or 0),
        "poly_close": float(bar.get("c", 0) or 0),
        "poly_volume": float(bar.get("v", 0) or 0),
        "poly_vwap": float(bar.get("vw", 0) or 0),
    }


def fetch_options_chain_proxy(symbol: str, current_price: float) -> dict:
    """Options-derived vol/skew proxies from DELAYED contract + prior-close data.

    Free tier blocks real-time snapshots (/v2,v3/snapshot/...) AND implied_volatility/greeks
    entirely (those only ever come back on a snapshot response). So this trades precision for
    availability: instead of IV, it prices an ATM straddle off yesterday's close as a level-of-
    fear proxy, and the OTM put/call price ratio as a skew proxy — both genuinely informative,
    just one day lagged and without a Black-Scholes inversion.

    Budget: 1 contracts-listing call + up to 3 prev-close calls.

    The straddle/skew price fields are absent whenever the chosen strike traded zero volume
    the prior session (resultsCount=0 from Polygon) — the reference/contracts endpoint has no
    volume/OI field to pick a liquid strike (that's snapshot-only data, also gated), so we
    can't avoid this without expensive multi-strike retries at 13s/call. poly_chain_call_count/
    put_count/skew_count are unaffected (contract-listing data, always present when the
    underlying has an options chain) — this is the same "missing data degrades gracefully,
    not silently wrong" pattern the rest of this codebase follows.
    """
    if current_price <= 0:
        return {}

    target_exp = (date.today() + timedelta(days=30)).isoformat()
    data = _get("/v3/reference/options/contracts", {
        "underlying_ticker": symbol,
        "expiration_date.gte": (date.today() + timedelta(days=20)).isoformat(),
        "expiration_date.lte": (date.today() + timedelta(days=45)).isoformat(),
        "limit": 250,
    })
    time.sleep(_REQUEST_GAP)
    contracts = (data or {}).get("results") or []
    if not contracts:
        return {}

    calls = [c for c in contracts if c.get("contract_type") == "call"]
    puts = [c for c in contracts if c.get("contract_type") == "put"]
    result = {
        "poly_chain_call_count": len(calls),
        "poly_chain_put_count": len(puts),
    }
    if calls and puts:
        result["poly_chain_skew_count"] = (len(calls) - len(puts)) / (len(calls) + len(puts))

    def _nearest(group: list[dict], strike: float) -> dict | None:
        return min(group, key=lambda c: abs(c.get("strike_price", 1e9) - strike), default=None)

    atm_call = _nearest(calls, current_price)
    atm_put = _nearest(puts, current_price)
    # 10%-OTM put (downside skew is what tail-risk pricing cares about).
    otm_put = _nearest(puts, current_price * 0.90)

    def _prev_close(contract: dict | None) -> float | None:
        ticker = (contract or {}).get("ticker")
        if not ticker:
            return None
        d = _get(f"/v2/aggs/ticker/{ticker}/prev")
        time.sleep(_REQUEST_GAP)
        bars = (d or {}).get("results") or []
        return float(bars[0]["c"]) if bars else None

    call_px = _prev_close(atm_call)
    put_px = _prev_close(atm_put)
    if call_px is not None and put_px is not None:
        result["poly_atm_straddle_pct"] = (call_px + put_px) / current_price
        result["poly_pc_price_ratio"] = put_px / call_px if call_px > 0 else None

    # Downside skew: 10%-OTM put price relative to the ATM put. Tail-risk demand pushes OTM
    # put prices up disproportionately (volatility smile) — a rising ratio = more fear priced
    # into crash protection specifically, distinct from the at-the-money vol level above.
    otm_put_px = _prev_close(otm_put)
    if otm_put_px is not None and put_px is not None and put_px > 0:
        result["poly_downside_skew"] = otm_put_px / put_px

    return {k: v for k, v in result.items() if v is not None}


def compute_polygon_features(universe: list[str]) -> pd.DataFrame:
    """Fetch Polygon signals for universe — rate-limited for free tier.

    Budget: 1 stock-snapshot call + ~4 options-chain-proxy calls per symbol, at the free
    tier's 13s/request gap = ~65s/symbol. Capped at 4 symbols (~4-5min total) — the
    scheduler's per-fetcher timeout for "polygon" must be >= that (see scheduler.py).
    Prioritize the highest-options-activity names in the universe.
    """
    if not POLYGON_API_KEY:
        logger.info("POLYGON_API_KEY not set — skipping Polygon ingest")
        return pd.DataFrame()

    priority = ["NVDA", "AMD", "SPY", "QQQ", "MSTR", "COIN", "PLTR", "AVGO"]
    targets = [s for s in priority if s in universe][:4]

    now = datetime.now(timezone.utc)
    rows = []
    for sym in targets:
        row: dict = {"symbol": sym, "asof_ts": now, "knowledge_ts": now}
        try:
            snap = fetch_stock_snapshot(sym)
            row.update(snap)
            chain = fetch_options_chain_proxy(sym, snap.get("poly_close", 0))
            row.update(chain)
            rows.append(row)
        except Exception as e:
            logger.warning(f"Polygon {sym}: {e}")

    return pd.DataFrame(rows) if rows else pd.DataFrame()
