"""Alpha Vantage fetcher — free tier: 25 req/day, 5 req/min.

Register at: https://www.alphavantage.co/support/#api-key
Set env var: ALPHAVANTAGE_API_KEY

Key endpoints (free tier):
  - CURRENCY_EXCHANGE_RATE — real-time FX rates for 200+ pairs
  - FX_DAILY — daily OHLCV for FX pairs (last 100 days)
  - DIGITAL_CURRENCY_DAILY — BTC/ETH/etc vs USD daily
  - TIME_SERIES_DAILY_ADJUSTED — equity OHLCV (backup to yfinance for non-US names)
  - ECONOMIC_INDICATORS — US GDP, real GDP, CPI, retail sales, unemployment rate
  - TREASURY_YIELD — US 2Y/5Y/10Y/30Y treasury yields
  - FEDERAL_FUNDS_RATE, INFLATION, UNEMPLOYMENT

At 25 calls/day with free tier, we focus on:
1. FX rates (USD pairs: EUR, JPY, GBP, CNY, INR, KRW, AUD) — 7 calls
2. Treasury yield curve (2Y + 10Y + FEDFUNDS) — 3 calls
3. Macro indicators (inflation, unemployment, retail sales) — 3 calls
= ~13 calls total, leaving buffer.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

AV_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")
_BASE = "https://www.alphavantage.co/query"
_REQUEST_GAP = 13  # 5/min limit

FX_PAIRS = [
    ("EUR", "USD"), ("JPY", "USD"), ("GBP", "USD"),
    ("CNY", "USD"), ("INR", "USD"), ("KRW", "USD"), ("AUD", "USD"),
]

MACRO_FUNCTIONS = {
    "real_gdp":      ("REAL_GDP", "quarterly"),
    "inflation":     ("INFLATION", "annual"),
    "retail_sales":  ("RETAIL_SALES", "monthly"),
    "unemployment":  ("UNEMPLOYMENT", "monthly"),
    "consumer_sent": ("CONSUMER_SENTIMENT", "monthly"),
    "nonfarm_payrolls": ("NONFARM_PAYROLL", "monthly"),
}

YIELD_MATURITIES = ["2year", "5year", "10year", "30year"]


def _av_get(params: dict) -> dict | None:
    if not AV_API_KEY:
        return None
    p = {"apikey": AV_API_KEY, **params}
    try:
        r = requests.get(_BASE, params=p, timeout=20)
        r.raise_for_status()
        data = r.json()
        if "Error Message" in data or "Note" in data:
            logger.warning(f"AlphaVantage response: {data.get('Note', data.get('Error Message'))}")
            return None
        return data
    except Exception as e:
        logger.warning(f"AlphaVantage {params.get('function')}: {e}")
        return None


def fetch_fx_rates() -> dict:
    """Fetch current FX rates for major USD pairs."""
    result = {}
    for from_ccy, to_ccy in FX_PAIRS:
        data = _av_get({"function": "CURRENCY_EXCHANGE_RATE",
                        "from_currency": from_ccy, "to_currency": to_ccy})
        time.sleep(_REQUEST_GAP)
        if not data:
            continue
        info = data.get("Realtime Currency Exchange Rate", {})
        rate = info.get("5. Exchange Rate")
        bid = info.get("8. Bid Price")
        ask = info.get("9. Ask Price")
        if rate:
            pair = f"{from_ccy}{to_ccy}"
            result[f"av_fx_{pair.lower()}"] = float(rate)
            if bid and ask:
                result[f"av_fx_{pair.lower()}_spread"] = float(ask) - float(bid)
    return result


def fetch_treasury_yields() -> dict:
    """Fetch US Treasury yield curve points."""
    result = {}
    for maturity in YIELD_MATURITIES[:2]:  # cap at 2 to save daily quota
        data = _av_get({"function": "TREASURY_YIELD", "interval": "monthly",
                        "maturity": maturity})
        time.sleep(_REQUEST_GAP)
        if not data:
            continue
        series = data.get("data", [])
        if series:
            latest = series[0]
            val = latest.get("value")
            if val and val != ".":
                result[f"av_yield_{maturity.replace('year', 'y')}"] = float(val)
    # Yield curve slope (10y - 2y)
    if "av_yield_10y" in result and "av_yield_2y" in result:
        result["av_yield_curve_slope"] = result["av_yield_10y"] - result["av_yield_2y"]
        result["av_yield_inverted"] = int(result["av_yield_curve_slope"] < 0)
    return result


def fetch_us_macro_indicators() -> dict:
    """Fetch US macro series from AlphaVantage economic indicators endpoint."""
    result = {}
    # Only fetch 2 to preserve daily quota
    priority = [("unemployment", "UNEMPLOYMENT", "monthly"),
                ("retail_sales", "RETAIL_SALES", "monthly")]
    for name, func, interval in priority:
        data = _av_get({"function": func, "interval": interval})
        time.sleep(_REQUEST_GAP)
        if not data:
            continue
        series = data.get("data", [])
        if not series:
            continue
        latest = series[0]
        val = latest.get("value")
        if val and val != ".":
            result[f"av_{name}_latest"] = float(val)
        if len(series) >= 2:
            prev = series[1].get("value")
            if prev and prev != ".":
                result[f"av_{name}_mom_chg"] = float(val) - float(prev)
        if len(series) >= 12:
            yoy_val = series[12].get("value") if len(series) > 12 else None
            if yoy_val and yoy_val != ".":
                result[f"av_{name}_yoy_chg"] = float(val) - float(yoy_val)
    return result


def compute_alphavantage_features() -> pd.DataFrame:
    """Compute AlphaVantage FX + macro features within daily quota."""
    if not AV_API_KEY:
        logger.info("ALPHAVANTAGE_API_KEY not set — skipping Alpha Vantage")
        return pd.DataFrame()

    now = datetime.now(timezone.utc)
    row: dict = {"asof_ts": now, "knowledge_ts": now}

    fx = fetch_fx_rates()
    row.update(fx)

    yields = fetch_treasury_yields()
    row.update(yields)

    macro = fetch_us_macro_indicators()
    row.update(macro)

    return pd.DataFrame([row])
