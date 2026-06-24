"""RBI (Reserve Bank of India) data fetcher — free, no API key.

Sources:
  - RBI DBIE (Database on Indian Economy) — REST API
  - RBI weekly statistical supplement (PDF-parsed via public URLs)
  - RBI press releases (rates, forex reserves)

Signals:
  - Repo rate, reverse repo rate (monetary policy stance)
  - CRR (Cash Reserve Ratio) — liquidity multiplier
  - SLR (Statutory Liquidity Ratio)
  - Forex reserves (weekly, $bn) — INR strength proxy
  - Bank credit growth (weekly YoY %)
  - Money supply (M3 growth) — inflation precursor
  - Call money rate (overnight interbank) — liquidity stress
  - INR/USD official rate

These are critical context for NSE signals: repo rate hikes → NIFTY Bank underperformance.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import urllib3
import pandas as pd
import requests

logger = logging.getLogger(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_DBIE_BASE = "https://dbie.rbi.org.in/DBIE/dbie.rbi"

# DBIE series codes
SERIES_MAP = {
    "repo_rate":        "BSR1:BSR1E014",     # Repo rate %
    "reverse_repo":     "BSR1:BSR1E015",     # Reverse repo rate %
    "crr":             "BSR1:BSR1E008",     # CRR %
    "forex_reserves":  "FER:FER1",          # Total forex reserves $mn
    "bank_credit":     "DBR:DBR1A",         # Non-food bank credit
    "m3":              "MSS:MSS1A",         # M3 (broad money)
}

_HEADERS = {"User-Agent": "MarketOS/0.1 research@guaqai.me"}


def _fetch_dbie(series_code: str, periods: int = 52) -> pd.Series:
    """Fetch a series from RBI DBIE API."""
    try:
        url = f"{_DBIE_BASE}?__action=api&type=LASTOBS&series={series_code}&n={periods}"
        # DBIE has a hostname mismatch SSL cert; suppress verification (read-only public data)
        r = requests.get(url, headers=_HEADERS, timeout=20, verify=False)
        r.raise_for_status()
        data = r.json()
        if not data or "data" not in data:
            return pd.Series(dtype=float)
        records = {}
        for item in data["data"]:
            if len(item) >= 2:
                try:
                    ts = pd.to_datetime(str(item[0]), dayfirst=True)
                    records[ts] = float(item[1])
                except Exception:
                    pass
        if not records:
            return pd.Series(dtype=float)
        s = pd.Series(records).sort_index()
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        return s
    except Exception as e:
        logger.warning(f"RBI DBIE {series_code}: {e}")
        return pd.Series(dtype=float)


def fetch_rbi_forex_reserves() -> dict:
    """Fetch RBI weekly forex reserves and compute features."""
    s = _fetch_dbie("FER:FER1", periods=55)
    if s.empty:
        return {}
    latest = float(s.iloc[-1])
    result = {"rbi_forex_reserves_mn": latest}
    if len(s) >= 4:
        prev_4w = float(s.iloc[-5:-1].mean())
        result["rbi_forex_4w_chg"] = latest - prev_4w
        result["rbi_forex_4w_chg_pct"] = (latest - prev_4w) / max(abs(prev_4w), 1.0)
    if len(s) >= 52:
        result["rbi_forex_52w_high"] = float(s.iloc[-52:].max())
        result["rbi_forex_52w_low"] = float(s.iloc[-52:].min())
        result["rbi_forex_52w_z"] = (latest - float(s.iloc[-52:].mean())) / max(float(s.iloc[-52:].std()), 1.0)
    return result


def fetch_rbi_rates() -> dict:
    """Fetch current RBI policy rates."""
    result = {}
    for name, code in [("repo_rate", "BSR1:BSR1E014"), ("reverse_repo", "BSR1:BSR1E015"),
                        ("crr", "BSR1:BSR1E008")]:
        s = _fetch_dbie(code, periods=24)
        if not s.empty:
            result[f"rbi_{name}"] = float(s.iloc[-1])
            if len(s) >= 2:
                result[f"rbi_{name}_chg_prev"] = float(s.iloc[-1] - s.iloc[-2])
            if len(s) >= 12:
                result[f"rbi_{name}_chg_12m"] = float(s.iloc[-1] - s.iloc[-12])
    return result


def fetch_india_inr_usd() -> dict:
    """Fetch INR/USD spot rate from RBI or Open-Exchange-Rates fallback."""
    try:
        # RBI publishes reference rates
        r = requests.get(
            "https://www.rbi.org.in/Scripts/ReferenceRateArchive.aspx",
            headers=_HEADERS, timeout=15
        )
        # Fallback: use exchangerate-api.com (free, no key for common pairs)
        r2 = requests.get(
            "https://open.er-api.com/v6/latest/USD",
            headers=_HEADERS, timeout=10
        )
        r2.raise_for_status()
        rates = r2.json().get("rates", {})
        inr = rates.get("INR", 0)
        if inr:
            return {
                "rbi_inr_usd": float(inr),
                "rbi_inr_per_dollar": float(inr),
            }
    except Exception as e:
        logger.warning(f"RBI INR/USD: {e}")
    return {}


def compute_rbi_features() -> pd.DataFrame:
    """Aggregate all RBI signals into one feature row."""
    now = datetime.now(timezone.utc)
    row: dict = {"asof_ts": now, "knowledge_ts": now}

    forex = fetch_rbi_forex_reserves()
    row.update(forex)

    rates = fetch_rbi_rates()
    row.update(rates)

    inr = fetch_india_inr_usd()
    row.update(inr)

    # Derived: rate spread (repo - reverse_repo = corridor width)
    if "rbi_repo_rate" in row and "rbi_reverse_repo" in row:
        row["rbi_rate_corridor"] = row["rbi_repo_rate"] - row["rbi_reverse_repo"]

    return pd.DataFrame([row])
