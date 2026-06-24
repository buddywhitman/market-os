"""UN Comtrade fetcher — 500 calls/day free, no key required for public data.

API v2: https://comtradeapi.un.org/public/v1/

Trade flow signals:
  - US imports of semiconductors (HS 8542) → NVDA/AMD/AVGO demand proxy
  - US exports of commercial aircraft (HS 8802) → Boeing/GEV/aerospace cycle
  - US imports of crude oil (HS 2709) → WTI demand signal
  - China imports of semiconductors → global chip demand
  - China exports to US → US-China trade tension barometer

Note: Comtrade data lags 2-6 months (monthly data with reporting delay).
It's a slow-moving structural signal, not a trading trigger.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, date
from calendar import monthrange

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BASE = "https://comtradeapi.un.org/public/v1/preview"
_HEADERS = {"User-Agent": "MarketOS/0.1 research@guaqai.me",
            "Accept": "application/json"}

# (reporter, partner, hs_code, friendly_name)
TRADE_QUERIES = [
    ("840", "0",   "8542", "us_semi_imports"),        # US ← World: semiconductors
    ("156", "0",   "8542", "china_semi_imports"),     # China ← World: semiconductors
    ("840", "156", "0",    "us_china_all_imports"),   # US ← China: all goods
    ("840", "0",   "2709", "us_crude_imports"),       # US ← World: crude oil
    ("840", "0",   "8802", "us_aircraft_exports"),    # US → World: aircraft (reversed)
]

# Comtrade country codes
COUNTRY_CODES = {"us": "840", "cn": "156", "world": "0"}


def _fetch_trade_series(reporter: str, partner: str, cmdcode: str,
                        flow: str = "M") -> pd.DataFrame:
    """Fetch monthly trade data for a specific commodity pair.

    reporter: ISO numeric code of reporting country
    partner: ISO numeric code of partner ('0' = world total)
    cmdcode: HS code ('0' = all goods)
    flow: 'M' = imports, 'X' = exports
    """
    # Get last available year's data
    today = date.today()
    period = f"{today.year - 1}"  # previous year (current year lags 6+ months)
    params = {
        "reporterCode": reporter,
        "partnerCode": partner,
        "period": period,
        "cmdCode": cmdcode if cmdcode != "0" else "TOTAL",
        "flowCode": flow,
        "includeDesc": "True",
    }
    try:
        url = f"{_BASE}/getTarifflineData"
        r = requests.get(url, headers=_HEADERS, params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        logger.warning(f"Comtrade {reporter}→{partner} HS{cmdcode}: {e}")
        return pd.DataFrame()


def _fetch_simple(reporter_code: str, partner_code: str, hs_code: str,
                  flow_code: str = "M") -> dict:
    """Simplified Comtrade query using the public preview endpoint."""
    try:
        # Use the simpler public data endpoint
        url = (f"https://comtradeapi.un.org/public/v1/preview/C/A/HS?"
               f"reporterCode={reporter_code}&partnerCode={partner_code}"
               f"&cmdCode={hs_code}&flowCode={flow_code}&period=2023")
        r = requests.get(url, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return {}
        row = data[0] if data else {}
        return {
            "primary_value": float(row.get("primaryValue", 0) or 0),
            "qty": float(row.get("qty", 0) or 0),
            "period": str(row.get("period", "")),
        }
    except Exception as e:
        logger.warning(f"Comtrade simple {reporter_code} HS{hs_code}: {e}")
        return {}


def compute_comtrade_features() -> pd.DataFrame:
    """Compute trade flow features for the feature store.

    These are annual/quarterly numbers; YoY change is the primary signal.
    Returns one-row DataFrame suitable for the feature store.
    """
    now = datetime.now(timezone.utc)
    row: dict = {"asof_ts": now, "knowledge_ts": now}

    queries = [
        # (reporter, partner, hs, flow, name)
        ("842", "0", "8542", "M", "us_semi_imports"),     # US semiconductor imports
        ("156", "0", "8542", "M", "china_semi_imports"),  # China semiconductor imports
        ("842", "0", "2709", "M", "us_crude_imports"),    # US crude oil imports
    ]

    for reporter, partner, hs, flow, name in queries:
        result = _fetch_simple(reporter, partner, hs, flow)
        if result:
            row[f"comtrade_{name}_value"] = result.get("primary_value", 0)
            row[f"comtrade_{name}_period"] = result.get("period", "")

    return pd.DataFrame([row]) if len(row) > 2 else pd.DataFrame()
