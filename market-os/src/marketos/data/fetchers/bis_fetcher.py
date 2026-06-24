"""BIS (Bank for International Settlements) fetcher — free, no API key.

BIS publishes structured datasets in CSV/XML via:
  https://data.bis.org/

Key datasets:
  - CBS: Consolidated Banking Statistics — cross-border bank claims
  - DSR: Debt Service Ratio — debt sustainability signal
  - LBS: Locational Banking Statistics — offshore banking flows
  - PP: Property Prices — global real estate cycle (housing bubble indicator)
  - CREDIT: Total credit to private non-financial sector (credit-to-GDP)
  - EFFECTIVE_ER: Effective exchange rates (BIS broad/narrow)

Credit-to-GDP gap (Basel III early-warning indicator):
  A credit gap > 10% signals banking crisis risk within 3 years.
  This is the most powerful macro-prudential signal in the BIS literature.
"""
from __future__ import annotations

import io
import logging
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BASE = "https://data.bis.org"
_HEADERS = {"User-Agent": "MarketOS/0.1 research@guaqai.me",
            "Accept": "text/csv"}

# BIS dataset key → (endpoint, description)
DATASETS = {
    "total_credit":    "WS_TC_PUB/1.0/data/TC/Q.US.P.A.M.770.XDC.A",
    "credit_gdp_us":   "WS_CREDIT_GAP/1.0/data/CREDIT_GAP/Q.US.H.A",
    "dsr_us":          "WS_DSR/1.0/data/DSR/Q.US.H.A",
    "property_prices": "WS_SPP/1.0/data/SPP/Q.US.R.N",
    "eff_er_usd":      "WS_EER/1.0/data/EER/M.B.B.USD",
}


def _fetch_bis_series(endpoint: str) -> pd.Series:
    """Fetch a BIS SDMX-CSV series."""
    url = f"{_BASE}/{endpoint}?format=csv"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        # SDMX CSV has columns: frequency, various keys, OBS_VALUE, TIME_PERIOD
        if "OBS_VALUE" not in df.columns or "TIME_PERIOD" not in df.columns:
            # Try long-format
            if len(df.columns) >= 2:
                df.columns.values[-2] = "TIME_PERIOD"
                df.columns.values[-1] = "OBS_VALUE"
        if "OBS_VALUE" not in df.columns:
            return pd.Series(dtype=float)
        df = df[["TIME_PERIOD", "OBS_VALUE"]].dropna()
        df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
        df = df.dropna()
        df.index = pd.to_datetime(df["TIME_PERIOD"]).dt.to_period("Q").dt.to_timestamp("Q")
        df.index = df.index.tz_localize("UTC")
        return df["OBS_VALUE"].sort_index()
    except Exception as e:
        logger.warning(f"BIS {endpoint}: {e}")
        return pd.Series(dtype=float)


def compute_bis_features() -> pd.DataFrame:
    """Compute BIS macro-prudential indicators.

    Returns one-row DataFrame with:
    - US credit-to-GDP gap (baseline warning signal)
    - US total private credit growth (YoY %)
    - US debt service ratio (income % going to debt repayment)
    - US real property prices (level + YoY)
    - USD real effective exchange rate
    """
    now = datetime.now(timezone.utc)
    row: dict = {"asof_ts": now, "knowledge_ts": now}

    for name, endpoint in DATASETS.items():
        s = _fetch_bis_series(endpoint)
        if s.empty:
            continue
        latest = float(s.iloc[-1])
        row[f"bis_{name}_latest"] = latest
        if len(s) >= 4:
            row[f"bis_{name}_qoq"] = latest - float(s.iloc[-2])
        if len(s) >= 5:
            row[f"bis_{name}_yoy"] = latest - float(s.iloc[-5])
        if len(s) >= 10:
            mean_10 = float(s.iloc[-10:].mean())
            std_10 = float(s.iloc[-10:].std())
            if std_10 > 0:
                row[f"bis_{name}_z"] = (latest - mean_10) / std_10

    # Credit-to-GDP gap: > 10% = Basel III amber signal, > 20% = red signal
    gap = row.get("bis_credit_gdp_us_latest", 0)
    if gap:
        row["bis_credit_gap_warning"] = int(gap > 10)
        row["bis_credit_gap_crisis"] = int(gap > 20)

    return pd.DataFrame([row])
