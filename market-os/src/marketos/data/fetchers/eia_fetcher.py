"""EIA (US Energy Information Administration) fetcher — free API key required.

Register at: https://www.eia.gov/opendata/register.php
Set env var: EIA_API_KEY

Key series:
  PET.WCRSTUS1.W   — Crude oil stocks (weekly, MBBL)
  PET.WGTSTUS1.W   — Gasoline stocks (weekly, MBBL)
  PET.WDISTUS1.W   — Distillate stocks / heating oil (weekly, MBBL)
  NG.NW2EVGW_R48_NUS_MMCF_W.W — Natural gas in storage (weekly, MMCF)
  ELEC.GEN.ALL-US-99.M         — Total US electricity generation (monthly)
  TOTAL.PAPIRIUS.M             — US petroleum imports (monthly)

Surprise signals:
  Weekly inventory vs 4-week rolling mean → directional price signal for:
    - CVX, XOM (crude/distillate → not in UNIVERSE but price-correlated)
    - VST, CEG, ETN (electricity generation/demand)
    - GEV (gas turbines → natgas demand proxy)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

EIA_API_KEY = os.environ.get("EIA_API_KEY", "")
_BASE = "https://api.eia.gov/v2"

# (series_id, friendly_name, frequency)
SERIES_MAP = {
    "crude_stocks":    ("PET.WCRSTUS1.W", "weekly"),
    "gasoline_stocks": ("PET.WGTSTUS1.W", "weekly"),
    "distillate_stocks": ("PET.WDISTUS1.W", "weekly"),
    "natgas_storage":  ("NG.NW2EVGW_R48_NUS_MMCF_W.W", "weekly"),
    "electricity_gen": ("ELEC.GEN.ALL-US-99.M", "monthly"),
}


def _fetch_series(series_id: str, length: int = 52) -> pd.Series:
    """Fetch a single EIA time series. Returns pd.Series indexed by UTC datetime."""
    if not EIA_API_KEY:
        return pd.Series(dtype=float)
    url = f"{_BASE}/seriesid/{series_id}"
    params = {"api_key": EIA_API_KEY, "length": length, "out": "json"}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json().get("response", {}).get("data", [])
        if not data:
            # Try v1 style fallback
            url_v1 = "https://api.eia.gov/series/"
            r = requests.get(url_v1, params={"api_key": EIA_API_KEY, "series_id": series_id,
                                              "num": length}, timeout=20)
            r.raise_for_status()
            series_data = r.json().get("series", [{}])[0].get("data", [])
            records = {row[0]: float(row[1]) for row in series_data if row[1] is not None}
            s = pd.Series(records)
            s.index = pd.to_datetime(s.index).tz_localize("UTC")
            return s.sort_index()

        records = {}
        for row in data:
            period = row.get("period", "")
            val = row.get("value")
            if period and val is not None:
                try:
                    ts = pd.to_datetime(period).tz_localize("UTC")
                    records[ts] = float(val)
                except Exception:
                    pass
        return pd.Series(records).sort_index()
    except Exception as e:
        logger.warning(f"EIA series {series_id}: {e}")
        return pd.Series(dtype=float)


def compute_eia_features() -> pd.DataFrame:
    """Compute EIA inventory surprise features.

    Returns one-row DataFrame with:
    - Latest levels for each series
    - 4-week rolling mean (naive forecast)
    - Surprise = latest - mean (positive = inventory build = bearish for price)
    - z-score of surprise vs 1yr distribution
    - Week-over-week change
    """
    row: dict = {}
    now = datetime.now(timezone.utc)

    for name, (series_id, freq) in SERIES_MAP.items():
        lookback = 54 if freq == "weekly" else 24
        s = _fetch_series(series_id, length=lookback)
        if s.empty:
            continue

        latest = float(s.iloc[-1])
        row[f"eia_{name}_latest"] = latest

        if len(s) >= 4:
            mean_4 = float(s.iloc[-5:-1].mean())  # prior 4 obs (not incl latest)
            row[f"eia_{name}_mean4"] = mean_4
            row[f"eia_{name}_surprise"] = latest - mean_4

        if len(s) >= 2:
            prev = float(s.iloc[-2])
            row[f"eia_{name}_wow"] = latest - prev
            row[f"eia_{name}_wow_pct"] = (latest - prev) / abs(prev) if prev != 0 else 0.0

        if len(s) >= 52:
            hist = s.iloc[-52:]
            surprises = hist.diff(1)
            mu = float(surprises.mean())
            sd = float(surprises.std())
            if sd > 0:
                latest_surprise = row.get(f"eia_{name}_surprise", 0.0)
                row[f"eia_{name}_surprise_z"] = (latest_surprise - mu) / sd

    row["asof_ts"] = now
    row["knowledge_ts"] = now
    return pd.DataFrame([row])
