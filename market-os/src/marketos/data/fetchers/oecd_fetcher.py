"""OECD fetcher — free REST API, no key required.

OECD SDMX API: https://data.oecd.org/api/sdmx-json-documentation/

Key datasets:
  - CLI: Composite Leading Indicators — predicts economic turning points 6-9mo ahead
  - CSESM: Consumer Sentiment (household confidence)
  - MEI: Main Economic Indicators — industrial production, CPI, trade balance
  - BLI: Better Life Index (annual)

Countries: US, EU, JP, CN, IN (India via partner data), KR, GB

CLI is the crown jewel: it's designed as a leading indicator and has documented
predictive power for equity markets 3-6 months forward. CLI > 100 = expansion,
< 100 = contraction. Rate of change matters more than level.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BASE = "https://stats.oecd.org/SDMX-JSON/data"
_HEADERS = {"Accept": "application/json", "User-Agent": "MarketOS/0.1 research@guaqai.me"}

# (dataset, key_filter, measure_key)
SERIES = {
    "cli_us":     ("MEI_CLI", "LOLITOAA.USA.M", "LOLITOAA"),
    "cli_eu":     ("MEI_CLI", "LOLITOAA.EU28.M", "LOLITOAA"),
    "cli_jp":     ("MEI_CLI", "LOLITOAA.JPN.M", "LOLITOAA"),
    "cli_cn":     ("MEI_CLI", "LOLITOAA.CHN.M", "LOLITOAA"),
    "cli_kr":     ("MEI_CLI", "LOLITOAA.KOR.M", "LOLITOAA"),
    "consumer_conf_us": ("CS_ESME", "USA.M", None),
    "consumer_conf_eu": ("CS_ESME", "EU28.M", None),
}


def _fetch_oecd(dataset: str, key: str, start_period: str = "2015-01") -> pd.Series:
    """Fetch an OECD SDMX series as a time-indexed pd.Series."""
    url = f"{_BASE}/{dataset}/{key}/all"
    params = {"startPeriod": start_period, "format": "sdmx-json"}
    try:
        r = requests.get(url, headers=_HEADERS, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        structure = data.get("structure", {})
        dataSets = data.get("dataSets", [{}])
        if not dataSets:
            return pd.Series(dtype=float)

        time_periods = structure.get("dimensions", {}).get("observation", [])
        if not time_periods:
            return pd.Series(dtype=float)
        time_dim = next((d for d in time_periods if d.get("id") == "TIME_PERIOD"), None)
        if not time_dim:
            return pd.Series(dtype=float)

        periods = [v.get("name", v.get("id", "")) for v in time_dim.get("values", [])]
        series_data = dataSets[0].get("series", {})
        # Typically one series; take first
        for _, obs_dict in series_data.items():
            obs = obs_dict.get("observations", {})
            records = {}
            for idx_str, values in obs.items():
                idx = int(idx_str)
                if idx < len(periods) and values and values[0] is not None:
                    try:
                        ts = pd.to_datetime(periods[idx]).tz_localize("UTC")
                        records[ts] = float(values[0])
                    except Exception:
                        pass
            if records:
                return pd.Series(records).sort_index()
        return pd.Series(dtype=float)
    except Exception as e:
        logger.warning(f"OECD {dataset}/{key}: {e}")
        return pd.Series(dtype=float)


def compute_oecd_features() -> pd.DataFrame:
    """Compute OECD CLI and confidence features for the feature store.

    CLI features:
    - Level (>100 = expansion phase)
    - 3-month change (acceleration/deceleration)
    - 6-month change (trend)
    - US vs EU divergence
    - Cross-country breadth (% with CLI > 100)
    """
    now = datetime.now(timezone.utc)
    row: dict = {"asof_ts": now, "knowledge_ts": now}

    cli_values = {}
    for name, (dataset, key, _) in SERIES.items():
        s = _fetch_oecd(dataset, key)
        if s.empty:
            continue
        latest = float(s.iloc[-1])
        row[f"oecd_{name}"] = latest

        if len(s) >= 3:
            row[f"oecd_{name}_3m_chg"] = latest - float(s.iloc[-4])
        if len(s) >= 6:
            row[f"oecd_{name}_6m_chg"] = latest - float(s.iloc[-7])
        if len(s) >= 12:
            row[f"oecd_{name}_yoy"] = latest - float(s.iloc[-13])

        if name.startswith("cli_"):
            cli_values[name] = latest

    # Cross-country CLI features
    if cli_values:
        vals = list(cli_values.values())
        row["oecd_cli_breadth"] = sum(1 for v in vals if v > 100) / len(vals)
        row["oecd_cli_global_mean"] = sum(vals) / len(vals)
        if "oecd_cli_us" in row and "oecd_cli_eu" in row:
            row["oecd_cli_us_eu_spread"] = row["oecd_cli_us"] - row["oecd_cli_eu"]

    return pd.DataFrame([row])
