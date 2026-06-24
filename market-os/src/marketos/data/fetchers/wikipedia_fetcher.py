"""Wikipedia pageviews fetcher — keyless, Wikimedia REST API.

API: https://wikimedia.org/api/rest_v1/metrics/pageviews/

Signal theory: pageviews are a direct measure of public attention/interest.
  - Company pages: attention spikes often precede earnings run-up or news
  - Technology concept pages: "artificial intelligence", "nuclear power", "cryptocurrency"
    → sector-level momentum/crowding signal
  - Macro-fear pages: "recession", "inflation", "bank run" → risk-off proxy

Research: Moat et al. 2013 (Scientific Reports), Preis et al. 2013 (Nature).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews"
_HEADERS = {"User-Agent": "MarketOS/0.1 (financial research; research@guaqai.me)"}

# ticker → Wikipedia article title
TICKER_PAGES = {
    "NVDA":  "Nvidia",
    "AMD":   "Advanced_Micro_Devices",
    "AVGO":  "Broadcom_Inc.",
    "MSFT":  "Microsoft",
    "PLTR":  "Palantir_Technologies",
    "COIN":  "Coinbase",
    "MSTR":  "MicroStrategy",
    "RKLB":  "Rocket_Lab",
    "PATH":  "UiPath",
    "LMT":   "Lockheed_Martin",
    "RTX":   "RTX_Corporation",
    "CCJ":   "Cameco",
}

# Thematic pages — sector/concept-level attention
THEME_PAGES = {
    "ai_attention":      "Artificial_intelligence",
    "semiconductor":     "Semiconductor",
    "nuclear_energy":    "Nuclear_power",
    "bitcoin_attention": "Bitcoin",
    "crypto_attention":  "Cryptocurrency",
    "recession_fear":    "Recession",
    "inflation_fear":    "Inflation",
    "space_economy":     "Commercial_spaceflight",
    "defense_spending":  "Military_budget",
    "robotics_attention":"Robotics",
}


def _fetch_pageviews(article: str, days: int = 30) -> pd.Series:
    """Fetch daily pageviews for a Wikipedia article. Returns pd.Series indexed by date."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    url = (
        f"{_BASE}/per-article/en.wikipedia/all-access/all-agents/"
        f"{article}/daily/"
        f"{start.strftime('%Y%m%d')}00/{end.strftime('%Y%m%d')}00"
    )
    try:
        r = requests.get(url, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        items = r.json().get("items", [])
        records = {}
        for item in items:
            ts_str = item.get("timestamp", "")
            views = item.get("views", 0)
            if ts_str:
                ts = pd.to_datetime(ts_str, format="%Y%m%d%H").tz_localize("UTC")
                records[ts] = int(views)
        return pd.Series(records).sort_index()
    except Exception as e:
        logger.debug(f"Wikipedia {article}: {e}")
        return pd.Series(dtype=float)


def _compute_attention_features(article: str, prefix: str, days: int = 30) -> dict:
    """Compute z-score and momentum from pageview series."""
    s = _fetch_pageviews(article, days=days)
    if s.empty or len(s) < 7:
        return {}
    latest = float(s.iloc[-1])
    mean_7 = float(s.iloc[-7:].mean())
    mean_all = float(s.mean())
    std_all = float(s.std())
    result = {
        f"{prefix}_views_latest": latest,
        f"{prefix}_views_7d_mean": mean_7,
        f"{prefix}_views_7d_vs_mean": mean_7 / max(mean_all, 1),
    }
    if std_all > 0:
        result[f"{prefix}_views_z"] = (latest - mean_all) / std_all
        result[f"{prefix}_views_7d_z"] = (mean_7 - mean_all) / std_all
    if len(s) >= 14:
        prev_7d = float(s.iloc[-14:-7].mean())
        result[f"{prefix}_views_7d_mom"] = (mean_7 - prev_7d) / max(prev_7d, 1)
    return result


def compute_wikipedia_features(universe: list[str], days: int = 30) -> pd.DataFrame:
    """Compute Wikipedia attention features for tickers and themes.

    Returns two frames:
    1. Per-symbol frame (one row per symbol with company page features)
    2. Aggregate market frame (one row with theme/concept-level attention)
    Both are returned as a concatenated DataFrame for the caller to split.
    """
    now = datetime.now(timezone.utc)
    sym_rows = []
    for sym in universe:
        if sym not in TICKER_PAGES:
            continue
        article = TICKER_PAGES[sym]
        feats = _compute_attention_features(article, "wiki", days=days)
        if feats:
            feats["symbol"] = sym
            feats["asof_ts"] = now
            feats["knowledge_ts"] = now
            sym_rows.append(feats)

    # Theme-level features → stored under "_attention" symbol
    theme_row: dict = {"symbol": "_attention", "asof_ts": now, "knowledge_ts": now}
    for theme_key, article in THEME_PAGES.items():
        feats = _compute_attention_features(article, f"wiki_{theme_key}", days=days)
        theme_row.update(feats)

    all_rows = sym_rows + ([theme_row] if len(theme_row) > 3 else [])
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
