"""Google Trends fetcher via pytrends.

Google Trends is a high-signal attention proxy:
  - Rising search interest precedes institutional coverage
  - Theme keywords (AI, nuclear, defense) track retail narrative momentum
  - Company-level searches correlate with earnings-period attention spikes
  - "recession" / "inflation" / "fed rate" queries track macro fear

Rate limits: unofficial API; Google blocks aggressive polling.
Strategy: weekly cadence, small batch sizes (5 keywords/request), 60s sleeps.

If pytrends is not installed, returns empty DataFrames gracefully.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta

import pandas as pd

# pytrends is optional — many environments won't have it
try:
    from pytrends.request import TrendReq
    _PYTRENDS_AVAILABLE = True
except ImportError:
    _PYTRENDS_AVAILABLE = False

# Keyword groups — max 5 per request (Google Trends limit)
THEME_KEYWORDS = {
    "AI": ["artificial intelligence", "AI stocks", "NVIDIA stock", "ChatGPT", "LLM"],
    "SEMI": ["semiconductor stocks", "chip shortage", "TSMC", "AMD stock", "fabless"],
    "POWER": ["nuclear energy stocks", "power grid", "electricity demand", "data center power", "uranium"],
    "DEFENSE": ["defense stocks", "military spending", "drone warfare", "LMT stock", "RTX stock"],
    "CRYPTO": ["bitcoin price", "ethereum", "crypto bull run", "BTC", "crypto fear"],
    "MACRO": ["recession 2025", "inflation rate", "fed rate cut", "yield curve", "dollar index"],
    "TECH_FEAR": ["tech layoffs", "AI bubble", "stock market crash", "bear market", "sell the news"],
    "ENERGY": ["oil price", "energy crisis", "natural gas", "Brent crude", "OPEC"],
}

COMPANY_KEYWORDS = {
    "NVDA": ["NVDA stock", "Nvidia earnings"],
    "MSFT": ["MSFT stock", "Microsoft earnings"],
    "AMD":  ["AMD stock", "AMD earnings"],
    "COIN": ["COIN stock", "Coinbase"],
    "PLTR": ["PLTR stock", "Palantir"],
}


def _build_client(timeout: int = 10, retries: int = 3) -> "TrendReq | None":
    if not _PYTRENDS_AVAILABLE:
        return None
    try:
        return TrendReq(hl="en-US", tz=0, timeout=(timeout, 25), retries=retries, backoff_factor=2)
    except Exception:
        return None


def fetch_trends_group(
    keywords: list[str],
    timeframe: str = "today 3-m",
    geo: str = "",
) -> pd.DataFrame:
    """Fetch Google Trends interest over time for up to 5 keywords.

    Returns DataFrame with DatetimeIndex and one column per keyword (0-100 interest).
    timeframe: 'today 3-m', 'today 12-m', 'today 5-y', 'now 7-d', etc.
    geo: '' = worldwide, 'US' = United States, 'IN' = India
    """
    pt = _build_client()
    if pt is None:
        return pd.DataFrame()
    try:
        kw = keywords[:5]  # hard cap
        pt.build_payload(kw, cat=0, timeframe=timeframe, geo=geo, gprop="")
        df = pt.interest_over_time()
        if df.empty:
            return pd.DataFrame()
        if "isPartial" in df.columns:
            df = df[~df["isPartial"]].drop(columns=["isPartial"])
        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        return df
    except Exception:
        return pd.DataFrame()


def fetch_trends_for_themes(
    themes: list[str] | None = None,
    timeframe: str = "today 3-m",
) -> dict[str, pd.DataFrame]:
    """Fetch trends for all theme keyword groups. Returns {theme: DataFrame}."""
    if not _PYTRENDS_AVAILABLE:
        return {}
    themes_to_fetch = themes or list(THEME_KEYWORDS.keys())
    results = {}
    for theme in themes_to_fetch:
        kw = THEME_KEYWORDS.get(theme, [theme])
        df = fetch_trends_group(kw, timeframe=timeframe)
        if not df.empty:
            results[theme] = df
        time.sleep(60)  # Google will block if you hammer it
    return results


def fetch_trends_for_universe(universe: list[str], timeframe: str = "today 3-m") -> pd.DataFrame:
    """Fetch Google Trends for company-level keywords in the universe."""
    if not _PYTRENDS_AVAILABLE:
        return pd.DataFrame()
    frames = []
    for ticker in universe:
        kw = COMPANY_KEYWORDS.get(ticker, [f"{ticker} stock"])
        df = fetch_trends_group(kw[:2], timeframe=timeframe)  # 2 kw per company
        if not df.empty:
            df["ticker"] = ticker
            frames.append(df)
        time.sleep(60)
    return pd.concat(frames) if frames else pd.DataFrame()


def build_trends_feature_row(
    themes: list[str] | None = None,
    lookback_weeks: int = 4,
) -> pd.DataFrame:
    """Derive scalar features from trend time series for the feature store.

    For each theme, computes:
    - current_interest: latest normalized value (0-100)
    - mom_4w: 4-week momentum (current vs 4 weeks ago)
    - z_score: z-score vs trailing 13 weeks
    """
    if not _PYTRENDS_AVAILABLE:
        return pd.DataFrame()
    all_themes = fetch_trends_for_themes(themes, timeframe="today 3-m")
    now = datetime.now(timezone.utc)
    row = {"asof_ts": now, "knowledge_ts": now}
    for theme, df in all_themes.items():
        # Sum across keywords in the group to get composite interest
        composite = df.sum(axis=1)
        if len(composite) < 2:
            continue
        current = float(composite.iloc[-1])
        four_weeks_ago = float(composite.iloc[-lookback_weeks]) if len(composite) >= lookback_weeks else float(composite.iloc[0])
        trailing = composite.iloc[-13:] if len(composite) >= 13 else composite
        z = (current - float(trailing.mean())) / max(float(trailing.std()), 1e-8)

        row[f"trends_{theme.lower()}_current"] = current
        row[f"trends_{theme.lower()}_mom4w"] = (current - four_weeks_ago) / max(four_weeks_ago, 1)
        row[f"trends_{theme.lower()}_z"] = z
    return pd.DataFrame([row]) if len(row) > 2 else pd.DataFrame()


def fetch_related_queries(keyword: str, top_n: int = 10) -> pd.DataFrame:
    """Fetch rising related queries for a keyword — useful for emerging narrative detection."""
    pt = _build_client()
    if pt is None:
        return pd.DataFrame()
    try:
        pt.build_payload([keyword], timeframe="today 3-m", geo="")
        related = pt.related_queries()
        rising = related.get(keyword, {}).get("rising")
        if rising is not None and not rising.empty:
            rising["keyword"] = keyword
            rising["asof_ts"] = datetime.now(timezone.utc)
            return rising.head(top_n)
    except Exception:
        pass
    return pd.DataFrame()
