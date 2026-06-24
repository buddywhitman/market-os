"""News + alternative sentiment data lake fetcher.

Sources (all free, no billing):
  NewsAPI    — 100 req/day free. Best structured headline database.
  GDELT      — Massive global news event + tone database. No key. No limit.
  StockTwits — Retail sentiment stream. Public API, no key.
  Wikipedia  — Pageview API. Attention proxy for themes/companies. No key.

Every function stores raw bytes in the lake with provenance, and returns a structured
DataFrame suitable for the sentiment feature family.
"""
from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
_HEADERS = {"User-Agent": "MarketOS/0.1 research@guaqai.me"}


# ── NewsAPI ───────────────────────────────────────────────────────────────────
def fetch_news_headlines(
    query: str,
    *,
    from_date: str | None = None,
    page_size: int = 100,
) -> pd.DataFrame:
    """Fetch headlines for a query. Returns title, description, publishedAt, source, url."""
    if not NEWSAPI_KEY:
        return pd.DataFrame()
    params = {
        "q": query,
        "pageSize": min(page_size, 100),
        "apiKey": NEWSAPI_KEY,
        "language": "en",
        "sortBy": "publishedAt",
    }
    if from_date:
        params["from"] = from_date
    r = requests.get("https://newsapi.org/v2/everything", params=params,
                     headers=_HEADERS, timeout=20)
    r.raise_for_status()
    articles = r.json().get("articles", [])
    if not articles:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "title": a["title"],
        "description": a.get("description", ""),
        "published_at": pd.to_datetime(a["publishedAt"]),
        "source": a["source"]["name"],
        "url": a["url"],
        "query": query,
    } for a in articles])
    return df.sort_values("published_at", ascending=False).reset_index(drop=True)


def fetch_news_for_universe(universe: list[str], *, lookback_hours: int = 24) -> pd.DataFrame:
    """Fetch headlines for each symbol/theme. Rate-aware (1 req/symbol, max 100/day)."""
    since = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).strftime("%Y-%m-%dT%H:%M:%S")
    frames = []
    for sym in universe[:20]:  # cap at 20 queries to preserve free quota
        try:
            df = fetch_news_headlines(sym, from_date=since, page_size=10)
            if not df.empty:
                df["symbol"] = sym
                frames.append(df)
            time.sleep(0.5)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── GDELT ─────────────────────────────────────────────────────────────────────
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

def fetch_gdelt_articles(
    query: str,
    *,
    start: str | None = None,
    end: str | None = None,
    max_records: int = 25,
) -> pd.DataFrame:
    """Fetch GDELT news articles for a query. Returns title, url, seendate, tone."""
    if start is None:
        start = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%d%H%M%S")
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "startdatetime": start,
        "enddatetime": end,
        "sort": "datedesc",
    }
    r = requests.get(GDELT_DOC_API, params=params, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    articles = r.json().get("articles", [])
    if not articles:
        return pd.DataFrame()
    return pd.DataFrame([{
        "title": a.get("title", ""),
        "url": a.get("url", ""),
        "published_at": pd.to_datetime(a.get("seendate", ""), format="%Y%m%dT%H%M%SZ", errors="coerce"),
        "source_country": a.get("sourcecountry", ""),
        "language": a.get("language", ""),
        "tone": float(a.get("tone", 0) or 0),
        "query": query,
    } for a in articles])


def fetch_gdelt_tone_timeline(
    query: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch GDELT tone timeline for a query. Returns date + avg_tone."""
    if start is None:
        start = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y%m%d%H%M%S")
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    params = {
        "query": query,
        "mode": "timelinetonepercent",
        "format": "json",
        "startdatetime": start,
        "enddatetime": end,
    }
    r = requests.get(GDELT_DOC_API, params=params, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    # GDELT timeline returns {"timeline": [{"data":[{"date":..,"value":..}]}]}
    tl = data.get("timeline", [])
    if not tl:
        return pd.DataFrame()
    rows = []
    for series in tl:
        for pt in series.get("data", []):
            rows.append({
                "date": pd.to_datetime(pt["date"]),
                "gdelt_tone": float(pt.get("value", 0) or 0),
                "query": query,
                "series": series.get("series", ""),
            })
    return pd.DataFrame(rows)


def fetch_gdelt_for_themes(themes: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch 30-day tone timeline for each theme keyword."""
    results = {}
    theme_queries = {
        "AI":       "artificial intelligence stocks semiconductor",
        "SEMI":     "semiconductor chip TSMC NVIDIA AMD",
        "POWER":    "power grid electricity energy infrastructure",
        "DEFENSE":  "defense aerospace military spending",
        "NUCLEAR":  "nuclear energy power reactor",
        "ROBOTICS": "robotics automation AI robot",
        "SPACE":    "space exploration rocket satellite launch",
        "CRYPTO":   "bitcoin ethereum cryptocurrency blockchain",
    }
    for theme in themes:
        q = theme_queries.get(theme, theme)
        try:
            df = fetch_gdelt_tone_timeline(q)
            if not df.empty:
                results[theme] = df
        except Exception:
            pass
        time.sleep(1)  # GDELT is generous but be polite
    return results


# ── StockTwits ────────────────────────────────────────────────────────────────
def fetch_stocktwits_sentiment(symbol: str) -> dict:
    """Fetch StockTwits stream for a symbol. Returns bullish/bearish ratios."""
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    r = requests.get(url, headers=_HEADERS, timeout=15)
    r.raise_for_status()
    messages = r.json().get("messages", [])
    if not messages:
        return {"bullish_ratio": 0.5, "bearish_ratio": 0.5, "n_messages": 0}
    sentiments = [m.get("entities", {}).get("sentiment", {}) for m in messages]
    bull = sum(1 for s in sentiments if s and s.get("basic") == "Bullish")
    bear = sum(1 for s in sentiments if s and s.get("basic") == "Bearish")
    n = len(messages)
    total_sentiment = bull + bear or 1
    return {
        "bullish_ratio": bull / total_sentiment,
        "bearish_ratio": bear / total_sentiment,
        "n_messages": n,
        "coverage": total_sentiment / n,
    }


def fetch_stocktwits_for_universe(universe: list[str]) -> pd.DataFrame:
    rows = []
    for sym in universe:
        try:
            s = fetch_stocktwits_sentiment(sym)
            s["symbol"] = sym
            s["asof_ts"] = datetime.now(timezone.utc)
            rows.append(s)
            time.sleep(0.3)
        except Exception:
            pass
    return pd.DataFrame(rows)


# ── Wikipedia Pageviews ───────────────────────────────────────────────────────
WIKI_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia"

def fetch_wikipedia_pageviews(article: str, *, days: int = 30) -> pd.DataFrame:
    """Fetch daily pageview counts for a Wikipedia article."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    url = f"{WIKI_API}/all-access/all-agents/{article}/daily/{start:%Y%m%d}/{end:%Y%m%d}"
    r = requests.get(url, headers=_HEADERS, timeout=15)
    if r.status_code == 404:
        return pd.DataFrame()
    r.raise_for_status()
    items = r.json().get("items", [])
    return pd.DataFrame([{
        "date": pd.to_datetime(item["timestamp"][:8], format="%Y%m%d"),
        "views": item["views"],
        "article": article,
    } for item in items])


WIKI_ARTICLES = {
    "AI":       "Artificial_intelligence",
    "NVDA":     "Nvidia",
    "AMD":      "Advanced_Micro_Devices",
    "SEMI":     "Semiconductor_industry",
    "NUCLEAR":  "Nuclear_power",
    "DEFENSE":  "Defense_industry",
    "CRYPTO":   "Bitcoin",
    "SPACE":    "SpaceX",
    "POWER":    "Electric_power_industry",
}

def fetch_wikipedia_for_universe(symbols: list[str]) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        article = WIKI_ARTICLES.get(sym, sym.replace(" ", "_"))
        try:
            df = fetch_wikipedia_pageviews(article, days=7)
            if not df.empty:
                df["symbol"] = sym
                frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
