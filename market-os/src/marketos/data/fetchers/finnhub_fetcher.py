"""Finnhub fetcher — 60 req/min free tier.

Register at: https://finnhub.io/register
Set env var: FINNHUB_API_KEY

Signals:
  - Earnings surprises (EPS actual vs estimate) — strongest single-stock signal
  - Insider sentiment (buy/sell ratio from Form 4 filings)
  - Company news sentiment scores (built-in NLP)
  - Peers comparison for relative positioning
  - Social sentiment (Reddit/Twitter aggregates)
  - Patent grants/R&D filings via analyst estimates
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone, date, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
_BASE = "https://finnhub.io/api/v1"

_REQUEST_GAP = 1.1  # seconds between calls to stay under 60/min


def _get(endpoint: str, params: dict | None = None) -> dict | list | None:
    if not FINNHUB_API_KEY:
        return None
    p = {"token": FINNHUB_API_KEY, **(params or {})}
    try:
        r = requests.get(f"{_BASE}/{endpoint}", params=p, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"Finnhub {endpoint}: {e}")
        return None


def fetch_earnings_surprises(symbol: str, lookback_quarters: int = 8) -> pd.DataFrame:
    """Fetch EPS actual vs estimate for last N quarters.

    Earnings surprise = (actual - estimate) / |estimate|
    Standardized unexpected earnings (SUE) momentum is a documented anomaly.
    """
    data = _get("stock/earnings", {"symbol": symbol, "limit": lookback_quarters})
    time.sleep(_REQUEST_GAP)
    if not data:
        return pd.DataFrame()
    rows = []
    for item in data:
        estimate = item.get("estimate")
        actual = item.get("actual")
        if estimate is None or actual is None:
            continue
        sue = (actual - estimate) / max(abs(estimate), 0.01)
        rows.append({
            "period": item.get("period", ""),
            "eps_actual": float(actual),
            "eps_estimate": float(estimate),
            "eps_surprise": float(actual - estimate),
            "sue": float(sue),
            "revenue_actual": float(item.get("revenueActual") or 0),
            "revenue_estimate": float(item.get("revenueEstimate") or 0),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_insider_sentiment(symbol: str) -> dict:
    """Fetch aggregate insider buy/sell sentiment (3-month window).

    MSPR (Monthly Share Purchase Ratio) from Finnhub aggregates Form 4 filings.
    Positive MSPR → net buying → bullish insider signal.
    """
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=90)).isoformat()
    data = _get("stock/insider-sentiment", {"symbol": symbol, "from": start, "to": end})
    time.sleep(_REQUEST_GAP)
    if not data or not data.get("data"):
        return {}
    items = data["data"]
    total_buy = sum(d.get("purchase", 0) for d in items)
    total_sell = sum(d.get("sale", 0) for d in items)
    total_change = sum(d.get("change", 0) for d in items)
    mspr_avg = sum(d.get("mspr", 0) for d in items) / max(len(items), 1)
    return {
        "insider_purchase_3m": float(total_buy),
        "insider_sale_3m": float(total_sell),
        "insider_net_3m": float(total_buy - total_sell),
        "insider_mspr_avg": float(mspr_avg),
        "insider_change_3m": float(total_change),
    }


def fetch_company_news_sentiment(symbol: str, days: int = 7) -> dict:
    """Fetch Finnhub's built-in news sentiment for a symbol."""
    data = _get("news-sentiment", {"symbol": symbol})
    time.sleep(_REQUEST_GAP)
    if not data:
        return {}
    buzz = data.get("buzz", {})
    sentiment = data.get("sentiment", {})
    return {
        "fh_news_articles_last_week": float(buzz.get("articlesInLastWeek", 0)),
        "fh_news_buzz_weekly_avg": float(buzz.get("weeklyAverage", 0)),
        "fh_news_buzz_score": float(buzz.get("buzz", 0)),
        "fh_news_bullish_pct": float(sentiment.get("bullishPercent", 0.5)),
        "fh_news_bearish_pct": float(sentiment.get("bearishPercent", 0.5)),
        "fh_news_score": float(data.get("companyNewsScore", 0)),
        "fh_social_sentiment": float(data.get("sectorAverageBullishPercent", 0.5)),
    }


def fetch_recommendation_trends(symbol: str) -> dict:
    """Fetch analyst recommendation distribution (strong buy → strong sell)."""
    data = _get("stock/recommendation", {"symbol": symbol})
    time.sleep(_REQUEST_GAP)
    if not data or not isinstance(data, list) or not data:
        return {}
    latest = data[0]
    strong_buy = float(latest.get("strongBuy", 0))
    buy = float(latest.get("buy", 0))
    hold = float(latest.get("hold", 0))
    sell = float(latest.get("sell", 0))
    strong_sell = float(latest.get("strongSell", 0))
    total = strong_buy + buy + hold + sell + strong_sell
    if total == 0:
        return {}
    bull_frac = (strong_buy + buy) / total
    bear_frac = (sell + strong_sell) / total
    score = (strong_buy * 2 + buy * 1 + hold * 0 + sell * -1 + strong_sell * -2) / total
    return {
        "analyst_strong_buy": strong_buy,
        "analyst_buy": buy,
        "analyst_hold": hold,
        "analyst_sell": sell,
        "analyst_strong_sell": strong_sell,
        "analyst_bull_frac": bull_frac,
        "analyst_bear_frac": bear_frac,
        "analyst_score": score,
        "analyst_total": total,
    }


def fetch_finnhub_universe(universe: list[str]) -> pd.DataFrame:
    """Fetch earnings surprises + insider sentiment + news for the full universe.

    Returns one row per symbol with all finnhub signals flattened.
    Rate-limited to ~1 call/second to stay within 60/min.
    """
    if not FINNHUB_API_KEY:
        logger.info("FINNHUB_API_KEY not set — skipping Finnhub ingest")
        return pd.DataFrame()

    now = datetime.now(timezone.utc)
    rows = []
    for sym in universe:
        try:
            row: dict = {"symbol": sym, "asof_ts": now, "knowledge_ts": now}
            # Earnings surprises — summarize last 4 quarters
            eps_df = fetch_earnings_surprises(sym, lookback_quarters=4)
            if not eps_df.empty:
                row["eps_sue_latest"] = float(eps_df["sue"].iloc[0])
                row["eps_sue_mean_4q"] = float(eps_df["sue"].mean())
                row["eps_sue_trend"] = float(eps_df["sue"].iloc[0] - eps_df["sue"].mean())
                row["eps_beat_streak"] = int((eps_df["sue"] > 0).cumprod().sum())
                row["eps_revenue_surprise_latest"] = float(
                    (eps_df["revenue_actual"].iloc[0] - eps_df["revenue_estimate"].iloc[0])
                    / max(abs(eps_df["revenue_estimate"].iloc[0]), 1.0)
                ) if "revenue_actual" in eps_df.columns else 0.0

            # Insider sentiment
            insider = fetch_insider_sentiment(sym)
            row.update(insider)

            # News sentiment — /news-sentiment is gated on the free tier (verified 2026-06-22:
            # explicit 403 "You don't have access to this resource" on every symbol). Calling
            # it anyway wastes a request + the _REQUEST_GAP sleep per symbol for a guaranteed
            # failure, which was enough to push the whole fetch past the scheduler's timeout.
            # fetch_company_news_sentiment() is left in the module for when the tier upgrades.

            # Analyst recommendations
            reco = fetch_recommendation_trends(sym)
            row.update(reco)

            rows.append(row)
        except Exception as e:
            logger.warning(f"Finnhub {sym}: {e}")

    return pd.DataFrame(rows) if rows else pd.DataFrame()
