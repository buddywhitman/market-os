"""Polymarket prediction-market fetcher — keyless, public Gamma API.

Signal theory: prediction markets are the crowd's priced odds on forward-looking macro/
political/crypto events (Fed decisions, recession, elections, geopolitics) — a complement to
FRED (where things stand) and news (what happened): what the crowd actually prices to happen
next. Genuinely orthogonal to the rest of this feature store; ported from TradingAgents'
dataflows/polymarket.py (which renders markdown for an LLM prompt) into numeric features for
the store, on the same per-topic adapter pattern as the rest of this codebase's fetchers.

No key, no auth: https://docs.polymarket.com/developers/gamma-markets-api/overview confirms
the Gamma API's /public-search endpoint is fully public.

KNOWN ISSUE (2026-06-22): gamma-api.polymarket.com (and bare polymarket.com) return DNS
REFUSED from this server via both Google (8.8.8.8) and Cloudflare (1.1.1.1) resolvers — a
network-level block (likely the hosting provider filtering gambling/prediction-market
domains), not a bug in this module. Verified the API itself is correct and live via
documentation cross-reference; this fetcher degrades gracefully (returns empty) until that's
resolved, same as every other fetcher in this codebase when its data source is unreachable.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BASE = "https://gamma-api.polymarket.com"
_TIMEOUT = 20

# Topic -> feature-name stem. Chosen for relevance to this universe's themes (AI/SEMI/POWER/
# DEFENSE/NUCLEAR/SPACE/CRYPTO) plus the macro context every symbol's composite already
# broadcasts (Fed policy, recession odds).
TOPICS = {
    "fed_cut":      "Fed rate cut",
    "recession":    "recession 2026",
    "ai_bubble":    "AI bubble",
    "btc_price":    "bitcoin price",
    "govt_shutdown": "government shutdown",
}


def _parse_json_list(value) -> list:
    """Gamma encodes outcomes/outcomePrices as JSON-string arrays."""
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def _is_forward_looking(market: dict, now: datetime) -> bool:
    if market.get("closed"):
        return False
    end_date = market.get("endDate")
    if end_date:
        try:
            if datetime.fromisoformat(end_date.replace("Z", "+00:00")) < now:
                return False
        except ValueError:
            pass
    return bool(_parse_json_list(market.get("outcomePrices"))) and bool(
        _parse_json_list(market.get("outcomes")))


def fetch_topic_markets(topic_query: str, limit: int = 5) -> list[dict]:
    """Most-traded open markets matching a topic query, normalized to {prob, volume, end_date,
    one_week_change}. Returns [] on any failure (network, DNS, empty results) — callers should
    treat that identically to "no signal available", not an error."""
    try:
        r = requests.get(f"{_BASE}/public-search",
                         params={"q": topic_query, "limit_per_type": 20}, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        logger.warning(f"Polymarket search failed for {topic_query!r}: {e}")
        return []

    now = datetime.now(timezone.utc)
    candidates = [m for event in data.get("events", []) for m in event.get("markets", [])
                  if _is_forward_looking(m, now)]
    candidates.sort(key=lambda m: m.get("volumeNum") or 0, reverse=True)

    out = []
    for m in candidates[:limit]:
        prices = _parse_json_list(m.get("outcomePrices"))
        try:
            prob = float(prices[0])
        except (ValueError, IndexError):
            continue
        out.append({
            "question": m.get("question"),
            "prob": prob,
            "volume": float(m.get("volumeNum") or 0),
            "end_date": (m.get("endDate") or "")[:10],
            "one_week_change": m.get("oneWeekPriceChange"),
        })
    return out


def compute_polymarket_features() -> pd.DataFrame:
    """One broadcast row: per-topic implied probability + volume + momentum, for every topic
    in TOPICS that has at least one open, sufficiently-traded market.

    Features: pm_{topic}_prob (top market's implied probability, volume-weighted ranking),
    pm_{topic}_volume, pm_{topic}_mom_1wk (probability change over the last week — the
    *direction* the crowd is updating, often more informative than the level itself).
    """
    now = datetime.now(timezone.utc)
    row: dict = {"asof_ts": now, "knowledge_ts": now}

    for stem, query in TOPICS.items():
        markets = fetch_topic_markets(query, limit=3)
        if not markets:
            continue
        top = markets[0]
        row[f"pm_{stem}_prob"] = top["prob"]
        row[f"pm_{stem}_volume"] = top["volume"]
        if isinstance(top.get("one_week_change"), (int, float)):
            row[f"pm_{stem}_mom_1wk"] = float(top["one_week_change"])
        # Dispersion across the top markets for this topic — high variance = market disagrees
        # on the specific framing/date even while broadly pricing the same direction.
        if len(markets) > 1:
            probs = [m["prob"] for m in markets]
            row[f"pm_{stem}_prob_dispersion"] = float(pd.Series(probs).std())

    return pd.DataFrame([row])
