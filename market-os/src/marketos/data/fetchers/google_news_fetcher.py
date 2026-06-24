"""Google News RSS — free headline sentiment for Indian (and any) tickers. Confirmed
reachable from this server. The feed's own copyright notice restricts use to "rendering
... within a personal feed reader for personal, non-commercial use" — this fits that
(a personal research tool reading headlines for one account's own trading decisions, the
same way a human would in an RSS reader), not redistribution.

HONEST SCOPE: there is no LLM-based sentiment scorer anywhere in this codebase to reuse
(checked — `sentiment/` is an empty package; the StockTwits integration uses ITS OWN
pre-computed bullish/bearish ratio, not raw-text scoring). Building LLM-based scoring is a
different cost category than free data fetching (API token spend, however small) — not
assumed here. This module ships a free, zero-dependency keyword heuristic as the default;
upgrading to LLM-based scoring (OpenRouter is already configured in .env) is a deliberate
future decision, not bundled into this pass.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from urllib.parse import quote

import requests

RSS_BASE = "https://news.google.com/rss/search"
_TIMEOUT = 15
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; marketos-research-bot/1.0)"}

# Deliberately crude, finance-context word lists — not a substitute for real NLP, just a
# free zero-cost signal until/unless LLM-based scoring is explicitly added later.
_POSITIVE_WORDS = {
    "surge", "soar", "rally", "jump", "gain", "rise", "beat", "outperform", "upgrade",
    "record", "growth", "profit", "bullish", "strong", "expansion", "win", "order",
    "buy", "upbeat", "robust", "boost",
}
_NEGATIVE_WORDS = {
    "plunge", "crash", "fall", "drop", "loss", "miss", "downgrade", "weak", "decline",
    "bearish", "concern", "probe", "fraud", "default", "cut", "sell", "warning",
    "lawsuit", "scandal", "slump",
}


def fetch_headlines(query: str, *, max_items: int = 20) -> list[dict]:
    """Recent headlines matching `query` (e.g. a company name or ticker). Returns
    [{title, link, pub_date}], oldest-safe empty list on any failure."""
    from bs4 import BeautifulSoup
    url = f"{RSS_BASE}?q={quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "xml")
    except Exception:
        return []
    out = []
    for item in soup.find_all("item")[:max_items]:
        title = item.find("title")
        link = item.find("link")
        pub_date = item.find("pubdate")
        out.append({
            "title": title.get_text(strip=True) if title else "",
            "link": link.get_text(strip=True) if link else "",
            "pub_date": pub_date.get_text(strip=True) if pub_date else "",
        })
    return out


def keyword_sentiment_score(headlines: list[dict]) -> dict:
    """Free, zero-dependency sentiment proxy: net (positive - negative) keyword hits
    across all headline titles, normalized to [-1, 1]. NOT a substitute for real NLP —
    a coarse signal only, explicitly labeled as such everywhere it's stored/displayed."""
    pos_hits = neg_hits = 0
    for h in headlines:
        words = set(re.findall(r"[a-z]+", (h.get("title") or "").lower()))
        pos_hits += len(words & _POSITIVE_WORDS)
        neg_hits += len(words & _NEGATIVE_WORDS)
    total = pos_hits + neg_hits
    score = (pos_hits - neg_hits) / total if total > 0 else 0.0
    now = datetime.now(timezone.utc)
    # asof_ts truncated to today's date — see screener_fetcher.py's identical fix for why
    # (the upsert conflict key is (symbol, asof_ts, family); without truncation, a
    # same-day rerun inserts a duplicate row instead of updating).
    return {
        "headline_count": len(headlines), "positive_hits": pos_hits, "negative_hits": neg_hits,
        "keyword_sentiment_score": round(score, 4),
        "asof_ts": now.replace(hour=0, minute=0, second=0, microsecond=0),
        "knowledge_ts": now,
    }
