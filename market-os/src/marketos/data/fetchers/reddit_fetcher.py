"""Reddit sentiment fetcher — uses public JSON endpoints (no OAuth required).

Reddit's `.json` endpoints require no authentication for public subreddits.
We extract mentions + sentiment from:
  - r/wallstreetbets  — retail momentum crowd
  - r/stocks          — retail fundamental discussion
  - r/investing       — longer-horizon discussion
  - r/options         — options-related activity

Signal theory: WSB mention spikes precede short-squeeze events (GME 2021, AMC).
Cross-subreddit divergence (WSB bull + r/investing skeptical) is a fade signal.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone

import os
import pandas as pd
import requests

logger = logging.getLogger(__name__)

REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "options"]
_USER_AGENT = "MarketOS/0.1 (financial research; contact: research@guaqai.me)"
_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_API_BASE = "https://oauth.reddit.com"
_PUBLIC_BASE = "https://www.reddit.com"

_cached_token: dict = {}


def _get_oauth_token() -> str:
    """Get application-only OAuth token (client credentials flow)."""
    import time
    global _cached_token
    now = time.time()
    if _cached_token.get("expires_at", 0) > now + 60:
        return _cached_token.get("access_token", "")
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        return ""
    try:
        r = requests.post(
            _TOKEN_URL,
            auth=(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": _USER_AGENT},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        _cached_token = {
            "access_token": data.get("access_token", ""),
            "expires_at": now + data.get("expires_in", 3600),
        }
        return _cached_token["access_token"]
    except Exception as e:
        logger.warning(f"Reddit OAuth: {e}")
        return ""

BULL_WORDS = frozenset([
    "bullish", "buy", "long", "calls", "moon", "rocket", "squeeze", "breakout",
    "undervalued", "accumulate", "strong buy", "beat", "upgrade",
])
BEAR_WORDS = frozenset([
    "bearish", "sell", "short", "puts", "dump", "crash", "bubble", "overvalued",
    "miss", "downgrade", "cut", "fraud", "bankruptcy",
])


def _score_text(text: str) -> float:
    """Simple lexical sentiment score in [-1, 1]."""
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    bull = len(words & BULL_WORDS)
    bear = len(words & BEAR_WORDS)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


def _fetch_via_rss(subreddit: str) -> pd.DataFrame:
    """Fallback: parse subreddit RSS feed (always public, no auth needed)."""
    try:
        url = f"https://www.reddit.com/r/{subreddit}/hot.rss"
        r = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=15)
        r.raise_for_status()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        rows = []
        for entry in root.findall(".//atom:entry", ns):
            title = entry.findtext("atom:title", "", ns)
            rows.append({
                "subreddit": subreddit,
                "title": title, "score": 0, "upvote_ratio": 0.5,
                "num_comments": 0, "text": "",
                "created_utc": datetime.now(timezone.utc),
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as e:
        logger.warning(f"Reddit RSS r/{subreddit}: {e}")
        return pd.DataFrame()


def fetch_subreddit_posts(subreddit: str, limit: int = 50) -> pd.DataFrame:
    """Fetch recent hot posts from a subreddit via OAuth API."""
    token = _get_oauth_token()
    if token:
        url = f"{_API_BASE}/r/{subreddit}/hot"
        headers = {"Authorization": f"bearer {token}", "User-Agent": _USER_AGENT}
    else:
        # Fallback to RSS (always public, returns JSON-convertible XML)
        return _fetch_via_rss(subreddit)
    try:
        r = requests.get(url, headers=headers, params={"limit": limit}, timeout=15)
        r.raise_for_status()
        posts = r.json().get("data", {}).get("children", [])
        rows = []
        for p in posts:
            d = p.get("data", {})
            rows.append({
                "subreddit": subreddit,
                "title": d.get("title", ""),
                "score": int(d.get("score", 0)),
                "upvote_ratio": float(d.get("upvote_ratio", 0.5)),
                "num_comments": int(d.get("num_comments", 0)),
                "text": d.get("selftext", ""),
                "created_utc": datetime.fromtimestamp(
                    d.get("created_utc", 0), tz=timezone.utc),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        logger.warning(f"Reddit r/{subreddit}: {e}")
        return pd.DataFrame()
    finally:
        time.sleep(1.0)  # be polite


def extract_symbol_mentions(df: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    """Count ticker mentions in posts and score sentiment per mention.

    Returns one row per (subreddit, symbol) pair with:
    - mention_count: raw mention count
    - weighted_score: upvote-weighted
    - avg_sentiment: mean lexical sentiment across mentioning posts
    - avg_comments: avg comment count (discussion depth signal)
    """
    rows = []
    for sym in universe:
        pattern = re.compile(rf'\b{re.escape(sym)}\b', re.IGNORECASE)
        for sub, group in df.groupby("subreddit"):
            matches = group[
                group["title"].str.contains(pattern, na=False) |
                group["text"].str.contains(pattern, na=False)
            ]
            if matches.empty:
                continue
            combined_text = " ".join(matches["title"].fillna("") + " " + matches["text"].fillna(""))
            rows.append({
                "symbol": sym,
                "subreddit": sub,
                "mention_count": len(matches),
                "weighted_score": float((matches["score"] * matches["upvote_ratio"]).sum()),
                "avg_sentiment": _score_text(combined_text),
                "avg_comments": float(matches["num_comments"].mean()),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_reddit_features(universe: list[str]) -> pd.DataFrame:
    """Fetch Reddit sentiment and aggregate into per-symbol feature row.

    Returns one row per symbol with:
    - reddit_mention_total: across all subreddits
    - reddit_wsb_mentions: WSB-specific (highest signal for squeezes)
    - reddit_sentiment_mean: mean sentiment across subreddits
    - reddit_wsb_sentiment: WSB-specific sentiment
    - reddit_engagement_score: upvote-weighted engagement
    - reddit_bullish_subreddits: count of subreddits with positive avg sentiment
    """
    all_posts = []
    for sub in SUBREDDITS:
        df = fetch_subreddit_posts(sub, limit=100)
        if not df.empty:
            all_posts.append(df)

    if not all_posts:
        return pd.DataFrame()

    combined = pd.concat(all_posts, ignore_index=True)
    now = datetime.now(timezone.utc)

    mention_df = extract_symbol_mentions(combined, universe)
    if mention_df.empty:
        return pd.DataFrame()

    rows = []
    for sym in universe:
        sym_data = mention_df[mention_df["symbol"] == sym]
        if sym_data.empty:
            continue

        wsb = sym_data[sym_data["subreddit"] == "wallstreetbets"]
        row: dict = {
            "symbol": sym,
            "asof_ts": now,
            "knowledge_ts": now,
            "reddit_mention_total": int(sym_data["mention_count"].sum()),
            "reddit_wsb_mentions": int(wsb["mention_count"].sum()) if not wsb.empty else 0,
            "reddit_sentiment_mean": float(sym_data["avg_sentiment"].mean()),
            "reddit_wsb_sentiment": float(wsb["avg_sentiment"].iloc[0]) if not wsb.empty else 0.0,
            "reddit_engagement_score": float(sym_data["weighted_score"].sum()),
            "reddit_avg_comments": float(sym_data["avg_comments"].mean()),
            "reddit_bullish_subreddits": int((sym_data["avg_sentiment"] > 0.1).sum()),
            "reddit_bearish_subreddits": int((sym_data["avg_sentiment"] < -0.1).sum()),
            "reddit_cross_sub_divergence": float(sym_data["avg_sentiment"].std()),
        }
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()
