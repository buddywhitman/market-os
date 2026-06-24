"""Screener.in — free Indian equity fundamentals + shareholding pattern. Confirmed
reachable from this server (HTTP 200, unlike NSE's own Akamai-blocked site) and
confirmed permitted by robots.txt for `/company/<symbol>/` pages (only user pages and a
few query-param paths are disallowed). Data is server-rendered static HTML — no JS engine
needed, plain requests + BeautifulSoup.

This is the per-stock institutional-ownership data source that DIDN'T exist anywhere else
in this codebase — `nse_fetcher.fetch_nse_fii_dii` is a market-wide AGGREGATE flow number,
not per-stock %. Screener.in's shareholding table gives real per-stock Promoter/FII/DII/
Public percentages, which is exactly what was missing.

Scope note: this covers the TOP ratio summary (Market Cap, P/E, ROCE, ROE, Book Value,
Dividend Yield) + the shareholding pattern table. Deeper data (sales/profit growth
trends, cash flow statements, peer comparison) lives in other page sections with more
complex table structures — not parsed here; a deliberate scope boundary, not an oversight,
to ship a working MVP rather than over-build before seeing whether this data is even used.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import requests

BASE_URL = "https://www.screener.in/company"
_TIMEOUT = 15
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; marketos-research-bot/1.0)"}


def _get_soup(symbol: str):
    from bs4 import BeautifulSoup
    r = requests.get(f"{BASE_URL}/{symbol}/", headers=_HEADERS, timeout=_TIMEOUT)
    if r.status_code != 200:
        return None
    return BeautifulSoup(r.text, "html.parser")


_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _parse_number(text: str) -> float | None:
    """Screener formats numbers like "₹ 1,234.56 Cr." or "12.3%" or "1,234" — extract the
    FIRST numeric token via regex rather than blanket-stripping non-digit characters from
    the whole string. The blanket-strip approach has a real bug (caught by a unit test
    before deploy): "₹ 1,234.56 Cr." has a period in "Cr." too, which survives the strip
    and produces "1234.56." — two decimal points, `float()` raises, silently returns None
    for a value that was actually parseable. Extracting one clean token avoids this and
    any other stray punctuation elsewhere in the string."""
    if not text:
        return None
    match = _NUMBER_RE.search(text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def fetch_company_ratios(symbol: str) -> dict:
    """Top ratio summary: market cap, P/E, ROCE, ROE, book value, dividend yield, etc.
    Returns {} if the page can't be fetched/parsed — graceful, same as every other
    fetcher in this codebase; a missing company page is not an exceptional error."""
    soup = _get_soup(symbol)
    if soup is None:
        return {}
    out: dict = {}
    for li in soup.find_all("li", attrs={"data-source": "default"}):
        name_el = li.find("span", class_="name")
        value_el = li.find("span", class_="value")
        if name_el is None or value_el is None:
            continue
        key = re.sub(r"\s+", "_", name_el.get_text(strip=True).lower())
        key = re.sub(r"[^\w_]", "", key)
        if not key:
            continue
        raw_value = value_el.get_text(strip=True)
        # "High / Low" (and similarly any other "/"-separated range) is TWO numbers, not
        # one — naively stripping non-digits from the whole string concatenates them into
        # garbage (caught running this against real data: "184729600.0" for a real
        # ~12,500/~8,200 range). Split on "/" first; a plain single value has no "/" and
        # falls through to the original single-number path unchanged.
        if "/" in raw_value:
            parts = raw_value.split("/")
            if len(parts) == 2:
                hi, lo = _parse_number(parts[0]), _parse_number(parts[1])
                if hi is not None:
                    out[f"{key}_high"] = hi
                if lo is not None:
                    out[f"{key}_low"] = lo
            continue
        val = _parse_number(raw_value)
        if val is not None:
            out[key] = val
    now = datetime.now(timezone.utc)
    # asof_ts is the upsert conflict key (symbol, asof_ts, family) in MarketosStore —
    # truncating to today's DATE (not full wall-clock time) means a same-day rerun
    # correctly UPDATES this row instead of inserting a new one. Caught running the real
    # job twice in one day (a manual smoke test, then the real job minutes later):
    # without this, fundamentals/sentiment rows accumulate one per RUN, not one per DAY,
    # unlike OHLCV-derived features where asof_ts is naturally the last bar's date and
    # stays stable across same-day reruns. knowledge_ts keeps the precise compute time.
    out["asof_ts"] = now.replace(hour=0, minute=0, second=0, microsecond=0)
    out["knowledge_ts"] = now
    return out


def fetch_shareholding_pattern(symbol: str) -> dict:
    """Latest-quarter Promoter/FII/DII/Public/Government shareholding %. This is the
    per-stock institutional-ownership data source no other fetcher in this codebase has —
    nse_fetcher.fetch_nse_fii_dii is a market-wide aggregate, not per-stock.
    """
    soup = _get_soup(symbol)
    if soup is None:
        return {}
    section = soup.find(id="shareholding")
    if section is None:
        return {}
    out: dict = {}
    _ROW_LABELS = {
        "promoters": "promoter_pct", "fiis": "fii_pct", "diis": "dii_pct",
        "government": "government_pct", "public": "public_pct",
    }
    for row in section.find_all("tr"):
        cells = row.find_all("td")
        header = row.find("th") or (row.find("td") if cells else None)
        if header is None:
            continue
        label = re.sub(r"\s+", " ", header.get_text(strip=True)).lower()
        label = label.replace("+", "").strip()
        matched_key = next((v for k, v in _ROW_LABELS.items() if k in label), None)
        if not matched_key or not cells:
            continue
        # Last cell is the most recent quarter — Screener lists quarters oldest-to-newest.
        latest = _parse_number(cells[-1].get_text(strip=True))
        if latest is not None:
            out[matched_key] = latest
    if out:
        out["fii_plus_dii_pct"] = round(out.get("fii_pct", 0.0) + out.get("dii_pct", 0.0), 2)
    now = datetime.now(timezone.utc)
    # asof_ts is the upsert conflict key (symbol, asof_ts, family) in MarketosStore —
    # truncating to today's DATE (not full wall-clock time) means a same-day rerun
    # correctly UPDATES this row instead of inserting a new one. Caught running the real
    # job twice in one day (a manual smoke test, then the real job minutes later):
    # without this, fundamentals/sentiment rows accumulate one per RUN, not one per DAY,
    # unlike OHLCV-derived features where asof_ts is naturally the last bar's date and
    # stays stable across same-day reruns. knowledge_ts keeps the precise compute time.
    out["asof_ts"] = now.replace(hour=0, minute=0, second=0, microsecond=0)
    out["knowledge_ts"] = now
    return out
