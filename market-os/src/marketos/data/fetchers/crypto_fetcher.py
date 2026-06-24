"""Crypto data fetcher — CoinGecko + Binance public API. No keys. No billing.

CoinGecko free tier: 30 req/min, unlimited daily. Best source for:
  - Prices, market cap, volume for 10k+ coins
  - Historical OHLCV
  - DeFi TVL, exchange volumes
  - Fear & Greed index proxy (market cap dominance)

Binance public API: no key, no limit for market data.
  - OHLCV for any pair
  - Order book depth (liquidity)
  - Funding rates (perpetuals)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

_HEADERS = {"User-Agent": "MarketOS/0.1 research@guaqai.me"}
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
BINANCE_BASE = "https://api.binance.com/api/v3"

# CoinGecko coin IDs for our universe
COIN_IDS = {
    "BTC":  "bitcoin",
    "ETH":  "ethereum",
    "SOL":  "solana",
    "BNB":  "binancecoin",
    "XRP":  "ripple",
    "DOGE": "dogecoin",
    "LINK": "chainlink",
    "AVAX": "avalanche-2",
    "MATIC":"matic-network",
    "ARB":  "arbitrum",
}

BINANCE_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]


# ── CoinGecko ─────────────────────────────────────────────────────────────────
def fetch_coingecko_prices(coin_ids: list[str] | None = None) -> pd.DataFrame:
    """Current price, market cap, volume, 24h change for key coins."""
    ids = coin_ids or list(COIN_IDS.values())
    params = {
        "ids": ",".join(ids),
        "vs_currencies": "usd",
        "include_market_cap": "true",
        "include_24hr_vol": "true",
        "include_24hr_change": "true",
        "include_last_updated_at": "true",
    }
    r = requests.get(f"{COINGECKO_BASE}/simple/price", params=params,
                     headers=_HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    rev = {v: k for k, v in COIN_IDS.items()}
    rows = []
    now = datetime.now(timezone.utc)
    for cid, vals in data.items():
        rows.append({
            "symbol": rev.get(cid, cid.upper()),
            "coin_id": cid,
            "price_usd": vals.get("usd", 0),
            "market_cap_usd": vals.get("usd_market_cap", 0),
            "volume_24h_usd": vals.get("usd_24h_vol", 0),
            "change_24h_pct": vals.get("usd_24h_change", 0),
            "asof_ts": now,
        })
    return pd.DataFrame(rows)


def fetch_coingecko_ohlcv(coin_id: str, *, days: int = 365) -> pd.DataFrame:
    """Historical OHLCV for a coin from CoinGecko."""
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    r = requests.get(f"{COINGECKO_BASE}/coins/{coin_id}/ohlc",
                     params=params, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    raw = r.json()  # [[ts_ms, open, high, low, close], ...]
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=["ts_ms", "open", "high", "low", "close"])
    df.index = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df[["open", "high", "low", "close"]].sort_index()


def fetch_coingecko_market_overview() -> pd.DataFrame:
    """Global crypto market: total market cap, BTC dominance, fear proxy."""
    r = requests.get(f"{COINGECKO_BASE}/global", headers=_HEADERS, timeout=20)
    r.raise_for_status()
    d = r.json().get("data", {})
    now = datetime.now(timezone.utc)
    return pd.DataFrame([{
        "total_market_cap_usd":    d.get("total_market_cap", {}).get("usd", 0),
        "total_volume_24h_usd":    d.get("total_volume", {}).get("usd", 0),
        "btc_dominance":           d.get("market_cap_percentage", {}).get("btc", 0),
        "eth_dominance":           d.get("market_cap_percentage", {}).get("eth", 0),
        "active_cryptocurrencies": d.get("active_cryptocurrencies", 0),
        "market_cap_change_24h":   d.get("market_cap_change_percentage_24h_usd", 0),
        "asof_ts": now,
    }])


def fetch_defi_tvl() -> pd.DataFrame:
    """Total DeFi TVL from DeFiLlama (no key, free). Good liquidity proxy."""
    try:
        r = requests.get("https://api.llama.fi/v2/historicalChainTvl", headers=_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        # Returns list of {date: unix_ts, tvl: float}
        rows = []
        for point in data[-365:]:  # last year
            rows.append({
                "date": pd.Timestamp(point["date"], unit="s", tz="UTC"),
                "defi_tvl_usd": float(point.get("tvl", 0)),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# ── Binance Public API ────────────────────────────────────────────────────────
def fetch_binance_ohlcv(
    symbol: str,
    *,
    interval: str = "1d",
    limit: int = 365,
) -> pd.DataFrame:
    """Fetch Binance klines (OHLCV) for a trading pair. No key needed."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(f"{BINANCE_BASE}/klines", params=params, headers=_HEADERS, timeout=20)
    r.raise_for_status()
    raw = r.json()
    df = pd.DataFrame(raw, columns=[
        "ts_open", "open", "high", "low", "close", "volume",
        "ts_close", "quote_vol", "n_trades", "taker_buy_vol",
        "taker_buy_quote_vol", "_ignore",
    ])
    df.index = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
    return df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()


def fetch_binance_funding_rates(symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Funding rates from Binance perps — a market sentiment proxy."""
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": symbol, "limit": 100},
            headers=_HEADERS, timeout=15,
        )
        r.raise_for_status()
        rows = r.json()
        return pd.DataFrame([{
            "ts": pd.Timestamp(row["fundingTime"], unit="ms", tz="UTC"),
            "funding_rate": float(row["fundingRate"]),
            "symbol": symbol,
        } for row in rows]).set_index("ts").sort_index()
    except Exception:
        return pd.DataFrame()


def fetch_crypto_panel() -> pd.DataFrame:
    """Run all crypto fetches and return a unified feature frame."""
    rows = []
    prices = fetch_coingecko_prices()
    overview = fetch_coingecko_market_overview()
    now = datetime.now(timezone.utc)

    # Build a single summary row per call
    summary = {
        "btc_dominance":         float(overview.iloc[0]["btc_dominance"]) if not overview.empty else 50.0,
        "total_market_cap_usd":  float(overview.iloc[0]["total_market_cap_usd"]) if not overview.empty else 0.0,
        "market_cap_change_24h": float(overview.iloc[0]["market_cap_change_24h"]) if not overview.empty else 0.0,
        "asof_ts": now,
        "knowledge_ts": now,
    }
    # Per-coin metrics
    for _, row in prices.iterrows():
        summary[f"{row['symbol']}_price"]   = row["price_usd"]
        summary[f"{row['symbol']}_mcap"]    = row["market_cap_usd"]
        summary[f"{row['symbol']}_vol24h"]  = row["volume_24h_usd"]
        summary[f"{row['symbol']}_chg24h"]  = row["change_24h_pct"]

    # BTC funding rate (latest)
    funding = fetch_binance_funding_rates("BTCUSDT")
    if not funding.empty:
        summary["btc_funding_rate"] = float(funding["funding_rate"].iloc[-1])

    return pd.DataFrame([summary])
