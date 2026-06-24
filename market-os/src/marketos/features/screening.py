"""Daily universe screening — narrows a BROAD candidate list down to a ranked shortlist
using ONLY cheap, yfinance-sourced technical criteria.

Why this exists: the quant sleeve's validated `UNIVERSE` (orchestrator.py) is deliberately
frozen at 19 names — it IS the cross-sectional pool the analog/latent fit was validated
against, so silently growing it would silently change the validated signal (see that
module's docstring). But "only 19 names, ever" was never the intent — the intent was to
validate a methodology on a tractable pool first. This module is the other half: a much
broader CANDIDATE universe, screened daily, so there's real visibility into what's moving
across AI/semis/memory/power/defense/space/biotech/crypto — without corrupting the
validated pool. Promoting a screened candidate INTO `UNIVERSE` is a deliberate, separate
re-validation decision, never automatic.

Operational constraint this respects (discovered the hard way — see project notes):
free-tier per-symbol APIs (Finnhub, Polygon, FMP fundamentals) already throttle at 19
symbols; naively scaling those calls 3-5x would break every rate-limited fetcher. yfinance
OHLCV has no such per-symbol cost at this scale. So screening uses ONLY price/volume —
no fundamentals, no sentiment, no broadcast families — by design, not as a shortcut.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# Min average dollar volume over the lookback window — filters out illiquid names a real
# account couldn't size into/out of without moving the price.
MIN_AVG_DOLLAR_VOLUME = 5_000_000
MIN_PRICE = 3.0
LOOKBACK_DAYS = 65  # ~3 months of trading days — enough for a stable SMA/momentum read


def screen_symbol(ohlcv: pd.DataFrame, *, min_avg_traded_value: float = MIN_AVG_DOLLAR_VOLUME,
                  min_price: float = MIN_PRICE) -> dict | None:
    """Cheap technical screen for one symbol. Returns None if there isn't enough history
    to judge it honestly (yet) — a thin screen is not the same as a failed screen.

    `min_avg_traded_value`/`min_price` default to the USD constants above but are
    overridable so the SAME liquidity/momentum logic applies to a different currency
    (e.g. INR for the NSE screen) without duplicating this function — the criteria are
    currency-agnostic, only the threshold magnitudes differ.
    """
    if ohlcv is None or len(ohlcv) < LOOKBACK_DAYS:
        return None
    recent = ohlcv.iloc[-LOOKBACK_DAYS:]
    close = recent["close"]
    traded_value = (recent["close"] * recent["volume"]).mean()
    price = float(close.iloc[-1])

    passes_liquidity = bool(traded_value >= min_avg_traded_value and price >= min_price)

    mom_63 = float(close.iloc[-1] / close.iloc[0] - 1.0)
    sma20 = close.rolling(20, min_periods=20).mean().iloc[-1]
    sma50 = close.rolling(min(50, len(close)), min_periods=min(50, len(close))).mean().iloc[-1]
    trend_healthy = bool(price > sma20 and sma20 > sma50) if pd.notna(sma20) and pd.notna(sma50) else False

    vol_21 = float(close.pct_change().tail(21).std() * np.sqrt(252)) if len(close) > 21 else None

    return {
        "price": round(price, 2),
        "avg_traded_value": round(float(traded_value), 0),
        "passes_liquidity": passes_liquidity,
        "mom_63d": round(mom_63, 4),
        "trend_healthy": trend_healthy,
        "annualized_vol_21d": round(vol_21, 4) if vol_21 is not None else None,
        # Composite screen score: liquidity is a hard gate (0 if failed), otherwise rank by
        # momentum with a bonus for a healthy trend — cheap proxy for "worth a closer look,"
        # NOT a return forecast (that's the validated UNIVERSE's job, not this module's).
        "screen_score": round(mom_63 + (0.05 if trend_healthy else 0.0), 4) if passes_liquidity else None,
    }


def fetch_bhavcopy_history(symbols: list[str], lookback_days: int = LOOKBACK_DAYS) -> dict[str, pd.DataFrame]:
    """Build per-symbol OHLCV history from NSE bhavcopy for the india screen.

    Cheap by construction: ONE bhavcopy file covers the ENTIRE NSE exchange for that day
    (~2000 symbols), so this is `lookback_days` HTTP calls total, not lookback_days *
    len(symbols) — unlike the yfinance per-symbol screen, there's no per-name API cost
    here at all, which is exactly why a much broader Indian candidate list is viable.
    Walks back day-by-day; NSE publishes nothing on weekends/holidays, so this fetches
    more calendar days than `lookback_days` to land on that many actual trading days.
    """
    from datetime import datetime, timedelta
    from marketos.data.fetchers.nse_fetcher import fetch_bhavcopy, IST

    wanted = set(symbols)
    per_symbol_rows: dict[str, list[dict]] = {s: [] for s in symbols}
    trading_days_found = 0
    calendar_days_back = 0
    max_calendar_days = int(lookback_days * 1.6) + 10  # buffer for weekends/holidays

    while trading_days_found < lookback_days and calendar_days_back < max_calendar_days:
        dt = datetime.now(IST) - timedelta(days=calendar_days_back)
        calendar_days_back += 1
        if dt.weekday() >= 5:
            continue
        day_df = fetch_bhavcopy(dt)
        if day_df.empty:
            continue
        trading_days_found += 1
        day_df = day_df[day_df["symbol"].isin(wanted)]
        for _, row in day_df.iterrows():
            # high/low carried through (not just close/volume) so callers can compute
            # ATR-based stops the same way the US sleeves do — bhavcopy has them for free.
            per_symbol_rows[row["symbol"]].append({
                "date": row["asof_ts"], "close": row["close"], "volume": row["volume"],
                "high": row.get("high", row["close"]), "low": row.get("low", row["close"]),
            })

    out = {}
    for sym, rows in per_symbol_rows.items():
        if not rows:
            continue
        df = pd.DataFrame(rows).sort_values("date").set_index("date")
        out[sym] = df
    return out


def fetch_angelone_history(symbols: list[str], lookback_days: int = LOOKBACK_DAYS) -> dict[str, pd.DataFrame]:
    """Per-symbol OHLCV history via AngelOne SmartAPI — the working replacement for
    `fetch_bhavcopy_history` (NSE's own website actively blocks this server's IP; see
    project notes). Unlike bhavcopy (one file/day covers the whole exchange), this is
    `len(symbols)` API calls, so a small delay between calls respects SmartAPI's
    documented per-second rate limit — symbol count here is the screening universe
    (dozens), not the whole exchange, so this stays well within budget.
    """
    import time
    from marketos.data.fetchers import angelone_fetcher as ao

    auth = ao.login()
    if auth.get("error"):
        return {}
    jwt_token = auth["jwt_token"]
    api_key = os.environ["ANGELONE_API_KEY"]

    tokens = ao.get_symbol_tokens(symbols, cache_dir=os.environ.get("DATA_LAKE_ROOT", "data_lake"))
    from_date = datetime.now() - timedelta(days=int(lookback_days * 1.6) + 10)
    to_date = datetime.now()

    out = {}
    for sym in symbols:
        token = tokens.get(sym)
        if not token:
            continue
        candles = ao.get_historical_candles(sym, token, jwt_token=jwt_token, api_key=api_key,
                                            from_date=from_date, to_date=to_date)
        time.sleep(0.34)  # ~3 req/sec — stay under SmartAPI's documented rate limit
        if not candles:
            continue
        df = pd.DataFrame(candles)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        df.index.name = None  # see angelone_fetcher.fetch_ohlcv_history for why
        out[sym] = df
    return out


def screen_universe(candidates: dict[str, pd.DataFrame], *,
                    min_avg_traded_value: float = MIN_AVG_DOLLAR_VOLUME,
                    min_price: float = MIN_PRICE) -> pd.DataFrame:
    """Screen every (symbol -> ohlcv) pair, return a ranked DataFrame (best screen_score
    first). Symbols that fail liquidity or lack enough history still appear, with
    screen_score=NaN, so the screen is auditable — nothing silently disappears."""
    now = datetime.now(timezone.utc)
    rows = []
    for sym, df in candidates.items():
        result = screen_symbol(df, min_avg_traded_value=min_avg_traded_value, min_price=min_price)
        if result is None:
            rows.append({"symbol": sym, "screen_score": None, "passes_liquidity": False,
                        "reason": "insufficient_history"})
            continue
        rows.append({"symbol": sym, **result})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["asof_ts"] = now
    out["knowledge_ts"] = now
    return out.sort_values("screen_score", ascending=False, na_position="last").reset_index(drop=True)
