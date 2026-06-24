"""Calibration study: how long do significant waves actually last in AI/semicon/crypto/
biotech names recently — and does that match the 20-trading-day forward horizon the analog
engine and latent-supervised fit are built around?

Method: a standard zigzag pivot detector. Track the running extreme since the last
confirmed pivot; flag a new pivot once price reverses by more than `threshold` from that
extreme. This is the textbook definition of "significant enough" swings — noise below the
threshold is ignored by construction, which is exactly what "average duration between
significant peaks and dips" requires (a naive local-max/local-min scan would count every
1-day wiggle as a swing).

Run on the server (needs yfinance + real history):
    python -m scripts.swing_duration_study
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Recent-months window. 1y gives enough swings per name for a real distribution while
# staying "recent" — a 5y window would average in pre-AI-cycle regime behavior.
LOOKBACK = "1y"

# Report at multiple thresholds since "significant" is a judgment call — if the conclusion
# only holds at one cherry-picked threshold, it isn't a real finding.
THRESHOLDS = (0.10, 0.15, 0.20)

UNIVERSE_FOR_STUDY = {
    "AI_SEMI": ["NVDA", "AMD", "AVGO", "MSFT", "PLTR", "TSM", "QCOM", "INTC", "MU", "WDC", "STX"],
    "POWER":   ["GEV", "VST", "CEG", "ETN"],
    "DEFENSE": ["LMT", "RTX", "NOC"],
    "SPACE_NUCLEAR_ROBOTICS": ["CCJ", "RKLB", "PATH"],
    "CRYPTO":  ["COIN", "MSTR", "BTC-USD"],
    "AGGRESSIVE_LEV": ["SOXL"],
    "BIOTECH": ["XBI"],  # sector ETF proxy — no biotech names in either sleeve yet
    "BENCHMARK": ["SPY", "QQQ"],
}


def zigzag_pivots(close: pd.Series, threshold: float) -> pd.DataFrame:
    """Return one row per confirmed pivot: date, price, kind ('peak'/'trough')."""
    prices = close.values
    dates = close.index
    pivots = []
    # Seed: assume the first bar is a pivot of unknown kind, direction TBD by the first move.
    extreme_idx, extreme_price = 0, prices[0]
    direction = None  # None until the first confirmed swing sets it
    for i in range(1, len(prices)):
        p = prices[i]
        if direction is None:
            move = (p - extreme_price) / extreme_price
            if move >= threshold:
                pivots.append((dates[extreme_idx], extreme_price, "trough"))
                direction, extreme_idx, extreme_price = "up", i, p
            elif move <= -threshold:
                pivots.append((dates[extreme_idx], extreme_price, "peak"))
                direction, extreme_idx, extreme_price = "down", i, p
            continue
        if direction == "up":
            if p > extreme_price:
                extreme_idx, extreme_price = i, p
            elif (extreme_price - p) / extreme_price >= threshold:
                pivots.append((dates[extreme_idx], extreme_price, "peak"))
                direction, extreme_idx, extreme_price = "down", i, p
        else:  # direction == "down"
            if p < extreme_price:
                extreme_idx, extreme_price = i, p
            elif (p - extreme_price) / extreme_price >= threshold:
                pivots.append((dates[extreme_idx], extreme_price, "trough"))
                direction, extreme_idx, extreme_price = "up", i, p
    return pd.DataFrame(pivots, columns=["date", "price", "kind"])


def pivot_durations(pivots: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Trading-day gap between each consecutive pair of confirmed pivots, labeled by the
    LEG it represents (rally = trough->peak, decline = peak->trough)."""
    if len(pivots) < 2:
        return pd.DataFrame(columns=["leg", "trading_days", "pct_move"])
    idx = {d: i for i, d in enumerate(dates)}
    rows = []
    for i in range(len(pivots) - 1):
        p0, p1 = pivots.iloc[i], pivots.iloc[i + 1]
        days = idx[p1["date"]] - idx[p0["date"]]
        leg = "rally" if p0["kind"] == "trough" else "decline"
        pct = (p1["price"] - p0["price"]) / p0["price"]
        rows.append({"leg": leg, "trading_days": days, "pct_move": pct})
    return pd.DataFrame(rows)


def run_study() -> pd.DataFrame:
    from marketos.data.lake import DataLake
    from marketos.data.fetchers.yfinance_fetcher import fetch_ohlcv
    import os

    lake = DataLake(os.environ.get("DATA_LAKE_ROOT", "data_lake"))
    all_durations = []
    for sector, symbols in UNIVERSE_FOR_STUDY.items():
        for sym in symbols:
            try:
                df = fetch_ohlcv(sym, lake=lake, period=LOOKBACK)
                if df.empty or len(df) < 30:
                    continue
                close = df["close"]
                for thresh in THRESHOLDS:
                    pivots = zigzag_pivots(close, thresh)
                    durs = pivot_durations(pivots, close.index)
                    if durs.empty:
                        continue
                    durs["symbol"] = sym
                    durs["sector"] = sector
                    durs["threshold"] = thresh
                    all_durations.append(durs)
            except Exception as exc:
                print(f"  {sym} failed: {exc}")
    if not all_durations:
        print("No data — cannot run study.")
        return pd.DataFrame()
    return pd.concat(all_durations, ignore_index=True)


def summarize(all_durs: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("SWING-DURATION STUDY — trading days between confirmed significant peaks/troughs")
    print(f"Lookback: {LOOKBACK} · current analog/latent forward horizon: 20 trading days")
    print("=" * 90)
    for thresh in THRESHOLDS:
        sub = all_durs[all_durs["threshold"] == thresh]
        if sub.empty:
            continue
        print(f"\n--- threshold = {thresh:.0%} move to confirm a pivot "
              f"(n={len(sub)} legs across {sub['symbol'].nunique()} symbols) ---")
        for leg in ("rally", "decline"):
            leg_durs = sub[sub["leg"] == leg]["trading_days"]
            if len(leg_durs) == 0:
                continue
            print(f"  {leg:8s}: median={leg_durs.median():5.1f}d  mean={leg_durs.mean():5.1f}d  "
                  f"p25={leg_durs.quantile(0.25):5.1f}d  p75={leg_durs.quantile(0.75):5.1f}d  "
                  f"n={len(leg_durs)}")
        print("  by sector (median trading days, both legs combined):")
        for sector, g in sub.groupby("sector"):
            print(f"    {sector:24s}: median={g['trading_days'].median():5.1f}d  n={len(g)}")

    print("\n" + "-" * 90)
    overall_median = all_durs[all_durs["threshold"] == 0.15]["trading_days"].median()
    print(f"AT 15% THRESHOLD (the middle, most defensible cut): overall median leg duration "
          f"= {overall_median:.1f} trading days.")
    print("READ: if this is well below 20d, the analog/latent 20d horizon is averaging across")
    print("      MULTIPLE complete waves — too slow for 'ride one wave, exit near its peak.'")
    print("      If well above 20d, 20d is cutting waves in half — too fast, exiting mid-wave.")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    durs = run_study()
    if not durs.empty:
        summarize(durs)
