"""Build the QUANT sleeve's daily opportunity set from what's already validated and
stored — the analog engine's outcome distributions (features/market_memory.py) and the
supervised-latent fit — then allocate and size real positions.

This is deliberately a CONNECTOR, not a new model. Per the project's own finding: the
bottleneck shifted from features to execution months ago. Every number below already
exists in Postgres before this module runs; this module's only job is reading it in the
shape `portfolio.allocator.allocate()` and `risk.sizing.position_size()` expect, so the
PM snapshot and dashboard show the real validated signal instead of `rng.uniform()`.

Mapping (analog family, per symbol, written weekly by prioritize_subspace_job):
  expectancy   <- analog_mean_ret_20d   (mean forward 20d return of historical analogs)
  sample_size  <- analog_n_effective    (uniqueness-weighted occurrence count, NOT raw n —
                                          using raw n here would let overlapping-window
                                          analogs masquerade as independent evidence)
  confidence   <- analog_win_rate_20d, discounted by analog_cross_regime_frac (an analog
                                          found by crossing into a different regime than
                                          today's is weaker evidence than a same-regime one)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from marketos.risk.sizing import RiskLimits, position_size, enforce_sector_caps
from marketos.portfolio.allocator import allocate


@dataclass(frozen=True)
class SizedPosition:
    symbol: str
    sector: str
    weight: float
    shares: float
    notional: float
    stop_price: float
    entry_price: float
    expectancy: float
    sample_size: float
    confidence: float


def build_opportunities(
    store, universe: list[str], sector_map: dict[str, str],
) -> pd.DataFrame:
    """Read the stored analog-engine output for each symbol and shape it into the
    `allocator.allocate()` input contract: symbol, sector, expectancy, sample_size,
    confidence. Symbols with no analog snapshot yet (e.g. prioritize_subspace_job hasn't
    run since they were added) are silently skipped — allocate() can't evaluate evidence
    that doesn't exist, and that's the honest behavior, not an error."""
    analog_data = store.get_latest_families(universe, ["analog"])
    rows = []
    for sym in universe:
        analog = analog_data.get((sym, "analog"), {})
        if not analog:
            continue
        expectancy = analog.get("analog_mean_ret_20d")
        sample_size = analog.get("analog_n_effective")
        win_rate = analog.get("analog_win_rate_20d")
        if expectancy is None or sample_size is None or win_rate is None:
            continue
        cross_regime_frac = analog.get("analog_cross_regime_frac", 0.0) or 0.0
        # Same-regime analogs are stronger evidence than cross-regime ones; discount
        # confidence proportionally rather than excluding cross-regime analogs outright
        # (the analog engine already prefers same-regime matches when enough exist).
        confidence = float(win_rate) * (1.0 - 0.5 * float(cross_regime_frac))
        rows.append({
            "symbol": sym,
            "sector": sector_map.get(sym, "UNKNOWN"),
            "expectancy": float(expectancy),
            "sample_size": float(sample_size),
            "confidence": confidence,
        })
    return pd.DataFrame(rows, columns=["symbol", "sector", "expectancy", "sample_size", "confidence"])


def size_positions(
    weighted: pd.DataFrame, prices: dict[str, float], atrs: dict[str, float],
    *, equity: float, limits: RiskLimits,
) -> list[SizedPosition]:
    """Convert allocator candidates into real share counts/stops via risk.sizing, THEN
    re-apply sector/gross caps to the resulting weights (not the allocator's intermediate
    edge-weight, which position_size() independently overrides with its own ATR-risk-based
    number — applying caps only to the discarded edge-weight would make sector caps a
    no-op on the actual sized output). A symbol missing a price or ATR (fetch failed, or
    too new for a 14-day ATR) is skipped rather than sized with a fabricated value."""
    raw: list[SizedPosition] = []
    for _, row in weighted.iterrows():
        sym = row["symbol"]
        price, atr = prices.get(sym), atrs.get(sym)
        if not price or not atr or row["weight"] <= 0:
            continue
        sized = position_size(equity, price, atr, limits=limits, edge_kelly=row["confidence"])
        raw.append(SizedPosition(
            symbol=sym, sector=row["sector"], weight=sized["weight"],
            shares=sized["shares"], notional=sized["notional"], stop_price=sized["stop_price"],
            entry_price=price, expectancy=row["expectancy"], sample_size=row["sample_size"],
            confidence=row["confidence"],
        ))
    if not raw:
        return raw

    sectors = {p.symbol: p.sector for p in raw}
    capped_weights = enforce_sector_caps({p.symbol: p.weight for p in raw}, sectors, limits)
    out = []
    for p in raw:
        new_w = capped_weights[p.symbol]
        scale = (new_w / p.weight) if p.weight > 0 else 0.0
        out.append(SizedPosition(
            symbol=p.symbol, sector=p.sector, weight=new_w, shares=p.shares * scale,
            notional=p.notional * scale, stop_price=p.stop_price, entry_price=p.entry_price,
            expectancy=p.expectancy, sample_size=p.sample_size, confidence=p.confidence,
        ))
    return out


# The analog engine's evidence is `analog_n_effective` — uniqueness-weighted, NOT a raw
# trade count (see features/market_memory.py / project notes: typically 5-15 even with 50
# raw matches, by deliberate design). allocate()'s default min_sample_size=30 assumes raw
# trade counts and would zero out every candidate against this evidence type, every time.
# 5.0 keeps genuinely-thin noise out (an analog count of 0-2) while not discarding the real
# signal this engine actually produces.
ANALOG_MIN_EFFECTIVE_N = 5.0


def build_quant_snapshot(
    store, *, universe: list[str], sector_map: dict[str, str], limits: RiskLimits,
    equity: float, max_positions: int = 15,
) -> dict:
    """End-to-end: stored analogs -> allocate() -> sized positions -> a snapshot dict ready
    for `MarketosStore.upsert_portfolio_snapshot`. Also fetches each candidate's latest
    close price + ATR(14) from the lake-backed fetcher (not stored in the feature vector
    itself) so sizing has real, current numbers — not last week's snapshot price.
    """
    from marketos.data.lake import DataLake
    from marketos.data.fetchers.yfinance_fetcher import fetch_ohlcv
    import os

    opps = build_opportunities(store, universe, sector_map)
    if opps.empty:
        return {"positions": [], "n_candidates": 0, "n_with_evidence": 0, "reason": "no_analog_data"}

    weighted = allocate(opps, limits=limits, max_positions=max_positions,
                        min_sample_size=ANALOG_MIN_EFFECTIVE_N)
    if weighted.empty:
        return {"positions": [], "n_candidates": len(opps), "n_with_evidence": 0,
               "reason": "no_positive_expectancy_with_adequate_evidence"}

    lake = DataLake(os.environ.get("DATA_LAKE_ROOT", "data_lake"))
    prices, atrs = {}, {}
    for sym in weighted["symbol"]:
        try:
            df = fetch_ohlcv(sym, lake=lake, period="3mo")
            if df.empty or len(df) < 15:
                continue
            high, low, close = df["high"], df["low"], df["close"]
            tr = pd.concat([high - low, (high - close.shift()).abs(),
                            (low - close.shift()).abs()], axis=1).max(axis=1)
            atrs[sym] = float(tr.rolling(14).mean().iloc[-1])
            prices[sym] = float(close.iloc[-1])
        except Exception:
            continue

    positions = size_positions(weighted, prices, atrs, equity=equity, limits=limits)
    return {
        "positions": positions,
        "n_candidates": len(opps),
        "n_with_evidence": len(opps),
        "n_allocated": len(weighted),
        "n_sized": len(positions),
    }


def snapshot_to_attribution(snapshot: dict, equity: float) -> dict:
    """Reshape a build_quant_snapshot() result into the kwargs upsert_portfolio_snapshot
    expects — weights/gross/cash/effective_n/top_positions/top_themes."""
    from marketos.portfolio.construction import effective_n
    import pandas as pd

    positions: list[SizedPosition] = snapshot.get("positions", [])
    weights = {p.symbol: p.weight for p in positions}
    gross = sum(weights.values())
    by_theme: dict[str, float] = {}
    for p in positions:
        by_theme[p.sector] = by_theme.get(p.sector, 0.0) + p.weight
    # Cash-normalized concentration, NOT raw 1/sum(w^2) on un-normalized weights — at low
    # gross exposure (e.g. 19%) the raw formula explodes past the position count (a 15-name,
    # 19%-gross book genuinely produced "effective_n=305" before this fix, which is
    # impossible — effective_n can never exceed the number of positions held).
    eff_n = effective_n(pd.Series(weights)) if weights else 0.0
    top_positions = sorted(
        [{"symbol": p.symbol, "weight": round(p.weight, 4), "notional": round(p.notional, 2),
          "stop_price": round(p.stop_price, 2), "entry_price": round(p.entry_price, 2),
          "expectancy": round(p.expectancy, 4), "confidence": round(p.confidence, 3),
          "sample_size": round(p.sample_size, 1)}
         for p in positions],
        key=lambda r: -r["weight"],
    )
    top_themes = sorted(
        [{"theme": k, "weight": round(v, 4)} for k, v in by_theme.items()],
        key=lambda r: -r["weight"],
    )
    return {
        "weights": weights,
        "gross_exposure": round(gross, 4),
        "cash_weight": round(max(0.0, 1.0 - gross), 4),
        "effective_n": round(eff_n, 2),
        "top_positions": top_positions,
        "top_themes": top_themes,
    }
