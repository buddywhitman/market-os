"""Live daily snapshot for the INDIA sleeve — real ₹5,000→₹50,000 growth capital, NSE via
AngelOne (once Phase 5 connects a broker; today this is MANUAL like every other sleeve).

UPDATED (2026-06-23): the India-equivalent analog engine now exists
(`prioritize_subspace_india_job`, reusing `features/latent_supervised.py` +
`features/market_memory.py` — both fully generic, no India-specific code needed there).
Where a candidate has analog evidence (family `analog_india`), this module sizes off it
the SAME way `opportunities.py` does for the US quant sleeve: expectancy=
analog_mean_ret_20d, confidence=analog_win_rate_20d (regime-discounted), sample_size=
analog_n_effective. Where a candidate DOESN'T have analog coverage yet (e.g. too new to
the weekly job, or insufficient history), this falls back to the technical screen score
alone — clearly distinguished in `reason`, never silently treated as equally strong.

Position count is deliberately small (a handful, not 15) — diversification matters
("don't put all eggs in one basket") but ₹5,000 split 15 ways makes each position too
small to matter even with a zero-brokerage discount broker, and dilutes the only real
signal this module has past the point of meaning anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from marketos.risk.sizing import RiskLimits, position_size

# How many of the day's top-screened names to actually size. Small on purpose — see
# module docstring. Diversified, not all-eggs-in-one-basket, but each position stays
# economically meaningful at ₹5,000 total capital.
MAX_POSITIONS = 5

# Same constant + reasoning as opportunities.py's ANALOG_MIN_EFFECTIVE_N: the analog
# engine's evidence is uniqueness-WEIGHTED effective n, typically 5-15 even with 50 raw
# matches — NOT a raw trade count. A naive >=30 threshold would silently exclude every
# real analog match, every time.
ANALOG_MIN_EFFECTIVE_N = 5.0


@dataclass(frozen=True)
class IndiaPosition:
    symbol: str
    sector: str
    weight: float
    shares: float
    notional_inr: float
    stop_price: float
    entry_price: float
    screen_score: float
    mom_63d: float
    reason: str
    confidence: float | None = None  # set only when backed by real analog evidence
    expectancy: float | None = None


def _analog_evidence(store, symbol: str) -> dict | None:
    """Real out-of-sample evidence for `symbol`, if the weekly India latent/analog job
    has covered it. None if no coverage, or coverage exists but is too thin to trust
    (sample_size < ANALOG_MIN_EFFECTIVE_N) — same discipline as opportunities.py."""
    analog = store.get_latest_family(symbol, "analog_india")
    if not analog:
        return None
    expectancy = analog.get("analog_mean_ret_20d")
    sample_size = analog.get("analog_n_effective")
    win_rate = analog.get("analog_win_rate_20d")
    if expectancy is None or sample_size is None or win_rate is None:
        return None
    if float(sample_size) < ANALOG_MIN_EFFECTIVE_N:
        return None
    cross_regime_frac = analog.get("analog_cross_regime_frac", 0.0) or 0.0
    confidence = float(win_rate) * (1.0 - 0.5 * float(cross_regime_frac))
    return {"expectancy": float(expectancy), "confidence": confidence,
           "sample_size": float(sample_size)}


def build_india_snapshot(store, *, limits: RiskLimits, capital_inr: float) -> dict:
    """Read today's stored India screen, rank qualifying candidates by REAL analog
    evidence where it exists (preferred) and technical screen score otherwise, take the
    top `MAX_POSITIONS`, fetch recent OHLCV for ATR, and size each within `capital_inr`."""
    from marketos.features.screening import fetch_angelone_history

    screen = store.get_latest_family("_screen_india", "screen")
    candidates = (screen or {}).get("top_candidates") or []
    if not candidates:
        return {"positions": [], "reason": "no_screen_data"}

    # Require CONFIRMED trend health + positive momentum, not just "best of whatever
    # happened to be in the top_candidates list" — with a small candidate pool (or an
    # unlucky day) the top-N-by-rank slice can include names that are liquid but not
    # actually trending up. Same discipline as the US sleeves' circuit_breaker: only buy
    # into a confirmed trend, never the least-bad option from a weak field.
    qualifying = [c for c in candidates
                 if c.get("trend_healthy") and (c.get("screen_score") or 0) > 0]
    if not qualifying:
        return {"positions": [], "reason": "no_candidate_passed_trend_health_filter"}

    # Enrich with analog evidence where it exists, and require a POSITIVE expectancy
    # there too — real evidence saying "don't buy this" must override a merely-positive
    # technical score, the same way the quant sleeve's allocate() filters on expectancy>0.
    enriched = []
    for c in qualifying:
        evidence = _analog_evidence(store, c["symbol"])
        if evidence is not None and evidence["expectancy"] <= 0:
            continue  # real evidence says no, regardless of technical score
        enriched.append((c, evidence))

    # Rank: analog-backed candidates first (real evidence beats a weaker proxy), each
    # group sorted by its own best signal. This is a ranking preference, not a hard cutoff
    # — a technical-only candidate can still make the cut if there aren't enough
    # analog-backed ones to fill MAX_POSITIONS.
    backed = sorted([(c, e) for c, e in enriched if e is not None],
                    key=lambda ce: -(ce[1]["expectancy"] * ce[1]["confidence"]))
    unbacked = sorted([(c, e) for c, e in enriched if e is None],
                      key=lambda ce: -ce[0]["screen_score"])
    top = (backed + unbacked)[:MAX_POSITIONS]
    if not top:
        return {"positions": [], "reason": "no_candidate_with_positive_expectancy"}

    symbols = [c["symbol"] for c, _ in top]
    history = fetch_angelone_history(symbols, lookback_days=20)  # just enough for ATR(14)

    positions: list[IndiaPosition] = []
    for c, evidence in top:
        sym = c["symbol"]
        df = history.get(sym)
        if df is None or len(df) < 15:
            continue
        close, high, low = df["close"], df["high"], df["low"]
        price = float(close.iloc[-1])
        tr = pd.concat([high - low, (high - close.shift()).abs(),
                        (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        if not atr or atr <= 0 or price <= 0:
            continue

        if evidence is not None:
            # Real out-of-sample evidence — size like the quant sleeve does, scaling by
            # confidence rather than assuming full edge_kelly=1.0.
            sized = position_size(capital_inr, price, atr, limits=limits,
                                  edge_kelly=evidence["confidence"])
            reason = (f"Analog evidence: {evidence['sample_size']:.1f} effective similar "
                     f"historical setups, mean fwd 20d return {evidence['expectancy']:+.1%}, "
                     f"confidence {evidence['confidence']:.0%}.")
        else:
            # No backtested edge to feed edge_kelly with — use 1.0 so sizing is governed
            # purely by limits.kelly_fraction/risk_per_trade, not a fabricated confidence
            # multiplier on top of an already-weak signal.
            sized = position_size(capital_inr, price, atr, limits=limits, edge_kelly=1.0)
            reason = ""
        positions.append(IndiaPosition(
            symbol=sym, sector=c.get("sector", "UNKNOWN"), weight=sized["weight"],
            shares=sized["shares"], notional_inr=sized["notional"],
            stop_price=sized["stop_price"], entry_price=price,
            screen_score=c.get("screen_score", 0.0), mom_63d=c.get("mom_63d", 0.0),
            confidence=evidence["confidence"] if evidence else None,
            expectancy=evidence["expectancy"] if evidence else None,
            reason=reason or (f"Technical screen rank only (momentum {c.get('mom_63d', 0):+.1%}, "
                   f"trend_healthy={c.get('trend_healthy')}) — no analog coverage for "
                   f"THIS symbol yet (too new to the weekly fit, or below the effective-n "
                   f"bar). Weaker evidence than the analog-backed positions above; "
                   f"weight this accordingly."),
        ))
    return {"positions": positions, "n_candidates": len(top)}


def snapshot_to_attribution(snapshot: dict) -> dict:
    from marketos.portfolio.construction import effective_n

    positions: list[IndiaPosition] = snapshot.get("positions", [])
    weights = {p.symbol: p.weight for p in positions}
    gross = sum(weights.values())
    eff_n = effective_n(pd.Series(weights)) if weights else 0.0
    by_sector: dict[str, float] = {}
    for p in positions:
        by_sector[p.sector] = by_sector.get(p.sector, 0.0) + p.weight
    top_positions = sorted(
        [{"symbol": p.symbol, "sector": p.sector, "weight": round(p.weight, 4),
          "notional": round(p.notional_inr, 2), "stop_price": round(p.stop_price, 2),
          "entry_price": round(p.entry_price, 2), "screen_score": round(p.screen_score, 4),
          "mom_63d": round(p.mom_63d, 4), "reason": p.reason,
          "confidence": round(p.confidence, 3) if p.confidence is not None else None,
          "expectancy": round(p.expectancy, 4) if p.expectancy is not None else None}
         for p in positions],
        key=lambda r: -r["weight"],
    )
    top_themes = sorted(
        [{"theme": k, "weight": round(v, 4)} for k, v in by_sector.items()],
        key=lambda r: -r["weight"],
    )
    return {
        "weights": weights, "gross_exposure": round(gross, 4),
        "cash_weight": round(max(0.0, 1.0 - gross), 4), "effective_n": round(eff_n, 2),
        "top_positions": top_positions, "top_themes": top_themes,
    }
