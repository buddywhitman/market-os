"""Live daily snapshot for the AGGRESSIVE sleeve — the circuit-breaker policy, applied to
TODAY's data, not a backtest. Mirrors `portfolio/opportunities.py`'s role for the quant
sleeve: reads/computes what's already validated and shapes it for storage + dashboard.

Policy used: `circuit_breaker` from `models/aggressive_sleeve_backtest.py` — the policy the
user explicitly LOCKED after the real backtest comparison (94% of buy-and-hold's CAGR, half
the drawdown, best Sharpe of the four tested; see project notes). Deliberately NOT using a
fabricated continuous confidence score for sizing: the backtest tested full deployment when
the breaker is IN and zero when OUT, so live sizing uses exactly that — `edge_kelly=1.0`
when in, weight=0 when out. Inventing a smoother confidence signal here would size
positions according to something that was never backtested.

Phase 4b — conviction-gated fast dip re-entry (portfolio/conviction_gate.py) is layered on
TOP of this base policy, not a replacement: when the circuit breaker is OUT (confirmed
sustained downtrend) AND `dip_reentry_signal` fires AND the gate passes (market-wide regime
+ this symbol's own analog history, where available), a PARTIAL position (DIP_REENTRY=0.6,
same fraction the backtest used) is taken instead of staying flat. This composition has NOT
been backtested as a unit — see conviction_gate.py's docstring for why — so its conviction
score and reasons are always recorded, making every gated decision auditable from day one.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from marketos.risk.sizing import RiskLimits, position_size
from marketos.models.aggressive_sleeve_backtest import deployment_weight, dip_reentry_signal, DIP_REENTRY
from marketos.portfolio.conviction_gate import conviction_gate


@dataclass(frozen=True)
class AggressivePosition:
    symbol: str
    in_position: bool
    weight: float
    shares: float
    notional: float
    stop_price: float
    entry_price: float
    reason: str
    conviction: float | None = None  # only set when a Phase 4b gated dip re-entry fired


def _evaluate_dip_reentry(
    sym: str, df: pd.DataFrame, price: float, *, store, spy_regime: int | None,
    equity: float, limits: RiskLimits,
) -> "AggressivePosition":
    """The circuit breaker is OUT for `sym`. Check the mechanical dip-reentry signal; if it
    fires, run it through the Phase 4b conviction gate before sizing anything. Always
    returns a position (in_position=False if the signal didn't fire or didn't pass the
    gate) — the reason/conviction fields make every branch auditable."""
    close = df["close"]
    base_out_reason = "circuit_breaker: price below a falling 200DMA (confirmed sustained downtrend)"
    try:
        dip = dip_reentry_signal(close)
        if not bool(dip.iloc[-1]):
            return AggressivePosition(sym, False, 0.0, 0.0, 0.0, 0.0, price, base_out_reason)
    except Exception:
        return AggressivePosition(sym, False, 0.0, 0.0, 0.0, 0.0, price, base_out_reason)

    analog = None
    if store is not None:
        try:
            analog = store.get_latest_family(sym, "analog") or None
        except Exception:
            analog = None

    gate = conviction_gate(sym, spy_regime=spy_regime, analog=analog)
    if not gate.gate_pass:
        return AggressivePosition(
            sym, False, 0.0, 0.0, 0.0, 0.0, price,
            f"{base_out_reason} | dip-reentry signal fired but gate FAILED "
            f"(conviction={gate.conviction}): {'; '.join(gate.reasons)}",
            conviction=gate.conviction,
        )

    # Gate passed — partial re-entry at DIP_REENTRY (0.6), same fraction the backtest used.
    # Still ATR-stopped like any other live position — a gated re-entry is a riskier bet,
    # not an excuse to skip the stop.
    high, low = df["high"], df["low"]
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    stop_price = price - atr * limits.atr_stop_mult if atr and atr > 0 else 0.0

    notional = equity * limits.max_name_weight * DIP_REENTRY
    shares = notional / price if price > 0 else 0.0
    return AggressivePosition(
        sym, True, limits.max_name_weight * DIP_REENTRY, shares, notional, stop_price, price,
        f"PHASE 4b GATED DIP RE-ENTRY (conviction={gate.conviction}): {'; '.join(gate.reasons)} "
        f"— UNVALIDATED as a unit, watch closely",
        conviction=gate.conviction,
    )


def build_aggressive_snapshot(
    universe: list[str], *, limits: RiskLimits, equity: float, store=None,
) -> dict:
    """Fetch live prices for the aggressive universe, evaluate the LOCKED circuit_breaker
    policy as of today's bar, and size every name currently flagged IN. Names flagged OUT
    get a Phase 4b conviction-gated dip-reentry check before being reported flat — never
    silently omitted either way; visibility into why a name is flat (or partially
    re-entered) matters as much as the sized positions themselves.

    `store` — a MarketosStore, used to read the market-wide regime + per-symbol analog
    evidence for the conviction gate. None (e.g. in tests) disables Phase 4b entirely and
    falls back to the plain circuit_breaker behavior — OUT means OUT, no gate to consult.
    """
    from marketos.data.lake import DataLake
    from marketos.data.fetchers.yfinance_fetcher import fetch_ohlcv
    import os

    lake = DataLake(os.environ.get("DATA_LAKE_ROOT", "data_lake"))
    spy_regime = None
    if store is not None:
        try:
            r = store.get_latest_regime("SPY")
            spy_regime = r["regime"] if r else None
        except Exception:
            spy_regime = None

    positions: list[AggressivePosition] = []
    for sym in universe:
        try:
            # 2y so the 200DMA + its 20d slope are warm; circuit_breaker needs both.
            df = fetch_ohlcv(sym, lake=lake, period="2y")
            if df.empty or len(df) < 200:
                positions.append(AggressivePosition(
                    sym, False, 0.0, 0.0, 0.0, 0.0, 0.0, "insufficient_history"))
                continue
            close = df["close"]
            w = deployment_weight(close, policy="circuit_breaker")
            in_position = bool(w.iloc[-1] > 0)
            price = float(close.iloc[-1])

            if not in_position:
                positions.append(_evaluate_dip_reentry(
                    sym, df, price, store=store, spy_regime=spy_regime,
                    equity=equity, limits=limits))
                continue

            high, low = df["high"], df["low"]
            tr = pd.concat([high - low, (high - close.shift()).abs(),
                            (low - close.shift()).abs()], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            if not atr or atr <= 0:
                positions.append(AggressivePosition(
                    sym, True, 0.0, 0.0, 0.0, 0.0, price, "circuit_breaker IN but ATR unsizeable"))
                continue

            sized = position_size(equity, price, atr, limits=limits, edge_kelly=1.0)
            positions.append(AggressivePosition(
                sym, True, sized["weight"], sized["shares"], sized["notional"],
                sized["stop_price"], price, "circuit_breaker: trend intact, fully deployed"))
        except Exception as exc:
            positions.append(AggressivePosition(sym, False, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                 f"fetch_failed: {exc}"))

    return {"positions": positions, "n_universe": len(universe),
           "n_in": sum(1 for p in positions if p.in_position)}


def snapshot_to_attribution(snapshot: dict) -> dict:
    """Same shape contract as opportunities.snapshot_to_attribution, for
    upsert_portfolio_snapshot under strategy_name='aggressive_sleeve'."""
    from marketos.portfolio.construction import effective_n

    positions: list[AggressivePosition] = snapshot.get("positions", [])
    weights = {p.symbol: p.weight for p in positions if p.in_position}
    gross = sum(weights.values())
    eff_n = effective_n(pd.Series(weights)) if weights else 0.0
    top_positions = sorted(
        [{"symbol": p.symbol, "in_position": p.in_position, "weight": round(p.weight, 4),
          "notional": round(p.notional, 2), "stop_price": round(p.stop_price, 2),
          "entry_price": round(p.entry_price, 2), "reason": p.reason,
          "conviction": p.conviction}
         for p in positions],
        key=lambda r: (-r["in_position"], -r["weight"]),
    )
    return {
        "weights": weights,
        "gross_exposure": round(gross, 4),
        "cash_weight": round(max(0.0, 1.0 - gross), 4),
        "effective_n": round(eff_n, 2),
        "top_positions": top_positions,
        "top_themes": [],
    }
