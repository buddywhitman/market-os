"""Per-trade attribution: the permanent record of what was known at entry and what
happened at exit. The point isn't aggregate Sharpe — it's eventually being able to ask
"which kinds of worlds produce the edge," which requires every trade to carry the regime,
the world state, and the reasoning it was opened under, not just its P&L.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class TradeAttribution:
    trade_id: str
    instrument: str
    strategy_name: str
    entry_ts: datetime
    entry_price: float
    position_size: float
    regime_label: int | None = None
    regime_name: str | None = None
    regime_probability: dict | None = None
    momentum_state: str | None = None
    momentum_200d: float | None = None
    momentum_50d: float | None = None
    vix_percentile: float | None = None
    spy_regime: str | None = None
    qqq_regime: str | None = None
    entry_reason: str | None = None
    volatility_at_entry: float | None = None
    portfolio_state: dict | None = None
    world_state_vector: dict | None = None
    world_state_hash: str | None = None
    nearest_analog_ids: list | None = None
    exit_ts: datetime | None = None
    exit_price: float | None = None
    holding_days: int | None = None
    realized_return: float | None = None
    stop_triggered: bool = False
    signal_decay_triggered: bool = False
    max_drawdown_during_trade: float | None = None

    @classmethod
    def open(cls, instrument, strategy_name, entry_ts, entry_price, position_size,
             **kwargs) -> "TradeAttribution":
        return cls(trade_id=str(uuid.uuid4()), instrument=instrument, strategy_name=strategy_name,
                   entry_ts=entry_ts, entry_price=entry_price, position_size=position_size, **kwargs)

    def close(self, exit_ts, exit_price, *, stop_triggered=False, signal_decay_triggered=False,
              max_drawdown_during_trade=None) -> None:
        self.exit_ts = exit_ts
        self.exit_price = exit_price
        self.holding_days = (exit_ts - self.entry_ts).days
        self.realized_return = (exit_price / self.entry_price - 1.0) if self.entry_price else None
        self.stop_triggered = stop_triggered
        self.signal_decay_triggered = signal_decay_triggered
        self.max_drawdown_during_trade = max_drawdown_during_trade

    def persist(self, store) -> None:
        store.open_trade(
            self.trade_id, self.instrument, self.strategy_name, self.entry_ts,
            regime_label=self.regime_label, regime_name=self.regime_name,
            regime_probability=self.regime_probability, momentum_state=self.momentum_state,
            momentum_200d=self.momentum_200d, momentum_50d=self.momentum_50d,
            vix_percentile=self.vix_percentile, spy_regime=self.spy_regime, qqq_regime=self.qqq_regime,
            entry_reason=self.entry_reason, entry_price=self.entry_price,
            volatility_at_entry=self.volatility_at_entry, position_size=self.position_size,
            portfolio_state=self.portfolio_state, world_state_vector=self.world_state_vector,
            world_state_hash=self.world_state_hash, nearest_analog_ids=self.nearest_analog_ids,
        )
        if self.exit_ts is not None:
            store.close_trade(
                self.trade_id, self.exit_ts, self.exit_price, self.realized_return,
                holding_days=self.holding_days, stop_triggered=self.stop_triggered,
                signal_decay_triggered=self.signal_decay_triggered,
                max_drawdown_during_trade=self.max_drawdown_during_trade,
            )


def world_state_hash(state: dict) -> str:
    """Deterministic short hash of a CATEGORICAL world-state snapshot, so trades entered
    under near-identical macro conditions group together exactly ("every trade entered
    when SPY=calm, VIX=low, SOXX=bullish, BTC=bullish") without comparing raw floats."""
    import hashlib
    import json
    canonical = json.dumps(state, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build_world_state(
    date, *, regime_by_instrument: dict[str, pd.Series] | None = None,
    momentum_by_instrument: dict[str, pd.Series] | None = None,
    vix_percentile: pd.Series | None = None, regime_bucket_names: dict[int, str] | None = None,
) -> dict:
    """Categorical snapshot of macro/theme conditions at one date: each instrument's
    regime bucketed to a name, each instrument's momentum bucketed to bullish/bearish,
    VIX percentile bucketed to low/mid/high. Categorical, not raw floats, so it's
    groupable and hashable -- the whole point of `world_state_hash`."""
    names = regime_bucket_names or {0: "calm", 1: "neutral", 2: "stress"}
    state: dict = {}
    for instr, regime_series in (regime_by_instrument or {}).items():
        if date in regime_series.index:
            r = int(regime_series.loc[date])
            state[f"{instr}_regime"] = names.get(r, f"state_{r}")
    for instr, mom_series in (momentum_by_instrument or {}).items():
        if date in mom_series.index:
            m = mom_series.loc[date]
            if pd.notna(m):
                state[f"{instr}_mom"] = "bullish" if m > 0 else "bearish"
    if vix_percentile is not None and date in vix_percentile.index:
        v = vix_percentile.loc[date]
        if pd.notna(v):
            state["VIX_bucket"] = "low" if v < 33 else ("high" if v > 66 else "mid")
    return state


def extract_trades_from_weights(
    weights: pd.Series, prices: pd.Series, *, instrument: str, strategy_name: str,
    entry_reason: str = "", regime_series: pd.Series | None = None,
    regime_names: dict[int, str] | None = None,
    momentum_200d: pd.Series | None = None, momentum_50d: pd.Series | None = None,
    vix_percentile: pd.Series | None = None,
    spy_regime: pd.Series | None = None, qqq_regime: pd.Series | None = None,
    world_state: pd.DataFrame | None = None,
    cross_asset_regimes: dict[str, pd.Series] | None = None,
    cross_asset_momentum: dict[str, pd.Series] | None = None,
) -> list[TradeAttribution]:
    """Walk a single-instrument 0/1 (in/out) weight series and emit one TradeAttribution
    per maximal non-zero run, populated with whatever regime/world-state context is
    available at entry. Built for the regime-conditional rules tested this round, but
    works for any binary-exposure weight series."""
    w = weights.values
    idx = weights.index
    px = prices.reindex(idx)
    trades: list[TradeAttribution] = []
    in_trade, entry_i = False, None
    for i in range(len(w)):
        active = w[i] != 0
        if active and not in_trade:
            in_trade, entry_i = True, i
        closing = in_trade and (i == len(w) - 1 or w[min(i + 1, len(w) - 1)] == 0)
        if closing:
            entry_ts, exit_ts = idx[entry_i], idx[i]
            entry_price, exit_price = float(px.iloc[entry_i]), float(px.iloc[i])
            regime_label = (int(regime_series.reindex(idx).iloc[entry_i])
                            if regime_series is not None else None)
            regime_name = (regime_names or {}).get(regime_label) if regime_label is not None else None

            def _at(series):
                if series is None:
                    return None
                v = series.reindex(idx).iloc[entry_i]
                return float(v) if pd.notna(v) else None

            world_snap = (world_state.reindex(idx).iloc[entry_i].dropna().to_dict()
                         if world_state is not None else None)
            state = build_world_state(
                entry_ts,
                regime_by_instrument={**({instrument: regime_series} if regime_series is not None else {}),
                                      **(cross_asset_regimes or {})},
                momentum_by_instrument=cross_asset_momentum or {},
                vix_percentile=vix_percentile,
            )
            trade = TradeAttribution.open(
                instrument=instrument, strategy_name=strategy_name, entry_ts=entry_ts,
                entry_price=entry_price, position_size=float(w[entry_i]),
                regime_label=regime_label, regime_name=regime_name,
                entry_reason=entry_reason,
                momentum_200d=_at(momentum_200d), momentum_50d=_at(momentum_50d),
                vix_percentile=_at(vix_percentile),
                spy_regime=(str(int(spy_regime.reindex(idx).iloc[entry_i]))
                           if spy_regime is not None and pd.notna(spy_regime.reindex(idx).iloc[entry_i]) else None),
                qqq_regime=(str(int(qqq_regime.reindex(idx).iloc[entry_i]))
                           if qqq_regime is not None and pd.notna(qqq_regime.reindex(idx).iloc[entry_i]) else None),
                world_state_vector=world_snap, world_state_hash=world_state_hash(state) if state else None,
            )
            path = px.iloc[entry_i:i + 1]
            running_max = path.cummax()
            dd = float(((path - running_max) / running_max).min()) if len(path) > 1 else 0.0
            trade.close(exit_ts, exit_price, max_drawdown_during_trade=dd)
            trades.append(trade)
            in_trade = False
    return trades
