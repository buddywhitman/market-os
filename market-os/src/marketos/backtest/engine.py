"""A small, honest vectorized backtest engine.

Honesty features baked in:
  * Signals act on the NEXT bar's open (no acting on the close you used to decide).
  * Transaction costs: brokerage (bps) + slippage (bps) + optional fixed per-trade.
  * Position changes only — costs are charged on turnover, not on holds.
  * Returns are post-cost; the equity curve is what you could actually have banked.

This is deliberately not a full event-driven simulator (see backtrader/vectorbt for
that). It exists to make the common case correct and fast, and to feed expectancy.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    brokerage_bps: float = 3.0     # round-trip-ish per side, in basis points
    slippage_bps: float = 5.0
    fixed_per_trade: float = 0.0   # currency units, on position change

    @property
    def per_side_frac(self) -> float:
        return (self.brokerage_bps + self.slippage_bps) / 1e4


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series
    period_returns: pd.Series      # post-cost, per period
    trade_returns: pd.Series       # per discrete trade (entry→exit), post-cost
    trade_holding_periods: pd.Series  # bars held per trade, same order/index as trade_returns
    turnover: pd.Series
    total_costs: float


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    cost: CostModel | None = None,
    price_col: str = "open",
) -> BacktestResult:
    """Backtest a weight schedule against prices.

    `prices`         — MultiIndex (ts, symbol) or wide frame; must contain `price_col`
                        and 'close'. Here we accept a wide frame of execution prices.
    `target_weights` — index=ts, columns=symbols; the weights we *decide* at each ts.
                        They are executed on the FOLLOWING bar to avoid look-ahead.
    """
    cost = cost or CostModel()
    w_decided = target_weights.fillna(0.0).sort_index()
    px = prices.sort_index()

    # Execute on the next bar: shift weights forward by one period.
    w_held = w_decided.shift(1).fillna(0.0)

    # Per-period asset returns from execution prices.
    asset_ret = px.pct_change().reindex(w_held.index).fillna(0.0)
    gross = (w_held * asset_ret).sum(axis=1)

    # Turnover & costs charged when the held weights change.
    dw = w_held.diff().abs().fillna(w_held.abs())
    turnover = dw.sum(axis=1)
    cost_frac = turnover * cost.per_side_frac
    net = gross - cost_frac

    equity = (1 + net).cumprod()
    total_costs = float(cost_frac.sum())

    trade_returns, trade_holding_periods = _extract_trade_returns(w_held, asset_ret, cost)

    return BacktestResult(
        equity_curve=equity,
        period_returns=net,
        trade_returns=trade_returns,
        trade_holding_periods=trade_holding_periods,
        turnover=turnover,
        total_costs=total_costs,
    )


def _extract_trade_returns(
    w_held: pd.DataFrame, asset_ret: pd.DataFrame, cost: CostModel
) -> tuple[pd.Series, pd.Series]:
    """Collapse continuous holdings into discrete per-trade returns (per symbol), and the
    number of bars each trade was held — needed for attribution (avg holding period) and
    for correctly annualizing trade-level Sharpe (sqrt(periods_per_year) assumes one
    observation per period; a 20-bar-average trade needs periods_per_year scaled down by
    that holding length, not left at the default daily-return assumption of 252).

    A 'trade' is a maximal run where the symbol's weight is non-zero. We compound the
    in-trade returns and net entry+exit costs. This is what feeds expectancy/win-rate.
    """
    returns_out, holding_out = [], []
    for sym in w_held.columns:
        w = w_held[sym].values
        r = asset_ret[sym].reindex(w_held.index).fillna(0.0).values
        in_trade = False
        comp = 1.0
        held = 0
        for i in range(len(w)):
            active = w[i] != 0
            if active and not in_trade:
                in_trade, comp, held = True, 1.0, 0
            if in_trade:
                comp *= 1 + r[i]
                held += 1
            closing = in_trade and (i == len(w) - 1 or w[min(i + 1, len(w) - 1)] == 0)
            if closing:
                comp_net = comp * (1 - cost.per_side_frac) ** 2  # entry + exit
                returns_out.append(comp_net - 1.0)
                holding_out.append(held)
                in_trade = False
    return (pd.Series(returns_out, dtype=float, name="trade_return"),
            pd.Series(holding_out, dtype=float, name="holding_periods"))
