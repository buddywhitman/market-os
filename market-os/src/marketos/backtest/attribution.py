"""Performance attribution — beta, alpha, capture ratios, exposure, turnover.

Sharpe alone hides the single most important question for judging a strategy: how much of
that return is just market beta? A Sharpe of 1.5 at beta=1.3 to the benchmark is
unimpressive — you could get most of it from a levered index fund with no model at all. A
Sharpe of 1.5 at beta=0.5 is genuinely differentiated. This module decomposes a backtest's
period returns against a benchmark so that comparison is explicit rather than implied.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AttributionReport:
    cagr: float
    avg_exposure: float              # mean gross weight over time (0-1+)
    avg_turnover_per_year: float
    avg_holding_days: float
    beta: float                      # vs the supplied benchmark
    alpha_annualized: float          # CAPM alpha: mean_strategy - beta*mean_benchmark, annualized
    correlation: float
    upside_capture: float            # strategy return / benchmark return, on benchmark-up days
    downside_capture: float          # same, on benchmark-down days — lower is better here

    def as_dict(self) -> dict:
        return asdict(self)


def _cagr(equity: pd.Series, periods_per_year: int) -> float:
    n = len(equity)
    if n < 2 or equity.iloc[0] <= 0:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0]
    years = n / periods_per_year
    return float(total_return ** (1 / years) - 1) if years > 0 and total_return > 0 else 0.0


def _capture_ratios(strat_ret: pd.Series, bench_ret: pd.Series) -> tuple[float, float]:
    """Average strategy return on benchmark-up (resp. -down) days, divided by the average
    benchmark return on those same days. >1 amplifies that side, <1 dampens it. A
    defensively-good strategy wants high upside capture and low downside capture —
    asymmetric capture is the entire point of a real edge, vs. just being levered beta."""
    joined = pd.DataFrame({"s": strat_ret, "b": bench_ret}).dropna()
    up = joined[joined["b"] > 0]
    down = joined[joined["b"] < 0]
    up_capture = float(up["s"].mean() / up["b"].mean()) if len(up) and up["b"].mean() != 0 else np.nan
    down_capture = (float(down["s"].mean() / down["b"].mean())
                    if len(down) and down["b"].mean() != 0 else np.nan)
    return up_capture, down_capture


def attribute_performance(
    period_returns: pd.Series,
    benchmark_returns: pd.Series,
    weights: pd.DataFrame,
    turnover: pd.Series,
    trade_holding_periods: pd.Series | None = None,
    *,
    periods_per_year: int = 252,
) -> AttributionReport:
    """Decompose a backtest's period returns against a benchmark (e.g. SPY or QQQ daily
    returns, aligned to the same calendar as `period_returns`).

    `weights`/`turnover`/`trade_holding_periods` come straight off BacktestResult — pass
    `bt.weights` isn't a field on BacktestResult itself, so callers should pass the
    target_weights frame they constructed; turnover and trade_holding_periods ARE on the
    result object.
    """
    joined = pd.DataFrame({"s": period_returns, "b": benchmark_returns}).dropna()
    if joined.empty:
        return AttributionReport(0, 0, 0, np.nan, 0, 0, 0, np.nan, np.nan)

    equity = (1 + joined["s"]).cumprod()
    cagr = _cagr(equity, periods_per_year)

    var_b = joined["b"].var()
    beta = float(joined["s"].cov(joined["b"]) / var_b) if var_b > 0 else 0.0
    mean_s = joined["s"].mean() * periods_per_year
    mean_b = joined["b"].mean() * periods_per_year
    alpha = float(mean_s - beta * mean_b)

    correlation = float(joined["s"].corr(joined["b"])) if len(joined) > 2 else 0.0
    up_capture, down_capture = _capture_ratios(joined["s"], joined["b"])

    avg_exposure = float(weights.abs().sum(axis=1).mean()) if weights is not None and not weights.empty else 0.0
    avg_turnover_per_year = float(turnover.mean() * periods_per_year) if turnover is not None and len(turnover) else 0.0
    avg_holding = (float(trade_holding_periods.mean())
                   if trade_holding_periods is not None and len(trade_holding_periods) else np.nan)

    return AttributionReport(
        cagr=cagr, avg_exposure=avg_exposure, avg_turnover_per_year=avg_turnover_per_year,
        avg_holding_days=avg_holding, beta=beta, alpha_annualized=alpha,
        correlation=correlation, upside_capture=up_capture, downside_capture=down_capture,
    )
