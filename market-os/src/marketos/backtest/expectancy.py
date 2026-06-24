"""Expectancy and trade-level statistics.

Expectancy is the system's north star: the average dollar (or R-multiple) outcome per
trade. A strategy with a high win rate and negative expectancy is worthless; a strategy
with a 35% win rate and positive expectancy compounds. We measure both, plus the
distribution around them, and we never trust a number without its sample size.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class ExpectancyReport:
    sample_size: int
    win_rate: float
    avg_win: float
    avg_loss: float          # stored as a negative number
    payoff_ratio: float      # avg_win / |avg_loss|
    profit_factor: float     # gross profit / gross loss
    expectancy: float        # mean per-trade return (the headline)
    expectancy_r: float      # mean per-trade return in R (risk) units
    sharpe: float
    sortino: float
    max_drawdown: float
    kelly_fraction: float    # full-Kelly; size at a fraction of this in production

    def as_dict(self) -> dict:
        return asdict(self)


def _drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak == 0, 1, peak)
    return float(dd.min())


def compute_expectancy(
    returns: np.ndarray,
    *,
    risk_per_trade: np.ndarray | None = None,
    periods_per_year: int = 252,
) -> ExpectancyReport:
    """Compute the full expectancy report from per-trade (or per-period) returns.

    `returns`        — array of realized fractional returns per trade.
    `risk_per_trade` — optional array of the risk taken on each trade (for R-multiples).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = r.size
    if n == 0:
        return ExpectancyReport(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = wins.size / n
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0  # negative
    payoff = avg_win / abs(avg_loss) if avg_loss != 0 else np.inf
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    expectancy = float(r.mean())

    if risk_per_trade is not None:
        risk = np.asarray(risk_per_trade, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_mult = np.where(risk > 0, r / risk, np.nan)
        expectancy_r = float(np.nanmean(r_mult))
    else:
        expectancy_r = 0.0

    std = r.std(ddof=1) if n > 1 else 0.0
    downside = losses.std(ddof=1) if losses.size > 1 else 0.0
    sharpe = (expectancy / std * np.sqrt(periods_per_year)) if std > 0 else 0.0
    sortino = (expectancy / downside * np.sqrt(periods_per_year)) if downside > 0 else 0.0

    equity = np.cumprod(1 + r)
    max_dd = _drawdown(equity)

    # Kelly for a binary-ish bet: f* = W/|loss| - (1-W)/win  (in odds form: W - (1-W)/payoff)
    kelly = (win_rate - (1 - win_rate) / payoff) if payoff not in (0, np.inf) else 0.0

    return ExpectancyReport(
        sample_size=n,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=float(payoff),
        profit_factor=float(profit_factor),
        expectancy=expectancy,
        expectancy_r=expectancy_r,
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=max_dd,
        kelly_fraction=float(max(0.0, kelly)),
    )


def monte_carlo_drawdown(
    returns: np.ndarray, n_paths: int = 2000, seed: int = 7
) -> dict[str, float]:
    """Bootstrap the distribution of terminal equity and max drawdown.

    A single backtest equity curve is one sample from a distribution. Resampling the
    trade sequence tells us how lucky (or unlucky) the realized path was, and what
    drawdown we should actually budget for.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return {"p05_terminal": 0, "p50_terminal": 0, "p95_terminal": 0, "p95_maxdd": 0}
    terminals, maxdds = [], []
    for _ in range(n_paths):
        sample = rng.choice(r, size=r.size, replace=True)
        equity = np.cumprod(1 + sample)
        terminals.append(equity[-1])
        maxdds.append(_drawdown(equity))
    return {
        "p05_terminal": float(np.percentile(terminals, 5)),
        "p50_terminal": float(np.percentile(terminals, 50)),
        "p95_terminal": float(np.percentile(terminals, 95)),
        "p95_maxdd": float(np.percentile(maxdds, 5)),  # 5th pct = worst-ish drawdown
    }
