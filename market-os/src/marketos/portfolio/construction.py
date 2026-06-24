"""Deterministic, unoptimized portfolio construction: regime confidence + momentum
strength + correlation-based uniqueness, inverse-vol scaled. No optimizer -- the point of
this stage is testing whether dumb, interpretable weights survive a portfolio-level null,
not squeezing out basis points via Optuna/HRP/risk-parity. Theme caps are deliberately
absent; correlation-based uniqueness penalizes overlap directly instead of via a crude
sector bucket.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def realized_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)


def regime_confidence_from_proba(regime_prob_cols: pd.DataFrame) -> pd.Series:
    """Max posterior probability across regime states -- how sure the model is, not
    which state it picked. Bounded [1/n_states, 1] by construction."""
    return regime_prob_cols.max(axis=1)


def regime_confidence_from_vol_percentile(pct_rank: pd.Series) -> pd.Series:
    """Fallback confidence for the no-HMM vol-percentile classifier: distance from the
    neutral midpoint (50), scaled to [0,1]. pct_rank near 10 or 90 is confidently
    calm/stress; pct_rank near 50 is genuinely ambiguous."""
    return (pct_rank - 50).abs() / 50.0


def momentum_strength_rank(momentum_by_instrument: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank of each instrument's momentum per date, among
    instruments with positive momentum (the only ones eligible to trade at all). Bounded
    [0,1] by construction -- avoids comparing raw momentum across instruments with very
    different volatility/return scales."""
    pos = momentum_by_instrument.where(momentum_by_instrument > 0)
    return pos.rank(axis=1, pct=True)


def uniqueness_scores(returns_by_instrument: pd.DataFrame,
                       windows: tuple[int, ...] = (21, 63, 252)) -> pd.DataFrame:
    """1/avg_correlation per instrument, blended across short/medium/long rolling
    windows, then cross-sectionally rank-normalized to [0,1]. Direct correlation penalty
    instead of a theme cap -- SOXX/SMH/QQQ/IGV/HACK overlapping heavily shows up here as
    low uniqueness for all five, without needing a hand-picked sector bucket."""
    instruments = list(returns_by_instrument.columns)
    avg_corr_per_window = []
    for w in windows:
        roll_corr = returns_by_instrument.rolling(w).corr()
        avg_corr = pd.DataFrame(index=returns_by_instrument.index, columns=instruments, dtype=float)
        for instr in instruments:
            row = roll_corr.xs(instr, level=1)
            others = row.drop(columns=[instr], errors="ignore")
            avg_corr[instr] = others.mean(axis=1)
        avg_corr_per_window.append(avg_corr)
    blended = sum(avg_corr_per_window) / len(avg_corr_per_window)
    uniqueness_raw = 1.0 / blended.clip(lower=0.05)  # floor avoids blowup as corr->0
    return uniqueness_raw.rank(axis=1, pct=True)


def effective_n(weights_row: pd.Series) -> float:
    """Effective number of bets among what's actually INVESTED, cash-normalized so the
    result is always bounded [1, len(weights_row)] regardless of gross exposure. Track
    gross_exposure/cash_weight separately for "how much is invested" -- this answers
    "how concentrated is it," and the two must not be conflated: with low gross exposure,
    raw 1/sum(w^2) on un-normalized weights can exceed the instrument count, which is
    meaningless as a "number of bets."""
    gross = float(weights_row.sum())
    if gross <= 0:
        return 0.0
    normalized = weights_row / gross
    sq = float((normalized ** 2).sum())
    return float(1.0 / sq) if sq > 0 else 0.0


def construct_weights(
    eligible: pd.DataFrame, regime_confidence: pd.DataFrame, momentum_rank: pd.DataFrame,
    uniqueness_rank: pd.DataFrame, realized_vol_df: pd.DataFrame,
    *, max_single: float = 0.15, reference_vol: float = 0.20,
) -> pd.DataFrame:
    """w_i ~ (0.35*regime_confidence + 0.35*momentum_strength + 0.30*uniqueness) / vol,
    capped at max_single per name, gross capped at 100% (scaled down only if exceeded --
    never scaled up). Cash is whatever's left over; it is never forced to zero."""
    score = (0.35 * regime_confidence.reindex_like(eligible).fillna(0)
             + 0.35 * momentum_rank.reindex_like(eligible).fillna(0)
             + 0.30 * uniqueness_rank.reindex_like(eligible).fillna(0))
    score = score.where(eligible, 0.0)
    raw_weight = max_single * score * (reference_vol / realized_vol_df.reindex_like(eligible).clip(lower=1e-4))
    raw_weight = raw_weight.clip(upper=max_single).fillna(0.0)
    gross = raw_weight.sum(axis=1)
    scale = (1.0 / gross.clip(lower=1.0))  # only scale DOWN when gross > 1.0, never up
    return raw_weight.mul(scale, axis=0)
