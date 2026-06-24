"""World-state market memory: SQL cohort queries + Gower-distance nearest-neighbor
search over the daily portfolio_attribution record. No embeddings, no FAISS, no LLMs --
the state vector is small (regime labels + VIX percentile + cash/effective_n/exposure/
turnover) and the history is a few thousand rows, exactly the regime this kind of search
is right-sized for. ML-based analog search (features/market_memory.py) operates on a
different, per-stock latent space; this module operates one level up, on world state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def gower_distance_matrix(
    query: dict, candidates: pd.DataFrame, *, categorical_cols: list[str],
    continuous_cols: list[str], continuous_ranges: dict[str, float] | None = None,
) -> pd.Series:
    """Gower distance from one query world-state to every candidate row. Categorical
    fields contribute 0 (match) or 1 (mismatch); continuous fields contribute
    |a-b|/range, range estimated from the candidate pool unless supplied. Equal-weighted
    average across all fields -- "dumb," no learned metric."""
    n_fields = len(categorical_cols) + len(continuous_cols)
    total = pd.Series(0.0, index=candidates.index)
    for col in categorical_cols:
        total += (candidates[col] != query.get(col)).astype(float)
    ranges = continuous_ranges or {}
    for col in continuous_cols:
        rng = ranges.get(col) or (candidates[col].max() - candidates[col].min())
        rng = rng if rng and rng > 0 else 1.0
        total += (candidates[col] - query.get(col, np.nan)).abs() / rng
    return total / n_fields


def find_nearest_worlds(query: dict, candidates: pd.DataFrame, *, categorical_cols: list[str],
                         continuous_cols: list[str], k: int = 50,
                         exclude_idx: pd.Index | None = None) -> pd.DataFrame:
    """Top-k nearest historical world-states by Gower distance, with their distance."""
    pool = candidates.drop(index=exclude_idx, errors="ignore") if exclude_idx is not None else candidates
    dist = gower_distance_matrix(query, pool, categorical_cols=categorical_cols,
                                 continuous_cols=continuous_cols)
    nearest = dist.nsmallest(k)
    out = pool.loc[nearest.index].copy()
    out["gower_distance"] = nearest
    return out.sort_values("gower_distance")


def analog_outcome_distribution(analog_dates: pd.DatetimeIndex, forward_returns: dict[int, pd.Series]) -> dict:
    """For a set of analog dates, summarize the forward-return distribution at each
    horizon (in trading days) already computed in `forward_returns`."""
    out = {}
    for horizon, fwd in forward_returns.items():
        vals = fwd.reindex(analog_dates).dropna()
        if len(vals) == 0:
            out[horizon] = {"n": 0}
            continue
        out[horizon] = {
            "n": int(len(vals)), "mean": float(vals.mean()), "median": float(vals.median()),
            "win_rate": float((vals > 0).mean()), "p5": float(vals.quantile(0.05)),
            "p95": float(vals.quantile(0.95)),
        }
    return out


def cohort_stats(rows: pd.DataFrame, *, return_col: str = "daily_return",
                  turnover_col: str = "turnover", periods_per_year: int = 252) -> dict:
    """n / mean return / Sharpe / max drawdown / mean turnover for an arbitrary cohort
    of days (the output of a world-state filter query)."""
    rets = rows[return_col].dropna()
    n = len(rets)
    if n == 0:
        return {"n": 0}
    mean_r, std_r = rets.mean(), rets.std()
    sharpe = float(mean_r / std_r * np.sqrt(periods_per_year)) if std_r > 0 else 0.0
    equity = (1 + rets).cumprod()
    maxdd = float(((equity - equity.cummax()) / equity.cummax()).min())
    mean_turnover = float(rows[turnover_col].mean()) if turnover_col in rows.columns else None
    return {"n": n, "mean_return": float(mean_r), "sharpe": sharpe, "max_drawdown": maxdd,
            "mean_turnover": mean_turnover}


def empirical_transition_matrix(bucketed_series: pd.Series, bucket_order: list) -> pd.DataFrame:
    """Empirical bucket-to-bucket transition frequency from a discretized series
    (regime labels, or a continuous series already cut into buckets via pd.cut). Plain
    counting, no model."""
    n = len(bucket_order)
    idx = {b: i for i, b in enumerate(bucket_order)}
    counts = np.zeros((n, n))
    vals = bucketed_series.values
    for t in range(len(vals) - 1):
        a, b = vals[t], vals[t + 1]
        if a in idx and b in idx:
            counts[idx[a], idx[b]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    return pd.DataFrame(probs, index=bucket_order, columns=bucket_order)
