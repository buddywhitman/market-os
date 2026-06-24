"""Label construction + leakage-proof validation — the missing layer underneath every
statistic this system has produced so far (IC in subspace.py, win-rates, any future MI/PBO).

Why this module exists, stated plainly: every cross-sectional IC, hit-rate, or "117 historical
analogs, 71% win rate" claim is *fiction* unless two problems are handled —

  1. Overlapping labels.  A 20-day forward return sampled daily shares 19/20 of its window with
     the next day's label. Consecutive observations are not independent. Treating them as IID
     when computing IC_std / t-stats / win-rates inflates apparent significance by roughly the
     horizon length — an "8 independent events" sample can masquerade as "117 observations."
     Fix: per-observation *uniqueness weights* (López de Prado, AFML ch.4) down-weight crowded
     labels so aggregate statistics reflect effective, not nominal, sample size.

  2. Walk-forward leakage.  A naive train/test split lets training rows whose label horizon
     extends past the split boundary "see" test-period price action. Fix: purge any training
     observation whose label window overlaps the test fold, and embargo a short window after
     the test fold (price/feature autocorrelation bleeds across the boundary too).

This module provides labels (triple-barrier — the standard meta-labeling target), the
uniqueness weights, and a deflated-Sharpe / probability-of-backtest-overfitting utility for
when many features/strategies are screened at once (this codebase screens 1,000+ features ×
several targets — exactly the regime multiple-testing inflation targets).

The purge+embargo SPLITTER itself lives in `backtest.walkforward` (it predates this module and
is also the one `models.alpha_model` trains against) — import `n_split_walk_forward` /
`walk_forward_splits` from there, not from here. A second, slightly-different purge convention
briefly existed in this file; it has been removed in favor of one canonical implementation.

References: Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"; López de Prado,
"Advances in Financial Machine Learning" (2018) ch. 3-4, 7, 11.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ── 1. Triple-barrier labeling ────────────────────────────────────────────────

def triple_barrier_labels(
    close: pd.Series,
    *,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    max_horizon: int = 10,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Label each bar by which barrier it hits first: profit-take, stop-loss, or time.

    Barriers are set at ±{pt_mult, sl_mult} × the bar's trailing realized volatility — so the
    label adapts to the prevailing vol regime instead of using a fixed % move (a fixed-%
    barrier is too tight in calm regimes and too loose in turmoil, which silently biases which
    *kind* of move gets labeled +1/-1 across history).

    Returns one row per bar with:
      label      : +1 (hit profit-take first), -1 (hit stop-loss first), 0 (timed out)
      ret        : realized return at the touch (or at max_horizon if timed out)
      t1         : the integer bar offset at which the label resolved (≤ max_horizon)
      touch_idx  : positional index into `close` where it resolved (for uniqueness counting)
    All causal: a label at bar i only uses information from bars i..i+t1, never before i.
    """
    n = len(close)
    px = close.values
    vol = close.pct_change().rolling(vol_window).std().values

    label = np.zeros(n)
    ret = np.full(n, np.nan)
    t1 = np.full(n, np.nan)
    touch_idx = np.full(n, np.nan)

    for i in range(n - 1):
        v = vol[i]
        if not np.isfinite(v) or v <= 0:
            continue
        upper = px[i] * (1 + pt_mult * v)
        lower = px[i] * (1 - sl_mult * v)
        end = min(i + max_horizon, n - 1)
        resolved = False
        for j in range(i + 1, end + 1):
            if px[j] >= upper:
                label[i], ret[i], t1[i], touch_idx[i] = 1, px[j] / px[i] - 1, j - i, j
                resolved = True
                break
            if px[j] <= lower:
                label[i], ret[i], t1[i], touch_idx[i] = -1, px[j] / px[i] - 1, j - i, j
                resolved = True
                break
        if not resolved and end > i:
            label[i], ret[i], t1[i], touch_idx[i] = 0, px[end] / px[i] - 1, end - i, end

    out = pd.DataFrame({"label": label, "ret": ret, "t1": t1, "touch_idx": touch_idx},
                       index=close.index)
    out.iloc[-1, :] = np.nan  # last bar has no future to resolve against
    return out


# ── 2. Sample uniqueness (de-overlap weighting) ───────────────────────────────

def sample_uniqueness(labels: pd.DataFrame) -> pd.Series:
    """Average uniqueness weight per labeled observation, from de Prado AFML ch.4.

    For each bar in [i, touch_idx_i], count how many *other* labels' windows also cover that
    bar (concurrency c_t). The observation's weight is the average of 1/c_t over its own window
    — a label that overlaps heavily with neighbors gets a small weight; an isolated label gets
    weight ≈1. Sum of weights ≈ the *effective* (independent) sample size, which is what every
    downstream IC/t-stat/MI computation should use instead of len(labels).
    """
    n = len(labels)
    touch = labels["touch_idx"].values
    concurrency = np.zeros(n)
    for i in range(n):
        t1 = touch[i]
        if not np.isfinite(t1):
            continue
        concurrency[int(i):int(t1) + 1] += 1.0

    weight = np.full(n, np.nan)
    for i in range(n):
        t1 = touch[i]
        if not np.isfinite(t1):
            continue
        span = concurrency[int(i):int(t1) + 1]
        span = span[span > 0]
        weight[i] = float((1.0 / span).mean()) if len(span) else np.nan

    return pd.Series(weight, index=labels.index, name="uniqueness")


def effective_sample_size(weights: pd.Series) -> float:
    """Sum of uniqueness weights = the effective (de-overlapped) N for honest stat tests."""
    return float(weights.dropna().sum())


# ── 3. Deflated Sharpe / PBO — guard against multiple-testing inflation ───────

def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_obs: float,
    n_trials: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_std_across_trials: float | None = None,
) -> float:
    """Probability the observed Sharpe is genuine, after correcting for (a) non-normal
    returns and (b) having tried `n_trials` features/strategies and reporting the best.

    `n_obs` should be the *effective* sample size (use effective_sample_size, not len()).
    `sharpe_std_across_trials`: the cross-trial std of Sharpe ratios, if you've actually run
    many trials (use the real dispersion). If omitted, falls back to the standard asymptotic
    approximation for the expected max of `n_trials` iid Gaussians — a reasonable default but
    weaker than passing the empirical dispersion when you have it.

    Returns DSR ∈ [0, 1]: probability of a true positive Sharpe ratio above zero, deflated for
    multiple testing. DSR < 0.95 on a "best of many" feature should not be trusted as alpha.
    """
    if n_trials <= 1:
        sr0 = 0.0
    else:
        if sharpe_std_across_trials is None:
            # Expected max of n iid N(0,1): asymptotic approx (Bailey & López de Prado 2014).
            euler_gamma = 0.5772156649
            ez = (1 - euler_gamma) * stats.norm.ppf(1 - 1.0 / n_trials) + \
                 euler_gamma * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
            sharpe_std_across_trials = 1.0  # unit std for the asymptotic max formula
            sr0 = sharpe_std_across_trials * ez
        else:
            sr0 = sharpe_std_across_trials * stats.norm.ppf(1 - 1.0 / n_trials)

    num = (observed_sharpe - sr0) * np.sqrt(n_obs - 1)
    den = np.sqrt(max(1e-12, 1 - skew * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe ** 2))
    z = num / den
    return float(stats.norm.cdf(z))


def probability_of_backtest_overfitting(in_sample_sharpes: np.ndarray,
                                         out_sample_sharpes: np.ndarray) -> float:
    """PBO (Bailey et al. 2015): fraction of trials where the best in-sample config ranks
    BELOW median out-of-sample — i.e. probability the winning feature/strategy was a fluke of
    in-sample search rather than a genuinely robust performer.

    Pass parallel arrays from the SAME trials (same feature/config, in-sample vs held-out
    fold). A PBO near 0.5 means the in-sample winner is no better than a coin flip
    out-of-sample; PBO should be well below 0.5 (ideally <0.2) before trusting a "best" feature.
    """
    in_s = np.asarray(in_sample_sharpes)
    out_s = np.asarray(out_sample_sharpes)
    if len(in_s) != len(out_s) or len(in_s) == 0:
        raise ValueError("in_sample_sharpes and out_sample_sharpes must be equal-length")
    best_idx = np.argmax(in_s)
    out_rank = stats.rankdata(out_s)[best_idx] / len(out_s)  # percentile rank, 1.0 = best
    return float(out_rank < 0.5)
