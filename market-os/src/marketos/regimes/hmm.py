"""Regime detection via a Gaussian Hidden Markov Model on returns/volatility.

Markets are nonstationary: a strategy with positive expectancy in a trending, low-vol
regime can bleed in a choppy, high-vol one. We estimate a latent regime so the portfolio
can scale risk, switch model weights, or stand aside. Uses hmmlearn if present; otherwise
falls back to a transparent volatility-tercile labeler so the pipeline always runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REGIME_NAMES = {0: "calm_trend", 1: "neutral", 2: "stress_chop"}


def detect_regimes(returns: pd.Series, n_states: int = 3, seed: int = 7) -> pd.DataFrame:
    """Return a frame with the regime label per timestamp (ordered by volatility), PLUS
    soft posterior probabilities per state (`regime_prob_0..N-1`).

    The hard label (`regime`) is what's been used so far for regime-CONDITIONAL analysis
    (subset the data where regime==X, recompute IC). The soft probabilities are for feeding
    regime uncertainty *into* a model as a continuous feature instead — e.g. AlphaModel
    trained on [sup_z1..z4, regime_prob_0, regime_prob_1, regime_prob_2] doesn't need a hard
    regime switch at the boundary, and partial-stress-probability days aren't artificially
    forced into "calm" or "stress." Mixture-of-experts (fully separate per-regime models) is
    deferred until there's enough effective-N per regime to support it — see project notes;
    this soft-probability feature is the right-sized step before that.
    """
    r = returns.dropna()
    feats = np.column_stack([
        r.values,
        r.rolling(10).std().bfill().values,
    ])
    try:
        from hmmlearn.hmm import GaussianHMM

        model = GaussianHMM(n_components=n_states, covariance_type="diag",
                            n_iter=200, random_state=seed)
        model.fit(feats)
        states = model.predict(feats)
        proba_raw = model.predict_proba(feats)  # columns indexed by RAW state id
        # order states by their volatility so labels (and probability columns) are stable
        vol_by_state = {s: feats[states == s, 1].mean() for s in np.unique(states)}
        order = {s: rank for rank, (s, _) in enumerate(sorted(vol_by_state.items(), key=lambda x: x[1]))}
        labels = np.array([order[s] for s in states])
        # Reindex probability columns into the same vol-ordered rank space as `labels`.
        proba = np.zeros_like(proba_raw)
        for raw_state, rank in order.items():
            proba[:, rank] = proba_raw[:, raw_state]
    except Exception:
        labels = _vol_tercile_labels(feats[:, 1])
        # No real posterior available from the tercile fallback — one-hot on the hard label
        # is the honest representation (zero uncertainty), not a fabricated soft estimate.
        proba = np.eye(n_states)[labels]

    out = pd.DataFrame(index=r.index)
    out["regime"] = labels
    out["regime_name"] = out["regime"].map(REGIME_NAMES).fillna("neutral")
    for i in range(proba.shape[1]):
        out[f"regime_prob_{i}"] = proba[:, i]
    return out


def _vol_tercile_labels(vol: np.ndarray) -> np.ndarray:
    q1, q2 = np.nanpercentile(vol, [33, 66])
    return np.where(vol <= q1, 0, np.where(vol <= q2, 1, 2))
