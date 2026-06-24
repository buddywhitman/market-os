"""Pre-flight diagnostics for the regime HMM: is this instrument's regime structure real,
or is the model degenerate / non-converged, making any downstream 'regime' label noise?

Run this BEFORE trusting detect_regimes() output for portfolio construction. A model that
"fits" without raising still silently produces garbage if it doesn't converge, collapses
to one dominant state, or assigns near-zero occupancy to a state — none of which raise an
exception, all of which were sitting unflagged in detect_regimes() until now.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class RegimeDiagnostics:
    instrument: str
    n_states: int
    converged: bool
    log_likelihood: float
    aic: float
    bic: float
    state_counts: dict
    occupancy_fraction: dict
    transition_matrix: list
    state_persistence: dict
    mean_return_per_state: dict
    volatility_per_state: dict
    mean_posterior_entropy: float
    verdict: str  # "good" | "marginal" | "bad"

    def as_dict(self) -> dict:
        return asdict(self)


def _build_feats(returns: pd.Series) -> tuple[pd.Series, np.ndarray]:
    r = returns.dropna()
    feats = np.column_stack([r.values, r.rolling(10).std().bfill().values])
    return r, feats


def _fit_and_diagnose(r: pd.Series, feats: np.ndarray, n_states: int, seed: int = 7) -> RegimeDiagnostics:
    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, random_state=seed)
    model.fit(feats)
    converged = bool(model.monitor_.converged)
    log_likelihood = float(model.score(feats))

    # diag covariance: n_states*(n_states-1) transmat free params + n_states*2 means
    # + n_states*2 variances + (n_states-1) initial-dist free params.
    n_params = n_states * (n_states - 1) + n_states * 2 * 2 + (n_states - 1)
    n_obs = len(feats)
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_obs) - 2 * log_likelihood

    states = model.predict(feats)
    proba = model.predict_proba(feats)
    transmat = model.transmat_

    state_counts = {int(s): int((states == s).sum()) for s in range(n_states)}
    occupancy = {k: v / n_obs for k, v in state_counts.items()}
    persistence = {int(s): float(transmat[s, s]) for s in range(n_states)}
    mean_ret = {int(s): (float(r.values[states == s].mean()) if (states == s).any() else float("nan"))
                for s in range(n_states)}
    vol = {int(s): (float(feats[states == s, 1].mean()) if (states == s).any() else float("nan"))
           for s in range(n_states)}
    row_entropy = -np.sum(proba * np.log(proba + 1e-12), axis=1)
    mean_entropy = float(row_entropy.mean())

    min_occupancy = min(occupancy.values())
    if not converged or min_occupancy < 0.03:
        verdict = "bad"
    elif min_occupancy < 0.08 or mean_entropy > 0.5:
        verdict = "marginal"
    else:
        verdict = "good"

    return RegimeDiagnostics(
        instrument="", n_states=n_states, converged=converged, log_likelihood=log_likelihood,
        aic=aic, bic=bic, state_counts=state_counts, occupancy_fraction=occupancy,
        transition_matrix=transmat.tolist(), state_persistence=persistence,
        mean_return_per_state=mean_ret, volatility_per_state=vol,
        mean_posterior_entropy=mean_entropy, verdict=verdict,
    )


def diagnose_instrument(instrument: str, returns: pd.Series,
                         candidate_n_states: tuple[int, ...] = (2, 3)) -> dict:
    """Fit each candidate n_states, score by BIC among converged+non-degenerate fits, and
    recommend either the best HMM config or a fallback if nothing qualifies."""
    r, feats = _build_feats(returns)
    candidates: dict[int, RegimeDiagnostics] = {}
    for n in candidate_n_states:
        try:
            diag = _fit_and_diagnose(r, feats, n)
            diag.instrument = instrument
            candidates[n] = diag
        except Exception:
            pass

    # verdict quality strictly outranks BIC: a "marginal" fit with better raw likelihood
    # is still a worse choice than a "good" fit with non-degenerate occupancy/entropy --
    # the whole point of this diagnostic is to not let goodness-of-fit hide a degenerate
    # state structure. Only fall back to BIC-among-marginal if no "good" candidate exists.
    good = {n: d for n, d in candidates.items() if d.verdict == "good"}
    marginal = {n: d for n, d in candidates.items() if d.verdict == "marginal"}
    if good:
        best_n = min(good, key=lambda n: good[n].bic)
        best, recommendation = good[best_n], f"hmm_n{best_n}"
    elif marginal:
        best_n = min(marginal, key=lambda n: marginal[n].bic)
        best, recommendation = marginal[best_n], f"hmm_n{best_n}_marginal"
    else:
        best_n, best, recommendation = None, None, "vol_percentile_fallback"

    return {"instrument": instrument, "candidates": candidates,
            "best_n_states": best_n, "best": best, "recommendation": recommendation}


def volatility_percentile_regime(returns: pd.Series, *, window: int = 20,
                                  calm_pct: float = 33.0, stress_pct: float = 66.0) -> pd.Series:
    """No HMM, no ML: bucket by trailing realized-vol's point-in-time expanding-window
    percentile rank (never looks at future vol, unlike a full-sample percentile). Returns
    0=calm, 1=neutral, 2=stress, matching detect_regimes()'s label convention. The
    fallback for instruments whose regime structure doesn't justify fitting a model at all.
    """
    vol = returns.rolling(window).std()
    min_periods = window * 3
    pct_rank = vol.expanding(min_periods=min_periods).apply(
        lambda s: float((s.values[:-1] < s.values[-1]).mean() * 100) if len(s) > 1 else 50.0,
        raw=False,
    )
    labels = pd.Series(1, index=returns.index, dtype=int)
    labels[pct_rank <= calm_pct] = 0
    labels[pct_rank >= stress_pct] = 2
    labels[pct_rank.isna()] = 1
    return labels
