"""Market Memory — historical analog search over the outcome-trained supervised-latent space.

The core idea (from the original roadmap critique): instead of

    1,169 features -> XGBoost -> point prediction

do

    current state -> outcome-trained latent embedding -> k nearest historical analogs
                   -> their realized-outcome distribution -> expectancy

This is what actually supports position sizing and risk: a distribution, not a point
estimate. Getting it right hinges on the SAME three things the rest of Phase 3/4 fixed:

  1. The distance metric must be the outcome-trained latent space (sup_z1..zk from
     latent_supervised.py), not raw features or unsupervised PCA. Euclidean distance over raw
     features is dominated by the high-variance, redundant momentum cluster — "nearest
     analogs" would just be other high-momentum days, not analogous regimes. PLS components
     are trained to predict the outcome, so distance in that space is distance in
     outcome-relevant terms.
  2. Regime-conditioning. A 2016 analog is drawn from a structurally different market than
     today (different rate regime, different vol structure). Matching only within the same
     HMM regime as "today" (when available) avoids comparing apples to a different fruit
     entirely; cross-regime matches are kept as a labeled fallback, not silently blended in.
  3. Honest occurrence counts. Forward-return labels overlap (a 20d-forward return sampled
     daily shares 19/20 of its window with the next day's), so raw "117 occurrences" inflates
     the true independent sample size by roughly the horizon. sample_uniqueness from
     labeling.py is reused here to report an effective occurrence count alongside the raw one.

No pgvector/faiss: at this scale (thousands of (date, symbol) snapshots, 4-dimensional
latent space) sklearn's NearestNeighbors is exact, fast, and adds no new infrastructure
dependency on the shared Postgres instance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from marketos.features.labeling import sample_uniqueness, effective_sample_size

OUTCOME_COLS = ["fwd_ret_5d", "fwd_ret_20d", "fwd_vol_20d", "fwd_large_move_20d"]


def _z_columns(panel: pd.DataFrame) -> list[str]:
    return sorted([c for c in panel.columns if c.startswith("sup_z")])


def find_analogs(
    current_z: dict[str, float],
    historical_panel: pd.DataFrame,
    *,
    current_date: pd.Timestamp | None = None,
    current_symbol: str | None = None,
    current_regime: int | None = None,
    k: int = 50,
    min_gap_days: int = 60,
) -> pd.DataFrame:
    """k nearest historical (date, symbol) analogs to current_z in latent-outcome space.

    Excludes any row within `min_gap_days` of current_date FOR THE SAME SYMBOL — without this,
    the nearest "analog" to today is trivially yesterday or last week (autocorrelated latent
    state), which tells you nothing. Cross-symbol matches close in time are fine and kept.

    If current_regime is given and the panel has a 'regime' column, restricts to same-regime
    history first; falls back to the full panel (regime-unfiltered, flagged via the returned
    'cross_regime' column) if that leaves fewer than `k` candidates.
    """
    z_cols = _z_columns(historical_panel)
    if not z_cols or historical_panel.empty:
        return pd.DataFrame()

    pool = historical_panel.copy()
    if current_symbol is not None and current_date is not None and "date" in pool.columns:
        # historical_panel's dates come from raw OHLCV (tz-naive); current_date is typically
        # datetime.now(timezone.utc) (tz-aware). Normalize both to tz-naive before subtracting
        # — pandas refuses to mix the two and the day-of-the-gap is all that matters here.
        pool_dates = pd.to_datetime(pool["date"])
        if pool_dates.dt.tz is not None:
            pool_dates = pool_dates.dt.tz_localize(None)
        cur_date = pd.to_datetime(current_date)
        if cur_date.tzinfo is not None:
            cur_date = cur_date.tz_localize(None)
        same_sym = pool["symbol"] == current_symbol
        too_close = same_sym & (pool_dates - cur_date).abs().dt.days.le(min_gap_days)
        pool = pool[~too_close]

    pool["cross_regime"] = False
    restricted = pool
    if current_regime is not None and "regime" in pool.columns:
        same_regime = pool[pool["regime"] == current_regime]
        if len(same_regime) >= k:
            restricted = same_regime
        else:
            pool = pool.copy()
            pool["cross_regime"] = pool["regime"] != current_regime

    if restricted.empty:
        return pd.DataFrame()

    x = restricted[z_cols].values
    q = np.array([[current_z.get(c, 0.0) for c in z_cols]])
    n_neighbors = min(k, len(restricted))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(x)
    dist, idx = nn.kneighbors(q)

    matches = restricted.iloc[idx[0]].copy()
    matches["distance"] = dist[0]
    return matches.sort_values("distance").reset_index(drop=True)


def summarize_analog_outcomes(matches: pd.DataFrame) -> dict:
    """Outcome-distribution summary over a set of analogs — occurrence counts (raw AND
    uniqueness-weighted effective N), win rate, central tendency, and tail risk. This is the
    "expectancy" half of the analog engine: what actually happened next, those times.
    """
    if matches.empty:
        return {}

    n = len(matches)
    out: dict = {"analog_n_raw": n}

    # Effective occurrence count: each match's forward-return window can overlap a NEIGHBORING
    # match's window (e.g. two analogs from adjacent trading days of the same historical
    # episode) — reuse the same uniqueness-weighting machinery as the OOS validation harness,
    # treating each match's max(20d) forward window as its "label span" for concurrency
    # purposes. Matches are already sorted by distance, not date, so re-sort by date per symbol
    # before computing touch_idx — concurrency only makes sense along a time axis.
    if "date" in matches.columns and "symbol" in matches.columns:
        eff_n = 0.0
        for sym, g in matches.groupby("symbol"):
            g = g.sort_values("date")
            touch = pd.Series(np.arange(len(g)) + 20, index=g.index)  # 20d = longest horizon
            fake_labels = pd.DataFrame({"touch_idx": touch.clip(upper=len(g) - 1).values},
                                       index=g.index)
            eff_n += effective_sample_size(sample_uniqueness(fake_labels))
        out["analog_n_effective"] = round(float(eff_n), 1)

    if "fwd_ret_20d" in matches.columns:
        r = matches["fwd_ret_20d"].dropna()
        if len(r):
            out["analog_win_rate_20d"] = float((r > 0).mean())
            out["analog_median_ret_20d"] = float(r.median())
            out["analog_mean_ret_20d"] = float(r.mean())
            out["analog_worst_ret_20d"] = float(r.min())
            out["analog_best_ret_20d"] = float(r.max())
            out["analog_p10_ret_20d"] = float(r.quantile(0.10))
            out["analog_p90_ret_20d"] = float(r.quantile(0.90))

    if "fwd_vol_20d" in matches.columns:
        v = matches["fwd_vol_20d"].dropna()
        if len(v):
            out["analog_median_vol_20d"] = float(v.median())

    if "cross_regime" in matches.columns:
        out["analog_cross_regime_frac"] = float(matches["cross_regime"].mean())

    return out
