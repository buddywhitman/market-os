"""Dynamic active-subspace prioritization — "knowing what to ignore."

There is no finite feature set that fully describes markets, and feature importance is
*non-stationary*: once alpha is discovered, capital exploits it and the signal decays
(Goodhart). At any moment only a sparse subspace of the ~200+ features is "live," and which
subspace is live rotates with regime. So rather than feeding everything into one model, we
continuously score each feature's *current* relevance and surface the active subspace.

Methodology — cross-sectional rank Information Coefficient (IC):
  For each date t, rank a feature across the universe and rank forward returns across the
  universe, then take their rank-correlation → IC_f[t]. This is the Fama-MacBeth / Grinold
  style cross-sectional IC. Stacking over t gives an IC time series per feature, from which:

    ic_mean    — average predictive sign/strength
    ic_ir      — mean / std  (information ratio; stability-adjusted strength)
    ic_t       — t-stat of the IC series (is it distinguishable from noise?)
    ic_decay   — recent-window IC minus older-window IC  (Goodhart detector: <0 = decaying)
    hit_rate   — fraction of weeks IC keeps its dominant sign  (consistency)
    regime_ic  — IC computed only within the current regime  (regime-conditional relevance)

  Active subspace = features ranked by a decay-penalized, stability-weighted score. A feature
  with strong but *fading* IC is deprioritized relative to a weaker but *stable, improving* one.

Caveat we make explicit: with a 19-name universe the cross-section is thin (Spearman over ~19
points), so single-week IC is noisy. The aggregation over many weeks + the IR/t-stat framing is
what makes it usable. This is a relevance *ranking*, not a trading signal by itself.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Bookkeeping columns never scored as features
_SKIP = {"asof_ts", "knowledge_ts", "symbol", "feature_family"}


def _rank_normalize(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank along columns (symbols) per row (date), scaled to [-0.5, 0.5]."""
    r = panel.rank(axis=1)
    n = panel.notna().sum(axis=1)
    return r.sub(0.5, axis=0).div(n.replace(0, np.nan), axis=0) - 0.0  # ~uniform centered


def _cross_sectional_ic(feature_wide: pd.DataFrame, fwd_wide: pd.DataFrame) -> pd.Series:
    """Vectorized per-date cross-sectional rank IC between two date×symbol matrices.

    Both inputs are date-indexed, symbol-columned, already aligned. Returns IC_f[t].
    """
    F = feature_wide.rank(axis=1)
    R = fwd_wide.rank(axis=1)
    # Row-wise (per-date) Pearson correlation of ranks == Spearman IC.
    Fc = F.sub(F.mean(axis=1), axis=0)
    Rc = R.sub(R.mean(axis=1), axis=0)
    num = (Fc * Rc).sum(axis=1)
    den = np.sqrt((Fc ** 2).sum(axis=1) * (Rc ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    return ic.dropna()


def compute_active_subspace(
    symbol_feature_history: dict[str, pd.DataFrame],
    symbol_ohlcv: dict[str, pd.DataFrame],
    *,
    fwd_horizon: int = 5,
    recent_window: int = 63,
    older_window: int = 189,
    regime_series: pd.Series | None = None,
    top_k: int = 50,
) -> pd.DataFrame:
    """Score every feature's current relevance and return a ranked relevance table.

    symbol_feature_history: {symbol: full technical feature history DataFrame (date-indexed)}
    symbol_ohlcv:           {symbol: ohlcv} used to build forward returns
    fwd_horizon:            forward-return horizon in bars for the IC target
    recent_window/older_window: trailing weeks for the Goodhart decay comparison
    regime_series:          optional date-indexed regime labels for regime-conditional IC
    top_k:                  how many features to flag as the active subspace

    Returns one row per feature with ic_mean/ic_ir/ic_t/ic_decay/hit_rate/regime_ic/active.
    """
    if not symbol_feature_history:
        return pd.DataFrame()

    # 1. Build aligned forward returns per symbol (target the IC predicts).
    fwd_by_sym: dict[str, pd.Series] = {}
    for sym, ohlcv in symbol_ohlcv.items():
        if ohlcv is None or ohlcv.empty or "close" not in ohlcv:
            continue
        close = ohlcv["close"]
        fwd = close.shift(-fwd_horizon) / close - 1.0   # forward return (causal target)
        fwd.index = pd.to_datetime(fwd.index)
        fwd_by_sym[sym] = fwd

    if not fwd_by_sym:
        return pd.DataFrame()
    fwd_wide_all = pd.DataFrame(fwd_by_sym)

    # 2. Determine the feature universe (intersection of columns across symbols).
    feature_cols: set[str] = set()
    for df in symbol_feature_history.values():
        feature_cols |= {c for c in df.columns if c not in _SKIP}
    feature_cols = sorted(feature_cols)

    # Pre-index each symbol's feature frame by datetime for fast pivoting.
    norm_hist = {}
    for sym, df in symbol_feature_history.items():
        d = df.copy()
        d.index = pd.to_datetime(d.index)
        norm_hist[sym] = d

    now = datetime.now(timezone.utc)
    rows = []

    for feat in feature_cols:
        # Build a date×symbol matrix for this feature.
        cols = {}
        for sym, df in norm_hist.items():
            if feat in df.columns:
                cols[sym] = df[feat]
        if len(cols) < 5:        # need a minimum cross-section
            continue
        feat_wide = pd.DataFrame(cols)
        # Align with forward returns on the shared dates/symbols.
        common_syms = [s for s in feat_wide.columns if s in fwd_wide_all.columns]
        if len(common_syms) < 5:
            continue
        fw = feat_wide[common_syms]
        rw = fwd_wide_all[common_syms].reindex(fw.index)
        # Drop rows where either side is all-NaN.
        valid = fw.notna().sum(axis=1).ge(5) & rw.notna().sum(axis=1).ge(5)
        fw, rw = fw[valid], rw[valid]
        if len(fw) < older_window // 2:
            continue

        ic = _cross_sectional_ic(fw, rw)
        if len(ic) < 20:
            continue

        ic_mean = float(ic.mean())
        ic_std = float(ic.std())
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
        ic_t = ic_mean / (ic_std / np.sqrt(len(ic))) if ic_std > 0 else 0.0
        dominant_sign = np.sign(ic_mean) if ic_mean != 0 else 1.0
        hit_rate = float((np.sign(ic) == dominant_sign).mean())

        recent_ic = float(ic.iloc[-recent_window:].mean()) if len(ic) >= recent_window else ic_mean
        older_slice = ic.iloc[-older_window:-recent_window]
        older_ic = float(older_slice.mean()) if len(older_slice) >= 10 else ic_mean
        # Decay measured on |IC| so it captures "signal fading" regardless of sign.
        ic_decay = abs(recent_ic) - abs(older_ic)

        # Regime-conditional IC (current regime only).
        regime_ic = np.nan
        if regime_series is not None and len(regime_series) > 0:
            rs = regime_series.reindex(ic.index).ffill()
            current_regime = rs.iloc[-1] if len(rs) else None
            if current_regime is not None:
                in_regime = ic[rs == current_regime]
                if len(in_regime) >= 10:
                    regime_ic = float(in_regime.mean())

        rows.append({
            "feature": feat,
            "ic_mean": ic_mean,
            "ic_ir": ic_ir,
            "ic_t": ic_t,
            "ic_recent": recent_ic,
            "ic_decay": ic_decay,
            "hit_rate": hit_rate,
            "regime_ic": regime_ic,
            "n_obs": len(ic),
        })

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)

    # 3. Composite relevance score — the heart of "what to pay attention to now":
    #    reward stable strength (|ic_ir|), reward consistency (hit_rate above coin-flip),
    #    reward current-regime fit, and PENALIZE decay (Goodhart). All standardized.
    def _z(s: pd.Series) -> pd.Series:
        sd = s.std()
        return (s - s.mean()) / sd if sd and np.isfinite(sd) and sd != 0 else s * 0.0

    regime_term = table["regime_ic"].abs().fillna(table["ic_recent"].abs())
    table["relevance"] = (
        1.0 * _z(table["ic_ir"].abs())
        + 0.6 * _z((table["hit_rate"] - 0.5).clip(lower=0))
        + 0.6 * _z(regime_term)
        + 0.8 * _z(table["ic_decay"])        # positive decay (improving) lifts; fading sinks
    )
    table = table.sort_values("relevance", ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    table["active"] = (table["rank"] <= top_k).astype(int)
    table["asof_ts"] = now
    table["knowledge_ts"] = now
    return table


def subspace_summary(table: pd.DataFrame, top_k: int = 50) -> dict:
    """Collapse the relevance table into a single feature row for the store.

    Stored under symbol `_subspace`. Captures the *shape* of the active subspace so we can
    track over time how concentrated / rotating attention is.
    """
    if table is None or table.empty:
        return {}
    active = table[table["active"] == 1] if "active" in table.columns else table.head(top_k)
    now = datetime.now(timezone.utc)
    out: dict = {
        "asof_ts": now,
        "knowledge_ts": now,
        "subspace_size": int(len(active)),
        "subspace_mean_ic_ir": float(active["ic_ir"].abs().mean()),
        "subspace_mean_hit_rate": float(active["hit_rate"].mean()),
        "subspace_decaying_frac": float((active["ic_decay"] < 0).mean()),
        "subspace_total_scored": int(len(table)),
        # Attention concentration: how much relevance mass sits in the top decile (Herfindahl-ish)
        "subspace_top_ic_ir": float(table["ic_ir"].abs().max()),
    }
    # Record the top features by name+score so the dashboard can render the live subspace.
    for i, (_, r) in enumerate(active.head(20).iterrows()):
        out[f"top{i+1:02d}_{r['feature']}"] = round(float(r["relevance"]), 4)
    return out
