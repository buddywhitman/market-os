"""Outcome-trained latent factors — the supervised upgrade to the cross-sectional PCA in
latent.py, validated through the purged walk-forward harness in labeling.py.

Why PLS, not an autoencoder: with ~19 symbols and a few years of history, a neural
autoencoder has far more parameters than independent samples — it would memorize, not
compress. Partial Least Squares finds latent components that maximize covariance with the
*targets* (not just variance of the inputs, like PCA), is a closed-form fit with no training
loop, and is exactly right-sized for this data regime. It is also a genuine drop-in: same
"per-symbol coordinates on k axes" interface as compute_latent_factors in latent.py.

Why 4 joint targets, not 1: PLS with a single scalar target concentrates almost all signal in
component 1 (rank-1 target). Using a *matrix* target — forward 5d return, forward 20d return,
forward 20d volatility, forward 20d max-abs-move (tail proxy) — lets PLS2 find several genuinely
distinct components, each loading on a different blend of direction/dispersion/tail risk. This
also directly answers the original ask ("MI against 5d return, 20d return, volatility, large
move probability") by making those four series the supervision signal for the whole latent
space, rather than four independent univariate screens.

Honesty constraint (the actual point of this module): every reported IC is OUT-OF-SAMPLE,
computed via the purged+embargoed walk-forward splits in labeling.py, with effective sample
size from sample_uniqueness — not the in-sample fit quality. A component that doesn't survive
this is reported as such (oos_ic_ir near/below 0), not hidden.

Scope limitation, stated explicitly: this trains on TECHNICAL features only, reconstructed
from raw OHLCV history (the only family with deep, honestly point-in-time-reconstructable
history right now — see prioritize_subspace_job). Fundamental/broadcast/xrank families have
only a single live Postgres snapshot; they cannot be honestly walk-forward validated until
daily history accumulates. Extending this module to those families is a TODO gated on that
history existing, not a design choice.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from marketos.backtest.walkforward import n_split_walk_forward
from marketos.features.labeling import (
    sample_uniqueness,
    effective_sample_size,
)

logger = logging.getLogger(__name__)

_SKIP = {"asof_ts", "knowledge_ts", "symbol", "feature_family"}
TARGET_NAMES = ["fwd_ret_5d", "fwd_ret_20d", "fwd_vol_20d", "fwd_large_move_20d"]
_MAX_HORIZON = 20  # = max horizon among targets; drives purge width


def build_targets(close: pd.Series) -> pd.DataFrame:
    """The 4 joint supervision targets, all strictly causal (each uses only future bars
    relative to its index date — verified via the shift-then-rolling identity below).

    fwd_vol_20d / fwd_large_move_20d trick: `ret.shift(-20).rolling(20)` computed at position i
    reads shifted values at [i-19..i], i.e. ret[i+1..i+20] — exactly "the 20 returns strictly
    after day i" with no look-back contamination.
    """
    ret = close.pct_change()
    out = pd.DataFrame(index=close.index)
    out["fwd_ret_5d"] = close.shift(-5) / close - 1.0
    out["fwd_ret_20d"] = close.shift(-20) / close - 1.0
    out["fwd_vol_20d"] = ret.shift(-20).rolling(20).std() * np.sqrt(252)
    out["fwd_large_move_20d"] = ret.abs().shift(-20).rolling(20).max()
    return out


def _fit_scaler(train_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Mean/std computed on TRAIN ONLY — applying these to test data is what keeps the
    standardization itself leakage-free (a global mean/std would leak test-period statistics
    into the training transform)."""
    mu = train_df.mean(axis=0)
    sd = train_df.std(axis=0).replace(0, np.nan).fillna(1.0)
    return mu, sd


def _apply_scaler(df: pd.DataFrame, mu: pd.Series, sd: pd.Series) -> pd.DataFrame:
    return ((df - mu) / sd).fillna(0.0)


def _cross_sectional_ic(pred: pd.Series, actual: pd.Series, dates: pd.Series) -> float:
    """Mean per-date cross-sectional rank-IC between a predicted score and an actual target,
    across whatever symbols/dates are present in this (already-pooled) frame."""
    df = pd.DataFrame({"pred": pred, "actual": actual, "date": dates}).dropna()
    if df.empty:
        return np.nan
    ics = df.groupby("date").apply(
        lambda g: g["pred"].corr(g["actual"], method="spearman") if len(g) >= 3 else np.nan
    )
    return float(ics.dropna().mean()) if ics.notna().any() else np.nan


def fit_supervised_latent(
    symbol_ohlcv: dict[str, pd.DataFrame],
    *,
    n_components: int = 4,
    n_splits: int = 4,
    embargo_frac: float = 0.03,
    min_rows_per_symbol: int = 300,
    regime_series: pd.Series | None = None,
) -> dict:
    """Pool a (date, symbol) panel of technical features + the 4 forward targets, validate
    PLS2 latent factors via purged walk-forward CV, then final-fit for production projection.

    regime_series: optional date-indexed regime label (e.g. from regimes.hmm.detect_regimes),
    used for the regime-conditional breakdown below. A global OOS IC can hide the fact that a
    "validated" component only works in calm markets and inverts during stress — exactly the
    kind of conditional/regime-dependent structure that survives crowding and that marginal
    screening (the original critique's complaint) would discard. If omitted, regime_validation
    is returned empty rather than silently skipped — callers can tell the difference.

    Returns:
      latent            : DataFrame, one row per symbol with z1..zk (the LATEST projection —
                          same shape as compute_latent_factors' per-symbol output).
      validation        : DataFrame, one row per (component, target) with oos_ic_mean,
                          oos_ic_ir, oos_ic_hit_rate, n_folds — pooled across all regimes.
      regime_validation : DataFrame, one row per (regime, component, target) — the same
                          statistics computed ONLY on out-of-sample dates falling in that
                          regime. Empty if regime_series wasn't provided.
      market            : dict — pooled effective sample size, n components actually validated.
    """
    from marketos.features.technical import build_technical_features

    # 1. Build per-symbol (features, targets) reindexed to a master calendar.
    #    Use the longest-history symbol present as the master clock so younger listings
    #    (PLTR/RKLB/GEV/CEG/VST) just contribute a shorter, NaN-padded ragged panel — the
    #    correct way to handle unequal IPO/spinoff dates rather than truncating everyone to
    #    the youngest name's history.
    feat_by_sym: dict[str, pd.DataFrame] = {}
    targ_by_sym: dict[str, pd.DataFrame] = {}
    for sym, ohlcv in symbol_ohlcv.items():
        if ohlcv is None or ohlcv.empty or len(ohlcv) < min_rows_per_symbol:
            continue
        try:
            feats = build_technical_features(ohlcv)
        except Exception as e:
            logger.warning(f"supervised_latent features {sym}: {e}")
            continue
        feats = feats.drop(columns=[c for c in _SKIP if c in feats.columns], errors="ignore")
        targs = build_targets(ohlcv["close"])
        feat_by_sym[sym] = feats
        targ_by_sym[sym] = targs

    if len(feat_by_sym) < 5:
        logger.warning("supervised_latent: fewer than 5 symbols with enough history, skipping")
        return {}

    master_dates = max((f.index for f in feat_by_sym.values()), key=len)
    feature_cols = sorted(set.intersection(*[set(f.columns) for f in feat_by_sym.values()]))
    if len(feature_cols) < 10:
        logger.warning("supervised_latent: too few common feature columns, skipping")
        return {}

    # 2. Master-date-indexed purged/embargoed split (the SAME date boundaries apply to every
    #    symbol — the calendar is shared even though each symbol's row availability differs).
    n_master = len(master_dates)
    try:
        splits = n_split_walk_forward(
            n_master, n_splits=n_splits, label_horizon=_MAX_HORIZON, embargo_frac=embargo_frac,
        )
    except ValueError as e:
        logger.warning(f"supervised_latent: {e}")
        return {}

    # 3. Walk-forward validation loop.
    fold_records: list[dict] = []  # one row per (fold, component, target)
    all_uniqueness_weights: list[pd.Series] = []
    oos_records: list[dict] = []  # raw (date, component, target, pred, actual) for regime cut

    for fold_i, fold in enumerate(splits):
        train_pos, test_pos = fold.train_idx, fold.test_idx
        if len(train_pos) < 50 or len(test_pos) < 10:
            continue
        train_dates = master_dates[train_pos]
        test_dates = master_dates[test_pos]

        X_train_parts, Y_train_parts = [], []
        X_test_parts, Y_test_parts, test_date_parts = [], [], []
        for sym in feat_by_sym:
            f = feat_by_sym[sym].reindex(master_dates)[feature_cols]
            t = targ_by_sym[sym].reindex(master_dates)[TARGET_NAMES]
            tr_mask = f.index.isin(train_dates) & f.notna().any(axis=1) & t.notna().all(axis=1)
            te_mask = f.index.isin(test_dates) & f.notna().any(axis=1) & t.notna().all(axis=1)
            if tr_mask.sum() > 0:
                X_train_parts.append(f[tr_mask])
                Y_train_parts.append(t[tr_mask])
            if te_mask.sum() > 0:
                X_test_parts.append(f[te_mask])
                Y_test_parts.append(t[te_mask])
                test_date_parts.append(pd.Series(f.index[te_mask], index=f.index[te_mask]))

        if not X_train_parts or not X_test_parts:
            continue
        X_train = pd.concat(X_train_parts)
        Y_train = pd.concat(Y_train_parts)
        X_test = pd.concat(X_test_parts)
        Y_test = pd.concat(Y_test_parts)
        test_dates_flat = pd.concat(test_date_parts)

        mu, sd = _fit_scaler(X_train)
        Xtr = _apply_scaler(X_train, mu, sd)
        Xte = _apply_scaler(X_test, mu, sd)
        Ymu, Ysd = _fit_scaler(Y_train)
        Ytr = _apply_scaler(Y_train, Ymu, Ysd)

        k = min(n_components, Xtr.shape[1], Ytr.shape[1], max(1, Xtr.shape[0] // 20))
        if k < 1:
            continue
        try:
            pls = PLSRegression(n_components=k, scale=False)
            pls.fit(Xtr.values, Ytr.values)
        except Exception as e:
            logger.warning(f"supervised_latent PLS fit fold {fold_i}: {e}")
            continue

        Ztest = pls.transform(Xte.values)  # out-of-sample latent scores
        for comp in range(k):
            score = pd.Series(Ztest[:, comp], index=Xte.index)
            for target in TARGET_NAMES:
                actual = Y_test[target]
                ic = _cross_sectional_ic(score, actual, test_dates_flat)
                if np.isfinite(ic):
                    fold_records.append({"fold": fold_i, "component": comp + 1,
                                        "target": target, "ic": ic})
                # Raw per-observation rows, kept so regime-conditional IC can be recomputed
                # below without re-running the fold loop (regime is a per-date cut on data
                # we already have — the dates here never overlap across folds by construction).
                valid = np.isfinite(score.values) & np.isfinite(actual.values)
                oos_records.extend({
                    "date": d, "component": comp + 1, "target": target,
                    "pred": p, "actual": a,
                } for d, p, a in zip(test_dates_flat.values[valid], score.values[valid],
                                     actual.values[valid]))

        # Uniqueness weighting for this fold's effective-N (fixed max-horizon labels: every
        # observation's window is [i, i+_MAX_HORIZON] — feed that directly to sample_uniqueness).
        touch = pd.Series(np.arange(len(X_test)) + _MAX_HORIZON, index=X_test.index)
        fake_labels = pd.DataFrame({"touch_idx": touch.clip(upper=len(X_test) - 1).values},
                                   index=X_test.index)
        all_uniqueness_weights.append(sample_uniqueness(fake_labels))

    if not fold_records:
        logger.warning("supervised_latent: no fold produced valid OOS IC, aborting")
        return {}

    fr = pd.DataFrame(fold_records)
    val_rows = []
    for (comp, target), g in fr.groupby(["component", "target"]):
        ic_mean = float(g["ic"].mean())
        ic_std = float(g["ic"].std())
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
        hit_rate = float((np.sign(g["ic"]) == np.sign(ic_mean)).mean()) if ic_mean != 0 else 0.0
        val_rows.append({"component": comp, "target": target, "oos_ic_mean": ic_mean,
                         "oos_ic_ir": ic_ir, "oos_ic_hit_rate": hit_rate, "n_folds": len(g)})
    validation = pd.DataFrame(val_rows).sort_values(["component", "target"]).reset_index(drop=True)

    # 3b. Regime-conditional breakdown — same per-date-IC-then-average methodology, just
    # subset to OOS dates falling in each regime. Reuses the raw oos_records collected above
    # rather than re-running the fold loop; dates never repeat across folds, so pooling them
    # here doesn't double-count anything.
    regime_validation = pd.DataFrame()
    if regime_series is not None and oos_records:
        oos_df = pd.DataFrame(oos_records)
        oos_df["date"] = pd.to_datetime(oos_df["date"])
        rs = regime_series.copy()
        rs.index = pd.to_datetime(rs.index)
        regime_lookup = rs.reindex(sorted(oos_df["date"].unique())).ffill()
        oos_df["regime"] = oos_df["date"].map(regime_lookup)
        oos_df = oos_df.dropna(subset=["regime"])

        regime_rows = []
        for (regime, comp, target), g in oos_df.groupby(["regime", "component", "target"]):
            per_date_ic = g.groupby("date").apply(
                lambda d: d["pred"].corr(d["actual"], method="spearman") if len(d) >= 3 else np.nan
            ).dropna()
            if len(per_date_ic) < 5:
                continue
            ic_mean = float(per_date_ic.mean())
            ic_std = float(per_date_ic.std())
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
            hit_rate = (float((np.sign(per_date_ic) == np.sign(ic_mean)).mean())
                       if ic_mean != 0 else 0.0)
            regime_rows.append({"regime": int(regime), "component": comp, "target": target,
                                "oos_ic_mean": ic_mean, "oos_ic_ir": ic_ir,
                                "oos_ic_hit_rate": hit_rate, "n_dates": len(per_date_ic)})
        if regime_rows:
            regime_validation = (pd.DataFrame(regime_rows)
                                 .sort_values(["regime", "component", "target"])
                                 .reset_index(drop=True))

    effective_n = sum(effective_sample_size(w) for w in all_uniqueness_weights)

    # 4. Final production fit — all data up to the last embargo boundary — then project the
    #    LATEST available row per symbol for the per-symbol output (same shape as latent.py).
    last_train_pos = splits[-1].train_idx
    if len(splits[-1].test_idx):
        last_train_pos = np.concatenate([last_train_pos, splits[-1].test_idx])
    final_dates = master_dates[np.sort(np.unique(last_train_pos))]

    X_final_parts, Y_final_parts, X_final_syms, latest_rows = [], [], [], {}
    for sym in feat_by_sym:
        f = feat_by_sym[sym].reindex(master_dates)[feature_cols]
        t = targ_by_sym[sym].reindex(master_dates)[TARGET_NAMES]
        mask = f.index.isin(final_dates) & f.notna().any(axis=1) & t.notna().all(axis=1)
        if mask.sum() > 0:
            X_final_parts.append(f[mask])
            Y_final_parts.append(t[mask])
            X_final_syms.append((sym, int(mask.sum())))
        # Latest row for projection: most recent date with a fully-finite feature vector.
        valid_idx = f.dropna(how="all").index
        if len(valid_idx):
            latest_rows[sym] = f.loc[valid_idx[-1]]

    if not X_final_parts:
        return {"latent": pd.DataFrame(), "validation": validation,
               "regime_validation": regime_validation, "historical_panel": pd.DataFrame(),
               "market": {"effective_n": effective_n, "n_components_validated": int(fr["component"].max())}}

    X_final = pd.concat(X_final_parts)
    Y_final = pd.concat(Y_final_parts)
    mu_f, sd_f = _fit_scaler(X_final)
    Yf_mu, Yf_sd = _fit_scaler(Y_final)
    Xf = _apply_scaler(X_final, mu_f, sd_f)
    Yf = _apply_scaler(Y_final, Yf_mu, Yf_sd)
    k_final = min(n_components, Xf.shape[1], Yf.shape[1])
    pls_final = PLSRegression(n_components=k_final, scale=False)
    pls_final.fit(Xf.values, Yf.values)

    # Project the ENTIRE final-fit panel (X_final spans nearly all of history, since the last
    # walk-forward fold's train set is an expanding window) into latent space, paired with the
    # RAW (un-standardized) realized outcomes. This is the substrate for the analog/market-
    # memory engine: "what happened, in outcome-trained-latent terms, every day in history" —
    # not just the latest snapshot. Carries a 'symbol' column reconstructed from which
    # X_final_parts block each row came from (concat preserves row order per block).
    sym_labels = np.concatenate([[sym] * n for sym, n in X_final_syms])
    Z_hist = pls_final.transform(Xf.values)
    historical_panel = pd.DataFrame(
        Z_hist, columns=[f"sup_z{i+1}" for i in range(k_final)], index=X_final.index)
    historical_panel["symbol"] = sym_labels
    historical_panel["date"] = X_final.index
    for col in TARGET_NAMES:
        historical_panel[col] = Y_final[col].values
    if regime_series is not None:
        rs = regime_series.copy()
        rs.index = pd.to_datetime(rs.index)
        regime_lookup = rs.reindex(sorted(set(X_final.index))).ffill()
        historical_panel["regime"] = pd.to_datetime(historical_panel["date"]).map(regime_lookup)
    historical_panel = historical_panel.reset_index(drop=True)

    now = datetime.now(timezone.utc)
    per_symbol_rows = []
    for sym, row in latest_rows.items():
        row_aligned = row.reindex(feature_cols)
        x_scaled = ((row_aligned - mu_f) / sd_f).fillna(0.0).values.reshape(1, -1)
        z = pls_final.transform(x_scaled)[0]
        rec = {"symbol": sym, **{f"sup_z{i+1}": float(z[i]) for i in range(k_final)}}
        per_symbol_rows.append(rec)
    latent_df = pd.DataFrame(per_symbol_rows)
    latent_df["asof_ts"] = now
    latent_df["knowledge_ts"] = now

    market = {
        "asof_ts": now, "knowledge_ts": now,
        "sup_latent_effective_n": float(effective_n),
        "sup_latent_n_components": int(k_final),
        "sup_latent_n_symbols": int(len(latest_rows)),
        "sup_latent_best_ic_ir": float(validation["oos_ic_ir"].abs().max()) if not validation.empty else 0.0,
    }
    if not regime_validation.empty:
        # Regime-robustness check: does the single strongest global component keep its SIGN
        # across every regime it has enough OOS observations in? A flip means the "validated"
        # signal is a regime artifact, not a stable factor — exactly what marginal screening
        # (the original critique) would hide by reporting only the pooled global IC.
        best_row = validation.loc[validation["oos_ic_ir"].abs().idxmax()]
        best_comp, best_target = best_row["component"], best_row["target"]
        same = regime_validation[(regime_validation["component"] == best_comp) &
                                 (regime_validation["target"] == best_target)]
        if len(same) > 1:
            market["sup_latent_best_regime_sign_stable"] = bool(
                same["oos_ic_mean"].apply(np.sign).nunique() == 1)
        market["sup_latent_n_regimes_validated"] = int(regime_validation["regime"].nunique())

    return {"latent": latent_df, "validation": validation,
           "regime_validation": regime_validation, "historical_panel": historical_panel,
           "market": market}
