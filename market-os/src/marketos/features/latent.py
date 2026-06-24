"""Latent-state compression — the Layer-3 seed (representation over enumeration).

The frontier isn't hand-listing oil/copper/flights; it's learning a compressed latent state Zₜ
that the raw observations Yₜ project onto — the way AlphaFold learned structure without being
handed chemistry equations. This module is the honest first cut at that: **cross-sectional
PCA** on the standardized feature panel.

Why PCA first (not a deep autoencoder yet): it is deterministic, reproducible, has no training
loop to babysit, and recovers exactly the statistical risk factors that APT/PCA-factor models
use. The leading components are the dominant *latent* directions the whole universe is moving
along right now — a data-driven basis, not a hand-specified one. An autoencoder is the natural
next upgrade once we want non-linear Zₜ; the interface here (return per-symbol latent
coordinates + market-level spectrum) is built so that swap is a drop-in.

Outputs:
  - Per symbol: its coordinates on the top-k latent axes (`z1..zk`) — where each name sits in
    the compressed state.
  - Market level (`_latent`): explained-variance spectrum, effective dimensionality
    (participation ratio), and PC1 share — how *concentrated* the market's variance is (a high
    PC1 share = everything moving together = correlation/risk-off regime).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SKIP = {"asof_ts", "knowledge_ts", "symbol", "feature_family"}


def compute_latent_factors(
    symbol_vectors: dict[str, dict],
    *,
    n_components: int = 8,
) -> tuple[pd.DataFrame, dict]:
    """Compress the cross-section of feature vectors into latent coordinates via PCA.

    symbol_vectors: {symbol: flat feature dict} — the latest composite vector per symbol.
    Returns (per_symbol_latent_df, market_latent_dict).

    The PCA is fit on the symbols×features matrix (each row a name, each column a feature),
    standardized per feature. With ~19 names the rank is bounded by n_symbols-1, so we clamp
    n_components accordingly and report what variance the retained axes explain.
    """
    # 1. Assemble the symbols × features matrix from the latest vectors.
    rows, syms = [], []
    for sym, vec in symbol_vectors.items():
        clean = {k: v for k, v in vec.items()
                 if k not in _SKIP and isinstance(v, (int, float)) and np.isfinite(v)}
        if clean:
            rows.append(clean)
            syms.append(sym)
    if len(rows) < 3:
        return pd.DataFrame(), {}

    mat = pd.DataFrame(rows, index=syms)
    # Keep only features present (non-NaN) for (almost) all names so PCA is well-posed.
    mat = mat.loc[:, mat.notna().mean() >= 0.8]
    mat = mat.fillna(mat.mean())
    if mat.shape[1] < 3:
        return pd.DataFrame(), {}

    # 2. Standardize per feature (PCA is scale-sensitive; raw units would let volume dominate).
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0).replace(0, np.nan)
    Z = ((mat - mu) / sd).fillna(0.0)

    # 3. SVD → principal components. Components limited by min(n_symbols-1, n_features).
    k = int(min(n_components, Z.shape[0] - 1, Z.shape[1]))
    if k < 1:
        return pd.DataFrame(), {}
    try:
        U, S, _Vt = np.linalg.svd(Z.values, full_matrices=False)
    except np.linalg.LinAlgError as e:
        logger.warning(f"latent PCA SVD failed: {e}")
        return pd.DataFrame(), {}

    var = (S ** 2)
    total_var = var.sum()
    explained = var / total_var if total_var > 0 else np.zeros_like(var)

    # Per-symbol scores on the top-k axes (U·S gives coordinates).
    scores = U[:, :k] * S[:k]
    now = datetime.now(timezone.utc)
    per_symbol = pd.DataFrame(scores, index=Z.index,
                              columns=[f"z{i+1}" for i in range(k)])
    per_symbol = per_symbol.reset_index().rename(columns={"index": "symbol"})
    per_symbol["asof_ts"] = now
    per_symbol["knowledge_ts"] = now

    # 4. Market-level spectrum + effective dimensionality.
    #    Participation ratio PR = (Σλ)² / Σλ²  — how many axes meaningfully carry variance.
    lam = var
    pr = float((lam.sum() ** 2) / (lam ** 2).sum()) if (lam ** 2).sum() > 0 else 0.0
    market = {
        "asof_ts": now,
        "knowledge_ts": now,
        "latent_pc1_share": float(explained[0]) if len(explained) else 0.0,
        "latent_pc2_share": float(explained[1]) if len(explained) > 1 else 0.0,
        "latent_pc3_share": float(explained[2]) if len(explained) > 2 else 0.0,
        "latent_top3_share": float(explained[:3].sum()) if len(explained) else 0.0,
        "latent_participation_ratio": pr,         # ↑ = diversified, ↓ = one-factor (risk-off)
        "latent_n_components": k,
        "latent_n_features": int(Z.shape[1]),
    }
    for i in range(min(k, 8)):
        market[f"latent_explained_{i+1}"] = float(explained[i])

    return per_symbol, market
