"""Ownership & positioning feature family.

Where technical features describe price's own geometry, ownership features describe *who holds
the position and how crowded it is* — a partial observation of the hidden positioning state.
The signal is almost always in the **extremes and the changes**, not the level: crowded longs
unwind, institutional accumulation precedes re-rating, insider clusters carry private
information.

Inputs (any subset; degrades gracefully):
  - cot_features:    market-level COT positioning (from cftc_fetcher)
  - fii_dii:         NSE foreign/domestic institutional net flows (from nse_fetcher)
  - insider_panel:   per-symbol insider net bias (from finnhub / edgar)

Design: this builder normalizes each raw flow into (a) a z-score vs its own history where we
have history, or (b) a sign/intensity feature where we only have a snapshot. Cross-sectional
ranks are applied to per-symbol insider data so a name's positioning is judged relative to
peers, not in absolute share counts.

Factor lineage: Lakonishok-Lee 2001 (insider trading predicts returns); Cohen-Malloy-Pomorski
2012 (opportunistic vs routine insiders); COT extremes (Briese, Williams).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_z(x: float, mu: float, sd: float) -> float:
    if sd is None or not np.isfinite(sd) or sd == 0:
        return 0.0
    return (x - mu) / sd


def build_market_positioning_features(
    cot_row: dict | None = None,
    fii_dii_history: pd.DataFrame | None = None,
) -> dict:
    """Market-level positioning features (one row, stored under `_positioning`).

    cot_row: latest COT feature dict (already percentile-indexed by cftc_fetcher).
    fii_dii_history: time-indexed frame with columns like 'fii_net','dii_net' for z-scoring.
    """
    out: dict = {}

    # COT already arrives normalized (index 0-100, net %OI, momentum). Pass through with a
    # composite crowding score: average distance of each market's index from the neutral 50.
    if cot_row:
        idx_cols = [k for k in cot_row if k.endswith("_index")]
        if idx_cols:
            extremity = np.mean([abs(cot_row[k] - 50.0) for k in idx_cols]) / 50.0
            out["pos_cot_crowding"] = float(extremity)  # 0=neutral, 1=max crowded somewhere
            out["pos_cot_net_tilt"] = float(np.mean([(cot_row[k] - 50.0) / 50.0 for k in idx_cols]))
        # Carry through the per-market normalized fields
        for k, v in cot_row.items():
            if k.startswith("cot_") and isinstance(v, (int, float)):
                out[f"pos_{k}"] = float(v)

    # FII/DII institutional flow z-scores (India). Net inflow is risk-on for INR assets.
    if fii_dii_history is not None and not fii_dii_history.empty:
        for col in ("fii_net", "dii_net", "foreign_institutional_investors_net",
                    "domestic_institutional_investors_net"):
            if col in fii_dii_history.columns:
                s = pd.to_numeric(fii_dii_history[col], errors="coerce").dropna()
                if len(s) >= 10:
                    latest = float(s.iloc[-1])
                    out[f"pos_{col}_z"] = _safe_z(latest, float(s.mean()), float(s.std()))
                    out[f"pos_{col}_5d_sum"] = float(s.iloc[-5:].sum())
                    # Flow persistence: sign agreement over last 5 obs
                    signs = np.sign(s.iloc[-5:])
                    out[f"pos_{col}_persistence"] = float(abs(signs.sum()) / 5.0)

    return out


def build_symbol_positioning_features(insider_panel: pd.DataFrame | None) -> pd.DataFrame:
    """Per-symbol positioning features from insider data, cross-sectionally ranked.

    insider_panel: one row per symbol with any of:
        insider_net_3m, insider_mspr_avg, insider_purchase_3m, insider_sale_3m
    Returns one row per symbol with z-scored / ranked insider conviction.
    """
    if insider_panel is None or insider_panel.empty or "symbol" not in insider_panel.columns:
        return pd.DataFrame()

    df = insider_panel.copy().set_index("symbol")
    out = pd.DataFrame(index=df.index)

    for col in ("insider_net_3m", "insider_mspr_avg"):
        if col in df.columns:
            raw = pd.to_numeric(df[col], errors="coerce")
            out[f"pos_{col}"] = raw
            mu, sd = raw.mean(skipna=True), raw.std(skipna=True)
            out[f"pos_{col}_xs_z"] = (raw - mu) / (sd if sd and np.isfinite(sd) and sd != 0 else 1.0)
            out[f"pos_{col}_xs_rank"] = raw.rank(pct=True)

    # Net buy/sell intensity (purchase vs sale dollar balance)
    if "insider_purchase_3m" in df.columns and "insider_sale_3m" in df.columns:
        buy = pd.to_numeric(df["insider_purchase_3m"], errors="coerce").fillna(0)
        sell = pd.to_numeric(df["insider_sale_3m"], errors="coerce").fillna(0)
        denom = (buy + sell).replace(0, np.nan)
        out["pos_insider_buy_ratio"] = (buy / denom).fillna(0.5)

    out["symbol"] = out.index
    return out.reset_index(drop=True)
