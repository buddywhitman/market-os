"""Fundamental feature family — valuation, quality, growth, leverage, insider activity.

Raw fundamentals (from FMP/EDGAR) are not directly comparable across names: a P/E of 30 is
cheap for a hypergrowth name and expensive for a utility. The signal lives in the
*cross-section* — where a name sits relative to its peers — and in *change* (improving vs
deteriorating). So we z-score and rank each metric across the universe and add trend deltas.

Causality: fundamentals carry `knowledge_ts` = filing date (lagged from the period end).
Cross-sectional ranks use only contemporaneously-known values. No forward fills across
filing dates.

Factor lineage: Fama-French (size, value); Novy-Marx 2013 (gross profitability);
Sloan 1996 (accruals); Piotroski 2000 (F-score quality); Asness 2019 (quality-minus-junk).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Raw metric definitions: field name → direction (+1 = higher is bullish)
# ---------------------------------------------------------------------------
VALUATION_METRICS = {
    "peRatio":                -1,   # lower P/E = cheaper
    "priceToBookRatio":       -1,
    "priceToSalesRatio":      -1,
    "evToEbitda":             -1,
    "evToOperatingCashFlow":  -1,
    "earningsYield":          +1,   # higher = cheaper (inverse PE)
    "freeCashFlowYield":      +1,
    "dividendYield":          +1,
}
QUALITY_METRICS = {
    "returnOnEquity":             +1,
    "returnOnAssets":             +1,
    "returnOnInvestedCapital":    +1,
    "grossProfitMargin":          +1,
    "operatingProfitMargin":      +1,
    "netProfitMargin":            +1,
    "interestCoverage":           +1,
    "debtToEquity":               -1,
    "debtRatio":                  -1,
    "netDebtToEBITDA":            -1,
}
LIQUIDITY_METRICS = {
    "currentRatio":   +1,
    "quickRatio":     +1,
    "cashRatio":      +1,
}
EFFICIENCY_METRICS = {
    "assetTurnover":        +1,
    "inventoryTurnover":    +1,
    "receivablesTurnover":  +1,
}
GROWTH_METRICS = {
    "revenueGrowth":    +1,
    "netIncomeGrowth":  +1,
    "epsgrowth":        +1,
}

ALL_DIRECTIONAL = {**VALUATION_METRICS, **QUALITY_METRICS,
                   **LIQUIDITY_METRICS, **EFFICIENCY_METRICS, **GROWTH_METRICS}


def _cross_sectional_z(series: pd.Series) -> pd.Series:
    """Z-score across the universe (cross-section), robust to NaN."""
    mu = series.mean(skipna=True)
    sd = series.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sd


def _safe_float(x) -> float | None:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Piotroski F-Score (0-9) — 9 binary accounting-health signals
# ---------------------------------------------------------------------------
def _piotroski_fscore(row: pd.Series) -> int:
    """Compute Piotroski F-score from one row of fundamentals."""
    score = 0
    roa = _safe_float(row.get("returnOnAssets"))
    ocf = _safe_float(row.get("operatingCashFlowPerShare"))
    roa_qoq = _safe_float(row.get("returnOnAssets_qoq"))
    debt_qoq = _safe_float(row.get("debtRatio_qoq"))
    liq_qoq = _safe_float(row.get("currentRatio_qoq"))
    gpm_qoq = _safe_float(row.get("grossProfitMargin_qoq"))
    turn_qoq = _safe_float(row.get("assetTurnover_qoq"))

    if roa is not None and roa > 0:
        score += 1
    if ocf is not None and ocf > 0:
        score += 1
    if roa is not None and ocf is not None and roa > 0 and ocf > roa:
        score += 1  # OCF > ROA → high accrual quality
    if roa_qoq is not None and roa_qoq > 0:
        score += 1
    if debt_qoq is not None and debt_qoq < 0:
        score += 1  # leverage decreased
    if liq_qoq is not None and liq_qoq > 0:
        score += 1
    score += 1  # dilution check requires prior-period shares — give benefit of doubt
    if gpm_qoq is not None and gpm_qoq > 0:
        score += 1
    if turn_qoq is not None and turn_qoq > 0:
        score += 1
    return score


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_fundamental_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Turn a one-row-per-ticker fundamental panel into normalized factor features.

    Returns one row per ticker with:
      - raw_*         : raw metric value
      - z_*           : cross-sectional z-score
      - rank_*        : percentile rank (0-1) in universe, adjusted for direction
      - *_qoq/_yoy    : QoQ/YoY trend deltas (already in panel from fetcher)
      - value_composite, quality_composite, growth_composite, liquidity_composite
      - fscore, fscore_high_quality, fscore_low_quality
      - size_log_mktcap, size_tier, z_size, z_beta
      - insider_* signals
      - qmj_score, junk_score, bab_value
    """
    if panel.empty or "ticker" not in panel.columns:
        return pd.DataFrame()

    df = panel.copy().set_index("ticker")
    out = pd.DataFrame(index=df.index)

    # ── 1. Level features (z-score + percentile rank per group) ──────────────
    group_components: dict[str, list[pd.Series]] = {
        "value": [], "quality": [], "growth": [], "liquidity": [], "efficiency": [],
    }
    group_map = {
        "value": VALUATION_METRICS,
        "quality": QUALITY_METRICS,
        "growth": GROWTH_METRICS,
        "liquidity": LIQUIDITY_METRICS,
        "efficiency": EFFICIENCY_METRICS,
    }

    for group, metrics in group_map.items():
        for metric, direction in metrics.items():
            if metric not in df.columns:
                continue
            raw = pd.to_numeric(df[metric], errors="coerce")
            if raw.notna().sum() < 2:
                continue
            out[f"raw_{metric}"] = raw
            z = _cross_sectional_z(raw) * direction
            out[f"z_{metric}"] = z
            pct = raw.rank(pct=True)
            out[f"rank_{metric}"] = pct if direction > 0 else (1 - pct)
            group_components[group].append(z)

    # Composite scores (equal-weighted mean of available z-scores in group)
    for group, components in group_components.items():
        if components:
            out[f"{group}_composite"] = pd.concat(components, axis=1).mean(axis=1)

    # ── 2. QoQ/YoY trend features (pass-through with cross-section z) ────────
    trend_cols = [c for c in df.columns if c.endswith("_qoq") or c.endswith("_yoy")]
    for col in trend_cols:
        raw = pd.to_numeric(df[col], errors="coerce")
        if raw.notna().sum() < 2:
            continue
        out[f"raw_{col}"] = raw
        out[f"z_{col}"] = _cross_sectional_z(raw)

    # ── 3. Size features ──────────────────────────────────────────────────────
    if "mktcap" in df.columns:
        mc = pd.to_numeric(df["mktcap"], errors="coerce")
        out["raw_mktcap"] = mc
        log_mc = mc.apply(lambda x: np.log(x) if isinstance(x, float) and x > 0 else np.nan)
        out["size_log_mktcap"] = log_mc
        out["z_size"] = _cross_sectional_z(log_mc)

        def _tier(x):
            if pd.isna(x):
                return None
            if x < 300e6:   return 0   # micro
            if x < 2e9:     return 1   # small
            if x < 10e9:    return 2   # mid
            if x < 200e9:   return 3   # large
            return 4                   # mega

        out["size_tier"] = mc.apply(_tier)

    if "beta" in df.columns:
        out["raw_beta"] = pd.to_numeric(df["beta"], errors="coerce")
        out["z_beta"] = _cross_sectional_z(out["raw_beta"])
        # Low-vol anomaly: beta < 1 = defensively positioned
        out["low_beta_flag"] = (out["raw_beta"] < 1.0).astype(int)

    # ── 4. Piotroski F-score ──────────────────────────────────────────────────
    fscores = []
    for ticker in df.index:
        row = df.loc[ticker]
        fs = _piotroski_fscore(row)
        fscores.append(fs)
    fscore_series = pd.Series(fscores, index=df.index)
    out["fscore"] = fscore_series
    out["z_fscore"] = _cross_sectional_z(fscore_series)
    out["fscore_high_quality"] = (fscore_series >= 7).astype(int)
    out["fscore_low_quality"] = (fscore_series <= 3).astype(int)

    # ── 5. Earnings yield vs total shareholder yield ──────────────────────────
    if "earningsYield" in df.columns and "dividendYield" in df.columns:
        ey = pd.to_numeric(df["earningsYield"], errors="coerce")
        dy = pd.to_numeric(df["dividendYield"], errors="coerce")
        out["total_yield"] = (ey.fillna(0) + dy.fillna(0)).clip(-1, 2)
        out["z_total_yield"] = _cross_sectional_z(out["total_yield"])

    # FCF yield as quality-growth bridge (Buffett-style "earnings power")
    if "freeCashFlowYield" in df.columns and "revenueGrowth" in df.columns:
        fcfy = pd.to_numeric(df["freeCashFlowYield"], errors="coerce")
        rg = pd.to_numeric(df["revenueGrowth"], errors="coerce")
        out["garp_score"] = _cross_sectional_z(fcfy) + _cross_sectional_z(rg)

    # ── 6. QMJ (Quality-Minus-Junk) composite ────────────────────────────────
    qmj_components = []
    for g in ["quality", "growth", "efficiency"]:
        if f"{g}_composite" in out.columns:
            qmj_components.append(out[f"{g}_composite"])
    if qmj_components:
        out["qmj_score"] = pd.concat(qmj_components, axis=1).mean(axis=1)

    # Junk score: high debt + low quality
    junk_components = []
    if "z_debtToEquity" in out.columns:
        junk_components.append(-out["z_debtToEquity"])
    if "quality_composite" in out.columns:
        junk_components.append(-out["quality_composite"])
    if junk_components:
        out["junk_score"] = pd.concat(junk_components, axis=1).mean(axis=1)

    # ── 7. Insider activity ───────────────────────────────────────────────────
    for col in ["insider_buy_count", "insider_sell_count", "insider_net_bias",
                "insider_buy_pct"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")

    if "insider_net_bias" in df.columns:
        bias = pd.to_numeric(df["insider_net_bias"], errors="coerce")
        out["z_insider_net_bias"] = _cross_sectional_z(bias)
        out["insider_bullish"] = (bias > 0).astype(int)
        out["insider_bearish"] = (bias < 0).astype(int)

    if "insider_buy_pct" in df.columns:
        bp = pd.to_numeric(df["insider_buy_pct"], errors="coerce")
        out["z_insider_buy_pct"] = _cross_sectional_z(bp)

    # ── 8. ETF flag ────────────────────────────────────────────────────────────
    if "is_etf" in df.columns:
        out["is_etf"] = pd.to_numeric(df["is_etf"], errors="coerce").fillna(0)

    # ── 9. Composite factor combos ────────────────────────────────────────────
    if "raw_beta" in out.columns and "value_composite" in out.columns:
        out["bab_value"] = out["value_composite"] - out["z_beta"].fillna(0)

    if "z_returnOnEquity_qoq" in out.columns and "quality_composite" in out.columns:
        out["fundamental_momentum"] = (out["quality_composite"] +
                                       out["z_returnOnEquity_qoq"].fillna(0))

    # ── 10. Greenblatt Magic Formula components ────────────────────────────────
    if "earningsYield" in df.columns and "returnOnInvestedCapital" in df.columns:
        ey = pd.to_numeric(df["earningsYield"], errors="coerce")
        roic = pd.to_numeric(df["returnOnInvestedCapital"], errors="coerce")
        out["magic_formula_ey"] = _cross_sectional_z(ey)
        out["magic_formula_roic"] = _cross_sectional_z(roic)
        out["magic_formula_score"] = out["magic_formula_ey"] + out["magic_formula_roic"]
        out["z_magic_formula"] = _cross_sectional_z(out["magic_formula_score"])

    # ── 11. Altman Z-score proxy ───────────────────────────────────────────────
    if "currentRatio" in df.columns and "debtRatio" in df.columns:
        cr = pd.to_numeric(df["currentRatio"], errors="coerce")
        dr = pd.to_numeric(df["debtRatio"], errors="coerce")
        roe_col = df.get("returnOnEquity", pd.Series(0.0, index=df.index))
        roe = pd.to_numeric(roe_col, errors="coerce").fillna(0)
        altman_proxy = 1.2 * (cr - 1).clip(-1, 2) + 1.4 * roe - 3.3 * dr.fillna(0.5)
        out["altman_z_proxy"] = altman_proxy
        out["z_altman_z"] = _cross_sectional_z(altman_proxy)
        out["distress_flag"] = (altman_proxy < -2.0).astype(int)
        out["safe_zone_flag"] = (altman_proxy > 2.6).astype(int)

    # ── 12. Profitability trend factors ───────────────────────────────────────
    for raw_col, trend_name in [
        ("returnOnEquity", "roe_trend"), ("grossProfitMargin", "gpm_trend"),
        ("netProfitMargin", "npm_trend"), ("revenueGrowth", "rev_growth_trend"),
    ]:
        qoq_col = f"z_{raw_col}_qoq"
        yoy_col = f"z_{raw_col}_yoy"
        if qoq_col in out.columns and yoy_col in out.columns:
            out[trend_name] = out[qoq_col].fillna(0) + out[yoy_col].fillna(0)
        elif qoq_col in out.columns:
            out[trend_name] = out[qoq_col]

    # ── 13. Combo alpha factors ────────────────────────────────────────────────
    if "value_composite" in out.columns and "quality_composite" in out.columns:
        out["value_quality_combo"] = out["value_composite"] + out["quality_composite"]
        if "growth_composite" in out.columns:
            out["garp_quality"] = out["quality_composite"] + out["growth_composite"]
    if "quality_composite" in out.columns and "z_fscore" in out.columns:
        out["hq_signal"] = out["quality_composite"] * out["z_fscore"].fillna(0)
    if "value_composite" in out.columns and "fscore" in out.columns:
        out["value_with_quality_gate"] = out["value_composite"] * (out["fscore"] >= 5).astype(float)

    # ── 14. Sector classification ──────────────────────────────────────────────
    if "sector" in df.columns:
        sector_map = {
            "Technology": 1, "Communication Services": 2, "Consumer Cyclical": 3,
            "Financial Services": 4, "Healthcare": 5, "Energy": 6,
            "Industrials": 7, "Consumer Defensive": 8, "Utilities": 9,
            "Real Estate": 10, "Basic Materials": 11,
        }
        out["sector_code"] = df["sector"].map(sector_map).fillna(0).astype(float)
        out["is_tech_sector"] = (df["sector"].isin(
            ["Technology", "Communication Services"])).astype(float)
        out["is_cyclical"] = (df["sector"].isin(
            ["Consumer Cyclical", "Financial Services", "Industrials", "Energy"])).astype(float)
        out["is_defensive"] = (df["sector"].isin(
            ["Healthcare", "Consumer Defensive", "Utilities", "Real Estate"])).astype(float)

    out["ticker"] = out.index
    return out.reset_index(drop=True)
