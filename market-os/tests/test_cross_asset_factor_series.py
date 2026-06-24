"""Regression guard for the India factor-panel bug: `build_cross_asset_features` used to
hardcode its factor-column filter against US-only names (FACTOR_SERIES), so a
differently-named factor panel (e.g. India's NIFTY50/BANKNIFTY/INDIA_VIX) silently
produced ZERO cross-asset features — no exception, just an empty result. Fixed via an
optional `factor_series` override; these tests guard both the default (US, unaffected)
and override (any other naming) paths.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from marketos.features.cross_asset import build_cross_asset_features, FACTOR_SERIES


def _synthetic_returns(n: int = 300, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n, tz="UTC")
    return pd.Series(rng.normal(0.0005, 0.02, n), index=dates)


def test_default_factor_series_still_matches_us_names():
    """Backward-compat guard: existing US callers that don't pass factor_series must keep
    getting real (non-empty) cross-asset features."""
    asset_ret = _synthetic_returns(seed=1)
    factor_panel = pd.DataFrame({
        "SPY": _synthetic_returns(seed=2), "QQQ": _synthetic_returns(seed=3),
        "VIX": _synthetic_returns(seed=4),
    })
    xa = build_cross_asset_features(asset_ret, factor_panel)
    real_cols = [c for c in xa.columns if c not in ("asof_ts", "knowledge_ts")]
    assert len(real_cols) > 0, "default factor_series must still match US factor names"
    assert any("spy" in c for c in real_cols)


def test_non_us_factor_names_produce_zero_features_WITHOUT_override():
    """Documents the exact failure mode that shipped: a non-US-named panel with no
    factor_series override silently produces nothing — confirms the bug is real, not
    fixed by accident elsewhere."""
    asset_ret = _synthetic_returns(seed=1)
    factor_panel = pd.DataFrame({
        "NIFTY50": _synthetic_returns(seed=2), "BANKNIFTY": _synthetic_returns(seed=3),
    })
    xa = build_cross_asset_features(asset_ret, factor_panel)  # no factor_series override
    real_cols = [c for c in xa.columns if c not in ("asof_ts", "knowledge_ts")]
    assert real_cols == [], "without an override, non-US factor names must produce nothing"


def test_custom_factor_series_unlocks_non_us_names():
    """The actual fix: passing factor_series matching the panel's real column names
    produces real beta/corr features, exactly as the US default does for US names."""
    asset_ret = _synthetic_returns(seed=1)
    factor_panel = pd.DataFrame({
        "NIFTY50": _synthetic_returns(seed=2), "BANKNIFTY": _synthetic_returns(seed=3),
        "INDIA_VIX": _synthetic_returns(seed=5),
    })
    xa = build_cross_asset_features(asset_ret, factor_panel,
                                    factor_series=["NIFTY50", "BANKNIFTY", "INDIA_VIX"])
    real_cols = [c for c in xa.columns if c not in ("asof_ts", "knowledge_ts")]
    assert len(real_cols) > 0
    assert any("nifty50" in c for c in real_cols)
    assert any("banknifty" in c for c in real_cols)


def test_us_derived_tilts_gracefully_absent_for_india_naming():
    """SPY/VIX/DXY/OIL-literal derived composites (excess_ret vs SPY, risk_on_score, ...)
    are US-specific and should be ABSENT (not crash) when the factor panel uses India
    names — an honest, bounded gap, not a silent corruption."""
    asset_ret = _synthetic_returns(seed=1)
    factor_panel = pd.DataFrame({
        "NIFTY50": _synthetic_returns(seed=2), "BANKNIFTY": _synthetic_returns(seed=3),
    })
    xa = build_cross_asset_features(asset_ret, factor_panel,
                                    factor_series=["NIFTY50", "BANKNIFTY"])
    assert "risk_on_score" not in xa.columns
    assert "excess_ret_63d" not in xa.columns
