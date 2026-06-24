"""Cross-asset feature family — relationships between an instrument and the wider complex.

A single stock never moves in isolation. Its sensitivity (beta) to the market, the dollar,
rates, oil, gold, and crypto carries information that price-only features cannot. When those
betas *shift*, the regime is changing. We compute rolling correlations and OLS betas at
multiple windows against a reference panel of "factor" series.

All features are causal: rolling windows look strictly backward. The reference panel must be
aligned (same DatetimeIndex, forward-filled for non-trading-day gaps) before computation.

References: Sharpe 1964 (CAPM beta); Fama-French factor sensitivities; Ang 2014 (factor
risk premia); cross-asset correlation regimes (Longin-Solnik 2001).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Reference factor series we compute sensitivities against.
# Keys are the column names we expect in the aligned reference frame.
FACTOR_SERIES = [
    # Market
    "SPY", "QQQ",
    # Macro / rates
    "VIX", "DXY", "US10Y",
    # Commodities / alternatives
    "OIL", "GOLD", "BTC",
    # Sector ETFs (rotation signals)
    "XLK", "XLF", "XLE", "XLRE", "XLV", "XLU", "XLI",
]
_BETA_WINDOWS = [5, 20, 60, 126, 252]


def _rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """OLS beta of y on x over a rolling window: cov(y,x)/var(x)."""
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var.replace(0, np.nan)


def build_cross_asset_features(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
    *,
    knowledge_lag: str = "0min",
    factor_series: list[str] | None = None,
) -> pd.DataFrame:
    """Compute cross-asset sensitivities for one instrument.

    asset_returns : daily returns of the target instrument (indexed by timestamp).
    factor_returns: DataFrame of daily returns for the factor panel; columns are a subset
                    of `factor_series`. Index must overlap asset_returns.
    factor_series : which factor_returns columns to compute beta/corr against. Defaults to
                    module-level `FACTOR_SERIES` (the US SPY/QQQ/VIX/... names) so existing
                    US callers are unaffected. MUST be overridden for a differently-named
                    factor panel (e.g. India's NIFTY50/BANKNIFTY/INDIA_VIX) — this was
                    hardcoded to US names until caught testing the India factor panel:
                    every per-factor beta/corr silently computed ZERO columns (not an
                    error, just an empty `available` list) because none of India's factor
                    names matched the US-only hardcoded list.
    Returns a frame with rolling correlation + beta to each available factor, plus
    derived spreads (e.g. risk-on/risk-off tilt — US-factor-name-specific, see below),
    with point-in-time columns.
    """
    idx = asset_returns.index
    factors = factor_returns.reindex(idx).ffill()
    _c: dict[str, pd.Series] = {}  # accumulate into dict, build DataFrame once at the end

    series = factor_series if factor_series is not None else FACTOR_SERIES
    available = [c for c in series if c in factors.columns]
    for fac in available:
        fr = factors[fac]
        for w in _BETA_WINDOWS:
            _c[f"corr_{fac.lower()}_{w}"] = asset_returns.rolling(w).corr(fr)
            _c[f"beta_{fac.lower()}_{w}"] = _rolling_beta(asset_returns, fr, w)
        # beta drift: short-window beta minus long-window beta
        _c[f"beta_{fac.lower()}_drift"] = (
            _c[f"beta_{fac.lower()}_20"] - _c[f"beta_{fac.lower()}_126"]
        )

    # ── Derived cross-asset tilts ──────────────────────────────────────────────
    if "VIX" in factors.columns and "SPY" in factors.columns:
        spy_beta60 = _c.get("beta_spy_60", pd.Series(0.0, index=idx))
        spy_resid = asset_returns - spy_beta60 * factors["SPY"]
        _c["idio_ret"] = spy_resid
        _c["idio_vol_20"] = spy_resid.rolling(20).std() * np.sqrt(252)
        _c["idio_vol_63"] = spy_resid.rolling(63).std() * np.sqrt(252)

    if "DXY" in factors.columns and "OIL" in factors.columns:
        _c["dollar_oil_sensitivity"] = (
            _c.get("beta_oil_60", pd.Series(np.nan, index=idx)) -
            _c.get("beta_dxy_60", pd.Series(np.nan, index=idx))
        )
    if "US10Y" in factors.columns:
        _c["rate_beta_60"] = _c.get("beta_us10y_60", pd.Series(np.nan, index=idx))
        if "beta_us10y_20" in _c and "beta_us10y_126" in _c:
            _c["rate_beta_drift"] = _c["beta_us10y_20"] - _c["beta_us10y_126"]

    # ── Rolling volatility correlation (regime alignment) ─────────────────────
    asset_rv20 = asset_returns.rolling(20).std() * np.sqrt(252)
    for fac in available:
        fac_rv20 = factors[fac].rolling(20).std() * np.sqrt(252)
        for w in [63, 126]:
            _c[f"volcorr_{fac.lower()}_{w}"] = asset_rv20.rolling(w).corr(fac_rv20)

    # ── Relative performance (active return vs SPY) ────────────────────────────
    if "SPY" in factors.columns:
        excess = asset_returns - factors["SPY"]
        excess_ret_63 = excess.rolling(63).sum() * 252 / 63
        te_63 = excess.rolling(63).std() * np.sqrt(252)
        _c["excess_ret_20d"] = excess.rolling(20).sum() * 252 / 20
        _c["excess_ret_63d"] = excess_ret_63
        _c["tracking_error_63"] = te_63
        _c["information_ratio_63"] = excess_ret_63 / te_63.replace(0, np.nan)

    # ── Regime positioning composites ──────────────────────────────────────────
    if "beta_spy_60" in _c and "beta_vix_60" in _c:
        _c["defensive_score"] = -_c["beta_spy_60"] + (-_c["beta_vix_60"])

    riskons = [_c[f] for f in ["beta_spy_60", "beta_qqq_60", "beta_btc_60"] if f in _c]
    if riskons:
        _c["risk_on_score"] = pd.concat(riskons, axis=1).mean(axis=1)

    if "corr_gold_60" in _c:
        _c["gold_equity_corr_63"] = _c["corr_gold_60"]
    if "corr_btc_60" in _c:
        _c["btc_equity_corr_63"] = _c["corr_btc_60"]

    _c["asof_ts"] = idx
    _c["knowledge_ts"] = idx + pd.Timedelta(knowledge_lag)
    return pd.DataFrame(_c, index=idx)


def build_factor_return_panel(price_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble a daily-return panel for the factor complex from raw price frames.

    price_frames maps a FACTOR_SERIES key → an OHLCV (or single-close) DataFrame.
    Missing factors are simply omitted; downstream code degrades gracefully.
    """
    cols = {}
    for key, frame in price_frames.items():
        if frame is None or frame.empty:
            continue
        close_col = "close" if "close" in frame.columns else frame.columns[0]
        cols[key] = frame[close_col].pct_change()
    if not cols:
        return pd.DataFrame()
    panel = pd.DataFrame(cols).sort_index()
    return panel
