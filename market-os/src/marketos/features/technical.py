"""Technical feature family — broad, literature-grounded, fully causal.

Every feature uses only data available up to and including the bar it is attributed to
(.shift / .rolling / .ewm are all backward-looking). We carry `asof_ts` and `knowledge_ts`
so the no-lookahead guard can verify causality downstream. Indicators use simple, auditable
math (no hidden TA-Lib state) so results reproduce on any machine.

Feature groups (≈200 columns):
  returns/momentum, moving-average structure, oscillators, volatility estimators,
  distributional moments, drawdown, range/channel position, trend strength (DMI/ADX/Vortex/
  Aroon), volume/flow, illiquidity, candle microstructure, streaks, persistence (Hurst/AC).

References: Wilder 1978 (RSI/ATR/ADX); Bollinger; Keltner; Donchian; Parkinson 1980;
Garman-Klass 1980; Rogers-Satchell 1991; Yang-Zhang 2000; Amihud 2002 (illiquidity);
Jegadeesh-Titman 1993 (momentum); Hurst 1951 (persistence).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Horizon banks reused across groups
_RET_HORIZONS = [1, 2, 3, 5, 10, 20, 40, 63, 126, 252]
_MA_SPANS = [5, 10, 20, 50, 100, 200]
_RSI_PERIODS = [2, 7, 14, 21]
_VOL_WINDOWS = [5, 10, 20, 60, 120]
_CHANNEL_WINDOWS = [10, 20, 55]


# ── primitive indicators ──────────────────────────────────────────────────────
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def _dmi(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (+DI, -DI, ADX)."""
    high, low = df["high"], df["low"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = _atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return plus_di, minus_di, adx


def _stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def _williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    return -100 * (high_max - df["close"]) / (high_max - low_min).replace(0, np.nan)


def _cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def _mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    rmf = tp * df["volume"]
    pos = rmf.where(tp > tp.shift(1), 0.0).rolling(period).sum()
    neg = rmf.where(tp < tp.shift(1), 0.0).rolling(period).sum()
    mfr = pos / neg.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def _aroon(df: pd.DataFrame, period: int = 25) -> tuple[pd.Series, pd.Series]:
    roll_high = df["high"].rolling(period + 1)
    roll_low = df["low"].rolling(period + 1)
    up = roll_high.apply(lambda x: (period - x.argmax()) / period * 100, raw=True)
    down = roll_low.apply(lambda x: (period - x.argmin()) / period * 100, raw=True)
    return up, down


def _vortex(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series]:
    tr = _true_range(df)
    vm_plus = (df["high"] - df["low"].shift(1)).abs()
    vm_minus = (df["low"] - df["high"].shift(1)).abs()
    tr_sum = tr.rolling(period).sum().replace(0, np.nan)
    return vm_plus.rolling(period).sum() / tr_sum, vm_minus.rolling(period).sum() / tr_sum


def _obv(close: pd.Series, vol: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * vol).cumsum()


def _chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.Series:
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    mfv = mfm * df["volume"]
    return mfv.rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)


def _hurst(ts: np.ndarray) -> float:
    """Rescaled-range Hurst exponent for a short window. 0.5=random, >0.5=trending."""
    ts = ts[~np.isnan(ts)]
    n = len(ts)
    if n < 20:
        return np.nan
    lags = range(2, min(20, n // 2))
    tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
    tau = np.array(tau)
    valid = tau > 0
    if valid.sum() < 3:
        return np.nan
    poly = np.polyfit(np.log(np.array(list(lags))[valid]), np.log(tau[valid]), 1)
    return poly[0]


def _parkinson_vol(df: pd.DataFrame, window: int) -> pd.Series:
    """Parkinson 1980 high-low range volatility estimator (annualized)."""
    hl = np.log(df["high"] / df["low"]) ** 2
    return np.sqrt(hl.rolling(window).mean() / (4 * np.log(2)) * 252)


def _garman_klass_vol(df: pd.DataFrame, window: int) -> pd.Series:
    """Garman-Klass 1980 OHLC volatility estimator (annualized)."""
    log_hl = np.log(df["high"] / df["low"]) ** 2
    log_co = np.log(df["close"] / df["open"]) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(gk.rolling(window).mean() * 252)


def _rogers_satchell_vol(df: pd.DataFrame, window: int) -> pd.Series:
    """Rogers-Satchell 1991 drift-independent volatility (annualized)."""
    log_ho = np.log(df["high"] / df["open"])
    log_lo = np.log(df["low"] / df["open"])
    log_co = np.log(df["close"] / df["open"])
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    return np.sqrt(rs.rolling(window).mean().clip(lower=0) * 252)


# ── main builder ───────────────────────────────────────────────────────────────
def build_technical_features(ohlcv: pd.DataFrame, *, knowledge_lag: str = "0min") -> pd.DataFrame:
    """Compute the full technical feature panel from an OHLCV frame.

    Required columns: open, high, low, close, volume. Index: timestamp.
    Returns a frame with `asof_ts` and `knowledge_ts` for point-in-time correctness.
    All ≈200 columns are causal; for daily bars values are known at the close.
    """
    df = ohlcv.copy().sort_index()
    o, h, l, close, vol = df["open"], df["high"], df["low"], df["close"], df["volume"]
    ret1 = close.pct_change()
    log_ret = np.log(close / close.shift(1))
    # Accumulate all series in a dict; concat once at the end to avoid DataFrame fragmentation
    _c: dict[str, pd.Series] = {}

    # ── 1. Returns & momentum across horizons ─────────────────────────────────
    for n in _RET_HORIZONS:
        _c[f"ret_{n}"] = close.pct_change(n)
        _c[f"logret_{n}"] = np.log(close / close.shift(n))
        _c[f"mom_{n}"] = close / close.shift(n) - 1
    # risk-adjusted momentum (Sharpe-like) at key horizons
    for n in [20, 63, 126, 252]:
        vol_n = ret1.rolling(n).std()
        _c[f"mom_riskadj_{n}"] = (close / close.shift(n) - 1) / (vol_n * np.sqrt(n)).replace(0, np.nan)
    # Jegadeesh-Titman 12-1 momentum (skip most recent month)
    _c["mom_12_1"] = close.shift(21) / close.shift(252) - 1
    # acceleration: change in 20d momentum
    _c["mom_accel_20"] = _c["mom_20"] - _c["mom_20"].shift(20)

    # ── 2. Moving-average structure ───────────────────────────────────────────
    for span in _MA_SPANS:
        sma = close.rolling(span).mean()
        ema = _ema(close, span)
        _c[f"sma_{span}_ratio"] = close / sma - 1
        _c[f"ema_{span}_ratio"] = close / ema - 1
        _c[f"sma_{span}_slope"] = sma.pct_change(5)
    # classic EMA pair + MACD family
    ema12, ema26 = _ema(close, 12), _ema(close, 26)
    _c["ema_ratio"] = ema12 / ema26 - 1
    macd = ema12 - ema26
    macd_signal = _ema(macd, 9)
    _c["macd"] = macd / close
    _c["macd_signal"] = macd_signal / close
    _c["macd_hist"] = (macd - macd_signal) / close
    # golden/death cross distance
    _c["sma_50_200_gap"] = close.rolling(50).mean() / close.rolling(200).mean() - 1
    # TRIX
    triple = _ema(_ema(_ema(close, 15), 15), 15)
    _c["trix_15"] = triple.pct_change()

    # ── 3. Oscillators ────────────────────────────────────────────────────────
    for p in _RSI_PERIODS:
        _c[f"rsi_{p}"] = _rsi(close, p)
    k, d = _stochastic(df, 14, 3)
    _c["stoch_k"] = k
    _c["stoch_d"] = d
    _c["stoch_diff"] = k - d
    _c["williams_r_14"] = _williams_r(df, 14)
    _c["cci_20"] = _cci(df, 20)
    _c["mfi_14"] = _mfi(df, 14)
    for n in [5, 10, 20]:
        _c[f"roc_{n}"] = close.pct_change(n) * 100

    # ── 4. Volatility estimators ──────────────────────────────────────────────
    for w in _VOL_WINDOWS:
        _c[f"realized_vol_{w}"] = ret1.rolling(w).std() * np.sqrt(252)
        _c[f"parkinson_vol_{w}"] = _parkinson_vol(df, w)
        _c[f"garman_klass_vol_{w}"] = _garman_klass_vol(df, w)
        _c[f"rogers_satchell_vol_{w}"] = _rogers_satchell_vol(df, w)
    # vol-of-vol and term structure
    _c["vol_of_vol_20"] = ret1.rolling(20).std().rolling(20).std() * np.sqrt(252)
    _c["vol_ratio_5_60"] = _c["realized_vol_5"] / _c["realized_vol_60"].replace(0, np.nan)
    _c["vol_ratio_20_120"] = _c["realized_vol_20"] / _c["realized_vol_120"].replace(0, np.nan)
    # ATR family
    for p in [7, 14, 21]:
        atr = _atr(df, p)
        _c[f"atr_{p}"] = atr
        _c[f"atr_pct_{p}"] = atr / close
    # downside / semivariance
    neg_ret = ret1.clip(upper=0)
    _c["downside_dev_20"] = neg_ret.rolling(20).std() * np.sqrt(252)
    _c["semivar_ratio_20"] = (neg_ret.rolling(20).std() /
                               ret1.clip(lower=0).rolling(20).std().replace(0, np.nan))

    # ── 5. Distributional moments ─────────────────────────────────────────────
    for w in [20, 60]:
        _c[f"ret_skew_{w}"] = ret1.rolling(w).skew()
        _c[f"ret_kurt_{w}"] = ret1.rolling(w).kurt()
    _c["zscore_20"] = (close - close.rolling(20).mean()) / close.rolling(20).std().replace(0, np.nan)
    _c["zscore_60"] = (close - close.rolling(60).mean()) / close.rolling(60).std().replace(0, np.nan)

    # ── 6. Drawdown & persistence ─────────────────────────────────────────────
    roll_max = close.rolling(252, min_periods=20).max()
    _c["drawdown_252"] = close / roll_max - 1
    _c["dist_52w_high"] = close / close.rolling(252, min_periods=20).max() - 1
    _c["dist_52w_low"] = close / close.rolling(252, min_periods=20).min() - 1
    _c["hurst_100"] = log_ret.rolling(100).apply(_hurst, raw=True)
    for lag in [1, 2, 5]:
        _c[f"autocorr_{lag}"] = ret1.rolling(60).apply(
            lambda x, L=lag: pd.Series(x).autocorr(L), raw=False)

    # ── 7. Range / channel position ───────────────────────────────────────────
    for w in _CHANNEL_WINDOWS:
        dc_high = h.rolling(w).max()
        dc_low = l.rolling(w).min()
        _c[f"donchian_pos_{w}"] = (close - dc_low) / (dc_high - dc_low).replace(0, np.nan)
    # Bollinger
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    _c["bb_pctb"] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)
    _c["bb_bandwidth"] = (4 * bb_std) / bb_mid.replace(0, np.nan)
    # Keltner
    kc_atr = _atr(df, 20)
    _c["keltner_pos"] = (close - bb_mid) / (2 * kc_atr).replace(0, np.nan)

    # ── 8. Trend strength ─────────────────────────────────────────────────────
    for p in [14, 28]:
        plus_di, minus_di, adx = _dmi(df, p)
        _c[f"adx_{p}"] = adx
        _c[f"di_diff_{p}"] = plus_di - minus_di
    vi_plus, vi_minus = _vortex(df, 14)
    _c["vortex_diff_14"] = vi_plus - vi_minus
    aroon_up, aroon_down = _aroon(df, 25)
    _c["aroon_osc_25"] = aroon_up - aroon_down

    # ── 9. Volume & flow ──────────────────────────────────────────────────────
    _c["vol_expansion"] = vol / vol.rolling(20).mean()
    _c["vol_zscore_20"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std().replace(0, np.nan)
    _c["dollar_volume"] = (close * vol)
    _c["dollar_vol_zscore_20"] = ((close * vol - (close * vol).rolling(20).mean()) /
                                   (close * vol).rolling(20).std().replace(0, np.nan))
    obv = _obv(close, vol)
    _c["obv_slope_20"] = obv.diff(20) / vol.rolling(20).mean().replace(0, np.nan)
    _c["cmf_20"] = _chaikin_money_flow(df, 20)
    # VWAP distance (rolling proxy)
    typical = (h + l + close) / 3
    vwap_20 = (typical * vol).rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan)
    _c["vwap_dist_20"] = close / vwap_20 - 1
    # Force index & ease of movement
    _c["force_index_13"] = (_ema(close.diff() * vol, 13)) / (close * vol.rolling(20).mean()).replace(0, np.nan)
    dist_moved = ((h + l) / 2 - (h.shift(1) + l.shift(1)) / 2)
    box_ratio = (vol / 1e8) / (h - l).replace(0, np.nan)
    _c["eom_14"] = (dist_moved / box_ratio.replace(0, np.nan)).rolling(14).mean()

    # ── 10. Illiquidity — multi-horizon (Amihud 2002) ─────────────────────────
    dv = (close * vol).replace(0, np.nan)  # dollar volume
    illiq_daily = ret1.abs() / dv
    for w in [5, 10, 20, 63, 126]:
        _c[f"amihud_illiq_{w}"] = illiq_daily.rolling(w).mean() * 1e9
    # Turnover ratio vs long-run mean
    for w in [5, 10, 20, 63]:
        _c[f"turnover_{w}"] = vol.rolling(w).mean() / vol.rolling(252).mean().replace(0, np.nan)
    # Roll (1984) bid-ask spread proxy: -2√(cov(ret_t, ret_{t-1})) for w-day windows
    for w in [20, 60]:
        cov = (ret1 * ret1.shift(1)).rolling(w).mean()
        roll_spread = (-2 * np.sqrt((-cov).clip(lower=0)))
        _c[f"roll_spread_{w}"] = roll_spread
    # Kyle (1985) lambda proxy: price impact per unit volume
    for w in [20, 60]:
        price_impact = (ret1.abs() / vol.replace(0, np.nan))
        _c[f"kyle_lambda_{w}"] = price_impact.rolling(w).mean() * 1e6
    # Illiquidity trend: ratio of short-term to long-term Amihud
    _c["illiq_trend_5_20"] = _c["amihud_illiq_5"] / _c["amihud_illiq_20"].replace(0, np.nan)
    _c["illiq_trend_20_63"] = _c["amihud_illiq_20"] / _c["amihud_illiq_63"].replace(0, np.nan)

    # ── 11. Candle microstructure ─────────────────────────────────────────────
    rng = (h - l).replace(0, np.nan)
    _c["candle_body"] = (close - o) / rng
    _c["upper_wick"] = (h - np.maximum(close, o)) / rng
    _c["lower_wick"] = (np.minimum(close, o) - l) / rng
    _c["gap"] = o / close.shift(1) - 1
    _c["true_range_pct"] = _true_range(df) / close
    _c["intraday_range"] = (h - l) / close
    _c["close_loc"] = (close - l) / rng  # where close sits in the day's range
    # Candle confirmation: body dominates → trend bar; wick dominates → indecision
    _c["body_wick_ratio"] = _c["candle_body"].abs() / (1 - _c["candle_body"].abs() + 1e-6)
    # Rolling candle pattern features
    for w in [5, 10, 20]:
        _c[f"avg_intraday_range_{w}"] = _c["intraday_range"].rolling(w).mean()
        _c[f"avg_body_{w}"] = _c["candle_body"].rolling(w).mean()
        _c[f"avg_gap_{w}"] = _c["gap"].rolling(w).mean()

    # ── 12. Streaks ───────────────────────────────────────────────────────────
    up_day = (ret1 > 0).astype(int)
    streak = up_day.groupby((up_day != up_day.shift()).cumsum()).cumcount() + 1
    _c["up_streak"] = streak.where(up_day == 1, 0)
    _c["down_streak"] = streak.where(up_day == 0, 0)
    for w in [5, 10, 20, 63]:
        _c[f"pct_up_days_{w}"] = up_day.rolling(w).mean()
    # large-move streaks
    big_up = (ret1 > ret1.rolling(252).std()).astype(int)
    big_dn = (ret1 < -ret1.rolling(252).std()).astype(int)
    _c["big_up_20"] = big_up.rolling(20).sum()
    _c["big_dn_20"] = big_dn.rolling(20).sum()

    # ── 13. Extended oscillators / price oscillators ──────────────────────────
    # DPO — Detrended Price Oscillator (removes long-term trend)
    for p in [14, 20]:
        shifted_ma = close.rolling(p // 2 + 1).mean().shift(p // 2 + 1)
        _c[f"dpo_{p}"] = close - shifted_ma
    # PPO — Percentage Price Oscillator (MACD normalized by slow EMA)
    _c["ppo_12_26"] = (ema12 - ema26) / ema26.replace(0, np.nan) * 100
    _c["ppo_signal_9"] = _ema(_c["ppo_12_26"], 9)
    _c["ppo_hist"] = _c["ppo_12_26"] - _c["ppo_signal_9"]
    # Elder Ray: Bull Power = High - EMA(13); Bear Power = Low - EMA(13)
    ema13 = _ema(close, 13)
    _c["bull_power_13"] = h - ema13
    _c["bear_power_13"] = l - ema13
    _c["elder_ray_ratio_13"] = _c["bull_power_13"] / (-_c["bear_power_13"]).replace(0, np.nan)
    # CMO — Chande Momentum Oscillator
    for p in [9, 14, 20]:
        su = ret1.clip(lower=0).rolling(p).sum()
        sd = (-ret1.clip(upper=0)).rolling(p).sum()
        _c[f"cmo_{p}"] = (su - sd) / (su + sd).replace(0, np.nan) * 100
    # KST — Know Sure Thing (rate-of-change smoothing composite)
    roc10 = close.pct_change(10)
    roc15 = close.pct_change(15)
    roc20 = close.pct_change(20)
    roc30 = close.pct_change(30)
    kst = (roc10.rolling(10).mean() * 1 + roc15.rolling(10).mean() * 2 +
           roc20.rolling(10).mean() * 3 + roc30.rolling(15).mean() * 4)
    _c["kst"] = kst
    _c["kst_signal"] = _ema(kst, 9)
    _c["kst_hist"] = kst - _c["kst_signal"]
    # DEMA / TEMA
    _c["dema_20_ratio"] = close / (2 * _ema(close, 20) - _ema(_ema(close, 20), 20)) - 1
    ema20 = _ema(close, 20)
    ema20_2 = _ema(ema20, 20)
    ema20_3 = _ema(ema20_2, 20)
    _c["tema_20_ratio"] = close / (3 * ema20 - 3 * ema20_2 + ema20_3).replace(0, np.nan) - 1
    # Hull MA (WMA(2·WMA(n/2) - WMA(n)), sqrt(n))
    def _wma(s: pd.Series, n: int) -> pd.Series:
        w = np.arange(1, n + 1)
        return s.rolling(n).apply(lambda x: (x * w[-len(x):]).sum() / w[-len(x):].sum(), raw=True)
    for n in [9, 16, 25]:
        sq = int(n ** 0.5)
        raw_hull = 2 * _wma(close, n // 2) - _wma(close, n)
        _c[f"hull_ma_{n}_ratio"] = close / _wma(raw_hull, sq).replace(0, np.nan) - 1

    # ── 14. Linear-regression based features ─────────────────────────────────
    for w in [20, 63, 126]:
        # slope of OLS regression of log-price on time (trend per bar)
        def _lr_slope(x):
            n_pts = len(x)
            if n_pts < 2:
                return np.nan
            t = np.arange(n_pts, dtype=float)
            A = np.vstack([t, np.ones(n_pts)]).T
            result = np.linalg.lstsq(A, x, rcond=None)
            return float(result[0][0])
        _c[f"lr_slope_{w}"] = np.log(close).rolling(w).apply(_lr_slope, raw=True)
        # R² of the linear fit (how "trend-like" vs noisy)
        def _lr_r2(x):
            n_pts = len(x)
            if n_pts < 2:
                return np.nan
            t = np.arange(n_pts, dtype=float)
            A = np.vstack([t, np.ones(n_pts)]).T
            result = np.linalg.lstsq(A, x, rcond=None)
            y_hat = A @ result[0]
            ss_res = ((x - y_hat) ** 2).sum()
            ss_tot = ((x - x.mean()) ** 2).sum()
            return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        _c[f"lr_r2_{w}"] = np.log(close).rolling(w).apply(_lr_r2, raw=True)
    # Mean reversion: deviation from linear regression channel
    for w in [20, 63]:
        _c[f"lr_dev_{w}"] = _c["zscore_20"] if w == 20 else _c["zscore_60"]

    # ── 15. Regime / state features ───────────────────────────────────────────
    # Trend consistency: fraction of 5d windows with positive return over past 63d
    for w_inner, w_outer in [(5, 63), (10, 126)]:
        _c[f"trend_consist_{w_inner}_{w_outer}"] = (
            close.pct_change(w_inner).rolling(w_outer).apply(
                lambda x: (x > 0).mean(), raw=True)
        )
    # Volatility regime z-score relative to own history
    rv20 = ret1.rolling(20).std() * np.sqrt(252)
    _c["vol_regime_zscore"] = (rv20 - rv20.rolling(252).mean()) / rv20.rolling(252).std().replace(0, np.nan)
    # Price acceleration (second derivative of price)
    _c["price_accel_5"] = close.pct_change(5) - close.pct_change(5).shift(5)
    _c["price_accel_20"] = close.pct_change(20) - close.pct_change(20).shift(20)

    # ── 16. Higher-moment / tail-risk features ────────────────────────────────
    # Realized skewness and kurtosis of returns at multiple windows
    for w in [20, 63, 126]:
        _c[f"skew_{w}"] = ret1.rolling(w).skew()
        _c[f"kurt_{w}"] = ret1.rolling(w).kurt()
    # Autocorrelation at lags 1, 5, 10, 21 (mean-reversion vs momentum signature)
    for lag in [1, 5, 10, 21]:
        _c[f"autocorr_lag{lag}"] = ret1.rolling(63).apply(
            lambda x: pd.Series(x).autocorr(lag=min(lag, len(x) - 1)), raw=False)
    # Value-at-Risk (5th percentile of rolling 63d returns)
    _c["var_95_63"] = ret1.rolling(63).quantile(0.05)
    _c["var_99_63"] = ret1.rolling(63).quantile(0.01)
    # Expected Shortfall / CVaR (mean of returns below VaR)
    def _cvar(x, q):
        threshold = np.quantile(x, q)
        tail = x[x <= threshold]
        return float(tail.mean()) if len(tail) > 0 else float(np.quantile(x, q))
    _c["cvar_95_63"] = ret1.rolling(63).apply(lambda x: _cvar(x, 0.05), raw=True)
    # Jump detection: return > 3σ counts in rolling windows
    sigma_63 = ret1.rolling(63).std()
    jumps = (ret1.abs() > 3 * sigma_63).astype(float)
    for w in [5, 20, 63]:
        _c[f"jump_count_{w}"] = jumps.rolling(w).sum()
    _c["jump_size_max_20"] = ret1.abs().rolling(20).max()  # largest single move
    # Tail ratio: magnitude of 95th vs 5th percentile (asymmetry)
    q95 = ret1.rolling(63).quantile(0.95)
    q05 = ret1.rolling(63).quantile(0.05)
    _c["tail_ratio_63"] = q95.abs() / q05.abs().replace(0, np.nan)
    _c["tail_asymmetry_63"] = q95 + q05  # positive = right-skewed (more big up days)

    # ── 17. Volatility surface proxies (from OHLC only) ──────────────────────
    # Yang-Zhang volatility estimator (most efficient OHLC estimator)
    if "open" in df.columns and "high" in df.columns and "low" in df.columns:
        o, h, l = df["open"], df["high"], df["low"]
        # Overnight: log(open/prev_close), Intraday: log(high/low)
        overnight = np.log(o / close.shift(1).replace(0, np.nan))
        intraday_hl = np.log(h / l.replace(0, np.nan))
        for w in [10, 21, 63]:
            k = 0.34 / (1.34 + (w + 1) / (w - 1))
            rs = intraday_hl.rolling(w).var()
            onight_var = overnight.rolling(w).var()
            cc_var = ret1.rolling(w).var()
            _c[f"yang_zhang_vol_{w}"] = np.sqrt((onight_var + k * cc_var + (1 - k) * rs) * 252)
    # Parkinson (high-low only) estimator
    if "high" in df.columns and "low" in df.columns:
        h, l = df["high"], df["low"]
        ln_hl = np.log(h / l.replace(0, np.nan))
        park = ln_hl ** 2 / (4 * np.log(2))
        for w in [10, 21, 63]:
            _c[f"parkinson_vol_{w}"] = np.sqrt(park.rolling(w).mean() * 252)
    # Volatility of volatility (vol-of-vol) — captures uncertainty about uncertainty
    rv21 = ret1.rolling(21).std() * np.sqrt(252)
    _c["vol_of_vol_63"] = rv21.rolling(63).std()
    _c["vol_of_vol_21"] = rv21.rolling(21).std()

    # ── 18. Return distribution features ─────────────────────────────────────
    # Gain/pain ratio: sum of gains / |sum of losses|
    def _gain_pain(x):
        gains = x[x > 0].sum()
        pains = abs(x[x < 0].sum())
        return gains / pains if pains > 0 else np.nan
    for w in [20, 63]:
        _c[f"gain_pain_{w}"] = ret1.rolling(w).apply(_gain_pain, raw=True)
    # Ulcer index (depth and duration of drawdowns)
    def _ulcer(x):
        # Percentage drawdown from rolling maximum
        roll_max = np.maximum.accumulate(x)
        dd = (x - roll_max) / roll_max
        return np.sqrt((dd ** 2).mean())
    for w in [14, 63]:
        _c[f"ulcer_{w}"] = close.rolling(w).apply(_ulcer, raw=True)
    # Martin ratio (return / ulcer) at 63d
    ret63_ann = close.pct_change(63) * (252 / 63)
    _c["martin_ratio_63"] = ret63_ann / _c["ulcer_63"].replace(0, np.nan)

    # ── 19. Momentum factor variants ──────────────────────────────────────────
    # mom_12_1 already computed in section 1 (Jegadeesh-Titman 12-1, skip most recent month).
    # 6-1 month momentum, same skip-month construction (price ratio, not pct_change ratio —
    # the latter divides by a near-zero pct_change(21) on flat patches and produces inf).
    _c["mom_6_1"] = close.shift(21) / close.shift(126) - 1
    # Intermediate momentum reversal (1-month)
    _c["mom_1_0"] = close.pct_change(21)
    # Short-term reversal (5 days)
    _c["reversal_5d"] = -close.pct_change(5)
    _c["reversal_10d"] = -close.pct_change(10)
    # Momentum quality: consistency of returns over 12m (Sharpe-like)
    _c["mom_quality_12m"] = (close.pct_change().rolling(252).mean() /
                              close.pct_change().rolling(252).std().replace(0, np.nan))
    # Momentum acceleration (12m mom trend)
    _c["mom_accel_12m"] = close.pct_change(252) - close.pct_change(252).shift(21)

    # ── 20. Price level / support-resistance features ─────────────────────────
    # Distance from 52-week high/low
    _c["dist_52w_high"] = close / close.rolling(252).max().replace(0, np.nan) - 1
    _c["dist_52w_low"] = close / close.rolling(252).min().replace(0, np.nan) - 1
    _c["dist_13w_high"] = close / close.rolling(63).max().replace(0, np.nan) - 1
    _c["dist_13w_low"] = close / close.rolling(63).min().replace(0, np.nan) - 1
    # Position within 52-week range (0=at 52w low, 1=at 52w high)
    w52_range = (close.rolling(252).max() - close.rolling(252).min()).replace(0, np.nan)
    _c["pos_in_52w_range"] = (close - close.rolling(252).min()) / w52_range
    w13_range = (close.rolling(63).max() - close.rolling(63).min()).replace(0, np.nan)
    _c["pos_in_13w_range"] = (close - close.rolling(63).min()) / w13_range
    # Breakout detection: close above rolling max (new high within window)
    _c["new_high_20"] = (close >= close.rolling(20).max().shift(1)).astype(float)
    _c["new_low_20"] = (close <= close.rolling(20).min().shift(1)).astype(float)
    _c["new_high_63"] = (close >= close.rolling(63).max().shift(1)).astype(float)
    _c["new_low_63"] = (close <= close.rolling(63).min().shift(1)).astype(float)
    # Days since 52-week high/low (how long since last extremum)
    def _days_since_max(x):
        idx = np.argmax(x)
        return float(len(x) - 1 - idx)
    _c["days_since_52w_high"] = close.rolling(252).apply(_days_since_max, raw=True)
    def _days_since_min(x):
        idx = np.argmin(x)
        return float(len(x) - 1 - idx)
    _c["days_since_52w_low"] = close.rolling(252).apply(_days_since_min, raw=True)

    # ── 21. Time-series percentile rank (where feature stands in own history) ──
    # Rolling rank of current value within its own trailing distribution.
    # Converts absolute signal level into "how extreme is this historically?"
    # Uses 252d lookback (1 year) so rank=1.0 means all-time-high in past year.
    _TS_RANK_COLS = [
        "rsi_14", "rsi_21",
        "macd_hist",
        "roc_20", "roc_63", "roc_126",
        "volatility_21d", "volatility_63d",
        "adx_14",
        "volume_surge_20d", "volume_surge_63d",
        "amihud_illiq_20", "amihud_illiq_63",
        "atr_14_pct",
        "bb_pctb",
        "mom_12_1", "mom_6_1",
        "skew_20", "kurt_20",
        "autocorr_lag1", "autocorr_lag5",
        "var_95_63",
        "ulcer_63",
        "gain_pain_63",
        "lr_slope_20", "lr_slope_63",
        "pos_in_52w_range",
        "dist_52w_high",
        "price_accel_20",
        "vol_regime_zscore",
        "trend_consist_5_63",
    ]
    for col in _TS_RANK_COLS:
        if col in _c:
            s = _c[col]
            if isinstance(s, pd.Series):
                _c[f"tsrank_{col}"] = s.rolling(252).apply(
                    lambda x: float(np.searchsorted(np.sort(x[:-1]), x[-1])) / max(len(x) - 1, 1),
                    raw=True,
                )

    # ── 22. Volume-weighted price features ────────────────────────────────────
    if "volume" in df.columns:
        vol_s = df["volume"].replace(0, np.nan)
        # Volume-weighted average price over rolling windows
        for w in [5, 10, 20, 63]:
            pv = (close * vol_s).rolling(w).sum()
            tv = vol_s.rolling(w).sum()
            vwap_w = pv / tv.replace(0, np.nan)
            _c[f"vwap_{w}_ratio"] = close / vwap_w.replace(0, np.nan) - 1
        # On-Balance Volume momentum
        obv_delta = (np.sign(ret1) * vol_s).fillna(0)
        obv = obv_delta.cumsum()
        obv_norm = obv / vol_s.rolling(20).mean().replace(0, np.nan)
        _c["obv_momentum_20"] = obv_norm.pct_change(20)
        _c["obv_momentum_63"] = obv_norm.pct_change(63)
        # Volume price trend (VPT)
        vpt = (ret1 * vol_s).fillna(0).cumsum()
        vpt_norm = vpt / vol_s.rolling(20).mean().replace(0, np.nan)
        _c["vpt_slope_20"] = vpt_norm.pct_change(20)

    # ── 23. Intraday bar structure features ───────────────────────────────────
    if "open" in df.columns and "high" in df.columns and "low" in df.columns:
        o, h, l = df["open"], df["high"], df["low"]
        # Shadow analysis
        true_range = (h - l).replace(0, np.nan)
        upper_shadow = (h - close.clip(lower=o))
        lower_shadow = (close.clip(upper=o) - l)
        # Doji detection: very small body relative to range
        body = (close - o).abs()
        _c["doji_flag"] = (body / true_range.replace(0, np.nan) < 0.1).astype(float)
        _c["doji_count_10"] = _c["doji_flag"].rolling(10).sum()
        # Upper/lower shadow dominance (bearish vs bullish pressure)
        _c["upper_shadow_pct"] = upper_shadow / true_range
        _c["lower_shadow_pct"] = lower_shadow / true_range
        _c["shadow_ratio"] = upper_shadow / lower_shadow.replace(0, np.nan)
        _c["shadow_ratio_ma10"] = _c["shadow_ratio"].rolling(10).mean()
        # Hammers and shooting stars
        _c["hammer_flag"] = (
            (lower_shadow / true_range > 0.6) & (upper_shadow / true_range < 0.1)
        ).astype(float)
        _c["shooting_star_flag"] = (
            (upper_shadow / true_range > 0.6) & (lower_shadow / true_range < 0.1)
        ).astype(float)
        _c["hammer_count_20"] = _c["hammer_flag"].rolling(20).sum()
        _c["shooting_star_count_20"] = _c["shooting_star_flag"].rolling(20).sum()

    # ── 24. Extended oscillators & additional trend indicators ───────────────
    # Williams %R (fast overbought/oversold, inverse of Stochastic %K)
    for n in [14, 21, 63]:
        _c[f"williams_r_{n}"] = -100 * (h.rolling(n).max() - close) / (
            h.rolling(n).max() - l.rolling(n).min()).replace(0, np.nan)

    # TRIX — triple-smoothed EMA % change (momentum with noise filter)
    for n in [14, 21]:
        t1 = _ema(close, n)
        t2 = _ema(t1, n)
        t3 = _ema(t2, n)
        _c[f"trix_{n}"] = t3.pct_change() * 100

    # Aroon Oscillator (trend age vs peers)
    for n in [14, 25]:
        ar_up = h.rolling(n + 1).apply(lambda x: (np.argmax(x) / (len(x) - 1)) * 100, raw=True)
        ar_dn = l.rolling(n + 1).apply(lambda x: (np.argmin(x) / (len(x) - 1)) * 100, raw=True)
        _c[f"aroon_osc_{n}"] = ar_up - ar_dn
        _c[f"aroon_up_{n}"] = ar_up

    # Vortex Indicator (directional strength)
    true_range_s = pd.concat([h - l,
                               (h - close.shift(1)).abs(),
                               (l - close.shift(1)).abs()], axis=1).max(axis=1)
    for n in [14, 21]:
        vm_pos = (h - l.shift(1)).abs().rolling(n).sum()
        vm_neg = (l - h.shift(1)).abs().rolling(n).sum()
        tr_sum = true_range_s.rolling(n).sum().replace(0, np.nan)
        _c[f"vortex_vi_plus_{n}"] = vm_pos / tr_sum
        _c[f"vortex_vi_minus_{n}"] = vm_neg / tr_sum
        _c[f"vortex_diff_{n}"] = _c[f"vortex_vi_plus_{n}"] - _c[f"vortex_vi_minus_{n}"]

    # Elder Force Index (price momentum × volume)
    efi_raw = ret1 * vol.replace(0, np.nan)
    for n in [13, 50]:
        _c[f"elder_force_{n}"] = _ema(efi_raw, n)
        _c[f"elder_force_{n}_norm"] = _c[f"elder_force_{n}"] / (
            _c[f"elder_force_{n}"].abs().rolling(252).max().replace(0, np.nan))

    # Mass Index (range expansion — presages trend reversals)
    ema9_hl = _ema(h - l, 9)
    ema9_ema9_hl = _ema(ema9_hl, 9)
    mi_ratio = ema9_hl / ema9_ema9_hl.replace(0, np.nan)
    _c["mass_index_25"] = mi_ratio.rolling(25).sum()

    # Chande Momentum Oscillator (CMO) at additional window
    for n in [30]:
        up_s = ret1.clip(lower=0).rolling(n).sum()
        dn_s = (-ret1).clip(lower=0).rolling(n).sum()
        _c[f"cmo_{n}"] = 100 * (up_s - dn_s) / (up_s + dn_s).replace(0, np.nan)

    # Donchian Channel features (breakout channel)
    for n in [20, 55]:
        dc_up = h.rolling(n).max()
        dc_dn = l.rolling(n).min()
        dc_mid = (dc_up + dc_dn) / 2
        _c[f"dc_pct_rank_{n}"] = (close - dc_dn) / (dc_up - dc_dn).replace(0, np.nan)
        _c[f"dc_width_{n}"] = (dc_up - dc_dn) / dc_mid.replace(0, np.nan)
        _c[f"dc_mid_ratio_{n}"] = close / dc_mid.replace(0, np.nan) - 1

    # Keltner Channel (EMA-based volatility channel vs Bollinger)
    for n in [20, 55]:
        kc_mid = _ema(close, n)
        atr_n = true_range_s.ewm(span=n, adjust=False).mean()
        _c[f"kc_pct_{n}"] = (close - kc_mid) / (2 * atr_n).replace(0, np.nan)
        # Squeeze: BB bandwidth vs KC width (low vol → upcoming expansion)
        if n == 20 and "bb_bandwidth" in _c:
            # bb_bandwidth = 4σ / mid; kc_width = 4×ATR / mid (approx)
            kc_width_norm = (4 * atr_n) / kc_mid.replace(0, np.nan)
            _c[f"squeeze_{n}"] = (_c["bb_bandwidth"] < kc_width_norm).astype(float)

    # ── 25. Z-score of close relative to rolling mean (mean-reversion distance) ──
    for n in [10, 20, 63, 126, 252]:
        rm = close.rolling(n).mean()
        rs = close.rolling(n).std()
        _c[f"zscore_close_{n}"] = (close - rm) / rs.replace(0, np.nan)

    # ── 26. Efficiency Ratio (Kaufman — measures trendiness 0=choppy, 1=trending) ─
    for n in [10, 21, 63]:
        net_move = (close - close.shift(n)).abs()
        path = ret1.abs().rolling(n).sum()
        _c[f"efficiency_ratio_{n}"] = net_move / path.replace(0, np.nan)

    # ── 27. Money Flow indicators ─────────────────────────────────────────────
    tp = (h + l + close) / 3   # typical price
    rmf = tp * vol              # raw money flow
    for n in [14, 21]:
        pos_mf = rmf.where(tp > tp.shift(1), 0.0).rolling(n).sum()
        neg_mf = rmf.where(tp <= tp.shift(1), 0.0).rolling(n).sum()
        mfi_ratio = pos_mf / neg_mf.replace(0, np.nan)
        _c[f"mfi_{n}"] = 100 - 100 / (1 + mfi_ratio)
    # Chaikin Money Flow
    clv = ((close - l) - (h - close)) / (h - l).replace(0, np.nan)
    for n in [20, 63]:
        _c[f"cmf_{n}"] = (clv * vol).rolling(n).sum() / vol.rolling(n).sum().replace(0, np.nan)

    # ── 28. Detrended Price Oscillator (removes dominant cycle, reveals shorter) ──
    for n in [14, 21, 63]:
        shift_n = n // 2 + 1
        _c[f"dpo_{n}"] = (close.shift(shift_n) - close.rolling(n).mean()) / close.replace(0, np.nan)

    # ── 29. Range statistics ────────────────────────────────────────────────────
    # High-low range as % of close midpoint (volatility proxy from range only)
    for n in [5, 20, 63, 126, 252]:
        _c[f"hl_ratio_{n}"] = (h.rolling(n).max() / l.rolling(n).min().replace(0, np.nan)) - 1
    # Average daily bar range as % of close
    daily_range_pct = (h - l) / close.replace(0, np.nan)
    for n in [5, 20, 63]:
        _c[f"avg_range_pct_{n}"] = daily_range_pct.rolling(n).mean()
    # Choppiness Index (log-ratio of ATR sum to total range; ~61.8 = choppy, ~38.2 = trending)
    for n in [14, 21]:
        atr_sum_n = true_range_s.rolling(n).sum()
        range_n = h.rolling(n).max() - l.rolling(n).min()
        _c[f"choppiness_{n}"] = 100 * np.log10(
            atr_sum_n / range_n.replace(0, np.nan)) / np.log10(n)
    # Range contraction ratio (short window range / long window range)
    _c["range_contraction_5_20"] = _c["avg_range_pct_5"] / _c["avg_range_pct_20"].replace(0, np.nan)
    _c["range_contraction_20_63"] = _c["avg_range_pct_20"] / _c["avg_range_pct_63"].replace(0, np.nan)

    # ── 30. Ultimate Oscillator (3-timeframe momentum composite) ─────────────────
    buy_pressure = close - pd.concat([l, close.shift(1)], axis=1).min(axis=1)
    for n1, n2, n3 in [(7, 14, 28)]:
        avg1 = buy_pressure.rolling(n1).sum() / true_range_s.rolling(n1).sum().replace(0, np.nan)
        avg2 = buy_pressure.rolling(n2).sum() / true_range_s.rolling(n2).sum().replace(0, np.nan)
        avg3 = buy_pressure.rolling(n3).sum() / true_range_s.rolling(n3).sum().replace(0, np.nan)
        _c["ultimate_osc"] = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

    # ── 31. Dollar volume metrics ─────────────────────────────────────────────
    dollar_vol = close * vol
    for n in [5, 20, 63]:
        dv_ma = dollar_vol.rolling(n).mean()
        dv_std = dollar_vol.rolling(n).std()
        _c[f"dollar_vol_ma_{n}"] = dv_ma
        _c[f"dollar_vol_zscore_{n}"] = (dollar_vol - dv_ma) / dv_std.replace(0, np.nan)

    # ── 32. Variance ratio test proxy (Lo-MacKinlay 1988) ────────────────────────
    # VR(q) = Var(q-period returns) / (q * Var(1-period returns)); >1=momentum, <1=reversion
    var_1 = ret1.rolling(252).var().replace(0, np.nan)
    for q in [4, 8, 16]:
        ret_q = close.pct_change(q)
        var_q = ret_q.rolling(252).var()
        _c[f"variance_ratio_{q}"] = var_q / (q * var_1)

    # ── 33. Additional time-series percentile ranks ───────────────────────────────
    _EXTRA_TSRANK = {
        "zscore_close_10": 252, "zscore_close_20": 252,
        "zscore_close_63": 252, "zscore_close_126": 252,
        "efficiency_ratio_10": 252, "efficiency_ratio_21": 252, "efficiency_ratio_63": 252,
        "mfi_14": 252, "mfi_21": 252,
        "cmf_20": 252, "cmf_63": 252,
        "choppiness_14": 252, "choppiness_21": 252,
        "hl_ratio_5": 252, "hl_ratio_20": 252, "hl_ratio_63": 252, "hl_ratio_126": 252,
        "dollar_vol_zscore_20": 252, "dollar_vol_zscore_63": 252,
        "ultimate_osc": 252,
        "variance_ratio_4": 252, "variance_ratio_8": 252, "variance_ratio_16": 252,
        "dpo_14": 252, "dpo_21": 252,
        "range_contraction_5_20": 252, "range_contraction_20_63": 252,
        "avg_range_pct_5": 252, "avg_range_pct_20": 252,
    }
    for col, w in _EXTRA_TSRANK.items():
        if col in _c:
            s = _c[col]
            if isinstance(s, pd.Series):
                _c[f"tsrank_{col}"] = s.rolling(w).rank(pct=True)

    _c["asof_ts"] = df.index
    _c["knowledge_ts"] = df.index + pd.Timedelta(knowledge_lag)
    return pd.DataFrame(_c, index=df.index)
