"""Phase 2 — the aggressive-sleeve REALITY-CHECK backtest (the GATE).

Before the aggressive ("burn") sleeve gets a rupee of real capital, it must answer one
question honestly: does a regime/trend throttle on a 3x daily-reset ETF (SOXL) and on
high-beta names actually AVOID the catastrophic drawdowns those instruments produce, while
still capturing enough of the secular wave to be worth it?

The brutal benchmark is SOXL buy-and-hold, which fell ~-90% peak-to-trough in 2022. If the
throttle can't dodge that, the aggressive thesis is dead on arrival regardless of any AI
narrative — that is the whole point of running this before building execution.

HONESTY FEATURES (deliberately, to avoid flattering ourselves):
  * Causal signals only. The trend (50/200 SMA) and the vol-regime (trailing realized vol
    vs its OWN expanding distribution) use information available up to day t. We do NOT use
    `regimes.hmm.detect_regimes` here — it fits the HMM on the full series, so its label at
    day t is informed by days after t. That look-ahead is acceptable for conditional
    research (its existing use) but would leak the future into a tradeable signal.
  * Next-bar execution + costs come from `backtest.engine` (weights are shifted one bar).
  * Reported headline Sharpe is on DAILY portfolio returns (periods_per_year=252) — the
    honest portfolio-level number, not the inflated trade-level one.
  * Every window reports the throttled strategy NEXT TO buy-and-hold so the comparison is
    explicit; "beats buy-and-hold" is the bar, not "made money in a bull market."

Run on the server (where yfinance + the lake live):
    python -m marketos.models.aggressive_sleeve_backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from marketos.backtest.engine import CostModel, run_backtest
from marketos.backtest.expectancy import compute_expectancy

# Deployment meter levels — the GREEN/YELLOW/RED throttle.
GREEN, YELLOW, RED = 1.0, 0.5, 0.0

# FULL_PERIOD starts here, not at each instrument's inception. AMD has history back to 1980,
# which would make a "full" buy-and-hold dominated by decades irrelevant to the current AI
# secular wave (and pre-dating SOXL/BTC/PLTR/COIN/RKLB entirely). 2015 = BTC liquid, SOXL
# established, the semis/AI era this sleeve is actually built to ride.
FULL_PERIOD_START = "2015-01-01"

# Stress windows the aggressive sleeve must be tested through. SOXL inception is March 2010
# so there is NO 2008 test; BTC liquid history starts ~2014. 2022 is the must-pass gate.
STRESS_WINDOWS = {
    "2018Q4_semis":  ("2018-09-01", "2019-01-31"),
    "covid_2020":    ("2020-02-01", "2020-06-30"),
    "bear_2022":     ("2022-01-01", "2022-12-31"),   # the gate — SOXL ~-90% buy-and-hold
    "recovery_2023": ("2023-01-01", "2023-12-31"),
}


# ── Causal signals ──────────────────────────────────────────────────────────────────────

def trend_up(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """Classic causal trend filter: price above the slow SMA AND fast SMA above slow SMA.
    Both rolling means look only backward, so the value at t uses data up to t."""
    sma_f = close.rolling(fast, min_periods=fast).mean()
    sma_s = close.rolling(slow, min_periods=slow).mean()
    return (close > sma_s) & (sma_f > sma_s)


def vol_regime_causal(returns: pd.Series, window: int = 63, min_history: int = 252) -> pd.Series:
    """Causal vol-regime in {0:calm, 1:neutral, 2:stress}. At each t we compute trailing
    realized vol over `window` days, then ask where THAT sits within the distribution of all
    trailing-vols observed UP TO t (expanding). <50th pct → calm, 50-80th → neutral,
    >80th → stress. Expanding rank = no look-ahead. Before `min_history` we default to
    neutral (not enough history to call calm/stress honestly)."""
    rvol = returns.rolling(window, min_periods=window).std()
    # Expanding percentile rank of the current rvol within all rvol seen so far.
    pct = rvol.expanding(min_periods=min_history).apply(
        lambda x: (x[:-1] <= x[-1]).mean() if len(x) > 1 else np.nan, raw=True)
    regime = pd.Series(1, index=returns.index, dtype=float)  # default neutral
    regime[pct < 0.50] = 0
    regime[pct > 0.80] = 2
    regime[pct.isna()] = 1
    return regime


# Deployment policies. The user's chosen base is "near buy-and-hold" (max terminal wealth,
# drawdowns accepted) — but a 3x DAILY-RESET ETF held through a sustained bear suffers
# volatility decay and can be permanently impaired even if the underlying recovers. So we
# offer three policies and compare them honestly:
#   "throttle"        — the original GREEN/YELLOW/RED (survival-first; caps upside hard)
#   "buyhold"         — always fully deployed (the user's stated preference)
#   "circuit_breaker" — fully deployed THROUGH dips; exits only in a CONFIRMED sustained
#                       downtrend (price below a falling 200DMA). Preserves nearly all
#                       upside while preventing the vol-decay-to-zero tail on SOXL.
#   "circuit_breaker_dip" — circuit_breaker PLUS conviction-based fast dip re-entry: when
#                       the breaker is OUT (confirmed bear), re-enter partially if a
#                       capitulation-reversal fires (deep drawdown + oversold RSI turning
#                       up + short-MA reclaim) — the causal price-only proxy for market-os's
#                       validated stress-regime bounce signal. Recaptures fast V-bottoms.
POLICIES = ("throttle", "buyhold", "circuit_breaker", "circuit_breaker_dip")

# Fraction to re-deploy on a high-conviction dip while the long-term breaker is still OUT.
# Partial (not full) because re-entering into a confirmed bear is the riskiest bet we make.
DIP_REENTRY = 0.6


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Causal Wilder-style RSI. Only past closes feed each value."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def dip_reentry_signal(close: pd.Series, *, dd_thresh: float = -0.20,
                       reentry_ma: int = 50, rsi_lo: float = 35.0) -> pd.Series:
    """Causal bounce-CONFIRMATION detector — a PERSISTENT 'safe to re-enter' state, not a
    one-day event (a one-day oversold tick is missed by a late-tripping breaker, see notes):
      (1) DIP HAPPENED: price fell >|dd_thresh| below its trailing 60-day high at some
          point recently (there was a real selloff to buy back into),
      (2) WAS OVERSOLD: RSI dipped below `rsi_lo` within the last 60 days (capitulation), and
      (3) BOUNCE CONFIRMED: price has reclaimed a RISING `reentry_ma`-day MA (the recovery is
          actually underway — momentum re-established, not a falling-knife catch).
    Persistent so it stays True through the recovery regardless of exactly when the oversold
    tick occurred. Causal throughout → tradeable next-bar."""
    roll_high = close.rolling(60, min_periods=20).max()
    dd_recent = (close / roll_high - 1.0).rolling(60, min_periods=20).min()
    rsi = _rsi(close)
    was_oversold = (rsi < rsi_lo).rolling(60, min_periods=1).max().astype(bool)
    sma_r = close.rolling(reentry_ma, min_periods=reentry_ma).mean()
    bounce_confirmed = (close > sma_r) & (sma_r.diff(5) > 0)
    return ((dd_recent < dd_thresh) & was_oversold & bounce_confirmed).fillna(False)


def deployment_weight(close: pd.Series, policy: str = "throttle") -> pd.Series:
    """Per-instrument deployment fraction under the chosen policy."""
    if policy == "buyhold":
        return pd.Series(1.0, index=close.index, dtype=float)

    if policy in ("circuit_breaker", "circuit_breaker_dip"):
        # Stay 100% invested unless price is below a 200DMA that is ALSO falling — i.e. a
        # confirmed multi-week downtrend, not a dip. A V-shaped dip keeps the 200DMA rising,
        # so the breaker does NOT trip and you ride the bounce (fixing the recovery-lag).
        sma_s = close.rolling(200, min_periods=200).mean()
        sustained_down = ((close < sma_s) & (sma_s.diff(20) < 0)).fillna(False)
        base = pd.Series(1.0, index=close.index, dtype=float)
        base[sustained_down] = 0.0
        if policy == "circuit_breaker":
            return base
        return _circuit_breaker_dip_weight(close, base)

    # Default: the GREEN/YELLOW/RED throttle.
    #   GREEN (1.0): trend up AND calm vol-regime; YELLOW (0.5): trend up AND neutral;
    #   RED (0.0): trend down OR stress vol-regime.
    ret = close.pct_change()
    up = trend_up(close)
    reg = vol_regime_causal(ret)
    w = pd.Series(RED, index=close.index, dtype=float)
    w[up & (reg == 1)] = YELLOW
    w[up & (reg == 0)] = GREEN
    return w


def _circuit_breaker_dip_weight(close: pd.Series, base: pd.Series) -> pd.Series:
    """Overlay fast dip re-entry on the circuit-breaker base. While the breaker is OUT
    (base==0, confirmed bear), latch into a partial position when `dip_reentry_signal`
    fires, hold it while price stays above its 20DMA, and exit if it breaks back below (the
    bounce failed). When the real trend reclaims (base flips to 1.0) we go fully invested
    and reset the latch. Stateful by nature → an explicit loop (≈3k rows, negligible)."""
    dip = dip_reentry_signal(close)
    sma20 = close.rolling(20, min_periods=20).mean()
    w = base.copy()
    latched = False
    for i in range(len(close)):
        if base.iloc[i] == 1.0:           # trend reclaimed → fully in, reset latch
            latched = False
            continue
        if latched:
            if close.iloc[i] < sma20.iloc[i]:   # failed bounce → stand down
                latched = False
                w.iloc[i] = 0.0
            else:
                w.iloc[i] = DIP_REENTRY
        elif dip.iloc[i]:                  # capitulation-reversal → latch in partially
            latched = True
            w.iloc[i] = DIP_REENTRY
        else:
            w.iloc[i] = 0.0
    return w


# ── Portfolio construction & per-window evaluation ────────────────────────────────────────

def build_target_weights(prices: dict[str, pd.Series], policy: str = "throttle",
                         max_gross: float = 1.0) -> pd.DataFrame:
    """Per-instrument deployment fractions under `policy`, equal-weighted across the active
    names and scaled so gross never exceeds `max_gross`. A name at 0 is held in cash."""
    raw = pd.DataFrame({sym: deployment_weight(px, policy) for sym, px in prices.items()}).fillna(0.0)
    # Equal-weight the sleeve: each name's target is its throttle fraction / N, then
    # renormalize down (never up) if gross would exceed the cap.
    n = max(len(prices), 1)
    w = raw / n
    gross = w.sum(axis=1)
    scale = np.where(gross > max_gross, max_gross / gross.replace(0, np.nan), 1.0)
    return w.mul(pd.Series(scale, index=w.index).fillna(1.0), axis=0)


def _curve_stats(equity: pd.Series, daily_ret: pd.Series) -> dict:
    """CAGR, annualized Sharpe (daily, 252), and max drawdown for an equity curve."""
    if len(equity) < 2:
        return {"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "total_return": 0.0}
    years = len(equity) / 252
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0
    sd = daily_ret.std()
    sharpe = float(daily_ret.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
    peak = equity.cummax()
    maxdd = float(((equity - peak) / peak).min())
    return {"cagr": cagr, "sharpe": sharpe, "max_drawdown": maxdd, "total_return": total}


def evaluate_window(prices: dict[str, pd.Series], label: str, start: str | None, end: str | None,
                    *, cost: CostModel | None = None, max_gross: float = 1.0,
                    policies: tuple[str, ...] = POLICIES) -> dict:
    """Run every policy over [start, end] alongside SOXL-only buy-and-hold (the brutal
    single-name benchmark). Returns one stat block per policy plus deployment levels."""
    px = pd.DataFrame(prices).sort_index()
    if start:
        px = px[px.index >= pd.Timestamp(start)]
    if end:
        px = px[px.index <= pd.Timestamp(end)]
    px = px.dropna(how="all")
    if len(px) < 30:
        return {"label": label, "n_days": len(px), "skipped": "insufficient history in window"}

    out = {"label": label, "n_days": len(px), "policies": {}, "deployment": {}}
    for policy in policies:
        # Build weights on the FULL series so SMAs/regimes are warm, then slice to the
        # window — a window shorter than 200 days would have an all-NaN trend filter.
        w = build_target_weights(prices, policy=policy, max_gross=max_gross).reindex(px.index).fillna(0.0)
        res = run_backtest(px, w, cost=cost, price_col="close")
        out["policies"][policy] = _curve_stats(res.equity_curve, res.period_returns)
        out["deployment"][policy] = float(w.sum(axis=1).mean())

    # SOXL-only buy-and-hold — the brutal single-name benchmark, if present.
    if "SOXL" in px.columns:
        s_ret = px["SOXL"].pct_change().fillna(0.0)
        out["buyhold_soxl"] = _curve_stats((1 + s_ret).cumprod(), s_ret)
    else:
        out["buyhold_soxl"] = None
    return out


# ── Entry point ───────────────────────────────────────────────────────────────────────────

def run_reality_check(price_overrides: dict[str, pd.Series] | None = None,
                      *, cost: CostModel | None = None) -> list[dict]:
    """Fetch the aggressive sleeve's max history (or use injected `price_overrides` for
    testing) and evaluate the throttle through every stress window plus the full period.
    Prints a comparison table and returns the per-window result dicts."""
    cost = cost or CostModel()
    if price_overrides is not None:
        prices = {k: v.sort_index() for k, v in price_overrides.items()}
    else:
        prices = _fetch_aggressive_prices()

    if not prices:
        print("No price data available — cannot run reality check.")
        return []

    # Normalize every series to BUSINESS-DAY frequency. This collapses BTC's 7-day calendar
    # onto the tradeable 5-day grid (weekend crypto moves fold into the next session's
    # return) and prevents the union-of-calendars artifact that otherwise inflates n_days.
    prices = {sym: s.reindex(pd.bdate_range(s.index.min(), s.index.max())).ffill()
              for sym, s in prices.items() if s is not None and not s.empty}

    windows = {"FULL_PERIOD": (FULL_PERIOD_START, None), **STRESS_WINDOWS}
    results = [evaluate_window(prices, lbl, s, e, cost=cost) for lbl, (s, e) in windows.items()]
    _print_report(results)
    return results


def _fetch_aggressive_prices() -> dict[str, pd.Series]:
    """Pull max-history close series for the aggressive sleeve via the lake-backed fetcher.
    Imported lazily so the module loads (and tests) without yfinance/network present."""
    import os
    from marketos.config import Config
    from marketos.portfolio.sleeves import load_sleeves
    from marketos.data.lake import DataLake
    from marketos.data.fetchers.yfinance_fetcher import fetch_ohlcv

    cfg = Config.load()
    sleeve = load_sleeves(cfg.raw)["aggressive"]
    lake = DataLake(os.environ.get("DATA_LAKE_ROOT", "data_lake"))
    out: dict[str, pd.Series] = {}
    for sym in sleeve.universe:
        try:
            df = fetch_ohlcv(sym, lake=lake, period="max")
            if not df.empty:
                out[sym] = df["close"]
        except Exception as exc:  # noqa: BLE001 — degrade gracefully, report which failed
            print(f"  fetch {sym} failed: {exc}")
    return out


def _fmt(s: dict | None) -> str:
    if not s:
        return "      n/a"
    return (f"ret={s['total_return']:+7.1%}  CAGR={s['cagr']:+6.1%}  "
            f"Sharpe={s['sharpe']:5.2f}  maxDD={s['max_drawdown']:6.1%}")


_POLICY_LABELS = {
    "buyhold":             "BUY&HOLD          ",
    "circuit_breaker_dip": "BREAKER+DIP-REENTRY",
    "circuit_breaker":     "CIRCUIT-BREAKER   ",
    "throttle":            "THROTTLE          ",
}


def _print_report(results: list[dict]) -> None:
    print("\n" + "=" * 96)
    print("AGGRESSIVE SLEEVE — POLICY COMPARISON (sleeve equal-weight, costs + next-bar exec)")
    print("=" * 96)
    for r in results:
        if r.get("skipped"):
            print(f"\n[{r['label']}] SKIPPED — {r['skipped']}")
            continue
        print(f"\n[{r['label']}]  n_days={r['n_days']}")
        for policy in ("buyhold", "circuit_breaker", "circuit_breaker_dip", "throttle"):
            stats = r["policies"].get(policy)
            dep = r["deployment"].get(policy, 0.0)
            print(f"  {_POLICY_LABELS[policy]}: {_fmt(stats)}  deploy={dep:.0%}")
        print(f"  buy&hold(SOXL) : {_fmt(r['buyhold_soxl'])}  [single-name vol-decay benchmark]")
    print("\n" + "-" * 96)
    print("READ: BUY&HOLD = max raw return, deepest drawdowns (vol-decay risk on SOXL).")
    print("      CIRCUIT-BREAKER = stays in through dips, exits only confirmed bears —")
    print("      should keep most upside while capping the -90% tail. THROTTLE = survival-first.")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    run_reality_check()
