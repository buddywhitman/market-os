"""The gate check: does the validated latent space + regime context actually trade?

Per the project's own promotion criteria (docs/BUILD_ORDER.md): a signal earns capital only
after positive OOS expectancy across walk-forward folds (n>=30), Monte-Carlo p05 terminal >
1.0, and sane attribution. This module answers that question for the supervised-latent
factors (latent_supervised.py) + regime context (regimes/hmm.py) — nothing else. No exits
beyond what backtest.engine already enforces (next-bar execution, real costs), no meta-
labeling, no portfolio allocator, no PM snapshot. Building those before this answer exists
would be wasted work if the answer is "no edge."

Feature set is deliberately narrow: sup_z1..zk (the PLS2 components, already validated
out-of-sample and regime-conditionally) + regime_prob_0..N-1 (soft HMM posteriors). NOT the
full 1,229-feature composite, NOT 50 features — exactly the dimensionality-reduction thesis
this project has been building toward. Cross-asset risk-on/breadth signals are a natural
next addition once a historical (not just latest-snapshot) series exists for them; out of
scope here to keep this pass narrowly focused on the one question it's meant to answer.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from marketos.backtest.engine import CostModel, run_backtest
from marketos.backtest.expectancy import compute_expectancy, monte_carlo_drawdown
from marketos.models.alpha_model import AlphaModel

logger = logging.getLogger(__name__)


def build_alpha_panel(historical_panel: pd.DataFrame, regime_probs: pd.DataFrame) -> pd.DataFrame:
    """Join the supervised-latent historical panel with regime soft-probabilities by date.

    historical_panel: latent_supervised.fit_supervised_latent()["historical_panel"] — one
                      row per (date, symbol) with sup_z1..zk, the 4 forward targets, and the
                      hard regime label already attached.
    regime_probs:     regimes.hmm.detect_regimes(...) output — date-indexed regime_prob_0..N-1.
    """
    df = historical_panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    rp = regime_probs.copy()
    rp.index = pd.to_datetime(rp.index)
    if rp.index.tz is not None:
        rp.index = rp.index.tz_localize(None)
    prob_cols = [c for c in rp.columns if c.startswith("regime_prob_")]

    joined = df.merge(rp[prob_cols], left_on="date", right_index=True, how="left")
    return joined.dropna(subset=prob_cols)


def alpha_feature_cols(panel: pd.DataFrame) -> list[str]:
    z_cols = sorted(c for c in panel.columns if c.startswith("sup_z"))
    prob_cols = sorted(c for c in panel.columns if c.startswith("regime_prob_"))
    return z_cols + prob_cols


def predicted_to_weights(pred_df: pd.DataFrame, *, top_n: int = 5,
                         rebalance_every: int = 20) -> pd.DataFrame:
    """Convert per-(date,symbol) predicted fwd_ret_20d into a discrete rebalance-every-N-days
    long-only equal-weight schedule.

    Rebalancing DAILY against a 20-day-forward prediction would create overlapping holding
    periods that don't correspond to any real decision (you'd be "re-deciding" a position
    every day based on a target that hasn't resolved yet). Rebalancing every `rebalance_every`
    (= the label horizon) days matches the actual decision cadence the target implies.
    """
    dates = sorted(pred_df["date"].unique())
    symbols = sorted(pred_df["symbol"].unique())
    weights = pd.DataFrame(0.0, index=dates, columns=symbols)
    current_w = pd.Series(0.0, index=symbols)
    for i, d in enumerate(dates):
        if i % rebalance_every == 0:
            day = pred_df[pred_df["date"] == d]
            top = day.nlargest(top_n, "pred")["symbol"]
            current_w = pd.Series(0.0, index=symbols)
            if len(top):
                current_w.loc[top] = 1.0 / len(top)
        weights.loc[d] = current_w.values
    return weights


def run_latent_alpha_gate_check(
    historical_panel: pd.DataFrame,
    regime_probs: pd.DataFrame,
    ohlcv_map: dict[str, pd.DataFrame],
    *,
    target_col: str = "fwd_ret_20d",
    label_horizon: int = 20,
    top_n: int = 5,
    train_days: int = 400,
    test_days: int = 60,
) -> dict:
    """latent+regime features -> AlphaModel walk-forward OOS predictions -> rebalance-every-
    label_horizon-days top-N long-only weights -> real backtest -> expectancy report.

    `train_days`/`test_days` are in TRADING DAYS — internally converted to row counts before
    calling AlphaModel, since the pooled (date,symbol) panel has ~n_symbols rows per date and
    AlphaModel.walk_forward_predict's train_periods/test_periods count rows, not days. Passing
    day-counts directly there would silently use a window ~19x too short.
    """
    panel = build_alpha_panel(historical_panel, regime_probs)
    rows_per_date = panel.groupby("date").size().mean() if not panel.empty else 0
    train_periods = int(train_days * rows_per_date)
    test_periods = int(test_days * rows_per_date)

    if panel.empty or len(panel) < train_periods + label_horizon + test_periods:
        logger.warning(f"latent_alpha_gate_check: not enough joined data "
                       f"(have {len(panel)} rows, need {train_periods + label_horizon + test_periods})")
        return {}

    feature_cols = alpha_feature_cols(panel)
    model = AlphaModel(feature_cols=feature_cols, target_col=target_col,
                       label_horizon=label_horizon, classification=False)
    pred_df = model.walk_forward_predict(panel, ts_col="date",
                                         train_periods=train_periods, test_periods=test_periods)
    if pred_df.empty:
        logger.warning("latent_alpha_gate_check: no OOS predictions produced "
                       "(panel too small for even one fold)")
        return {}

    weights = predicted_to_weights(pred_df, top_n=top_n, rebalance_every=label_horizon)

    open_px = pd.DataFrame({sym: df["open"] for sym, df in ohlcv_map.items()
                            if sym in weights.columns})
    open_px.index = pd.to_datetime(open_px.index)
    if open_px.index.tz is not None:
        open_px.index = open_px.index.tz_localize(None)
    open_px = open_px.reindex(weights.index).ffill()

    bt = run_backtest(open_px, weights, cost=CostModel())
    # compute_expectancy's Sharpe/Sortino annualize via sqrt(periods_per_year), defaulting to
    # 252 (daily returns). bt.trade_returns is PER-TRADE, and each trade spans ~label_horizon
    # trading days — annualizing with 252 here would inflate Sharpe by roughly
    # sqrt(252/label_horizon), since it implicitly assumes ~252 independent trade-periods/year
    # when there are really only ~252/label_horizon. This is the single most important line
    # in this module for getting an honest number, not an inflated one.
    trades_per_year = 252 / label_horizon
    report = compute_expectancy(bt.trade_returns.values, periods_per_year=trades_per_year)
    mc = monte_carlo_drawdown(bt.trade_returns.values)

    return {
        "feature_cols": feature_cols,
        "n_oos_rows": len(pred_df),
        "n_oos_dates": pred_df["date"].nunique(),
        "predictions": pred_df,
        "weights": weights,
        "backtest": bt,
        "expectancy": report,
        "monte_carlo": mc,
    }
