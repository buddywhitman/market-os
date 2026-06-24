"""Invariants that keep us honest. These tests are the teeth behind the rules."""

import numpy as np
import pandas as pd
import pytest

from marketos.principles import (
    PrincipleViolation,
    assert_no_lookahead,
    assert_includes_delisted,
    assert_positive_expectancy,
    assert_survives_label_shuffle,
)
from marketos.backtest.expectancy import compute_expectancy
from marketos.backtest.walkforward import walk_forward_splits
from marketos.risk.sizing import RiskLimits, position_size


def test_no_lookahead_catches_future_knowledge():
    df = pd.DataFrame({
        "asof_ts": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "knowledge_ts": pd.to_datetime(["2024-01-01", "2024-01-05"]),  # 2nd row leaks
        "x": [1.0, 2.0],
    })
    with pytest.raises(PrincipleViolation):
        assert_no_lookahead(df)


def test_no_lookahead_passes_when_causal():
    df = pd.DataFrame({
        "asof_ts": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "knowledge_ts": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "x": [1.0, 2.0],
    })
    assert_no_lookahead(df)  # should not raise


def test_survivorship_guard():
    alive_only = pd.DataFrame({"status": ["listed", "listed"]})
    with pytest.raises(PrincipleViolation):
        assert_includes_delisted(alive_only)
    with_graveyard = pd.DataFrame({"status": ["listed", "delisted"]})
    assert_includes_delisted(with_graveyard)


def test_positive_expectancy_gate():
    with pytest.raises(PrincipleViolation):
        assert_positive_expectancy(0.05, sample_size=10)   # too few trades
    with pytest.raises(PrincipleViolation):
        assert_positive_expectancy(-0.01, sample_size=100)  # negative edge
    assert_positive_expectancy(0.02, sample_size=100) is None


def test_label_shuffle_guard():
    # real edge mostly survives shuffling -> fine
    assert_survives_label_shuffle(real_sharpe=1.5, shuffled_sharpe=0.2) is None
    # shuffled labels retain most of the real Sharpe -> artifact, must raise
    with pytest.raises(PrincipleViolation):
        assert_survives_label_shuffle(real_sharpe=1.59, shuffled_sharpe=1.16)


def test_expectancy_math():
    # 2 wins of +0.10, 1 loss of -0.05 → expectancy = (0.10+0.10-0.05)/3
    rep = compute_expectancy(np.array([0.10, 0.10, -0.05]))
    assert rep.sample_size == 3
    assert rep.win_rate == pytest.approx(2 / 3)
    assert rep.expectancy == pytest.approx(0.05)
    assert rep.profit_factor == pytest.approx(0.20 / 0.05)


def test_walkforward_is_purged_and_ordered():
    ts = pd.Series(pd.bdate_range("2020-01-01", periods=300))
    splits = list(walk_forward_splits(ts, train_periods=100, test_periods=20, label_horizon=5))
    assert len(splits) > 0
    for s in splits:
        # test always strictly after train, with the purge gap respected
        assert s.test_idx.min() - s.train_idx.max() >= 5
        assert s.train_idx.max() < s.test_idx.min()


def test_position_size_respects_risk_budget():
    limits = RiskLimits(risk_per_trade=0.01, atr_stop_mult=2.0, max_name_weight=0.10)
    pos = position_size(equity=100_000, entry_price=100, atr=2.0, limits=limits)
    # risk = 1% of 100k = 1000; stop dist = 2*ATR = 4; shares ≈ 250 → notional 25k
    # but name cap is 10% → clamped to 10k notional
    assert pos["weight"] <= 0.10 + 1e-9
    assert pos["stop_price"] < 100
