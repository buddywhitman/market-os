"""Unit tests for Phase 4b's glue logic in portfolio/aggressive_snapshot.py — the
conviction-gated dip re-entry layered on top of the locked circuit_breaker policy.

`conviction_gate.py` and `dip_reentry_signal` already have their own verified behavior
(see project notes / aggressive_sleeve_backtest.py); these tests target the NEW composition
logic in `_evaluate_dip_reentry` — does it call through correctly, does it size/stop a
gated re-entry, does it stay flat with an honest reason when the gate fails or the signal
never fires, does store=None disable Phase 4b entirely (graceful degradation).
"""
from __future__ import annotations

import pandas as pd
import pytest

from marketos.portfolio.aggressive_snapshot import _evaluate_dip_reentry
from marketos.portfolio.conviction_gate import ConvictionResult
from marketos.risk.sizing import RiskLimits


def _synthetic_df(price: float = 100.0, n: int = 60) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = pd.Series([price] * n, index=dates)
    return pd.DataFrame({"close": close, "high": close * 1.01, "low": close * 0.99}, index=dates)


@pytest.fixture
def limits():
    return RiskLimits(max_name_weight=0.40, atr_stop_mult=3.0)


def test_dip_signal_not_fired_stays_flat(monkeypatch, limits):
    df = _synthetic_df()
    monkeypatch.setattr("marketos.portfolio.aggressive_snapshot.dip_reentry_signal",
                        lambda close: pd.Series([False] * len(close), index=close.index))
    pos = _evaluate_dip_reentry("SOXL", df, 100.0, store=None, spy_regime=None,
                                equity=100_000, limits=limits)
    assert pos.in_position is False
    assert pos.weight == 0.0
    assert pos.conviction is None
    assert "falling 200DMA" in pos.reason


def test_dip_signal_fires_but_gate_fails(monkeypatch, limits):
    df = _synthetic_df()
    monkeypatch.setattr("marketos.portfolio.aggressive_snapshot.dip_reentry_signal",
                        lambda close: pd.Series([True] * len(close), index=close.index))
    monkeypatch.setattr("marketos.portfolio.aggressive_snapshot.conviction_gate",
                        lambda symbol, **k: ConvictionResult(gate_pass=False, conviction=0.4,
                                                              reasons=["test: gate failed"]))
    pos = _evaluate_dip_reentry("SOXL", df, 100.0, store=None, spy_regime=None,
                                equity=100_000, limits=limits)
    assert pos.in_position is False
    assert pos.weight == 0.0
    assert pos.conviction == 0.4
    assert "gate FAILED" in pos.reason


def test_dip_signal_fires_and_gate_passes_sizes_a_partial_position(monkeypatch, limits):
    df = _synthetic_df(price=100.0)
    monkeypatch.setattr("marketos.portfolio.aggressive_snapshot.dip_reentry_signal",
                        lambda close: pd.Series([True] * len(close), index=close.index))
    monkeypatch.setattr("marketos.portfolio.aggressive_snapshot.conviction_gate",
                        lambda symbol, **k: ConvictionResult(gate_pass=True, conviction=0.85,
                                                              reasons=["test: gate passed"]))
    pos = _evaluate_dip_reentry("MSTR", df, 100.0, store=None, spy_regime=2,
                                equity=100_000, limits=limits)
    assert pos.in_position is True
    assert pos.conviction == 0.85
    assert "GATED DIP RE-ENTRY" in pos.reason
    # Partial = DIP_REENTRY (0.6) fraction of max_name_weight, never the full circuit_breaker
    # weight a confirmed-trend IN position would get — re-entering a confirmed bear is the
    # riskiest bet this system makes, so it must never be sized as confidently as a normal IN.
    from marketos.models.aggressive_sleeve_backtest import DIP_REENTRY
    assert pos.weight == pytest.approx(limits.max_name_weight * DIP_REENTRY)
    assert pos.notional > 0
    assert pos.shares > 0
    assert pos.stop_price > 0  # must still be ATR-stopped, not "no stop because it's gated"
    assert pos.stop_price < pos.entry_price


def test_store_none_disables_phase_4b_gate_lookup(monkeypatch, limits):
    """store=None must not attempt any analog/regime lookup — Phase 4b should be fully
    disabled, not silently degrade with fabricated evidence."""
    df = _synthetic_df()
    monkeypatch.setattr("marketos.portfolio.aggressive_snapshot.dip_reentry_signal",
                        lambda close: pd.Series([True] * len(close), index=close.index))

    captured = {}

    def fake_gate(symbol, *, spy_regime, analog):
        captured["analog"] = analog
        return ConvictionResult(gate_pass=False, conviction=0.5, reasons=[])

    monkeypatch.setattr("marketos.portfolio.aggressive_snapshot.conviction_gate", fake_gate)
    _evaluate_dip_reentry("SOXL", df, 100.0, store=None, spy_regime=None,
                          equity=100_000, limits=limits)
    assert captured["analog"] is None
