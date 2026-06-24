"""Unit tests for portfolio/india_snapshot.py — the india sleeve's real ₹5,000 sizing.

NSE bhavcopy access is currently blocked (Akamai bot-protection on the server's IP, see
project notes) so these test the LOGIC with a fake store + synthetic price history, the
same pattern used everywhere else in this session. Will run identically against real data
once the data-access blocker is resolved (AngelOne SmartAPI recommended).
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from marketos.portfolio.india_snapshot import build_india_snapshot, snapshot_to_attribution
from marketos.risk.sizing import RiskLimits


class _FakeStore:
    def __init__(self, candidates, analog: dict | None = None):
        self._candidates = candidates
        self._analog = analog or {}  # {symbol: analog_dict}

    def get_latest_family(self, symbol, family):
        if family == "screen":
            return {"top_candidates": self._candidates}
        if family == "analog_india":
            return self._analog.get(symbol, {})
        return {}


def _synthetic_history(prices: dict[str, float], n: int = 20) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2026-01-01", periods=n)
    out = {}
    for sym, price in prices.items():
        close = pd.Series(np.linspace(price * 0.9, price, n), index=dates)
        out[sym] = pd.DataFrame(
            {"close": close, "high": close * 1.02, "low": close * 0.98,
             "volume": np.full(n, 100_000)}, index=dates)
    return out


@pytest.fixture
def limits():
    return RiskLimits(risk_per_trade=0.05, max_name_weight=0.25, kelly_fraction=0.30,
                      atr_stop_mult=2.5)


def test_no_screen_data_returns_empty_with_reason(limits):
    snap = build_india_snapshot(_FakeStore([]), limits=limits, capital_inr=5000.0)
    assert snap["positions"] == []
    assert snap["reason"] == "no_screen_data"


def test_trend_unhealthy_candidate_excluded_even_if_present(limits):
    """Regression guard: a candidate that ranked into the stored top_candidates list but
    has trend_healthy=False and/or non-positive screen_score must NOT be sized, even if
    the candidate pool is small enough that it would otherwise make a naive top-N cut."""
    candidates = [
        {"symbol": "DIXON", "sector": "ELECTRONICS_VLSI_EMS", "screen_score": 0.25,
         "mom_63d": 0.20, "trend_healthy": True},
        {"symbol": "TCS", "sector": "IT_SERVICES", "screen_score": 0.05,
         "mom_63d": 0.02, "trend_healthy": False},
    ]
    history = _synthetic_history({"DIXON": 15000.0, "TCS": 3800.0})
    with patch("marketos.features.screening.fetch_angelone_history", return_value=history):
        snap = build_india_snapshot(_FakeStore(candidates), limits=limits, capital_inr=5000.0)
    symbols = [p.symbol for p in snap["positions"]]
    assert "TCS" not in symbols
    assert "DIXON" in symbols


def test_all_candidates_trend_unhealthy_returns_empty_with_reason(limits):
    candidates = [{"symbol": "X", "sector": "S", "screen_score": -0.1,
                  "mom_63d": -0.05, "trend_healthy": False}]
    snap = build_india_snapshot(_FakeStore(candidates), limits=limits, capital_inr=5000.0)
    assert snap["positions"] == []
    assert snap["reason"] == "no_candidate_passed_trend_health_filter"


def test_position_count_capped_at_max_positions(limits):
    from marketos.portfolio.india_snapshot import MAX_POSITIONS
    candidates = [
        {"symbol": f"S{i}", "sector": "X", "screen_score": 0.30 - i * 0.01,
         "mom_63d": 0.1, "trend_healthy": True}
        for i in range(10)
    ]
    history = _synthetic_history({f"S{i}": 100.0 + i for i in range(10)})
    with patch("marketos.features.screening.fetch_angelone_history", return_value=history):
        snap = build_india_snapshot(_FakeStore(candidates), limits=limits, capital_inr=5000.0)
    assert len(snap["positions"]) <= MAX_POSITIONS


def test_sizing_uses_real_capital_not_a_paper_figure(limits):
    """The whole point of this rework: ₹5,000 must produce ₹5,000-scale notionals, not
    leftover $100k-paper-figure math."""
    candidates = [{"symbol": "BEL", "sector": "DEFENSE_ELECTRONICS", "screen_score": 0.2,
                  "mom_63d": 0.15, "trend_healthy": True}]
    history = _synthetic_history({"BEL": 280.0})
    with patch("marketos.features.screening.fetch_angelone_history", return_value=history):
        snap = build_india_snapshot(_FakeStore(candidates), limits=limits, capital_inr=5000.0)
    assert len(snap["positions"]) == 1
    p = snap["positions"][0]
    assert p.notional_inr < 5000.0  # can't exceed total real capital
    assert p.notional_inr > 0
    assert "no analog coverage" in p.reason  # honesty flag: this symbol lacks real evidence
    assert p.confidence is None  # no fabricated confidence when there's no real evidence


def test_attribution_shape_for_storage(limits):
    candidates = [{"symbol": "BEL", "sector": "DEFENSE_ELECTRONICS", "screen_score": 0.2,
                  "mom_63d": 0.15, "trend_healthy": True}]
    history = _synthetic_history({"BEL": 280.0})
    with patch("marketos.features.screening.fetch_angelone_history", return_value=history):
        snap = build_india_snapshot(_FakeStore(candidates), limits=limits, capital_inr=5000.0)
    attr = snapshot_to_attribution(snap)
    assert set(attr.keys()) == {"weights", "gross_exposure", "cash_weight", "effective_n",
                                "top_positions", "top_themes"}
    assert 0.0 <= attr["gross_exposure"] <= 1.0001


def test_analog_backed_candidate_ranked_above_technical_only(limits):
    """The core new behavior: a candidate WITH real analog evidence must rank ahead of a
    technical-only candidate, even if the technical-only one has a higher screen_score."""
    candidates = [
        {"symbol": "TECHONLY", "sector": "X", "screen_score": 0.50,  # highest screen_score
         "mom_63d": 0.30, "trend_healthy": True},
        {"symbol": "ANALOGBACKED", "sector": "Y", "screen_score": 0.10,  # lowest screen_score
         "mom_63d": 0.05, "trend_healthy": True},
    ]
    analog = {"ANALOGBACKED": {"analog_mean_ret_20d": 0.04, "analog_n_effective": 15.0,
                               "analog_win_rate_20d": 0.65, "analog_cross_regime_frac": 0.0}}
    history = _synthetic_history({"TECHONLY": 100.0, "ANALOGBACKED": 200.0})
    with patch("marketos.features.screening.fetch_angelone_history", return_value=history):
        snap = build_india_snapshot(_FakeStore(candidates, analog), limits=limits, capital_inr=5000.0)
    symbols_in_order = [p.symbol for p in snap["positions"]]
    assert symbols_in_order[0] == "ANALOGBACKED", \
        "real evidence must outrank a technical-only proxy regardless of screen_score"


def test_analog_negative_expectancy_excludes_even_with_good_screen_score():
    """Real evidence saying 'historically this loses money' must override a positive
    technical screen score — direct evidence beats a weaker proxy in BOTH directions."""
    limits = RiskLimits(risk_per_trade=0.05, max_name_weight=0.25, kelly_fraction=0.30,
                        atr_stop_mult=2.5)
    candidates = [{"symbol": "BADHISTORY", "sector": "X", "screen_score": 0.40,
                  "mom_63d": 0.25, "trend_healthy": True}]
    analog = {"BADHISTORY": {"analog_mean_ret_20d": -0.02, "analog_n_effective": 20.0,
                             "analog_win_rate_20d": 0.30, "analog_cross_regime_frac": 0.0}}
    snap = build_india_snapshot(_FakeStore(candidates, analog), limits=limits, capital_inr=5000.0)
    assert snap["positions"] == []


def test_analog_evidence_below_min_effective_n_treated_as_no_coverage(limits):
    """Thin analog coverage (effective n below the bar) must fall back to technical-only,
    not be treated as real evidence just because a record exists."""
    candidates = [{"symbol": "THIN", "sector": "X", "screen_score": 0.20,
                  "mom_63d": 0.10, "trend_healthy": True}]
    analog = {"THIN": {"analog_mean_ret_20d": 0.05, "analog_n_effective": 2.0,  # below 5.0 bar
                       "analog_win_rate_20d": 0.80, "analog_cross_regime_frac": 0.0}}
    history = _synthetic_history({"THIN": 150.0})
    with patch("marketos.features.screening.fetch_angelone_history", return_value=history):
        snap = build_india_snapshot(_FakeStore(candidates, analog), limits=limits, capital_inr=5000.0)
    assert len(snap["positions"]) == 1
    assert snap["positions"][0].confidence is None  # thin coverage = treated as no coverage
