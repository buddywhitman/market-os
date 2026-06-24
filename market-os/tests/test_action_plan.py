"""Unit tests for portfolio/action_plan.py — the highest-stakes rendering surface in the
dashboard (a newbie investor acting directly on this page's numbers). Pure-logic module,
no Streamlit involved, so these test the actual decision rules directly.
"""
from __future__ import annotations

from marketos.portfolio.action_plan import build_action_items


def test_quant_zero_weight_excluded():
    quant = [{"symbol": "X", "weight": 0.0, "notional": 0.0, "entry_price": 100.0,
             "stop_price": 90.0, "confidence": 0.6, "expectancy": 0.02}]
    items = build_action_items(quant, [])
    assert items == []


def test_quant_too_small_to_buy_one_share_excluded():
    """$50 notional at $1000/share rounds to 0 whole shares — not an actionable item."""
    quant = [{"symbol": "X", "weight": 0.01, "notional": 50.0, "entry_price": 1000.0,
             "stop_price": 900.0, "confidence": 0.6, "expectancy": 0.02}]
    items = build_action_items(quant, [])
    assert items == []


def test_quant_buy_item_shape_and_math():
    quant = [{"symbol": "NVDA", "weight": 0.05, "notional": 1000.0, "entry_price": 200.0,
             "stop_price": 180.0, "confidence": 0.7, "expectancy": 0.03}]
    items = build_action_items(quant, [])
    assert len(items) == 1
    it = items[0]
    assert it.sleeve == "quant"
    assert it.action == "BUY/HOLD"
    assert it.order_type == "LIMIT"
    assert it.quantity == 5  # floor(1000/200)
    assert it.limit_price > 200.0  # buffer applied above last price
    assert it.stop_loss_price == 180.0
    assert "MANUAL" in it.execution  # no broker exists — never claims auto, regardless of sleeve


def test_aggressive_in_full_trend_buy():
    agg = [{"symbol": "NVDA", "in_position": True, "weight": 0.4, "notional": 4000.0,
           "entry_price": 200.0, "stop_price": 180.0, "reason": "trend intact",
           "conviction": None}]
    items = build_action_items([], agg)
    assert len(items) == 1
    assert items[0].action == "BUY/HOLD"
    assert "MANUAL" in items[0].execution


def test_aggressive_gated_dip_reentry_is_flagged_smaller_and_cautious():
    agg = [{"symbol": "MSTR", "in_position": True, "weight": 0.06, "notional": 600.0,
           "entry_price": 100.0, "stop_price": 85.0, "reason": "gated", "conviction": 0.8}]
    items = build_action_items([], agg)
    assert len(items) == 1
    it = items[0]
    assert "cautious" in it.action.lower() or "smaller" in it.action.lower()
    assert "80%" in it.reason or "0.8" in it.reason


def test_aggressive_out_of_position_is_exit_or_avoid_with_real_reason():
    agg = [{"symbol": "COIN", "in_position": False, "weight": 0.0, "notional": 0.0,
           "entry_price": 150.0, "stop_price": 0.0,
           "reason": "circuit_breaker: confirmed sustained downtrend", "conviction": None}]
    items = build_action_items([], agg)
    assert len(items) == 1
    it = items[0]
    assert it.action == "EXIT IF HELD / AVOID NEW BUYS"
    assert it.quantity == 0
    assert "downtrend" in it.reason


def test_ordering_buys_before_avoid_and_quant_before_aggressive():
    quant = [{"symbol": "Q1", "weight": 0.05, "notional": 1000.0, "entry_price": 100.0,
             "stop_price": 90.0, "confidence": 0.6, "expectancy": 0.02}]
    agg = [
        {"symbol": "AVOID1", "in_position": False, "weight": 0.0, "notional": 0.0,
         "entry_price": 50.0, "stop_price": 0.0, "reason": "out", "conviction": None},
        {"symbol": "BUY1", "in_position": True, "weight": 0.3, "notional": 3000.0,
         "entry_price": 100.0, "stop_price": 90.0, "reason": "in", "conviction": None},
    ]
    items = build_action_items(quant, agg)
    symbols_in_order = [it.symbol for it in items]
    assert symbols_in_order == ["Q1", "BUY1", "AVOID1"]


def test_never_claims_auto_execution_regardless_of_sleeve():
    """The single most consequential thing this module must never get wrong: no broker
    is connected. Every item, from either sleeve, must say MANUAL."""
    quant = [{"symbol": "Q1", "weight": 0.05, "notional": 1000.0, "entry_price": 100.0,
             "stop_price": 90.0, "confidence": 0.6, "expectancy": 0.02}]
    agg = [{"symbol": "A1", "in_position": True, "weight": 0.3, "notional": 3000.0,
           "entry_price": 100.0, "stop_price": 90.0, "reason": "in", "conviction": None}]
    items = build_action_items(quant, agg)
    assert all("MANUAL" in it.execution and "auto" not in it.execution.lower() for it in items)


def test_india_item_uses_inr_not_usd():
    india = [{"symbol": "BEL", "weight": 0.15, "notional": 750.0, "entry_price": 280.0,
             "stop_price": 252.0, "reason": "screen rank only"}]
    items = build_action_items([], [], india)
    assert len(items) == 1
    it = items[0]
    assert it.sleeve == "india"
    assert it.currency == "INR"
    assert it.notional_usd == 0.0
    assert it.notional_inr > 0
    assert "MANUAL" in it.execution


def test_india_positions_defaults_to_none_safely():
    quant = [{"symbol": "Q1", "weight": 0.05, "notional": 1000.0, "entry_price": 100.0,
             "stop_price": 90.0, "confidence": 0.6, "expectancy": 0.02}]
    items_without_india = build_action_items(quant, [])
    items_with_empty_india = build_action_items(quant, [], [])
    assert len(items_without_india) == len(items_with_empty_india) == 1
