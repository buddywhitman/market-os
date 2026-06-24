"""Unit tests for notify/briefing.py — the Telegram-formatted version of the same
decision logic the Action Plan dashboard page renders. Tests the FORMATTING here;
the underlying decisions (action_plan.build_action_items) already have their own
dedicated test suite — not re-tested here, only reused.
"""
from __future__ import annotations

from marketos.notify.briefing import (
    build_india_morning_briefing, build_us_evening_briefing, format_briefing,
)
from marketos.portfolio.action_plan import build_action_items


class _FakeStore:
    def __init__(self, rows_by_strategy: dict[str, list[dict]] | None = None):
        self._rows = rows_by_strategy or {}

    def get_portfolio_history(self, strategy_name, limit=1):
        return self._rows.get(strategy_name, [])


def test_empty_items_produces_a_clear_no_signal_message_not_silence():
    """A scheduled briefing with nothing to report must say so explicitly — silence on a
    push notification reads as 'did this even run,' not 'no signal today.'"""
    text = format_briefing([], title="Test Briefing")
    assert "No actionable items" in text
    assert "Test Briefing" in text


def test_india_briefing_with_no_data_does_not_crash():
    text = build_india_morning_briefing(_FakeStore())
    assert "No actionable items" in text


def test_india_briefing_renders_real_shaped_buy_item():
    rows = {"india_sleeve": [{"top_positions": [
        {"symbol": "BEL", "sector": "DEFENSE_ELECTRONICS", "weight": 0.15,
         "notional": 750.0, "entry_price": 280.0, "stop_price": 252.0,
         "confidence": 0.78, "expectancy": 0.042, "reason": "Analog evidence."},
    ]}]}
    text = build_india_morning_briefing(_FakeStore(rows))
    assert "BEL" in text
    assert "₹" in text  # INR, not USD, for the india sleeve
    assert "$" not in text
    assert "MANUAL" in text  # the no-broker invariant must survive into the briefing too


def test_us_briefing_renders_usd_not_inr():
    rows = {"quant_sleeve": [{"top_positions": [
        {"symbol": "NVDA", "weight": 0.05, "notional": 1000.0, "entry_price": 200.0,
         "stop_price": 180.0, "confidence": 0.7, "expectancy": 0.03},
    ]}], "aggressive_sleeve": []}
    text = build_us_evening_briefing(_FakeStore(rows))
    assert "NVDA" in text
    assert "$" in text
    assert "₹" not in text


def test_briefing_always_states_manual_execution_invariant():
    """The single most consequential thing this must never get wrong, same as the
    dashboard's Action Plan page — never imply automatic execution exists."""
    rows = {"india_sleeve": [{"top_positions": [
        {"symbol": "BEL", "weight": 0.15, "notional": 750.0, "entry_price": 280.0,
         "stop_price": 252.0, "confidence": 0.78, "expectancy": 0.042, "reason": "x"},
    ]}]}
    text = build_india_morning_briefing(_FakeStore(rows))
    assert "no automatic order placement" in text.lower()
    assert "MANUAL" in text


def test_avoid_items_separated_from_buy_items_in_output():
    items = build_action_items(
        [], [{"symbol": "COIN", "in_position": False, "weight": 0.0, "notional": 0.0,
             "entry_price": 150.0, "stop_price": 0.0, "reason": "trend broke",
             "conviction": None}],
    )
    text = format_briefing(items, title="Test")
    assert "Exit if held" in text
    assert "COIN" in text
