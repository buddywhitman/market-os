"""Unit tests for data/fetchers/angelone_orders.py — the highest-stakes module in this
project (it can place real trades on a real broker account). These tests exist
specifically to guard the dry_run safety invariant: dry_run defaults to True, and a
dry-run call must NEVER make a network request.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from marketos.data.fetchers.angelone_orders import place_order


def test_dry_run_is_the_default():
    """If a caller forgets to pass dry_run at all, it must default to safe."""
    with patch("marketos.data.fetchers.angelone_orders.requests.post") as mock_post:
        result = place_order(
            tradingsymbol="BEL-EQ", symboltoken="383", transaction_type="BUY",
            quantity=1, price=280.0, jwt_token="fake", api_key="fake",
        )
    assert result["dry_run"] is True
    mock_post.assert_not_called()  # the entire point — no network call on dry-run


def test_dry_run_never_calls_the_network_even_when_explicit():
    with patch("marketos.data.fetchers.angelone_orders.requests.post") as mock_post:
        place_order(
            tradingsymbol="BEL-EQ", symboltoken="383", transaction_type="BUY",
            quantity=1, price=280.0, jwt_token="fake", api_key="fake", dry_run=True,
        )
    mock_post.assert_not_called()


def test_live_order_requires_explicit_dry_run_false():
    """The real API path is only reachable by explicitly passing dry_run=False — confirms
    the live branch exists and is gated correctly, without actually depending on a real
    network response (mocked)."""
    with patch("marketos.data.fetchers.angelone_orders.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "status": True, "data": {"orderid": "FAKE123"}, "message": "SUCCESS",
        }
        result = place_order(
            tradingsymbol="BEL-EQ", symboltoken="383", transaction_type="BUY",
            quantity=1, price=280.0, jwt_token="fake", api_key="fake", dry_run=False,
        )
    mock_post.assert_called_once()
    assert result["dry_run"] is False
    assert result["order_id"] == "FAKE123"


def test_invalid_transaction_type_rejected_before_any_network_call():
    with patch("marketos.data.fetchers.angelone_orders.requests.post") as mock_post:
        with pytest.raises(ValueError):
            place_order(
                tradingsymbol="BEL-EQ", symboltoken="383", transaction_type="HOLD",
                quantity=1, price=280.0, jwt_token="fake", api_key="fake", dry_run=False,
            )
    mock_post.assert_not_called()


def test_zero_or_negative_quantity_rejected():
    with pytest.raises(ValueError):
        place_order(
            tradingsymbol="BEL-EQ", symboltoken="383", transaction_type="BUY",
            quantity=0, price=280.0, jwt_token="fake", api_key="fake",
        )


def test_dry_run_request_body_shape_is_correct():
    """The dry-run body is what would ACTUALLY be sent if dry_run were False — verify its
    shape now, since this is exactly what a future approval-flow UI would show the user
    before they confirm a real order."""
    result = place_order(
        tradingsymbol="BEL-EQ", symboltoken="383", transaction_type="BUY",
        quantity=3, price=280.5, jwt_token="fake", api_key="fake",
    )
    body = result["would_send"]
    assert body["tradingsymbol"] == "BEL-EQ"
    assert body["transactiontype"] == "BUY"
    assert body["quantity"] == "3"
    assert body["price"] == "280.5"
    assert body["producttype"] == "DELIVERY"  # swing-holding default, not intraday


def test_market_order_ignores_price():
    result = place_order(
        tradingsymbol="BEL-EQ", symboltoken="383", transaction_type="SELL",
        quantity=1, price=None, jwt_token="fake", api_key="fake", order_type="MARKET",
    )
    assert result["would_send"]["price"] == "0"
