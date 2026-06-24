"""AngelOne SmartAPI — order placement. Deliberately a SEPARATE module from
angelone_fetcher.py (auth + read-only data): placing a real order is a categorically
different risk tier than reading a quote, and keeping them apart means importing this
module is itself a visible signal that order-affecting code is in play.

HARD SAFETY RULE: `dry_run` defaults to True on every function here and stays True
regardless of the account's current balance. The account being empty today is not a
reason to ship code that defaults to live — this code keeps running after the account
gets funded (the whole point is ₹5,000→₹50,000), and "harmless because broke" is not a
property that survives. A caller must pass `dry_run=False` explicitly, every single time,
to place a real order. There is no global "go live" switch — that's deliberate.

Every order attempt — dry-run AND real — is logged via `log_order_attempt` (caller's
responsibility to call it) so there is always a full audit trail, matching the project's
existing trade_attribution discipline for the US sleeves.

This module does NOT decide WHEN to trade. It only knows how to (attempt to) place,
check, and read back orders once something else has already decided what to do. Wiring
this into any automatic/scheduled execution path is a separate, much bigger decision than
building the primitive — not done here, not implied by this module existing.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import requests

SMARTAPI_BASE = "https://apiconnect.angelbroking.com"
_TIMEOUT = 10


def _headers(jwt_token: str, api_key: str) -> dict:
    return {
        "Content-Type": "application/json", "Accept": "application/json",
        "X-UserType": "USER", "X-SourceID": "WEB",
        "X-ClientLocalIP": "127.0.0.1", "X-ClientPublicIP": "127.0.0.1",
        "X-MACAddress": "00:00:00:00:00:00", "X-PrivateKey": api_key,
        "Authorization": f"Bearer {jwt_token}",
    }


def place_order(
    *, tradingsymbol: str, symboltoken: str, transaction_type: str, quantity: int,
    price: float | None, jwt_token: str, api_key: str,
    order_type: str = "LIMIT", product_type: str = "DELIVERY",
    dry_run: bool = True,
) -> dict:
    """Place (or simulate placing) one order.

    `transaction_type` — "BUY" or "SELL". `order_type` — "LIMIT" or "MARKET" (price
    ignored for MARKET). `product_type` — "DELIVERY" (matches the india sleeve's swing/
    multi-day holding design, NOT intraday) by default.

    `dry_run=True` (the default, and the only safe way to call this until a deliberate,
    separate decision is made to go live): builds the exact request that WOULD be sent,
    does NOT call the API, and returns it tagged `dry_run: True` — lets every caller
    (Action Plan page, a future approval flow) show "this is what would happen" before
    anything real occurs.
    """
    if transaction_type not in ("BUY", "SELL"):
        raise ValueError(f"transaction_type must be BUY or SELL, got {transaction_type!r}")
    if quantity <= 0:
        raise ValueError(f"quantity must be positive, got {quantity}")

    body = {
        "variety": "NORMAL", "tradingsymbol": tradingsymbol, "symboltoken": symboltoken,
        "transactiontype": transaction_type, "exchange": "NSE", "ordertype": order_type,
        "producttype": product_type, "duration": "DAY",
        "price": "0" if order_type == "MARKET" else str(price),
        "squareoff": "0", "stoploss": "0", "quantity": str(quantity),
    }

    if dry_run:
        return {"dry_run": True, "would_send": body, "status": None, "order_id": None,
               "checked_at": datetime.now(timezone.utc)}

    url = f"{SMARTAPI_BASE}/rest/secure/angelbroking/order/v1/placeOrder"
    try:
        r = requests.post(url, json=body, headers=_headers(jwt_token, api_key), timeout=_TIMEOUT)
        payload = r.json()
    except (requests.RequestException, ValueError) as exc:
        return {"dry_run": False, "would_send": body, "status": False,
               "error": str(exc), "order_id": None}

    return {
        "dry_run": False, "would_send": body, "status": payload.get("status"),
        "order_id": (payload.get("data") or {}).get("orderid"),
        "message": payload.get("message"), "error_code": payload.get("errorcode"),
        "http_status": r.status_code, "placed_at": datetime.now(timezone.utc),
    }


def get_order_book(*, jwt_token: str, api_key: str) -> list[dict]:
    """Read-only — today's order book (status of every order placed today). Safe to call
    any time; returns [] on any failure rather than raising."""
    url = f"{SMARTAPI_BASE}/rest/secure/angelbroking/order/v1/getOrderBook"
    try:
        r = requests.get(url, headers=_headers(jwt_token, api_key), timeout=_TIMEOUT)
        payload = r.json()
    except (requests.RequestException, ValueError):
        return []
    return payload.get("data") or [] if payload.get("status") else []


def get_positions(*, jwt_token: str, api_key: str) -> list[dict]:
    """Read-only — current open positions. Safe to call any time."""
    url = f"{SMARTAPI_BASE}/rest/secure/angelbroking/order/v1/getPosition"
    try:
        r = requests.get(url, headers=_headers(jwt_token, api_key), timeout=_TIMEOUT)
        payload = r.json()
    except (requests.RequestException, ValueError):
        return []
    return payload.get("data") or [] if payload.get("status") else []


def get_holdings(*, jwt_token: str, api_key: str) -> list[dict]:
    """Read-only — current delivery holdings (what's actually owned, not just today's
    intraday position). Safe to call any time."""
    url = f"{SMARTAPI_BASE}/rest/secure/angelbroking/portfolio/v1/getHolding"
    try:
        r = requests.get(url, headers=_headers(jwt_token, api_key), timeout=_TIMEOUT)
        payload = r.json()
    except (requests.RequestException, ValueError):
        return []
    return payload.get("data") or [] if payload.get("status") else []


def log_order_attempt(store, order_result: dict, *, symbol: str, reason: str) -> None:
    """Audit trail for every order attempt — dry-run AND real, same as the US sleeves'
    trade_attribution discipline. Call this for EVERY place_order() result, regardless of
    dry_run, so there is always a complete record of what was attempted, decided, and (if
    real) what happened.
    """
    from marketos.data.fetchers.orchestrator import _store_features
    now = datetime.now(timezone.utc)
    _store_features(store, f"_order_{symbol}", "angelone_order_log", {
        "asof_ts": now, "knowledge_ts": now, "symbol": symbol, "reason": reason,
        "dry_run": order_result.get("dry_run"), "status": order_result.get("status"),
        "order_id": order_result.get("order_id"), "would_send": order_result.get("would_send"),
        "error": order_result.get("error"), "message": order_result.get("message"),
    })
