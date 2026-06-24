"""Plain-language action items — turns the quant + aggressive sleeve snapshots into
exactly what a newbie investor would need to type into a broker: action, order type,
quantity, limit price, stop-loss. Pure logic, no Streamlit — kept separately testable
because this is the highest-stakes rendering surface in the whole dashboard (a newbie
acting on a wrong number here is worse than a wrong number anywhere else in the app).

HONESTY BOUNDARY this module exists to enforce: no broker integration exists yet (Phase 5
was never built — see project notes). Every item says MANUAL, always, regardless of which
sleeve's locked execution mode is nominally "auto" — claiming otherwise here would be the
single most consequential misstatement in the whole system.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionItem:
    sleeve: str               # "quant" | "aggressive" | "india"
    symbol: str
    action: str                # "BUY/HOLD" | "EXIT IF HELD / AVOID" | "WATCH (no action)"
    order_type: str            # "MARKET" | "LIMIT"
    limit_price: float | None  # set when order_type == "LIMIT"
    quantity: int              # whole shares — see note in build_action_items
    notional_usd: float        # 0.0 for india items — see notional_inr instead
    stop_loss_price: float | None
    reason: str
    execution: str = "MANUAL — no broker connected yet"
    currency: str = "USD"
    notional_inr: float = 0.0  # set for india items; native INR, no FX conversion needed


# A LIMIT order this far above the last price gives the order room to fill on a normal
# day's wiggle without chasing a fast-moving name — newbie-safe default, not a forecast.
LIMIT_BUFFER = 0.005


def _quant_item(p: dict) -> ActionItem | None:
    weight = p.get("weight", 0) or 0
    if weight <= 0:
        return None
    price = p.get("entry_price", 0) or 0
    notional = p.get("notional", 0) or 0
    qty = int(notional // price) if price > 0 else 0
    if qty <= 0:
        return None
    return ActionItem(
        sleeve="quant", symbol=p["symbol"], action="BUY/HOLD",
        order_type="LIMIT", limit_price=round(price * (1 + LIMIT_BUFFER), 2),
        quantity=qty, notional_usd=round(qty * price, 2),
        stop_loss_price=p.get("stop_price"),
        reason=(f"Historically similar setups for this stock went on to average "
               f"{p.get('expectancy', 0):+.1%} over the next ~4 weeks, with roughly "
               f"{p.get('confidence', 0):.0%} of those cases ending positive."),
    )


def _aggressive_item(p: dict) -> ActionItem | None:
    price = p.get("entry_price", 0) or 0
    if p.get("in_position"):
        conviction = p.get("conviction")
        if conviction is not None:
            # Phase 4b gated dip re-entry — smaller, explicitly flagged as unproven.
            notional = p.get("notional", 0) or 0
            qty = int(notional // price) if price > 0 else 0
            if qty <= 0:
                return None
            return ActionItem(
                sleeve="aggressive", symbol=p["symbol"], action="BUY/HOLD (smaller, cautious size)",
                order_type="LIMIT", limit_price=round(price * (1 + LIMIT_BUFFER), 2),
                quantity=qty, notional_usd=round(qty * price, 2),
                stop_loss_price=p.get("stop_price") or None,
                reason=("This is a fast 'buy-the-dip' signal that hasn't built a track "
                       f"record yet — confidence score {conviction:.0%}. Sized smaller "
                       "than a normal trend-confirmed buy on purpose. Watch this one closely."),
            )
        notional = p.get("notional", 0) or 0
        qty = int(notional // price) if price > 0 else 0
        if qty <= 0:
            return None
        return ActionItem(
            sleeve="aggressive", symbol=p["symbol"], action="BUY/HOLD",
            order_type="LIMIT", limit_price=round(price * (1 + LIMIT_BUFFER), 2),
            quantity=qty, notional_usd=round(qty * price, 2),
            stop_loss_price=p.get("stop_price") or None,
            reason="This stock's price trend is intact (above its long-term average, "
                  "which is itself still rising). Full position for this fast-moving book.",
        )
    return ActionItem(
        sleeve="aggressive", symbol=p["symbol"],
        action="EXIT IF HELD / AVOID NEW BUYS", order_type="MARKET", limit_price=None,
        quantity=0, notional_usd=0.0, stop_loss_price=None,
        reason=p.get("reason", "Trend has broken down — this book exits losers fast on "
                              "purpose, since it accepts large drawdowns to chase upside."),
    )


def _india_item(p: dict) -> ActionItem | None:
    weight = p.get("weight", 0) or 0
    if weight <= 0:
        return None
    price = p.get("entry_price", 0) or 0
    notional = p.get("notional", 0) or 0
    qty = int(notional // price) if price > 0 else 0
    if qty <= 0:
        return None
    return ActionItem(
        sleeve="india", symbol=p["symbol"], action="BUY/HOLD",
        order_type="LIMIT", limit_price=round(price * (1 + LIMIT_BUFFER), 2),
        quantity=qty, notional_usd=0.0, stop_loss_price=p.get("stop_price"),
        reason=p.get("reason", "Technical screen rank (momentum + trend health) — no "
                              "backtested win-rate exists for Indian stocks yet."),
        currency="INR", notional_inr=round(qty * price, 2),
    )


def build_action_items(quant_positions: list[dict], aggressive_positions: list[dict],
                       india_positions: list[dict] | None = None) -> list[ActionItem]:
    """`quant_positions`/`aggressive_positions`/`india_positions` are each sleeve's
    `top_positions` array already stored by its snapshot job. Returns BUY/HOLD items first
    (quant, then aggressive, then india), then EXIT/AVOID items last — the things to
    actually go do, before the things to just be aware of. `india_positions` defaults to
    None/[] so existing callers (and tests) that only know about the two US sleeves keep
    working unchanged."""
    quant_items = [it for it in (_quant_item(p) for p in quant_positions) if it]
    india_items = [it for it in (_india_item(p) for p in (india_positions or [])) if it]
    agg_buy = [it for it in (_aggressive_item(p) for p in aggressive_positions) if it]
    actionable = [it for it in agg_buy if it.action != "EXIT IF HELD / AVOID NEW BUYS"]
    avoid = [it for it in agg_buy if it.action == "EXIT IF HELD / AVOID NEW BUYS"]
    return quant_items + india_items + actionable + avoid
