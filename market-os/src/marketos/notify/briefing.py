"""Briefing content — the SAME decision logic as the dashboard's Action Plan page
(`portfolio/action_plan.py`), reformatted as plain text for a push notification instead of
a Streamlit page. Deliberately does not duplicate any decision logic; `build_action_items`
is the single source of truth for "what to do" — this module only renders it differently.
"""
from __future__ import annotations

from marketos.portfolio.action_plan import ActionItem, build_action_items


def _load_top_positions(store, strategy_name: str) -> list[dict]:
    rows = store.get_portfolio_history(strategy_name, limit=1)
    return (rows[0].get("top_positions") or []) if rows else []


def _format_item(it: ActionItem) -> str:
    lines = [f"*{it.symbol}* — {it.action}"]
    if "BUY" in it.action:
        qty_line = f"{it.order_type} {it.quantity} shares"
        if it.limit_price:
            qty_line += f" @ ₹{it.limit_price:,.2f}" if it.currency == "INR" else f" @ ${it.limit_price:,.2f}"
        lines.append(qty_line)
        if it.stop_loss_price:
            stop_str = f"₹{it.stop_loss_price:,.2f}" if it.currency == "INR" else f"${it.stop_loss_price:,.2f}"
            lines.append(f"Stop-loss: {stop_str}")
        cost = it.notional_inr if it.currency == "INR" else it.notional_usd
        cost_str = f"₹{cost:,.0f}" if it.currency == "INR" else f"${cost:,.0f}"
        lines.append(f"Cost: ~{cost_str}")
    lines.append(f"_{it.reason}_")
    lines.append(f"({it.execution})")
    return "\n".join(lines)


def format_briefing(items: list[ActionItem], *, title: str) -> str:
    """Plain-text/Markdown briefing body. Returns a "nothing actionable" message rather
    than an empty string when `items` is empty — silence on a scheduled briefing reads as
    "did this even run," not "no signal today."""
    if not items:
        return f"*{title}*\n\nNo actionable items today — either no snapshot has run "\
               f"yet, or nothing currently clears the bar. This is a normal outcome, not "\
               f"an error; absence of a signal is itself information."

    # startswith(), NOT substring-contains — "EXIT IF HELD / AVOID NEW BUYS" contains the
    # literal substring "BUY" (inside "NEW BUYS"); see dashboard/pages/1_Action_Plan.py
    # for the identical bug this caught there too.
    buys = [it for it in items if it.action.startswith("BUY")]
    avoids = [it for it in items if not it.action.startswith("BUY")]
    sections = [f"*{title}*\n"]
    if buys:
        sections.append(f"✅ *Buy / Hold ({len(buys)})*\n")
        sections.extend(_format_item(it) + "\n" for it in buys)
    if avoids:
        sections.append(f"⛔ *Exit if held / avoid ({len(avoids)})*\n")
        sections.extend(_format_item(it) + "\n" for it in avoids)
    sections.append("⚠️ No automatic order placement is connected. Every item above must "
                    "be placed manually.")
    return "\n".join(sections)


def build_india_morning_briefing(store) -> str:
    """India sleeve only — sent before NSE's 9:15 AM IST open."""
    india_positions = _load_top_positions(store, "india_sleeve")
    items = [it for it in build_action_items([], [], india_positions) if it.sleeve == "india"]
    return format_briefing(items, title="🇮🇳 India morning briefing")


def build_us_evening_briefing(store) -> str:
    """US sleeves (quant + aggressive) only — sent around the US market's 9:30 AM ET open."""
    quant_positions = _load_top_positions(store, "quant_sleeve")
    aggressive_positions = _load_top_positions(store, "aggressive_sleeve")
    items = [it for it in build_action_items(quant_positions, aggressive_positions)
            if it.sleeve in ("quant", "aggressive")]
    return format_briefing(items, title="🇺🇸 US evening briefing")
