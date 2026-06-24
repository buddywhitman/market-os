"""Action Plan — exactly what to do today, in plain language, for a newbie investor.

No jargon-first numbers here (those live on the main PM cockpit page). This page only
answers: what to buy/avoid, what order type, how many shares, what price, what stop-loss —
the things you'd actually type into a broker app. Reads the SAME two stored snapshots as
the main page (quant_sleeve + aggressive_sleeve); does not compute anything new.
"""
from __future__ import annotations

import os

import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None

from marketos.portfolio.action_plan import build_action_items
from marketos.utils.fx import get_usdinr_rate, usd_to_inr


def _load_top_positions(strategy_name: str) -> list[dict]:
    dsn = os.environ.get("POSTGRES_DSN", "")
    if not dsn:
        return []
    try:
        from marketos.db.store import MarketosStore
        store = MarketosStore(dsn)
        rows = store.get_portfolio_history(strategy_name, limit=1)
        return (rows[0].get("top_positions") or []) if rows else []
    except Exception:
        return []


def main() -> None:
    if st is None:
        print("streamlit not installed")
        return
    st.set_page_config(page_title="market-os — Action Plan", layout="wide")
    st.title("📋 Action Plan — what to do today")

    st.error(
        "**No automatic order placement is connected to any broker yet.** Every item "
        "below must be placed by you, manually, in USD. ₹ amounts are shown only so you "
        "can sanity-check the size — they are NOT what you type into your broker; the "
        "broker order itself is always in USD."
    )

    fx = get_usdinr_rate()
    if fx.is_live:
        st.caption(f"💱 Reference rate: USD/INR ≈ ₹{fx.rate:.2f} (live)")
    else:
        st.warning(f"⚠️ Could not fetch a live USD/INR rate — ₹ amounts below use a STALE "
                  f"fallback (₹{fx.rate:.2f}/USD). Check a live rate yourself before "
                  f"trusting any ₹ figure on this page.")

    with st.expander("New to investing? Quick glossary (click to expand)"):
        st.write("- **LIMIT order**: you set the highest price you're willing to pay. "
                "Safer than a market order on a fast-moving stock.")
        st.write("- **MARKET order**: buy/sell immediately at whatever the current price is.")
        st.write("- **Stop-loss**: an order that automatically sells if the price falls to "
                "a level you set — your built-in 'cut the loss' safety net.")
        st.write("- **Quantity**: whole shares only here. If your broker supports "
                "fractional shares, you can also just spend the USD amount shown directly.")

    quant_positions = _load_top_positions("quant_sleeve")
    aggressive_positions = _load_top_positions("aggressive_sleeve")
    india_positions = _load_top_positions("india_sleeve")
    items = build_action_items(quant_positions, aggressive_positions, india_positions)

    if not india_positions:
        st.caption("ℹ️ India sleeve: no positions yet. NSE data access is currently "
                  "blocked from the server (a known infrastructure issue, not something "
                  "wrong with your account) — this section activates automatically once "
                  "that's resolved.")

    if not items:
        st.info("No actionable items right now — either no snapshot has run yet, or "
               "nothing currently clears the bar. Check back after the next daily run.")
        return

    # REAL BUG fixed here (caught by a test written for the new Telegram briefing module,
    # which copied this exact pattern — this dashboard page had been shipping with it):
    # "BUY" in it.action is a SUBSTRING check, and "EXIT IF HELD / AVOID NEW BUYS"
    # literally contains "BUY" inside "NEW BUYS" — every avoid item was silently
    # misclassified into the Buy/Hold section. startswith() is exact for the three real
    # action strings ("BUY/HOLD", "BUY/HOLD (smaller...)", "EXIT IF HELD / AVOID NEW BUYS").
    buys = [it for it in items if it.action.startswith("BUY")]
    avoids = [it for it in items if not it.action.startswith("BUY")]

    _SLEEVE_LABELS = {
        "quant": "Mid-term (quant)", "aggressive": "Short-term wave-riding (aggressive)",
        "india": "Real ₹5k growth capital (india, via AngelOne)",
    }

    if buys:
        st.subheader(f"✅ Buy / Hold — {len(buys)} item(s)")
        for it in buys:
            sleeve_label = _SLEEVE_LABELS.get(it.sleeve, it.sleeve)
            with st.container(border=True):
                st.markdown(f"### {it.symbol} — {it.action}")
                st.caption(f"{sleeve_label} · Execution: {it.execution}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Order type", it.order_type)
                c2.metric("Quantity", f"{it.quantity} shares")
                if it.currency == "INR":
                    # India sleeve capital IS INR already — no FX conversion, no USD
                    # figure to show (there isn't one; this is the real currency).
                    if it.limit_price:
                        c3.metric("Limit price", f"₹{it.limit_price:,.2f}")
                    st.write(f"**Total cost:** ≈ ₹{it.notional_inr:,.2f}")
                    if it.stop_loss_price:
                        st.write(f"**Set a stop-loss sell order at:** ₹{it.stop_loss_price:,.2f}")
                else:
                    if it.limit_price:
                        c3.metric("Limit price", f"\\${it.limit_price:,.2f}",
                                 help=f"≈ ₹{usd_to_inr(it.limit_price, fx):,.0f}")
                    st.write(f"**Total cost:** ≈ \\${it.notional_usd:,.2f} "
                            f"(≈ ₹{usd_to_inr(it.notional_usd, fx):,.0f})")
                    if it.stop_loss_price:
                        st.write(f"**Set a stop-loss sell order at:** \\${it.stop_loss_price:,.2f} "
                                f"(≈ ₹{usd_to_inr(it.stop_loss_price, fx):,.0f})")
                st.caption(f"Why: {it.reason}")

    if avoids:
        st.subheader(f"⛔ Exit if held / avoid new buys — {len(avoids)} item(s)")
        for it in avoids:
            with st.container(border=True):
                st.markdown(f"### {it.symbol} — {it.action}")
                st.caption(f"Why: {it.reason}")

    st.divider()
    st.caption("Generated from the same daily snapshots shown on the main PM cockpit page. "
              "Nothing on this page is a forecast or a guarantee — it's a structured "
              "summary of validated historical patterns and current trend status.")


if __name__ == "__main__":
    main()
