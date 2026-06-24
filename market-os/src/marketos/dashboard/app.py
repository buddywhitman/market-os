"""PM cockpit — the daily decision surface.

Reads the QUANT sleeve's real daily snapshot from `marketos.portfolio_attribution`
(written by `pipelines.scheduler.pm_snapshot_job`, fed by `portfolio.opportunities` from
the analog engine's validated, out-of-sample evidence — see that module's docstring for
the exact field mapping). If no live snapshot exists yet (job hasn't run, or Postgres is
unreachable), falls back to clearly-labeled DEMO data rather than silently showing
fabricated numbers as if they were real. Run with `make dashboard`.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:  # allow import without streamlit installed
    st = None

from marketos.utils.fx import get_usdinr_rate, usd_to_inr


def _add_inr_columns(df: pd.DataFrame, fx, usd_cols: tuple[str, ...]) -> pd.DataFrame:
    """Add a `_inr` column next to each USD column, for comprehension only. Every
    instrument here is USD-denominated and the actual order (manual or auto) is placed in
    USD — these columns answer 'roughly how many rupees is that' for a newbie, they are
    NOT what you type into a broker. The Action Plan page says this explicitly too."""
    out = df.copy()
    for col in usd_cols:
        if col in out.columns:
            out[f"{col}_inr"] = out[col].apply(
                lambda v: round(usd_to_inr(v, fx), 0) if pd.notna(v) else v)
    return out


def _demo_opportunities() -> pd.DataFrame:
    """Clearly-labeled fallback shape, used only when no live snapshot is reachable."""
    rng = np.random.default_rng(7)
    syms = ["NVDA", "GEV", "CCJ", "LMT", "RKLB", "AVGO", "PLTR", "VST"]
    themes = ["AI_SEMI", "POWER", "NUCLEAR", "DEFENSE", "SPACE", "AI_SEMI", "AI_SEMI", "POWER"]
    df = pd.DataFrame({
        "symbol": syms,
        "theme": themes,
        "weight": rng.uniform(0.02, 0.10, len(syms)).round(4),
        "expectancy": rng.uniform(0.005, 0.04, len(syms)).round(4),
        "confidence": rng.uniform(0.42, 0.63, len(syms)).round(3),
        "sample_size": rng.uniform(5, 40, len(syms)).round(1),
        "notional": rng.uniform(2000, 9000, len(syms)).round(0),
        "stop_price": rng.uniform(50, 900, len(syms)).round(2),
        "entry_price": rng.uniform(60, 1000, len(syms)).round(2),
    })
    return df.sort_values("weight", ascending=False)


def _load_live_snapshot() -> tuple[pd.DataFrame, dict] | None:
    """Read the most recent quant_sleeve row from Postgres. Returns None (not an empty
    frame) on any failure — DSN unset, DB unreachable, table empty, or job never run —
    so the caller can fall back to demo data with an honest label instead of an error."""
    dsn = os.environ.get("POSTGRES_DSN", "")
    if not dsn:
        return None
    try:
        from marketos.db.store import MarketosStore
        store = MarketosStore(dsn)
        rows = store.get_portfolio_history("quant_sleeve", limit=1)
        if not rows:
            return None
        row = rows[0]
        positions = row.get("top_positions") or []
        if not positions:
            return None
        df = pd.DataFrame(positions)
        # The metrics/detail sections below assume this exact column set. A stored snapshot
        # missing one (e.g. an older row written before a field was added) should render
        # with that metric blank, not crash the whole page — this bit us once already
        # (sample_size was omitted from opportunities.py's payload).
        for col in ("symbol", "weight", "notional", "stop_price", "entry_price",
                   "expectancy", "confidence", "sample_size"):
            if col not in df.columns:
                df[col] = float("nan")
        df["theme"] = df["symbol"].map(_sector_map())
        meta = {
            "date": row.get("date"), "gross_exposure": row.get("gross_exposure"),
            "cash_weight": row.get("cash_weight"), "effective_n": row.get("effective_n"),
            "regime_snapshot": row.get("regime_snapshot") or {},
        }
        return df, meta
    except Exception:
        return None


def _styled_or_plain(df: pd.DataFrame):
    """Color-gradient the table if matplotlib is available; otherwise the plain frame.

    `Styler.background_gradient(...)` only REGISTERS a pending style function — pandas
    doesn't actually run it (and doesn't import matplotlib) until `Styler._compute()` is
    called. Streamlit's `st.dataframe()` triggers that compute internally, so a try/except
    around the `.background_gradient()` call itself never sees the ImportError; it surfaces
    several stack frames later, inside Streamlit's marshalling code. Forcing `._compute()`
    here — the exact call visible in the original traceback — makes the failure happen
    inside OUR try/except instead, where it's actually catchable. This was the bug that
    shipped uncaught: trapping the call that builds the styler caught nothing.

    Catches AttributeError too: `.style` itself raises that (not ImportError) when jinja2
    is missing — a DIFFERENT optional dependency than matplotlib, surfaced by actually
    running this without jinja2 installed locally, not by reasoning about the one reported
    bug. Both are "optional styling dependency absent," same fallback applies."""
    try:
        styler = df.style.background_gradient(subset=["weight", "expectancy"], cmap="Greens")
        styler._compute()
        return styler
    except (ImportError, AttributeError):
        return df


def _sector_map() -> dict[str, str]:
    try:
        from marketos.data.fetchers.orchestrator import SECTOR_MAP
        return SECTOR_MAP
    except Exception:
        return {}


def main() -> None:
    if st is None:
        print("streamlit not installed; run: pip install -e '.[dash]'")
        return
    st.set_page_config(page_title="market-os PM cockpit", layout="wide")
    st.title("market-os — PM cockpit")
    st.caption("Distributions and expectancy, not point forecasts. Every number carries its sample size.")

    fx = get_usdinr_rate()
    if fx.is_live:
        st.caption(f"💱 USD/INR ≈ ₹{fx.rate:.2f} (live) — INR figures below are for "
                  f"reference only. Every instrument is USD-denominated; orders are "
                  f"placed in USD regardless of how funding reaches the broker.")
    else:
        st.warning(f"⚠️ Could not fetch a live USD/INR rate — showing a STALE approximate "
                  f"rate (₹{fx.rate:.2f}/USD) for INR figures below. Check a live rate "
                  f"before trusting any ₹ amount on this page.")

    live = _load_live_snapshot()
    if live is not None:
        df, meta = live
        st.success(f"Live snapshot · {meta['date']} · gross={meta['gross_exposure']:.0%} · "
                  f"cash={meta['cash_weight']:.0%} · effective_n={meta['effective_n']:.1f}")
        regime = meta.get("regime_snapshot", {})
        if regime:
            st.caption(f"Regime: {regime}")
    else:
        df = _demo_opportunities()
        st.warning("No live pm_snapshot found (job hasn't run or Postgres unreachable) — "
                  "showing DEMO data, not real positions.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Positions", len(df))
    c2.metric("Median expectancy", f"{df['expectancy'].median():.3f}")
    c3.metric("Median confidence", f"{df['confidence'].median():.0%}")
    c4.metric("Median sample size", f"{df['sample_size'].median():.1f}")

    st.subheader("Sized positions")
    st.dataframe(_styled_or_plain(_add_inr_columns(df, fx, ("notional", "entry_price", "stop_price"))),
                use_container_width=True)

    st.subheader("Position detail")
    pick = st.selectbox("Symbol", df["symbol"].tolist())
    row = df[df["symbol"] == pick].iloc[0]
    a, b = st.columns(2)
    with a:
        st.write(f"**Theme:** {row['theme']}")
        st.write(f"**Expectancy (analog fwd 20d):** {row['expectancy']:.4f}  "
                f"(effective n={row['sample_size']:.1f})")
        st.write(f"**Confidence (analog win-rate, regime-discounted):** {row['confidence']:.0%}")
        # Escape every literal "$" — Streamlit's markdown renderer treats unescaped pairs
        # of "$" as LaTeX math-mode delimiters, so "$X ... **bold** ... $Y" silently eats
        # the bold markdown between them (this is the exact bug that shipped: "Entry:"
        # rendered bold because it sat before the first "$", but "**Stop:**" landed INSIDE
        # the math span between two unescaped "$" and rendered as literal asterisks).
        st.write(f"**Weight:** {row['weight']:.1%}  ·  **Notional:** \\${row['notional']:,.0f} "
                f"(≈ ₹{usd_to_inr(row['notional'], fx):,.0f})")
    with b:
        st.write(f"**Entry:** \\${row['entry_price']:,.2f} (≈ ₹{usd_to_inr(row['entry_price'], fx):,.0f})"
                f"  ·  **Stop:** \\${row['stop_price']:,.2f} "
                f"(≈ ₹{usd_to_inr(row['stop_price'], fx):,.0f})")
        st.write("**Feature attribution (SHAP):** wire in models.alpha_model → SHAP here.")
        st.write("**Catalysts:** Phase 4b — ground TradingAgents on this snapshot for "
                "fundamental/sentiment context, not shown yet.")
    try:
        st.page_link("pages/1_Action_Plan.py",
                     label="→ Go to the Action Plan page for exact order instructions", icon="📋")
    except Exception:
        st.caption("→ See the 'Action Plan' page in the sidebar for exact order instructions.")

    _render_aggressive_sleeve()
    _render_screened_candidates()


def _render_aggressive_sleeve() -> None:
    """The AGGRESSIVE sleeve's live circuit-breaker status — a SEPARATE row
    (strategy_name='aggressive_sleeve') from the quant snapshot above, by design (the
    capital ring-fence). Shows every name in the sleeve, in or out, with the reason —
    visibility into why a name is flat matters as much as what's currently held."""
    st.divider()
    st.subheader("Aggressive sleeve — live circuit-breaker status")
    st.caption("Wave-riding book. Willing to lose all. LOCKED policy: fully deployed while "
              "trend is intact, exits only on a confirmed sustained downtrend (price below "
              "a falling 200DMA). No conviction-gated dip re-entry yet (Phase 4b).")
    dsn = os.environ.get("POSTGRES_DSN", "")
    if not dsn:
        st.caption("No POSTGRES_DSN — aggressive sleeve status unavailable.")
        return
    try:
        from marketos.db.store import MarketosStore
        store = MarketosStore(dsn)
        rows = store.get_portfolio_history("aggressive_sleeve", limit=1)
    except Exception:
        rows = []
    if not rows:
        st.caption("No aggressive_snapshot run yet.")
        return
    row = rows[0]
    positions = row.get("top_positions") or []
    if not positions:
        st.caption("Aggressive snapshot ran but produced no rows.")
        return
    df = pd.DataFrame(positions)
    # Same defensive backfill as _load_live_snapshot, for the same reason: a stored row
    # whose shape doesn't exactly match what this section expects must degrade, not
    # KeyError. Caught by this module's OWN test suite before it ever shipped.
    for col in ("symbol", "in_position", "weight", "notional", "stop_price",
               "entry_price", "reason", "conviction"):
        if col not in df.columns:
            df[col] = False if col == "in_position" else float("nan")
    n_in = int(df["in_position"].sum()) if df["in_position"].notna().any() else 0
    st.caption(f"{row.get('date')} · {n_in}/{len(df)} names IN · "
              f"gross={row.get('gross_exposure', 0):.0%}")
    st.dataframe(df, use_container_width=True)


def _render_screened_candidates() -> None:
    """The BROAD candidate universe's daily technical screen — deliberately separate from
    the validated 15 positions above. These are NOT sized, NOT in the quant cross-section,
    and have NOT earned capital — they're visibility into what else is moving, pending a
    deliberate promotion decision into UNIVERSE (see features/screening.py)."""
    st.divider()
    st.subheader("Candidate universe — daily technical screen (not yet validated)")
    dsn = os.environ.get("POSTGRES_DSN", "")
    if not dsn:
        st.caption("No POSTGRES_DSN — candidate screen unavailable.")
        return
    try:
        from marketos.db.store import MarketosStore
        store = MarketosStore(dsn)
        screen = store.get_latest_family("_screen", "screen")
    except Exception:
        screen = {}
    candidates = screen.get("top_candidates") if screen else None
    if not candidates:
        st.caption("No screen run yet — `screen_universe_job` hasn't fired.")
        return
    st.caption(f"{screen.get('n_screened', 0)} candidates screened, "
              f"{screen.get('n_passed_liquidity', 0)} passed the liquidity gate. "
              "Ranked by 63d momentum + trend health, yfinance-only (no fundamentals/sentiment).")
    st.dataframe(pd.DataFrame(candidates), use_container_width=True)


if __name__ == "__main__":
    main()
