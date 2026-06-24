"""Runtime smoke test for the dashboard — exercises `main()` end-to-end, not just imports.

Two real bugs shipped past a code read and past curl-based manual checks because neither
actually executes the Streamlit script body: a missing `sample_size` key (KeyError) and an
optional-dependency failure in `.style.background_gradient()` (ImportError/AttributeError,
depending on which of matplotlib/jinja2 is absent). Streamlit only runs `main()` inside a
live websocket session — a plain HTTP GET to "/" returns the static shell and proves
nothing. This test stubs every `st.*` call used by the dashboard with a no-op recorder and
actually calls `main()`, so it fails the same way the real app would, locally, before any
deploy.
"""
from __future__ import annotations

import sys
import types

import pandas as pd
import pytest


class _FakeColumn:
    def metric(self, *a, **k):
        pass

    def write(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeStreamlit(types.SimpleNamespace):
    """Records nothing, raises nothing — the point is to let the REAL pandas/python code
    in main() run to completion, so bugs in df construction / f-string formatting / styler
    computation surface exactly as they would under the real Streamlit runtime."""

    def set_page_config(self, *a, **k): pass
    def title(self, *a, **k): pass
    def caption(self, *a, **k): pass
    def success(self, *a, **k): pass
    def info(self, *a, **k): pass
    def subheader(self, *a, **k): pass
    def divider(self, *a, **k): pass

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.written: list[str] = []
        self.warned: list[str] = []

    def write(self, text, *a, **k):
        self.written.append(text)

    def warning(self, text, *a, **k):
        self.warned.append(text)

    def columns(self, n):
        return [_FakeColumn() for _ in range(n)]

    def dataframe(self, data, **k):
        # Force any lazy pandas Styler computation NOW, same as Streamlit's real
        # marshalling does internally — this is exactly where both real bugs surfaced.
        if hasattr(data, "_compute"):
            data._compute()

    def page_link(self, *a, **k): pass

    def selectbox(self, label, options, **k):
        return options[0]


@pytest.fixture
def fake_streamlit(monkeypatch):
    fake = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    # Deterministic by default — tests that care about the candidate-screen section set
    # POSTGRES_DSN explicitly. Without this, a real .env in the test environment could
    # make _render_screened_candidates() try a real DB connection during unrelated tests.
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    # dashboard.app imports streamlit at module load time into a local `st` binding —
    # reload so it picks up the fake module instead of the cached real/absent one.
    import importlib
    import marketos.dashboard.app as app
    importlib.reload(app)
    # Deterministic + no real network call: locally there's no yfinance so this already
    # falls back fast, but on a server WITH yfinance installed this would otherwise hit
    # the network on every test run. Tests dedicated to the FX banner override this.
    from marketos.utils.fx import FxRate
    monkeypatch.setattr(app, "get_usdinr_rate", lambda: FxRate(rate=86.0, is_live=True))
    yield app


def test_main_runs_against_demo_fallback(fake_streamlit, monkeypatch):
    """No live snapshot reachable -> demo path. Must not raise."""
    monkeypatch.setattr(fake_streamlit, "_load_live_snapshot", lambda: None)
    fake_streamlit.main()  # raises on any of: KeyError, ImportError, AttributeError, ...


def test_main_runs_against_live_snapshot_shape(fake_streamlit, monkeypatch):
    """The exact shape `opportunities.snapshot_to_attribution` produces — every column the
    live path's f-strings/metrics touch must be present, or this must catch it, not main()."""
    live_df = pd.DataFrame([
        {"symbol": "NVDA", "weight": 0.011, "notional": 1100.0, "stop_price": 188.0,
         "entry_price": 208.0, "expectancy": 0.0266, "confidence": 0.56, "sample_size": 15.3,
         "theme": "AI_SEMI"},
        {"symbol": "LMT", "weight": 0.02, "notional": 1998.0, "stop_price": 458.0,
         "entry_price": 493.0, "expectancy": 0.062, "confidence": 0.76, "sample_size": 13.9,
         "theme": "DEFENSE"},
    ])
    meta = {"date": "2026-06-23", "gross_exposure": 0.187, "cash_weight": 0.813,
            "effective_n": 10.7, "regime_snapshot": {"SPY": 0}}
    monkeypatch.setattr(fake_streamlit, "_load_live_snapshot", lambda: (live_df, meta))
    fake_streamlit.main()

    # Regression guard for the exact rendering bug that shipped: Streamlit's markdown
    # renderer treats an unescaped pair of "$" as LaTeX math-mode delimiters, so
    # "$X ... **bold** ... $Y" silently eats the bold markdown between the two "$"s.
    # Every literal dollar sign written must be escaped ("\$"), never bare.
    st_module = sys.modules["streamlit"]
    dollar_lines = [t for t in st_module.written if "$" in t]
    assert dollar_lines, "expected at least one written line containing a dollar amount"
    for line in dollar_lines:
        bare = line.replace("\\$", "")  # drop escaped ones, anything left is a bare "$"
        assert "$" not in bare, f"unescaped '$' will eat markdown as LaTeX math-mode: {line!r}"


def test_live_snapshot_missing_a_column_does_not_crash_main(fake_streamlit, monkeypatch):
    """Regression guard for the exact bug that shipped: a stored snapshot missing one
    expected column (e.g. written before a field existed) must degrade, not KeyError.

    Patches at the STORE seam, not `_load_live_snapshot` itself — the column-backfill
    logic this test exists to verify lives INSIDE `_load_live_snapshot`. Stubbing that
    function out (as the other tests do) would skip the exact code under test and pass
    even if the backfill were deleted entirely.
    """
    class _FakeStore:
        def __init__(self, dsn):
            pass

        def get_portfolio_history(self, strategy_name, limit=1):
            return [{
                "date": "2026-06-23", "gross_exposure": 0.01, "cash_weight": 0.99,
                "effective_n": 1.0, "regime_snapshot": {},
                "top_positions": [
                    {"symbol": "NVDA", "weight": 0.011, "notional": 1100.0,
                     "stop_price": 188.0, "entry_price": 208.0, "expectancy": 0.0266,
                     "confidence": 0.56},
                    # sample_size deliberately omitted — the exact shape that shipped broken
                ],
            }]

    monkeypatch.setenv("POSTGRES_DSN", "postgresql://fake/dsn")
    monkeypatch.setattr("marketos.db.store.MarketosStore", _FakeStore)
    fake_streamlit.main()


def test_styled_or_plain_falls_back_without_matplotlib(fake_streamlit, monkeypatch):
    """Force the ImportError/AttributeError path directly, independent of whatever
    optional deps happen to be installed in whichever environment runs this test."""
    df = pd.DataFrame({"weight": [0.1, 0.2], "expectancy": [0.01, 0.02], "symbol": ["A", "B"]})

    class _BoomStyler:
        def _compute(self):
            raise ImportError("matplotlib required")

    class _BoomDF(pd.DataFrame):
        @property
        def style(self):
            return types.SimpleNamespace(
                background_gradient=lambda **k: _BoomStyler())

    result = fake_streamlit._styled_or_plain(_BoomDF(df))
    assert result is not None  # fell back to *something* renderable, didn't propagate


def test_screened_candidates_section_with_no_dsn_does_not_crash(fake_streamlit, monkeypatch):
    """No POSTGRES_DSN -> the new candidate-screen section must degrade silently, not
    attempt a real DB connection or raise."""
    monkeypatch.setattr(fake_streamlit, "_load_live_snapshot", lambda: None)
    fake_streamlit.main()


def test_screened_candidates_section_renders_real_shape(fake_streamlit, monkeypatch):
    """The exact shape `screen_universe_job` writes — top_candidates list of dicts with
    symbol/sector/screen_score/mom_63d/trend_healthy. Must render without raising."""
    class _FakeStore:
        def __init__(self, dsn): pass
        def get_portfolio_history(self, strategy_name, limit=1):
            return []  # forces demo fallback for the main snapshot section

        def get_latest_family(self, symbol, family):
            assert (symbol, family) == ("_screen", "screen")
            return {
                "n_screened": 70, "n_passed_liquidity": 42,
                "top_candidates": [
                    {"symbol": "MU", "sector": "MEMORY", "screen_score": 0.31,
                     "mom_63d": 0.26, "trend_healthy": True},
                    {"symbol": "TSM", "sector": "OTHER_SEMI", "screen_score": 0.18,
                     "mom_63d": 0.13, "trend_healthy": True},
                ],
            }

    monkeypatch.setenv("POSTGRES_DSN", "postgresql://fake/dsn")
    monkeypatch.setattr("marketos.db.store.MarketosStore", _FakeStore)
    fake_streamlit.main()


def test_aggressive_sleeve_section_with_no_dsn_does_not_crash(fake_streamlit, monkeypatch):
    monkeypatch.setattr(fake_streamlit, "_load_live_snapshot", lambda: None)
    fake_streamlit.main()


def test_aggressive_sleeve_section_renders_real_shape(fake_streamlit, monkeypatch):
    """The exact shape `aggressive_snapshot.snapshot_to_attribution` writes — a mix of
    in_position=True (sized) and in_position=False (flat, with a reason) rows."""
    class _FakeStore:
        def __init__(self, dsn): pass

        def get_portfolio_history(self, strategy_name, limit=1):
            if strategy_name == "aggressive_sleeve":
                return [{
                    "date": "2026-06-23", "gross_exposure": 0.45,
                    "top_positions": [
                        {"symbol": "NVDA", "in_position": True, "weight": 0.40,
                         "notional": 40000.0, "stop_price": 180.0, "entry_price": 208.0,
                         "reason": "circuit_breaker: trend intact, fully deployed"},
                        {"symbol": "SOXL", "in_position": False, "weight": 0.0,
                         "notional": 0.0, "stop_price": 0.0, "entry_price": 22.0,
                         "reason": "circuit_breaker: price below a falling 200DMA "
                                  "(confirmed sustained downtrend)"},
                    ],
                }]
            return []  # quant_sleeve -> demo fallback

        def get_latest_family(self, symbol, family):
            return {}

    monkeypatch.setenv("POSTGRES_DSN", "postgresql://fake/dsn")
    monkeypatch.setattr("marketos.db.store.MarketosStore", _FakeStore)
    fake_streamlit.main()


def test_aggressive_sleeve_section_with_gated_dip_reentry_row(fake_streamlit, monkeypatch):
    """Phase 4b's gated dip-reentry adds a `conviction` field to some rows but not others
    (only set when the gate actually fired) — must render without raising either way."""
    class _FakeStore:
        def __init__(self, dsn): pass

        def get_portfolio_history(self, strategy_name, limit=1):
            if strategy_name == "aggressive_sleeve":
                return [{
                    "date": "2026-06-23", "gross_exposure": 0.30,
                    "top_positions": [
                        {"symbol": "NVDA", "in_position": True, "weight": 0.40,
                         "notional": 40000.0, "stop_price": 180.0, "entry_price": 208.0,
                         "reason": "circuit_breaker: trend intact, fully deployed",
                         "conviction": None},
                        {"symbol": "MSTR", "in_position": True, "weight": 0.06,
                         "notional": 6000.0, "stop_price": 95.0, "entry_price": 109.0,
                         "reason": "PHASE 4b GATED DIP RE-ENTRY (conviction=0.8): ...",
                         "conviction": 0.8},
                        {"symbol": "COIN", "in_position": False, "weight": 0.0,
                         "notional": 0.0, "stop_price": 0.0, "entry_price": 200.0,
                         "reason": "circuit_breaker OUT | dip signal fired but gate FAILED "
                                  "(conviction=0.4): ...",
                         "conviction": 0.4},
                    ],
                }]
            return []

        def get_latest_family(self, symbol, family):
            return {}

    monkeypatch.setenv("POSTGRES_DSN", "postgresql://fake/dsn")
    monkeypatch.setattr("marketos.db.store.MarketosStore", _FakeStore)
    fake_streamlit.main()


def test_fx_staleness_warning_shown_when_not_live(fake_streamlit, monkeypatch):
    """If the live USD/INR fetch fails, the page must visibly warn, not silently show a
    fallback number as if it were current — FX rates move daily and training data has a
    cutoff, so a wrong number shown confidently is worse than no number."""
    from marketos.utils.fx import FxRate
    monkeypatch.setattr(fake_streamlit, "get_usdinr_rate", lambda: FxRate(rate=86.0, is_live=False))
    monkeypatch.setattr(fake_streamlit, "_load_live_snapshot", lambda: None)
    fake_streamlit.main()
    st_module = sys.modules["streamlit"]
    assert any("STALE" in w or "stale" in w for w in st_module.warned), (
        f"expected a staleness warning when is_live=False, got: {st_module.warned}")


def test_fx_no_staleness_warning_when_live(fake_streamlit, monkeypatch):
    """Conversely: when the rate IS live, must not falsely warn the user about staleness."""
    from marketos.utils.fx import FxRate
    monkeypatch.setattr(fake_streamlit, "get_usdinr_rate", lambda: FxRate(rate=86.0, is_live=True))
    monkeypatch.setattr(fake_streamlit, "_load_live_snapshot", lambda: None)
    fake_streamlit.main()
    st_module = sys.modules["streamlit"]
    assert not any("STALE" in w or "stale" in w for w in st_module.warned), (
        f"should not warn about staleness when is_live=True, got: {st_module.warned}")
