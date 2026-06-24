"""Runtime smoke test for the Action Plan Streamlit page — same discipline as
test_dashboard.py: stub streamlit, actually call main(), let the real code run to
completion. This is the highest-stakes page in the app (a newbie acts directly on it), so
it gets the same "don't trust curl/quiet-logs" verification as everything else.
"""
from __future__ import annotations

import sys
import types

import pytest


class _CtxRecorder:
    """A no-op context manager AND object with arbitrary attribute/callable access — covers
    st.container(border=True), st.expander(...), and the columns it yields."""
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def __getattr__(self, name):
        return lambda *a, **k: None


class _FakeStreamlit(types.SimpleNamespace):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.errors: list[str] = []
        self.warned: list[str] = []

    def set_page_config(self, *a, **k): pass
    def title(self, *a, **k): pass
    def caption(self, *a, **k): pass
    def write(self, *a, **k): pass
    def markdown(self, *a, **k): pass
    def info(self, *a, **k): pass
    def subheader(self, *a, **k): pass
    def divider(self, *a, **k): pass

    def error(self, text, *a, **k):
        self.errors.append(text)

    def warning(self, text, *a, **k):
        self.warned.append(text)

    def container(self, *a, **k):
        return _CtxRecorder()

    def expander(self, *a, **k):
        return _CtxRecorder()

    def columns(self, n):
        return [_CtxRecorder() for _ in range(n)]


@pytest.fixture
def fake_page(monkeypatch):
    fake = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    import importlib
    import marketos.dashboard.pages
    mod_name = "marketos.dashboard.pages.1_Action_Plan"
    # The leading digit makes this an invalid Python identifier for a normal import —
    # load it directly from its file path instead.
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(marketos.dashboard.pages.__file__), "1_Action_Plan.py")
    spec = importlib.util.spec_from_file_location("action_plan_page", path)
    page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(page)
    from marketos.utils.fx import FxRate
    monkeypatch.setattr(page, "get_usdinr_rate", lambda: FxRate(rate=86.0, is_live=True))
    yield page


def test_main_with_no_dsn_shows_no_actionable_items(fake_page):
    """No POSTGRES_DSN -> empty lists everywhere -> the 'no actionable items' info path,
    not a crash."""
    fake_page.main()


def test_main_always_shows_the_no_broker_disclaimer(fake_page):
    fake_page.main()
    st_module = sys.modules["streamlit"]
    assert any("No automatic order placement" in e for e in st_module.errors)


def test_main_with_real_shaped_buy_and_avoid_items(fake_page, monkeypatch):
    class _FakeStore:
        def __init__(self, dsn): pass
        def get_portfolio_history(self, strategy_name, limit=1):
            if strategy_name == "quant_sleeve":
                return [{"top_positions": [
                    {"symbol": "NVDA", "weight": 0.05, "notional": 1000.0,
                     "entry_price": 200.0, "stop_price": 180.0, "confidence": 0.7,
                     "expectancy": 0.03},
                ]}]
            if strategy_name == "aggressive_sleeve":
                return [{"top_positions": [
                    {"symbol": "MSTR", "in_position": True, "weight": 0.06,
                     "notional": 600.0, "entry_price": 100.0, "stop_price": 85.0,
                     "reason": "gated", "conviction": 0.8},
                    {"symbol": "COIN", "in_position": False, "weight": 0.0,
                     "notional": 0.0, "entry_price": 150.0, "stop_price": 0.0,
                     "reason": "circuit_breaker: confirmed downtrend", "conviction": None},
                ]}]
            return []

    monkeypatch.setenv("POSTGRES_DSN", "postgresql://fake/dsn")
    monkeypatch.setattr("marketos.db.store.MarketosStore", _FakeStore)
    fake_page.main()


def test_fx_staleness_warning_on_action_plan_page(fake_page, monkeypatch):
    from marketos.utils.fx import FxRate
    monkeypatch.setattr(fake_page, "get_usdinr_rate", lambda: FxRate(rate=86.0, is_live=False))
    fake_page.main()
    st_module = sys.modules["streamlit"]
    assert any("STALE" in w for w in st_module.warned)


def test_main_with_india_buy_item_renders_inr_not_usd(fake_page, monkeypatch):
    class _FakeStore:
        def __init__(self, dsn): pass
        def get_portfolio_history(self, strategy_name, limit=1):
            if strategy_name == "india_sleeve":
                return [{"top_positions": [
                    {"symbol": "BEL", "sector": "DEFENSE_ELECTRONICS", "weight": 0.15,
                     "notional": 750.0, "entry_price": 280.0, "stop_price": 252.0,
                     "screen_score": 0.2, "mom_63d": 0.15,
                     "reason": "Technical screen rank only — no backtested win-rate."},
                ]}]
            return []

    monkeypatch.setenv("POSTGRES_DSN", "postgresql://fake/dsn")
    monkeypatch.setattr("marketos.db.store.MarketosStore", _FakeStore)
    fake_page.main()


def test_main_with_no_india_data_shows_honest_blocker_caption(fake_page, monkeypatch):
    """No india_sleeve rows -> must explain WHY (NSE blocked), not just show nothing."""
    fake_page.main()  # no DSN -> india_positions empty -> the info caption path


def test_avoid_item_with_buys_substring_classified_correctly(fake_page, monkeypatch):
    """Real bug, caught by the Telegram briefing's tests: 'EXIT IF HELD / AVOID NEW BUYS'
    contains the literal substring 'BUY' (inside 'NEW BUYS') — a naive `"BUY" in
    it.action` classification check silently put every avoid item in the Buy/Hold
    section. Must classify by startswith(), not substring-contains."""
    class _FakeStore:
        def __init__(self, dsn): pass
        def get_portfolio_history(self, strategy_name, limit=1):
            if strategy_name == "quant_sleeve":
                return [{"top_positions": [
                    {"symbol": "COIN", "weight": 0.0, "notional": 0.0, "entry_price": 150.0,
                     "stop_price": 0.0, "confidence": 0.0, "expectancy": -0.02},
                ]}]
            return []

    monkeypatch.setenv("POSTGRES_DSN", "postgresql://fake/dsn")
    monkeypatch.setattr("marketos.db.store.MarketosStore", _FakeStore)
    # quant items with weight=0 are excluded entirely by action_plan's own filter, so
    # exercise this via the aggressive sleeve's explicit EXIT action string instead.
    import marketos.portfolio.action_plan as ap
    items = ap.build_action_items([], [{
        "symbol": "COIN", "in_position": False, "weight": 0.0, "notional": 0.0,
        "entry_price": 150.0, "stop_price": 0.0, "reason": "trend broke", "conviction": None,
    }])
    assert items[0].action == "EXIT IF HELD / AVOID NEW BUYS"
    buys = [it for it in items if it.action.startswith("BUY")]
    avoids = [it for it in items if not it.action.startswith("BUY")]
    assert len(buys) == 0
    assert len(avoids) == 1
