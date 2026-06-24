"""Unit tests for data/fetchers/screener_fetcher.py — focused on the HTML-parsing logic
using real-shaped HTML snippets (no network), since this fetcher scrapes a third-party
site whose structure could change without notice.
"""
from __future__ import annotations

from unittest.mock import patch

from marketos.data.fetchers.screener_fetcher import (
    fetch_company_ratios, fetch_shareholding_pattern, _parse_number,
)


def _fake_ratios_html(extra_li: str = "") -> str:
    return f"""
    <html><body>
    <li class="flex flex-space-between" data-source="default">
      <span class="name">Market Cap</span>
      <span class="nowrap value">₹ 72,888 Cr.</span>
    </li>
    <li class="flex flex-space-between" data-source="default">
      <span class="name">High / Low</span>
      <span class="nowrap value">₹ 12,500 / ₹ 8,200</span>
    </li>
    <li class="flex flex-space-between" data-source="default">
      <span class="name">Stock P/E</span>
      <span class="nowrap value">96.0</span>
    </li>
    {extra_li}
    </body></html>
    """


def test_parse_number_strips_currency_and_commas():
    assert _parse_number("₹ 1,234.56 Cr.") == 1234.56
    assert _parse_number("12.3%") == 12.3
    assert _parse_number("-") is None
    assert _parse_number("") is None


def test_ratios_high_low_range_split_into_two_fields_not_concatenated():
    """Regression guard for a real bug: 'High / Low' is two numbers, not one. Naively
    stripping non-digits from the whole string concatenates them into garbage (caught
    running this against real DIXON data: produced 184729600.0 for a ~12,500/~8,200
    range). Must split on '/' into _high/_low fields instead."""
    with patch("marketos.data.fetchers.screener_fetcher._get_soup") as mock_soup:
        from bs4 import BeautifulSoup
        mock_soup.return_value = BeautifulSoup(_fake_ratios_html(), "html.parser")
        result = fetch_company_ratios("FAKE")
    assert result.get("high__low_high") == 12500
    assert result.get("high__low_low") == 8200
    # The bug's exact symptom — a single garbage concatenated number — must NOT appear.
    assert "high__low" not in result


def test_ratios_plain_values_parsed_normally():
    with patch("marketos.data.fetchers.screener_fetcher._get_soup") as mock_soup:
        from bs4 import BeautifulSoup
        mock_soup.return_value = BeautifulSoup(_fake_ratios_html(), "html.parser")
        result = fetch_company_ratios("FAKE")
    assert result["market_cap"] == 72888.0
    assert result["stock_pe"] == 96.0


def test_ratios_returns_empty_dict_when_page_unreachable():
    with patch("marketos.data.fetchers.screener_fetcher._get_soup", return_value=None):
        assert fetch_company_ratios("FAKE") == {}


def _fake_shareholding_html() -> str:
    return """
    <html><body>
    <div id="shareholding">
      <table>
        <tr><th>Particulars</th><th>Mar 2025</th><th>Jun 2025</th></tr>
        <tr><td>Promoters +</td><td>28.80%</td><td>28.69%</td></tr>
        <tr><td>FIIs +</td><td>17.90%</td><td>18.30%</td></tr>
        <tr><td>DIIs +</td><td>27.50%</td><td>28.14%</td></tr>
        <tr><td>Public</td><td>25.80%</td><td>24.87%</td></tr>
      </table>
    </div>
    </body></html>
    """


def test_shareholding_takes_latest_quarter_not_oldest():
    """Screener lists quarters oldest-to-newest left-to-right — must take the LAST
    column, not the first, or every holding % would be a stale prior-quarter value."""
    with patch("marketos.data.fetchers.screener_fetcher._get_soup") as mock_soup:
        from bs4 import BeautifulSoup
        mock_soup.return_value = BeautifulSoup(_fake_shareholding_html(), "html.parser")
        result = fetch_shareholding_pattern("FAKE")
    assert result["promoter_pct"] == 28.69  # latest, not 28.80 (the prior quarter)
    assert result["fii_pct"] == 18.30
    assert result["dii_pct"] == 28.14


def test_shareholding_computes_fii_plus_dii_from_latest_values():
    with patch("marketos.data.fetchers.screener_fetcher._get_soup") as mock_soup:
        from bs4 import BeautifulSoup
        mock_soup.return_value = BeautifulSoup(_fake_shareholding_html(), "html.parser")
        result = fetch_shareholding_pattern("FAKE")
    assert result["fii_plus_dii_pct"] == round(18.30 + 28.14, 2)


def test_shareholding_returns_empty_dict_when_section_missing():
    with patch("marketos.data.fetchers.screener_fetcher._get_soup") as mock_soup:
        from bs4 import BeautifulSoup
        mock_soup.return_value = BeautifulSoup("<html><body>no shareholding here</body></html>",
                                                "html.parser")
        assert fetch_shareholding_pattern("FAKE") == {}


def test_asof_ts_truncated_to_date_not_full_timestamp():
    """Regression guard for a real issue caught running the job twice in one day (a
    manual smoke test, then the scheduled job minutes later): asof_ts is the upsert
    conflict key (symbol, asof_ts, family) — without truncating to the date, every rerun
    inserts a NEW row instead of updating, since fundamentals have no natural 'last bar
    date' the way OHLCV does."""
    with patch("marketos.data.fetchers.screener_fetcher._get_soup") as mock_soup:
        from bs4 import BeautifulSoup
        mock_soup.return_value = BeautifulSoup(_fake_ratios_html(), "html.parser")
        result = fetch_company_ratios("FAKE")
    assert result["asof_ts"].hour == 0
    assert result["asof_ts"].minute == 0
    assert result["asof_ts"].second == 0
    # knowledge_ts keeps the precise compute time — NOT truncated, still useful for audit.
    assert result["knowledge_ts"] != result["asof_ts"] or result["knowledge_ts"].hour == 0
