"""SEC EDGAR + FMP fundamentals fetcher.

Sources:
  SEC EDGAR XBRL Company Facts — free, no key. Full structured financials for all
    US public companies. Revenue, EPS, net income, total assets, etc.
  FMP v4 stable endpoints — key required (free tier). Income statements, key metrics,
    earnings calendar, insider transactions, institutional holders.

EDGAR is authoritative but batchy (quarterly filings). FMP bridges the gap with
normalized TTM metrics and forward-looking estimates.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

FMP_KEY = os.environ.get("FMP_API_KEY", "")
FMP_BASE = "https://financialmodelingprep.com/stable"
EDGAR_BASE = "https://data.sec.gov"

_HEADERS = {
    "User-Agent": "MarketOS/0.1 research@guaqai.me",
    "Accept": "application/json",
}

# Map ticker → SEC CIK (hardcoded for our universe; saves an API call per symbol)
# Fetch dynamically via tickers.json for new symbols
TICKER_TO_CIK = {
    "NVDA": "0001045810",
    "AMD":  "0000002488",
    "AVGO": "0001441816",
    "MSFT": "0000789019",
    "PLTR": "0001321655",
    "GEV":  "0001852633",
    "VST":  "0001692819",
    "CEG":  "0001679273",
    "ETN":  "0000031462",
    "LMT":  "0000936468",
    "RTX":  "0000101829",
    "NOC":  "0001133421",
    "CCJ":  "0000016040",
    "RKLB": "0001819989",
    "PATH": "0001680346",
    "COIN": "0001679788",
    "MSTR": "0001050446",
    "SPY":  None,  # ETF — no EDGAR filings
    "QQQ":  None,
}

CONCEPT_MAP = {
    # Revenue
    "revenue": ["us-gaap/Revenues", "us-gaap/RevenueFromContractWithCustomerExcludingAssessedTax"],
    # Net income
    "net_income": ["us-gaap/NetIncomeLoss"],
    # EPS diluted
    "eps_diluted": ["us-gaap/EarningsPerShareDiluted"],
    # Operating income
    "op_income": ["us-gaap/OperatingIncomeLoss"],
    # Total assets
    "total_assets": ["us-gaap/Assets"],
    # Free cash flow components
    "cfo": ["us-gaap/NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["us-gaap/PaymentsToAcquirePropertyPlantAndEquipment"],
    # R&D
    "rd_expense": ["us-gaap/ResearchAndDevelopmentExpense"],
    # Shares outstanding
    "shares_diluted": ["us-gaap/CommonStockSharesOutstanding"],
}


def _cik_for_ticker(ticker: str) -> str | None:
    """Resolve ticker → CIK. Checks hardcoded map first, then EDGAR lookup."""
    cik = TICKER_TO_CIK.get(ticker)
    if cik is not None:
        return cik
    try:
        r = requests.get(
            f"{EDGAR_BASE}/files/company_tickers.json",
            headers=_HEADERS, timeout=15,
        )
        r.raise_for_status()
        for _, entry in r.json().items():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except Exception:
        pass
    return None


def fetch_edgar_concept(ticker: str, concept_path: str) -> pd.DataFrame:
    """Fetch a single XBRL concept for a ticker from EDGAR Company Facts.

    concept_path example: 'us-gaap/Revenues'
    Returns DataFrame with columns: end, val, accn, form, filed, unit.
    """
    cik = _cik_for_ticker(ticker)
    if not cik:
        return pd.DataFrame()
    ns, tag = concept_path.split("/", 1)
    url = f"{EDGAR_BASE}/api/xbrl/companyconcept/CIK{cik}/{ns}/{tag}.json"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=20)
        if r.status_code == 404:
            return pd.DataFrame()
        r.raise_for_status()
        data = r.json()
        # Units: pick USD for financial items, shares for share counts
        units = data.get("units", {})
        rows = []
        for unit_key, observations in units.items():
            for obs in observations:
                if obs.get("form") in ("10-K", "10-Q"):
                    rows.append({
                        "end": pd.to_datetime(obs["end"]),
                        "val": float(obs["val"]),
                        "accn": obs.get("accn", ""),
                        "form": obs.get("form", ""),
                        "filed": pd.to_datetime(obs.get("filed", obs["end"])),
                        "unit": unit_key,
                        "ticker": ticker,
                        "concept": concept_path,
                    })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values("end")
        # Drop duplicates: keep the latest filing for each period end
        df = df.drop_duplicates(subset=["end", "concept"], keep="last")
        return df
    except Exception:
        return pd.DataFrame()


def fetch_edgar_financials(ticker: str) -> pd.DataFrame:
    """Pull all configured XBRL concepts for a ticker, returning a wide pivoted frame."""
    frames = []
    for fname, paths in CONCEPT_MAP.items():
        for path in paths:
            df = fetch_edgar_concept(ticker, path)
            if not df.empty:
                df = df[["end", "filed", "val", "form"]].copy()
                df.columns = ["period_end", "filed_at", fname, "form"]
                frames.append(df)
                break  # use first path that returns data
        time.sleep(0.1)  # polite to EDGAR
    if not frames:
        return pd.DataFrame()
    from functools import reduce
    merged = reduce(
        lambda a, b: pd.merge(a, b, on=["period_end", "filed_at", "form"], how="outer"),
        frames,
    )
    merged["ticker"] = ticker
    merged["asof_ts"] = merged["period_end"].dt.tz_localize("UTC")
    merged["knowledge_ts"] = merged["filed_at"].dt.tz_localize("UTC")
    return merged.sort_values("period_end")


# ── FMP v4 stable endpoints ───────────────────────────────────────────────────

def fetch_fmp_income(ticker: str, period: str = "annual", limit: int = 20) -> pd.DataFrame:
    """Income statement from FMP stable endpoint."""
    if not FMP_KEY:
        return pd.DataFrame()
    params = {"symbol": ticker, "period": period, "limit": limit, "apikey": FMP_KEY}
    try:
        r = requests.get(f"{FMP_BASE}/income-statement", params=params,
                         headers=_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["period_end"] = pd.to_datetime(df["date"])
        df["asof_ts"] = df["period_end"].dt.tz_localize("UTC")
        df["knowledge_ts"] = pd.to_datetime(df.get("fillingDate", df["date"])).dt.tz_localize("UTC")
        return df.sort_values("period_end")
    except Exception:
        return pd.DataFrame()


def fetch_fmp_key_metrics(ticker: str, period: str = "annual", limit: int = 20) -> pd.DataFrame:
    """Key metrics (PE, EV/EBITDA, FCF yield, ROE, etc.) from FMP."""
    if not FMP_KEY:
        return pd.DataFrame()
    params = {"symbol": ticker, "period": period, "limit": limit, "apikey": FMP_KEY}
    try:
        r = requests.get(f"{FMP_BASE}/key-metrics", params=params,
                         headers=_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["period_end"] = pd.to_datetime(df["date"])
        df["asof_ts"] = df["period_end"].dt.tz_localize("UTC")
        df["knowledge_ts"] = df["asof_ts"]
        return df.sort_values("period_end")
    except Exception:
        return pd.DataFrame()


def fetch_fmp_earnings_calendar(tickers: list[str] | None = None,
                                 from_date: str | None = None,
                                 to_date: str | None = None) -> pd.DataFrame:
    """Upcoming earnings dates for the universe."""
    if not FMP_KEY:
        return pd.DataFrame()
    from datetime import timedelta
    if from_date is None:
        from_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if to_date is None:
        to_date = (datetime.now(timezone.utc) + timedelta(days=90)).strftime("%Y-%m-%d")
    params = {"from": from_date, "to": to_date, "apikey": FMP_KEY}
    try:
        r = requests.get(f"{FMP_BASE}/earnings-calendar", params=params,
                         headers=_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if tickers:
            df = df[df["symbol"].isin(tickers)]
        return df
    except Exception:
        return pd.DataFrame()


def fetch_fmp_insider_trades(ticker: str, limit: int = 50) -> pd.DataFrame:
    """SEC Form 4 insider transactions from FMP."""
    if not FMP_KEY:
        return pd.DataFrame()
    params = {"symbol": ticker, "limit": limit, "apikey": FMP_KEY}
    try:
        r = requests.get(f"{FMP_BASE}/insider-trading", params=params,
                         headers=_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["transaction_date"] = pd.to_datetime(df.get("transactionDate", df.get("date", "")))
        df["asof_ts"] = df["transaction_date"].dt.tz_localize("UTC")
        df["knowledge_ts"] = df["asof_ts"]
        return df.sort_values("transaction_date")
    except Exception:
        return pd.DataFrame()


def fetch_fmp_institutional_holders(ticker: str) -> pd.DataFrame:
    """Institutional ownership from FMP."""
    if not FMP_KEY:
        return pd.DataFrame()
    params = {"symbol": ticker, "apikey": FMP_KEY}
    try:
        r = requests.get(f"{FMP_BASE}/institutional-ownership/institutional-holders/symbol",
                         params=params, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()


def fetch_fmp_profile(ticker: str) -> dict:
    """Company profile — market cap, sector, beta, shares outstanding."""
    if not FMP_KEY:
        return {}
    try:
        r = requests.get(f"{FMP_BASE}/profile", params={"symbol": ticker, "apikey": FMP_KEY},
                         headers=_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data[0] if data else {}
    except Exception:
        return {}


def fetch_fmp_ratios(ticker: str, limit: int = 8) -> pd.DataFrame:
    """Financial ratios — margins, coverage, liquidity, per-share metrics."""
    if not FMP_KEY:
        return pd.DataFrame()
    try:
        r = requests.get(f"{FMP_BASE}/ratios", params={"symbol": ticker, "period": "annual",
                         "limit": limit, "apikey": FMP_KEY}, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["period_end"] = pd.to_datetime(df.get("date", ""))
        return df.sort_values("period_end")
    except Exception:
        return pd.DataFrame()


def fetch_fundamental_panel(universe: list[str]) -> pd.DataFrame:
    """One-row-per-ticker summary of the latest fundamental metrics."""
    # FMP's "stable" endpoints renamed several fields vs the legacy v3 names this
    # module was written against. Map actual FMP column -> the name fundamental.py expects.
    _FMP_ALIASES = {
        "priceToEarningsRatio": "peRatio",
        "evToEBITDA": "evToEbitda",
        "debtToEquityRatio": "debtToEquity",
        "debtToAssetsRatio": "debtRatio",
        "interestCoverageRatio": "interestCoverage",
    }
    _KEY_METRIC_COLS = [
        "peRatio", "evToEbitda", "freeCashFlowYield", "returnOnEquity",
        "debtToEquity", "priceToBookRatio", "revenuePerShare",
        "priceToSalesRatio", "evToOperatingCashFlow", "earningsYield",
        "currentRatio", "quickRatio",
        "returnOnAssets", "returnOnInvestedCapital", "netDebtToEBITDA",
        "enterpriseValue", "marketCap",
        "grossProfitMargin", "operatingProfitMargin", "netProfitMargin",
        "dividendYield",
    ]
    _RATIO_COLS = [
        "grossProfitMargin", "operatingProfitMargin", "netProfitMargin",
        "returnOnEquity", "returnOnAssets",
        "debtRatio", "debtToEquity", "interestCoverage",
        "currentRatio", "quickRatio", "cashRatio",
        "assetTurnover", "inventoryTurnover", "receivablesTurnover",
        "operatingCashFlowPerShare", "freeCashFlowPerShare",
        "peRatio", "priceToBookRatio", "priceToSalesRatio",
        "dividendYield",
    ]

    rows = []
    for ticker in universe:
        row: dict = {"ticker": ticker, "asof_ts": datetime.now(timezone.utc)}

        # FMP key metrics (last 8 fiscal years for trend features — quarterly period
        # requires a paid FMP plan, so we use annual periods on the free tier)
        km = fetch_fmp_key_metrics(ticker, limit=8).rename(columns=_FMP_ALIASES)
        if not km.empty:
            latest = km.iloc[-1]
            for col in _KEY_METRIC_COLS:
                if col in latest.index:
                    row[col] = latest[col]
            # YoY change for key fundamentals (kept the _qoq suffix for downstream compatibility)
            if len(km) >= 2:
                prev = km.iloc[-2]
                for col in ["peRatio", "evToEbitda", "returnOnEquity", "debtToEquity",
                            "grossProfitMargin", "netProfitMargin", "freeCashFlowYield"]:
                    if col in latest.index and col in prev.index:
                        try:
                            v_now = float(latest[col]) if latest[col] is not None else None
                            v_prev = float(prev[col]) if prev[col] is not None else None
                            if v_now is not None and v_prev is not None and v_prev != 0:
                                row[f"{col}_qoq"] = (v_now - v_prev) / abs(v_prev)
                        except (TypeError, ValueError):
                            pass
            # YoY change (q[-1] vs q[-5])
            if len(km) >= 5:
                prev_yr = km.iloc[-5]
                for col in ["peRatio", "returnOnEquity", "grossProfitMargin", "revenueGrowth"]:
                    if col in latest.index and col in prev_yr.index:
                        try:
                            v_now = float(latest[col]) if latest[col] is not None else None
                            v_prev = float(prev_yr[col]) if prev_yr[col] is not None else None
                            if v_now is not None and v_prev is not None and v_prev != 0:
                                row[f"{col}_yoy"] = (v_now - v_prev) / abs(v_prev)
                        except (TypeError, ValueError):
                            pass

        # FMP financial ratios (additional margin/quality/liquidity metrics)
        ratios = fetch_fmp_ratios(ticker, limit=4).rename(columns=_FMP_ALIASES)
        if not ratios.empty:
            latest_r = ratios.iloc[-1]
            for col in _RATIO_COLS:
                if col in latest_r.index and col not in row:
                    row[col] = latest_r[col]

        # Company profile (sector beta, market cap tier)
        profile = fetch_fmp_profile(ticker)
        if profile:
            row["beta"] = profile.get("beta")
            row["sector"] = profile.get("sector", "")
            row["mktcap"] = profile.get("mktCap")
            row["shares_outstanding"] = profile.get("sharesOutstanding")
            row["is_etf"] = int(bool(profile.get("isEtf", False)))

        # Insider net buys (last 90 days)
        insiders = fetch_fmp_insider_trades(ticker, limit=50)
        if not insiders.empty and "transactionType" in insiders.columns:
            buys = insiders[insiders["transactionType"].str.contains("P-Purchase", na=False)]
            sells = insiders[insiders["transactionType"].str.contains("S-Sale", na=False)]
            row["insider_buy_count"] = len(buys)
            row["insider_sell_count"] = len(sells)
            row["insider_net_bias"] = len(buys) - len(sells)
            if "securitiesTransacted" in insiders.columns:
                buy_vol = pd.to_numeric(buys.get("securitiesTransacted", pd.Series(dtype=float)),
                                        errors="coerce").sum()
                sell_vol = pd.to_numeric(sells.get("securitiesTransacted", pd.Series(dtype=float)),
                                         errors="coerce").sum()
                total_vol = buy_vol + sell_vol
                row["insider_buy_pct"] = buy_vol / total_vol if total_vol > 0 else 0.5

        rows.append(row)
        time.sleep(0.3)  # FMP rate limit

    return pd.DataFrame(rows)
