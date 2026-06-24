"""Macro data fetcher — all free, no billing required.

Sources:
  FRED       — 800k+ macro series. API key required (free registration).
               Covers: VIX, DXY, US10Y, OIL (WTI), Fed Funds, CPI, PMI,
               USDINR, India10Y, M2, credit spreads, yield curve.
  BLS        — Bureau of Labor Statistics. No key. CPI, PPI, payrolls, unemployment.
  IMF WEO    — GDP growth + inflation forecasts for all countries. No key.
  World Bank — Long-run development indicators. No key.
  CBOE       — Full VIX history from 1990. CSV download. No key.

All data is normalized to daily frequency (forward-fill from lower-frequency sources).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from io import StringIO

import pandas as pd
import requests

FRED_KEY = os.environ.get("FRED_API_KEY", "")
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_HEADERS = {"User-Agent": "MarketOS/0.1 research@guaqai.me"}

# ── FRED series catalogue ─────────────────────────────────────────────────────
FRED_SERIES = {
    # Volatility & risk
    "VIX":          "VIXCLS",
    "MOVE":         "BAMLH0A0HYM2",     # HY spread as bond vol proxy
    "CREDIT_SPRD":  "BAMLC0A0CM",       # investment grade spread
    # Dollar & FX
    "DXY":          "DTWEXBGS",          # broad dollar index
    "USDINR":       "DEXINUS",
    "EURUSD":       "DEXUSEU",
    "JPYUSD":       "DEXJPUS",
    # Rates
    "US2Y":         "DGS2",
    "US10Y":        "DGS10",
    "US30Y":        "DGS30",
    "FED_RATE":     "FEDFUNDS",
    "TIPS_10Y":     "DFII10",            # real rate
    # Commodities
    "OIL_WTI":      "DCOILWTICO",
    "OIL_BRENT":    "DCOILBRENTEU",
    "NATGAS":       "DHHNGSP",
    "GOLD":         "GOLDAMGBD228NLBM",
    "COPPER":       "PCOPPUSDM",         # monthly
    # Inflation & growth
    "CPI_YOY":      "CPIAUCSL",
    "CORE_CPI":     "CPILFESL",
    "PPI":          "PPIACO",
    "US_GDP":       "GDP",               # quarterly
    # Labour
    "UNEMPLOYMENT": "UNRATE",
    "PAYROLLS":     "PAYEMS",
    # Money
    "M2":           "M2SL",
    # India proxies (FRED)
    "INDIA_CPI":    "INDCPIALLMINMEI",   # monthly
}


def fetch_fred(series_id: str, start: str = "2010-01-01") -> pd.Series:
    """Fetch a FRED series. Returns a pd.Series with DatetimeIndex."""
    if not FRED_KEY:
        raise RuntimeError("FRED_API_KEY not set")
    params = {
        "series_id": series_id,
        "observation_start": start,
        "api_key": FRED_KEY,
        "file_type": "json",
        "sort_order": "asc",
    }
    r = requests.get(FRED_BASE, params=params, headers=_HEADERS, timeout=20)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    data = {o["date"]: float(o["value"]) for o in obs if o["value"] not in (".", "")}
    return pd.Series(data, name=series_id, dtype=float)


def fetch_macro_panel(start: str = "2010-01-01") -> pd.DataFrame:
    """Fetch all FRED macro series into a daily wide DataFrame."""
    frames = {}
    for name, sid in FRED_SERIES.items():
        try:
            s = fetch_fred(sid, start)
            s.index = pd.to_datetime(s.index)
            frames[name] = s
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames)
    df = df.reindex(pd.bdate_range(df.index.min(), df.index.max()))
    return df.ffill().sort_index()


def compute_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive analytical features from raw macro panel."""
    f = pd.DataFrame(index=df.index)

    # Yield curve
    if "US10Y" in df and "US2Y" in df:
        f["yield_curve_2_10"] = df["US10Y"] - df["US2Y"]
        f["yield_curve_inverted"] = (f["yield_curve_2_10"] < 0).astype(float)

    # Real rate
    if "US10Y" in df and "TIPS_10Y" in df:
        f["real_rate_10y"] = df["TIPS_10Y"]
        f["breakeven_inflation"] = df["US10Y"] - df["TIPS_10Y"]

    # VIX regime
    if "VIX" in df:
        f["vix"] = df["VIX"]
        f["vix_z30"] = (df["VIX"] - df["VIX"].rolling(30).mean()) / df["VIX"].rolling(30).std()
        f["vix_spike"] = (df["VIX"] > df["VIX"].rolling(252).quantile(0.85)).astype(float)

    # Dollar momentum
    if "DXY" in df:
        f["dxy"] = df["DXY"]
        f["dxy_mom_20"] = df["DXY"].pct_change(20)

    # Oil
    if "OIL_WTI" in df:
        f["oil_wti"] = df["OIL_WTI"]
        f["oil_mom_20"] = df["OIL_WTI"].pct_change(20)
        f["oil_yoy"] = df["OIL_WTI"].pct_change(252)

    # US10Y momentum
    if "US10Y" in df:
        f["us10y"] = df["US10Y"]
        f["us10y_chg_20d"] = df["US10Y"].diff(20)

    # Credit conditions
    if "CREDIT_SPRD" in df:
        f["credit_spread"] = df["CREDIT_SPRD"]
        f["credit_tightening"] = (df["CREDIT_SPRD"].diff(5) > 0).astype(float)

    f["asof_ts"] = df.index
    f["knowledge_ts"] = df.index  # macro data is lagged but we use what's published daily
    return f.dropna(how="all")


# ── BLS (Bureau of Labor Statistics) — no key needed ─────────────────────────
BLS_SERIES = {
    "CPI_ALL":    "CUUR0000SA0",    # CPI all items
    "CPI_CORE":   "CUUR0000SA0L1E", # CPI ex food & energy
    "PPI_ALL":    "WPSFD49207",     # PPI finished goods
    "PAYROLLS":   "CES0000000001",  # total nonfarm payrolls
    "UNEMPLOYMENT":"LNS14000000",   # unemployment rate
}

def fetch_bls(series_ids: list[str] | None = None) -> pd.DataFrame:
    """Fetch BLS public data (no registration, 25 series/query, 500 queries/day)."""
    series_ids = series_ids or list(BLS_SERIES.values())
    payload = {"seriesid": series_ids, "startyear": "2010", "endyear": str(datetime.now().year)}
    r = requests.post("https://api.bls.gov/publicAPI/v2/timeseries/data/",
                      json=payload, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    result = r.json()
    frames = {}
    rev_map = {v: k for k, v in BLS_SERIES.items()}
    for series in result.get("Results", {}).get("series", []):
        sid = series["seriesID"]
        name = rev_map.get(sid, sid)
        data = {}
        for point in series.get("data", []):
            try:
                year, period = point["year"], point["period"]
                if period.startswith("M"):  # monthly
                    month = int(period[1:])
                    date = pd.Timestamp(int(year), month, 1)
                    data[date] = float(point["value"])
            except (ValueError, KeyError):
                pass
        if data:
            frames[name] = pd.Series(data, dtype=float)
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames)
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index().ffill()


# ── IMF World Economic Outlook — no key needed ────────────────────────────────
IMF_INDICATORS = {
    "GDP_GROWTH":  "NGDP_RPCH",    # real GDP growth %
    "INFLATION":   "PCPIPCH",      # CPI inflation %
    "CURR_ACCT":   "BCA_NGDPD",    # current account % GDP
    "UNEMPLOYMENT":"LUR",           # unemployment rate
}
IMF_COUNTRIES = {"USA": "US", "IND": "IN", "CHN": "CN", "GBR": "GB", "JPN": "JP"}

def fetch_imf_weo() -> pd.DataFrame:
    """Fetch IMF WEO forecasts for key countries and indicators."""
    rows = []
    current_year = datetime.now().year
    periods = ",".join(str(y) for y in range(2015, current_year + 3))
    for imf_code, iso2 in IMF_COUNTRIES.items():
        for fname, code in IMF_INDICATORS.items():
            try:
                url = f"https://www.imf.org/external/datamapper/api/v1/{code}/{imf_code}?periods={periods}"
                r = requests.get(url, headers=_HEADERS, timeout=15)
                r.raise_for_status()
                vals = r.json().get("values", {}).get(code, {}).get(imf_code, {})
                for year, val in vals.items():
                    if val is not None:
                        rows.append({"date": pd.Timestamp(int(year), 12, 31),
                                     "country": iso2, "indicator": fname, "value": float(val)})
            except Exception:
                pass
    return pd.DataFrame(rows)


# ── World Bank — no key needed ────────────────────────────────────────────────
WB_INDICATORS = {
    "GDP_GROWTH":    "NY.GDP.MKTP.KD.ZG",
    "INFLATION":     "FP.CPI.TOTL.ZG",
    "FDI_PCT_GDP":   "BX.KLT.DINV.WD.GD.ZS",
    "TRADE_PCT_GDP": "NE.TRD.GNFS.ZS",
}
WB_COUNTRIES = ["US", "IN", "CN", "GB", "JP", "DE", "BR"]

def fetch_world_bank(indicator: str, countries: list[str] | None = None) -> pd.DataFrame:
    """Fetch a World Bank indicator for multiple countries."""
    countries = countries or WB_COUNTRIES
    ctry_str = ";".join(countries)
    url = f"https://api.worldbank.org/v2/country/{ctry_str}/indicator/{indicator}"
    params = {"format": "json", "per_page": 500, "mrv": 15}
    r = requests.get(url, params=params, headers=_HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    if len(data) < 2 or not data[1]:
        return pd.DataFrame()
    rows = []
    for item in data[1]:
        if item.get("value") is not None:
            rows.append({
                "date": pd.Timestamp(int(item["date"]), 12, 31),
                "country": item["countryiso3code"],
                "value": float(item["value"]),
                "indicator": indicator,
            })
    return pd.DataFrame(rows)


# ── CBOE VIX History (full history from 1990) — no key ───────────────────────
def fetch_cboe_vix_history() -> pd.DataFrame:
    """Fetch full CBOE VIX daily history. Independent source to cross-validate FRED."""
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    r = requests.get(url, headers=_HEADERS, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), parse_dates=["DATE"])
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()
