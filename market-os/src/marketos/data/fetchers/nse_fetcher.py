"""NSE India market data fetcher — no billing required.

Sources:
  NSE Bhavcopy    — Official daily OHLCV CSV for all NSE-listed equities.
                    Published ~18:30 IST each trading day. Free, no key.
  NSE FII/DII     — Institutional flow data (Foreign + Domestic).
                    Published daily. Strong market breadth signal.
  NSEPython       — Community wrapper for NSE option chains, circuit breakers,
                    advances/declines. Falls back gracefully if not installed.
  NSE Indices     — NIFTY 50, NIFTY Bank, NIFTY IT OHLCV via NSE REST.

India-specific note: all timestamps are IST (UTC+5:30). We convert to UTC on ingestion.
"""
from __future__ import annotations

import io
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

_HEADERS = {
    "User-Agent": "MarketOS/0.1 research@guaqai.me",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

IST = ZoneInfo("Asia/Kolkata")
NSE_BASE = "https://www.nseindia.com"
NSE_ARCHIVE = "https://nsearchives.nseindia.com"

# NSE index symbols we care about
NSE_INDICES = ["NIFTY 50", "NIFTY BANK", "NIFTY IT", "NIFTY MIDCAP 100"]


_NSE_TIMEOUT = 8  # seconds per request — NSE blocks servers; fail fast rather than hang


def _nse_session() -> requests.Session:
    """Build a session with the NSE cookie that bypasses bot detection."""
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get(f"{NSE_BASE}/", timeout=_NSE_TIMEOUT)
        time.sleep(0.3)
    except Exception:
        pass
    return s


def fetch_latest_bhavcopy(max_lookback: int = 6) -> pd.DataFrame:
    """Walk back from today to the most recent published bhavcopy.

    NSE publishes nothing on weekends/holidays, so on a Saturday the daily job must
    fall back to Friday's file. We try up to `max_lookback` calendar days back.
    """
    for back in range(max_lookback + 1):
        dt = datetime.now(IST) - timedelta(days=back)
        if dt.weekday() >= 5:  # skip Sat/Sun outright
            continue
        df = fetch_bhavcopy(dt)
        if not df.empty:
            return df
    return pd.DataFrame()


def fetch_bhavcopy(date: datetime | None = None) -> pd.DataFrame:
    """Download NSE equity bhavcopy (OHLCV) for a given date.

    date defaults to today (or previous trading day if called early morning).
    Returns OHLCV with a DatetimeIndex (UTC).
    """
    if date is None:
        date = datetime.now(IST)
    # NSE publishes bhavcopy for format: BhavCopy_NSE_CM_0d0d0d0d_0d0d_0d0d_F0001.csv.zip
    # or the legacy simpler format at nsearchives
    dt_str = date.strftime("%d%b%Y").upper()  # e.g. 20JUN2025
    # Try new format first (post 2024)
    yyyy = date.strftime("%Y")
    mm = date.strftime("%m")
    dd = date.strftime("%d")
    mon = date.strftime("%b").upper()
    new_url = (
        f"{NSE_ARCHIVE}/products/content/sec_bhavdata_full_"
        f"{dd}{mon}{yyyy}.csv"
    )
    legacy_url = (
        f"{NSE_ARCHIVE}/content/historical/EQUITIES/"
        f"{yyyy}/{mon}/cm{dt_str}bhav.csv.zip"
    )
    session = _nse_session()
    for url in [new_url, legacy_url]:
        try:
            r = session.get(url, timeout=_NSE_TIMEOUT)
            if r.status_code != 200:
                continue
            if url.endswith(".zip"):
                import zipfile
                zf = zipfile.ZipFile(io.BytesIO(r.content))
                fname = zf.namelist()[0]
                df = pd.read_csv(zf.open(fname))
            else:
                df = pd.read_csv(io.StringIO(r.text))
            # Normalize columns
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
            # Keep only EQ series
            if "series" in df.columns:
                df = df[df["series"].str.strip() == "EQ"]
            # Standardize column names
            col_map = {
                "symbol": "symbol", "open_price": "open", "high_price": "high",
                "low_price": "low", "close_price": "close", "tottrdqty": "volume",
                "totaltradedquantity": "volume", "last_price": "close",
                "open": "open", "high": "high", "low": "low", "close": "close",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            date_utc = date.replace(hour=15, minute=30, second=0, microsecond=0,
                                    tzinfo=IST).astimezone(timezone.utc)
            df["asof_ts"] = date_utc
            df["knowledge_ts"] = date_utc
            df["source"] = "NSE_BHAVCOPY"
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for c in numeric_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df[["symbol"] + [c for c in ["open", "high", "low", "close", "volume",
                                                  "asof_ts", "knowledge_ts", "source"]
                                     if c in df.columns]].dropna(subset=["close"])
        except Exception:
            continue
    return pd.DataFrame()


def fetch_nse_fii_dii(date: datetime | None = None) -> pd.DataFrame:
    """Fetch NSE FII/DII provisional flow data.

    Returns a single-row DataFrame with net buy/sell values for the date.
    FII net positive = foreign money flowing in (bullish for INR assets).
    """
    if date is None:
        date = datetime.now(IST)
    dt_str = date.strftime("%d-%m-%Y")
    session = _nse_session()
    url = f"{NSE_BASE}/api/fiidiiTradeReact?date={dt_str}"
    try:
        r = session.get(url, timeout=_NSE_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()
        rows = []
        now_utc = datetime.now(timezone.utc)
        date_utc = date.replace(hour=18, minute=30, second=0, microsecond=0,
                                 tzinfo=IST).astimezone(timezone.utc)
        for entry in data:
            category = entry.get("category", "").strip()
            rows.append({
                "date": date_utc,
                "category": category,
                "buy_value": float(str(entry.get("buyValue", 0)).replace(",", "") or 0),
                "sell_value": float(str(entry.get("sellValue", 0)).replace(",", "") or 0),
                "net_value": float(str(entry.get("netValue", 0)).replace(",", "") or 0),
                "asof_ts": date_utc,
                "knowledge_ts": now_utc,
            })
        df = pd.DataFrame(rows)
        # Pivot to wide: fii_net, dii_net
        wide = {}
        for _, row in df.iterrows():
            cat = row["category"].lower().replace(" ", "_")
            wide[f"{cat}_net"] = row["net_value"]
            wide[f"{cat}_buy"] = row["buy_value"]
            wide[f"{cat}_sell"] = row["sell_value"]
        wide["date"] = date_utc
        wide["asof_ts"] = date_utc
        wide["knowledge_ts"] = now_utc
        return pd.DataFrame([wide])
    except Exception:
        return pd.DataFrame()


def fetch_nse_index_ohlcv(index_name: str = "NIFTY 50", days: int = 365) -> pd.DataFrame:
    """Fetch historical OHLCV for an NSE index."""
    end = datetime.now(IST)
    start = end - timedelta(days=days)
    session = _nse_session()
    url = f"{NSE_BASE}/api/historicalOR/indicesHistory"
    params = {
        "indexType": index_name,
        "from": start.strftime("%d-%m-%Y"),
        "to": end.strftime("%d-%m-%Y"),
    }
    try:
        r = session.get(url, params=params, timeout=_NSE_TIMEOUT)
        r.raise_for_status()
        data = r.json().get("data", {}).get("indexCloseOnlineRecords", [])
        if not data:
            return pd.DataFrame()
        rows = []
        for rec in data:
            date_ist = datetime.strptime(rec["EOD_TIMESTAMP"], "%d-%b-%Y").replace(
                hour=15, minute=30, tzinfo=IST
            )
            rows.append({
                "open": float(rec.get("EOD_OPEN_INDEX_VAL", 0) or 0),
                "high": float(rec.get("EOD_HIGH_INDEX_VAL", 0) or 0),
                "low": float(rec.get("EOD_LOW_INDEX_VAL", 0) or 0),
                "close": float(rec.get("EOD_CLOSE_INDEX_VAL", 0) or 0),
                "volume": float(rec.get("EOD_TRADED_QTY", 0) or 0),
                "asof_ts": date_ist.astimezone(timezone.utc),
                "knowledge_ts": date_ist.astimezone(timezone.utc),
                "index": index_name,
            })
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["asof_ts"])
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


def fetch_nse_advances_declines() -> dict:
    """Market breadth snapshot: advances, declines, unchanged."""
    session = _nse_session()
    try:
        r = session.get(f"{NSE_BASE}/api/allIndices", timeout=_NSE_TIMEOUT)
        r.raise_for_status()
        data = r.json().get("data", [])
        nifty50 = next((d for d in data if d.get("index") == "NIFTY 50"), {})
        return {
            "advances": int(nifty50.get("advances", 0)),
            "declines": int(nifty50.get("declines", 0)),
            "unchanged": int(nifty50.get("unchanged", 0)),
            "asof_ts": datetime.now(timezone.utc),
        }
    except Exception:
        return {}


def fetch_nse_option_chain(symbol: str = "NIFTY") -> pd.DataFrame:
    """Fetch NSE option chain for PCR (put-call ratio) sentiment signal.

    PCR > 1.2 often indicates oversold / bullish reversal zone.
    PCR < 0.7 often indicates overbought / bearish reversal zone.
    """
    session = _nse_session()
    try:
        url = f"{NSE_BASE}/api/option-chain-indices?symbol={symbol}"
        r = session.get(url, timeout=_NSE_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        records = data.get("records", {}).get("data", [])
        total_ce_oi = sum(r.get("CE", {}).get("openInterest", 0) for r in records if r.get("CE"))
        total_pe_oi = sum(r.get("PE", {}).get("openInterest", 0) for r in records if r.get("PE"))
        pcr = total_pe_oi / max(total_ce_oi, 1)
        now = datetime.now(timezone.utc)
        return pd.DataFrame([{
            "symbol": symbol,
            "total_ce_oi": total_ce_oi,
            "total_pe_oi": total_pe_oi,
            "pcr": pcr,
            "asof_ts": now,
            "knowledge_ts": now,
        }])
    except Exception:
        return pd.DataFrame()


def fetch_nse_panel() -> pd.DataFrame:
    """Unified NSE snapshot for daily feature generation."""
    now = datetime.now(timezone.utc)
    row = {"asof_ts": now, "knowledge_ts": now}

    # FII/DII flows
    fii = fetch_nse_fii_dii()
    if not fii.empty:
        for col in fii.columns:
            if col not in ("asof_ts", "knowledge_ts", "date"):
                row[f"nse_{col}"] = fii.iloc[0].get(col, 0)

    # Advances/Declines
    breadth = fetch_nse_advances_declines()
    total = (breadth.get("advances", 0) + breadth.get("declines", 0) +
             breadth.get("unchanged", 0)) or 1
    row["nse_advance_ratio"] = breadth.get("advances", 0) / total
    row["nse_decline_ratio"] = breadth.get("declines", 0) / total

    # PCR
    pcr_df = fetch_nse_option_chain("NIFTY")
    if not pcr_df.empty:
        row["nse_nifty_pcr"] = float(pcr_df.iloc[0]["pcr"])

    # Nifty 50 close (last available)
    nifty = fetch_nse_index_ohlcv("NIFTY 50", days=5)
    if not nifty.empty:
        row["nse_nifty50_close"] = float(nifty["close"].iloc[-1])
        row["nse_nifty50_chg"] = float(nifty["close"].pct_change().iloc[-1])

    return pd.DataFrame([row])
