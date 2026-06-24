"""CFTC Commitment of Traders (COT) fetcher — free, no key.

Source: https://www.cftc.gov/dea/newcot/ (weekly, published Friday for Tuesday positions)

Why COT is special (POMDP framing): price/volume tell you what happened, but COT tells you
*who is positioned how* — large speculators (non-commercials), commercials (hedgers), and
small traders. This is a partial view of the otherwise-hidden positioning state. Positioning
extremes are mean-reverting: when large specs are maximally long, marginal buyers are
exhausted (contrarian sell signal), and vice versa.

Williams COT Index = percentile rank of net non-commercial positioning over a trailing window
(0 = most short in window, 100 = most long). >80 or <20 are the actionable extremes.

Markets mapped to our universe context:
  - E-mini S&P 500  → SPY/QQQ broad risk
  - Gold            → GOLD factor / risk-off
  - Crude Oil WTI   → energy names (VST/CEG/GEV indirectly)
  - Nasdaq-100      → tech beta
  - Bitcoin         → COIN/MSTR crypto beta
"""
from __future__ import annotations

import io
import logging
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "MarketOS/0.1 research@guaqai.me"}

# CFTC market names (as they appear in the legacy futures-only report) → friendly key
COT_MARKETS = {
    "E-MINI S&P 500": "spx",
    "NASDAQ-100 STOCK INDEX (MINI)": "ndx",
    "GOLD": "gold",
    "CRUDE OIL, LIGHT SWEET-WTI": "wti",
    "BITCOIN": "btc",
    "U.S. DOLLAR INDEX": "dxy",
}

# Current-year legacy futures-only combined report (CSV in a zip)
_COT_URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_2025.zip"
_COT_FALLBACK = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_2025.zip"


def _download_cot() -> pd.DataFrame:
    """Download and parse the current-year COT report."""
    for url in (_COT_URL, _COT_FALLBACK):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=60)
            if r.status_code != 200:
                continue
            import zipfile
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            fname = zf.namelist()[0]
            df = pd.read_csv(zf.open(fname), low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as e:
            logger.warning(f"CFTC download {url}: {e}")
    return pd.DataFrame()


def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Find the first matching column name (CFTC renames columns between report types)."""
    for c in candidates:
        if c in df.columns:
            return c
    # fuzzy: case-insensitive contains
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for lc, orig in low.items():
            if cand.lower() in lc:
                return orig
    return None


def compute_cot_features(window: int = 156) -> pd.DataFrame:
    """Compute COT positioning features (Williams index + net % of OI + weekly change).

    window: trailing weeks for the percentile index (156 ≈ 3 years).
    Returns one row with per-market positioning features, stored under symbol `_positioning`.
    """
    raw = _download_cot()
    if raw.empty:
        return pd.DataFrame()

    name_col = _col(raw, "Market_and_Exchange_Names", "Market and Exchange Names")
    date_col = _col(raw, "Report_Date_as_YYYY-MM-DD", "As_of_Date_In_Form_YYMMDD",
                    "Report_Date_as_MM_DD_YYYY")
    nc_long = _col(raw, "NonComm_Positions_Long_All", "Noncommercial Positions-Long (All)")
    nc_short = _col(raw, "NonComm_Positions_Short_All", "Noncommercial Positions-Short (All)")
    oi_col = _col(raw, "Open_Interest_All", "Open Interest (All)")

    if not all([name_col, date_col, nc_long, nc_short]):
        logger.warning("CFTC: required columns not found")
        return pd.DataFrame()

    now = datetime.now(timezone.utc)
    row: dict = {"asof_ts": now, "knowledge_ts": now}

    for cftc_name, key in COT_MARKETS.items():
        # Match the market (names sometimes carry exchange suffixes)
        mask = raw[name_col].astype(str).str.upper().str.contains(cftc_name, regex=False, na=False)
        sub = raw[mask].copy()
        if sub.empty:
            continue
        # Parse dates and sort
        sub["_date"] = pd.to_datetime(sub[date_col], errors="coerce")
        sub = sub.dropna(subset=["_date"]).sort_values("_date")
        if sub.empty:
            continue

        long_p = pd.to_numeric(sub[nc_long], errors="coerce")
        short_p = pd.to_numeric(sub[nc_short], errors="coerce")
        net = (long_p - short_p)
        net = net.dropna()
        if len(net) < 8:
            continue

        latest_net = float(net.iloc[-1])
        row[f"cot_{key}_net"] = latest_net

        # Williams COT Index: percentile of latest net within trailing window
        win = net.iloc[-window:]
        lo, hi = float(win.min()), float(win.max())
        if hi > lo:
            row[f"cot_{key}_index"] = 100.0 * (latest_net - lo) / (hi - lo)
        # Net as % of open interest (positioning intensity)
        if oi_col is not None:
            oi = pd.to_numeric(sub[oi_col], errors="coerce")
            if not oi.empty and float(oi.iloc[-1]) > 0:
                row[f"cot_{key}_net_pct_oi"] = 100.0 * latest_net / float(oi.iloc[-1])
        # Weekly change (flow direction)
        if len(net) >= 2:
            row[f"cot_{key}_wow"] = latest_net - float(net.iloc[-2])
        # 4-week momentum of positioning
        if len(net) >= 5:
            row[f"cot_{key}_mom4w"] = latest_net - float(net.iloc[-5])
        # Extreme flags (actionable contrarian zones)
        idx = row.get(f"cot_{key}_index")
        if idx is not None:
            row[f"cot_{key}_extreme_long"] = int(idx > 80)
            row[f"cot_{key}_extreme_short"] = int(idx < 20)

    return pd.DataFrame([row]) if len(row) > 2 else pd.DataFrame()
