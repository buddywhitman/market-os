"""AngelOne SmartAPI — authentication + market data. Intended to replace the NSE bhavcopy
scraper (currently blocked by Akamai bot-protection on this server's IP — see project
notes) with a legitimate, ToS-compliant API, AND to eventually provide real order
execution for the india sleeve.

SAFETY BOUNDARY, stated up front: this module is AUTH + DATA ONLY. Order placement is
intentionally NOT implemented here — that needs its own explicit confirmation flow,
dry-run mode, and almost certainly a human-approval step before any live order, given the
stakes. Building that alongside auth/data in the same pass would conflate "can we read
data" with "can we place real trades" — two very different risk levels.

Credentials (ANGELONE_API_KEY, ANGELONE_CLIENT_CODE, ANGELONE_MPIN, ANGELONE_TOTP_SECRET)
come from the server's .env, never hardcoded. The TOTP secret is the base32 string from
AngelOne's 2FA setup — `pyotp.TOTP(secret).now()` generates a fresh 6-digit code per call,
so this can authenticate unattended (no human typing a code each session).

KNOWN CONSTRAINT (see project notes): this server's outbound IP rotates (CGNAT), not
static. AngelOne's developer portal asks for a static IP at app-registration time: if
they enforce it, calls from here may be rejected unpredictably until a stable egress IP
exists (a small cloud VPS as a Tailscale exit node was the recommended fix, not yet built).
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

SMARTAPI_BASE = "https://apiconnect.angelbroking.com"
SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
_TIMEOUT = 10
_SCRIP_MASTER_TIMEOUT = 60  # the file is ~23MB; the per-request default is too short
_SCRIP_CACHE_MAX_AGE_DAYS = 7  # tokens are stable; refresh weekly, not every run


def _totp_code() -> str:
    import pyotp
    secret = os.environ["ANGELONE_TOTP_SECRET"]
    return pyotp.TOTP(secret).now()


def login() -> dict:
    """Authenticate via clientcode + MPIN + TOTP. Returns a dict with jwt_token/
    refresh_token/feed_token on success, or {"error": ...} on failure — never raises for
    an authentication failure (a wrong/dummy credential is an expected, common case here,
    not an exceptional one), only for missing env vars or a genuine network failure.
    """
    api_key = os.environ["ANGELONE_API_KEY"]
    client_code = os.environ["ANGELONE_CLIENT_CODE"]
    mpin = os.environ["ANGELONE_MPIN"]
    totp = _totp_code()

    url = f"{SMARTAPI_BASE}/rest/auth/angelbroking/user/v1/loginByPassword"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-UserType": "USER",
        "X-SourceID": "WEB",
        "X-ClientLocalIP": "127.0.0.1",
        "X-ClientPublicIP": "127.0.0.1",
        "X-MACAddress": "00:00:00:00:00:00",
        "X-PrivateKey": api_key,
    }
    body = {"clientcode": client_code, "password": mpin, "totp": totp}

    try:
        r = requests.post(url, json=body, headers=headers, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        return {"error": f"network_failure: {exc}", "http_status": None}

    try:
        payload = r.json()
    except ValueError:
        return {"error": f"non_json_response (HTTP {r.status_code}): {r.text[:300]}",
               "http_status": r.status_code}

    if not payload.get("status"):
        return {"error": payload.get("message", "login_failed"), "http_status": r.status_code,
               "error_code": payload.get("errorcode"), "raw": payload}

    data = payload.get("data", {})
    return {
        "jwt_token": data.get("jwtToken"), "refresh_token": data.get("refreshToken"),
        "feed_token": data.get("feedToken"), "http_status": r.status_code,
        "authenticated_at": datetime.now(timezone.utc),
    }


def _scrip_cache_path(cache_dir: str) -> Path:
    return Path(cache_dir) / "angelone_scrip_tokens.json"


def get_symbol_tokens(symbols: list[str], *, cache_dir: str = "data_lake",
                      exchange: str = "NSE") -> dict[str, str]:
    """Map trading symbols (e.g. "DIXON") to AngelOne's internal numeric instrument
    tokens — required by every market-data and order-placement endpoint; SmartAPI does
    NOT accept a plain trading symbol. Source is AngelOne's public scrip master (~23MB,
    every instrument across every exchange) — downloaded once and cached locally for
    `_SCRIP_CACHE_MAX_AGE_DAYS`, since tokens are stable and re-downloading 23MB on every
    job run would be wasteful. Equity series only ("-EQ" suffix in the master file).
    """
    cache_path = _scrip_cache_path(cache_dir)
    cached: dict | None = None
    if cache_path.exists():
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(
            cache_path.stat().st_mtime, tz=timezone.utc)
        if age < timedelta(days=_SCRIP_CACHE_MAX_AGE_DAYS):
            try:
                cached = json.loads(cache_path.read_text())
            except (json.JSONDecodeError, OSError):
                cached = None

    if cached is None:
        r = requests.get(SCRIP_MASTER_URL, timeout=_SCRIP_MASTER_TIMEOUT)
        r.raise_for_status()
        master = r.json()
        # Filter to the requested exchange's equity series before caching — the full file
        # covers every exchange/derivative/commodity; no need to cache or scan all of it
        # on every lookup.
        cached = {
            entry["symbol"].replace("-EQ", ""): entry["token"]
            for entry in master
            if entry.get("exch_seg") == exchange and str(entry.get("symbol", "")).endswith("-EQ")
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached))

    return {sym: cached[sym] for sym in symbols if sym in cached}


def get_historical_candles(
    symbol: str, token: str, *, jwt_token: str, api_key: str,
    from_date: datetime, to_date: datetime, interval: str = "ONE_DAY",
) -> list[dict]:
    """Daily (or other interval) OHLCV candles for one instrument. Returns a list of
    {date, open, high, low, close, volume} dicts, oldest first — empty list on any
    failure (auth expired, rate-limited, no data for the range), never raises; the caller
    already has a "no data this run" degrade path from the bhavcopy-based screen, reused
    here so the migration to AngelOne as the data source needs no caller-side changes.
    """
    url = f"{SMARTAPI_BASE}/rest/secure/angelbroking/historical/v1/getCandleData"
    headers = {
        "Content-Type": "application/json", "Accept": "application/json",
        "X-UserType": "USER", "X-SourceID": "WEB",
        "X-ClientLocalIP": "127.0.0.1", "X-ClientPublicIP": "127.0.0.1",
        "X-MACAddress": "00:00:00:00:00:00", "X-PrivateKey": api_key,
        "Authorization": f"Bearer {jwt_token}",
    }
    body = {
        "exchange": "NSE", "symboltoken": token, "interval": interval,
        "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
        "todate": to_date.strftime("%Y-%m-%d %H:%M"),
    }
    try:
        r = requests.post(url, json=body, headers=headers, timeout=_TIMEOUT)
        payload = r.json()
    except (requests.RequestException, ValueError):
        return []
    if not payload.get("status"):
        return []
    rows = payload.get("data") or []
    # AngelOne returns [timestamp, open, high, low, close, volume] arrays, not dicts.
    return [{"date": row[0], "open": row[1], "high": row[2], "low": row[3],
            "close": row[4], "volume": row[5]} for row in rows]


def fetch_ohlcv_history(
    symbol: str, token: str, *, jwt_token: str, api_key: str, years: float = 5.0,
) -> "pd.DataFrame":
    """Multi-year daily OHLCV for ONE symbol, same column contract as
    `yfinance_fetcher.fetch_ohlcv` (open/high/low/close/volume, indexed by date) — so
    `features/technical.py::build_technical_features()` runs on it unchanged.

    Verified directly against the real API (2026-06-23): a single un-chunked 5-year
    ONE_DAY request for BEL returned all 1238 trading days with no truncation, so this
    does NOT chunk — that's a deliberate finding, not an assumption. If a future symbol or
    longer range silently truncates, that would need to be caught and chunking added then,
    not pre-built speculatively for a limit that doesn't appear to exist.
    """
    import pandas as pd

    candles = get_historical_candles(symbol, token, jwt_token=jwt_token, api_key=api_key,
                                     from_date=datetime.now() - timedelta(days=years * 365),
                                     to_date=datetime.now())
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(candles)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.drop_duplicates(subset="date").sort_values("date").set_index("date")
    # Clear the index name (real bug, caught running the real India latent fit): an
    # index literally named "date" collides with a "date" COLUMN that
    # `latent_supervised.py`'s internal processing creates somewhere downstream —
    # `groupby("date")` then raises "ambiguous" because both an index level and a column
    # share that exact name. yfinance's OHLCV index is named "Date" (capital D), which
    # never collided — matching that quirk would be fragile; no name at all is safer.
    df.index.name = None
    return df[["open", "high", "low", "close", "volume"]]
