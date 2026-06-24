"""Data lake ingestion orchestrator.

Called by the APScheduler jobs. Each function is a discrete ingest task that
can be run independently or as part of the daily/weekly pipeline.

All functions:
  1. Fetch data from external source
  2. Store raw bytes in the data lake (content-addressed)
  3. Write normalized features to Postgres marketos.features
  4. Return a summary dict for scheduler logging
"""
from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from marketos.data.lake import DataLake
from marketos.db.store import MarketosStore
from marketos.principles import content_hash

logger = logging.getLogger(__name__)

CODE_VERSION = os.environ.get("CODE_VERSION", "0.1.0")
DATA_LAKE_ROOT = os.environ.get("DATA_LAKE_ROOT", "data_lake")
PG_DSN = os.environ.get("POSTGRES_DSN", "")

# The validated QUANT cross-section. This list — and ONLY this list — defines the pool
# for cross-sectional ranks (xrank_*) and the supervised-latent/analog fit. Changing it
# changes the validated quant signal, so it is kept frozen at the 19 names that were
# validated. Do NOT add aggressive-sleeve instruments here.
UNIVERSE = [
    "NVDA", "AMD", "AVGO", "MSFT", "PLTR",
    "GEV", "VST", "CEG", "ETN",
    "LMT", "RTX", "NOC",
    "CCJ", "RKLB", "PATH",
    "COIN", "MSTR",
    "SPY", "QQQ",
]

# Aggressive-sleeve instruments that need PER-SYMBOL features computed/stored but must
# NOT enter the quant cross-sectional pool above. SOXL is a 3x daily-reset ETF; BTC-USD
# trades 7 days/week — both would distort every quant percentile and the PLS latent fit.
# They get the full per-symbol technical/composite vector (so the aggressive sleeve and
# its backtest have data) but are ranked AGAINST the quant pool, never added to it.
AGGRESSIVE_EXTRAS = ["SOXL", "BTC-USD", "MU", "WDC", "STX", "TSM", "QCOM", "INTC"]

# What compute_features iterates for per-symbol vectors: quant + aggressive extras.
COMPUTE_UNIVERSE = UNIVERSE + AGGRESSIVE_EXTRAS

# Broad CANDIDATE universe for daily technical screening (features/screening.py) — a much
# wider net than the frozen, validated 19-name UNIVERSE above. This does NOT feed the
# analog/latent/xrank pipeline; it's the "what else is out there" visibility layer the
# screening step narrows down daily. Promotion into UNIVERSE is a deliberate, separate
# re-validation decision, never automatic — see that module's docstring for why.
# yfinance-only by design (no Finnhub/Polygon/FMP per-symbol calls) — those are already
# rate-limit-capped at 19 symbols; this list can be 4-5x larger only because it touches
# none of them.
CANDIDATE_UNIVERSE = {
    "AI_SEMI": ["GOOGL", "META", "ASML", "AMAT", "LRCX", "KLAC", "ARM", "MRVL", "ON",
               "MCHP", "TXN", "ADI", "NXPI", "SNPS", "CDNS"],
    "MEMORY": ["MU", "WDC", "STX"],
    "OTHER_SEMI": ["TSM", "QCOM", "INTC"],
    "POWER_UTILITY": ["NEE", "DUK", "SO", "AEP", "EXC", "PCG", "XEL"],
    "DEFENSE": ["GD", "LHX", "HII", "TXT"],
    "NUCLEAR": ["LEU", "SMR", "BWXT", "OKLO", "NNE"],
    "SPACE": ["ASTS", "RDW", "LUNR"],
    "ROBOTICS_AUTOMATION": ["ISRG", "ABB", "ROK", "TER", "IRBT"],
    "CRYPTO_MINERS": ["MARA", "RIOT", "CLSK", "HUT", "CIFR"],
    "BIOTECH": ["VRTX", "REGN", "AMGN", "GILD", "MRNA", "CRSP", "NTLA", "BEAM"],
    "AI_SOFTWARE_INFRA": ["ORCL", "CRM", "SNOW", "DDOG", "NOW", "PANW", "CRWD"],
}

# Indian (NSE) candidate universe for the india sleeve — screened from `fetch_bhavcopy`
# (NSE-wide daily OHLCV; one HTTP call covers the whole exchange, so this list can be
# broad without per-symbol API cost). NSE symbols, no ".NS" suffix — bhavcopy is native.
# No per-stock FII/DII-ownership filter is applied (Rule 4 from the user's proposed
# ruleset) because no per-stock institutional-holding data source exists in this codebase
# — fetch_nse_fii_dii is a MARKET-WIDE aggregate flow number, not per-stock ownership %.
# Liquidity + momentum (computed from real bhavcopy history) is the actual filter; see
# screening.py's screen_symbol(), reused as-is for India.
# ELECTRONICS_VLSI is the closest domestic analog to the US AI/semicon wave (PLI-scheme
# EMS/chip-design-services names) — the user's own domain (AI/VLSI/embedded) maps here
# most directly, but the rest of the list exists so the screen isn't artificially narrowed
# to one theme; let the evidence decide where the real opportunities are.
INDIA_CANDIDATE_UNIVERSE = {
    "ELECTRONICS_VLSI_EMS": ["DIXON", "KAYNES", "AMBER", "SYRMA", "TATAELXSI", "CYIENT", "LTTS"],
    "DEFENSE_ELECTRONICS": ["BEL", "HAL", "BDL", "MAZDOCK", "COCHINSHIP"],
    "POWER_ENERGY": ["NTPC", "POWERGRID", "TATAPOWER", "ADANIPOWER", "NHPC"],
    "RENEWABLE_EV": ["SUZLON", "ADANIGREEN", "TATAMOTORS"],
    "IT_SERVICES": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"],
    "BANKING_FINANCE": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK"],
    "MANUFACTURING_CAPEX": ["LT", "SIEMENS", "ABB", "CGPOWER"],
    "PHARMA": ["SUNPHARMA", "CIPLA", "DRREDDY"],
    "AUTO": ["MARUTI", "M&M"],
}

# The validated INDIA cross-section — the analog to the US side's frozen `UNIVERSE`.
# Declared 2026-06-23 after `prioritize_subspace_india_job` successfully fit exactly these
# 38 names (real OOS validation: best_oos_ic_ir=10.74, effective_n=1560, market_memory
# matched against 44,161 historical snapshots — see project notes).
#
# DELIBERATELY a separate, independently-typed-out list, NOT `list(INDIA_CANDIDATE_UNIVERSE
# flattened)` — an earlier draft derived it that way, which silently defeats the entire
# point: editing INDIA_CANDIDATE_UNIVERSE (e.g. adding a new screening candidate) would
# have auto-grown this list too, with no re-validation step in between. Same mistake the US
# side's UNIVERSE/CANDIDATE_UNIVERSE split was built to prevent. Growing this list for real
# means re-running prioritize_subspace_india_job, confirming the validation numbers still
# look sane, and then deliberately editing this literal list — not a side effect of
# touching the candidate pool.
INDIA_UNIVERSE = [
    "DIXON", "KAYNES", "AMBER", "SYRMA", "TATAELXSI", "CYIENT", "LTTS",
    "BEL", "HAL", "BDL", "MAZDOCK", "COCHINSHIP",
    "NTPC", "POWERGRID", "TATAPOWER", "ADANIPOWER", "NHPC",
    "SUZLON", "ADANIGREEN", "TATAMOTORS",
    "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
    "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK",
    "LT", "SIEMENS", "ABB", "CGPOWER",
    "SUNPHARMA", "CIPLA", "DRREDDY",
    "MARUTI", "M&M",
]

# India factor panel — the NIFTY/BANKNIFTY/sector-index analog to FACTOR_EQUITY_SYMBOLS
# (registry.py, SPY/QQQ/XLK/etc) for the US side. These are AngelOne index instruments
# (instrumenttype=AMXIDX in the scrip master), NOT equities — they don't appear in the
# "-EQ" filtered map `angelone_fetcher.get_symbol_tokens` returns, so tokens are hardcoded
# here (looked up once directly from the scrip master, 2026-06-23 — stable reference
# instruments, not expected to change token IDs the way a company's listing could).
# "INDIA_VIX" is the direct analog to the US side's VIX macro factor.
INDIA_FACTOR_INDICES = {
    "NIFTY50": "99926000", "NIFTY500": "99926004", "NIFTYIT": "99926008",
    "BANKNIFTY": "99926009", "NIFTY100": "99926012", "NIFTYMIDCAP100": "99926011",
    "INDIA_VIX": "99926017", "NIFTYREALTY": "99926018", "NIFTYENERGY": "99926020",
    "NIFTYFMCG": "99926021", "NIFTYPHARMA": "99926023", "NIFTYPSE": "99926024",
    "NIFTYPSUBANK": "99926025", "NIFTYAUTO": "99926029", "NIFTYMETAL": "99926030",
}

# Sector/theme bucket for each quant-universe name — used by risk.sizing.enforce_sector_caps
# so the allocator can't silently concentrate the whole book in one theme (e.g. all-AI).
# Benchmarks (SPY/QQQ) get their own bucket since they're not thesis positions to size like
# the rest — kept out of MAX_SECTOR_WEIGHT enforcement by being a 100%-cap bucket of one.
SECTOR_MAP = {
    "NVDA": "AI_SEMI", "AMD": "AI_SEMI", "AVGO": "AI_SEMI", "MSFT": "AI_SEMI", "PLTR": "AI_SEMI",
    "GEV": "POWER", "VST": "POWER", "CEG": "POWER", "ETN": "POWER",
    "LMT": "DEFENSE", "RTX": "DEFENSE", "NOC": "DEFENSE",
    "CCJ": "NUCLEAR",
    "RKLB": "SPACE",
    "PATH": "ROBOTICS",
    "COIN": "CRYPTO", "MSTR": "CRYPTO",
    "SPY": "BENCHMARK", "QQQ": "BENCHMARK",
}

THEMES = ["AI", "SEMI", "POWER", "DEFENSE", "NUCLEAR", "ROBOTICS", "SPACE", "CRYPTO"]


def _lake() -> DataLake:
    return DataLake(DATA_LAKE_ROOT)


def _store_features(store: MarketosStore, symbol: str, family: str, row: dict):
    """Extract asof/knowledge timestamps from row dict and call store with correct signature."""
    now = datetime.now(timezone.utc)
    asof = row.get("asof_ts", now)
    knowledge = row.get("knowledge_ts", now)
    # Convert numpy/pandas timestamps to Python datetime
    if hasattr(asof, "to_pydatetime"):
        asof = asof.to_pydatetime()
    if hasattr(knowledge, "to_pydatetime"):
        knowledge = knowledge.to_pydatetime()
    # Ensure UTC-aware
    if hasattr(asof, "tzinfo") and asof.tzinfo is None:
        asof = asof.replace(tzinfo=timezone.utc)
    if hasattr(knowledge, "tzinfo") and knowledge.tzinfo is None:
        knowledge = knowledge.replace(tzinfo=timezone.utc)
    # Strip timestamps from the features payload (they're stored in dedicated columns)
    features = {k: v for k, v in row.items()
                if k not in ("asof_ts", "knowledge_ts", "symbol", "feature_family")}
    # Postgres JSON rejects NaN/Inf as invalid tokens — replace with None (SQL NULL)
    features = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in features.items()}
    store.upsert_features(symbol, asof, knowledge, family, features)


def _store() -> MarketosStore:
    return MarketosStore(PG_DSN)


def _safe_run(name: str, fn, *args, **kwargs) -> dict:
    """Run fn, catching all exceptions so one failed source doesn't kill the pipeline."""
    try:
        result = fn(*args, **kwargs)
        logger.info(f"[{name}] OK")
        return {"task": name, "status": "ok", "result": result}
    except Exception as e:
        logger.error(f"[{name}] FAILED: {e}\n{traceback.format_exc()}")
        return {"task": name, "status": "error", "error": str(e)}


# ── Daily: equity OHLCV ───────────────────────────────────────────────────────

def ingest_ohlcv() -> dict:
    """Fetch OHLCV for the full universe via yfinance."""
    from marketos.data.fetchers.yfinance_fetcher import fetch_ohlcv
    lake = _lake()
    store = _store()
    results = []
    for sym in UNIVERSE:
        try:
            df = fetch_ohlcv(sym, lake=lake, period="5y", code_version=CODE_VERSION)
            if df.empty:
                continue
            # Store latest bar as feature row
            row = df.iloc[-1].to_dict()
            row["symbol"] = sym
            row["feature_family"] = "ohlcv"
            _store_features(store, sym, "ohlcv", row)
            results.append(sym)
        except Exception as e:
            logger.warning(f"OHLCV {sym}: {e}")
    return {"ingested_symbols": results}


# ── Daily: macro ──────────────────────────────────────────────────────────────

def ingest_macro() -> dict:
    """Fetch FRED macro panel and compute derived features."""
    from marketos.data.fetchers.macro_fetcher import fetch_macro_panel, compute_macro_features
    store = _store()
    lake = _lake()
    raw = fetch_macro_panel()
    if raw.empty:
        return {"status": "no_data"}
    lake.put_raw(
        "macro", raw.to_csv().encode(),
        source="FRED", ext="csv", code_version=CODE_VERSION, extra={"rows": len(raw)},
    )
    feats = compute_macro_features(raw)
    if not feats.empty:
        row = feats.iloc[-1].to_dict()
        row["feature_family"] = "macro"
        _store_features(store, "_market", "macro", row)
    return {"macro_rows": len(raw), "feature_rows": len(feats)}


def ingest_bls() -> dict:
    """Fetch BLS inflation + labor data."""
    from marketos.data.fetchers.macro_fetcher import fetch_bls
    store = _store()
    df = fetch_bls()
    if df.empty:
        return {"status": "no_data"}
    row = df.iloc[-1].to_dict()
    row["feature_family"] = "macro_bls"
    _store_features(store, "_market", "macro_bls", row)
    return {"rows": len(df)}


# ── Daily: crypto ─────────────────────────────────────────────────────────────

def ingest_crypto() -> dict:
    """Fetch CoinGecko + Binance crypto panel."""
    from marketos.data.fetchers.crypto_fetcher import fetch_crypto_panel
    store = _store()
    lake = _lake()
    df = fetch_crypto_panel()
    if df.empty:
        return {"status": "no_data"}
    lake.put_raw(
        "crypto", df.to_json().encode(),
        source="CoinGecko+Binance", ext="json", code_version=CODE_VERSION, extra={},
    )
    row = df.iloc[0].to_dict()
    row["feature_family"] = "crypto"
    _store_features(store, "_crypto", "crypto", row)
    return {"columns": list(df.columns)}


# ── Daily: news + sentiment ───────────────────────────────────────────────────

def ingest_news() -> dict:
    """Fetch GDELT tone timelines + NewsAPI headlines."""
    from marketos.data.fetchers.news_fetcher import (
        fetch_news_for_universe,
        fetch_gdelt_for_themes,
        fetch_stocktwits_for_universe,
    )
    store = _store()
    lake = _lake()
    # GDELT tone by theme
    tone_data = fetch_gdelt_for_themes(THEMES)
    for theme, df in tone_data.items():
        if not df.empty:
            lake.put_raw(
                "news", df.to_json().encode(),
                source="GDELT", ext="json", code_version=CODE_VERSION, extra={"theme": theme},
            )
            row = {"theme": theme, "feature_family": "gdelt_tone",
                   "asof_ts": datetime.now(timezone.utc)}
            if "gdelt_tone" in df.columns:
                row["gdelt_tone_latest"] = float(df["gdelt_tone"].iloc[-1])
                row["gdelt_tone_mean_30d"] = float(df["gdelt_tone"].mean())
            store.upsert_theme_score(datetime.now(timezone.utc), theme,
                                     row.get("gdelt_tone_latest", 0.0),
                                     {"source": "gdelt"})

    # StockTwits sentiment
    twits = fetch_stocktwits_for_universe(UNIVERSE[:10])  # cap to preserve rate limits
    if not twits.empty:
        lake.put_raw(
            "news", twits.to_json().encode(),
            source="StockTwits", ext="json", code_version=CODE_VERSION, extra={},
        )
        for _, row in twits.iterrows():
            sym = row.get("symbol", "")
            if sym:
                store.cache_sentiment(
                    content_hash(f"stocktwits:{sym}".encode()),
                    "StockTwits",
                    datetime.now(timezone.utc),
                    {"bullish_ratio": row.get("bullish_ratio", 0.5),
                     "bearish_ratio": row.get("bearish_ratio", 0.5)},
                    "stocktwits_v1",
                    symbol=sym,
                )

    # NewsAPI headlines → OpenRouter structured sentiment (cached one-call-per-headline)
    news = fetch_news_for_universe(UNIVERSE, lookback_hours=24)
    scored_count = 0
    if not news.empty:
        lake.put_raw(
            "news", news.to_json().encode(),
            source="NewsAPI", ext="json", code_version=CODE_VERSION, extra={"articles": len(news)},
        )
        from marketos.sentiment.openrouter import score_text
        # Aggregate per-symbol sentiment so it can enter the feature store
        per_symbol: dict[str, list[dict]] = {}
        for _, art in news.iterrows():
            sym = art.get("symbol", "")
            text = f"{art.get('title', '')}. {art.get('description', '')}".strip()
            if not text or text == ".":
                continue
            try:
                scores = score_text(
                    text, store=store, symbol=sym or None,
                    asof_ts=art.get("published_at"),
                )
                scored_count += 1
                if sym:
                    per_symbol.setdefault(sym, []).append(scores)
            except Exception as e:
                logger.warning(f"sentiment score failed for {sym}: {e}")
        # Roll up per-symbol mean sentiment into the feature store
        now = datetime.now(timezone.utc)
        for sym, score_list in per_symbol.items():
            if not score_list:
                continue
            agg = {}
            keys = score_list[0].keys()
            for k in keys:
                vals = [s.get(k, 0.0) for s in score_list if isinstance(s.get(k), (int, float))]
                if vals:
                    agg[f"news_{k}_mean"] = float(np.mean(vals))
                    agg[f"news_{k}_max"] = float(np.max(vals))
            agg["news_article_count"] = len(score_list)
            agg["asof_ts"] = now
            agg["knowledge_ts"] = now
            _store_features(store, sym, "sentiment", agg)

    return {
        "gdelt_themes": list(tone_data.keys()),
        "news_articles": len(news) if not news.empty else 0,
        "headlines_scored": scored_count,
    }


# ── Daily: fundamentals ───────────────────────────────────────────────────────

def ingest_fundamentals() -> dict:
    """Fetch FMP + EDGAR fundamentals for the universe."""
    from marketos.data.fetchers.edgar_fetcher import fetch_fundamental_panel, fetch_fmp_earnings_calendar
    store = _store()
    lake = _lake()
    panel = fetch_fundamental_panel(UNIVERSE)
    if not panel.empty:
        lake.put_raw(
            "fundamentals", panel.to_json().encode(),
            source="FMP+EDGAR", ext="json", code_version=CODE_VERSION, extra={"symbols": len(panel)},
        )
        # Normalize into cross-sectional valuation/quality factors
        from marketos.features.fundamental import build_fundamental_features
        factors = build_fundamental_features(panel)
        now = datetime.now(timezone.utc)
        if not factors.empty:
            for _, row in factors.iterrows():
                sym = row.get("ticker", "")
                if sym:
                    d = row.to_dict()
                    d["asof_ts"] = now
                    d["knowledge_ts"] = now
                    _store_features(store, sym, "fundamental", d)

    # Earnings calendar (next 90 days)
    cal = fetch_fmp_earnings_calendar(UNIVERSE)
    if not cal.empty:
        lake.put_raw(
            "fundamentals", cal.to_json().encode(),
            source="FMP_EARNINGS", ext="json", code_version=CODE_VERSION, extra={},
        )

    return {"symbols": len(panel) if not panel.empty else 0,
            "earnings_events": len(cal) if not cal.empty else 0}


# ── Daily: NSE India ──────────────────────────────────────────────────────────

def ingest_nse() -> dict:
    """Fetch NSE bhavcopy + FII/DII flows + option chain PCR."""
    from marketos.data.fetchers.nse_fetcher import fetch_latest_bhavcopy, fetch_nse_panel
    store = _store()
    lake = _lake()
    # Bhavcopy (walks back to last trading day on weekends/holidays)
    bhav = fetch_latest_bhavcopy()
    if not bhav.empty:
        lake.put_raw(
            "nse", bhav.to_csv(index=False).encode(),
            source="NSE_BHAVCOPY", ext="csv", code_version=CODE_VERSION, extra={"rows": len(bhav)},
        )
    # Panel (FII/DII + PCR + breadth)
    panel = fetch_nse_panel()
    if not panel.empty:
        row = panel.iloc[0].to_dict()
        row["feature_family"] = "nse"
        _store_features(store, "_nse", "nse", row)
    return {"bhavcopy_rows": len(bhav) if not bhav.empty else 0}


# ── Daily: weather (Open-Meteo, keyless) ─────────────────────────────────────

def ingest_weather() -> dict:
    """Fetch Open-Meteo historical weather for economically-significant locations."""
    from marketos.data.fetchers.openmeteo_fetcher import compute_weather_features
    store = _store()
    df = compute_weather_features(days=90)
    if df.empty:
        return {"status": "no_data"}
    row = df.iloc[0].to_dict()
    row["feature_family"] = "weather"
    _store_features(store, "_weather", "weather", row)
    return {"features": len([k for k in row if k.startswith("wx_")])}


# ── Daily: RBI India ──────────────────────────────────────────────────────────

def ingest_rbi() -> dict:
    """Fetch RBI policy rates, forex reserves, INR/USD."""
    from marketos.data.fetchers.rbi_fetcher import compute_rbi_features
    store = _store()
    df = compute_rbi_features()
    if df.empty:
        return {"status": "no_data"}
    row = df.iloc[0].to_dict()
    row["feature_family"] = "rbi"
    _store_features(store, "_india", "rbi", row)
    return {"features": len([k for k in row if k.startswith("rbi_")])}


# ── Daily: CBOE volatility surface ───────────────────────────────────────────

def ingest_cboe() -> dict:
    """Fetch CBOE VIX term structure, SKEW, VVIX, and sector vols via yfinance."""
    from marketos.data.fetchers.cboe_fetcher import compute_cboe_features
    store = _store()
    row = compute_cboe_features()
    if len(row) <= 2:  # only timestamps, nothing fetched
        return {"status": "no_data"}
    row["feature_family"] = "cboe_vol"
    _store_features(store, "_cboe", "cboe_vol", row)
    return {"features": len([k for k in row if k not in ("asof_ts", "knowledge_ts", "feature_family")])}


# ── Daily: Reddit sentiment ───────────────────────────────────────────────────

def ingest_reddit() -> dict:
    """Fetch Reddit WSB/stocks/investing/options sentiment for universe."""
    from marketos.data.fetchers.reddit_fetcher import compute_reddit_features
    store = _store()
    lake = _lake()
    df = compute_reddit_features(UNIVERSE)
    if df.empty:
        return {"status": "no_data"}
    lake.put_raw(
        "news", df.to_json().encode(),
        source="Reddit", ext="json", code_version=CODE_VERSION, extra={"symbols": len(df)},
    )
    for _, row in df.iterrows():
        sym = row.get("symbol", "")
        if sym:
            d = row.to_dict()
            d["feature_family"] = "reddit"
            _store_features(store, sym, "reddit", d)
    return {"symbols_with_mentions": len(df)}


# ── Weekly: EIA energy inventories ───────────────────────────────────────────

def ingest_eia() -> dict:
    """Fetch EIA petroleum/natgas inventory surprises."""
    from marketos.data.fetchers.eia_fetcher import compute_eia_features
    store = _store()
    df = compute_eia_features()
    if df.empty:
        return {"status": "no_data"}
    row = df.iloc[0].to_dict()
    row["feature_family"] = "eia_energy"
    _store_features(store, "_energy", "eia_energy", row)
    return {"features": len([k for k in row if k.startswith("eia_")])}


# ── Weekly: OECD leading indicators ──────────────────────────────────────────

def ingest_oecd() -> dict:
    """Fetch OECD Composite Leading Indicators for major economies."""
    from marketos.data.fetchers.oecd_fetcher import compute_oecd_features
    store = _store()
    df = compute_oecd_features()
    if df.empty:
        return {"status": "no_data"}
    row = df.iloc[0].to_dict()
    row["feature_family"] = "oecd"
    _store_features(store, "_global", "oecd", row)
    return {"features": len([k for k in row if k.startswith("oecd_")])}


# ── Weekly: Finnhub market signals ───────────────────────────────────────────

def ingest_finnhub() -> dict:
    """Fetch Finnhub earnings surprises, insider sentiment, analyst ratings."""
    from marketos.data.fetchers.finnhub_fetcher import fetch_finnhub_universe
    store = _store()
    lake = _lake()
    df = fetch_finnhub_universe(UNIVERSE)
    if df.empty:
        return {"status": "no_data"}
    lake.put_raw(
        "fundamentals", df.to_json().encode(),
        source="Finnhub", ext="json", code_version=CODE_VERSION, extra={"symbols": len(df)},
    )
    for _, row in df.iterrows():
        sym = row.get("symbol", "")
        if sym:
            d = row.to_dict()
            d["feature_family"] = "finnhub"
            _store_features(store, sym, "finnhub", d)
    return {"symbols": len(df)}


# ── Weekly: GitHub developer activity ────────────────────────────────────────

def ingest_github() -> dict:
    """Fetch GitHub repo stars/commits for AI/tech companies in universe."""
    from marketos.data.fetchers.github_fetcher import compute_github_features
    store = _store()
    df = compute_github_features(UNIVERSE)
    if df.empty:
        return {"status": "no_data"}
    row = df.iloc[0].to_dict()
    row["feature_family"] = "github"
    _store_features(store, "_github", "github", row)
    return {"features": len([k for k in row if k.startswith("gh_")])}


# ── Daily: Hiring trends (open requisitions from public ATS APIs) ─────────────

def ingest_hiring() -> dict:
    """Fetch open-requisition counts/recency/function-mix per company careers site.

    Reads the prior `_hiring` snapshot so momentum (req growth) is computed in-fetcher —
    the change is the tradeable signal, not the level. Stored as a broadcast family.
    """
    from marketos.data.fetchers.hiring_fetcher import compute_hiring_features
    store = _store()
    prior = store.get_latest_family("_hiring", "hiring")
    df = compute_hiring_features(UNIVERSE, prior=prior)
    if df.empty:
        return {"status": "no_data"}
    row = df.iloc[0].to_dict()
    row["feature_family"] = "hiring"
    _store_features(store, "_hiring", "hiring", row)
    return {"features": len([k for k in row if k.startswith("hiring_")])}


# ── Daily: Aviation (OpenSky) ─────────────────────────────────────────────────

def ingest_aviation() -> dict:
    """Fetch OpenSky flight counts as economic activity proxy."""
    from marketos.data.fetchers.opensky_fetcher import compute_aviation_features
    store = _store()
    df = compute_aviation_features()
    if df.empty:
        return {"status": "no_data"}
    row = df.iloc[0].to_dict()
    row["feature_family"] = "aviation"
    _store_features(store, "_aviation", "aviation", row)
    return {"total_tracked": row.get("aviation_total_tracked", 0)}


# ── Weekly: AlphaVantage FX + macro ──────────────────────────────────────────

def ingest_alphavantage() -> dict:
    """Fetch AlphaVantage FX rates, treasury yields, macro indicators."""
    from marketos.data.fetchers.alphavantage_fetcher import compute_alphavantage_features
    store = _store()
    df = compute_alphavantage_features()
    if df.empty:
        return {"status": "no_data"}
    row = df.iloc[0].to_dict()
    row["feature_family"] = "alphavantage"
    _store_features(store, "_macro", "alphavantage", row)
    return {"features": len([k for k in row if k.startswith("av_")])}


# ── Weekly: Polygon options flow ──────────────────────────────────────────────

def ingest_polygon() -> dict:
    """Fetch Polygon options flow, IV surface, and stock snapshots."""
    from marketos.data.fetchers.polygon_fetcher import compute_polygon_features
    store = _store()
    df = compute_polygon_features(UNIVERSE)
    if df.empty:
        return {"status": "no_data"}
    for _, row in df.iterrows():
        sym = row.get("symbol", "")
        if sym:
            d = row.to_dict()
            d["feature_family"] = "options"
            _store_features(store, sym, "options", d)
    return {"symbols": len(df)}


# ── Daily: Wikipedia attention ────────────────────────────────────────────────

def ingest_wikipedia() -> dict:
    """Fetch Wikipedia pageviews for tickers and theme pages."""
    from marketos.data.fetchers.wikipedia_fetcher import compute_wikipedia_features
    store = _store()
    df = compute_wikipedia_features(UNIVERSE, days=30)
    if df.empty:
        return {"status": "no_data"}
    count = 0
    for _, row in df.iterrows():
        sym = row.get("symbol", "")
        if sym:
            d = row.to_dict()
            family = "wiki_themes" if sym.startswith("_") else "wiki_attention"
            d["feature_family"] = family
            _store_features(store, sym, family, d)
            count += 1
    return {"symbols_stored": count}


# ── Weekly: CFTC COT positioning ─────────────────────────────────────────────

def ingest_positioning() -> dict:
    """Fetch CFTC COT and build market-level positioning (ownership) features."""
    from marketos.data.fetchers.cftc_fetcher import compute_cot_features
    from marketos.features.ownership import build_market_positioning_features
    store = _store()
    lake = _lake()
    cot = compute_cot_features()
    if cot.empty:
        return {"status": "no_data"}
    cot_row = cot.iloc[0].to_dict()
    lake.put_raw(
        "positioning", cot.to_json().encode(),
        source="CFTC_COT", ext="json", code_version=CODE_VERSION, extra={},
    )
    feats = build_market_positioning_features(cot_row=cot_row)
    now = datetime.now(timezone.utc)
    feats["asof_ts"] = now
    feats["knowledge_ts"] = now
    feats["feature_family"] = "positioning"
    _store_features(store, "_positioning", "positioning", feats)
    return {"features": len([k for k in feats if k.startswith("pos_")])}


# ── Weekly: UN Comtrade trade flows ──────────────────────────────────────────

def ingest_comtrade() -> dict:
    """Fetch UN Comtrade semiconductor and crude oil import flows."""
    from marketos.data.fetchers.comtrade_fetcher import compute_comtrade_features
    store = _store()
    df = compute_comtrade_features()
    if df.empty:
        return {"status": "no_data"}
    row = df.iloc[0].to_dict()
    row["feature_family"] = "comtrade"
    _store_features(store, "_trade", "comtrade", row)
    return {"features": len([k for k in row if k.startswith("comtrade_")])}


# ── Weekly: Polymarket prediction-market probabilities ───────────────────────

def ingest_polymarket() -> dict:
    """Fetch implied probabilities for Fed/recession/AI-bubble/BTC/macro event markets."""
    from marketos.data.fetchers.polymarket_fetcher import compute_polymarket_features
    store = _store()
    df = compute_polymarket_features()
    if df.empty:
        return {"status": "no_data"}
    row = df.iloc[0].to_dict()
    row["feature_family"] = "polymarket"
    _store_features(store, "_polymarket", "polymarket", row)
    return {"features": len([k for k in row if k.startswith("pm_")])}


# ── Weekly: BIS macro-prudential ─────────────────────────────────────────────

def ingest_bis() -> dict:
    """Fetch BIS credit-to-GDP gap, debt service ratio, effective FX rate."""
    from marketos.data.fetchers.bis_fetcher import compute_bis_features
    store = _store()
    df = compute_bis_features()
    if df.empty:
        return {"status": "no_data"}
    row = df.iloc[0].to_dict()
    row["feature_family"] = "bis"
    _store_features(store, "_macro", "bis", row)
    return {"features": len([k for k in row if k.startswith("bis_")])}


# ── Weekly: Google Trends ─────────────────────────────────────────────────────

def ingest_google_trends() -> dict:
    """Fetch Google Trends for all theme keyword groups. Runs weekly."""
    from marketos.data.fetchers.google_trends_fetcher import build_trends_feature_row
    store = _store()
    df = build_trends_feature_row()
    if not df.empty:
        row = df.iloc[0].to_dict()
        row["feature_family"] = "google_trends"
        _store_features(store, "_trends", "google_trends", row)
    return {"features": len(df.columns) if not df.empty else 0}


# ── Weekly: IMF + World Bank ──────────────────────────────────────────────────

def ingest_global_macro() -> dict:
    """Fetch IMF WEO + World Bank indicators. Runs weekly (data updates monthly)."""
    from marketos.data.fetchers.macro_fetcher import fetch_imf_weo, fetch_world_bank, WB_INDICATORS
    lake = _lake()
    # IMF
    imf = fetch_imf_weo()
    if not imf.empty:
        lake.put_raw(
            "macro", imf.to_json().encode(),
            source="IMF_WEO", ext="json", code_version=CODE_VERSION, extra={"rows": len(imf)},
        )
    # World Bank
    wb_frames = []
    for indicator in list(WB_INDICATORS.values())[:2]:  # cap at 2 to avoid rate limits
        df = fetch_world_bank(indicator)
        if not df.empty:
            wb_frames.append(df)
    if wb_frames:
        # Each fetch_world_bank() call returns its own 0..N-1 RangeIndex; concatenating
        # multiple indicators duplicates index values, which orient='columns' (.to_json()'s
        # default) rejects. orient='records' doesn't depend on index uniqueness at all, and
        # is the natural shape for "one row per (country, date, indicator)" data anyway.
        wb = pd.concat(wb_frames, ignore_index=True)
        lake.put_raw(
            "macro", wb.to_json(orient="records").encode(),
            source="WORLD_BANK", ext="json", code_version=CODE_VERSION, extra={"rows": len(wb)},
        )
    return {"imf_rows": len(imf) if not imf.empty else 0,
            "wb_rows": sum(len(f) for f in wb_frames)}


# ── Master entry points ───────────────────────────────────────────────────────

def run_daily_ingest() -> list[dict]:
    """Run all daily ingest tasks. Called by scheduler at 02:00 UTC."""
    tasks = [
        ("ohlcv", ingest_ohlcv),
        ("macro", ingest_macro),
        ("bls", ingest_bls),
        ("crypto", ingest_crypto),
        ("news", ingest_news),
        ("nse", ingest_nse),
        ("weather", ingest_weather),
        ("rbi", ingest_rbi),
        ("cboe", ingest_cboe),
        ("reddit", ingest_reddit),
        ("aviation", ingest_aviation),
        ("wikipedia", ingest_wikipedia),
        ("hiring", ingest_hiring),
    ]
    return [_safe_run(name, fn) for name, fn in tasks]


def run_weekly_ingest() -> list[dict]:
    """Run weekly-cadence tasks. Called by scheduler on Sunday 01:00 UTC."""
    tasks = [
        ("fundamentals", ingest_fundamentals),
        ("google_trends", ingest_google_trends),
        ("global_macro", ingest_global_macro),
        ("eia", ingest_eia),
        ("oecd", ingest_oecd),
        ("finnhub", ingest_finnhub),
        ("github", ingest_github),
        ("polygon", ingest_polygon),
        ("alphavantage", ingest_alphavantage),
        ("bis", ingest_bis),
        ("comtrade", ingest_comtrade),
        ("positioning", ingest_positioning),
        ("polymarket", ingest_polymarket),
    ]
    return [_safe_run(name, fn) for name, fn in tasks]
