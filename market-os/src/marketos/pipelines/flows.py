"""Orchestration flows (Prefect if installed, plain functions otherwise).

Three cadences:
  * daily_research_flow  — fetch → lake → features → theme scores → alpha → PM snapshot
  * intraday_flow        — light refresh of prices/technicals for active names
  * research_flow        — heavy, ad-hoc: regime re-fit, feature-importance audits

Designed to run cheaply on Machine 2 (8GB, 24/7). Heavy model training is offloaded to
Machine 1 (RTX 3070Ti) on demand.
"""

from __future__ import annotations

import pandas as pd

from marketos.config import Config
from marketos.data.lake import DataLake
from marketos.data.fetchers.yfinance_fetcher import fetch_ohlcv
from marketos.features.technical import build_technical_features
from marketos.themes.theme_hunter import ThemeHunter
from marketos.principles import assert_no_lookahead

try:
    from prefect import flow, task
except Exception:  # graceful no-op decorators when Prefect is absent
    def task(fn=None, **_):
        return fn if fn else (lambda f: f)

    def flow(fn=None, **_):
        return fn if fn else (lambda f: f)


@task
def _ingest(symbol: str, lake: DataLake) -> pd.DataFrame:
    return fetch_ohlcv(symbol, lake=lake)


@task
def _featurize(ohlcv: pd.DataFrame) -> pd.DataFrame:
    feats = build_technical_features(ohlcv)
    assert_no_lookahead(feats)
    return feats


@flow(name="daily_research_flow")
def daily_research_flow(config_path: str | None = None) -> pd.DataFrame:
    cfg = Config.load(config_path) if config_path else Config.load()
    lake = DataLake(cfg.data_lake_root)
    universe = cfg.universe or ["AI", "SEMI", "POWER"]

    rows = []
    for sym in universe:
        ohlcv = _ingest(sym, lake)
        feats = _featurize(ohlcv)
        last = feats.dropna().iloc[-1]
        rows.append({
            "theme": sym,
            "narrative": 0.5,
            "momentum": last["mom_63"],
            "expansion": last["vol_expansion"],
            "rel_strength": last["mom_20"],
            "breadth": last["adx_14"] / 100,
        })
    return ThemeHunter().score_themes(pd.DataFrame(rows))


if __name__ == "__main__":
    print(daily_research_flow().to_string(index=False))
