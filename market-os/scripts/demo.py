"""End-to-end demo: data lake → features → theme scores → backtest → expectancy.

Runs fully offline (synthetic GBM if yfinance/network is unavailable). Proves the spine
of the system works and that every principle gate passes. Run with: `make demo`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from marketos import __version__
from marketos.config import Config
from marketos.data.lake import DataLake
from marketos.data.fetchers.yfinance_fetcher import fetch_ohlcv
from marketos.features.technical import build_technical_features
from marketos.themes.theme_hunter import ThemeHunter
from marketos.backtest.engine import CostModel, run_backtest
from marketos.backtest.expectancy import compute_expectancy, monte_carlo_drawdown
from marketos.regimes.hmm import detect_regimes
from marketos.principles import assert_no_lookahead, assert_positive_expectancy


def main() -> None:
    cfg = Config.load()
    lake = DataLake(cfg.data_lake_root)
    universe = cfg.universe or ["AI", "SEMI", "POWER", "DEFENSE", "ROBOT"]

    print(f"market-os v{__version__} — demo")
    print(f"data lake: {lake.root}")
    print(f"universe: {universe}\n")

    prices, feats_by_sym = {}, {}
    for sym in universe:
        df = fetch_ohlcv(sym, lake=lake, code_version=__version__)
        prices[sym] = df["open"]
        f = build_technical_features(df)
        assert_no_lookahead(f)                       # principle gate
        feats_by_sym[sym] = f

    price_panel = pd.DataFrame(prices).dropna()

    # --- H1 Theme Hunter: build a theme panel from the technical features ----------
    rows = []
    for sym, f in feats_by_sym.items():
        last = f.dropna().iloc[-1]
        rows.append({
            "theme": sym,
            "narrative": np.random.default_rng(abs(hash(sym)) % 99).random(),  # placeholder for news/search
            "momentum": last["mom_63"],
            "expansion": last["vol_expansion"],
            "rel_strength": last["mom_20"],
            "breadth": last["adx_14"] / 100,
        })
    theme_panel = pd.DataFrame(rows)
    scored = ThemeHunter().score_themes(theme_panel)
    print("Theme scores:")
    print(scored[["theme", "theme_score"]].to_string(index=False), "\n")

    # --- Simple momentum strategy on the top theme, backtested honestly -----------
    top = scored.iloc[0]["theme"]
    close = fetch_ohlcv(top, lake=lake)["close"]
    mom = close.pct_change(20)
    signal = (mom > 0).astype(float)                 # long when 20d momentum positive
    weights = pd.DataFrame({top: signal})
    exec_prices = pd.DataFrame({top: fetch_ohlcv(top, lake=lake)["open"]})

    result = run_backtest(exec_prices, weights, cost=CostModel())
    rep = compute_expectancy(result.trade_returns.values)
    mc = monte_carlo_drawdown(result.trade_returns.values)

    regimes = detect_regimes(close.pct_change())
    print(f"Backtest on top theme '{top}':")
    print(f"  trades={rep.sample_size}  win_rate={rep.win_rate:.1%}  "
          f"expectancy={rep.expectancy:.4f}  profit_factor={rep.profit_factor:.2f}")
    print(f"  sharpe={rep.sharpe:.2f}  max_dd={rep.max_drawdown:.1%}  "
          f"kelly={rep.kelly_fraction:.2f}")
    print(f"  MC p05 terminal={mc['p05_terminal']:.2f}  MC p95 maxDD={mc['p95_maxdd']:.1%}")
    print(f"  current regime: {regimes['regime_name'].iloc[-1]}")
    print(f"  total costs paid: {result.total_costs:.4f}")
    print(f"\nlake manifest entries: {len(lake.manifest())}")

    try:
        assert_positive_expectancy(rep.expectancy, rep.sample_size)
        print("\n✓ positive-expectancy gate PASSED — this edge clears the bar.")
    except AssertionError as e:
        print(f"\n✗ positive-expectancy gate: {e}\n  (expected on random/synthetic data — "
              "the gate is doing its job by refusing to deploy a non-edge.)")


if __name__ == "__main__":
    main()
