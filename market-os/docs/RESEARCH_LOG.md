# Research log

Append-only. One entry per experiment. The log is how we avoid re-running dead ends and
how we prove (later) that a deployed edge was discovered honestly, before it was sized.

## Template
```
### YYYY-MM-DD — <hypothesis>
- universe / period:
- features tested:
- validation: walk-forward folds, n per fold
- OOS expectancy: (with CI)  | Monte-Carlo p05 terminal:
- SHAP sanity: (top features, any leakage suspected?)
- decision: KEEP / CUT / ITERATE  — and why
```

### 2026-06-22 — does the PLS latent-space + regime-probability alpha model have real predictive
skill, or is it riding basket beta?
- universe / period: 19-symbol AI/semis/power/defense/space/crypto basket (NVDA, AMD, AVGO,
  MSFT, PLTR, GEV, VST, CEG, ETN, LMT, RTX, NOC, CCJ, RKLB, PATH, COIN, MSTR, SPY, QQQ),
  ~5y daily OHLCV (2021-06 to 2026-06), train=300d/test=60d walk-forward, top-5 of 19,
  20d label horizon, rebalance every 20d.
- features tested: 4 PLS2 supervised-latent components (`sup_z1..z4`) trained against
  fwd_ret_5d/20d, fwd_vol_20d, fwd_large_move_20d, plus 3 HMM regime-probability columns,
  fed into walk-forward XGBoost/HistGBM (`AlphaModel`).
- validation: purged+embargoed walk-forward (4 folds for the PLS fit, rolling for the
  alpha model), n_oos=138 closed trades on the baseline config.
- OOS expectancy (baseline, real labels): Sharpe=1.592, win_rate=0.674, PF=3.81,
  expectancy=0.0956/trade, maxdd=-0.580. Passes `assert_positive_expectancy` (n=138≥30).
- **Randomized-label null test (5 seeds, same fitted panel)**: target column permuted
  before training, breaking any real (feature→target) relationship. Shuffled Sharpe:
  mean=1.073, std=0.164, range [0.86, 1.31] across 5 seeds — i.e. ~67% of the real Sharpe
  (1.592) is retained even with the target completely randomized. **This is the decisive
  finding**: the bulk of the apparent edge is a backtest-construction artifact (picking
  top-5-of-19 names from a basket that mostly went up over this window), not validated
  predictive skill from sup_z+regime. Added `principles.assert_survives_label_shuffle`
  to catch this going forward (max_retained_fraction=0.35 default; this result would fail
  it at ~0.67-0.82 retained).
- symbol/sector leave-one-out: drop NVDA → Sharpe 1.122, win_rate 0.626, maxdd -0.725
  (notably worse — NVDA carries disproportionate weight). Drop all 3 semis → Sharpe
  1.232, win_rate 0.711 (holds up reasonably). Drop both crypto names → Sharpe 1.152
  (holds up). Sector-level robustness is OK; symbol-level (NVDA) concentration risk is
  real.
- 2022-only OOS window (semis correction + crypto collapse + rate hikes — the one
  adversarial period actually reachable in this 5y history): Sharpe=1.051 on n=12 trades.
  Directionally fine but n=12 fails the project's own n≥30 evidentiary bar — inconclusive,
  not supportive evidence either way.
- market-memory-only (kNN over sup_z, no XGBoost) comparison: Sharpe=2.003, PF=74.93 on
  n=12 trades. PF=74.93 is a small-sample artifact, not a real result — explicitly not
  reporting this as "market memory beats the model," it's uninterpretable at this n.
- regime transition matrix (cheap diagnostic, no model needed): regimes are highly
  persistent (mean diagonal=0.965; stress sojourn ~14 days, calm ~29 days, neutral ~36
  days) — supports pursuing sequential/hidden-state modeling later, independent of the
  alpha-validity question above.
- survivorship-bias check: `assert_includes_delisted` exists in principles.py but has
  never been wired to the actual UNIVERSE (which has zero delisted names — a real
  violation of the project's own Rule 3b). Attempted to fix by fetching known
  delisted/bankrupt thematically-adjacent tickers (VORB, RIDE, FFIE) via yfinance — all
  returned zero data (expected, free retail sources don't carry delisted history). SI and
  BBBY returned data, but those tickers were reassigned to unrelated companies after the
  originals delisted — using that data would mislabel a different company's history as
  the failed one's. **Blocked by data vendor, not by code.** Needs a paid point-in-time
  universe vendor (CRSP/Norgate/Sharadar) to actually fix; not attempted further this
  round.
- SHAP sanity: not re-run this round (deferred; the shuffle-test finding makes SHAP
  analysis on the baseline model less urgent than fixing the underlying validation gap).
- decision: **ITERATE, not KEEP.** The originally-reported "modest edge over buy-and-hold"
  framing from the prior round is downgraded — most of it doesn't survive the shuffle
  test. Before any further claim of edge: (1) the alpha model needs to beat its own
  label-shuffle null by a real margin, not just beat buy-and-hold: re-architect either the
  selection rule (e.g. rank within a market-neutral or sector-neutral universe instead of
  picking raw top-N by predicted return, which is what's amplifying basket beta) or add a
  beta-hedge leg; (2) `assert_survives_label_shuffle` should gate every future gate-check
  before it's reported as a finding; (3) NVDA-specific concentration (item above) needs an
  explicit position cap rather than relying on top-N selection to diversify it away.
