# Build order

Build bottom-up. Every phase must be *trustworthy* before the next is allowed to depend
on it. A later phase built on a leaky earlier phase is negative value.

| Phase | Deliverable | Done-when (gate) | Status |
|------|-------------|------------------|--------|
| 1 | **Data lake** | Raw immutable + hashed + manifested; reproducible re-fetch | ✅ scaffolded (`data/lake.py`) |
| 2 | **Feature store** | Point-in-time, `no_lookahead` passes in CI | ◑ technical family done; add fundamental/macro/sentiment/event |
| 3 | **Theme hunter (H1)** | Theme scores + leader ranking from real signals | ✅ scaffolded (`themes/`) |
| 4 | **Backtest infra** | No-lookahead engine, costs, walk-forward, expectancy, Monte Carlo | ✅ scaffolded (`backtest/`) |
| 5 | **XGBoost alpha models** | Walk-forward OOS preds + SHAP; beats baseline expectancy OOS | ✅ scaffolded (`models/alpha_model.py`) |
| 6 | **Agents** | Research/Macro/Technical/Sentiment emit structured features only | ◑ base contract done (`agents/base.py`) |
| 7 | **PM dashboard (H2)** | Daily cockpit with expectancy + SHAP + risk per name | ✅ scaffolded (`dashboard/app.py`) |
| 8 | **Regime models** | HMM/Kalman regimes feed sizing + model switching | ✅ scaffolded (`regimes/hmm.py`) |
| 9 | **Knowledge graph** | Entities, supply chains, ownership, causal links in pgvector | ☐ stub dir |
| 10 | **Foundation models** | Chronos/PatchTST/TimesFM/Mamba/TFT as feature generators | ☐ stub dir |
| 11 | **Synthetic markets** | Agent zoo (Buffett/CTA/MM/HFT/RL/LLM) + evolution + self-play | ☐ stub dir |
| 12 | **World models** | Latent state-space market models; counterfactual simulation | ☐ stub dir |

## Promotion criteria (how a feature/strategy earns capital)
1. Passes `no_lookahead` and survivorship checks.
2. Positive **out-of-sample** expectancy across walk-forward folds, n ≥ 30 per fold.
3. Monte-Carlo p05 terminal > 1.0 (survives unlucky trade ordering).
4. SHAP attributions are sane (no single leaky feature dominating).
5. Improves the *portfolio's* expectancy, not just its own (low marginal correlation).

Anything failing these is removed, not "tuned until it passes." Tuning until it passes is
how you overfit the graveyard onto the future.
