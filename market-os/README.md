# market-os

> A continuously evolving **market operating system**: it extracts information, discovers
> alpha, adapts to regime change, and compounds capital across decades — while remaining
> explainable, reproducible, and cheap to run.

This is **not** a money-printing machine. It is an apparatus for staying *slightly less
wrong than competing models*, forever.

---

## The one-paragraph thesis

Markets are complex, adaptive, nonstationary, reflexive, partially observable, and
adversarial. Perfect prediction is impossible and chasing it is how you blow up. Durable
edge comes from a loop:

```
information extraction → adaptation → risk allocation → continuous learning → (repeat)
```

Everything in this repo serves that loop. Anything that does not improve **out-of-sample
expectancy** is deleted.

---

## Three horizons

| Horizon | Name | Holding | Status | What it does |
|--------|------|---------|--------|--------------|
| **H1** | Theme Hunter | days–months | scaffolded, runnable | Find explosive themes + ride trend persistence |
| **H2** | Mini Hedge Fund | days–weeks | scaffolded | Research/Macro/Technical/Sentiment/Quant/Risk/Portfolio agents → daily PM dashboard |
| **H3** | Market OS | years | research stubs | Knowledge graph, regime/world models, synthetic markets, self-play, evolution |

H2 is the **primary focus**. H1 funds attention and proves the data plumbing. H3 is the
long game.

---

## Non-negotiable rules (enforced, not aspirational)

These are encoded in `src/marketos/principles.py` and tested in `tests/`.

1. **No black boxes.** Every prediction ships with SHAP attributions.
2. **Everything versioned, timestamped, reproducible.** Raw data is immutable & hashed.
3. **No data leakage. No survivorship bias. No future information.** Point-in-time
   feature store; a `no_lookahead` guard runs in CI.
4. **No magical claims.** We report distributions and expectancy, never point forecasts
   dressed as certainty.
5. **If a feature does not improve out-of-sample expectancy, remove it.**
6. **Simplicity over complexity. Evidence over narrative.**

If you are about to violate one of these, the system should make it hard. That is by
design.

---

## Quickstart

```bash
make setup           # create venv, install deps
make infra-up        # postgres + redis + pgvector via docker compose
make demo            # end-to-end: fetch sample OHLCV → features → theme scores → backtest → expectancy
make dashboard       # streamlit PM dashboard
make test            # principles + no-lookahead + expectancy unit tests
```

No GPU, no paid API, and no internet beyond yfinance is required for `make demo`.

---

## Repository map

```
config/                 declarative knobs — universe, themes, costs, risk limits
data_lake/              immutable raw/  +  derived curated/ features/ models/ backtests/
src/marketos/
  principles.py         the rules above, as runtime-checkable invariants
  config.py             typed config loader
  data/                 fetchers → parsers → normalizers → catalog (DuckDB/Parquet)
  features/             point-in-time feature store + families (technical, …)
  themes/               H1 theme hunter + leader ranking
  models/               XGBoost/LightGBM alpha stack, walk-forward, SHAP
  regimes/              HMM / Kalman / particle filters
  backtest/             no-lookahead engine, costs, walk-forward, expectancy, Monte Carlo
  risk/                 ATR stops, Kelly-scaled sizing, exposure caps
  portfolio/            opportunity ranking + capital allocation
  sentiment/            news/social/transcript embeddings → features (RAG)
  agents/               research / macro / technical / sentiment / quant / risk / pm
  knowledge_graph/      entities, supply chains, ownership, causal links
  world_models/         state-space latent market models
  simulations/          synthetic markets, agent zoo, evolution, self-play
  dashboard/            streamlit PM cockpit
  pipelines/            Prefect flows (daily/intraday/research)
  utils/                time, hashing, ids
docs/                   ARCHITECTURE, BUILD_ORDER, DATA_CONTRACTS, RESEARCH_LOG
tests/                  invariants that keep us honest
```

## Build order

See [`docs/BUILD_ORDER.md`](docs/BUILD_ORDER.md). Short version: **data lake → feature
store → theme hunter → backtest → alpha models → agents → dashboard → regimes →
knowledge graph → foundation models → synthetic markets → world models.**

Do not skip ahead. A world model trained on leaky features is worse than no model.
