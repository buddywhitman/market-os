# Architecture

## The loop (everything serves this)
```
            ┌──────────────────────────────────────────────────────────┐
            │                                                          │
   information extraction → adaptation → risk allocation → learning ──┘
```

## Layered view

```
┌─────────────────────────────────────────────────────────────────────────┐
│  H3  MARKET OS   knowledge graph · regimes · world models · synthetic     │
│                  markets · evolution · self-play            (years)       │
├─────────────────────────────────────────────────────────────────────────┤
│  H2  MINI HEDGE FUND   research · macro · technical · sentiment · quant   │
│                        · risk · portfolio  →  PM dashboard  (days–weeks)  │
├─────────────────────────────────────────────────────────────────────────┤
│  H1  THEME HUNTER   theme scoring · leader ranking          (days–months)│
├─────────────────────────────────────────────────────────────────────────┤
│  CORE   backtest (no-lookahead, costs, walk-forward, expectancy, MC)      │
│         · feature store (point-in-time) · alpha models (GBM + SHAP)       │
├─────────────────────────────────────────────────────────────────────────┤
│  DATA   immutable raw lake (hashed, manifested) · fetchers · parsers      │
│         · catalog (DuckDB/Parquet) · Postgres+pgvector · Redis            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data flow
1. **Fetchers** pull from official sources first (NSE/BSE/SEBI/RBI/SEC EDGAR/FRED),
   then low-cost APIs (OpenBB/yfinance/FMP), then alternative (Trends/Reddit/X/GitHub).
2. **Lake** stores raw bytes immutably, content-addressed, with provenance.
3. **Parsers/normalizers** turn raw → typed tables in the DuckDB/Parquet catalog.
4. **Feature store** computes families with point-in-time `asof_ts`/`knowledge_ts`.
5. **Agents** reduce data (including LLM/RAG sentiment) to **structured features only**.
6. **Alpha models** train walk-forward, emit OOS probabilities + SHAP.
7. **Quant/Risk/Portfolio** convert edge → sized, capped positions.
8. **Dashboard** is the human decision surface; **Prefect** runs the cadences.

## Hardware mapping
- **Machine 1** (RTX 3070Ti, i7, 32GB): model training, foundation TS models, SHAP,
  simulations, evolution. On-demand, heavy.
- **Machine 2** (i5, 8GB, 24/7): ingestion, lake, feature refresh, daily flows,
  dashboard, Postgres/Redis. Always-on, light.

## Why distributions, not point forecasts
A point forecast hides its own uncertainty and invites overbetting. We carry the whole
distribution (or at least a calibrated probability + expectancy + Monte-Carlo drawdown)
so sizing can be Kelly-aware and survival-first. "Slightly less wrong than competing
models, with calibrated uncertainty" is the entire game.
