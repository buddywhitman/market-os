# Data contracts

Every table that crosses a module boundary obeys a contract. Contracts are what let us
swap a fetcher, add a feature family, or retrain a model without silent corruption.

## Universal columns
| Column | Type | Meaning |
|--------|------|---------|
| `asof_ts` | tz-aware UTC | the time the row is *attributed* to (e.g. bar close) |
| `knowledge_ts` | tz-aware UTC | the time the value actually became *knowable* |

`knowledge_ts > asof_ts` is a leak and is rejected by `principles.assert_no_lookahead`.

## Raw lake object (immutable)
Stored at `data_lake/raw/<domain>/<YYYY>/<MM>/<DD>/<sha256>.<ext>` and manifested in
`data_lake/raw/_manifest.jsonl` with: `domain, path, bytes, source, fetched_at, sha256,
code_version` (+ optional `symbol`, `interval`, …). Raw is **append-only**.

## OHLCV (normalized)
Index: tz-aware timestamp. Columns: `open, high, low, close, volume` (adjusted).

## Technical feature frame
All of OHLCV-derived columns in `features/technical.py` **plus** the universal columns.
Causal by construction; verified in `make demo` and CI.

## Universe (for historical studies)
Columns must include `symbol`, `status` ∈ {`listed`,`delisted`,`merged`,…}, listing and
delisting dates. A universe lacking delisted names is rejected by
`assert_includes_delisted` — survivorship bias is not allowed into a backtest.

## Opportunity row (portfolio input)
`symbol, sector, expectancy, sample_size, confidence` (+ optional `alpha_score`). Only
positive-expectancy rows with `sample_size ≥ 30` receive capital.

## Model prediction series
Only **out-of-sample, walk-forward** predictions (`models.alpha_model`) may enter a
backtest. In-sample predictions are for diagnostics only and never sized.
