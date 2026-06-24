"""Postgres-backed store for the marketos schema.

All writes are idempotent (ON CONFLICT DO NOTHING / DO UPDATE). The store is the
single point where Python data structures land in durable, queryable form. DuckDB
reads from Postgres via the postgres_scanner extension for in-process OLAP queries.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
from psycopg2.extras import Json


class MarketosStore:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._conn = None

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._dsn)
        return self._conn

    def _exec(self, sql: str, params=None, fetch=False):
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            if fetch:
                return cur.fetchall()
        conn.commit()

    # ── Raw manifest ──────────────────────────────────────────────────────────
    def upsert_raw_manifest(self, domain: str, source: str, fetched_at: datetime,
                             sha256: str, path: str, bytes_: int,
                             symbol: str | None = None, metadata: dict | None = None,
                             code_version: str = "0.1.0"):
        self._exec("""
            INSERT INTO marketos.raw_manifest
                (domain, source, fetched_at, sha256, path, bytes, symbol, metadata, code_version)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (sha256, domain) DO NOTHING
        """, (domain, source, fetched_at, sha256, path, bytes_,
              symbol, Json(metadata or {}), code_version))

    # ── Features ──────────────────────────────────────────────────────────────
    def upsert_features(self, symbol: str, asof_ts: datetime, knowledge_ts: datetime,
                         family: str, features: dict):
        self._exec("""
            INSERT INTO marketos.features (symbol, asof_ts, knowledge_ts, family, features)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT (symbol, asof_ts, family) DO UPDATE SET features=EXCLUDED.features
        """, (symbol, asof_ts, knowledge_ts, family, Json(features)))

    # ── Theme scores ──────────────────────────────────────────────────────────
    def upsert_theme_score(self, computed_at: datetime, theme: str,
                            score: float, inputs: dict):
        self._exec("""
            INSERT INTO marketos.theme_scores (computed_at, theme, theme_score, inputs)
            VALUES (%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
        """, (computed_at, theme, score, Json(inputs)))

    # ── Alpha signals ─────────────────────────────────────────────────────────
    def upsert_signal(self, symbol: str, asof_ts: datetime, model_version: str,
                       pred: float, shap_top: dict | None = None):
        self._exec("""
            INSERT INTO marketos.alpha_signals
                (symbol, asof_ts, model_version, pred, shap_top)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT (symbol, asof_ts, model_version) DO UPDATE SET pred=EXCLUDED.pred
        """, (symbol, asof_ts, model_version, pred, Json(shap_top or {})))

    # ── Backtest results ──────────────────────────────────────────────────────
    def insert_backtest(self, strategy_name: str, params: dict, report: dict):
        self._exec("""
            INSERT INTO marketos.backtest_results
                (strategy_name, params, expectancy, win_rate, profit_factor,
                 sharpe, max_drawdown, sample_size, report)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (strategy_name, Json(params),
              report.get("expectancy"), report.get("win_rate"),
              report.get("profit_factor"), report.get("sharpe"),
              report.get("max_drawdown"), report.get("sample_size"),
              Json(report)))

    # ── Regime labels ─────────────────────────────────────────────────────────
    def upsert_regime(self, asof_ts: datetime, market: str, regime: int, regime_name: str):
        self._exec("""
            INSERT INTO marketos.regime_labels (asof_ts, market, regime, regime_name)
            VALUES (%s,%s,%s,%s)
            ON CONFLICT (asof_ts, market) DO UPDATE SET regime=EXCLUDED.regime,
                regime_name=EXCLUDED.regime_name
        """, (asof_ts, market, regime, regime_name))

    def get_latest_regime(self, market: str) -> dict | None:
        """Most recent regime label for `market` (written by regime_update_job). None if
        the regime_update job hasn't run yet."""
        rows = self._exec(
            "SELECT asof_ts, regime, regime_name FROM marketos.regime_labels "
            "WHERE market=%s ORDER BY asof_ts DESC LIMIT 1",
            (market,), fetch=True)
        return dict(rows[0]) if rows else None

    # ── Sentiment cache ───────────────────────────────────────────────────────
    def get_cached_sentiment(self, content_hash: str) -> dict | None:
        rows = self._exec(
            "SELECT scores FROM marketos.sentiment_cache WHERE content_hash=%s",
            (content_hash,), fetch=True)
        return rows[0]["scores"] if rows else None

    def cache_sentiment(self, content_hash: str, source: str, asof_ts: datetime,
                         scores: dict, model_id: str,
                         symbol: str | None = None, theme: str | None = None):
        self._exec("""
            INSERT INTO marketos.sentiment_cache
                (content_hash, source, asof_ts, symbol, theme, scores, model_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (content_hash) DO NOTHING
        """, (content_hash, source, asof_ts, symbol, theme, Json(scores), model_id))

    # ── Trade attribution ────────────────────────────────────────────────────
    def ensure_trade_attribution_table(self):
        """Create marketos.trade_attribution if it doesn't exist. The permanent
        per-trade laboratory notebook: every position stores what was known at entry
        (regime, world state, reasoning) and what happened at exit, so future analysis
        can ask "which kinds of worlds produce the edge" instead of just aggregate Sharpe."""
        self._exec("""
            CREATE TABLE IF NOT EXISTS marketos.trade_attribution (
                trade_id                   text PRIMARY KEY,
                instrument                 text NOT NULL,
                strategy_name              text NOT NULL,
                entry_ts                   timestamptz NOT NULL,
                exit_ts                    timestamptz,
                status                     text NOT NULL DEFAULT 'open',

                regime_label               integer,
                regime_name                text,
                regime_probability         jsonb,
                momentum_state             text,
                momentum_200d              double precision,
                momentum_50d               double precision,
                vix_percentile             double precision,
                spy_regime                 text,
                qqq_regime                 text,
                entry_reason               text,

                entry_price                double precision,
                exit_price                 double precision,
                holding_days               integer,
                stop_triggered             boolean DEFAULT false,
                signal_decay_triggered     boolean DEFAULT false,
                realized_return            double precision,
                max_drawdown_during_trade  double precision,
                volatility_at_entry        double precision,
                position_size              double precision,

                portfolio_state            jsonb,
                world_state_vector         jsonb,
                world_state_hash           text,
                nearest_analog_ids         jsonb,

                created_at  timestamptz NOT NULL DEFAULT now(),
                updated_at  timestamptz NOT NULL DEFAULT now()
            )
        """)
        # additive migration for tables created before these columns existed
        for col, coltype in [
            ("momentum_200d", "double precision"), ("momentum_50d", "double precision"),
            ("vix_percentile", "double precision"), ("spy_regime", "text"), ("qqq_regime", "text"),
            ("world_state_hash", "text"),
        ]:
            self._exec(f"ALTER TABLE marketos.trade_attribution ADD COLUMN IF NOT EXISTS {col} {coltype}")
        self._exec("CREATE INDEX IF NOT EXISTS idx_trade_attribution_instrument "
                   "ON marketos.trade_attribution(instrument)")
        self._exec("CREATE INDEX IF NOT EXISTS idx_trade_attribution_entry_ts "
                   "ON marketos.trade_attribution(entry_ts)")
        self._exec("CREATE INDEX IF NOT EXISTS idx_trade_attribution_status "
                   "ON marketos.trade_attribution(status)")
        self._exec("CREATE INDEX IF NOT EXISTS idx_trade_attribution_world_state_hash "
                   "ON marketos.trade_attribution(world_state_hash)")

    def open_trade(self, trade_id: str, instrument: str, strategy_name: str, entry_ts: datetime,
                    *, regime_label: int | None = None, regime_name: str | None = None,
                    regime_probability: dict | None = None, momentum_state: str | None = None,
                    momentum_200d: float | None = None, momentum_50d: float | None = None,
                    vix_percentile: float | None = None, spy_regime: str | None = None,
                    qqq_regime: str | None = None,
                    entry_reason: str | None = None, entry_price: float | None = None,
                    volatility_at_entry: float | None = None, position_size: float | None = None,
                    portfolio_state: dict | None = None, world_state_vector: dict | None = None,
                    world_state_hash: str | None = None, nearest_analog_ids: list | None = None):
        self._exec("""
            INSERT INTO marketos.trade_attribution
                (trade_id, instrument, strategy_name, entry_ts, status,
                 regime_label, regime_name, regime_probability, momentum_state,
                 momentum_200d, momentum_50d, vix_percentile, spy_regime, qqq_regime, entry_reason,
                 entry_price, volatility_at_entry, position_size,
                 portfolio_state, world_state_vector, world_state_hash, nearest_analog_ids)
            VALUES (%s,%s,%s,%s,'open',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (trade_id) DO NOTHING
        """, (trade_id, instrument, strategy_name, entry_ts,
              regime_label, regime_name, Json(regime_probability or {}), momentum_state,
              momentum_200d, momentum_50d, vix_percentile, spy_regime, qqq_regime, entry_reason,
              entry_price, volatility_at_entry, position_size,
              Json(portfolio_state or {}), Json(world_state_vector or {}), world_state_hash,
              Json(nearest_analog_ids or [])))

    def close_trade(self, trade_id: str, exit_ts: datetime, exit_price: float,
                     realized_return: float | None, *, holding_days: int | None = None,
                     stop_triggered: bool = False, signal_decay_triggered: bool = False,
                     max_drawdown_during_trade: float | None = None):
        self._exec("""
            UPDATE marketos.trade_attribution
            SET exit_ts=%s, exit_price=%s, realized_return=%s, holding_days=%s,
                stop_triggered=%s, signal_decay_triggered=%s,
                max_drawdown_during_trade=%s, status='closed', updated_at=now()
            WHERE trade_id=%s
        """, (exit_ts, exit_price, realized_return, holding_days,
              stop_triggered, signal_decay_triggered, max_drawdown_during_trade, trade_id))

    def get_open_trades(self, instrument: str | None = None) -> list[dict]:
        if instrument:
            return self._exec(
                "SELECT * FROM marketos.trade_attribution WHERE status='open' AND instrument=%s",
                (instrument,), fetch=True) or []
        return self._exec(
            "SELECT * FROM marketos.trade_attribution WHERE status='open'", fetch=True) or []

    def get_trade_history(self, instrument: str | None = None, strategy_name: str | None = None,
                           limit: int = 1000) -> list[dict]:
        conditions, params = [], []
        if instrument:
            conditions.append("instrument=%s")
            params.append(instrument)
        if strategy_name:
            conditions.append("strategy_name=%s")
            params.append(strategy_name)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        return self._exec(
            f"SELECT * FROM marketos.trade_attribution {where} ORDER BY entry_ts DESC LIMIT %s",
            tuple(params), fetch=True) or []

    # ── Portfolio attribution ────────────────────────────────────────────────
    def ensure_portfolio_attribution_table(self):
        """Daily PM notebook: weights, exposure, effective_n, turnover, regime context.
        One row per (date, strategy_name)."""
        self._exec("""
            CREATE TABLE IF NOT EXISTS marketos.portfolio_attribution (
                date                    date NOT NULL,
                strategy_name           text NOT NULL,
                weights                 jsonb,
                gross_exposure          double precision,
                cash_weight             double precision,
                effective_n             double precision,
                realized_vol            double precision,
                turnover                double precision,
                regime_snapshot         jsonb,
                correlation_matrix_hash text,
                top_positions           jsonb,
                top_themes              jsonb,
                vix_percentile          double precision,
                daily_return            double precision,
                created_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (date, strategy_name)
            )
        """)
        for col, coltype in [("vix_percentile", "double precision"), ("daily_return", "double precision")]:
            self._exec(f"ALTER TABLE marketos.portfolio_attribution ADD COLUMN IF NOT EXISTS {col} {coltype}")
        self._exec("CREATE INDEX IF NOT EXISTS idx_portfolio_attribution_date "
                   "ON marketos.portfolio_attribution(date)")

    def upsert_portfolio_snapshot(self, date, strategy_name, *, weights: dict | None = None,
                                   gross_exposure: float | None = None, cash_weight: float | None = None,
                                   effective_n: float | None = None, realized_vol: float | None = None,
                                   turnover: float | None = None, regime_snapshot: dict | None = None,
                                   correlation_matrix_hash: str | None = None,
                                   top_positions: list | None = None, top_themes: list | None = None,
                                   vix_percentile: float | None = None, daily_return: float | None = None):
        self._exec("""
            INSERT INTO marketos.portfolio_attribution
                (date, strategy_name, weights, gross_exposure, cash_weight, effective_n,
                 realized_vol, turnover, regime_snapshot, correlation_matrix_hash,
                 top_positions, top_themes, vix_percentile, daily_return)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (date, strategy_name) DO UPDATE SET
                weights=EXCLUDED.weights, gross_exposure=EXCLUDED.gross_exposure,
                cash_weight=EXCLUDED.cash_weight, effective_n=EXCLUDED.effective_n,
                realized_vol=EXCLUDED.realized_vol, turnover=EXCLUDED.turnover,
                regime_snapshot=EXCLUDED.regime_snapshot,
                correlation_matrix_hash=EXCLUDED.correlation_matrix_hash,
                top_positions=EXCLUDED.top_positions, top_themes=EXCLUDED.top_themes,
                vix_percentile=EXCLUDED.vix_percentile, daily_return=EXCLUDED.daily_return
        """, (date, strategy_name, Json(weights or {}), gross_exposure, cash_weight, effective_n,
              realized_vol, turnover, Json(regime_snapshot or {}), correlation_matrix_hash,
              Json(top_positions or []), Json(top_themes or []), vix_percentile, daily_return))

    def get_portfolio_history(self, strategy_name: str, limit: int = 2000) -> list[dict]:
        return self._exec(
            "SELECT * FROM marketos.portfolio_attribution WHERE strategy_name=%s "
            "ORDER BY date DESC LIMIT %s", (strategy_name, limit), fetch=True) or []

    def query_world_cohort(self, strategy_name: str, *, regime_filters: dict[str, int] | None = None,
                            vix_min: float | None = None, vix_max: float | None = None,
                            effective_n_min: float | None = None, effective_n_max: float | None = None,
                            cash_min: float | None = None, cash_max: float | None = None,
                            limit: int = 10000) -> list[dict]:
        """SQL market-memory query: every day matching a categorical world description
        ("SPY calm, VIX low, effective_n>4, cash<40%"), no embeddings, no FAISS -- just
        WHERE clauses against the daily attribution table."""
        conditions, params = ["strategy_name=%s"], [strategy_name]
        for instr, regime in (regime_filters or {}).items():
            conditions.append("(regime_snapshot->>%s)::int = %s")
            params.extend([instr, regime])
        if vix_min is not None:
            conditions.append("vix_percentile >= %s")
            params.append(vix_min)
        if vix_max is not None:
            conditions.append("vix_percentile <= %s")
            params.append(vix_max)
        if effective_n_min is not None:
            conditions.append("effective_n >= %s")
            params.append(effective_n_min)
        if effective_n_max is not None:
            conditions.append("effective_n <= %s")
            params.append(effective_n_max)
        if cash_min is not None:
            conditions.append("cash_weight >= %s")
            params.append(cash_min)
        if cash_max is not None:
            conditions.append("cash_weight <= %s")
            params.append(cash_max)
        params.append(limit)
        where = " AND ".join(conditions)
        return self._exec(
            f"SELECT * FROM marketos.portfolio_attribution WHERE {where} ORDER BY date LIMIT %s",
            tuple(params), fetch=True) or []

    # ── Feature reads ─────────────────────────────────────────────────────────
    def get_latest_family(self, symbol: str, family: str) -> dict:
        """Return the most recent features dict for a (symbol, family) pair.

        Returns empty dict if no row found. The returned dict has raw values from the
        JSONB column — NaN/Inf are not possible (they're sanitized on write).
        """
        rows = self._exec("""
            SELECT features FROM marketos.features
            WHERE symbol=%s AND family=%s
            ORDER BY asof_ts DESC LIMIT 1
        """, (symbol, family), fetch=True)
        if not rows:
            return {}
        f = rows[0]["features"]
        return f if isinstance(f, dict) else {}

    def get_latest_families(self, symbols: list[str], families: list[str]) -> dict[tuple, dict]:
        """Batch-read latest features for many (symbol, family) pairs.

        Returns {(symbol, family): features_dict}.
        Single query using DISTINCT ON to avoid N×M round-trips.
        """
        if not symbols or not families:
            return {}
        rows = self._exec("""
            SELECT DISTINCT ON (symbol, family) symbol, family, features
            FROM marketos.features
            WHERE symbol = ANY(%s) AND family = ANY(%s)
            ORDER BY symbol, family, asof_ts DESC
        """, (symbols, families), fetch=True)
        return {(r["symbol"], r["family"]): (r["features"] if isinstance(r["features"], dict) else {})
                for r in (rows or [])}

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
