"""APScheduler-based pipeline runner.

Uses BlockingScheduler with SQLAlchemyJobStore so job state survives
container restarts. No Prefect server required (~10MB vs ~1GB).

Schedule (all UTC):
  Daily:
    01:30  ingest_news         — GDELT, NewsAPI, StockTwits, Wikipedia
    02:00  ingest_daily        — OHLCV, macro, crypto, NSE, BLS
    03:00  compute_features    — technical + macro feature derivation
    04:00  score_themes        — theme hunter ranking
    05:00  regime_update       — HMM regime detection
    05:30  pm_snapshot         — portfolio manager metrics to Postgres
  Weekly (Sunday):
    01:00  ingest_weekly       — FMP fundamentals, Google Trends, IMF/WB

Entry: python -m marketos.pipelines.scheduler
"""
from __future__ import annotations

import logging
import os
import sys
import traceback
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("marketos.scheduler")

PG_DSN = os.environ.get("POSTGRES_DSN", "")
CODE_VERSION = os.environ.get("CODE_VERSION", "0.1.0")


# ── Job functions ─────────────────────────────────────────────────────────────

def ingest_news_job():
    logger.info("--- JOB: ingest_news ---")
    try:
        from marketos.data.fetchers.orchestrator import ingest_news
        r = ingest_news()
        logger.info(f"ingest_news done: {r}")
    except Exception:
        logger.error(traceback.format_exc())


def ingest_daily_job():
    logger.info("--- JOB: ingest_daily ---")
    try:
        from marketos.data.fetchers.orchestrator import (
            ingest_ohlcv, ingest_macro, ingest_bls, ingest_crypto, ingest_nse,
            ingest_weather, ingest_rbi, ingest_cboe, ingest_reddit, ingest_aviation,
            ingest_wikipedia, ingest_hiring,
        )
        for name, fn in [
            ("ohlcv", ingest_ohlcv),
            ("macro", ingest_macro),
            ("bls", ingest_bls),
            ("crypto", ingest_crypto),
            ("nse", ingest_nse),
            ("weather", ingest_weather),
            ("rbi", ingest_rbi),
            ("cboe", ingest_cboe),
            ("reddit", ingest_reddit),
            ("aviation", ingest_aviation),
            ("wikipedia", ingest_wikipedia),
            ("hiring", ingest_hiring),
        ]:
            try:
                r = fn()
                logger.info(f"  {name}: {r}")
            except Exception:
                logger.error(f"  {name} FAILED:\n{traceback.format_exc()}")
    except Exception:
        logger.error(traceback.format_exc())


def ingest_weekly_job():
    logger.info("--- JOB: ingest_weekly ---")
    try:
        from marketos.data.fetchers.orchestrator import (
            ingest_fundamentals, ingest_google_trends, ingest_global_macro,
            ingest_eia, ingest_oecd, ingest_finnhub, ingest_github, ingest_polygon,
            ingest_alphavantage, ingest_bis, ingest_comtrade, ingest_positioning,
            ingest_polymarket,
        )
        for name, fn in [
            ("fundamentals", ingest_fundamentals),
            ("finnhub", ingest_finnhub),
            ("polygon", ingest_polygon),
            ("global_macro", ingest_global_macro),
            ("eia", ingest_eia),
            ("oecd", ingest_oecd),
            ("github", ingest_github),
            ("alphavantage", ingest_alphavantage),
            ("bis", ingest_bis),
            ("comtrade", ingest_comtrade),
            ("positioning", ingest_positioning),
            ("polymarket", ingest_polymarket),
            # google_trends (pytrends) can hang for many minutes under rate-limit backoff —
            # run it last, with a hard timeout, so a hang can't block compute_features.
            ("google_trends", ingest_google_trends),
        ]:
            import concurrent.futures
            # Per-fetcher timeout budget. polygon's free tier throttles to 1 request every
            # 13s and the chain-proxy needs ~5 calls/symbol x 4 symbols = ~260s honest cost —
            # the prior 120s default was killing it before it could do any real work.
            _TIMEOUTS = {"google_trends": 90, "polygon": 300}
            try:
                timeout = _TIMEOUTS.get(name, 120)
                # Don't use the executor as a context manager: __exit__ calls
                # shutdown(wait=True), which blocks until the thread finishes —
                # defeating the timeout below. Let the thread become orphaned instead.
                ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                r = ex.submit(fn).result(timeout=timeout)
                ex.shutdown(wait=False)
                logger.info(f"  {name}: {r}")
            except concurrent.futures.TimeoutError:
                logger.error(f"  {name} TIMED OUT after {timeout}s — skipping")
            except Exception:
                logger.error(f"  {name} FAILED:\n{traceback.format_exc()}")
    except Exception:
        logger.error(traceback.format_exc())


def compute_features_job():
    logger.info("--- JOB: compute_features ---")
    try:
        from marketos.data.fetchers.yfinance_fetcher import fetch_ohlcv
        from marketos.data.lake import DataLake
        from marketos.db.store import MarketosStore
        from marketos.data.fetchers.orchestrator import _store_features, UNIVERSE, COMPUTE_UNIVERSE
        from marketos.features.registry import (
            compute_symbol_features, build_factor_panel, feature_count,
            FACTOR_EQUITY_SYMBOLS,
        )
        from marketos.data.fetchers.macro_fetcher import fetch_macro_panel
        import os

        lake = DataLake(os.environ.get("DATA_LAKE_ROOT", "data_lake"))
        store = MarketosStore(PG_DSN)

        # 1. Build the factor panel: equity/crypto factor returns + macro factor returns
        price_frames = {}
        for fkey, ftick in FACTOR_EQUITY_SYMBOLS.items():
            try:
                price_frames[fkey] = fetch_ohlcv(ftick, lake=lake, period="2y",
                                                 code_version=CODE_VERSION)
            except Exception:
                pass
        try:
            macro_panel = fetch_macro_panel()
        except Exception:
            macro_panel = None
        factor_panel = build_factor_panel(price_frames, macro_panel)

        # 2. Latest macro broadcast row (regime context attached to every symbol)
        macro_broadcast = {}
        if macro_panel is not None and not macro_panel.empty:
            from marketos.data.fetchers.macro_fetcher import compute_macro_features
            mf = compute_macro_features(macro_panel)
            if not mf.empty:
                macro_broadcast = {k: v for k, v in mf.iloc[-1].to_dict().items()
                                   if k not in ("asof_ts", "knowledge_ts")}

        # 3a. Read broadcast (global/market-level) features from store — these are attached
        #     to every symbol in the composite so they enter the cross-sectional model.
        _BROADCAST_MAP = {
            # (store_symbol, family) → key_prefix in composite
            ("_cboe",        "cboe_vol"):      "cboe_",
            ("_weather",     "weather"):        "wx_",
            ("_india",       "rbi"):            "rbi_",
            ("_aviation",    "aviation"):       "avia_",
            ("_energy",      "eia_energy"):     "eia_",
            ("_global",      "oecd"):           "oecd_",
            ("_macro",       "alphavantage"):   "av_",
            ("_macro",       "bis"):            "bis_",
            ("_nse",         "nse"):            "nse_",       # India market FII/DII/PCR
            ("_crypto",      "crypto"):         "crypto_",    # BTC/ETH/dominance
            ("_positioning", "positioning"):    "pos_",       # CFTC COT
            ("_github",      "github"):         "gh_",        # dev activity
            ("_hiring",      "hiring"):         "hiring_",    # open-req counts/recency/function mix
            ("_trends",      "google_trends"):  "gtrend_",    # search interest
            ("_trade",       "comtrade"):       "trade_",     # semiconductor/oil flows
            ("_polymarket",  "polymarket"):     "pm_",        # prediction-market implied probs
            ("_market",      "macro"):          "macro_",     # FRED/USDT/yield curve
            ("_market",      "macro_bls"):      "bls_",       # BLS employment
        }
        broadcast_features: dict[str, object] = {}
        for (bsym, bfam), prefix in _BROADCAST_MAP.items():
            try:
                fdata = store.get_latest_family(bsym, bfam)
                for k, v in fdata.items():
                    if k not in ("asof_ts", "knowledge_ts", "feature_family"):
                        broadcast_features[f"{prefix}{k}" if not k.startswith(prefix) else k] = v
            except Exception:
                pass

        # Broadcast GDELT theme tones (latest score per theme → market sentiment signal)
        try:
            from marketos.data.fetchers.orchestrator import THEMES
            cur = store._get_conn().cursor()
            cur.execute("""
                SELECT DISTINCT ON (theme) theme, theme_score, inputs
                FROM marketos.theme_scores
                ORDER BY theme, computed_at DESC
            """)
            rows = cur.fetchall()
            for row in (rows or []):
                t = str(row[0]).lower().replace(" ", "_")
                broadcast_features[f"gdelt_{t}_tone"] = float(row[1]) if row[1] is not None else None
                inp = row[2] if isinstance(row[2], dict) else {}
                broadcast_features[f"gdelt_{t}_mean"] = inp.get("gdelt_tone_mean_30d")
        except Exception:
            pass

        # 3b. Read per-symbol stored families (reddit, wiki, finnhub, options, fundamental, sentiment)
        _PER_SYMBOL_FAMILIES = ["reddit", "wiki_attention", "finnhub", "options",
                                 "fundamental", "sentiment"]
        per_symbol_stored = store.get_latest_families(COMPUTE_UNIVERSE, _PER_SYMBOL_FAMILIES)

        # 3. Compute the full composite vector per symbol (quant + aggressive extras)
        max_count = 0
        ok = 0
        symbol_vectors: dict[str, dict] = {}
        for sym in COMPUTE_UNIVERSE:
            try:
                ohlcv = fetch_ohlcv(sym, lake=lake, period="5y", code_version=CODE_VERSION)
                if ohlcv.empty:
                    continue
                vector = compute_symbol_features(
                    sym, ohlcv,
                    factor_panel=factor_panel,
                    macro_broadcast=macro_broadcast,
                )
                if not vector:
                    continue
                # Merge broadcast families (global signals)
                vector.update(broadcast_features)
                # Merge per-symbol stored families
                for fam in _PER_SYMBOL_FAMILIES:
                    fdata = per_symbol_stored.get((sym, fam), {})
                    for k, v in fdata.items():
                        if k not in ("asof_ts", "knowledge_ts", "feature_family", "symbol"):
                            vector[k] = v
                symbol_vectors[sym] = vector
                ok += 1
            except Exception:
                logger.warning(f"features {sym}: {traceback.format_exc()}")

        # 3c. Cross-sectional rank features — where each symbol stands vs the universe.
        #     Added *after* all individual vectors are built so they use contemporaneous peers.
        #     Each xrank_* is a percentile (0=worst, 1=best in universe, direction-adjusted).
        _RANK_KEYS_HIGHER_IS_BETTER = [
            # Momentum
            "roc_5", "roc_10", "roc_20", "roc_63", "roc_126",
            "mom_12_1", "mom_6_1", "mom_quality_12m", "mom_accel_12m",
            # Trend strength
            "adx_14", "adx_21",
            "lr_r2_20", "lr_r2_63", "lr_slope_20", "lr_slope_63",
            "trend_consist_5_63", "trend_consist_10_126",
            "macd_line", "macd_hist",
            # Oscillators (mid-level RSI often better for cross-rank)
            "rsi_14", "rsi_21",
            "cci_20", "ppo_hist",
            # Regime
            "vol_regime_zscore", "price_accel_5", "price_accel_20",
            # Fundamental quality
            "value_composite", "quality_composite", "growth_composite",
            "liquidity_composite", "efficiency_composite",
            "qmj_score", "fscore", "garp_score", "fundamental_momentum",
            "z_returnOnEquity", "z_grossProfitMargin", "z_freeCashFlowYield",
            "z_revenueGrowth", "z_netIncomeGrowth",
            "z_returnOnAssets", "z_operatingProfitMargin", "z_netProfitMargin",
            # Analyst/insider
            "analyst_score", "analyst_bull_frac", "z_insider_net_bias",
            "eps_sue_latest", "eps_sue_mean_4q", "eps_beat_streak",
            # Social/attention
            "reddit_mention_count", "reddit_sentiment_mean",
            "wiki_views_30d",
            # Cross-asset relative strength
            "xa_excess_ret_20d", "xa_excess_ret_63d",
            "xa_information_ratio_63",
            "xa_idio_vol_20",
            "xa_risk_on_score",
            # Gain/pain & drawdown
            "gain_pain_20", "gain_pain_63",
            "martin_ratio_63",
            # Yield & valuation
            "z_total_yield", "z_dividendYield", "z_earningsYield",
            # Position in range
            "pos_in_52w_range", "pos_in_13w_range",
            # Volume
            "volume_surge_20d", "volume_surge_63d",
            "obv_momentum_20", "vpt_slope_20",
            # Candle quality
            "lower_shadow_pct",
            # Time-series ranks (how current level ranks in own history)
            "tsrank_rsi_14", "tsrank_rsi_21", "tsrank_roc_20", "tsrank_roc_63",
            "tsrank_volume_surge_20d", "tsrank_volume_surge_63d",
            "tsrank_pos_in_52w_range",
            "tsrank_adx_14", "tsrank_adx_21",
            "tsrank_lr_slope_20", "tsrank_lr_slope_63",
            "tsrank_gain_pain_20",
            # Short-term return
            "mom_1_0",
            # Distance from range extremes (closer to high = stronger)
            "dist_52w_high", "dist_52w_low",
            "dist_13w_high", "dist_13w_low",
            # Recent-high flags (more 20/63d highs = stronger trend)
            "new_high_20", "new_high_63",
            # Fundamental combos
            "magic_formula_score", "z_altman_z",
            "value_quality_combo", "hq_signal", "garp_quality",
            # Money flow
            "mfi_14", "mfi_21",
            "cmf_20", "cmf_63",
            "ultimate_osc",
            # Trendiness (higher efficiency ratio = more trending)
            "efficiency_ratio_21", "efficiency_ratio_63",
            # Dollar volume acceleration
            "dollar_vol_zscore_20",
            # Variance ratio >1 = momentum signal
            "variance_ratio_4", "variance_ratio_8",
        ]
        _RANK_KEYS_LOWER_IS_BETTER = [
            # Volatility (lower = less risky = better rank)
            "volatility_21d", "volatility_63d", "volatility_126d",
            "atr_14_pct", "yang_zhang_vol_21", "yang_zhang_vol_63",
            "parkinson_vol_21", "parkinson_vol_63",
            "vol_of_vol_63", "vol_of_vol_21",
            # Tail risk
            "var_95_63", "cvar_95_63",
            "jump_count_20", "jump_count_63",
            "jump_max_20", "jump_max_63",
            "ulcer_63", "ulcer_14",
            "tail_ratio_63",
            # Fat tails (high kurtosis = fatter tails = riskier)
            "kurt_20", "kurt_63", "kurt_126",
            # Autocorrelation (high autocorr → trend exhaustion risk)
            "autocorr_1", "autocorr_5",
            # Illiquidity (lower = more liquid = better)
            "amihud_illiq_20", "amihud_illiq_63", "amihud_illiq_126",
            "roll_spread_20", "roll_spread_60",
            "kyle_lambda_20", "kyle_lambda_60",
            # Valuation (lower P/E = cheaper = better rank)
            "z_peRatio", "z_evToEbitda", "z_debtToEquity",
            "z_priceToBookRatio", "z_priceToSalesRatio",
            "junk_score",
            # Tracking error
            "xa_tracking_error_63",
            # Short-term reversal
            "reversal_5d", "reversal_10d",
            # Days since high (lower = more recent high = stronger trend)
            "days_since_52w_high",
            # Skewness (negative skew = bad)
            "skew_20", "skew_63", "skew_126",
            # New lows (fewer new lows = better trend health)
            "new_low_20", "new_low_63",
            # Candle bearish
            "upper_shadow_pct", "shooting_star_count_20", "shooting_star_count_63",
            # Choppiness (higher = more choppy = worse for trend-following)
            "choppiness_14", "choppiness_21",
            # Price z-score (extreme over-extension from mean)
            "zscore_close_20", "zscore_close_63",
        ]
        try:
            import numpy as np
            all_rank_keys = (
                [(k, True)  for k in _RANK_KEYS_HIGHER_IS_BETTER] +
                [(k, False) for k in _RANK_KEYS_LOWER_IS_BETTER]
            )
            # The rank POOL is the validated quant universe ONLY. Aggressive extras
            # (SOXL/BTC-USD) are ranked against this pool below but are never added to it —
            # adding a 3x ETF / crypto would shift every quant percentile and silently
            # mutate the validated cross-sectional signal.
            universe_vals: dict[str, list] = {k: [] for k, _ in all_rank_keys}
            for sym in UNIVERSE:
                vec = symbol_vectors.get(sym)
                if not vec:
                    continue
                for k, _ in all_rank_keys:
                    v = vec.get(k)
                    if v is not None:
                        try:
                            fv = float(v)
                            if np.isfinite(fv):
                                universe_vals[k].append((sym, fv))
                        except (TypeError, ValueError):
                            pass

            for sym in list(symbol_vectors.keys()):
                for k, higher_is_better in all_rank_keys:
                    pairs = universe_vals[k]
                    if len(pairs) < 2:
                        continue
                    sym_val = symbol_vectors[sym].get(k)
                    if sym_val is None:
                        continue
                    try:
                        sym_fv = float(sym_val)
                        if not np.isfinite(sym_fv):
                            continue
                    except (TypeError, ValueError):
                        continue
                    vals = sorted([v for _, v in pairs])
                    pct_rank = sum(1 for v in vals if v <= sym_fv) / len(vals)
                    # Flip so 1=best regardless of direction
                    symbol_vectors[sym][f"xrank_{k}"] = round(
                        pct_rank if higher_is_better else 1.0 - pct_rank, 4)
        except Exception:
            logger.warning(f"cross-sectional ranks: {traceback.format_exc()}")

        for sym, vector in symbol_vectors.items():
            try:
                n = feature_count(vector)
                max_count = max(max_count, n)
                _store_features(store, sym, "composite", vector)
            except Exception:
                logger.warning(f"store composite {sym}: {traceback.format_exc()}")

        logger.info(f"compute_features done: {ok} symbols, {max_count} features/symbol")

        # 4. Latent-state compression (Layer-3): cross-sectional PCA on the final vectors.
        try:
            from marketos.features.latent import compute_latent_factors
            per_symbol, market = compute_latent_factors(symbol_vectors)
            if not per_symbol.empty:
                for _, r in per_symbol.iterrows():
                    d = r.to_dict()
                    _store_features(store, d["symbol"], "latent", d)
            if market:
                _store_features(store, "_latent", "latent", market)
                logger.info(f"latent done: PC1 share={market.get('latent_pc1_share', 0):.3f}, "
                            f"participation={market.get('latent_participation_ratio', 0):.2f}")
        except Exception:
            logger.warning(f"latent computation: {traceback.format_exc()}")
    except Exception:
        logger.error(traceback.format_exc())


def prioritize_subspace_job():
    """Dynamic active-subspace prioritization — rank which features are 'live' now.

    Heavy (rebuilds full feature history + cross-sectional IC), so runs weekly. Surfaces the
    sparse active subspace and tracks Goodhart decay over time.
    """
    logger.info("--- JOB: prioritize_subspace ---")
    try:
        from marketos.data.fetchers.yfinance_fetcher import fetch_ohlcv
        from marketos.data.lake import DataLake
        from marketos.db.store import MarketosStore
        from marketos.data.fetchers.orchestrator import _store_features, UNIVERSE
        from marketos.features.technical import build_technical_features
        from marketos.features.subspace import compute_active_subspace, subspace_summary
        from marketos.regimes.hmm import detect_regimes
        import os
        import pandas as pd

        lake = DataLake(os.environ.get("DATA_LAKE_ROOT", "data_lake"))
        store = MarketosStore(PG_DSN)

        feature_hist, ohlcv_map = {}, {}
        for sym in UNIVERSE:
            try:
                ohlcv = fetch_ohlcv(sym, lake=lake, period="5y", code_version=CODE_VERSION)
                if ohlcv.empty:
                    continue
                ohlcv_map[sym] = ohlcv
                feature_hist[sym] = build_technical_features(ohlcv)
            except Exception:
                logger.warning(f"subspace hist {sym}: {traceback.format_exc()}")

        if not feature_hist:
            logger.warning("prioritize_subspace: no feature history built")
            return

        # Regime series from SPY for regime-conditional IC.
        regime_series = None
        try:
            if "SPY" in ohlcv_map:
                rets = ohlcv_map["SPY"]["close"].pct_change().dropna()
                reg = detect_regimes(rets)
                regime_series = reg["regime"]
                regime_series.index = pd.to_datetime(ohlcv_map["SPY"].index[-len(regime_series):])
        except Exception:
            logger.warning(f"subspace regime: {traceback.format_exc()}")

        table = compute_active_subspace(
            feature_hist, ohlcv_map, fwd_horizon=5, regime_series=regime_series, top_k=50,
        )
        if table.empty:
            logger.warning("prioritize_subspace: empty relevance table")
            return

        summary = subspace_summary(table, top_k=50)
        _store_features(store, "_subspace", "subspace", summary)
        top5 = ", ".join(table.head(5)["feature"].tolist())
        logger.info(f"prioritize_subspace done: {len(table)} scored, "
                    f"active={summary.get('subspace_size')}, "
                    f"decaying={summary.get('subspace_decaying_frac', 0):.0%}, top: {top5}")

        # Outcome-trained PLS latent factors, validated via purged walk-forward CV —
        # reuses the same ohlcv_map already fetched above, no extra network calls.
        try:
            from marketos.features.latent_supervised import fit_supervised_latent
            sup = fit_supervised_latent(ohlcv_map, n_components=4, n_splits=4, embargo_frac=0.03,
                                        regime_series=regime_series)
            if sup and not sup["latent"].empty:
                for _, r in sup["latent"].iterrows():
                    d = r.to_dict()
                    _store_features(store, d["symbol"], "latent_supervised", d)
                _store_features(store, "_latent", "latent_supervised", sup["market"])
                # Validation table: one broadcast row per (component, target) — the honest,
                # out-of-sample, uniqueness-weighted statistic. Flattened so it fits the
                # symbol/family/features(dict) store shape.
                val_row = {"asof_ts": sup["market"]["asof_ts"],
                          "knowledge_ts": sup["market"]["knowledge_ts"]}
                for _, vr in sup["validation"].iterrows():
                    key = f"c{int(vr['component'])}_{vr['target']}"
                    val_row[f"{key}_ic"] = round(float(vr["oos_ic_mean"]), 4)
                    val_row[f"{key}_ir"] = round(float(vr["oos_ic_ir"]), 3)
                    val_row[f"{key}_hitrate"] = round(float(vr["oos_ic_hit_rate"]), 3)
                _store_features(store, "_latent", "latent_validation", val_row)

                # Regime-conditional breakdown: does the validated signal hold its sign across
                # calm/neutral/stress regimes, or is the pooled IC above hiding a flip?
                rv = sup.get("regime_validation")
                if rv is not None and not rv.empty:
                    regime_row = {"asof_ts": sup["market"]["asof_ts"],
                                 "knowledge_ts": sup["market"]["knowledge_ts"]}
                    for _, vr in rv.iterrows():
                        key = f"r{int(vr['regime'])}_c{int(vr['component'])}_{vr['target']}"
                        regime_row[f"{key}_ic"] = round(float(vr["oos_ic_mean"]), 4)
                        regime_row[f"{key}_ir"] = round(float(vr["oos_ic_ir"]), 3)
                    _store_features(store, "_latent", "latent_regime_validation", regime_row)
                best = sup["market"].get("sup_latent_best_ic_ir", 0.0)
                stable = sup["market"].get("sup_latent_best_regime_sign_stable")
                logger.info(f"latent_supervised done: {sup['market'].get('sup_latent_n_symbols')} "
                            f"symbols, {sup['market'].get('sup_latent_n_components')} components, "
                            f"effective_n={sup['market'].get('sup_latent_effective_n', 0):.0f}, "
                            f"best_oos_ic_ir={best:.2f}, "
                            f"regime_sign_stable={stable}")

                # Market Memory: nearest-analog outcome distribution for each symbol's latest
                # supervised-latent snapshot. Distance is in outcome-trained latent space (not
                # raw features), regime-restricted when enough same-regime history exists.
                hist = sup.get("historical_panel")
                if hist is not None and not hist.empty:
                    from marketos.features.market_memory import find_analogs, summarize_analog_outcomes
                    current_regime = int(regime_series.iloc[-1]) if regime_series is not None and len(regime_series) else None
                    z_cols = [c for c in sup["latent"].columns if c.startswith("sup_z")]
                    n_with_analogs = 0
                    for _, srow in sup["latent"].iterrows():
                        sym = srow["symbol"]
                        current_z = {c: float(srow[c]) for c in z_cols}
                        matches = find_analogs(
                            current_z, hist, current_date=srow["asof_ts"], current_symbol=sym,
                            current_regime=current_regime, k=50, min_gap_days=60,
                        )
                        if matches.empty:
                            continue
                        analog_row = summarize_analog_outcomes(matches)
                        analog_row["asof_ts"] = srow["asof_ts"]
                        analog_row["knowledge_ts"] = srow["knowledge_ts"]
                        _store_features(store, sym, "analog", analog_row)
                        n_with_analogs += 1
                    logger.info(f"market_memory done: {n_with_analogs} symbols matched against "
                                f"{len(hist)} historical (date,symbol) snapshots, "
                                f"current_regime={current_regime}")
            else:
                logger.warning("latent_supervised: empty result")
        except Exception:
            logger.warning(f"latent_supervised: {traceback.format_exc()}")
    except Exception:
        logger.error(traceback.format_exc())


def screen_universe_job():
    """Daily technical screen across the BROAD candidate universe (60+ names) — yfinance
    OHLCV only, no rate-limited per-symbol fetchers. Does NOT touch the validated 19-name
    UNIVERSE or its cross-sectional fit; this is the separate "what else is moving"
    visibility layer (features/screening.py). Cheap enough to run daily even though
    compute_features (which fetches fundamentals/sentiment broadcasts) only covers the
    frozen pool.
    """
    logger.info("--- JOB: screen_universe ---")
    try:
        from marketos.data.fetchers.yfinance_fetcher import fetch_ohlcv
        from marketos.data.lake import DataLake
        from marketos.db.store import MarketosStore
        from marketos.data.fetchers.orchestrator import _store_features, CANDIDATE_UNIVERSE
        from marketos.features.screening import screen_universe
        import os

        lake = DataLake(os.environ.get("DATA_LAKE_ROOT", "data_lake"))
        store = MarketosStore(PG_DSN)
        flat = [sym for syms in CANDIDATE_UNIVERSE.values() for sym in syms]
        sector_of = {sym: sec for sec, syms in CANDIDATE_UNIVERSE.items() for sym in syms}

        candidates = {}
        for sym in flat:
            try:
                df = fetch_ohlcv(sym, lake=lake, period="6mo", code_version=CODE_VERSION)
                if not df.empty:
                    candidates[sym] = df
            except Exception:
                logger.warning(f"screen fetch {sym} failed:\n{traceback.format_exc()}")

        ranked = screen_universe(candidates)
        if ranked.empty:
            logger.warning("screen_universe: no candidates screened")
            return

        ranked["sector"] = ranked["symbol"].map(sector_of)
        now = datetime.now(timezone.utc)
        summary = {
            "asof_ts": now, "knowledge_ts": now,
            "n_screened": int(len(ranked)),
            "n_passed_liquidity": int(ranked["passes_liquidity"].sum()),
            "top_candidates": ranked[ranked["passes_liquidity"]].head(20)[
                ["symbol", "sector", "screen_score", "mom_63d", "trend_healthy"]
            ].to_dict(orient="records"),
        }
        _store_features(store, "_screen", "screen", summary)
        logger.info(f"screen_universe done: {summary['n_screened']} screened, "
                   f"{summary['n_passed_liquidity']} passed liquidity, "
                   f"top: {', '.join(c['symbol'] for c in summary['top_candidates'][:5])}")
    except Exception:
        logger.error(traceback.format_exc())


def screen_india_universe_job():
    """Daily technical screen across the Indian (NSE) candidate universe — for the india
    sleeve (AngelOne execution target). Built from `fetch_bhavcopy` (one HTTP call covers
    the WHOLE exchange per day, so this has no per-symbol API cost at all, unlike the US
    screen's per-symbol yfinance calls). INR liquidity threshold (₹2 crore/day average
    traded value) is intentionally much lower than the US $5M bar — appropriate for a
    retail account this small, where market impact from the account's OWN size is a
    non-concern; raise it later if real capital grows enough for that to change.
    """
    logger.info("--- JOB: screen_india_universe ---")
    try:
        from marketos.db.store import MarketosStore
        from marketos.data.fetchers.orchestrator import _store_features, INDIA_CANDIDATE_UNIVERSE
        from marketos.features.screening import fetch_angelone_history, screen_universe

        store = MarketosStore(PG_DSN)
        flat = [sym for syms in INDIA_CANDIDATE_UNIVERSE.values() for sym in syms]
        sector_of = {sym: sec for sec, syms in INDIA_CANDIDATE_UNIVERSE.items() for sym in syms}

        candidates = fetch_angelone_history(flat)
        if not candidates:
            logger.warning("screen_india_universe: no bhavcopy data fetched")
            return

        # ₹2 crore/day ≈ 20,000,000 in raw close*volume units; ₹50 minimum price filters
        # out illiquid penny stocks without excluding genuinely smaller, liquid names.
        ranked = screen_universe(candidates, min_avg_traded_value=20_000_000, min_price=50.0)
        if ranked.empty:
            logger.warning("screen_india_universe: no candidates screened")
            return

        ranked["sector"] = ranked["symbol"].map(sector_of)
        now = datetime.now(timezone.utc)
        summary = {
            "asof_ts": now, "knowledge_ts": now,
            "n_screened": int(len(ranked)),
            "n_passed_liquidity": int(ranked["passes_liquidity"].sum()),
            "top_candidates": ranked[ranked["passes_liquidity"]].head(20)[
                ["symbol", "sector", "screen_score", "mom_63d", "trend_healthy"]
            ].to_dict(orient="records"),
        }
        _store_features(store, "_screen_india", "screen", summary)
        logger.info(f"screen_india_universe done: {summary['n_screened']} screened, "
                   f"{summary['n_passed_liquidity']} passed liquidity, "
                   f"top: {', '.join(c['symbol'] for c in summary['top_candidates'][:5])}")
    except Exception:
        logger.error(traceback.format_exc())


def score_themes_job():
    logger.info("--- JOB: score_themes ---")
    try:
        from marketos.db.store import MarketosStore
        from marketos.themes.theme_hunter import ThemeHunter
        import pandas as pd

        store = MarketosStore(PG_DSN)
        # Build a minimal theme panel from cached features
        # Full implementation reads from Postgres features table
        hunter = ThemeHunter()
        # Placeholder: in full version pull from store.get_theme_features()
        now = datetime.now(timezone.utc)
        logger.info("score_themes done (stub — awaiting full feature store query)")
    except Exception:
        logger.error(traceback.format_exc())


def regime_update_job():
    logger.info("--- JOB: regime_update ---")
    try:
        from marketos.data.fetchers.yfinance_fetcher import fetch_ohlcv
        from marketos.regimes.hmm import detect_regimes
        from marketos.data.lake import DataLake
        from marketos.db.store import MarketosStore
        import os

        lake = DataLake(os.environ.get("DATA_LAKE_ROOT", "data_lake"))
        store = MarketosStore(PG_DSN)
        spy = fetch_ohlcv("SPY", lake=lake, period="2y", code_version=CODE_VERSION)
        if spy.empty:
            logger.warning("regime_update: no SPY data")
            return
        returns = spy["close"].pct_change().dropna()
        regimes = detect_regimes(returns)
        current = regimes.iloc[-1]
        store.upsert_regime(
            datetime.now(timezone.utc),
            "SPY",
            int(current["regime"]),
            str(current["regime_name"]),
        )
        logger.info(f"regime_update done: {current['regime_name']}")
    except Exception:
        logger.error(traceback.format_exc())


def regime_update_india_job():
    """Same regime detection as regime_update_job, off NIFTY50 instead of SPY —
    `regimes/hmm.py::detect_regimes` is fully generic (any return series in, regime labels
    out), so this is a straight reuse, not a reimplementation. Stored under market="NIFTY50"
    (a separate row from "SPY" in the same regime_labels table) so the conviction gate and
    any future india-sleeve logic can read the market-appropriate regime.
    """
    logger.info("--- JOB: regime_update_india ---")
    try:
        from marketos.data.fetchers.angelone_fetcher import login, fetch_ohlcv_history
        from marketos.data.fetchers.orchestrator import INDIA_FACTOR_INDICES
        from marketos.regimes.hmm import detect_regimes
        from marketos.db.store import MarketosStore
        import os

        store = MarketosStore(PG_DSN)
        auth = login()
        if auth.get("error"):
            logger.warning(f"regime_update_india: login failed: {auth['error']}")
            return
        nifty = fetch_ohlcv_history("NIFTY50", INDIA_FACTOR_INDICES["NIFTY50"],
                                    jwt_token=auth["jwt_token"],
                                    api_key=os.environ["ANGELONE_API_KEY"], years=2)
        if nifty.empty:
            logger.warning("regime_update_india: no NIFTY50 data")
            return
        returns = nifty["close"].pct_change().dropna()
        regimes = detect_regimes(returns)
        current = regimes.iloc[-1]
        store.upsert_regime(
            datetime.now(timezone.utc), "NIFTY50",
            int(current["regime"]), str(current["regime_name"]),
        )
        logger.info(f"regime_update_india done: {current['regime_name']}")
    except Exception:
        logger.error(traceback.format_exc())


def pm_snapshot_job():
    """Real PM cockpit snapshot for the QUANT sleeve: stored analog evidence -> allocate()
    -> sized positions -> marketos.portfolio_attribution. Replaces the prior heartbeat-only
    stub. Heartbeat write is kept (cheap, proves the worker is alive even if the snapshot
    logic throws) but is no longer the only thing this job does.
    """
    logger.info("--- JOB: pm_snapshot ---")
    now = datetime.now(timezone.utc)
    try:
        from marketos.db.store import MarketosStore
        from marketos.data.fetchers.orchestrator import _store_features, UNIVERSE, SECTOR_MAP
        store = MarketosStore(PG_DSN)
        _store_features(store, "_system", "heartbeat", {
            "asof_ts": now, "knowledge_ts": now, "status": "alive", "code_version": CODE_VERSION,
        })
    except Exception:
        logger.error(f"heartbeat write failed:\n{traceback.format_exc()}")
        return

    try:
        from marketos.portfolio.sleeves import load_sleeves
        from marketos.portfolio.opportunities import build_quant_snapshot, snapshot_to_attribution
        from marketos.config import Config

        cfg = Config.load()
        quant = load_sleeves(cfg.raw)["quant"]
        equity = quant.capital

        snap = build_quant_snapshot(
            store, universe=UNIVERSE, sector_map=SECTOR_MAP, limits=quant.limits, equity=equity,
        )
        if not snap.get("positions"):
            logger.info(f"pm_snapshot: no positions sized ({snap.get('reason', 'unknown')}) "
                       f"— writing empty/cash snapshot")

        attribution = snapshot_to_attribution(snap, equity)
        regime = None
        try:
            regime = store.get_latest_regime("SPY")
        except Exception:
            pass
        store.ensure_portfolio_attribution_table()
        store.upsert_portfolio_snapshot(
            now.date(), "quant_sleeve",
            weights=attribution["weights"], gross_exposure=attribution["gross_exposure"],
            cash_weight=attribution["cash_weight"], effective_n=attribution["effective_n"],
            top_positions=attribution["top_positions"], top_themes=attribution["top_themes"],
            regime_snapshot={"SPY": regime["regime"]} if regime else {},
        )
        logger.info(f"pm_snapshot done: {len(attribution['weights'])} positions, "
                   f"gross={attribution['gross_exposure']:.0%}, "
                   f"effective_n={attribution['effective_n']:.1f}")
    except Exception:
        logger.error(traceback.format_exc())


def aggressive_snapshot_job():
    """Live daily snapshot for the AGGRESSIVE sleeve — applies the LOCKED circuit_breaker
    policy (see models/aggressive_sleeve_backtest.py) to today's data. Parallel to
    pm_snapshot_job but for the wave-riding sleeve; written under strategy_name
    'aggressive_sleeve' so the two sleeves never share a row.
    """
    logger.info("--- JOB: aggressive_snapshot ---")
    try:
        from marketos.db.store import MarketosStore
        from marketos.portfolio.sleeves import load_sleeves
        from marketos.portfolio.aggressive_snapshot import (
            build_aggressive_snapshot, snapshot_to_attribution as agg_attribution,
        )
        from marketos.config import Config

        store = MarketosStore(PG_DSN)
        cfg = Config.load()
        aggressive = load_sleeves(cfg.raw)["aggressive"]
        equity = aggressive.capital

        snap = build_aggressive_snapshot(list(aggressive.universe), limits=aggressive.limits,
                                         equity=equity, store=store)
        attribution = agg_attribution(snap)
        store.ensure_portfolio_attribution_table()
        store.upsert_portfolio_snapshot(
            datetime.now(timezone.utc).date(), "aggressive_sleeve",
            weights=attribution["weights"], gross_exposure=attribution["gross_exposure"],
            cash_weight=attribution["cash_weight"], effective_n=attribution["effective_n"],
            top_positions=attribution["top_positions"], top_themes=attribution["top_themes"],
        )
        logger.info(f"aggressive_snapshot done: {snap['n_in']}/{snap['n_universe']} IN, "
                   f"gross={attribution['gross_exposure']:.0%}")
    except Exception:
        logger.error(traceback.format_exc())


def india_morning_briefing_job():
    """Pre-market India briefing — runs AFTER india_snapshot (07:00 IST), well before the
    9:15 AM IST open. Pushes via Telegram if TELEGRAM_BOT_TOKEN/CHAT_ID are set; logs the
    full briefing text regardless (so it's visible in `docker logs` even before Telegram
    is configured) — never silently does nothing.
    """
    logger.info("--- JOB: india_morning_briefing ---")
    try:
        from marketos.db.store import MarketosStore
        from marketos.notify.briefing import build_india_morning_briefing
        from marketos.notify.telegram import send_message

        store = MarketosStore(PG_DSN)
        text = build_india_morning_briefing(store)
        logger.info(f"india_morning_briefing content:\n{text}")
        result = send_message(text)
        if result["sent"]:
            logger.info("india_morning_briefing sent via Telegram")
        else:
            logger.warning(f"india_morning_briefing NOT sent via Telegram: {result['error']} "
                           f"(content was still logged above)")
    except Exception:
        logger.error(traceback.format_exc())


def us_evening_briefing_job():
    """US sleeves briefing — timed to the US market's actual open (9:35 AM ET, 5 minutes
    after the bell) using an explicit America/New_York timezone so it's automatically
    DST-correct (no hand-computed UTC offset to get wrong twice a year). By this point
    pm_snapshot/aggressive_snapshot have been fresh for many hours.
    """
    logger.info("--- JOB: us_evening_briefing ---")
    try:
        from marketos.db.store import MarketosStore
        from marketos.notify.briefing import build_us_evening_briefing
        from marketos.notify.telegram import send_message

        store = MarketosStore(PG_DSN)
        text = build_us_evening_briefing(store)
        logger.info(f"us_evening_briefing content:\n{text}")
        result = send_message(text)
        if result["sent"]:
            logger.info("us_evening_briefing sent via Telegram")
        else:
            logger.warning(f"us_evening_briefing NOT sent via Telegram: {result['error']} "
                           f"(content was still logged above)")
    except Exception:
        logger.error(traceback.format_exc())


def ingest_india_fundamentals_job():
    """Weekly (fundamentals/shareholding don't move daily, unlike price): Screener.in
    ratios + shareholding pattern, and Google News headline keyword-sentiment, for every
    INDIA_UNIVERSE name. Both sources are free, confirmed reachable, and were tested
    against real data before being wired in here (see project notes). A real 2-second
    sleep between symbols is deliberate — these are free community sites, not a paid API
    with a documented rate limit, so courtesy matters more here than speed.
    """
    logger.info("--- JOB: ingest_india_fundamentals ---")
    try:
        import time
        from marketos.db.store import MarketosStore
        from marketos.data.fetchers.orchestrator import _store_features, INDIA_UNIVERSE
        from marketos.data.fetchers.screener_fetcher import fetch_company_ratios, fetch_shareholding_pattern
        from marketos.data.fetchers.google_news_fetcher import fetch_headlines, keyword_sentiment_score

        store = MarketosStore(PG_DSN)
        ok_fund = ok_sent = 0
        for sym in INDIA_UNIVERSE:
            try:
                ratios = fetch_company_ratios(sym)
                shareholding = fetch_shareholding_pattern(sym)
                if ratios or shareholding:
                    combined = {**ratios, **shareholding}
                    _store_features(store, sym, "fundamental_india", combined)
                    ok_fund += 1
            except Exception:
                logger.warning(f"ingest_india_fundamentals ratios {sym}:\n{traceback.format_exc()}")
            time.sleep(2.0)

            try:
                heads = fetch_headlines(f"{sym} NSE")
                score = keyword_sentiment_score(heads)
                _store_features(store, sym, "sentiment_india", score)
                ok_sent += 1
            except Exception:
                logger.warning(f"ingest_india_fundamentals news {sym}:\n{traceback.format_exc()}")
            time.sleep(2.0)

        logger.info(f"ingest_india_fundamentals done: {ok_fund}/{len(INDIA_UNIVERSE)} fundamentals, "
                   f"{ok_sent}/{len(INDIA_UNIVERSE)} sentiment")
    except Exception:
        logger.error(traceback.format_exc())


def prioritize_subspace_india_job():
    """India equivalent of prioritize_subspace_job's latent+analog section — the actual
    evidence engine india_snapshot.py is currently missing (it sizes off technical screen
    rank only; see that module's docstring). `fit_supervised_latent` and
    `features/market_memory.py` are fully generic functions of any historical OHLCV panel
    — verified by direct reading, no India-specific assumptions found — so this is a
    straight reuse with India's candidate universe + AngelOne data + NIFTY50 regime,
    mirroring the US job's structure exactly. Heavy (rebuilds full multi-year history for
    every candidate), so weekly cadence like its US counterpart.

    Active-subspace ranking (the US job's `compute_active_subspace` section) is
    deliberately NOT included here yet — this pass prioritizes the higher-value
    latent/analog evidence; subspace ranking can be added later without restructuring this.
    """
    logger.info("--- JOB: prioritize_subspace_india ---")
    try:
        from marketos.db.store import MarketosStore
        from marketos.data.fetchers.angelone_fetcher import login, get_symbol_tokens, fetch_ohlcv_history
        from marketos.data.fetchers.orchestrator import _store_features, INDIA_UNIVERSE, INDIA_FACTOR_INDICES
        from marketos.regimes.hmm import detect_regimes
        from marketos.features.latent_supervised import fit_supervised_latent
        from marketos.features.technical import build_technical_features
        from marketos.features.subspace import compute_active_subspace, subspace_summary
        import os
        import pandas as pd

        store = MarketosStore(PG_DSN)
        auth = login()
        if auth.get("error"):
            logger.warning(f"prioritize_subspace_india: login failed: {auth['error']}")
            return
        jwt, key = auth["jwt_token"], os.environ["ANGELONE_API_KEY"]

        tokens = get_symbol_tokens(INDIA_UNIVERSE, cache_dir=os.environ.get("DATA_LAKE_ROOT", "data_lake"))

        ohlcv_map, feature_hist = {}, {}
        for sym in INDIA_UNIVERSE:
            token = tokens.get(sym)
            if not token:
                continue
            try:
                ohlcv = fetch_ohlcv_history(sym, token, jwt_token=jwt, api_key=key, years=5)
                if not ohlcv.empty:
                    ohlcv_map[sym] = ohlcv
                    feature_hist[sym] = build_technical_features(ohlcv)
            except Exception:
                logger.warning(f"subspace_india hist {sym}: {traceback.format_exc()}")

        if not ohlcv_map:
            logger.warning("prioritize_subspace_india: no history fetched")
            return

        # Regime series from NIFTY50 for regime-conditional IC (same role SPY plays in
        # the US job).
        regime_series = None
        try:
            nifty = fetch_ohlcv_history("NIFTY50", INDIA_FACTOR_INDICES["NIFTY50"],
                                        jwt_token=jwt, api_key=key, years=5)
            if not nifty.empty:
                rets = nifty["close"].pct_change().dropna()
                reg = detect_regimes(rets)
                regime_series = reg["regime"]
                regime_series.index = pd.to_datetime(nifty.index[-len(regime_series):])
        except Exception:
            logger.warning(f"subspace_india regime: {traceback.format_exc()}")

        # Active-subspace ranking — same role as the US job's equivalent section, reusing
        # compute_active_subspace/subspace_summary unchanged (fully generic, confirmed by
        # reading both — no US-specific assumptions found, unlike cross_asset.py's bug).
        try:
            table = compute_active_subspace(feature_hist, ohlcv_map, fwd_horizon=5,
                                            regime_series=regime_series, top_k=50)
            if not table.empty:
                summary = subspace_summary(table, top_k=50)
                _store_features(store, "_subspace_india", "subspace", summary)
                top5 = ", ".join(table.head(5)["feature"].tolist())
                logger.info(f"subspace_india done: {len(table)} scored, "
                           f"active={summary.get('subspace_size')}, top: {top5}")
        except Exception:
            logger.warning(f"subspace_india: {traceback.format_exc()}")

        sup = fit_supervised_latent(ohlcv_map, n_components=4, n_splits=4, embargo_frac=0.03,
                                    regime_series=regime_series)
        if not sup or sup["latent"].empty:
            logger.warning("prioritize_subspace_india: latent_supervised empty result")
            return

        for _, r in sup["latent"].iterrows():
            d = r.to_dict()
            _store_features(store, d["symbol"], "latent_supervised_india", d)
        _store_features(store, "_latent_india", "latent_supervised", sup["market"])

        val_row = {"asof_ts": sup["market"]["asof_ts"], "knowledge_ts": sup["market"]["knowledge_ts"]}
        for _, vr in sup["validation"].iterrows():
            key2 = f"c{int(vr['component'])}_{vr['target']}"
            val_row[f"{key2}_ic"] = round(float(vr["oos_ic_mean"]), 4)
            val_row[f"{key2}_ir"] = round(float(vr["oos_ic_ir"]), 3)
            val_row[f"{key2}_hitrate"] = round(float(vr["oos_ic_hit_rate"]), 3)
        _store_features(store, "_latent_india", "latent_validation", val_row)

        best = sup["market"].get("sup_latent_best_ic_ir", 0.0)
        logger.info(f"latent_supervised_india done: {sup['market'].get('sup_latent_n_symbols')} "
                   f"symbols, effective_n={sup['market'].get('sup_latent_effective_n', 0):.0f}, "
                   f"best_oos_ic_ir={best:.2f}")

        # Market Memory analog engine — same role as the US side.
        hist = sup.get("historical_panel")
        if hist is not None and not hist.empty:
            from marketos.features.market_memory import find_analogs, summarize_analog_outcomes
            current_regime = int(regime_series.iloc[-1]) if regime_series is not None and len(regime_series) else None
            z_cols = [c for c in sup["latent"].columns if c.startswith("sup_z")]
            n_with_analogs = 0
            for _, srow in sup["latent"].iterrows():
                sym = srow["symbol"]
                current_z = {c: float(srow[c]) for c in z_cols}
                matches = find_analogs(current_z, hist, current_date=srow["asof_ts"],
                                       current_symbol=sym, current_regime=current_regime,
                                       k=50, min_gap_days=60)
                if matches.empty:
                    continue
                analog_row = summarize_analog_outcomes(matches)
                analog_row["asof_ts"] = srow["asof_ts"]
                analog_row["knowledge_ts"] = srow["knowledge_ts"]
                _store_features(store, sym, "analog_india", analog_row)
                n_with_analogs += 1
            logger.info(f"market_memory_india done: {n_with_analogs} symbols matched against "
                       f"{len(hist)} historical (date,symbol) snapshots, current_regime={current_regime}")
    except Exception:
        logger.error(traceback.format_exc())


def india_snapshot_job():
    """Live daily snapshot for the INDIA sleeve — real ₹5,000 growth capital. Reads the
    day's stored screen (screen_india_universe_job), sizes the top few qualifying names.
    See portfolio/india_snapshot.py's docstring for the honest limitation: this sizes off
    technical screen rank only, NOT a backtested win-rate like the quant sleeve.
    """
    logger.info("--- JOB: india_snapshot ---")
    try:
        from marketos.db.store import MarketosStore
        from marketos.portfolio.sleeves import load_sleeves
        from marketos.portfolio.india_snapshot import build_india_snapshot, snapshot_to_attribution as india_attribution
        from marketos.config import Config

        store = MarketosStore(PG_DSN)
        cfg = Config.load()
        india = load_sleeves(cfg.raw)["india"]

        snap = build_india_snapshot(store, limits=india.limits, capital_inr=india.capital)
        if not snap.get("positions"):
            logger.info(f"india_snapshot: no positions sized ({snap.get('reason', 'unknown')})")

        attribution = india_attribution(snap)
        store.ensure_portfolio_attribution_table()
        store.upsert_portfolio_snapshot(
            datetime.now(timezone.utc).date(), "india_sleeve",
            weights=attribution["weights"], gross_exposure=attribution["gross_exposure"],
            cash_weight=attribution["cash_weight"], effective_n=attribution["effective_n"],
            top_positions=attribution["top_positions"], top_themes=attribution["top_themes"],
        )
        logger.info(f"india_snapshot done: {len(attribution['weights'])} positions, "
                   f"gross={attribution['gross_exposure']:.0%} of ₹{india.capital:,.0f}")
    except Exception:
        logger.error(traceback.format_exc())


# ── Scheduler entry point ─────────────────────────────────────────────────────

def run_all_now():
    """Run all daily jobs sequentially for the initial full pass."""
    logger.info("=== INITIAL FULL PASS ===")
    for name, fn in [
        ("ingest_news", ingest_news_job),
        ("ingest_daily", ingest_daily_job),
        ("ingest_weekly", ingest_weekly_job),
        ("compute_features", compute_features_job),
        ("screen_universe", screen_universe_job),
        ("screen_india_universe", screen_india_universe_job),
        ("score_themes", score_themes_job),
        ("regime_update", regime_update_job),
        ("regime_update_india", regime_update_india_job),
        ("pm_snapshot", pm_snapshot_job),
        ("aggressive_snapshot", aggressive_snapshot_job),
        ("india_snapshot", india_snapshot_job),
        ("india_morning_briefing", india_morning_briefing_job),
        ("us_evening_briefing", us_evening_briefing_job),
        ("prioritize_subspace", prioritize_subspace_job),
        ("prioritize_subspace_india", prioritize_subspace_india_job),
        ("ingest_india_fundamentals", ingest_india_fundamentals_job),
    ]:
        logger.info(f"Running {name}...")
        try:
            fn()
        except Exception:
            logger.error(f"{name} FAILED:\n{traceback.format_exc()}")
    logger.info("=== INITIAL PASS COMPLETE ===")


def main():
    if not PG_DSN:
        logger.error("POSTGRES_DSN not set — scheduler cannot persist job state")
        sys.exit(1)

    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
        from apscheduler.executors.pool import ThreadPoolExecutor
    except ImportError as e:
        logger.error(f"APScheduler not installed: {e}")
        sys.exit(1)

    jobstores = {"default": SQLAlchemyJobStore(url=PG_DSN)}
    executors = {"default": ThreadPoolExecutor(max_workers=4)}
    scheduler = BlockingScheduler(jobstores=jobstores, executors=executors,
                                  timezone="UTC")

    # Daily jobs
    scheduler.add_job(ingest_news_job,     "cron", hour=1,  minute=30, id="ingest_news",
                      replace_existing=True, misfire_grace_time=3600)
    scheduler.add_job(ingest_daily_job,    "cron", hour=2,  minute=0,  id="ingest_daily",
                      replace_existing=True, misfire_grace_time=3600)
    scheduler.add_job(compute_features_job,"cron", hour=3,  minute=0,  id="compute_features",
                      replace_existing=True, misfire_grace_time=3600)
    scheduler.add_job(screen_universe_job, "cron", hour=3,  minute=30, id="screen_universe",
                      replace_existing=True, misfire_grace_time=3600)
    scheduler.add_job(score_themes_job,    "cron", hour=4,  minute=0,  id="score_themes",
                      replace_existing=True, misfire_grace_time=3600)
    scheduler.add_job(regime_update_job,   "cron", hour=5,  minute=0,  id="regime_update",
                      replace_existing=True, misfire_grace_time=3600)
    scheduler.add_job(pm_snapshot_job,     "cron", hour=5,  minute=30, id="pm_snapshot",
                      replace_existing=True, misfire_grace_time=3600)
    scheduler.add_job(aggressive_snapshot_job, "cron", hour=5, minute=45, id="aggressive_snapshot",
                      replace_existing=True, misfire_grace_time=3600)

    # INDIA daily jobs — deliberately on a DIFFERENT schedule than their US counterparts
    # above, and registered with an EXPLICIT timezone rather than a hand-computed UTC
    # offset. NSE opens 9:15 AM IST; the US-mirrored times (03:45/05:15/06:00 UTC, chosen
    # to follow the US jobs' convention) land DURING Indian trading hours, not before — a
    # "pre-market briefing" was literally impossible on that schedule (caught designing
    # Phase 5's alert timing, not by reasoning about the cron times in isolation). These
    # three are self-contained (fetch live via AngelOne, no dependency on the US-side jobs
    # above) so they run on whatever schedule actually serves the Indian trading day —
    # completing by 7:00 AM IST, well ahead of the 9:15 AM open. India has no DST, so a
    # fixed IST time needs no special handling, but using the named timezone (rather than
    # its current UTC+5:30 offset baked into hour/minute numbers) is clearer to read and
    # impossible to get subtly wrong if this is ever touched again.
    scheduler.add_job(screen_india_universe_job, "cron", hour=6, minute=0, timezone="Asia/Kolkata",
                      id="screen_india_universe", replace_existing=True, misfire_grace_time=3600)
    scheduler.add_job(regime_update_india_job, "cron", hour=6, minute=30, timezone="Asia/Kolkata",
                      id="regime_update_india",
                      replace_existing=True, misfire_grace_time=3600)
    scheduler.add_job(india_snapshot_job, "cron", hour=7, minute=0, timezone="Asia/Kolkata",
                      id="india_snapshot", replace_existing=True, misfire_grace_time=3600)
    # India briefing: 7:30 AM IST — 30 min after india_snapshot completes, 1h45m before
    # the 9:15 AM IST NSE open.
    scheduler.add_job(india_morning_briefing_job, "cron", hour=7, minute=30, timezone="Asia/Kolkata",
                      id="india_morning_briefing", replace_existing=True, misfire_grace_time=3600)
    # US briefing: 9:35 AM ET — 5 min after the market bell. America/New_York handles
    # EST/EDT automatically; pm_snapshot (05:30 UTC)/aggressive_snapshot (05:45 UTC) are
    # many hours fresh by market open regardless of which side of DST it is.
    scheduler.add_job(us_evening_briefing_job, "cron", hour=9, minute=35, timezone="America/New_York",
                      id="us_evening_briefing", replace_existing=True, misfire_grace_time=3600)

    # Weekly jobs (Sunday)
    scheduler.add_job(ingest_weekly_job,   "cron", day_of_week="sun", hour=1, minute=0,
                      id="ingest_weekly", replace_existing=True, misfire_grace_time=7200)
    # Active-subspace prioritization after weekly ingest settles (Sunday 06:00 UTC)
    scheduler.add_job(prioritize_subspace_job, "cron", day_of_week="sun", hour=6, minute=0,
                      id="prioritize_subspace", replace_existing=True, misfire_grace_time=7200)
    scheduler.add_job(prioritize_subspace_india_job, "cron", day_of_week="sun", hour=6, minute=30,
                      id="prioritize_subspace_india", replace_existing=True, misfire_grace_time=7200)
    scheduler.add_job(ingest_india_fundamentals_job, "cron", day_of_week="sun", hour=7, minute=0,
                      id="ingest_india_fundamentals", replace_existing=True, misfire_grace_time=7200)

    # Run everything immediately on startup so no day-0 data gap
    run_all_now()

    logger.info("Scheduler started. Waiting for next trigger...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
