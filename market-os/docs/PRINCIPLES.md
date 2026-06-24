# Core Principles & Enforcement

Market OS operates under six non-negotiable principles. These are **not aspirational**—they are runtime-checked invariants with automated enforcement.

---

## 1. No Black Boxes — Full Attribution Required

**Principle**: Every prediction must ship with feature-level attribution. We report **why**, not just **what**.

**Enforcement**:
```python
# ✗ NOT ALLOWED
prediction = model.predict(features)  # Returns only point estimate

# ✓ REQUIRED
from shap import TreeExplainer
shap_explainer = TreeExplainer(model)
prediction = model.predict(features)
shap_values = shap_explainer.shap_values(features)

# Attribution breakdown:
for feature, contribution in zip(feature_names, shap_values[0]):
    print(f"{feature}: {contribution:.4f}")
```

**Test coverage**: `tests/test_explainability.py`
- All models must implement `explain()` method
- SHAP or equivalent attribution on 100% of predictions
- Mean absolute contribution must sum to variance explained

**Real-world example**:
```
Model predicts AAPL: +15% in 30 days
Top contributors:
  • PE ratio (mean reversion):       +8.0%
  • Earnings growth trend:             +5.2%
  • Sector momentum:                   +3.5%
  • Sentiment (news):                 -1.7%  (contrarian signal)
```

---

## 2. Everything Versioned, Timestamped, Reproducible

**Principle**: Every data artifact, feature, model, and backtest result is immutable and traceable to its source.

**Enforcement**:

### Data immutability
```python
from marketos.data import immutable_hash

raw_data = fetch_ohlcv('AAPL', '2024-01-01')
content_hash = immutable_hash(raw_data.to_parquet_bytes())
# Store with: data/raw/{symbol}/{date}_{hash}.parquet

# Verify on load:
loaded = load_with_hash(path, expected_hash)  # Fails if corrupted
```

### Feature versioning
```python
# Every feature row includes:
features['_feature_version'] = 'v1.2.3'  # Code version
features['_compute_timestamp'] = pd.Timestamp.utcnow()  # When computed
features['_asof_date'] = asof_date  # When data became available
features['_dependencies'] = ['ohlcv:v1.0', 'macro:v2.1']  # Data lineage
```

### Model lineage
```yaml
# model_manifest.yml — stored alongside trained model
model:
  type: XGBoost
  version: v1.0.0
  trained_by: quant-agent-v2
  trained_at: 2024-06-22T10:30:00Z
  git_commit: abc123def456
  features:
    - name: rsi_14
      source: technical:v1.0
      importance: 0.15
    - name: sentiment_z_score
      source: sentiment:v2.1
      importance: 0.12
  backtests:
    - start: 2020-01-01
      end: 2024-01-01
      sharpe: 0.87
      max_dd: -0.18
      win_rate: 0.53
  trained_on:
    n_samples: 987000
    date_range: [2015-01-01, 2023-12-31]
    universe: [sp500, nasdaq100]
```

**Test coverage**: `tests/test_versioning.py`
- Hashes verified on every data load
- Model artifacts reject prediction if git_commit is unknown
- Backtest results include exact library versions (xgboost==2.x.x, pandas==2.x.x)

---

## 3. No Data Leakage. No Survivorship Bias. No Future Information.

**Principle**: A feature value at timestamp T reflects **only** information available before T.

**Enforcement**:

### Point-in-time feature computation
```python
# ✓ CORRECT: Compute metrics using only historical data
for asof_date in trading_dates:
    # Use data available BEFORE asof_date
    lookback_data = ohlcv[ohlcv.index < asof_date].iloc[-252:]
    rsi = compute_rsi(lookback_data)
    features.loc[asof_date, 'rsi_14'] = rsi

# ✗ WRONG: Using forward data
features['rsi_14'] = compute_rsi(ohlcv)  # May include future values
```

### Survivorship bias prevention
```python
# ✓ CORRECT: Include delisted, bankrupt, merged companies
available_at_date = {
    'AAPL': (date(2000, 1), date.today()),
    'BANKRUPT_TICKER': (date(2010, 1), date(2015, 6)),  # Include!
}

# ✗ WRONG: Only current universe
backtest_universe = sp500_current  # Missing delisted symbols

# Adjustment for returns:
returns['ticker'] = ticker
returns['available'] = date.isin(available_at_date[ticker])
backtest_metrics = compute_metrics(
    returns[returns['available'] == True]
)
```

### Forward-lookahead guard (automated CI check)
```python
# This test runs before every model training:
def test_no_lookahead():
    """Detect columns with future information"""
    # For each feature at timestamp T, compute correlation
    # with returns at T+1, T+2, ... T+20
    
    for feature in model_features:
        for lag in range(1, 21):
            future_return = returns.shift(-lag)
            correlation = feature.corr(future_return)
            
            # Flag suspiciously high correlations
            assert abs(correlation) < 0.15, \
                f"{feature} has {correlation:.3f} correlation " \
                f"with return at T+{lag} days (likely lookahead)"
```

**Test coverage**: `tests/test_no_lookahead.py`
- Runs automatically before backtest and training
- Detects both obvious (future OHLC) and subtle (index rebalance dates) leakage
- Fails CI if any feature has suspicious forward correlation

**Real example of what we catch**:
```python
# Found bug: P/E ratio included analyst earnings revisions
# published AFTER market close — not available at order time
# Fixed: Use only consensus estimates from 8:00 AM EST window
```

---

## 4. No Magical Claims — Report Distributions, Not Point Forecasts

**Principle**: Never claim "the model predicts X will be Y." Always report uncertainty.

**Enforcement**:

### Distributions, not predictions
```python
# ✗ NOT ALLOWED
print(f"AAPL will return {model.predict(features):.2%} next month")

# ✓ REQUIRED
returns_samples = model.predict_samples(features, n_samples=10000)
return_mean = returns_samples.mean()
return_std = returns_samples.std()
return_ci_low = np.percentile(returns_samples, 5)
return_ci_high = np.percentile(returns_samples, 95)

print(f"Expected return: {return_mean:.2%} ± {return_std:.2%}")
print(f"90% confidence interval: [{return_ci_low:.2%}, {return_ci_high:.2%}]")
```

### Quantile outputs preferred
```python
# Model outputs probabilities across return quantiles:
quantiles = model.predict_quantiles(features)
# quantiles = {
#     0.05: -0.08,   # 5th percentile: -8% (tail risk)
#     0.25: -0.02,   # 25th: -2%
#     0.50:  0.03,   # median: +3%
#     0.75:  0.09,   # 75th: +9%
#     0.95:  0.25    # 95th: +25% (tail opportunity)
# }
```

### Explicit uncertainty and confidence
```python
# Every decision includes confidence interval:
signal = {
    'action': 'BUY',
    'position_size': 0.03,  # 3% of portfolio
    'conviction': 0.62,     # 62% of max bet
    'ci_low': 0.48,         # 48th percentile of expected return
    'ci_high': 0.15,        # 15th percentile loss (downside)
    'win_rate': 0.52,       # 52% (barely above coin flip)
    'model_sharpe': 0.87    # Out-of-sample Sharpe
}
```

**Test coverage**: `tests/test_no_point_forecasts.py`
- Rejects models that output single point predictions
- Requires quantile or probability distribution output
- Verifies confidence intervals on backtest results

---

## 5. Feature Importance: No Feature Without Empirical Edge

**Principle**: If a feature doesn't improve out-of-sample expectancy, delete it. Ruthlessly.

**Enforcement**:

### Walk-forward importance validation
```python
from marketos.features import validate_feature_importance

for test_period in test_periods:
    # Train without feature
    model_without = train(X_without_feature, y)
    sharpe_without = evaluate(model_without, y_test)
    
    # Train with feature
    model_with = train(X_with_feature, y)
    sharpe_with = evaluate(model_with, y_test)
    
    # Feature must add positive edge
    edge = sharpe_with - sharpe_without
    if edge < 0.02:  # Minimum 2 bps of Sharpe improvement
        mark_for_deletion(feature)

# Clean up low-impact features
universe = remove_low_impact_features(universe)
```

### Feature cost-benefit
```python
# Some features are expensive to compute (LLM embeddings, satellite data)
# Must justify the cost:

feature_cost = {
    'technical_rsi':        0.001,  # bps per computation
    'fundamental_pe':       0.005,  # requires API call
    'sentiment_embedding':  0.05,   # LLM inference
    'alternative_sat_data': 0.50    # expensive data subscription
}

feature_edge = {
    'technical_rsi':        0.03,   # 3 bps of Sharpe improvement
    'fundamental_pe':       0.08,   # 8 bps improvement
    'sentiment_embedding':  0.15,   # 15 bps improvement
    'alternative_sat_data': 0.04    # 4 bps improvement — too cheap!
}

# Keep only if ROI > 3.0x
for feature in features:
    roi = feature_edge[feature] / feature_cost[feature]
    if roi < 3.0:
        delete_feature(feature)
```

**Test coverage**: `tests/test_feature_edge.py`
- Compares with/without feature on validation set
- Tracks feature importance stability across market regimes
- Flags features with declining importance over time

---

## 6. Simplicity Over Complexity. Evidence Over Narrative.

**Principle**: Prefer interpretable models, simple rules, and empirical proof over sophisticated but black-box approaches.

**Enforcement**:

### Bias toward interpretability
```python
# ✓ PREFERRED: Tree-based (XGBoost, LightGBM)
#  - Feature importances are local and global
#  - Decisions are explicit
#  - Fast inference

# ⚠ TOLERATED: Linear (Ridge, Lasso)
#  - Fully interpretable, but less expressive

# ✗ LAST RESORT: Deep neural networks
#  - Only if interpretability layer (SHAP) added
#  - Only if sharply better out-of-sample than tree model
#  - Must pass blind comparison test
```

### Complexity penalty
```python
# Model selection includes regularization for complexity:
# Not: AIC = 2k - 2ln(L)
# Instead: AIC = 2k - 2ln(L) + 5k^2  # Penalty for high k
#          (encourages parsimony)

# Model candidates must clear hurdle rate:
#   Simple model sharpe:      0.85
#   Complex model must beat:  0.85 + (0.05 * complexity_penalty)
#
#   Only adopt complex model if sharpe > 0.91+
```

### Evidence standard
```python
# Decision: Should we add LLM-based sentiment signals?

# Analysis required:
1. Walk-forward backtest on sentiment alone
2. Correlation with price data (avoid overfitting to same signal)
3. Impact when combined with existing factors
4. Cost of LLM embeddings vs. edge gained
5. Stability across market regimes (bull, bear, sideways)
6. Comparison to simpler sentiment baseline (VIX, put/call ratio)

# Decision only made if:
#  - Edge consistent across all 5 test periods
#  - Cost/benefit > 2.0x
#  - No model overfitting detected
```

---

## Enforcement in CI/CD

Every commit triggers these checks:

```bash
# 1. Principles test — highest priority
pytest tests/test_principles.py
# Fails if: black box pred, no versioning, lookahead, magical claims, etc.

# 2. Data integrity
pytest tests/test_data_contracts.py
# Fails if: hash mismatch, missing asof_date, survivorship bias

# 3. No lookahead — sophisticated detector
pytest tests/test_no_lookahead.py
# Scans for forward-leaking features, looks ahead in backtest

# 4. Expectancy validation
pytest tests/test_expectancy.py
# Verifies out-of-sample sharpe, drawdown, win rates are as reported

# 5. Reproducibility
pytest tests/test_reproducibility.py
# Re-runs backtest with same code → identical results (bit-for-bit)

# If any test fails, PR is blocked until fixed
```

---

## Violations Caught in the Wild

### Example 1: Hidden forward information
**Violation**: News sentiment included earnings announcement *dates*.  
**Impact**: Model "predicts" the move after the announcement.  
**Fix**: Use only news published *before* prediction timestamp.

### Example 2: Survivorship bias
**Violation**: Backtest used current S&P 500 index (delisted companies missing).  
**Impact**: Sharpe inflated by 0.3 due to successful survivorship.  
**Fix**: Include delisted tickers with actual removal dates.

### Example 3: Look-ahead in calendar
**Violation**: Fed meeting feature included *actual Fed decision*.  
**Impact**: Model correlated with future price moves (0.32 correlation at T+1).  
**Fix**: Use only pre-announced meeting dates, not decisions.

### Example 4: Point forecast marketing
**Violation**: Dashboard showed "Model predicts +15% for XYZ."  
**Impact**: Clients thought prediction was certain.  
**Fix**: Show distribution, highlight 40% drawdown scenario, confidence interval.

---

## Summary: How to Verify Compliance

**For researchers**:
1. Run `make test` — all checks must pass
2. Review backtest report for "win rate", "max drawdown", "Monte Carlo 95% confidence interval"
3. Request SHAP attribution for any prediction
4. Ask: "What would this look like in a different market regime?"

**For teams integrating Market OS**:
1. Audit `test_principles.py` — understand the enforcement level
2. Run your own data through feature validation
3. Verify CI passes on your branch
4. Cross-validate against your proprietary models

**For compliance/risk teams**:
1. Data lineage: Trace any position back to raw source and timestamp
2. Attribution: Every trade decision includes feature-level explanation
3. Audit trail: Immutable log of all computations and decisions
4. Scenario analysis: Monte Carlo shows tail risk distribution

---

**Version**: 1.0  
**Last reviewed**: June 2026  
**Enforcement**: Automated via CI, manual via code review
