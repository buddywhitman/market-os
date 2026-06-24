# Quick Start — Market OS

Get up and running in 15 minutes. No paid APIs, no GPU required.

---

## 1. Setup (2 minutes)

```bash
cd /path/to/market-os

# Create virtual environment + install dependencies
make setup

# Start PostgreSQL + Redis (Docker)
make infra-up
```

**Expected output**:
```
✓ Virtual environment created
✓ Dependencies installed
✓ PostgreSQL and Redis started
```

---

## 2. Run Demo (5 minutes)

```bash
make demo
```

This runs end-to-end:
1. Fetches OHLCV from yfinance (no API key needed)
2. Computes features (technical, fundamental via free sources)
3. Trains XGBoost model (walk-forward)
4. Backtests with realistic costs
5. Reports expectancy metrics

**Expected output**:
```
Fetching data... ✓ (5 min)
Computing features... ✓ (1 min)
Training models... ✓ (30 sec)
Backtesting... ✓ (2 min)

Results:
─────────────────────────────
Sharpe ratio:        0.74
Win rate:            52%
Max drawdown:        -14%
Compounded return:    9.1%
```

---

## 3. View Dashboard (3 minutes)

```bash
make dashboard
```

Opens Streamlit at `http://localhost:8501`

**Dashboard tabs**:
- **Portfolio**: Current positions, allocations, Greeks
- **Backtest**: Historical returns, drawdown chart, monthly breakdown
- **Factors**: Feature importance, SHAP values
- **Alerts**: Risk warnings, regime changes, execution issues

---

## 4. Run Tests (2 minutes)

```bash
make test
```

Verifies:
- ✅ No black boxes (SHAP attribution present)
- ✅ No data leakage (forward-lookahead guard)
- ✅ Reproducibility (backtest bit-for-bit identical)
- ✅ Principles enforcement (all 6 rules checked)

**If tests fail**: Indicates violation of core principles. Fix before deploying.

---

## 5. Understand the Code Structure

```
market-os/
├── Makefile                    ← Run make setup|demo|dashboard|test
├── pyproject.toml              ← Dependencies
├── config/                     ← Universe, themes, costs, limits
├── data_lake/                  ← Immutable raw + derived data
├── src/marketos/
│   ├── data/                   ← Fetchers, parsers, catalog
│   ├── features/               ← Feature store + families
│   ├── models/                 ← XGBoost training, walk-forward
│   ├── backtest/               ← Engine, costs, expectancy
│   ├── risk/                   ← Regime, stops, sizing
│   ├── agents/                 ← Multi-agent decision system
│   └── dashboard/              ← Streamlit UI
├── tests/                      ← Principles + no-lookahead + expectancy
└── docs/                       ← Architecture, build order, etc.
```

---

## 6. Next Steps

### To understand the system:
1. Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (15 min)
2. Read [`docs/PRINCIPLES.md`](docs/PRINCIPLES.md) (20 min)
3. Read [`docs/RESEARCH_SUMMARY.md`](docs/RESEARCH_SUMMARY.md) (executive overview)

### To customize for your data:
```python
# In src/marketos/data/__init__.py, add your data source:
from marketos.data import DataLake

dl = DataLake(base_path='./data_lake')
dl.register_source(
    'my_ticks',
    fetcher=your_fetch_function,
    parser=your_parse_function,
    update_freq='daily'
)
```

### To add your own model:
```python
# In src/marketos/models/__init__.py:
from marketos.models import ModelRegistry

registry = ModelRegistry()
registry.register(
    'my_model',
    train_fn=your_train_fn,
    predict_fn=your_predict_fn,
    feature_deps=['technical', 'sentiment']
)
```

### To deploy to production:
See `docs/ARCHITECTURE.md` → Deployment section

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `docker: command not found` | Install Docker: https://docker.com/install |
| `make: command not found` | Install GNU Make: `apt-get install make` (Linux) or Homebrew (Mac) |
| `PostgreSQL connection refused` | Run `make infra-up` (containers may need restart) |
| Test failures | Check Git commit is in `CHANGELOG.md` (versioning requirement) |
| Dashboard doesn't load | Port 8501 may be busy; try `streamlit run ... --server.port 8502` |

---

## Key Files to Know

| File | Purpose |
|------|---------|
| `src/marketos/principles.py` | The 6 non-negotiable rules |
| `tests/test_principles.py` | Enforces those rules |
| `tests/test_no_lookahead.py` | Detects forward-leaking features |
| `config/default.yaml` | Universe, rebalance frequency, cost model |
| `src/marketos/backtest/engine.py` | Core backtesting logic |
| `docs/DATA_CONTRACTS.md` | Data format specifications |

---

## Glossary

**Walk-forward backtest**: Train on period A, test on period B (no future data leak). Repeat for sliding windows.

**SHAP**: Explainable AI technique showing feature-level contribution to each prediction.

**Sharpe ratio**: Return / volatility. >0.5 is decent, >1.0 is excellent.

**Max drawdown**: Largest peak-to-trough decline. -20% means portfolio lost 20% from high point.

**Regime**: Market state (trending vs. mean-revert, high vol vs. low vol). Models adapt to regime.

**Kelly criterion**: Optimal bet sizing = (win_rate - loss_rate) / odds. Often scaled down 25-50% for safety.

---

## Further Reading

- **Quantitative Trading**: "Quantitative Trading" by Ernest P. Chan
- **Risk Management**: "The Intelligent Investor" by Benjamin Graham (classics apply)
- **Feature Engineering**: "Feature Engineering for Machine Learning" by Zheng & Casari
- **Backtesting**: "Advances in Financial Machine Learning" by Marcos López de Prado

---

## Getting Help

- **Code questions**: Open an issue on GitHub or email pulkit.talks@gmail.com
- **Strategic questions**: Schedule a call for evaluation/partnership
- **Integration questions**: See `docs/TECHNICAL_SPECS.md` → Integration Patterns

---

**You're now ready to backtest, explore, and build!** 🚀
