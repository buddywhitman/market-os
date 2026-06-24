# Contributing to Market OS Research

Thank you for your interest in contributing! This document outlines how to participate in the project.

---

## Code of Conduct

- Be respectful and constructive
- Focus on the work, not the person
- Help others learn
- Maintain scientific rigor and reproducibility

---

## Getting Started

### 1. Fork & Clone
```bash
git clone https://github.com/YOUR_USERNAME/market-os-research.git
cd market-os-research
git remote add upstream https://github.com/buddywhitman/market-os-research.git
```

### 2. Set Up Development Environment
```bash
cd market-os
make setup
make infra-up
```

### 3. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

---

## Development Workflow

### Before Making Changes

1. **Read the principles** — [`market-os/docs/PRINCIPLES.md`](market-os/docs/PRINCIPLES.md)
2. **Understand the architecture** — [`market-os/docs/ARCHITECTURE.md`](market-os/docs/ARCHITECTURE.md)
3. **Check for existing issues** — Don't duplicate effort

### Making Changes

**Never violate the 6 core principles:**

1. **No black boxes** — Add SHAP attribution to any new predictions
2. **Full versioning** — All changes must be versioned and reproducible
3. **No lookahead** — Features must use only data available at decision time
4. **No point forecasts** — Report distributions, not point estimates
5. **Feature edge required** — Only add features that improve out-of-sample Sharpe
6. **Simplicity over complexity** — Prefer interpretable models

### Testing

```bash
cd market-os

# Run ALL tests (this is required)
make test

# Run specific test suite
pytest tests/test_principles.py -v
pytest tests/test_no_lookahead.py -v

# Run with coverage
pytest tests/ --cov=src/marketos --cov-report=html
```

**All tests must pass before opening a PR.** CI/CD will verify:
- ✅ Principles enforcement
- ✅ No lookahead detection
- ✅ Linting (Black, isort, Ruff)
- ✅ Documentation completeness

### Code Style

```bash
# Format with Black
black --line-length 100 market-os/src/ market-os/tests/

# Sort imports with isort
isort market-os/src/ market-os/tests/

# Lint with Ruff
ruff check market-os/src/ market-os/tests/
```

---

## Types of Contributions

### Bug Fixes
- Clear title: "Fix: [description]"
- Link to issue if it exists
- Include test case that reproduces the bug
- Verify fix doesn't break other tests

### New Features
- Discuss in an issue first (avoid wasted effort)
- Implement with full SHAP attribution
- Walk-forward backtest to show edge
- Cost-benefit analysis if expensive
- Update relevant documentation

### New Data Sources
- Add fetcher in `src/marketos/data/fetchers/`
- Implement error handling and retries
- Add to `tests/` with mock data
- Document in `docs/DATA_CONTRACTS.md`

### New Features / Signals
- Add to appropriate family in `src/marketos/features/`
- Compute point-in-time (no lookahead)
- Walk-forward test for edge
- Add to `RESEARCH_LOG.md` with findings

### Documentation
- Clarify ambiguities
- Add examples
- Fix typos
- Link related sections

### Research / Analysis
- Report findings in `market-os/docs/RESEARCH_LOG.md`
- Include: date, question, method, result, conclusion
- Attach Jupyter notebook or backtest results
- Propose improvements if applicable

---

## Pull Request Process

### 1. Before Submission

```bash
# Sync with upstream
git fetch upstream
git rebase upstream/main

# Run full test suite locally
cd market-os && make test

# Format code
black --line-length 100 src/ tests/
isort src/ tests/
```

### 2. Open PR with Template

**Title**: `[Type] Brief description`
- Types: `Feature`, `Fix`, `Docs`, `Refactor`, `Test`, `Research`
- Examples: `[Feature] Add regime-aware position sizing`, `[Fix] Correct lookahead in RSI calculation`

**Description** (use the template):
```markdown
## Summary
[What does this change do? Why?]

## Type
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Research finding
- [ ] Refactor

## Testing
- [ ] Added tests for new code
- [ ] All tests pass (`make test`)
- [ ] Walk-forward backtest shows edge (if applicable)

## Checklist
- [ ] Code follows style guidelines (Black, isort, Ruff)
- [ ] No black boxes (SHAP attribution added if needed)
- [ ] No lookahead (verified against test_no_lookahead.py)
- [ ] Documentation updated
- [ ] Principles still enforced

## Related Issues
Closes #123 (if applicable)
```

### 3. Review Process

- CI/CD must pass (linting, tests, security)
- Code review: 1-2 approvals
- No conflicts with main branch
- Documentation reviewed if applicable

---

## Common Contribution Patterns

### Adding a Feature to Feature Store

```python
# market-os/src/marketos/features/technical.py

from marketos.features.registry import FeatureRegistry

registry = FeatureRegistry()

@registry.register(
    name="my_signal",
    families=["technical"],
    lookahead_days=1  # Enforce minimum delay
)
def compute_my_signal(ohlcv: pd.DataFrame) -> pd.Series:
    """
    Compute custom signal.
    
    Args:
        ohlcv: OHLCV data (columns: open, high, low, close, volume)
    
    Returns:
        Series with signal values, indexed by date
    """
    # Use only historical data (no future information)
    signal = ohlcv['close'].rolling(20).mean()
    return signal
```

Then walk-forward test:

```python
# market-os/tests/test_my_signal.py

def test_my_signal_edge():
    """Verify signal improves out-of-sample Sharpe"""
    from marketos.backtest import Backtester
    from marketos.features import compute_my_signal
    
    # Load test data (never seen by model)
    ohlcv = load_test_data('2020-01-01', '2024-01-01')
    
    # Compute signal
    signal = compute_my_signal(ohlcv)
    
    # Backtest with signal
    bt = Backtester(ohlcv=ohlcv, signal=signal)
    results = bt.run()
    
    # Verify edge
    assert results.sharpe > 0.5, "Signal doesn't improve Sharpe"
```

### Adding Data Source

```python
# market-os/src/marketos/data/fetchers/my_fetcher.py

from marketos.data.fetchers import BaseFetcher

class MyFetcher(BaseFetcher):
    """Fetch data from My API."""
    
    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """
        Fetch OHLCV data.
        
        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        try:
            # Call API
            data = self._call_my_api(symbol, start, end)
            # Parse and normalize
            df = self._parse(data)
            return df
        except Exception as e:
            self.logger.error(f"Failed to fetch {symbol}: {e}")
            raise
    
    def _parse(self, raw_data: dict) -> pd.DataFrame:
        """Normalize to standard format."""
        df = pd.DataFrame(raw_data)
        df.rename(columns={...}, inplace=True)  # Map API columns
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
```

---

## Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` (added automatically)
- Release notes (major contributors)
- Commit history (Git attribution)

---

## Questions?

- **GitHub Issues** — For bugs and feature requests
- **Discussions** — For ideas and design questions (coming soon)
- **Email** — pulkit.talks@gmail.com for partnership inquiries

---

## License

By contributing, you agree that your contributions are licensed under the same license as the project (see `LICENSE`).

---

Thank you for helping advance quantitative trading research! 🚀
