# Documentation Index — Complete Reference

A comprehensive guide to navigating the research portfolio and finding information quickly.

---

## Quick Navigation

### 🚀 Get Started (15 minutes)
1. **New to this project?** → [`market-os/QUICKSTART.md`](market-os/QUICKSTART.md)
2. **Decision-maker review** → [`PORTFOLIO_OVERVIEW.md`](PORTFOLIO_OVERVIEW.md)
3. **Run demo** → `cd market-os && make demo`

### 📚 Understand the System (2-4 hours)
1. [`market-os/README.md`](market-os/README.md) — Project overview
2. [`market-os/docs/RESEARCH_SUMMARY.md`](market-os/docs/RESEARCH_SUMMARY.md) — Results & benchmarks
3. [`market-os/docs/ARCHITECTURE.md`](market-os/docs/ARCHITECTURE.md) — System design
4. [`market-os/docs/PRINCIPLES.md`](market-os/docs/PRINCIPLES.md) — Enforcement rules

### 🔧 Build & Integrate (4-8 hours)
1. [`market-os/docs/BUILD_ORDER.md`](market-os/docs/BUILD_ORDER.md) — Dependency graph
2. [`market-os/docs/TECHNICAL_SPECS.md`](market-os/docs/TECHNICAL_SPECS.md) — Integration guide
3. [`market-os/docs/DATA_CONTRACTS.md`](market-os/docs/DATA_CONTRACTS.md) — Data formats
4. Code: `src/marketos/` for implementation details

---

## By Role

### Portfolio Manager / CIO
**Goal**: Decide if this is valuable for your organization

**Read** (30 minutes):
- [`PORTFOLIO_OVERVIEW.md`](PORTFOLIO_OVERVIEW.md) — This file
- [`market-os/docs/RESEARCH_SUMMARY.md`](market-os/docs/RESEARCH_SUMMARY.md) — Results & benchmarks

**Next**: Schedule demo (30 min) and technical discussion (1-2 hours)

### Quantitative Researcher
**Goal**: Understand the system deeply, evaluate methodology, consider contributions

**Read** (2-4 hours):
1. [`market-os/README.md`](market-os/README.md)
2. [`market-os/docs/RESEARCH_SUMMARY.md`](market-os/docs/RESEARCH_SUMMARY.md)
3. [`market-os/docs/ARCHITECTURE.md`](market-os/docs/ARCHITECTURE.md)
4. [`market-os/docs/PRINCIPLES.md`](market-os/docs/PRINCIPLES.md)

**Explore** (2-4 hours):
- Run `make demo` and examine backtest output
- Review `tests/test_principles.py` to understand enforcement
- Walk through `src/marketos/features/` for feature engineering patterns
- Study `src/marketos/backtest/engine.py` for validation logic

### Software Engineer / DevOps
**Goal**: Understand architecture, deployment, integration points

**Read** (1-2 hours):
1. [`market-os/README.md`](market-os/README.md)
2. [`market-os/docs/TECHNICAL_SPECS.md`](market-os/docs/TECHNICAL_SPECS.md)
3. [`market-os/docs/BUILD_ORDER.md`](market-os/docs/BUILD_ORDER.md)

**Explore** (2-4 hours):
- Review `Makefile` for build/deploy patterns
- Check `docker-compose.yml` for infrastructure stack
- Walk through `src/marketos/integration/` for extension points
- Read `pyproject.toml` for dependencies

### Compliance / Risk Officer
**Goal**: Verify integrity, audit trail, risk controls

**Read** (1-2 hours):
1. [`market-os/docs/PRINCIPLES.md`](market-os/docs/PRINCIPLES.md) — Core rules
2. [`market-os/docs/TECHNICAL_SPECS.md`](market-os/docs/TECHNICAL_SPECS.md) → "Compliance & Audit" section
3. [`research-artifacts/README.md`](research-artifacts/README.md) → Reproducibility section

**Verify**:
- Run `make test` to confirm principles enforcement
- Check `tests/test_no_lookahead.py` for data integrity validation
- Review audit trail format in backtest output
- Validate that SHAP attributions are required for all predictions

---

## By Topic

### Backtesting & Validation
- [`market-os/docs/ARCHITECTURE.md`](market-os/docs/ARCHITECTURE.md) → "Layer 5: Backtesting & Expectancy"
- [`market-os/docs/PRINCIPLES.md`](market-os/docs/PRINCIPLES.md) → Principle #3 (no lookahead)
- [`src/marketos/backtest/engine.py`](market-os/src/marketos/backtest/engine.py)
- [`tests/test_no_lookahead.py`](market-os/tests/test_no_lookahead.py)

### Feature Engineering
- [`market-os/docs/ARCHITECTURE.md`](market-os/docs/ARCHITECTURE.md) → "Layer 2: Feature Store"
- [`src/marketos/features/`](market-os/src/marketos/features/)
- [`src/marketos/agents/utils/`](market-os/src/marketos/agents/utils/) for signal definitions

### Risk Management
- [`market-os/docs/ARCHITECTURE.md`](market-os/docs/ARCHITECTURE.md) → "Layer 4: Risk & Portfolio"
- [`market-os/docs/TECHNICAL_SPECS.md`](market-os/docs/TECHNICAL_SPECS.md) → "Risk modules" table
- [`src/marketos/risk/`](market-os/src/marketos/risk/)

### Explainability
- [`market-os/docs/PRINCIPLES.md`](market-os/docs/PRINCIPLES.md) → Principle #1 (no black boxes)
- [`market-os/docs/TECHNICAL_SPECS.md`](market-os/docs/TECHNICAL_SPECS.md) → Model interpretability section
- [`src/marketos/models/`](market-os/src/marketos/models/) for SHAP integration

### Multi-Agent Systems
- [`market-os/docs/ARCHITECTURE.md`](market-os/docs/ARCHITECTURE.md) → "Layer 6: Agents"
- [`market-os/docs/TECHNICAL_SPECS.md`](market-os/docs/TECHNICAL_SPECS.md) → Agent framework
- [`TradingAgents/README.md`](TradingAgents/README.md) for reference implementation
- [`src/marketos/agents/`](market-os/src/marketos/agents/)

### Data Pipelines
- [`market-os/docs/DATA_CONTRACTS.md`](market-os/docs/DATA_CONTRACTS.md)
- [`market-os/docs/ARCHITECTURE.md`](market-os/docs/ARCHITECTURE.md) → "Layer 1: Data Lake"
- [`src/marketos/data/`](market-os/src/marketos/data/)

### Integration with Proprietary Systems
- [`market-os/docs/TECHNICAL_SPECS.md`](market-os/docs/TECHNICAL_SPECS.md) → "Integration Patterns"
- [`src/marketos/integration/`](market-os/src/marketos/integration/) for examples
- [`docs/BUILD_ORDER.md`](market-os/docs/BUILD_ORDER.md) for module dependencies

---

## By Question

**Q: How do I run this?**  
→ [`market-os/QUICKSTART.md`](market-os/QUICKSTART.md)

**Q: What are the results?**  
→ [`market-os/docs/RESEARCH_SUMMARY.md`](market-os/docs/RESEARCH_SUMMARY.md)

**Q: How do I know this isn't cheating (lookahead bias)?**  
→ [`market-os/docs/PRINCIPLES.md`](market-os/docs/PRINCIPLES.md) + `make test`

**Q: Can I trust the backtest numbers?**  
→ [`market-os/docs/PRINCIPLES.md`](market-os/docs/PRINCIPLES.md) → Principle #2 (versioning/reproducibility)

**Q: How does this compare to [my system]?**  
→ [`market-os/docs/RESEARCH_SUMMARY.md`](market-os/docs/RESEARCH_SUMMARY.md) → Competitive Differentiation

**Q: How do I integrate this with my data?**  
→ [`market-os/docs/TECHNICAL_SPECS.md`](market-os/docs/TECHNICAL_SPECS.md) → "Bring Your Own Data"

**Q: How do I add my own models?**  
→ [`market-os/docs/TECHNICAL_SPECS.md`](market-os/docs/TECHNICAL_SPECS.md) → "Bring Your Own Model"

**Q: What's the computational cost?**  
→ [`market-os/docs/TECHNICAL_SPECS.md`](market-os/docs/TECHNICAL_SPECS.md) → Performance & Scalability

**Q: Can we run this in production?**  
→ [`market-os/docs/ARCHITECTURE.md`](market-os/docs/ARCHITECTURE.md) → Layer 7 (Deployment)

**Q: What license is this under?**  
→ See `LICENSE` file in repository root

---

## File Structure Quick Reference

```
/home/buddywhitman/
├── README.md                      ← Start here (portfolio intro)
├── PORTFOLIO_OVERVIEW.md          ← For decision-makers (this role-based doc)
├── DOCUMENTATION_INDEX.md         ← This file (navigation guide)
│
├── market-os/                     ← Main project (Market OS)
│   ├── README.md
│   ├── QUICKSTART.md              ← 15-min setup
│   ├── Makefile                   ← Commands: setup, demo, test, dashboard
│   ├── docs/
│   │   ├── ARCHITECTURE.md        ← System design (7 layers)
│   │   ├── BUILD_ORDER.md         ← Dependencies
│   │   ├── DATA_CONTRACTS.md      ← Data formats
│   │   ├── PRINCIPLES.md          ← 6 enforcement rules
│   │   ├── RESEARCH_SUMMARY.md    ← Results & benchmarks
│   │   ├── RESEARCH_LOG.md        ← Dated experiments
│   │   └── TECHNICAL_SPECS.md     ← Integration guide
│   ├── src/marketos/              ← Source code
│   ├── tests/                     ← Test suite
│   └── config/                    ← Configuration
│
├── TradingAgents/                 ← Reference (Multi-agent LLM)
│   ├── README.md
│   └── tradingagents/             ← Source
│
├── research-artifacts/            ← Papers, benchmarks, case studies
│   └── README.md
│
└── [other projects]
    └── (OpenLane, aiter, mpi_workspace, etc.)
```

---

## Documentation Hierarchy

```
Level 1: Portfolio & Entry Points
  ├── README.md (portfolio intro)
  ├── PORTFOLIO_OVERVIEW.md (commercial positioning)
  └── DOCUMENTATION_INDEX.md (this file)

Level 2: Project Overview
  ├── market-os/README.md (project thesis)
  ├── market-os/QUICKSTART.md (get running)
  └── market-os/docs/RESEARCH_SUMMARY.md (results)

Level 3: Deep Dives
  ├── market-os/docs/ARCHITECTURE.md (design)
  ├── market-os/docs/PRINCIPLES.md (enforcement)
  └── market-os/docs/TECHNICAL_SPECS.md (integration)

Level 4: Implementation Details
  ├── market-os/docs/BUILD_ORDER.md (dependencies)
  ├── market-os/docs/DATA_CONTRACTS.md (schemas)
  └── src/marketos/ (source code)
```

---

## Recommended Reading Order by Goal

### Goal: Evaluate for Investment (1-2 hours)
1. This file (DOCUMENTATION_INDEX.md) — 5 min
2. [`PORTFOLIO_OVERVIEW.md`](PORTFOLIO_OVERVIEW.md) — 15 min
3. [`market-os/docs/RESEARCH_SUMMARY.md`](market-os/docs/RESEARCH_SUMMARY.md) — 20 min
4. [`market-os/QUICKSTART.md`](market-os/QUICKSTART.md) (run demo) — 15 min
5. Call for questions — 30+ min

### Goal: Use for Research (4-8 hours)
1. [`market-os/README.md`](market-os/README.md) — 10 min
2. [`market-os/QUICKSTART.md`](market-os/QUICKSTART.md) + run demo — 30 min
3. [`market-os/docs/ARCHITECTURE.md`](market-os/docs/ARCHITECTURE.md) — 45 min
4. [`market-os/docs/RESEARCH_SUMMARY.md`](market-os/docs/RESEARCH_SUMMARY.md) — 30 min
5. [`market-os/docs/PRINCIPLES.md`](market-os/docs/PRINCIPLES.md) — 60 min
6. Code exploration (features, models, backtest) — 2-3 hours

### Goal: Integrate into Production (2-4 weeks)
1. Complete "Use for Research" path above
2. [`market-os/docs/TECHNICAL_SPECS.md`](market-os/docs/TECHNICAL_SPECS.md) — 2 hours
3. [`market-os/docs/BUILD_ORDER.md`](market-os/docs/BUILD_ORDER.md) — 1 hour
4. [`market-os/docs/DATA_CONTRACTS.md`](market-os/docs/DATA_CONTRACTS.md) — 1 hour
5. Implementation (custom data, models) — 1-2 weeks
6. Validation (backtest, live paper trading) — 1-2 weeks

---

## Contact & Support

- **General questions**: Email pulkit.talks@gmail.com
- **Community**: [TauricResearch Discord](https://discord.com/invite/hk9PGKShPK)
- **Issues**: GitHub issues (link TBD)
- **Consulting**: Available for enterprise integration

---

**Version**: 1.0  
**Last updated**: June 2026  
**Next review**: September 2026
