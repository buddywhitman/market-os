# Research & Development Portfolio

A curated collection of quantitative trading systems, market microstructure research, and AI-driven financial intelligence frameworks.

## Core Projects

### 🎯 **Market OS** — Quantitative Trading Infrastructure
*Primary focus: Explainable alpha generation with risk-aware capital allocation*

A continuously evolving market operating system for systematic trading. Enforces strict principles around reproducibility, data integrity, and explainability—designed for institutional quant and HFT workflows.

**Key characteristics:**
- **Three investment horizons**: Theme Hunter (days–months), Mini Hedge Fund (days–weeks), Market OS (years+)
- **Six core principles**: No black boxes, full versioning, zero data leakage, distributions not point forecasts, empirical evidence, ruthless simplicity
- **Multi-agent architecture**: Research, Macro, Technical, Sentiment, Quant, Risk, Portfolio Management
- **Feature store**: Point-in-time, SHAP-attributed, walk-forward validated
- **Backtesting**: No-lookahead, cost-inclusive, Monte Carlo expectancy

**Status**: H1 (Theme Hunter) runnable; H2 (Mini Hedge Fund) scaffolded; H3 (Market OS knowledge graph) research stubs

**Quick start:**
```bash
cd market-os
make setup
make demo          # fetch → features → backtest → expectancy
make dashboard     # Streamlit PM cockpit
make test          # Verify principles enforcement
```

---

### 📊 TradingAgents — Multi-Agent LLM Trading Framework
*Reference architecture for collaborative AI trading systems*

Sophisticated multi-agent framework that mirrors dynamics of real trading firms. LLM-powered agents (fundamental analysts, sentiment experts, technicians, risk managers) collaborate to evaluate markets and inform decisions.

**Specialized roles:**
- Analyst Team: Fundamentals, Sentiment, News, Technical
- Researcher Team: Bull/Bear debate and synthesis
- Execution: Trader, Risk Management, Portfolio Manager
- Multi-LLM support: OpenAI, Anthropic, Azure, Google, Bedrock, DeepSeek, Qwen

---

### 🔬 Additional Research
- **aiter**: ML inference and distributed training infrastructure
- **mpi_workspace**: High-performance computing frameworks
- **OpenLane**: Hardware-software co-design research

---

## Documentation Structure

```
.
├── README.md                          (this file — start here)
├── PORTFOLIO_SUMMARY.md               (if this exists, full project overview)
│
├── market-os/                         ← MAIN PROJECT
│   ├── README.md                      (project overview)
│   ├── QUICKSTART.md                  (get running in 15 min)
│   ├── Makefile                       (setup, demo, test, dashboard)
│   ├── docs/
│   │   ├── ARCHITECTURE.md            (system design, 7 layers, diagram)
│   │   ├── BUILD_ORDER.md             (dependency graph, execution order)
│   │   ├── DATA_CONTRACTS.md          (data formats, schemas, contracts)
│   │   ├── RESEARCH_LOG.md            (dated findings, experiments)
│   │   ├── TECHNICAL_SPECS.md         (for quant/HFT integration)
│   │   ├── PRINCIPLES.md              (6 enforcement rules + test examples)
│   │   └── RESEARCH_SUMMARY.md        (executive summary, benchmarks, results)
│   ├── src/marketos/                  (source code)
│   │   ├── data/                      (fetchers, parsers, catalog)
│   │   ├── features/                  (feature store, SHAP attribution)
│   │   ├── models/                    (XGBoost, walk-forward validation)
│   │   ├── backtest/                  (no-lookahead engine, expectancy)
│   │   ├── risk/                      (regime, stops, Kelly sizing)
│   │   ├── agents/                    (research, macro, sentiment, risk, PM)
│   │   └── dashboard/                 (Streamlit UI)
│   ├── tests/                         (principles, no-lookahead, expectancy)
│   └── config/                        (universes, themes, costs, limits)
│
├── TradingAgents/                     ← REFERENCE FRAMEWORK
│   ├── README.md                      (multi-agent LLM trading system)
│   ├── tradingagents/                 (source code)
│   │   ├── agents/                    (analysts, researchers, managers)
│   │   ├── dataflows/                 (data integration)
│   │   ├── llm_clients/               (multi-provider support)
│   │   └── graph/                     (workflow orchestration)
│   ├── cli/                           (command-line interface)
│   └── tests/                         (integration tests)
│
├── research-artifacts/                ← RESEARCH & BENCHMARKS
│   ├── README.md                      (overview of papers, case studies)
│   ├── papers/                        (whitepapers, technical specs)
│   ├── benchmarks/                    (backtest results, comparisons)
│   ├── case_studies/                  (real-world examples, learnings)
│   └── technical_specs/               (integration guides, API docs)
│
└── [other projects]
    ├── aiter/                         (ML inference infrastructure)
    ├── mpi_workspace/                 (HPC frameworks)
    ├── OpenLane/                      (hardware/chip design)
    └── nanoInfer, pocl, etc.          (specialized research)
```

---

## For Quant & HFT Firms

**Market OS** is positioned as:

1. **Operational template**: Reference architecture for combining traditional quant with modern AI/LLM reasoning
2. **Reproducibility standard**: Demonstrates end-to-end data integrity, feature engineering, and backtest verification
3. **Risk framework**: Explicit regime detection, Kelly scaling, ATR stops, portfolio constraints
4. **Integration point**: Modular, extensible design for bridging proprietary systems

**Key differentiators:**
- Explainability via SHAP/attribution (not black-box NN)
- Walk-forward validation with proper test set isolation
- Multi-horizon diversification (not just alpha chasing)
- Decoupled from any single LLM provider

---

## Getting Started

### Minimal setup (no cloud, no paid APIs)
```bash
# Market OS end-to-end demo
cd market-os
make setup infra-up demo test

# Visit dashboard
make dashboard  # http://localhost:8501
```

### Recommended for evaluation
1. Review [`market-os/docs/ARCHITECTURE.md`](market-os/docs/ARCHITECTURE.md) — understand the system design
2. Run `make demo` — verify feature store and backtest pipeline
3. Check [`market-os/tests/test_principles.py`](market-os/tests/test_principles.py) — see enforcement in action
4. Explore [`docs/RESEARCH_LOG.md`](market-os/docs/RESEARCH_LOG.md) — recent findings and improvements

---

## Research Artifacts

Papers, benchmarks, and case studies available in `/research-artifacts/`:
- Technical whitepapers
- Backtest benchmarks (Sharpe, drawdown, calmar ratios)
- Real-world case studies
- Data pipeline documentation

---

## Citation & Collaboration

If this research informs your work, please cite:
```bibtex
@software{marketos2024,
  title={Market OS: A Quantitative Trading Infrastructure},
  author={Whitman, Pulkit},
  year={2024},
  url={https://github.com/...}
}
```

For integration inquiries, technical questions, or collaboration opportunities:
- Email: pulkit.talks@gmail.com
- Discord: [TauricResearch Community](https://discord.com/invite/hk9PGKShPK)

---

## License & Disclaimer

This code is provided for **research and educational purposes**. It is **not** investment advice, financial advice, or a guarantee of returns.

- ✅ Use for backtesting, feature research, educational exploration
- ❌ Do not use for real trading without your own risk/compliance review
- ❌ Past performance is not indicative of future results

See [`market-os/docs/DISCLAIMER.md`](market-os/docs/DISCLAIMER.md) for full terms.

---

**Last updated**: June 2026
