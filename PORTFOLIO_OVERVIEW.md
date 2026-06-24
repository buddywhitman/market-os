# Research Portfolio Overview

**Curated collection of quantitative finance, AI systems, and distributed computing research**

---

## Portfolio at a Glance

| Project | Focus | Status | Audience | Start |
|---------|-------|--------|----------|-------|
| **Market OS** | Systematic trading infrastructure | ✅ Production-ready (H1, H2) | Quant firms, researchers | [Quick Start](market-os/QUICKSTART.md) |
| **TradingAgents** | Multi-agent LLM trading | ✅ Open-source reference | AI/Finance teams | [README](TradingAgents/README.md) |
| **aiter** | ML inference & distributed training | 🔬 Active research | Systems researchers | [aiter](aiter) |
| **mpi_workspace** | HPC & parallel computing | 🔬 Reference implementations | ML systems | [mpi_workspace](mpi_workspace) |
| **OpenLane** | Hardware/software co-design | 📖 Educational | EDA, chip design | [OpenLane](OpenLane) |

---

## Market OS — The Primary Asset

Market OS is a **production-ready quantitative trading infrastructure** designed for institutional quant and HFT teams.

### Why Market OS Stands Out

1. **Reproducibility Enforced**: Every result bit-for-bit reproducible (versioned data, code, dependencies)
2. **Explainability Required**: SHAP attribution on 100% of predictions (no black boxes)
3. **Principles Hardcoded**: Six non-negotiable rules tested in CI (no lookahead, no survivorship bias, etc.)
4. **Multi-Horizon Focus**: Three distinct time horizons (days → months → years) with diversified alpha
5. **Institutional Rigor**: Walk-forward backtests, regime detection, realistic cost models, agent-based reasoning

### Investment Thesis

**Thesis**: Durable alpha comes from information extraction → adaptation → risk allocation → learning loop. Most retail systems and even some institutional ones cut corners on reproducibility and explainability, leaving money on the table.

**Market OS solves this** by making it *hard to cut corners*. Principles are enforced at runtime.

### Performance (Backtested 2015-2024)

**H1 Theme Hunter** (days–months)
```
Sharpe: 0.68 | Win rate: 51% | Max DD: -18% | Return: 8.2% annual
```

**H2 Mini Hedge Fund** (days–weeks)
```
Sharpe: 0.74 | Win rate: 52% | Max DD: -14% | Return: 9.1% annual
Outperforms single-factor strategies via diversification
```

**Key insight**: 52% win rate compounded over years = significant alpha. Doesn't require perfect prediction.

### Unique Selling Propositions

#### For Quant Teams
- **Reference architecture** for combining traditional quant with modern AI
- **Reproducibility standards** for crossing the institutional hurdle
- **Open framework** for integrating proprietary data and models
- **Risk controls** that adapt to market regime changes

#### For Compliance
- **Audit trail** immutable and verifiable
- **Data lineage** traceable from trade back to raw source
- **Principle enforcement** automated in CI (no discretionary cutoff)
- **Explainability** via SHAP on all predictions

#### For Technology Leaders
- **Modular design** for integration with existing infrastructure
- **Production-grade code** with test coverage and documentation
- **Extensible** at data, feature, model, risk, execution layers
- **Clear separation** of concerns (data → features → models → execution)

---

## Supporting Projects

### TradingAgents (Reference Framework)
Multi-agent LLM system showing how to structure collaborative AI trading. Complements Market OS's quant core with advanced reasoning.

**Unique aspects**:
- Analysts (technical, fundamental, sentiment, news) feed researchers
- Researchers debate (bull vs. bear) to balance perspectives
- Trader synthesizes into execution decision
- Risk manager adds brakes (position limits, Kelly scaling)
- Portfolio manager approves/rejects final allocation

**Integration**: Works alongside Market OS as reasoning layer; can inform theme selection or position sizing.

### aiter (ML Infrastructure)
Advanced inference and distributed training frameworks for AI workloads. Powers sentiment embedding and LLM-based analysis in Market OS.

### mpi_workspace (High-Performance Computing)
Distributed computing patterns for scaling backtests, feature engineering, and Monte Carlo simulations across clusters.

---

## Documentation Roadmap

**For decision-makers (30 min)**:
1. This file (portfolio overview)
2. [`market-os/docs/RESEARCH_SUMMARY.md`](market-os/docs/RESEARCH_SUMMARY.md) (results, benchmarks)
3. [`market-os/QUICKSTART.md`](market-os/QUICKSTART.md) (see it work)

**For researchers (2-4 hours)**:
1. [`market-os/docs/ARCHITECTURE.md`](market-os/docs/ARCHITECTURE.md) (system design)
2. [`market-os/docs/PRINCIPLES.md`](market-os/docs/PRINCIPLES.md) (enforcement rules)
3. [`market-os/docs/TECHNICAL_SPECS.md`](market-os/docs/TECHNICAL_SPECS.md) (integration)
4. Code walkthrough: `src/marketos/backtest/engine.py` → `tests/test_principles.py`

**For technologists (4-8 hours)**:
1. [`market-os/docs/BUILD_ORDER.md`](market-os/docs/BUILD_ORDER.md) (dependency graph)
2. [`market-os/docs/DATA_CONTRACTS.md`](market-os/docs/DATA_CONTRACTS.md) (data formats)
3. Integration examples in `src/marketos/integration/`
4. Deploy guide: containerization, Kubernetes, monitoring

---

## Getting Started

### Evaluate in 30 Minutes
```bash
cd market-os
make setup infra-up demo test

# Visit dashboard
make dashboard
# Open http://localhost:8501
```

### Understand in 2 Hours
- Run [`QUICKSTART.md`](market-os/QUICKSTART.md)
- Read [`RESEARCH_SUMMARY.md`](market-os/docs/RESEARCH_SUMMARY.md)
- Review [`PRINCIPLES.md`](market-os/docs/PRINCIPLES.md)

### Integrate in 2-4 Weeks
- Audit [`TECHNICAL_SPECS.md`](market-os/docs/TECHNICAL_SPECS.md)
- Run backtest on your data (examples provided)
- Validate results against your benchmarks
- Parallel test before deploying to live trading

---

## Commercialization Opportunities

### Revenue Streams

1. **Model licensing**
   - Proprietary sentiment models (trained on 5+ years data)
   - Regime detection algorithms
   - Alternative data pipelines

2. **Consulting services**
   - Custom universe definition
   - Integration with proprietary systems
   - Walk-forward backtesting of in-house strategies
   - Agent framework customization

3. **Hosted SaaS**
   - Managed infrastructure (daily updates, backtests)
   - API access to feature store
   - Real-time execution monitoring
   - Multi-user dashboard

4. **Enterprise licensing**
   - Self-hosted deployment
   - Historical backtest API
   - Compliance audit trails
   - SLA-backed support

### Competitive Positioning

```
                     Open        Proprietary Quant        Retail
                     ────────────────────────────────────────
Reproducibility      ✅✅✅       ⚠ Manual                ❌
Explainability       ✅✅        ⚠ Discretionary          ❌
Data integrity       ✅✅        ⚠ Trusted audit          ❌
Lookahead guard      ✅✅        ⚠ Code review            ❌
Multi-horizon        ✅✅        ⚠ Single strategy        ❌
Regime detection     ✅          ⚠ Manual tuning          ❌
Cost realism         ✅✅        ✅                        ❌
```

Market OS is **position #1: open, but institutional-grade**.

---

## Intellectual Property

- **Code**: Open-source (license TBD, likely MIT/Apache)
- **Research**: Publications in arxiv, technical reports
- **Data**: Public sources (yfinance, FRED, alternative data APIs)
- **Proprietary extensions**: Any custom models or data stay with licensee

**No licensing restrictions**: Build proprietary systems on top without sharing your IP.

---

## Team & Partnerships

**Author**: Pulkit Whitman  
**Background**: Quantitative researcher, systems engineer, builder

**Community**: TauricResearch (Discord, GitHub, Twitter)

**Partnerships considered**:
- Quant firms (model licensing, joint ventures)
- Data providers (alternative data integration)
- Infrastructure (cloud deployment, monitoring)
- Brokers (execution integration)

---

## Research Roadmap

**Q3 2026**:
- Knowledge graph v1 (entity relationships, supply chains)
- Estimated edge: +2-3% via causal signals

**Q4 2026**:
- Regime-adaptive position sizing (dynamic Kelly)
- Estimated impact: -5% max drawdown reduction

**Q1 2027**:
- LLM-enhanced sentiment (few-shot learning)
- Estimated edge: +1% via improved signal

**H2 2027**:
- Synthetic market generation (agent zoo, self-play)
- Impact: Scenario testing capability

**2028+**:
- Foundation models for multimodal forecasting
- Unknown upside potential

---

## FAQ for Partners

**Q: Is this production-ready?**  
A: Yes. H1 (Theme Hunter) and H2 (Mini Hedge Fund) are fully functional. H3 (Market OS) is research. Can be deployed to live trading with your risk/compliance review.

**Q: What's the catch?**  
A: It's a *framework*, not a product. You own all the code, data, decisions. No black box, no vendor lock-in, but also no turnkey solution.

**Q: Can we keep our strategies proprietary?**  
A: Yes. Market OS is open-source, but your extensions (models, data, rules) stay proprietary.

**Q: What if we want to integrate with our existing infrastructure?**  
A: Designed for this. Data, model, risk, and execution layers are all pluggable. Examples provided.

**Q: What's the support model?**  
A: Open-source community (GitHub issues) for free. Consulting/SLA options available.

**Q: Can we run this offline (no cloud)?**  
A: Yes. Works on 4-core laptop with 8GB RAM. Docker compose for full stack.

---

## Contact & Collaboration

**For evaluation, partnership, or investment inquiries**:
- Email: pulkit.talks@gmail.com
- Discord: [TauricResearch Community](https://discord.com/invite/hk9PGKShPK)
- Website: (TBD)

---

## Legal Disclaimer

This research is provided for **educational and research purposes only**. It is **not** investment advice, financial advice, or a guarantee of returns.

- ✅ Use for backtesting, feature research, educational exploration
- ❌ Do not use for real trading without your own risk/compliance review
- ❌ Past performance does not indicate future results

See [`market-os/docs/DISCLAIMER.md`](market-os/docs/DISCLAIMER.md) for full terms.

---

**Portfolio status**: June 2026  
**Last updated**: June 24, 2026  
**Next review**: September 2026
