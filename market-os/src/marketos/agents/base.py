"""Agent base contract.

An 'agent' here is NOT an LLM chat loop. It is a deterministic, testable transform that
ingests data and emits **structured features or metrics** — never narratives. This keeps
the H2 mini hedge fund explainable and reproducible. LLMs, where used (sentiment/RAG),
sit *behind* an agent and their output is reduced to numbers before it leaves.

Each agent: declares its inputs, emits a typed frame, and stamps point-in-time columns.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import pandas as pd

from marketos.principles import assert_no_lookahead


@dataclass
class AgentOutput:
    name: str
    features: pd.DataFrame   # must carry asof_ts + knowledge_ts

    def validate(self) -> "AgentOutput":
        assert_no_lookahead(self.features)
        return self


class Agent(abc.ABC):
    """Base class for Research / Macro / Technical / Sentiment / Quant / Risk / PM agents."""

    name: str = "agent"

    @abc.abstractmethod
    def run(self, **inputs) -> AgentOutput:
        """Produce structured features. Implementations must NOT emit free text."""
        raise NotImplementedError


# Roster (to be implemented per docs/BUILD_ORDER.md). Listed here as the contract surface.
AGENT_ROSTER = [
    "research",    # filings, results, presentations, concalls, ownership → growth/quality/valuation metrics
    "macro",       # oil, DXY, USDINR, rates, VIX, PMI, policy → macro & regime features
    "technical",   # EMA/ADX/RSI/ATR/VWAP/RS/momentum → technical features
    "sentiment",   # news/Reddit/X/transcripts via embeddings+RAG → sentiment features
    "quant",       # historical analogs → win rate/expectancy/Sharpe/Monte Carlo
    "risk",        # sizing/stops/Kelly/exposure caps/drawdown limits
    "portfolio",   # rank opportunities, allocate capital
]
