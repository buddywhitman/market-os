"""The non-negotiable rules, as runtime-checkable invariants.

This module is intentionally dependency-light so it can be imported anywhere
(fetchers, feature builders, backtests) to assert correctness at the boundary.

Philosophy: it should be *hard* to violate a principle by accident. Where we cannot
make violation impossible, we make it loud.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd


class PrincipleViolation(AssertionError):
    """Raised when a non-negotiable rule is broken. Never caught silently."""


# --------------------------------------------------------------------------------------
# Rule 3: No future information / no look-ahead.
# --------------------------------------------------------------------------------------
def assert_no_lookahead(
    df: pd.DataFrame,
    *,
    asof_col: str = "asof_ts",
    knowledge_col: str = "knowledge_ts",
) -> None:
    """Every row must be *knowable* at the time it is used.

    `asof_ts`     — the timestamp the row is attributed to (e.g. bar close).
    `knowledge_ts`— the wall-clock time the value actually became available
                    (e.g. when a filing was published, or a fundamental restated).

    A feature is leaky if knowledge_ts > asof_ts: it encodes information from the
    future relative to when we claim to act. This is the single most common way
    backtests lie.
    """
    if asof_col not in df.columns or knowledge_col not in df.columns:
        raise PrincipleViolation(
            f"point-in-time columns missing: need '{asof_col}' and '{knowledge_col}'. "
            "Every feature row must carry when it was knowable."
        )
    bad = df[df[knowledge_col] > df[asof_col]]
    if len(bad):
        raise PrincipleViolation(
            f"look-ahead detected in {len(bad)} rows: knowledge_ts is after asof_ts. "
            "You are using information from the future."
        )


def assert_monotonic_time(df: pd.DataFrame, ts_col: str = "asof_ts") -> None:
    """Time must not go backwards within a series; duplicates/reordering hide bugs."""
    if not df[ts_col].is_monotonic_increasing:
        raise PrincipleViolation(f"'{ts_col}' is not monotonically increasing.")


# --------------------------------------------------------------------------------------
# Rule 3b: No survivorship bias.
# --------------------------------------------------------------------------------------
def assert_includes_delisted(universe: pd.DataFrame, *, status_col: str = "status") -> None:
    """A backtest universe that contains only currently-listed names is a trap.

    The graveyard (delisted, merged, bankrupt) must be present for any historical
    study, or your win rate is fiction.
    """
    if status_col not in universe.columns:
        raise PrincipleViolation(
            "universe has no lifecycle status column; cannot prove it isn't survivorship-biased."
        )
    if (universe[status_col] == "delisted").sum() == 0:
        raise PrincipleViolation(
            "universe contains zero delisted names — almost certainly survivorship-biased."
        )


# --------------------------------------------------------------------------------------
# Rule 2: Everything versioned, timestamped, reproducible.
# --------------------------------------------------------------------------------------
def content_hash(payload: bytes) -> str:
    """Deterministic content address for immutable raw storage."""
    return hashlib.sha256(payload).hexdigest()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Provenance:
    """Stamped onto every artifact. If you can't say where it came from, you can't use it."""

    source: str          # e.g. "yfinance", "NSE bhavcopy", "SEC EDGAR"
    fetched_at: datetime  # wall-clock acquisition time (knowledge time)
    sha256: str           # content hash of the raw bytes
    code_version: str     # git sha / package version that produced this

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat(),
            "sha256": self.sha256,
            "code_version": self.code_version,
        }


# --------------------------------------------------------------------------------------
# Rule 4 & 5: distributions over point forecasts; expectancy is the arbiter.
# --------------------------------------------------------------------------------------
def assert_positive_expectancy(expectancy: float, sample_size: int, min_n: int = 30) -> None:
    """We only deploy edges with positive expected value *and* enough evidence."""
    if sample_size < min_n:
        raise PrincipleViolation(
            f"sample size {sample_size} < {min_n}: insufficient evidence to claim an edge."
        )
    if expectancy <= 0:
        raise PrincipleViolation(
            f"expectancy {expectancy:.4f} <= 0: this is not an edge, it is a donation."
        )


# --------------------------------------------------------------------------------------
# Rule 3c: a positive backtest must not survive having its labels shuffled.
# --------------------------------------------------------------------------------------
def assert_survives_label_shuffle(
    real_sharpe: float, shuffled_sharpe: float, *, max_retained_fraction: float = 0.35
) -> None:
    """A label-shuffle null test breaks any real (feature -> target) relationship while
    leaving the backtest mechanics (basket composition, rebalancing, costs) untouched. If
    performance barely drops when the target is randomized, the measured 'edge' is coming
    from those mechanics — e.g. basket beta concentrated by a top-N selection rule — not
    from the model. This caught a real false positive on 2026-06-22: a 19-symbol AI/semis
    basket retained ~73% of its Sharpe under shuffled labels, because picking almost any
    5-of-19 names from a basket that mostly went up still looks like skill.

    `max_retained_fraction` is the most the shuffled-label Sharpe may retain of the real
    Sharpe before we call the result an artifact rather than evidence of skill.
    """
    if real_sharpe <= 0:
        raise PrincipleViolation(f"real_sharpe {real_sharpe:.4f} <= 0: nothing to validate.")
    retained = shuffled_sharpe / real_sharpe
    if retained > max_retained_fraction:
        raise PrincipleViolation(
            f"shuffled-label Sharpe ({shuffled_sharpe:.3f}) retains {retained:.0%} of the "
            f"real Sharpe ({real_sharpe:.3f}) — exceeds the {max_retained_fraction:.0%} "
            "threshold. The apparent edge is likely a backtest-construction artifact "
            "(basket beta, selection-rule concentration), not validated predictive skill."
        )


THE_RULES = [
    "No black boxes — every prediction carries SHAP attributions.",
    "Everything versioned, timestamped, reproducible — raw data is immutable & hashed.",
    "No data leakage, no survivorship bias, no future information.",
    "No magical claims — report distributions and expectancy, not point forecasts.",
    "If a feature doesn't improve out-of-sample expectancy, remove it.",
    "Simplicity over complexity. Evidence over narrative.",
    "Optimize for long-term information extraction and adaptation.",
]
