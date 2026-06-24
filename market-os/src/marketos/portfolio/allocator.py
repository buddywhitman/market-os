"""Portfolio agent — rank opportunities, allocate capital, respect every cap.

Inputs are per-opportunity rows carrying an alpha score and an expectancy report. We
allocate proportionally to *evidence-adjusted edge* (expectancy discounted by sample
size and confidence), then hand off to risk.sizing to enforce name/sector/gross caps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from marketos.risk.sizing import RiskLimits, enforce_sector_caps


def _evidence_discount(sample_size: pd.Series, k: int = 60) -> pd.Series:
    """Shrink edge for thin samples: n/(n+k) → 0 when n is tiny, →1 when n is large."""
    return sample_size / (sample_size + k)


def allocate(
    opportunities: pd.DataFrame,
    *,
    limits: RiskLimits | None = None,
    max_positions: int = 15,
    min_sample_size: float = 30,
) -> pd.DataFrame:
    """Allocate weights across opportunities.

    Expected columns: symbol, sector, expectancy, sample_size, confidence (0–1).
    Returns the frame with a `weight` column after applying sector and gross caps.
    Only positive-expectancy ideas with adequate evidence get capital.

    `min_sample_size` defaults to 30 (a raw trade count) but MUST be lowered for evidence
    sources that report uniqueness-weighted EFFECTIVE n rather than raw n — e.g. the analog
    engine's `analog_n_effective`, which is deliberately discounted for overlapping windows
    and typically sits in the 5-15 range even with 50 raw matches (see
    features/market_memory.py). Passing 30 against effective-n evidence silently filters
    every candidate to zero, every time — not a cautious default, a broken one.
    """
    limits = limits or RiskLimits()
    df = opportunities.copy()
    df = df[(df["expectancy"] > 0) & (df["sample_size"] >= min_sample_size)].copy()
    if df.empty:
        df["weight"] = []
        return df

    df["edge"] = (
        df["expectancy"].clip(lower=0)
        * _evidence_discount(df["sample_size"])
        * df.get("confidence", 1.0)
    )
    df = df.sort_values("edge", ascending=False).head(max_positions)

    total = df["edge"].sum()
    df["weight"] = (df["edge"] / total) * limits.max_gross_exposure if total > 0 else 0.0

    sectors = dict(zip(df["symbol"], df.get("sector", "UNKNOWN")))
    capped = enforce_sector_caps(dict(zip(df["symbol"], df["weight"])), sectors, limits)
    df["weight"] = df["symbol"].map(capped)
    return df.reset_index(drop=True)
