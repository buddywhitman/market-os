"""H1 — Theme Hunter.

Goal: find explosive themes and ride trend persistence (days→months). We do NOT try to
call bottoms or tops. We score themes on multiple orthogonal signals, then rank the
leaders *inside* the strongest themes.

Theme Score blends:
    narrative intensity (news/search frequency)
  × price momentum
  × volume / volatility expansion
  × relative strength vs the broad market
  × breadth (how many constituents participate)

Every input is normalized cross-sectionally so no single axis dominates by units.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    return (s - mu) / sd if sd > 0 else s * 0.0


def _minmax_100(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) * 100 if hi > lo else s * 0 + 50


@dataclass
class ThemeWeights:
    narrative: float = 0.20
    momentum: float = 0.30
    expansion: float = 0.20
    rel_strength: float = 0.20
    breadth: float = 0.10


@dataclass
class ThemeHunter:
    weights: ThemeWeights = field(default_factory=ThemeWeights)

    def score_themes(self, theme_panel: pd.DataFrame) -> pd.DataFrame:
        """Score themes from a panel with one row per theme.

        Expected columns (raw, any scale — we normalize):
          narrative, momentum, expansion, rel_strength, breadth
        Returns the panel with z-scored inputs and a 0–100 `theme_score`, sorted desc.
        """
        w = self.weights
        df = theme_panel.copy()
        cols = ["narrative", "momentum", "expansion", "rel_strength", "breadth"]
        for c in cols:
            df[f"z_{c}"] = _zscore(df[c])
        blended = (
            w.narrative * df["z_narrative"]
            + w.momentum * df["z_momentum"]
            + w.expansion * df["z_expansion"]
            + w.rel_strength * df["z_rel_strength"]
            + w.breadth * df["z_breadth"]
        )
        df["theme_score"] = _minmax_100(blended).round(1)
        return df.sort_values("theme_score", ascending=False)

    def rank_leaders(self, constituents: pd.DataFrame, theme_score: float) -> pd.DataFrame:
        """Rank names inside a theme.

        leader_score = theme_score
                     × momentum × volatility_expansion × liquidity
                     × relative_strength × fundamental_acceleration

        Expected columns: symbol, momentum, vol_expansion, liquidity,
                          rel_strength, fundamental_accel (all raw; normalized to ~[0,2]).
        """
        df = constituents.copy()
        factors = ["momentum", "vol_expansion", "liquidity", "rel_strength", "fundamental_accel"]
        norm = pd.DataFrame(index=df.index)
        for f in factors:
            # map to a positive multiplier centered near 1 via rank-based scaling
            r = df[f].rank(pct=True)
            norm[f] = 0.5 + r  # in [0.5, 1.5]
        leader = (theme_score / 100.0)
        for f in factors:
            leader = leader * norm[f]
        df["leader_score"] = (leader * 100).round(1)
        return df.sort_values("leader_score", ascending=False)
