"""Walk-forward validation — the only validation we trust.

Random k-fold cross-validation leaks the future into the past on time series. We instead
train on a rolling/expanding window and test strictly on the *subsequent* out-of-sample
window, with a purge+embargo gap between them so that label horizons cannot bleed across
the boundary (cf. Lopez de Prado).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_span: tuple[pd.Timestamp, pd.Timestamp]
    test_span: tuple[pd.Timestamp, pd.Timestamp]


def walk_forward_splits(
    timestamps: pd.Series,
    *,
    train_periods: int,
    test_periods: int,
    label_horizon: int,
    expanding: bool = False,
    embargo: int = 0,
) -> Iterator[WalkForwardSplit]:
    """Yield purged, embargoed walk-forward splits.

    `label_horizon` is how many periods into the future the target looks; we purge that
    many periods between train and test so a training label cannot overlap test inputs.
    `embargo` adds extra spacing after the test window before the next train window.
    """
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    n = len(ts)
    purge = label_horizon
    start = 0
    train_end = train_periods
    while train_end + purge + test_periods <= n:
        train_lo = 0 if expanding else start
        train_hi = train_end                       # exclusive
        test_lo = train_end + purge                 # purge the label horizon
        test_hi = min(test_lo + test_periods, n)

        train_idx = np.arange(train_lo, train_hi)
        test_idx = np.arange(test_lo, test_hi)
        if test_idx.size == 0:
            break

        yield WalkForwardSplit(
            train_idx=train_idx,
            test_idx=test_idx,
            train_span=(ts.iloc[train_lo], ts.iloc[train_hi - 1]),
            test_span=(ts.iloc[test_lo], ts.iloc[test_hi - 1]),
        )

        step = test_periods + embargo
        start += step
        train_end += step


def n_split_walk_forward(
    n_obs: int,
    *,
    n_splits: int,
    label_horizon: int,
    embargo_frac: float = 0.03,
) -> list[WalkForwardSplit]:
    """Convenience wrapper: "give me exactly (up to) n_splits expanding-window folds"
    instead of hand-picking train_periods/test_periods. Calibrates window sizes to use
    roughly all of `n_obs`, then delegates 100% of the actual purge+embargo arithmetic to
    `walk_forward_splits` above — this is the ONLY purge/embargo implementation in the
    codebase; nothing should hand-roll its own (a second one already caused two divergent
    fold-boundary conventions to exist briefly in this project — see git history / CHANGELOG).

    Takes a plain integer `n_obs` rather than real timestamps where the caller doesn't have
    a meaningful calendar (e.g. a synthetic/positional index) — synthesizes a daily
    DatetimeIndex internally since `walk_forward_splits` needs *some* timestamp series for
    its span metadata, but only the integer positions are used by callers in that case.
    """
    embargo = max(1, int(n_obs * embargo_frac))
    train_periods = (n_obs - label_horizon - (n_splits - 1) * embargo) // (n_splits + 1)
    test_periods = train_periods
    if train_periods < 10:
        raise ValueError(f"n_obs={n_obs} too small for {n_splits} splits "
                         f"(label_horizon={label_horizon}, embargo_frac={embargo_frac})")
    dummy_dates = pd.bdate_range("2000-01-01", periods=n_obs)
    folds = list(walk_forward_splits(
        pd.Series(dummy_dates), train_periods=train_periods, test_periods=test_periods,
        label_horizon=label_horizon, expanding=True, embargo=embargo,
    ))
    return folds[-n_splits:] if len(folds) > n_splits else folds
