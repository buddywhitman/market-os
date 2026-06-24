"""Phase 4b — conviction-gate the aggressive sleeve's mechanical dip re-entry signal.

Phase 4a's finding (real backtest, see project notes): a PURE PRICE dip-reentry signal
(50DMA reclaim after a deep, oversold drawdown) helps in genuine recoveries (+17pts in
2023) but HURTS in grinding bear-market rallies (-7pts in 2022) — because price alone
cannot distinguish a real bottom from a dead-cat bounce. That is exactly the discriminator
market-os's validated signals exist to provide:

  * The market-wide HMM regime (regimes/hmm.py) found, via regime-conditional validation,
    that the stress_chop regime is specifically where the latent direction signal shows
    strong trend-continuation/bounce behavior (IC=0.75 conditional vs ~0.02 pooled — see
    project notes, Phase 4 entry). A capitulation-reversal setup occurring WHILE the market
    is in stress_chop is on much firmer ground than the same setup in a calm/neutral tape.
  * The analog engine's per-symbol win-rate (`analog_win_rate_20d`), where available, is
    direct historical evidence for THIS symbol's specific behavior after similar setups.

Deliberately NOT backtested as a unified gated strategy — reconstructing the regime AND
analog evidence causally for every historical day would be heavy and leak-prone (the HMM
fit point was explicitly flagged as look-ahead for backtesting in
models/aggressive_sleeve_backtest.py's docstring). This module is LIVE-ONLY: it composes
two already-validated-separately signals into a conviction score for TODAY's decision, and
its own track record must accumulate from here forward, not be assumed from the pieces.
"""
from __future__ import annotations

from dataclasses import dataclass

# Below this conviction score, a fired dip-reentry signal is NOT acted on — same spirit as
# the backtest's partial (0.6, not 1.0) re-entry weight: re-entering into a still-falling
# market is the riskiest bet this system makes, so the bar is deliberately not "any signal."
GATE_THRESHOLD = 0.55

# Stress-regime bonus mirrors the regime-conditional finding (IC 0.75 in stress vs ~0.02
# pooled) — a large, specific, already-validated number, not an arbitrary tuning constant.
STRESS_REGIME_BONUS = 0.30
ANALOG_WIN_RATE_WEIGHT = 0.6  # how much a symbol's own analog win-rate can move the score

# A symbol's OWN analog history below this is a hard veto, regardless of regime. The
# market-wide regime bonus reflects "conditions favor SOME quality dip bouncing" — it
# does not mean THIS symbol will, and direct symbol-specific evidence this bad should not
# be overridable by a market-wide prior (caught during local verification: a COIN-like
# 10% win-rate symbol cleared the score threshold purely off the stress-regime bonus
# before this guard existed).
ANALOG_HARD_VETO_WIN_RATE = 0.30


@dataclass(frozen=True)
class ConvictionResult:
    gate_pass: bool
    conviction: float
    reasons: list[str]


def conviction_gate(
    symbol: str, *, spy_regime: int | None, analog: dict | None,
) -> ConvictionResult:
    """Score a fired dip-reentry signal's conviction from 0 (don't act) to 1 (strong).

    `spy_regime` — current market-wide regime label (0=calm_trend, 1=neutral,
        2=stress_chop) from `store.get_latest_regime("SPY")`. None if regime_update hasn't
        run — treated as neutral (no bonus, no penalty), not as a silent failure.
    `analog` — this symbol's latest `analog` family dict, or None/empty if the symbol isn't
        in the validated quant UNIVERSE (most aggressive-sleeve-only names, e.g. SOXL/
        BTC-USD/MU, have no analog history — the gate must degrade gracefully, not
        require evidence that structurally doesn't exist for that symbol).
    """
    score = 0.5  # neutral prior: a confirmed bounce-pattern is mildly informative on its own
    reasons = []

    if spy_regime == 2:
        score += STRESS_REGIME_BONUS
        reasons.append("market-wide regime is stress_chop — exactly where the validated "
                       "direction signal shows trend-continuation/bounce behavior")
    elif spy_regime == 0:
        score -= 0.15
        reasons.append("market-wide regime is calm_trend — a dip here is less likely to be "
                       "the kind of capitulation the validated signal was found in")
    else:
        reasons.append("market-wide regime is neutral or unknown — no regime adjustment")

    win_rate = (analog or {}).get("analog_win_rate_20d")
    hard_veto = False
    if win_rate is not None:
        score += (float(win_rate) - 0.5) * ANALOG_WIN_RATE_WEIGHT
        reasons.append(f"this symbol's own analog win-rate is {win_rate:.0%} "
                       f"(n_effective={(analog or {}).get('analog_n_effective', '?')})")
        if float(win_rate) < ANALOG_HARD_VETO_WIN_RATE:
            hard_veto = True
            reasons.append(f"VETO: this symbol's own win-rate ({win_rate:.0%}) is below "
                           f"{ANALOG_HARD_VETO_WIN_RATE:.0%} — direct symbol-specific "
                           f"evidence this poor overrides any regime-based bonus")
    else:
        reasons.append(f"no analog history for {symbol} (not in the validated quant "
                       f"UNIVERSE) — relying on regime context alone")

    score = max(0.0, min(1.0, score))
    return ConvictionResult(gate_pass=(not hard_veto) and score >= GATE_THRESHOLD,
                            conviction=round(score, 3), reasons=reasons)
