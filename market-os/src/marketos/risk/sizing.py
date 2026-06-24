"""Position sizing & risk allocation.

Survival first. We never let a single idea, a single theme, or a single regime shift
take the book below its drawdown budget. Sizing is volatility-aware (ATR), edge-aware
(fractional Kelly), and capped at every level (name / sector / gross).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    risk_per_trade: float = 0.0075    # fraction of equity risked to the stop (0.75%)
    max_name_weight: float = 0.10
    max_sector_weight: float = 0.30
    max_gross_exposure: float = 1.50
    kelly_fraction: float = 0.25      # size at quarter-Kelly; full-Kelly is for the brave
    atr_stop_mult: float = 2.5
    max_portfolio_drawdown: float = 0.20  # de-risk trigger

    @classmethod
    def from_dict(cls, d: dict | None) -> "RiskLimits":
        """Build from a (partial) config dict, ignoring unknown keys and keeping
        dataclass defaults for anything absent. Lets config.yaml be the single source
        of truth without forcing every sleeve to specify every field."""
        if not d:
            return cls()
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: float(v) for k, v in d.items() if k in known and v is not None})


def atr_stop_distance(atr: float, mult: float) -> float:
    """Stop distance in price units = mult × ATR. Wider stops in noisier names."""
    return max(atr, 1e-9) * mult


def position_size(
    equity: float,
    entry_price: float,
    atr: float,
    *,
    limits: RiskLimits,
    edge_kelly: float | None = None,
) -> dict:
    """Size a position so the loss-to-stop equals `risk_per_trade` of equity,
    then scale by fractional Kelly if an edge estimate is supplied, then clamp to the
    per-name weight cap.

    Returns shares, notional, weight, stop price, and the risk actually taken.
    """
    stop_dist = atr_stop_distance(atr, limits.atr_stop_mult)
    dollar_risk = equity * limits.risk_per_trade

    if edge_kelly is not None:
        scale = max(0.0, min(1.0, edge_kelly)) * limits.kelly_fraction
        dollar_risk *= scale if scale > 0 else 0.0

    raw_shares = dollar_risk / stop_dist if stop_dist > 0 else 0.0
    notional = raw_shares * entry_price
    weight = notional / equity if equity > 0 else 0.0

    # Clamp to the per-name cap.
    if weight > limits.max_name_weight:
        weight = limits.max_name_weight
        notional = weight * equity
        raw_shares = notional / entry_price if entry_price > 0 else 0.0

    return {
        "shares": float(raw_shares),
        "notional": float(notional),
        "weight": float(weight),
        "stop_price": float(entry_price - stop_dist),
        "risk_amount": float(min(dollar_risk, weight * equity / max(stop_dist, 1e-9) * stop_dist)),
        "stop_distance": float(stop_dist),
    }


def enforce_sector_caps(weights: dict[str, float], sectors: dict[str, str],
                        limits: RiskLimits) -> dict[str, float]:
    """Scale down names so no sector exceeds its cap and gross stays within budget."""
    out = dict(weights)

    # Sector cap.
    by_sector: dict[str, float] = {}
    for sym, w in out.items():
        by_sector[sectors.get(sym, "UNKNOWN")] = by_sector.get(sectors.get(sym, "UNKNOWN"), 0) + w
    for sec, total in by_sector.items():
        if total > limits.max_sector_weight and total > 0:
            factor = limits.max_sector_weight / total
            for sym in out:
                if sectors.get(sym, "UNKNOWN") == sec:
                    out[sym] *= factor

    # Gross cap.
    gross = sum(abs(w) for w in out.values())
    if gross > limits.max_gross_exposure and gross > 0:
        factor = limits.max_gross_exposure / gross
        out = {sym: w * factor for sym, w in out.items()}
    return out
