"""USD/INR spot rate — for display only (converting USD position sizes to an INR-equivalent
a newbie investor can reason about). NEVER used for sizing/risk math — every instrument in
both sleeves is USD-denominated and stays that way; this is purely a comprehension aid.

Always tries a live fetch first via yfinance's "INR=X" ticker (Yahoo's USD/INR spot). The
hardcoded fallback below is a rough historical anchor, NOT a current rate — it WILL be
wrong by the time anyone reads this, since FX moves daily and training data has a cutoff.
Every caller must check `is_live` and show a visible staleness warning when False; silently
displaying a fallback number as if it were today's rate would mislead exactly the user this
feature is meant to help.
"""
from __future__ import annotations

from dataclasses import dataclass

# Rough historical anchor ONLY — not a current rate. Last known order-of-magnitude during
# this project's training data window. Any caller using this MUST surface `is_live=False`.
_FALLBACK_USDINR = 86.0


@dataclass(frozen=True)
class FxRate:
    rate: float
    is_live: bool


def get_usdinr_rate() -> FxRate:
    """Live USD/INR spot, or the stale fallback with is_live=False if the fetch fails."""
    try:
        import yfinance as yf
        data = yf.download("INR=X", period="5d", interval="1d", progress=False, auto_adjust=True)
        if not data.empty:
            close = data["Close"]
            # yfinance can return a 1-col MultiIndex frame for a single ticker.
            val = float(close.iloc[-1].iloc[0]) if hasattr(close.iloc[-1], "iloc") else float(close.iloc[-1])
            if val > 0:
                return FxRate(rate=val, is_live=True)
    except Exception:
        pass
    return FxRate(rate=_FALLBACK_USDINR, is_live=False)


def usd_to_inr(usd: float, fx: FxRate) -> float:
    return usd * fx.rate
