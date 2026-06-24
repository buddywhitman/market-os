"""Capital ring-fencing — three sleeves, three temperaments, three risk budgets.

The whole point of this module is a HARD boundary: goal-1 wave-riding aggression must
never be able to draw down the goal-2 quant book. Allocation and sizing always run
*within* one sleeve; there is no cross-sleeve netting or rebalancing. Each sleeve owns:

  * its own universe (which instruments it may hold),
  * its own ``RiskLimits`` (how aggressively it may size and when it de-risks),
  * an execution mode (auto / briefing / none),
  * two honesty flags (``can_lose_all``, ``allow_leverage``) that downstream code can
    assert against before, say, putting a 3x ETF in the survival-first book.

Config lives in ``config/config.yaml`` under ``sleeves:`` so the ring-fence is versioned
in git, not buried in code. Code defaults below mirror that file so the module is still
usable (and testable) with no config present.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from marketos.risk.sizing import RiskLimits

# Valid execution modes. "auto" = system places orders (Indian instruments via AngelOne);
# "briefing" = morning briefing + realtime alerts, human places the trade (US instruments);
# "none" = research only, no capital.
EXECUTION_MODES = ("auto", "briefing", "none")

# Instruments whose price action is itself leveraged (daily-reset ETFs, etc.). These may
# only live in a sleeve with allow_leverage=True — putting a 3x daily-reset product in the
# survival-first book would blow the drawdown budget through path dependency alone, well
# before any stop fires. Extend as leveraged names are added (SOXS, TQQQ, UPRO, ...).
LEVERAGED_INSTRUMENTS = frozenset({"SOXL", "SOXS", "TQQQ", "SQQQ", "UPRO", "SPXU", "TECL", "TECS"})


@dataclass(frozen=True)
class Sleeve:
    name: str
    description: str
    universe: tuple[str, ...]
    limits: RiskLimits
    can_lose_all: bool = False
    allow_leverage: bool = False
    execution: str = "none"
    currency: str = "USD"
    # REAL capital, not a paper/demo figure — sizing math must reflect what's actually
    # there. A ₹5,000 account produces meaningfully different (and more concentrated)
    # position sizes than a $100k one; using a fantasy number here was the bug the user
    # caught (the india/quant sleeves were both implicitly sized off a $100k paper
    # capital_base regardless of real account size).
    capital: float = 100_000.0

    def __post_init__(self) -> None:
        if self.execution not in EXECUTION_MODES:
            raise ValueError(
                f"sleeve {self.name!r}: execution={self.execution!r} not in {EXECUTION_MODES}")

    def holds(self, symbol: str) -> bool:
        return symbol in self.universe


# ── Code defaults (mirror config/config.yaml so the module works config-free) ──────────

_DEFAULT_SLEEVES = {
    "aggressive": Sleeve(
        name="aggressive",
        description="Short-term secular-wave riding. Willing to lose all. LLM+regime driven.",
        universe=("SOXL", "NVDA", "AMD", "AVGO", "PLTR", "RKLB", "COIN", "MSTR", "BTC-USD",
                  # Memory + other semicons added 2026-06-23 per user request — not in the
                  # quant sleeve's frozen 19-name validated cross-section.
                  "MU", "WDC", "STX", "TSM", "QCOM", "INTC"),
        limits=RiskLimits(
            risk_per_trade=0.03, max_name_weight=0.40, max_sector_weight=1.00,
            max_gross_exposure=1.00, kelly_fraction=0.20, atr_stop_mult=3.0,
            max_portfolio_drawdown=0.50,
        ),
        can_lose_all=True, allow_leverage=True, execution="briefing",
    ),
    "quant": Sleeve(
        name="quant",
        description="Mid-term quant wealth engine. Survival-first, OOS-validated.",
        universe=("NVDA", "AMD", "AVGO", "MSFT", "PLTR", "GEV", "VST", "CEG", "ETN",
                  "LMT", "RTX", "NOC", "CCJ", "RKLB", "PATH", "COIN", "MSTR", "SPY", "QQQ"),
        limits=RiskLimits(),  # book-wide survival-first defaults
        can_lose_all=False, allow_leverage=False, execution="auto",
    ),
    "india": Sleeve(
        name="india",
        description="Real ₹5,000→₹50,000 growth capital. NSE via AngelOne. High order "
                    "frequency, diversified across screened names (not one bet) — but "
                    "concentrated enough that each position is still meaningful at this "
                    "account size. Graduates to wealth-protection sizing once ₹50k is hit.",
        universe=tuple(),  # the screened/ranked candidate list IS the live universe, not a fixed one
        limits=RiskLimits(
            risk_per_trade=0.05, max_name_weight=0.25, max_sector_weight=0.50,
            max_gross_exposure=1.00, kelly_fraction=0.30, atr_stop_mult=2.5,
            max_portfolio_drawdown=0.40,
        ),
        can_lose_all=True, allow_leverage=False, execution="auto",
        currency="INR", capital=5_000.0,
    ),
    "research": Sleeve(
        name="research",
        description="Research, edge detection, world-models. No capital.",
        universe=(), limits=RiskLimits(),
        can_lose_all=False, allow_leverage=False, execution="none",
    ),
}


def load_sleeves(config: dict | None = None) -> dict[str, Sleeve]:
    """Build the sleeve map from a parsed config dict (the ``raw`` field of
    ``marketos.config.Config``). Falls back to code defaults when ``sleeves:`` is absent.

    A sleeve that omits its own ``risk:`` block inherits the book-wide ``risk:`` defaults,
    which themselves fall back to ``RiskLimits()`` — so the survival-first numbers live in
    exactly one place and a partial override never silently resets the rest of the fields.
    """
    if not config or "sleeves" not in config:
        return dict(_DEFAULT_SLEEVES)

    book_default = RiskLimits.from_dict(config.get("risk"))
    out: dict[str, Sleeve] = {}
    for name, spec in (config["sleeves"] or {}).items():
        spec = spec or {}
        limits = RiskLimits.from_dict(spec["risk"]) if spec.get("risk") else book_default
        out[name] = Sleeve(
            name=name,
            description=spec.get("description", ""),
            universe=tuple(spec.get("universe", []) or []),
            limits=limits,
            can_lose_all=bool(spec.get("can_lose_all", False)),
            allow_leverage=bool(spec.get("allow_leverage", False)),
            execution=spec.get("execution", "none"),
            currency=spec.get("currency", "USD"),
            # Real capital, read from config so it can be updated as the account actually
            # grows (₹5,000 → ₹50,000) WITHOUT a code change — falls back to the dataclass
            # default (100,000) only if a sleeve's config omits it entirely.
            capital=float(spec.get("capital", 100_000.0)),
        )
    _validate_ringfence(out)
    return out


def _validate_ringfence(sleeves: dict[str, Sleeve]) -> None:
    """Fail loud at load time if a leveraged instrument leaks into a non-leverage sleeve.
    A silent leak here is exactly the cross-contamination the whole module exists to prevent."""
    for s in sleeves.values():
        if s.allow_leverage:
            continue
        leaked = LEVERAGED_INSTRUMENTS & set(s.universe)
        if leaked:
            raise ValueError(
                f"sleeve {s.name!r} has allow_leverage=False but holds leveraged "
                f"instrument(s) {sorted(leaked)} — this would breach its drawdown budget "
                f"through path dependency. Move them to a leverage-enabled sleeve.")


def sleeve_for_symbol(symbol: str, sleeves: dict[str, Sleeve]) -> list[str]:
    """Which sleeves are allowed to hold ``symbol``. A name can legitimately appear in
    more than one sleeve (e.g. NVDA in both aggressive and quant) — they are sized
    independently under each sleeve's own risk budget, never netted."""
    return [s.name for s in sleeves.values() if s.holds(symbol)]
