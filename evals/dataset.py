"""Eval dataset — questions with gold expectations, plus the entity gold for extraction.

Two cases run through the *whole* pipeline:
* ``value-inflation`` — a real, regime-linked premium that must come out ``promising`` and verified;
* ``size-overfit``    — the deliberately-overfit fixture (no planted premium) the Critic must flag.
"""

from __future__ import annotations

from dataclasses import dataclass

from factorforge.models import Recommendation

# Entities the extraction must recover from the bundled corpus.
GOLD_ENTITIES = frozenset(
    {
        "factor:value", "factor:momentum", "factor:size", "factor:quality", "factor:low_volatility",
        "regime:high_inflation", "regime:recession", "regime:market_recovery",
        "asset:equities", "author:Fama", "author:Jegadeesh", "author:Banz",
    }
)


@dataclass(frozen=True)
class EvalCase:
    name: str
    question: str
    gold_factor: str
    gold_recommendation: Recommendation
    must_surface: tuple[str, ...]          # substrings the retrieved evidence must contain
    is_overfit_fixture: bool = False
    golden_sharpe: float | None = None     # frozen value to catch math regressions
    sharpe_tol: float = 0.05


def load_cases() -> list[EvalCase]:
    return [
        EvalCase(
            name="value-inflation",
            question="Does the value premium depend on the inflation regime?",
            gold_factor="value",
            gold_recommendation=Recommendation.PROMISING,
            must_surface=("high_inflation", "value", "inflation"),
            golden_sharpe=1.68,
            sharpe_tol=0.05,
        ),
        EvalCase(
            name="size-overfit",
            question="Is the small-cap size premium reliable out of sample?",
            gold_factor="size",
            gold_recommendation=Recommendation.LIKELY_OVERFIT,
            must_surface=("size",),
            is_overfit_fixture=True,
        ),
    ]
