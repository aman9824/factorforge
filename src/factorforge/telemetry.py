"""Cost telemetry — make the vectorless tradeoff visible instead of hiding it.

Vectorless retrieval costs more model calls than vector RAG. Rather than bury that, we count it:
the :class:`CostTracker` accumulates model calls + token usage per stage so a run can report how
much the (token-heavier) navigation actually cost. The mock path reports zero (no model calls).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostTracker:
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_stage: dict[str, int] = field(default_factory=dict)

    def add(self, stage: str, input_tokens: int = 0, output_tokens: int = 0, calls: int = 1) -> None:
        self.model_calls += calls
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.by_stage[stage] = self.by_stage.get(stage, 0) + input_tokens + output_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def summary(self) -> dict[str, object]:
        return {
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "by_stage": dict(self.by_stage),
        }
