"""The agent-backend seam.

A backend executes a single agent *role* against a context and returns structured output. This is
where the system talks to an LLM (or its deterministic stand-in), so it is the one place we
abstract — the orchestrator drives the same Researcher→Hypothesizer→Backtester→Critic→Reporter
sequence regardless of whether real Claude agents or the mock are underneath.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from factorforge.agents.roles import Role
from factorforge.telemetry import CostTracker


class AgentBackend(ABC):
    name: str = "base"
    # Optional cost sink (see LLMProvider.tracker): the Claude backend records each agent turn's
    # token usage here when the orchestrator attaches a tracker. None (no-op) for the mock.
    tracker: CostTracker | None = None

    @abstractmethod
    def run_role(self, role: Role, context: dict[str, Any]) -> dict[str, Any]:
        """Execute ``role`` against ``context``; return a dict matching the role's output shape."""
        raise NotImplementedError
