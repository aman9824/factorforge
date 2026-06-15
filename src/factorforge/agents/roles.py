"""Agent role definitions.

Each role pairs a name with a real Anthropic **Agent Skill** (``skills/<role>/SKILL.md``) and the
tools it is allowed to call. The Claude backend loads the skill text as the agent's system prompt
and uses ``tools`` to decide which MCP servers/allowed-tools to wire; the mock backend reads the
same registry for dispatch. Declarative roles are what let the orchestrator treat all five agents
uniformly.

Two tool families: the knowledge-graph tools (served by the external MCP server) and the backtest
tools (deterministic, in-process). The split per role is what drives the wiring in the backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_SKILLS_DIR = Path(__file__).parent / "skills"

KG_TOOLS = ("get_evidence", "graph_search", "get_neighbors", "navigate_document", "list_factors", "get_section")
BACKTEST_TOOLS = ("run_backtest",)
DIAGNOSTIC_TOOLS = ("run_diagnostics",)


@dataclass(frozen=True)
class Role:
    name: str
    skill: str                      # subdirectory under skills/
    tools: tuple[str, ...] = ()

    def skill_text(self) -> str:
        return (_SKILLS_DIR / self.skill / "SKILL.md").read_text(encoding="utf-8")


RESEARCHER = Role(name="researcher", skill="researcher", tools=KG_TOOLS)
HYPOTHESIZER = Role(name="hypothesizer", skill="hypothesizer", tools=("get_evidence", "list_factors"))
BACKTESTER = Role(name="backtester", skill="backtester", tools=BACKTEST_TOOLS)
CRITIC = Role(name="critic", skill="critic", tools=DIAGNOSTIC_TOOLS)
REPORTER = Role(name="reporter", skill="reporter", tools=())

ALL_ROLES = (RESEARCHER, HYPOTHESIZER, BACKTESTER, CRITIC, REPORTER)
