"""Real multi-agent backend — Claude Agent SDK on Vertex AI.

Each role runs as its own Claude agent whose ``SKILL.md`` is the system prompt:

* **Researcher / Hypothesizer** connect to the **external** knowledge-graph MCP server — the SDK
  spawns ``python -m factorforge.mcp_server`` as a stdio subprocess and exposes its tools as
  ``mcp__kg__*``. This is the standalone-MCP-server technique (the JD deliverable).
* **Backtester / Critic** get **in-process** SDK MCP tools (``create_sdk_mcp_server``) that wrap the
  deterministic backtest tools. Crucially we **capture the tool's real output** rather than trusting
  the model's text — so numbers never originate from the model.
* **Reporter** has no tools.

Imported lazily (see ``factory``) so ``claude-agent-sdk`` stays an optional dependency. Auth is ADC;
the agent uses the ``[1m]``-suffixed model id; the spawned MCP server inherits the ``FF_*`` settings
so its knowledge base matches.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from factorforge.agents.roles import Role
from factorforge.backends.base import AgentBackend
from factorforge.backtest.data import DataSource
from factorforge.backtest.tools import (
    BACKTEST_TOOL,
    DIAGNOSTICS_TOOL,
    run_backtest_tool,
    run_diagnostics_tool,
)
from factorforge.citations.verifier import resolve_citation
from factorforge.config import Settings
from factorforge.knowledge import KnowledgeBase
from factorforge.logging import get_logger
from factorforge.models import (
    Citation,
    EvidencePath,
    Finding,
    Recommendation,
    RiskAssessment,
    RiskFlag,
    Severity,
)

log = get_logger(__name__)

_OUTPUT_HINT = {
    "researcher": '{"findings": [{"claim": "<one sentence>", "doc_id": "<id>", "quote": "<verbatim quote from that doc>"}]}',
    "hypothesizer": '{"factor_name": "value|momentum|quality|size|low_volatility", "thesis": "<paragraph>", "rank_signal": "<characteristic>", "direction": "high_minus_low|low_minus_high", "quantiles": 5}',
    "critic": '{"recommendation": "promising|inconclusive|likely_overfit", "overfitting_risk": "low|medium|high", "risk_flags": [{"title": "...", "severity": "low|medium|high", "detail": "..."}], "rationale": "..."}',
}


class ClaudeAgentBackend(AgentBackend):
    name = "claude"

    def __init__(self, settings: Settings, data_source: DataSource, kb: KnowledgeBase | None = None) -> None:
        self.settings = settings
        self.data_source = data_source
        self.kb = kb

    # ── environments ────────────────────────────────────────────────────────────
    def _agent_env(self) -> dict[str, str]:
        return {
            "CLAUDE_CODE_USE_VERTEX": "1",
            "ANTHROPIC_VERTEX_PROJECT_ID": self.settings.vertex_project_id,
            "CLOUD_ML_REGION": self.settings.vertex_region,
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_SKIP_UPDATE_CHECK": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }

    def _server_env(self) -> dict[str, str]:
        # The spawned MCP server is a plain process: pass through the FF_* settings so its knowledge
        # base is built with the same provider, plus a headless matplotlib backend.
        passthrough = {k: v for k, v in os.environ.items() if k.startswith("FF_") or k in ("PATH", "PYTHONPATH", "HOME", "GOOGLE_APPLICATION_CREDENTIALS")}
        passthrough.update(
            {
                "FF_LLM_PROVIDER": self.settings.llm_provider.value,
                "FF_VERTEX_PROJECT_ID": self.settings.vertex_project_id,
                "FF_VERTEX_REGION": self.settings.vertex_region,
                "FF_VERTEX_MODEL": self.settings.vertex_model,
                "MPLBACKEND": "Agg",
            }
        )
        return passthrough

    # ── dispatch ──────────────────────────────────────────────────────────────────
    def run_role(self, role: Role, context: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        return asyncio.run(self._arun_role(role, context))

    async def _arun_role(self, role: Role, context: dict[str, Any]) -> dict[str, Any]:
        if role.name in ("researcher", "hypothesizer"):
            return await self._kg_role(role, context)
        if role.name == "backtester":
            return await self._backtest_role(role, context)
        if role.name == "critic":
            return await self._critic_role(role, context)
        return await self._reporter_role(role, context)

    # ── driver ────────────────────────────────────────────────────────────────────
    async def _drive(self, options: Any, prompt: str) -> str:
        from claude_agent_sdk import AssistantMessage, TextBlock, query

        parts: list[str] = []

        async def _run() -> None:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            parts.append(block.text)

        import asyncio

        await asyncio.wait_for(_run(), timeout=self.settings.step_timeout_s)
        return "".join(parts)

    # ── KG roles (external MCP server over stdio) ──────────────────────────────────
    async def _kg_role(self, role: Role, context: dict[str, Any]) -> dict[str, Any]:
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(
            model=self.settings.agent_model,
            system_prompt=role.skill_text(),
            permission_mode="bypassPermissions",
            max_turns=self.settings.max_turns,
            allowed_tools=[f"mcp__kg__{t}" for t in role.tools],
            mcp_servers={
                "kg": {
                    "command": sys.executable,
                    "args": ["-m", "factorforge.mcp_server"],
                    "env": self._server_env(),
                }
            },
            env=self._agent_env(),
        )
        prompt = self._build_prompt(role, context)
        raw = await self._drive(options, prompt)
        log.info("claude.kg_role", role=role.name, chars=len(raw))
        if role.name == "researcher":
            return self._parse_researcher(context, raw)
        return self._parse_hypothesizer(context, raw)

    # ── backtest role (in-process tool, captured) ──────────────────────────────────
    async def _backtest_role(self, role: Role, context: dict[str, Any]) -> dict[str, Any]:
        from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool

        captured: dict[str, Any] = {}

        @tool(BACKTEST_TOOL.name, BACKTEST_TOOL.description, {"inputs": dict})
        async def _run_backtest(args: dict[str, Any]) -> dict[str, Any]:
            result = run_backtest_tool(args["inputs"], self.data_source)
            captured["result"] = result
            return {"content": [{"type": "text", "text": json.dumps(result)}]}

        server = create_sdk_mcp_server(name="backtest", tools=[_run_backtest])
        options = ClaudeAgentOptions(
            model=self.settings.agent_model,
            system_prompt=role.skill_text(),
            permission_mode="bypassPermissions",
            max_turns=self.settings.max_turns,
            allowed_tools=["mcp__backtest__run_backtest"],
            mcp_servers={"backtest": server},
            env=self._agent_env(),
        )
        await self._drive(options, self._build_prompt(role, context))
        # Numbers come from the tool, never the model's text.
        result = captured.get("result") or run_backtest_tool(context["hypothesis"], self.data_source)
        return {"reported_stats": result["stats"], "backtest": result}

    # ── critic role (in-process tool + parsed judgment) ────────────────────────────
    async def _critic_role(self, role: Role, context: dict[str, Any]) -> dict[str, Any]:
        from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool

        captured: dict[str, Any] = {}
        diag_inputs = dict(context["hypothesis"])
        if context.get("n_trials") is not None:
            diag_inputs["n_trials"] = context["n_trials"]

        @tool(DIAGNOSTICS_TOOL.name, DIAGNOSTICS_TOOL.description, {"inputs": dict})
        async def _run_diagnostics(args: dict[str, Any]) -> dict[str, Any]:
            result = run_diagnostics_tool(args["inputs"], self.data_source)
            captured["result"] = result
            return {"content": [{"type": "text", "text": json.dumps(result)}]}

        server = create_sdk_mcp_server(name="diagnostics", tools=[_run_diagnostics])
        options = ClaudeAgentOptions(
            model=self.settings.agent_model,
            system_prompt=role.skill_text(),
            permission_mode="bypassPermissions",
            max_turns=self.settings.max_turns,
            allowed_tools=["mcp__diagnostics__run_diagnostics"],
            mcp_servers={"diagnostics": server},
            env=self._agent_env(),
        )
        raw = await self._drive(options, self._build_prompt(role, {"inputs": diag_inputs, **context}))
        diagnostics = captured.get("result") or run_diagnostics_tool(diag_inputs, self.data_source)
        risk = self._parse_risk(raw)
        return {"diagnostics": diagnostics, "risk": risk.model_dump()}

    # ── reporter role (no tools) ────────────────────────────────────────────────────
    async def _reporter_role(self, role: Role, context: dict[str, Any]) -> dict[str, Any]:
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(
            model=self.settings.agent_model,
            system_prompt=role.skill_text(),
            permission_mode="bypassPermissions",
            max_turns=self.settings.max_turns,
            env=self._agent_env(),
        )
        narrative = await self._drive(options, self._build_prompt(role, context))
        return {"narrative": narrative.strip()}

    # ── prompts + parsing ──────────────────────────────────────────────────────────
    def _build_prompt(self, role: Role, context: dict[str, Any]) -> str:
        ctx = json.dumps(context, indent=2, default=str)
        hint = _OUTPUT_HINT.get(role.name)
        tail = (
            f"\n\nWhen done, respond with ONLY a JSON object of exactly this shape:\n{hint}\n"
            "Do not include any prose outside the JSON."
            if hint
            else "\n\nWrite the final report as plain text."
        )
        return f"Context (JSON):\n{ctx}{tail}"

    def _parse_researcher(self, context: dict[str, Any], raw: str) -> dict[str, Any]:
        data = _extract_json(raw)
        findings: list[Finding] = []
        for item in data.get("findings", []):
            citation = self._verify_quote(item.get("doc_id", ""), item.get("quote", ""))
            if citation is not None:
                findings.append(Finding(claim=str(item.get("claim", "")).strip(), citations=[citation]))
        return {"question": context.get("question", ""), "findings": [f.model_dump() for f in findings], "evidence_paths": []}

    def _parse_hypothesizer(self, context: dict[str, Any], raw: str) -> dict[str, Any]:
        data = _extract_json(raw)
        paths = [EvidencePath.model_validate(p) for p in context.get("evidence_paths", [])]
        factor = str(data.get("factor_name", "value"))
        supporting = [c for p in paths for c in p.citations if f"factor:{factor}" in " ".join(s.label for s in p.steps)][:3]
        return {
            "hypothesis": {
                "factor_name": factor,
                "thesis": str(data.get("thesis", "")),
                "rank_signal": str(data.get("rank_signal", "characteristic")),
                "direction": data.get("direction", "high_minus_low"),
                "quantiles": int(data.get("quantiles", 5)),
                "rebalance": "M",
                "universe": [],
                "supporting_citations": [c.model_dump() for c in supporting],
            }
        }

    def _parse_risk(self, raw: str) -> RiskAssessment:
        data = _extract_json(raw)
        flags = [
            RiskFlag(title=str(f.get("title", "")), severity=_severity(f.get("severity")), detail=str(f.get("detail", "")))
            for f in data.get("risk_flags", [])
        ]
        return RiskAssessment(
            recommendation=_recommendation(data.get("recommendation")),
            overfitting_risk=_severity(data.get("overfitting_risk")),
            risk_flags=flags,
            rationale=str(data.get("rationale", "")),
        )

    def _verify_quote(self, doc_id: str, quote: str) -> Citation | None:
        if self.kb is None or not doc_id or not quote:
            return None
        doc, tree = self.kb.get_doc(doc_id), self.kb.get_tree(doc_id)
        if doc is None or tree is None:
            return None
        return resolve_citation(doc, tree, quote)


def _severity(raw: object) -> Severity:
    try:
        return Severity(str(raw).lower())
    except ValueError:
        return Severity.MEDIUM


def _recommendation(raw: object) -> Recommendation:
    try:
        return Recommendation(str(raw).lower())
    except ValueError:
        return Recommendation.INCONCLUSIVE


def _extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.removeprefix("json").strip()
            if part.startswith("{"):
                text = part
                break
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        parsed: dict[str, Any] = json.loads(text[start : end + 1])
        return parsed
    except json.JSONDecodeError:
        return {}
