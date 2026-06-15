"""Deterministic, zero-credential agent backend.

This is **not** the AI. It is a deterministic stand-in that runs the *real* retrieval, graph, and
backtest tools — only the reasoning prose and the judgment are rule-based. So a mock pipeline run
produces genuine evidence paths and genuine, verifiable backtest numbers; the real reasoning lives
in :class:`~factorforge.backends.claude.ClaudeAgentBackend`.
"""

from __future__ import annotations

from typing import Any

from src.factorforge.agents.roles import Role
from src.factorforge.backends.base import AgentBackend
from src.factorforge.backtest.data import DataSource
from src.factorforge.backtest.tools import run_backtest_tool, run_diagnostics_tool
from src.factorforge.config import Settings
from src.factorforge.knowledge import KnowledgeBase
from src.factorforge.models import (
    BacktestResult,
    EvidencePath,
    FactorDirection,
    Finding,
    OverfittingReport,
    Recommendation,
    RiskAssessment,
    RiskFlag,
    Severity,
)
from src.factorforge.providers.base import LLMProvider
from src.factorforge.retrieval.retriever import retrieve

# Conventional construction per factor (so the mock forms a sensible hypothesis).
_DIRECTION = {
    "value": FactorDirection.HIGH_MINUS_LOW,
    "momentum": FactorDirection.HIGH_MINUS_LOW,
    "quality": FactorDirection.HIGH_MINUS_LOW,
    "size": FactorDirection.LOW_MINUS_HIGH,
    "low_volatility": FactorDirection.LOW_MINUS_HIGH,
}
_RANK_SIGNAL = {
    "value": "book_to_market",
    "momentum": "trailing_12_1_return",
    "quality": "gross_profitability",
    "size": "market_cap",
    "low_volatility": "trailing_volatility",
}


def _first_sentence(text: str, cap: int = 200) -> str:
    text = " ".join(text.split())
    for end in (". ", "! ", "? "):
        idx = text.find(end)
        if 0 < idx < cap:
            return text[: idx + 1]
    return text[:cap]


class MockBackend(AgentBackend):
    name = "mock"

    def __init__(
        self, settings: Settings, kb: KnowledgeBase, data_source: DataSource, provider: LLMProvider
    ) -> None:
        self.settings = settings
        self.kb = kb
        self.data_source = data_source
        self.provider = provider

    def run_role(self, role: Role, context: dict[str, Any]) -> dict[str, Any]:
        handler = {
            "researcher": self._research,
            "hypothesizer": self._hypothesize,
            "backtester": self._backtest,
            "critic": self._critique,
            "reporter": self._report,
        }.get(role.name)
        if handler is None:
            raise ValueError(f"MockBackend has no handler for role {role.name!r}")
        return handler(context)

    # ── researcher ─────────────────────────────────────────────────────────────
    def _research(self, ctx: dict[str, Any]) -> dict[str, Any]:
        question = ctx["question"]
        # The orchestrator supplies the authoritative evidence; fall back to retrieving if absent.
        if ctx.get("evidence_paths"):
            paths = [EvidencePath.model_validate(p) for p in ctx["evidence_paths"]]
        else:
            paths = retrieve(self.kb, question, self.provider, self.settings).paths

        findings: list[Finding] = []
        seen: set[str] = set()
        for path in paths:
            if not path.citations:
                continue
            claim = _first_sentence(path.citations[0].quote)
            if claim in seen:
                continue
            seen.add(claim)
            findings.append(Finding(claim=claim, citations=path.citations[:2]))
            if len(findings) >= 6:
                break

        return {
            "question": question,
            "findings": [f.model_dump() for f in findings],
            "evidence_paths": [p.model_dump() for p in paths],
        }

    # ── hypothesizer ───────────────────────────────────────────────────────────
    def _hypothesize(self, ctx: dict[str, Any]) -> dict[str, Any]:
        paths = [EvidencePath.model_validate(p) for p in ctx.get("evidence_paths", [])]
        factor = self._infer_factor(paths)
        citations = [c for p in paths for c in p.citations if f"factor:{factor}" in " ".join(s.label for s in p.steps)][:3]
        direction = _DIRECTION.get(factor, FactorDirection.HIGH_MINUS_LOW)
        thesis = (
            f"The {factor} factor is supported by the retrieved evidence for the question "
            f"'{ctx.get('question', '')}'. Test it as a cross-sectional long/short portfolio."
        )
        return {
            "hypothesis": {
                "factor_name": factor,
                "thesis": thesis,
                "rank_signal": _RANK_SIGNAL.get(factor, "characteristic"),
                "direction": direction.value,
                "quantiles": 5,
                "rebalance": "M",
                "universe": [],
                "supporting_citations": [c.model_dump() for c in citations],
            }
        }

    def _infer_factor(self, paths: list[EvidencePath]) -> str:
        counts: dict[str, int] = {}
        for path in paths:
            for step in path.steps:
                for token in step.label.split():
                    if token.startswith("factor:"):
                        name = token.split(":", 1)[1].split("-")[0]
                        counts[name] = counts.get(name, 0) + 1
        if counts:
            return max(counts, key=lambda k: (counts[k], k))
        return "value"

    # ── backtester ─────────────────────────────────────────────────────────────
    def _backtest(self, ctx: dict[str, Any]) -> dict[str, Any]:
        result = run_backtest_tool(ctx["hypothesis"], self.data_source)  # the real, authoritative tool
        return {
            "reported_stats": result["stats"],
            "backtest": result,
        }

    # ── critic ───────────────────────────────────────────────────────────────────
    def _critique(self, ctx: dict[str, Any]) -> dict[str, Any]:
        inputs = dict(ctx["hypothesis"])
        if ctx.get("n_trials") is not None:
            inputs["n_trials"] = ctx["n_trials"]
        report = OverfittingReport.model_validate(run_diagnostics_tool(inputs, self.data_source))
        risk = self._judge(report)
        return {"diagnostics": report.model_dump(), "risk": risk.model_dump()}

    @staticmethod
    def _judge(d: OverfittingReport) -> RiskAssessment:
        flags: list[RiskFlag] = []
        # Deflated Sharpe is the headline anti-overfitting test (it already accounts for the number
        # of trials). PBO via CSCV is reported but NOT a gate here: with few similar-quality configs
        # it is underpowered and ~random, so it would false-positive on robust factors.
        if d.deflated_sharpe < 0.90:
            flags.append(RiskFlag(title="Low deflated Sharpe", severity=Severity.HIGH,
                                  detail=f"Deflated Sharpe {d.deflated_sharpe:.2f} < 0.90 after adjusting for {d.n_trials} trials."))
        if d.sharpe_decay > 1.0:
            flags.append(RiskFlag(title="Severe in/out-of-sample decay", severity=Severity.HIGH,
                                  detail=f"Sharpe falls {d.sharpe_decay:.2f} from in- to out-of-sample."))
        elif d.sharpe_decay > 0.5:
            flags.append(RiskFlag(title="In/out-of-sample decay", severity=Severity.MEDIUM,
                                  detail=f"Sharpe falls {d.sharpe_decay:.2f} from in- to out-of-sample."))
        if d.param_sensitivity > 0.5:
            flags.append(RiskFlag(title="Parameter sensitivity", severity=Severity.MEDIUM,
                                  detail=f"Sharpe dispersion across the parameter grid is {d.param_sensitivity:.2f}."))

        has_high = any(f.severity == Severity.HIGH for f in flags)
        if has_high:
            rec, risk = Recommendation.LIKELY_OVERFIT, Severity.HIGH
        elif flags:
            rec, risk = Recommendation.INCONCLUSIVE, Severity.MEDIUM
        else:
            rec, risk = Recommendation.PROMISING, Severity.LOW

        rationale = (
            f"Deflated Sharpe {d.deflated_sharpe:.2f}, PBO {d.pbo:.0%}, IS/OOS decay "
            f"{d.sharpe_decay:+.2f} over {d.n_trials} trials -> '{rec.value}'."
        )
        return RiskAssessment(recommendation=rec, overfitting_risk=risk, risk_flags=flags, rationale=rationale)

    # ── reporter ─────────────────────────────────────────────────────────────────
    def _report(self, ctx: dict[str, Any]) -> dict[str, Any]:
        hyp = ctx["hypothesis"]
        bt = BacktestResult.model_validate(ctx["backtest"])
        risk = RiskAssessment.model_validate(ctx["risk"])
        n_cites = sum(len(EvidencePath.model_validate(p).citations) for p in ctx.get("evidence_paths", []))
        narrative = (
            f"Question: {ctx.get('question', '')}\n"
            f"Verdict: {risk.recommendation.value} (overfitting risk: {risk.overfitting_risk.value}).\n"
            f"The evidence ({n_cites} citations) supports testing the {hyp['factor_name']} factor as a "
            f"{hyp['direction']} long/short. Backtest (data: {bt.data_source}): Sharpe "
            f"{bt.stats.sharpe:.2f}, CAGR {bt.stats.cagr:.1%}, max drawdown {bt.stats.max_drawdown:.1%}, "
            f"IC {bt.information_coefficient:+.2f}. {risk.rationale} "
            f"This is research/educational only and NOT financial advice."
        )
        return {"narrative": narrative}
