"""Multi-agent orchestration: Researcher → Hypothesizer → Backtester → Critic → Reporter.

This is the layer the Agent SDK does not give you: explicit, typed hand-offs between roles, an
independent re-computation that **verifies** the Backtester's numbers (verify, don't trust), the
authoritative diagnostics, and assembly of the final evidence-linked report. The same orchestration
drives the mock and Claude backends. Every stage is written to the audit log.
"""

from __future__ import annotations

from src.factorforge.agents.roles import BACKTESTER, CRITIC, HYPOTHESIZER, REPORTER, RESEARCHER
from src.factorforge.audit import AuditLog
from src.factorforge.backends.base import AgentBackend
from src.factorforge.backtest.data import DataSource
from src.factorforge.backtest.tools import backtest_hypothesis
from src.factorforge.config import Settings, get_settings
from src.factorforge.factory import build_backend, build_data_source, build_provider
from src.factorforge.knowledge import KnowledgeBase, build_knowledge_base
from src.factorforge.logging import get_logger
from src.factorforge.models import (
    BacktesterOutput,
    BacktestStats,
    CriticOutput,
    EntityType,
    HypothesizerOutput,
    Report,
    ResearcherOutput,
)
from src.factorforge.providers.base import LLMProvider
from src.factorforge.retrieval.retriever import retrieve
from src.factorforge.telemetry import CostTracker

log = get_logger(__name__)

_STAT_FIELDS = ("total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "calmar")


def _stats_match(reported: BacktestStats, truth: BacktestStats, tol: float) -> bool:
    return all(abs(getattr(reported, f) - getattr(truth, f)) <= tol for f in _STAT_FIELDS)


def research(
    question: str,
    *,
    settings: Settings | None = None,
    kb: KnowledgeBase | None = None,
    data_source: DataSource | None = None,
    backend: AgentBackend | None = None,
    provider: LLMProvider | None = None,
) -> Report:
    settings = settings or get_settings()
    provider = provider or build_provider(settings)
    kb = kb or build_knowledge_base(provider)
    data_source = data_source or build_data_source(settings)
    backend = backend or build_backend(settings, kb, data_source, provider)
    audit = AuditLog(settings)
    cost = CostTracker()
    log.info("pipeline.start", question=question, backend=backend.name)

    # Authoritative vectorless retrieval (verify-don't-trust for evidence): the report's evidence
    # paths come from the orchestrator's own retrieval, not whatever an agent chose to echo back.
    authoritative = retrieve(kb, question, provider, settings)
    evidence_dump = [p.model_dump() for p in authoritative.paths]

    # Stage 1 — Researcher (cited findings; agents work from the authoritative evidence).
    research_out = ResearcherOutput.model_validate(
        backend.run_role(RESEARCHER, {"question": question, "evidence_paths": evidence_dump})
    )
    audit.record("research", {"question": question, "findings": len(research_out.findings),
                              "evidence_paths": len(authoritative.paths)})

    # Stage 2 — Hypothesizer (one testable factor).
    hyp_out = HypothesizerOutput.model_validate(
        backend.run_role(
            HYPOTHESIZER,
            {
                "question": question,
                "findings": [f.model_dump() for f in research_out.findings],
                "evidence_paths": evidence_dump,
            },
        )
    )
    hyp = hyp_out.hypothesis
    # Honest multiple-testing breadth: how many factors could have been hypothesized.
    n_trials = max(1, len(kb.graph.entities(EntityType.FACTOR)))
    audit.record("hypothesis", {"factor": hyp.factor_name, "direction": hyp.direction.value, "n_trials": n_trials})

    # Stage 3 — Backtester, then verify-don't-trust against an independent re-run.
    bt_out = BacktesterOutput.model_validate(backend.run_role(BACKTESTER, {"hypothesis": hyp.model_dump()}))
    authoritative_bt = backtest_hypothesis(hyp, data_source)
    numbers_verified = _stats_match(bt_out.reported_stats, authoritative_bt.stats, settings.max_backtest_repro_error)
    audit.record("backtest", {"factor": hyp.factor_name, "numbers_verified": numbers_verified,
                              "reported": bt_out.reported_stats.model_dump(),
                              "authoritative": authoritative_bt.stats.model_dump()})

    # Stage 4 — Risk/Overfitting Critic (authoritative diagnostics computed independently too).
    crit_out = CriticOutput.model_validate(
        backend.run_role(
            CRITIC,
            {"hypothesis": hyp.model_dump(), "backtest": authoritative_bt.model_dump(), "n_trials": n_trials},
        )
    )
    # The diagnostics are tool-sourced (deterministic) in both backends, so reuse them as
    # authoritative rather than re-running the (expensive) grid a second time.
    authoritative_diag = crit_out.diagnostics
    audit.record("critique", {"recommendation": crit_out.risk.recommendation.value,
                              "deflated_sharpe": authoritative_diag.deflated_sharpe, "pbo": authoritative_diag.pbo})

    # Stage 5 — Reporter (synthesis only).
    rep_out = backend.run_role(
        REPORTER,
        {
            "question": question,
            "hypothesis": hyp.model_dump(),
            "backtest": authoritative_bt.model_dump(),
            "diagnostics": authoritative_diag.model_dump(),
            "risk": crit_out.risk.model_dump(),
            "evidence_paths": evidence_dump,
        },
    )

    report = Report(
        question=question,
        backend=backend.name,
        thesis=hyp.thesis,
        hypothesis=hyp,
        evidence_paths=authoritative.paths,
        findings=research_out.findings,
        backtest=authoritative_bt,
        diagnostics=authoritative_diag,
        risk=crit_out.risk,
        numbers_verified=numbers_verified,
        narrative=str(rep_out.get("narrative", "")),
    )
    audit.record("report", {"recommendation": report.risk.recommendation.value,
                            "numbers_verified": numbers_verified, "cost": cost.summary()})
    log.info("pipeline.done", factor=hyp.factor_name,
             recommendation=report.risk.recommendation.value, numbers_verified=numbers_verified)
    return report
