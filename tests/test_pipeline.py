"""End-to-end multi-agent pipeline on the deterministic mock backend."""

from __future__ import annotations

from factorforge.config import Settings
from factorforge.models import DISCLAIMER, Recommendation
from factorforge.orchestrator import research

QUESTION = "Does the value premium depend on the inflation regime?"


def _settings() -> Settings:
    return Settings(audit_enabled=False)  # no disk writes in tests


def test_pipeline_produces_verified_promising_value_report() -> None:
    report = research(QUESTION, settings=_settings())

    assert report.backend == "mock"
    assert report.hypothesis.factor_name == "value"
    assert report.numbers_verified is True            # agent's reported stats matched the re-run
    assert report.backtest.stats.sharpe > 1.0          # the planted premium
    assert report.risk.recommendation == Recommendation.PROMISING


def test_report_is_evidence_linked_and_cited() -> None:
    report = research(QUESTION, settings=_settings())

    assert report.evidence_paths and report.findings
    # Every citation in the report resolves to its exact source span (rebuild the corpus to check).
    from factorforge.corpus.loader import load_corpus

    docs = {d.id: d for d in load_corpus()}
    for path in report.evidence_paths:
        for cit in path.citations:
            assert docs[cit.doc_id].text[cit.start : cit.end] == cit.quote
    for finding in report.findings:
        assert finding.citations
        for cit in finding.citations:
            assert docs[cit.doc_id].text[cit.start : cit.end] == cit.quote


def test_report_is_honest_and_deterministic() -> None:
    a = research(QUESTION, settings=_settings())
    b = research(QUESTION, settings=_settings())
    assert a.model_dump() == b.model_dump()          # deterministic mock pipeline
    assert a.disclaimer == DISCLAIMER
    assert "not financial advice" in a.narrative.lower()
    assert "value" in a.narrative.lower()
