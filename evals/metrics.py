"""Eval scoring — turn pipeline reports into gated metrics."""

from __future__ import annotations

from dataclasses import dataclass

from evals.dataset import EvalCase
from factorforge.models import Document, Report


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


@dataclass
class CaseScore:
    name: str
    factor_match: bool
    recommendation: str
    recommendation_match: bool
    numbers_verified: bool
    golden_ok: bool
    surfaced: float
    citation_coverage: float
    is_overfit_fixture: bool


def _evidence_blob(report: Report) -> str:
    parts: list[str] = []
    for path in report.evidence_paths:
        parts.append(path.text)
        parts.extend(s.label for s in path.steps)
        parts.extend(c.section or "" for c in path.citations)
    return " ".join(parts).lower()


def score_case(case: EvalCase, report: Report, docs: dict[str, Document]) -> CaseScore:
    blob = _evidence_blob(report)
    surfaced = _mean([1.0 if m.lower() in blob else 0.0 for m in case.must_surface])

    citations = [c for p in report.evidence_paths for c in p.citations]
    citations += [c for f in report.findings for c in f.citations]
    resolved = [
        1.0 if (c.doc_id in docs and docs[c.doc_id].text[c.start : c.end] == c.quote) else 0.0
        for c in citations
    ]
    coverage = _mean(resolved) if resolved else 1.0

    golden_ok = case.golden_sharpe is None or abs(report.backtest.stats.sharpe - case.golden_sharpe) <= case.sharpe_tol

    return CaseScore(
        name=case.name,
        factor_match=report.hypothesis.factor_name == case.gold_factor,
        recommendation=report.risk.recommendation.value,
        recommendation_match=report.risk.recommendation == case.gold_recommendation,
        numbers_verified=report.numbers_verified,
        golden_ok=golden_ok,
        surfaced=surfaced,
        citation_coverage=coverage,
        is_overfit_fixture=case.is_overfit_fixture,
    )


@dataclass
class Scorecard:
    cases: list[CaseScore]
    extraction_accuracy: float

    @property
    def factor_accuracy(self) -> float:
        return _mean([1.0 if c.factor_match else 0.0 for c in self.cases])

    @property
    def recommendation_accuracy(self) -> float:
        return _mean([1.0 if c.recommendation_match else 0.0 for c in self.cases])

    @property
    def citation_coverage(self) -> float:
        return _mean([c.citation_coverage for c in self.cases])

    @property
    def retrieval_path_accuracy(self) -> float:
        return _mean([c.surfaced for c in self.cases])

    @property
    def overfitting_catch_rate(self) -> float:
        fixtures = [c for c in self.cases if c.is_overfit_fixture]
        return _mean([1.0 if c.recommendation_match else 0.0 for c in fixtures]) if fixtures else 1.0

    @property
    def backtest_repro_ok(self) -> bool:
        return all(c.numbers_verified and c.golden_ok for c in self.cases)
