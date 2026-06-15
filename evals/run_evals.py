"""Run the eval suite and gate on thresholds.

    python -m evals.run_evals                       # mock backend (default) — deterministic, used in CI
    FF_AGENT_BACKEND=claude python -m evals.run_evals   # score the real multi-agent run

Exits non-zero if any threshold is missed, so CI fails when extraction regresses, a citation is
dropped, retrieval stops surfacing the right evidence, a backtest number drifts, or the Critic
stops catching the overfit fixture. Also prints the vectorless-retrieval cost telemetry.
"""

from __future__ import annotations

import sys

from evals.dataset import GOLD_ENTITIES, load_cases
from evals.metrics import Scorecard, score_case
from factorforge.config import get_settings
from factorforge.corpus.loader import load_corpus
from factorforge.factory import build_data_source, build_provider
from factorforge.knowledge import build_knowledge_base
from factorforge.logging import configure_logging
from factorforge.orchestrator import research
from factorforge.retrieval.retriever import retrieve


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    provider = build_provider(settings)
    kb = build_knowledge_base(provider)
    data_source = build_data_source(settings)
    docs = {d.id: d for d in load_corpus()}

    present = {e.id for e in kb.graph.entities()}
    extraction_accuracy = len(GOLD_ENTITIES & present) / len(GOLD_ENTITIES)

    scores = []
    navigations = 0
    tokens = 0
    for case in load_cases():
        report = research(case.question, settings=settings, kb=kb, data_source=data_source, provider=provider)
        scores.append(score_case(case, report, docs))
        rr = retrieve(kb, case.question, provider, settings)
        navigations += rr.navigations
        tokens += rr.input_tokens + rr.output_tokens
    card = Scorecard(scores, extraction_accuracy)

    print("\n=== FactorForge Eval Scorecard ===")
    print(f"backend={settings.agent_backend.value}  llm={settings.llm_provider.value}  "
          f"data={settings.data_source.value}  cases={len(scores)}\n")
    for c in scores:
        print(f"  {c.name:16} factor={'ok' if c.factor_match else 'NO':3}  rec={c.recommendation:14} "
              f"({'ok' if c.recommendation_match else 'NO'})  verified={str(c.numbers_verified):5}  "
              f"golden={'ok' if c.golden_ok else 'NO':3}  surfaced={c.surfaced:4.0%}  cites={c.citation_coverage:4.0%}")
    print(
        f"\n  AGGREGATE  extraction={card.extraction_accuracy:.0%}  cites={card.citation_coverage:.0%}  "
        f"retrieval={card.retrieval_path_accuracy:.0%}  factor={card.factor_accuracy:.0%}  "
        f"overfit_catch={card.overfitting_catch_rate:.0%}  backtest_repro={'ok' if card.backtest_repro_ok else 'FAIL'}"
    )
    print(f"  COST       vectorless navigations={navigations}  retrieval_tokens={tokens}  "
          f"(each navigation = 1 model call under vertex; mock reports 0 tokens)")

    failures: list[str] = []
    if card.extraction_accuracy < settings.min_extraction_accuracy:
        failures.append(f"extraction_accuracy {card.extraction_accuracy:.0%} < {settings.min_extraction_accuracy:.0%}")
    if card.citation_coverage < settings.min_citation_coverage:
        failures.append(f"citation_coverage {card.citation_coverage:.0%} < {settings.min_citation_coverage:.0%}")
    if card.retrieval_path_accuracy < settings.min_retrieval_path_accuracy:
        failures.append(f"retrieval_path_accuracy {card.retrieval_path_accuracy:.0%} < {settings.min_retrieval_path_accuracy:.0%}")
    if card.overfitting_catch_rate < settings.min_overfitting_catch_rate:
        failures.append(f"overfitting_catch_rate {card.overfitting_catch_rate:.0%} < {settings.min_overfitting_catch_rate:.0%}")
    if card.factor_accuracy < 1.0:
        failures.append(f"factor_accuracy {card.factor_accuracy:.0%} < 100%")
    if not card.backtest_repro_ok:
        failures.append("backtest reproducibility failed (a number was unverified or drifted from gold)")

    if failures:
        print("\nFAIL — thresholds not met:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS - all thresholds met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
