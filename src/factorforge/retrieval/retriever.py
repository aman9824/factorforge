"""The vectorless retriever — composes graph nav + tree search + structured filters.

Flow for a query:
1. **anchor** it to concept entities (graph term search);
2. **filter** the corpus to documents that mention those anchors (graph-driven scoping);
3. gather **graph** evidence (cited edges + a connecting path);
4. **navigate** each scoped document's tree for section-level evidence;
5. score every path by query-term overlap, sort, and cap.

Every result is an :class:`EvidencePath` with a readable step trace and proven citations — no
embeddings, no opaque similarity score.
"""

from __future__ import annotations

from factorforge.config import Settings
from factorforge.extract.vocab import tokenize
from factorforge.knowledge import KnowledgeBase
from factorforge.models import EvidencePath, RetrievalResult
from factorforge.providers.base import LLMProvider
from factorforge.retrieval.filters import anchor_entities, docs_mentioning
from factorforge.retrieval.graph_nav import graph_evidence
from factorforge.retrieval.tree_search import tree_evidence


def _score(query_tokens: set[str], path: EvidencePath) -> float:
    text_overlap = len(query_tokens & tokenize(path.text))
    label_overlap = len(query_tokens & tokenize(" ".join(s.label for s in path.steps)))
    return float(2 * text_overlap + label_overlap)


def retrieve(
    kb: KnowledgeBase, query: str, provider: LLMProvider, settings: Settings
) -> RetrievalResult:
    anchors = anchor_entities(kb, query, limit=6)

    # Graph-driven scoping: only navigate documents the graph says mention an anchor.
    candidates = {doc.id: doc for anchor in anchors for doc in docs_mentioning(kb, anchor.id)}
    if not candidates:
        candidates = dict(kb.docs)

    paths: list[EvidencePath] = graph_evidence(kb, query, anchors, settings.max_evidence_paths)

    # Measure the token cost of *this* retrieval's navigations via the provider's cost tracker
    # (graph evidence above makes no model call). The delta isolates the navigate() usage; with the
    # mock provider — or no tracker attached — it stays zero.
    tracker = provider.tracker
    in_before = tracker.input_tokens if tracker is not None else 0
    out_before = tracker.output_tokens if tracker is not None else 0

    for doc_id in sorted(candidates):
        path = tree_evidence(
            candidates[doc_id], kb.trees[doc_id], query, provider, settings.max_nav_nodes
        )
        if path is not None:
            paths.append(path)

    query_tokens = tokenize(query)
    for path in paths:
        path.score = _score(query_tokens, path)
    paths.sort(key=lambda p: -p.score)

    nav_in = (tracker.input_tokens - in_before) if tracker is not None else 0
    nav_out = (tracker.output_tokens - out_before) if tracker is not None else 0

    # One hierarchical-doc navigation per scoped document (each is a model call in vertex mode).
    return RetrievalResult(
        query=query,
        paths=paths[: settings.max_evidence_paths],
        navigations=len(candidates),
        input_tokens=nav_in,
        output_tokens=nav_out,
    )
