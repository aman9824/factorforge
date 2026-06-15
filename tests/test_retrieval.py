"""Vectorless retrieval: graph nav + tree search + filters → cited evidence paths."""

from __future__ import annotations

from factorforge.config import get_settings
from factorforge.knowledge import KnowledgeBase, build_knowledge_base
from factorforge.models import PathKind
from factorforge.providers.mock import MockProvider
from factorforge.retrieval.retriever import retrieve

QUERY = "Does the value premium depend on the inflation regime?"


def _kb() -> KnowledgeBase:
    return build_knowledge_base(MockProvider())


def test_retrieval_returns_ranked_cited_paths() -> None:
    kb = _kb()
    result = retrieve(kb, QUERY, MockProvider(), get_settings())

    assert result.paths, "expected at least one evidence path"
    scores = [p.score for p in result.paths]
    assert scores == sorted(scores, reverse=True)  # ranked
    assert scores[0] > 0


def test_graph_path_connects_regime_to_factor() -> None:
    kb = _kb()
    result = retrieve(kb, QUERY, MockProvider(), get_settings())

    graph_paths = [p for p in result.paths if p.kind == PathKind.GRAPH]
    assert graph_paths
    # Some graph step explicitly links the inflation regime to the value factor.
    labels = [s.label for p in graph_paths for s in p.steps]
    assert any("high_inflation" in lbl and "value" in lbl for lbl in labels)


def test_tree_path_navigates_to_a_regime_section() -> None:
    kb = _kb()
    result = retrieve(kb, QUERY, MockProvider(), get_settings())

    tree_paths = [p for p in result.paths if p.kind == PathKind.TREE]
    assert tree_paths
    sections = [c.section or "" for p in tree_paths for c in p.citations]
    assert any("inflation" in s.lower() or "regime" in s.lower() for s in sections)


def test_every_citation_resolves_to_its_exact_span() -> None:
    kb = _kb()
    result = retrieve(kb, QUERY, MockProvider(), get_settings())
    for path in result.paths:
        for cit in path.citations:
            doc = kb.docs[cit.doc_id]
            assert doc.text[cit.start : cit.end] == cit.quote


def test_scoping_limits_to_relevant_documents() -> None:
    kb = _kb()
    result = retrieve(kb, QUERY, MockProvider(), get_settings())
    tree_docs = {c.doc_id for p in result.paths if p.kind == PathKind.TREE for c in p.citations}
    # The value note must be searched; an unrelated note (low-volatility) must not be.
    assert "value-premium" in tree_docs
    assert "low-volatility" not in tree_docs


def test_a_different_query_surfaces_different_evidence() -> None:
    kb = _kb()
    result = retrieve(kb, "Why does momentum crash during a market recovery?", MockProvider(), get_settings())
    blob = " ".join(p.text.lower() for p in result.paths)
    assert "momentum" in blob
    assert "recovery" in blob or "crash" in blob
