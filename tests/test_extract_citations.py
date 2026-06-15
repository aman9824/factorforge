"""Mock extraction + span-citation verification."""

from __future__ import annotations

from factorforge.citations.verifier import verify_extraction
from factorforge.corpus.loader import load_corpus
from factorforge.corpus.structure import build_doc_tree
from factorforge.extract.entities import index_document
from factorforge.models import (
    Document,
    EntityType,
    RawEntity,
    RawExtraction,
    RelationType,
)
from factorforge.providers.mock import MockProvider


def _value_doc() -> Document:
    return next(d for d in load_corpus() if d.id == "value-premium")


def test_mock_extracts_expected_entities_with_proven_citations() -> None:
    indexed = index_document(_value_doc(), MockProvider())
    ids = {e.id for e in indexed.extraction.entities}

    assert "factor:value" in ids
    assert "regime:high_inflation" in ids
    assert "asset:equities" in ids

    # Every entity carries at least one citation, and every citation's span resolves back to the
    # exact quoted text in the source (this is the proof, not a promise).
    doc = indexed.doc
    for entity in indexed.extraction.entities:
        assert entity.citations, f"{entity.id} has no citation"
        for cit in entity.citations:
            assert doc.text[cit.start : cit.end] == cit.quote
            assert cit.section  # attributed to a tree section


def test_key_regime_relation_is_present_and_cited() -> None:
    indexed = index_document(_value_doc(), MockProvider())
    edges = {(r.type, r.source, r.target): r for r in indexed.extraction.relations}

    key = (RelationType.AFFECTS, "regime:high_inflation", "factor:value")
    assert key in edges, "expected the value<-high_inflation regime edge for the demo question"

    rel = edges[key]
    assert rel.citation is not None
    assert "inflation" in rel.citation.quote.lower()

    # factor -> asset edge too
    assert (RelationType.AFFECTS, "factor:value", "asset:equities") in edges


def test_relation_endpoints_reference_existing_entities() -> None:
    indexed = index_document(_value_doc(), MockProvider())
    ids = {e.id for e in indexed.extraction.entities}
    for rel in indexed.extraction.relations:
        assert rel.source in ids and rel.target in ids


def test_clean_mock_extraction_has_no_rejections() -> None:
    indexed = index_document(_value_doc(), MockProvider())
    # The mock quotes are real substrings, so nothing should be rejected.
    assert indexed.extraction.rejected == []


def test_verifier_rejects_a_fabricated_quote() -> None:
    doc = _value_doc()
    tree = build_doc_tree(doc)
    raw = RawExtraction(
        entities=[
            RawEntity(type=EntityType.FACTOR, name="value", quote="The value premium"),
            RawEntity(
                type=EntityType.FACTOR,
                name="momentum",
                quote="this exact sentence does not appear in the document at all zzz",
            ),
        ]
    )
    verified = verify_extraction(raw, doc, tree)
    ids = {e.id for e in verified.entities}

    assert "factor:value" in ids          # real quote -> accepted
    assert "factor:momentum" not in ids    # fabricated quote -> rejected
    assert any(r.reason == "quote not found in source" for r in verified.rejected)


def test_mock_navigation_surfaces_relevant_sections() -> None:
    doc = _value_doc()
    tree = build_doc_tree(doc)
    sel = MockProvider().navigate("does the value premium depend on the inflation regime?", tree, max_nodes=4)

    assert sel.node_ids
    from factorforge.corpus.structure import find_node

    titles = [find_node(tree.root, nid).title.lower() for nid in sel.node_ids]  # type: ignore[union-attr]
    assert any("inflation" in t or "regime" in t for t in titles)
    assert sel.thinking  # the reasoning trace is populated
