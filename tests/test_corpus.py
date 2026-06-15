"""Corpus loading + hierarchical structure tree."""

from __future__ import annotations

from factorforge.corpus.loader import load_corpus
from factorforge.corpus.structure import (
    build_doc_tree,
    find_node,
    iter_nodes,
    node_own_text,
    node_text,
)
from factorforge.models import Document


def _doc(docs: list[Document], doc_id: str) -> Document:
    return next(d for d in docs if d.id == doc_id)


def test_load_corpus_parses_frontmatter_and_strips_it() -> None:
    docs = load_corpus()
    assert len(docs) == 6
    value = _doc(docs, "value-premium")
    assert value.title == "The Value Premium"
    assert value.authors == ["Fama", "French"]
    assert value.source == "factorforge:research-notes"
    # Frontmatter must NOT leak into the body text (offsets index into the body only).
    assert "---" not in value.text
    assert "id: value-premium" not in value.text
    assert value.text.lstrip().startswith("# The Value Premium")


def test_doc_tree_mirrors_header_hierarchy() -> None:
    value = _doc(load_corpus(), "value-premium")
    tree = build_doc_tree(value)

    assert tree.root.title == "The Value Premium"
    assert tree.root.level == 1

    h2_titles = [c.title for c in tree.root.children]
    assert h2_titles == [
        "Summary",
        "Definition",
        "Empirical evidence",
        "Regime behavior",
        "Risks and caveats",
        "References",
    ]

    regime = next(c for c in tree.root.children if c.title == "Regime behavior")
    assert [c.title for c in regime.children] == ["High-inflation regimes", "Recessions"]


def test_node_ids_unique_and_spans_valid() -> None:
    value = _doc(load_corpus(), "value-premium")
    tree = build_doc_tree(value)
    nodes = list(iter_nodes(tree.root))

    # The value note has 1 H1 + 6 H2 + 2 H3 = 9 nodes.
    assert len(nodes) == 9
    ids = [n.node_id for n in nodes]
    assert len(set(ids)) == len(ids)  # unique

    for n in nodes:
        assert 0 <= n.start < n.end <= len(value.text)
        # A child's span is contained within its parent's span.
        for c in n.children:
            assert n.start <= c.start and c.end <= n.end


def test_node_text_and_summary_resolve_from_spans() -> None:
    value = _doc(load_corpus(), "value-premium")
    tree = build_doc_tree(value)

    regime = next(c for c in tree.root.children if c.title == "Regime behavior")
    high_infl = find_node(tree.root, regime.children[0].node_id)
    assert high_infl is not None and high_infl.title == "High-inflation regimes"

    text = node_text(value, high_infl)
    assert "inflation" in text.lower()
    assert "High-inflation regimes" in text
    assert high_infl.summary  # non-empty, rule-based gist
    assert "inflation" in high_infl.summary.lower()

    # Own-text of the "Regime behavior" parent excludes its child subtrees.
    own = node_own_text(value, regime)
    assert "High-inflation regimes" not in own


def test_every_corpus_doc_builds_a_tree() -> None:
    for doc in load_corpus():
        tree = build_doc_tree(doc)
        assert tree.root.start == 0
        assert tree.root.end == len(doc.text)
        assert list(iter_nodes(tree.root))  # at least the root
