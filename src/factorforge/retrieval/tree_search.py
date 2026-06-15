"""Hierarchical document navigation (PageIndex-style) — the second vectorless mechanism.

The provider navigates a document's structure tree (titles + summaries, no body text) and returns
the relevant node ids plus its reasoning. We resolve those nodes back to their spans, producing an
evidence path whose steps name the exact sections walked and whose citations are the section spans.
No embeddings — just structure + reasoning.
"""

from __future__ import annotations

from factorforge.corpus.structure import find_node, node_own_text, node_text
from factorforge.models import (
    Citation,
    DocTree,
    Document,
    EvidencePath,
    PathKind,
    PathStep,
)
from factorforge.providers.base import LLMProvider


def tree_evidence(
    doc: Document, tree: DocTree, query: str, provider: LLMProvider, max_nodes: int
) -> EvidencePath | None:
    selection = provider.navigate(query, tree, max_nodes=max_nodes)

    steps: list[PathStep] = []
    citations: list[Citation] = []
    chunks: list[str] = []
    for node_id in selection.node_ids:
        node = find_node(tree.root, node_id)
        if node is None:
            continue
        span_text = node_text(doc, node)
        citations.append(
            Citation(
                doc_id=doc.id, quote=span_text, start=node.start, end=node.end,
                node_id=node_id, section=node.title,
            )
        )
        steps.append(
            PathStep(
                kind=PathKind.TREE, ref=node_id,
                label=f"{doc.id} > {node.title}", detail=selection.thinking,
            )
        )
        chunks.append(f"[{doc.title} > {node.title}] {node_own_text(doc, node).strip()}")

    if not citations:
        return None
    return EvidencePath(
        query=query, kind=PathKind.TREE, steps=steps, citations=citations, text="\n\n".join(chunks)
    )
