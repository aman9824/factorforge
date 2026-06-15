"""Orchestrate per-document indexing: structure tree + verified extraction.

An :class:`IndexedDoc` bundles everything the rest of the system needs about one document — the
normalized text, its hierarchical tree, and its quote-verified entities/relations. The knowledge
graph (Phase 3) is assembled from a list of these; retrieval (Phase 4) navigates their trees.
"""

from __future__ import annotations

from dataclasses import dataclass

from factorforge.citations.verifier import VerifiedExtraction, verify_extraction
from factorforge.corpus.structure import build_doc_tree
from factorforge.models import DocTree, Document
from factorforge.providers.base import LLMProvider


@dataclass(frozen=True)
class IndexedDoc:
    doc: Document
    tree: DocTree
    extraction: VerifiedExtraction


def index_document(doc: Document, provider: LLMProvider) -> IndexedDoc:
    tree = build_doc_tree(doc)
    raw = provider.extract(doc)
    return IndexedDoc(doc=doc, tree=tree, extraction=verify_extraction(raw, doc, tree))


def index_corpus(docs: list[Document], provider: LLMProvider) -> list[IndexedDoc]:
    return [index_document(doc, provider) for doc in docs]
