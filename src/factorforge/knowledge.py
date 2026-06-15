"""The :class:`KnowledgeBase` — the assembled, queryable knowledge the system reasons over.

Bundles the indexed documents (text + structure trees + verified extractions) with the knowledge
graph built from them. This is the single object retrieval (Phase 4) and the MCP server (Phase 5)
operate against. Built once from a provider; deterministic for the mock provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from factorforge.corpus.loader import load_corpus
from factorforge.extract.entities import IndexedDoc, index_corpus
from factorforge.graph.build import build_graph
from factorforge.graph.store import GraphStore
from factorforge.models import DocTree, Document
from factorforge.providers.base import LLMProvider


@dataclass
class KnowledgeBase:
    indexed: list[IndexedDoc]
    graph: GraphStore
    docs: dict[str, Document]
    trees: dict[str, DocTree]

    def get_doc(self, doc_id: str) -> Document | None:
        return self.docs.get(doc_id)

    def get_tree(self, doc_id: str) -> DocTree | None:
        return self.trees.get(doc_id)


def build_knowledge_base(provider: LLMProvider, corpus_dir: Path | None = None) -> KnowledgeBase:
    """Load the corpus, index every document, and build the knowledge graph."""
    docs = load_corpus(corpus_dir)
    indexed = index_corpus(docs, provider)
    graph = build_graph(indexed)
    return KnowledgeBase(
        indexed=indexed,
        graph=graph,
        docs={item.doc.id: item.doc for item in indexed},
        trees={item.doc.id: item.tree for item in indexed},
    )
