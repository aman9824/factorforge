"""Structured / metadata filters — the third (non-LLM) retrieval mechanism.

These are plain graph queries used to *scope* retrieval: which entities anchor a query, which
documents mention them, which an author wrote. The retriever uses them to focus the (token-heavy)
tree search on only the documents the graph says are relevant — graph informing tree search.
"""

from __future__ import annotations

from factorforge.knowledge import KnowledgeBase
from factorforge.models import Document, Entity, EntityType, RelationType

# Anchors are *concepts*, not provenance nodes.
_ANCHOR_TYPES = {EntityType.FACTOR, EntityType.REGIME, EntityType.ASSET}


def anchor_entities(kb: KnowledgeBase, query: str, limit: int = 6) -> list[Entity]:
    """The concept entities (factor/regime/asset) a query is about, best first."""
    hits = kb.graph.find_entities(query, limit=limit * 3)
    return [e for e in hits if e.type in _ANCHOR_TYPES][:limit]


def _doc_id_of(entity_id: str) -> str:
    return entity_id.split(":", 1)[1]


def docs_mentioning(kb: KnowledgeBase, entity_id: str) -> list[Document]:
    """Documents whose extraction mentions ``entity_id`` (via the ``mentions`` edge)."""
    out: list[Document] = []
    for nb in kb.graph.neighbors(entity_id, etype=EntityType.DOCUMENT):
        if nb.direction == "in" and nb.relation.type == RelationType.MENTIONS:
            doc = kb.get_doc(_doc_id_of(nb.entity.id))
            if doc is not None:
                out.append(doc)
    return out


def docs_by_author(kb: KnowledgeBase, author: str) -> list[Document]:
    """Documents authored by ``author``."""
    out: list[Document] = []
    for nb in kb.graph.neighbors(f"author:{author}", etype=EntityType.DOCUMENT):
        if nb.direction == "in" and nb.relation.type == RelationType.AUTHORED_BY:
            doc = kb.get_doc(_doc_id_of(nb.entity.id))
            if doc is not None:
                out.append(doc)
    return out


def entities_by_type(kb: KnowledgeBase, etype: EntityType) -> list[Entity]:
    return kb.graph.entities(etype)
