"""Assemble the knowledge graph from per-document extractions.

Combines two kinds of knowledge:

* **structural** — each document node, its authors (``authored_by``), and what it ``mentions``;
* **extracted** — the quote-verified factor/regime/asset entities and their ``affects`` /
  ``related_to`` relations.

The result is one graph where, e.g., ``document:value-premium`` ``mentions`` ``factor:value``,
which is ``affects``-linked from ``regime:high_inflation`` — every extracted edge carrying a span
citation.
"""

from __future__ import annotations

from factorforge.extract.entities import IndexedDoc
from factorforge.graph.store import GraphStore, NetworkxGraphStore
from factorforge.models import Entity, EntityType, Relation, RelationType


def build_graph(indexed: list[IndexedDoc], store: GraphStore | None = None) -> GraphStore:
    store = store or NetworkxGraphStore()

    for item in indexed:
        doc = item.doc
        doc_id = f"document:{doc.id}"
        store.add_entity(
            Entity(
                id=doc_id,
                type=EntityType.DOCUMENT,
                name=doc.title,
                attributes={"source": doc.source, "published": doc.published},
            )
        )

        for author in doc.authors:
            author_id = f"author:{author}"
            store.add_entity(Entity(id=author_id, type=EntityType.AUTHOR, name=author))
            store.add_relation(
                Relation(type=RelationType.AUTHORED_BY, source=doc_id, target=author_id)
            )

        for entity in item.extraction.entities:
            store.add_entity(entity)
            store.add_relation(
                Relation(
                    type=RelationType.MENTIONS,
                    source=doc_id,
                    target=entity.id,
                    citation=entity.citations[0] if entity.citations else None,
                )
            )

        for relation in item.extraction.relations:
            store.add_relation(relation)

    return store
