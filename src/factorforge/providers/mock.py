"""Deterministic, zero-credential provider.

This is **not** the AI. It is a deterministic stand-in that does genuine rule-based work:
sentence-level vocabulary matching for extraction, and term-overlap scoring for tree navigation.
The output goes through the exact same citation verifier as the real model, so a mock-built graph
is real (every edge is backed by a quote) — only the *judgment* is rule-based rather than learned.
"""

from __future__ import annotations

from factorforge.corpus.structure import iter_nodes
from factorforge.extract.vocab import find_mentions, split_sentences, tokenize
from factorforge.models import (
    DocTree,
    Document,
    EntityType,
    NavSelection,
    RawEntity,
    RawExtraction,
    RawRelation,
    RelationType,
)
from factorforge.providers.base import LLMProvider


class MockProvider(LLMProvider):
    name = "mock"

    def extract(self, doc: Document) -> RawExtraction:
        entities: dict[tuple[EntityType, str], RawEntity] = {}
        relations: dict[tuple[RelationType, str, str], RawRelation] = {}

        for sentence in split_sentences(doc.text):
            mentions = find_mentions(sentence)
            for etype, canon in mentions:
                entities.setdefault(
                    (etype, canon), RawEntity(type=etype, name=canon, quote=sentence)
                )

            factors = [c for e, c in mentions if e == EntityType.FACTOR]
            regimes = [c for e, c in mentions if e == EntityType.REGIME]
            assets = [c for e, c in mentions if e == EntityType.ASSET]

            # regime --affects--> factor
            for regime in regimes:
                for factor in factors:
                    relations.setdefault(
                        (RelationType.AFFECTS, regime, factor),
                        RawRelation(type=RelationType.AFFECTS, source=regime, target=factor, quote=sentence),
                    )
            # factor --affects--> asset
            for factor in factors:
                for asset in assets:
                    relations.setdefault(
                        (RelationType.AFFECTS, factor, asset),
                        RawRelation(type=RelationType.AFFECTS, source=factor, target=asset, quote=sentence),
                    )
            # factor <-> factor (co-mention)
            for i, f1 in enumerate(factors):
                for f2 in factors[i + 1 :]:
                    relations.setdefault(
                        (RelationType.RELATED_TO, f1, f2),
                        RawRelation(type=RelationType.RELATED_TO, source=f1, target=f2, quote=sentence),
                    )

        return RawExtraction(entities=list(entities.values()), relations=list(relations.values()))

    def navigate(self, query: str, tree: DocTree, max_nodes: int = 5) -> NavSelection:
        q_tokens = tokenize(query)
        scored: list[tuple[int, str, str]] = []  # (overlap, node_id, title)
        for node in iter_nodes(tree.root):
            overlap = len(q_tokens & tokenize(f"{node.title} {node.summary}"))
            if overlap > 0:
                scored.append((overlap, node.node_id, node.title))

        scored.sort(key=lambda t: (-t[0], t[1]))
        top = scored[:max_nodes]
        if top:
            thinking = "Selected by term overlap with the query: " + ", ".join(
                f"{title} (score {score})" for score, _, title in top
            )
        else:
            thinking = "No section titles or summaries overlapped the query terms."
        return NavSelection(thinking=thinking, node_ids=[node_id for _, node_id, _ in top])
