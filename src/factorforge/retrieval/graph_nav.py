"""Graph-traversal retrieval — turn graph edges into traceable, cited evidence paths.

Two kinds of evidence come out of the graph:

1. **anchor edges** — the cited semantic relations (``affects`` / ``related_to`` / ``concerns``)
   incident to the query's anchor entities;
2. **a connecting path** — the shortest path between a factor anchor and a regime anchor, which
   answers "how is X connected to Y?" as a readable multi-hop chain.

Structural edges (``mentions`` / ``authored_by``) are excluded here — document content is the job
of the tree search.
"""

from __future__ import annotations

from factorforge.knowledge import KnowledgeBase
from factorforge.models import (
    Entity,
    EntityType,
    EvidencePath,
    PathKind,
    PathStep,
    Relation,
    RelationType,
)

_SEMANTIC = {RelationType.AFFECTS, RelationType.RELATED_TO, RelationType.CONCERNS, RelationType.CONTRADICTS}


def _edge_label(rel: Relation) -> str:
    return f"{rel.source} --{rel.type.value}--> {rel.target}"


def _step(rel: Relation) -> PathStep:
    return PathStep(
        kind=PathKind.GRAPH,
        ref=f"{rel.source}->{rel.target}",
        label=_edge_label(rel),
        detail=(rel.citation.section or "") if rel.citation else "",
    )


def graph_evidence(
    kb: KnowledgeBase, query: str, anchors: list[Entity], max_paths: int
) -> list[EvidencePath]:
    g = kb.graph
    paths: list[EvidencePath] = []
    seen: set[tuple[str, str, str]] = set()

    # 1) A connecting path between a factor anchor and a regime anchor (the "does X depend on Y").
    factor = next((a for a in anchors if a.type == EntityType.FACTOR), None)
    regime = next((a for a in anchors if a.type == EntityType.REGIME), None)
    if factor is not None and regime is not None:
        gp = g.shortest_path(regime.id, factor.id)
        if gp is not None and gp.relations:
            cits = [r.citation for r in gp.relations if r.citation is not None]
            text = " | ".join(c.quote for c in cits) or " -> ".join(e.name for e in gp.entities)
            for r in gp.relations:
                seen.add((r.type.value, r.source, r.target))  # don't re-emit these as anchor edges
            paths.append(
                EvidencePath(
                    query=query, kind=PathKind.GRAPH,
                    steps=[_step(r) for r in gp.relations], citations=cits, text=text,
                )
            )

    # 2) Cited semantic edges incident to each anchor (excluding any already on the path above).
    for anchor in anchors:
        for nb in g.neighbors(anchor.id):
            rel = nb.relation
            if rel.type not in _SEMANTIC or rel.citation is None:
                continue
            key = (rel.type.value, rel.source, rel.target)
            if key in seen:
                continue
            seen.add(key)
            paths.append(
                EvidencePath(
                    query=query, kind=PathKind.GRAPH,
                    steps=[_step(rel)], citations=[rel.citation], text=rel.citation.quote,
                )
            )

    return paths[:max_paths]
