"""Knowledge graph construction, traversal, and JSON round-trip."""

from __future__ import annotations

from factorforge.graph.store import NetworkxGraphStore
from factorforge.knowledge import build_knowledge_base
from factorforge.models import EntityType, RelationType
from factorforge.providers.mock import MockProvider


def _kb():  # type: ignore[no-untyped-def]
    return build_knowledge_base(MockProvider())


def test_graph_has_structural_and_extracted_nodes() -> None:
    g = _kb().graph
    stats = g.stats()
    assert stats["entities"] > 10 and stats["relations"] > 10

    for eid in ("factor:value", "regime:high_inflation", "asset:equities",
                "document:value-premium", "author:Fama"):
        assert g.has_entity(eid), f"missing {eid}"

    # Author and factor partitions are populated.
    assert {e.id for e in g.entities(EntityType.FACTOR)} >= {
        "factor:value", "factor:momentum", "factor:size", "factor:quality", "factor:low_volatility"
    }


def test_authored_by_and_mentions_edges() -> None:
    g = _kb().graph
    doc_neighbors = g.neighbors("document:value-premium")
    rels = {(n.relation.type, n.entity.id) for n in doc_neighbors}
    assert (RelationType.AUTHORED_BY, "author:Fama") in rels
    assert (RelationType.MENTIONS, "factor:value") in rels


def test_regime_affects_factor_is_traversable_and_cited() -> None:
    g = _kb().graph
    # From the value factor, the high-inflation regime is an *incoming* affects edge.
    nbrs = g.neighbors("factor:value")
    affects_in = [
        n for n in nbrs
        if n.relation.type == RelationType.AFFECTS and n.direction == "in"
        and n.entity.id == "regime:high_inflation"
    ]
    assert affects_in, "expected regime:high_inflation --affects--> factor:value"
    assert affects_in[0].relation.citation is not None  # the edge is evidence-backed


def test_shortest_path_between_regime_and_factor() -> None:
    g = _kb().graph
    path = g.shortest_path("regime:high_inflation", "factor:value")
    assert path is not None
    ids = [e.id for e in path.entities]
    assert ids[0] == "regime:high_inflation" and ids[-1] == "factor:value"
    assert len(path.relations) == len(path.entities) - 1


def test_find_entities_by_terms() -> None:
    g = _kb().graph
    hits = g.find_entities("inflation regime")
    assert any(e.id == "regime:high_inflation" for e in hits)

    momentum = g.find_entities("momentum", etype=EntityType.FACTOR)
    assert momentum and momentum[0].id == "factor:momentum"


def test_graph_json_round_trips() -> None:
    g = _kb().graph
    snapshot = g.to_json()
    rebuilt = NetworkxGraphStore.from_json(snapshot)
    assert rebuilt.stats() == g.stats()
    original = g.get_entity("factor:value")
    restored = rebuilt.get_entity("factor:value")
    assert original is not None and restored is not None
    assert restored.model_dump() == original.model_dump()
