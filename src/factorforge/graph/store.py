"""The knowledge-graph store — a swappable interface over an in-process ``networkx`` graph.

Why a graph (and not a vector index): retrieval becomes *traversal*, and a result is a readable
path of typed edges (``factor:value <-affects- regime:high_inflation``) with a citation on each
hop — explainable and auditable, which is the whole point of FactorForge.

``GraphStore`` is an ABC so the backing store can later be swapped (e.g. Neo4j) without touching
retrieval or the MCP server. The bundled implementation uses ``networkx.MultiDiGraph`` — zero
infra, deterministic, JSON-serializable — which fits the offline/CI constraints.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import networkx as nx

from factorforge.extract.vocab import tokenize
from factorforge.graph.schema import GraphPath, Neighbor
from factorforge.models import Entity, EntityType, Relation


def _parse_id(entity_id: str) -> tuple[EntityType | None, str]:
    etype_str, _, name = entity_id.partition(":")
    try:
        return EntityType(etype_str), name
    except ValueError:
        return None, entity_id


class GraphStore(ABC):
    """Interface for the knowledge graph."""

    @abstractmethod
    def add_entity(self, entity: Entity) -> None: ...

    @abstractmethod
    def add_relation(self, relation: Relation) -> None: ...

    @abstractmethod
    def get_entity(self, entity_id: str) -> Entity | None: ...

    @abstractmethod
    def has_entity(self, entity_id: str) -> bool: ...

    @abstractmethod
    def entities(self, etype: EntityType | None = None) -> list[Entity]: ...

    @abstractmethod
    def relations(self) -> list[Relation]: ...

    @abstractmethod
    def neighbors(self, entity_id: str, etype: EntityType | None = None) -> list[Neighbor]: ...

    @abstractmethod
    def find_entities(self, query: str, etype: EntityType | None = None, limit: int = 10) -> list[Entity]: ...

    @abstractmethod
    def shortest_path(self, source_id: str, target_id: str) -> GraphPath | None: ...

    @abstractmethod
    def to_json(self) -> dict[str, list[dict[str, Any]]]: ...

    def stats(self) -> dict[str, int]:
        return {"entities": len(self.entities()), "relations": len(self.relations())}


class NetworkxGraphStore(GraphStore):
    def __init__(self) -> None:
        self._g = nx.MultiDiGraph()

    # ── writes ────────────────────────────────────────────────────────────────
    def add_entity(self, entity: Entity) -> None:
        existing = self.get_entity(entity.id)
        if existing is not None:
            # Merge: an entity is mentioned across many docs; accumulate its citations.
            seen = {(c.doc_id, c.start, c.end) for c in existing.citations}
            for cit in entity.citations:
                if (cit.doc_id, cit.start, cit.end) not in seen:
                    existing.citations.append(cit)
                    seen.add((cit.doc_id, cit.start, cit.end))
            existing.attributes.update(entity.attributes)
            self._g.nodes[entity.id]["data"] = existing.model_dump()
            return
        self._g.add_node(entity.id, data=entity.model_dump())

    def _ensure_node(self, entity_id: str) -> None:
        if entity_id not in self._g:
            etype, name = _parse_id(entity_id)
            placeholder = Entity(id=entity_id, type=etype or EntityType.CLAIM, name=name)
            self._g.add_node(entity_id, data=placeholder.model_dump())

    def add_relation(self, relation: Relation) -> None:
        self._ensure_node(relation.source)
        self._ensure_node(relation.target)
        self._g.add_edge(
            relation.source, relation.target, key=relation.type.value, data=relation.model_dump()
        )

    # ── reads ─────────────────────────────────────────────────────────────────
    def get_entity(self, entity_id: str) -> Entity | None:
        if entity_id not in self._g:
            return None
        return Entity.model_validate(self._g.nodes[entity_id]["data"])

    def has_entity(self, entity_id: str) -> bool:
        return bool(entity_id in self._g)

    def entities(self, etype: EntityType | None = None) -> list[Entity]:
        out: list[Entity] = []
        for _, data in self._g.nodes(data=True):
            ent = Entity.model_validate(data["data"])
            if etype is None or ent.type == etype:
                out.append(ent)
        return out

    def relations(self) -> list[Relation]:
        return [Relation.model_validate(data["data"]) for _, _, data in self._g.edges(data=True)]

    def neighbors(self, entity_id: str, etype: EntityType | None = None) -> list[Neighbor]:
        if entity_id not in self._g:
            return []
        out: list[Neighbor] = []
        for _, tgt, data in self._g.out_edges(entity_id, data=True):
            ent = self.get_entity(tgt)
            if ent is not None and (etype is None or ent.type == etype):
                out.append(Neighbor(relation=Relation.model_validate(data["data"]), entity=ent, direction="out"))
        for src, _, data in self._g.in_edges(entity_id, data=True):
            ent = self.get_entity(src)
            if ent is not None and (etype is None or ent.type == etype):
                out.append(Neighbor(relation=Relation.model_validate(data["data"]), entity=ent, direction="in"))
        return out

    def find_entities(self, query: str, etype: EntityType | None = None, limit: int = 10) -> list[Entity]:
        q_tokens = tokenize(query)
        needle = query.strip().lower()
        scored: list[tuple[int, Entity]] = []
        for ent in self.entities(etype):
            score = len(q_tokens & tokenize(ent.name))
            if needle and needle in (ent.id.lower(), ent.name.lower()):
                score += 100  # exact id/name match wins
            if score > 0:
                scored.append((score, ent))
        scored.sort(key=lambda t: (-t[0], t[1].id))
        return [ent for _, ent in scored[:limit]]

    def shortest_path(self, source_id: str, target_id: str) -> GraphPath | None:
        if source_id not in self._g or target_id not in self._g:
            return None
        undirected = self._g.to_undirected(as_view=True)
        try:
            node_ids = nx.shortest_path(undirected, source_id, target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        entities = [e for e in (self.get_entity(n) for n in node_ids) if e is not None]
        relations = [
            r
            for r in (self._edge_between(a, b) for a, b in zip(node_ids, node_ids[1:], strict=False))
            if r is not None
        ]
        return GraphPath(entities=entities, relations=relations)

    def _edge_between(self, a: str, b: str) -> Relation | None:
        for src, dst in ((a, b), (b, a)):
            if self._g.has_edge(src, dst):
                data = next(iter(self._g.get_edge_data(src, dst).values()))
                return Relation.model_validate(data["data"])
        return None

    # ── (de)serialization ──────────────────────────────────────────────────────
    def to_json(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "entities": [e.model_dump() for e in self.entities()],
            "relations": [r.model_dump() for r in self.relations()],
        }

    @classmethod
    def from_json(cls, data: dict[str, list[dict[str, Any]]]) -> NetworkxGraphStore:
        store = cls()
        for ent in data.get("entities", []):
            store.add_entity(Entity.model_validate(ent))
        for rel in data.get("relations", []):
            store.add_relation(Relation.model_validate(rel))
        return store
