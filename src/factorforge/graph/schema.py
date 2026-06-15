"""Graph query result types returned by the :class:`~factorforge.graph.store.GraphStore`.

The entity/relation node types live in :mod:`factorforge.models`; these are the *shapes returned
by traversals* — a neighbor (the edge + the node on the other end) and a path (an alternating
sequence of entities and the relations that connect them). Both are JSON-serializable so they can
cross the MCP boundary and become :class:`~factorforge.models.EvidencePath` steps.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from factorforge.models import Entity, Relation


class Neighbor(BaseModel):
    """A relation plus the entity on its far end, as seen from a focal entity."""

    relation: Relation
    entity: Entity
    direction: Literal["out", "in"]


class GraphPath(BaseModel):
    """A traversal path: ``entities[i]`` is linked to ``entities[i+1]`` by ``relations[i]``."""

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
