"""Citation verification — the trust core.

Every entity and relation an LLM (or the mock) *claims* carries a supporting ``quote``. Here we
independently re-check that quote against the canonical document text. A claim is promoted to a
graph entity/relation only if its quote resolves to a real ``(start, end)`` span; anything else is
rejected and recorded for transparency. This turns "the model said it cited a source" into "we
proved it did" — exactly P1's stance, applied to a knowledge graph.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from factorforge.corpus.structure import node_for_offset
from factorforge.models import (
    Citation,
    DocTree,
    Document,
    Entity,
    EntityType,
    RawExtraction,
    Relation,
)


class RejectedItem(BaseModel):
    kind: str          # "entity" | "relation"
    ref: str           # entity id or "source -type-> target"
    quote: str
    reason: str


class VerifiedExtraction(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    rejected: list[RejectedItem] = Field(default_factory=list)


def _find_span(source: str, quote: str) -> tuple[int, int] | None:
    """Resolve ``quote`` to an exact ``(start, end)`` span in ``source``, or ``None``."""
    q = quote.strip()
    # Tool results truncate long quotes with a trailing ellipsis; the remainder is still a verbatim
    # prefix of a real span, so drop the ellipsis before matching (paraphrases still won't match).
    for suffix in ("...", "…"):
        if q.endswith(suffix):
            q = q[: -len(suffix)].strip()
    if not q:
        return None
    pos = source.find(q)
    if pos != -1:
        return pos, pos + len(q)
    # Whitespace-tolerant: the model may have re-flowed spaces/newlines (e.g. the mock collapses
    # wrapped lines into single-space sentences). Match any run of whitespace between tokens.
    tokens = [re.escape(t) for t in q.split()]
    if tokens:
        m = re.search(r"\s+".join(tokens), source)
        if m:
            return m.start(), m.end()
    return None


def _citation(doc: Document, tree: DocTree, quote: str) -> Citation | None:
    span = _find_span(doc.text, quote)
    if span is None:
        return None
    start, end = span
    node = node_for_offset(tree.root, start)
    return Citation(
        doc_id=doc.id,
        quote=doc.text[start:end],
        start=start,
        end=end,
        node_id=node.node_id if node else None,
        section=node.title if node else None,
    )


def resolve_citation(doc: Document, tree: DocTree, quote: str) -> Citation | None:
    """Public: verify an (untrusted) quote against a document, returning a proven span or None.

    Used to turn an agent's claimed quote into a real citation (verify, don't trust).
    """
    return _citation(doc, tree, quote)


def verify_extraction(raw: RawExtraction, doc: Document, tree: DocTree) -> VerifiedExtraction:
    """Promote only quote-verified entities/relations; record the rest as rejected."""
    out = VerifiedExtraction()

    name_type: dict[str, EntityType] = {}
    for ent in raw.entities:
        name_type.setdefault(ent.name, ent.type)

    by_id: dict[str, Entity] = {}
    for raw_ent in raw.entities:
        eid = f"{raw_ent.type.value}:{raw_ent.name}"
        citation = _citation(doc, tree, raw_ent.quote) if raw_ent.quote else None
        if raw_ent.quote and citation is None:
            out.rejected.append(
                RejectedItem(kind="entity", ref=eid, quote=raw_ent.quote, reason="quote not found in source")
            )
            continue
        entity = by_id.get(eid)
        if entity is None:
            entity = Entity(id=eid, type=raw_ent.type, name=raw_ent.name, attributes=raw_ent.attributes)
            by_id[eid] = entity
            out.entities.append(entity)
        if citation is not None:
            entity.citations.append(citation)

    for rel in raw.relations:
        src_type = name_type.get(rel.source)
        tgt_type = name_type.get(rel.target)
        ref = f"{rel.source} -{rel.type.value}-> {rel.target}"
        if src_type is None or tgt_type is None:
            out.rejected.append(
                RejectedItem(kind="relation", ref=ref, quote=rel.quote, reason="unknown relation endpoint")
            )
            continue
        citation = _citation(doc, tree, rel.quote) if rel.quote else None
        if rel.quote and citation is None:
            out.rejected.append(
                RejectedItem(kind="relation", ref=ref, quote=rel.quote, reason="quote not found in source")
            )
            continue
        out.relations.append(
            Relation(
                type=rel.type,
                source=f"{src_type.value}:{rel.source}",
                target=f"{tgt_type.value}:{rel.target}",
                citation=citation,
                attributes=rel.attributes,
            )
        )

    return out
