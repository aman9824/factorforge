"""The standalone knowledge-graph MCP server (FastMCP).

This is where the JD's "MCP servers" technique lives. It exposes the FactorForge knowledge graph
and the vectorless retriever to *any* MCP client (the Claude Agent SDK, Claude Desktop, Claude
Code) as tools + resources. It runs as its own process — over stdio (the agent spawns it as a
subprocess) or streamable-HTTP (``make serve-mcp``).

Tool *logic* lives in module-level functions so it is unit-testable without the MCP machinery;
``build_server`` registers thin ``@mcp.tool()`` wrappers around them.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.factorforge.config import Settings, get_settings
from src.factorforge.corpus.structure import find_node, node_text, render_outline
from src.factorforge.factory import build_provider
from src.factorforge.knowledge import KnowledgeBase, build_knowledge_base
from src.factorforge.models import EntityType, EvidencePath
from src.factorforge.providers.base import LLMProvider
from src.factorforge.retrieval.retriever import retrieve
from src.factorforge.retrieval.tree_search import tree_evidence

_QUOTE_CAP = 320  # cap quoted text so evidence stays token-cheap for the agent


def _compact_path(path: EvidencePath) -> dict[str, Any]:
    """A token-lean view of an evidence path for tool results."""
    return {
        "kind": path.kind.value,
        "score": path.score,
        "steps": [s.label for s in path.steps],
        "citations": [
            {
                "doc_id": c.doc_id,
                "section": c.section,
                "node_id": c.node_id,
                "start": c.start,
                "end": c.end,
                "quote": (c.quote[:_QUOTE_CAP] + "...") if len(c.quote) > _QUOTE_CAP else c.quote,
            }
            for c in path.citations
        ],
    }


# ── tool logic (unit-testable, MCP-agnostic) ─────────────────────────────────

def tool_graph_search(kb: KnowledgeBase, query: str, limit: int = 8) -> str:
    ents = kb.graph.find_entities(query, limit=limit)
    return json.dumps([{"id": e.id, "type": e.type.value, "name": e.name} for e in ents], indent=2)


def tool_get_neighbors(kb: KnowledgeBase, entity_id: str) -> str:
    nbrs = kb.graph.neighbors(entity_id)
    return json.dumps(
        [
            {
                "relation": n.relation.type.value,
                "direction": n.direction,
                "neighbor_id": n.entity.id,
                "neighbor_name": n.entity.name,
                "citation": (
                    {"doc_id": n.relation.citation.doc_id, "section": n.relation.citation.section,
                     "quote": n.relation.citation.quote[:_QUOTE_CAP]}
                    if n.relation.citation else None
                ),
            }
            for n in nbrs
        ],
        indent=2,
    )


def tool_list_factors(kb: KnowledgeBase) -> str:
    factors = kb.graph.entities(EntityType.FACTOR)
    return json.dumps([{"id": e.id, "name": e.name} for e in factors], indent=2)


def tool_navigate_document(
    kb: KnowledgeBase, provider: LLMProvider, settings: Settings, doc_id: str, query: str
) -> str:
    doc, tree = kb.get_doc(doc_id), kb.get_tree(doc_id)
    if doc is None or tree is None:
        return json.dumps({"error": f"unknown doc_id: {doc_id}"})
    path = tree_evidence(doc, tree, query, provider, settings.max_nav_nodes)
    if path is None:
        return json.dumps({"thinking": "no relevant sections found", "citations": []})
    return json.dumps(_compact_path(path), indent=2)


def tool_get_evidence(
    kb: KnowledgeBase, provider: LLMProvider, settings: Settings, query: str
) -> str:
    result = retrieve(kb, query, provider, settings)
    return json.dumps([_compact_path(p) for p in result.paths], indent=2)


def tool_get_section(kb: KnowledgeBase, doc_id: str, node_id: str) -> str:
    doc, tree = kb.get_doc(doc_id), kb.get_tree(doc_id)
    if doc is None or tree is None:
        return json.dumps({"error": f"unknown doc_id: {doc_id}"})
    node = find_node(tree.root, node_id)
    if node is None:
        return json.dumps({"error": f"unknown node_id: {node_id}"})
    return json.dumps(
        {"doc_id": doc_id, "node_id": node_id, "title": node.title, "text": node_text(doc, node)}
    )


def resource_kg_snapshot(kb: KnowledgeBase) -> str:
    return json.dumps(kb.graph.to_json())


def resource_doc_outline(kb: KnowledgeBase, doc_id: str) -> str:
    tree = kb.get_tree(doc_id)
    if tree is None:
        return json.dumps({"error": f"unknown doc_id: {doc_id}"})
    return render_outline(tree)


# ── server assembly ──────────────────────────────────────────────────────────

def build_server(kb: KnowledgeBase | None = None, settings: Settings | None = None) -> FastMCP:
    settings = settings or get_settings()
    provider = build_provider(settings)
    kb = kb or build_knowledge_base(provider)

    mcp = FastMCP("factorforge-kg", host=settings.mcp_http_host, port=settings.mcp_http_port)

    @mcp.tool()
    def graph_search(query: str, limit: int = 8) -> str:
        """Search the knowledge graph for entities (factors, regimes, assets, authors, documents)
        matching a query. Returns each entity's id, type, and name."""
        return tool_graph_search(kb, query, limit)

    @mcp.tool()
    def get_neighbors(entity_id: str) -> str:
        """Return an entity's graph neighbors with the relation type, direction, the neighbor, and
        the span citation that backs each edge."""
        return tool_get_neighbors(kb, entity_id)

    @mcp.tool()
    def list_factors() -> str:
        """List every factor entity known to the knowledge graph."""
        return tool_list_factors(kb)

    @mcp.tool()
    def navigate_document(doc_id: str, query: str) -> str:
        """Hierarchically navigate a document's structure tree (PageIndex-style) to find the
        sections relevant to a query. Returns the chosen sections with span citations."""
        return tool_navigate_document(kb, provider, settings, doc_id, query)

    @mcp.tool()
    def get_evidence(query: str) -> str:
        """Run vectorless retrieval (graph traversal + tree search + filters) for a query and
        return ranked evidence paths, each a traceable step trace with span citations. This is the
        primary research tool."""
        return tool_get_evidence(kb, provider, settings, query)

    @mcp.tool()
    def get_section(doc_id: str, node_id: str) -> str:
        """Return the full text of one document section (tree node), e.g. after navigate_document."""
        return tool_get_section(kb, doc_id, node_id)

    @mcp.resource("kg://snapshot")
    def kg_snapshot() -> str:
        """A JSON snapshot of the entire knowledge graph (entities + relations)."""
        return resource_kg_snapshot(kb)

    @mcp.resource("doc://{doc_id}")
    def doc_outline(doc_id: str) -> str:
        """The hierarchical outline (table of contents) of a document."""
        return resource_doc_outline(kb, doc_id)

    return mcp
