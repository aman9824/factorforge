"""Knowledge-graph MCP server: tool logic + a real round-trip through the FastMCP API."""

from __future__ import annotations

import asyncio
import json

from factorforge.config import get_settings
from factorforge.knowledge import KnowledgeBase, build_knowledge_base
from factorforge.mcp_server.server import (
    build_server,
    resource_doc_outline,
    resource_kg_snapshot,
    tool_get_evidence,
    tool_get_neighbors,
    tool_get_section,
    tool_graph_search,
    tool_list_factors,
    tool_navigate_document,
)
from factorforge.providers.mock import MockProvider


def _kb() -> KnowledgeBase:
    return build_knowledge_base(MockProvider())


def test_graph_search_and_neighbors_tools() -> None:
    kb = _kb()
    hits = json.loads(tool_graph_search(kb, "value factor"))
    assert any(h["id"] == "factor:value" for h in hits)

    nbrs = json.loads(tool_get_neighbors(kb, "factor:value"))
    assert any(n["neighbor_id"] == "regime:high_inflation" for n in nbrs)
    # The affects edge carries a citation through the MCP boundary.
    affects = [n for n in nbrs if n["neighbor_id"] == "regime:high_inflation"]
    assert affects[0]["citation"] is not None


def test_list_factors_tool() -> None:
    factors = json.loads(tool_list_factors(_kb()))
    names = {f["name"] for f in factors}
    assert {"value", "momentum", "size", "quality", "low_volatility"} <= names


def test_navigate_and_section_tools() -> None:
    kb = _kb()
    nav = json.loads(tool_navigate_document(kb, MockProvider(), get_settings(), "value-premium",
                                            "inflation regime"))
    assert nav["citations"], "navigation should select at least one section"
    node_id = nav["citations"][0]["node_id"]

    section = json.loads(tool_get_section(kb, "value-premium", node_id))
    assert section["text"] and section["doc_id"] == "value-premium"

    # Unknown ids fail gracefully.
    assert "error" in json.loads(tool_get_section(kb, "value-premium", "9999"))
    assert "error" in json.loads(tool_navigate_document(kb, MockProvider(), get_settings(), "nope", "x"))


def test_get_evidence_tool_returns_cited_paths() -> None:
    kb = _kb()
    paths = json.loads(tool_get_evidence(kb, MockProvider(), get_settings(),
                                         "Does the value premium depend on the inflation regime?"))
    assert paths
    assert any(c["section"] for p in paths for c in p["citations"])


def test_resources() -> None:
    kb = _kb()
    snap = json.loads(resource_kg_snapshot(kb))
    assert snap["entities"] and snap["relations"]

    outline = resource_doc_outline(kb, "value-premium")
    assert "Regime behavior" in outline and "[0001]" in outline


def test_fastmcp_registration_and_call_tool_roundtrip() -> None:
    server = build_server(kb=_kb(), settings=get_settings())

    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert {"graph_search", "get_neighbors", "list_factors",
            "navigate_document", "get_evidence", "get_section"} <= names

    # Round-trip an actual tool call through the MCP layer.
    result = asyncio.run(server.call_tool("list_factors", {}))
    assert "factor:value" in str(result)
