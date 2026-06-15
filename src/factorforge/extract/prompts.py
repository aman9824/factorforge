"""Provider-agnostic prompts + tool schemas for the real (Vertex) path.

These are plain strings/dicts (no SDK types) so the prompt layer stays decoupled from any
particular client. The Vertex provider passes the tool schemas to ``messages.create`` and forces
the corresponding tool call, guaranteeing schema-valid JSON back.
"""

from __future__ import annotations

from typing import Any

from src.factorforge.models import NavSelection, RawExtraction

# ── Extraction ───────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = (
    "You are a meticulous quant-finance knowledge engineer. Extract entities and relations from a "
    "research note about investment factors. Entity types: factor, asset, regime, claim, author. "
    "Relation types: affects (factor/regime -> asset/factor), related_to, concerns, proposes, "
    "evidenced_by, authored_by, mentions. For EVERY entity and relation you MUST include a short "
    "verbatim `quote` copied exactly from the document that supports it. Do not invent facts; if "
    "there is no supporting sentence, do not emit the item. Prefer canonical lowercase names with "
    "underscores (e.g. 'value', 'high_inflation', 'small_cap')."
)

EXTRACTION_TOOL: dict[str, Any] = {
    "name": "record_extraction",
    "description": "Record the entities and relations extracted from the document.",
    "input_schema": RawExtraction.model_json_schema(),
}


def build_extraction_prompt(doc_title: str, doc_text: str) -> str:
    return (
        f"Document title: {doc_title}\n\n"
        f"Document text:\n{doc_text}\n\n"
        "Extract the factor/asset/regime/author entities and the relations among them. "
        "Every item needs a verbatim supporting quote from the text above."
    )


# ── Navigation (hierarchical tree search) ────────────────────────────────────

NAVIGATION_SYSTEM = (
    "You navigate a document's table-of-contents tree to answer a query. You are given the tree as "
    "an outline of [node_id] Title — summary lines. Think about which sections most likely contain "
    "the answer, then return the relevant node_ids. Select only nodes that are genuinely relevant; "
    "fewer is better. Always explain your reasoning in `thinking`."
)

NAVIGATION_TOOL: dict[str, Any] = {
    "name": "select_nodes",
    "description": "Select the relevant tree node ids for the query, with reasoning.",
    "input_schema": NavSelection.model_json_schema(),
}


def build_navigation_prompt(query: str, outline: str, max_nodes: int) -> str:
    return (
        f"Query: {query}\n\n"
        f"Document outline (node_id, title, summary):\n{outline}\n\n"
        f"Return at most {max_nodes} relevant node_ids that should be read to answer the query."
    )
