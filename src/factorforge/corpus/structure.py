"""Build a hierarchical structure tree (PageIndex-style) for a document — **no embeddings**.

The tree mirrors the document's own markdown header hierarchy. Each node carries a title, a
char-offset span into the document body, a short rule-based summary (the leading text of the
section), and nested children. Retrieval later navigates this tree by title + summary and resolves
the chosen nodes back to their spans — a fully traceable, explainable path rather than an opaque
similarity score.

The structure here is 100% deterministic (derived from headers); only the *navigation judgment*
(which nodes are relevant to a query) is delegated to the LLM provider seam, and even that has a
deterministic mock.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from src.factorforge.models import DocNode, DocTree, Document

_HEADER = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_WS = re.compile(r"\s+")
_SENT = re.compile(r"(?<=[.!?])\s+")


def _summary(text: str, max_chars: int = 220) -> str:
    """A short, deterministic gist of a section: its first sentence (capped)."""
    collapsed = _WS.sub(" ", text).strip()
    if not collapsed:
        return ""
    first = _SENT.split(collapsed, maxsplit=1)[0]
    return first if len(first) <= max_chars else collapsed[:max_chars].rstrip()


def build_doc_tree(doc: Document) -> DocTree:
    """Construct the structure tree for ``doc`` from its markdown headers."""
    body = doc.text
    matches = list(_HEADER.finditer(body))

    if not matches:
        root = DocNode(
            node_id="0000", title=doc.title, summary=_summary(body),
            start=0, end=len(body), level=0,
        )
        return DocTree(doc_id=doc.id, title=doc.title, root=root)

    # (level, title, header_start, content_start) for each header in document order.
    heads = [(len(m.group(1)), m.group(2).strip(), m.start(), m.end()) for m in matches]
    n = len(heads)

    # A node's span ends at the next header whose level is <= its own (a sibling/ancestor break),
    # so the span covers the header plus its content plus all descendants.
    ends = [len(body)] * n
    for i in range(n):
        for j in range(i + 1, n):
            if heads[j][0] <= heads[i][0]:
                ends[i] = heads[j][2]
                break

    flat: list[DocNode] = []
    for i, (level, title, hstart, hcontent) in enumerate(heads):
        # "Own text" is the content between this header and the next header overall (i.e. excluding
        # descendant sections), used only to derive the summary.
        own_end = heads[i + 1][2] if i + 1 < n and heads[i + 1][2] < ends[i] else ends[i]
        flat.append(
            DocNode(
                node_id=str(i).zfill(4), title=title, summary=_summary(body[hcontent:own_end]),
                start=hstart, end=ends[i], level=level, children=[],
            )
        )

    # Nest via a stack: pop until the top is a strict ancestor (lower level), then attach.
    roots: list[DocNode] = []
    stack: list[DocNode] = []
    for node in flat:
        while stack and stack[-1].level >= node.level:
            stack.pop()
        (stack[-1].children if stack else roots).append(node)
        stack.append(node)

    if len(roots) == 1:
        root = roots[0]
    else:
        # Multiple top-level headers: wrap them under a synthetic document root.
        root = DocNode(
            node_id="root", title=doc.title, summary=doc.title,
            start=0, end=len(body), level=0, children=roots,
        )
    return DocTree(doc_id=doc.id, title=doc.title, root=root)


def iter_nodes(node: DocNode) -> Iterator[DocNode]:
    """Pre-order traversal over a node and all its descendants."""
    yield node
    for child in node.children:
        yield from iter_nodes(child)


def find_node(root: DocNode, node_id: str) -> DocNode | None:
    """Return the node with ``node_id`` in the tree rooted at ``root``, or ``None``."""
    for node in iter_nodes(root):
        if node.node_id == node_id:
            return node
    return None


def node_text(doc: Document, node: DocNode) -> str:
    """The full text of ``node`` (header + content + descendants)."""
    return doc.text[node.start : node.end]


def node_own_text(doc: Document, node: DocNode) -> str:
    """The text of ``node`` excluding its child subtrees (header + its direct content)."""
    if not node.children:
        return doc.text[node.start : node.end]
    return doc.text[node.start : node.children[0].start]


def node_for_offset(root: DocNode, offset: int) -> DocNode | None:
    """The deepest node whose span contains ``offset`` (used to attribute a citation to a section)."""
    best: DocNode | None = None
    for node in iter_nodes(root):
        if node.start <= offset < node.end and (best is None or node.level >= best.level):
            best = node
    return best


def render_outline(tree: DocTree) -> str:
    """Render a token-cheap text outline (id · title · summary) for LLM navigation — no body text.

    This is the PageIndex principle: send the model the *structure*, not the content, and let it
    pick the relevant node ids to expand.
    """
    lines: list[str] = []
    for node in iter_nodes(tree.root):
        indent = "  " * node.level
        summary = f" - {node.summary}" if node.summary else ""
        lines.append(f"{indent}[{node.node_id}] {node.title}{summary}")
    return "\n".join(lines)
