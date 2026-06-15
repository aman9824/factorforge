"""Load the bundled corpus into normalized :class:`Document` objects.

Each document is a small markdown research note with a trivial ``--- key: value ---`` frontmatter
block. We use a hand-rolled frontmatter parser (no YAML dependency): the format is intentionally
tiny, and keeping the dependency surface small matters more than generality here.

``Document.text`` is the **body only** (frontmatter stripped), so every char offset used downstream
— tree spans, citations — indexes into the same canonical string. Newlines are normalized to ``\\n``
so offsets are stable across platforms.
"""

from __future__ import annotations

from pathlib import Path

from src.factorforge.models import Document

# src/factorforge/corpus/loader.py -> parents[3] == project root (projects/factorforge)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS_DIR = _PROJECT_ROOT / "data" / "corpus"


def _split_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Return ``(metadata, body)``. Frontmatter is an optional leading ``---`` fenced block."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    if not text.startswith("---\n"):
        return {}, text.lstrip("\n")

    close = text.find("\n---", 4)
    if close == -1:
        return {}, text.lstrip("\n")

    block = text[4:close]
    after = text[close + 1 :]            # starts at the closing '---' line
    nl = after.find("\n")
    body = after[nl + 1 :] if nl != -1 else ""

    meta: dict[str, str] = {}
    for line in block.split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, body.lstrip("\n")


def parse_document(path: Path) -> Document:
    """Parse a single ``.md`` file into a :class:`Document`."""
    meta, body = _split_frontmatter(path.read_text(encoding="utf-8"))
    authors = [a.strip() for a in meta.get("authors", "").split(";") if a.strip()]
    return Document(
        id=meta.get("id", path.stem),
        title=meta.get("title", path.stem),
        source=meta.get("source", ""),
        authors=authors,
        published=meta.get("published", ""),
        text=body,
    )


def load_corpus(corpus_dir: Path | None = None) -> list[Document]:
    """Load every ``.md`` document from ``corpus_dir`` (default: the bundled corpus), sorted by path."""
    directory = corpus_dir or DEFAULT_CORPUS_DIR
    docs = [parse_document(p) for p in sorted(directory.glob("*.md"))]
    if not docs:
        raise FileNotFoundError(f"No corpus documents found in {directory}")
    return docs
