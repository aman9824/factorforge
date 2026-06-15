"""Controlled vocabulary + lightweight text utilities for the deterministic mock path.

The corpus is curated, so a small controlled vocabulary of factor / regime / asset terms lets the
**mock** provider do genuine, deterministic entity & relation extraction (and navigation ranking)
with zero model calls. The real ``vertex`` provider does open-ended extraction instead — but both
are checked by the same citation verifier, so the mock is held to the same evidence bar.

Canonical entity names are globally unique across types, so a relation can reference an entity by
name without ambiguity (e.g. ``high_inflation`` is only ever a regime).
"""

from __future__ import annotations

import re

from factorforge.models import EntityType

# Canonical name -> surface aliases. Aliases are curated to match the corpus precisely while
# avoiding cross-document false positives (e.g. bare "value" is omitted so "market value" in the
# size note does not register as the value factor).
VOCAB: dict[EntityType, dict[str, list[str]]] = {
    EntityType.FACTOR: {
        "value": ["value factor", "value premium", "value stock", "book-to-market", "hml"],
        "momentum": ["momentum factor", "momentum", "wml", "winners minus losers"],
        "size": ["size factor", "size effect", "size premium", "smb", "small-cap premium"],
        "quality": ["quality factor", "quality premium", "gross profitability", "profitability", "rmw", "quality"],
        "low_volatility": ["low-volatility", "low volatility", "low-vol", "volatility anomaly"],
    },
    EntityType.REGIME: {
        "high_inflation": ["high-inflation", "inflation"],
        "recession": ["recession"],
        "high_volatility": ["high-volatility", "high volatility"],
        "market_recovery": ["market recovery", "recovery"],
    },
    EntityType.ASSET: {
        "equities": ["equities", "equity", "stocks"],
        "small_cap": ["small-cap stock", "small-cap", "small cap"],
        "large_cap": ["large-cap stock", "large-cap", "large cap"],
    },
}

# Compiled (entity_type, canonical, regex) triples, longest-alias-first so specific phrases win.
_PATTERNS: list[tuple[EntityType, str, re.Pattern[str]]] = []
for _etype, _canon_map in VOCAB.items():
    for _canon, _aliases in _canon_map.items():
        _alt = "|".join(re.escape(a) for a in sorted(_aliases, key=len, reverse=True))
        _PATTERNS.append((_etype, _canon, re.compile(rf"\b(?:{_alt})\b", re.IGNORECASE)))

_HEADER_LINE = re.compile(r"^[ \t]*#{1,6}[ \t].*$", re.MULTILINE)
_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    """a an the of in on and or to for is are was were be been do does did this that these those
    it its as at by with from has have had not no than then when where which who whom whose why how
    can could should would may might will shall about into over under more most less least very
    you your we our they their he she his her them""".split()  # noqa: SIM905
)


def find_mentions(text: str) -> list[tuple[EntityType, str]]:
    """Unique ``(entity_type, canonical)`` mentions in ``text``, in stable VOCAB order."""
    out: list[tuple[EntityType, str]] = []
    for etype, canon, pat in _PATTERNS:
        if pat.search(text):
            out.append((etype, canon))
    return out


def split_sentences(text: str) -> list[str]:
    """Split body text into clean, whitespace-collapsed sentences (header lines removed)."""
    no_headers = _HEADER_LINE.sub("", text)
    collapsed = _WS.sub(" ", no_headers).strip()
    if not collapsed:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(collapsed) if s.strip()]


def tokenize(text: str) -> set[str]:
    """Content tokens for overlap scoring: lowercased, de-pluralized, stop-worded, len >= 3."""
    tokens: set[str] = set()
    for raw in _TOKEN.findall(text.lower()):
        tok = raw[:-1] if len(raw) > 3 and raw.endswith("s") else raw
        if len(tok) >= 3 and tok not in _STOPWORDS:
            tokens.add(tok)
    return tokens
