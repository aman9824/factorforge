"""The LLM provider seam — the one place the system does structured generation.

Two operations need a language model, so they are the two methods we abstract:

* :meth:`extract`  — pull entities & relations (with supporting quotes) out of a document.
* :meth:`navigate` — choose the relevant nodes of a document's structure tree for a query
  (PageIndex-style tree search), returning the reasoning and the chosen node ids.

Both return *unverified* output. Verification (re-checking every quote against the source) happens
downstream in :mod:`factorforge.citations.verifier`, so the mock and the real model are held to
the exact same evidence bar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.factorforge.models import DocTree, Document, NavSelection, RawExtraction


class LLMProvider(ABC):
    """Strategy interface for structured generation."""

    name: str = "base"

    @abstractmethod
    def extract(self, doc: Document) -> RawExtraction:
        """Return entities & relations found in ``doc``, each with a supporting quote."""
        raise NotImplementedError

    @abstractmethod
    def navigate(self, query: str, tree: DocTree, max_nodes: int = 5) -> NavSelection:
        """Choose up to ``max_nodes`` relevant tree nodes for ``query`` (with reasoning)."""
        raise NotImplementedError
