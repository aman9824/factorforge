"""Real provider — Claude on Google Vertex AI via the Anthropic SDK.

Uses ``AnthropicVertex`` (ADC auth: ``gcloud auth application-default login``) and **forces** a
single tool call so the model must return schema-valid JSON. The output is still run through the
same citation verifier as the mock — the model is never trusted to have cited correctly; it is
checked. ``anthropic`` is imported lazily (see ``factory``) so it stays an optional dependency.

Note the model id: the AnthropicVertex SDK takes the **bare** id (``claude-opus-4-8``), unlike the
Claude Agent SDK which wants the ``[1m]``-suffixed id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.factorforge.config import Settings
from src.factorforge.corpus.structure import render_outline
from src.factorforge.extract.prompts import (
    EXTRACTION_SYSTEM,
    EXTRACTION_TOOL,
    NAVIGATION_SYSTEM,
    NAVIGATION_TOOL,
    build_extraction_prompt,
    build_navigation_prompt,
)
from src.factorforge.logging import get_logger
from src.factorforge.models import DocTree, Document, NavSelection, RawExtraction
from src.factorforge.providers.base import LLMProvider

if TYPE_CHECKING:
    from anthropic.types import MessageParam, ToolChoiceToolParam, ToolParam

log = get_logger(__name__)


class VertexProvider(LLMProvider):
    name = "vertex"

    def __init__(self, settings: Settings) -> None:
        from anthropic import AnthropicVertex  # optional dep, imported on use

        self.settings = settings
        self.model = settings.vertex_model
        self._client = AnthropicVertex(
            project_id=settings.vertex_project_id,
            region=settings.vertex_region,
            timeout=float(settings.request_timeout_s),
        )

    def extract(self, doc: Document) -> RawExtraction:
        data = self._call(
            EXTRACTION_SYSTEM,
            EXTRACTION_TOOL,
            build_extraction_prompt(doc.title, doc.text),
        )
        log.info("vertex.extract", document=doc.id, model=self.model)
        return RawExtraction.model_validate(data)

    def navigate(self, query: str, tree: DocTree, max_nodes: int = 5) -> NavSelection:
        data = self._call(
            NAVIGATION_SYSTEM,
            NAVIGATION_TOOL,
            build_navigation_prompt(query, render_outline(tree), max_nodes),
        )
        log.info("vertex.navigate", document=tree.doc_id, model=self.model)
        return NavSelection.model_validate(data)

    def _call(self, system: str, tool: dict[str, Any], user: str) -> dict[str, Any]:
        # Retry only TRANSIENT failures; 4xx are permanent and just waste calls.
        from anthropic import APIConnectionError, InternalServerError, RateLimitError
        from anthropic.types import ToolUseBlock

        tool_name = tool["name"]

        @retry(
            retry=retry_if_exception_type(
                (APIConnectionError, RateLimitError, InternalServerError)
            ),
            stop=stop_after_attempt(self.settings.max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            reraise=True,
        )
        def _do() -> dict[str, Any]:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                tools=[cast("ToolParam", tool)],
                tool_choice=cast("ToolChoiceToolParam", {"type": "tool", "name": tool_name}),
                messages=[cast("MessageParam", {"role": "user", "content": user})],
            )
            for block in resp.content:
                if isinstance(block, ToolUseBlock) and block.name == tool_name:
                    return cast("dict[str, Any]", block.input)
            raise ValueError(f"Vertex response did not contain a {tool_name} tool call")

        return _do()
