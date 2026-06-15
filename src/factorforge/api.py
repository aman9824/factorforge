"""FastAPI surface for FactorForge.

Endpoints: ``/health``, ``POST /research`` (run the pipeline), ``GET /graph`` (graph stats).
A simple token check (RBAC-lite) gates the data endpoints when ``FF_API_REQUIRE_AUTH=true``; it is
off by default so the demo is open. The knowledge base is built once and cached.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from factorforge.config import get_settings
from factorforge.knowledge import KnowledgeBase, build_knowledge_base
from factorforge.models import Report

app = FastAPI(
    title="FactorForge",
    description="Auditable vectorless quant-research engine. Research/educational — not financial advice.",
    version="0.1.0",
)


class ResearchRequest(BaseModel):
    question: str


@lru_cache(maxsize=1)
def _knowledge_base() -> KnowledgeBase:
    from factorforge.factory import build_provider

    return build_knowledge_base(build_provider(get_settings()))


def require_auth(x_api_token: str | None = Header(default=None)) -> None:
    """RBAC-lite: enforce a shared token when FF_API_REQUIRE_AUTH is set."""
    settings = get_settings()
    if settings.api_require_auth and x_api_token != settings.api_token:
        raise HTTPException(status_code=401, detail="invalid or missing API token")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research", dependencies=[Depends(require_auth)])
def research_endpoint(request: ResearchRequest) -> Report:
    from factorforge.orchestrator import research

    settings = get_settings()
    return research(request.question, settings=settings, kb=_knowledge_base())


@app.get("/graph", dependencies=[Depends(require_auth)])
def graph_endpoint() -> dict[str, int]:
    return _knowledge_base().graph.stats()
