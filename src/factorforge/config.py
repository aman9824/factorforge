"""Typed, validated configuration loaded from environment / ``.env``.

Single source of truth for runtime knobs. **Every default makes the whole pipeline run offline
and deterministically** — the mock LLM provider, the mock agent backend, and the synthetic data
source — so a fresh clone runs ``make demo`` with zero credentials and CI runs free.

Three independent seams are selected here:

* ``llm_provider``  — structured generation (extraction, summaries, tree navigation).  mock | vertex
* ``agent_backend`` — the 5-role research pipeline.                                    mock | claude
* ``data_source``   — the price/return panel the backtest runs on.                     synthetic | french

The real Claude path needs only the ``FF_VERTEX_*`` block plus ``FF_LLM_PROVIDER=vertex`` and/or
``FF_AGENT_BACKEND=claude``. The real-data path needs ``FF_DATA_SOURCE=french`` after
``make fetch-french`` (which writes to a git-ignored cache; the data is never redistributed).
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderKind(StrEnum):
    """Structured-generation backend (extraction / summaries / navigation judgments)."""

    MOCK = "mock"
    VERTEX = "vertex"


class BackendKind(StrEnum):
    """Multi-agent pipeline execution backend."""

    MOCK = "mock"
    CLAUDE = "claude"


class DataSourceKind(StrEnum):
    """Backtest price/return panel source."""

    SYNTHETIC = "synthetic"
    FRENCH = "french"


class Settings(BaseSettings):
    """Application settings. Prefix every env var with ``FF_`` (e.g. ``FF_AGENT_BACKEND``)."""

    model_config = SettingsConfigDict(
        env_prefix="FF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Seam selection (all default to deterministic/offline) ────────────────
    llm_provider: ProviderKind = ProviderKind.MOCK
    agent_backend: BackendKind = BackendKind.MOCK
    data_source: DataSourceKind = DataSourceKind.SYNTHETIC

    # ── Orchestration reliability ────────────────────────────────────────────
    step_timeout_s: int = Field(default=240, ge=1, le=1800)
    max_turns: int = Field(default=60, ge=1, le=500)

    # ── Vectorless retrieval knobs ───────────────────────────────────────────
    max_nav_nodes: int = Field(default=8, ge=1, le=100)
    """Cap on nodes a single hierarchical-document tree search may return."""
    max_evidence_paths: int = Field(default=12, ge=1, le=200)
    """Cap on evidence paths surfaced per research query (bounds token cost)."""

    # ── Backtest ─────────────────────────────────────────────────────────────
    bt_initial_capital: float = Field(default=1_000_000.0, gt=0)
    synthetic_seed: int = Field(default=7, ge=0)
    """Seed for the deterministic synthetic returns panel — reproducibility knob."""

    # ── MCP server ───────────────────────────────────────────────────────────
    mcp_transport: Literal["stdio", "http"] = "stdio"
    mcp_http_host: str = "127.0.0.1"
    mcp_http_port: int = Field(default=8848, ge=1, le=65535)

    # ── Eval gate thresholds (CI fails when any is missed) ───────────────────
    min_extraction_accuracy: float = Field(default=0.90, ge=0.0, le=1.0)
    min_citation_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    min_retrieval_path_accuracy: float = Field(default=0.90, ge=0.0, le=1.0)
    max_backtest_repro_error: float = Field(default=1e-6, ge=0.0)
    """Max allowed |agent-reported - independently-recomputed| backtest stat (verify-don't-trust)."""
    min_overfitting_catch_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    """Fraction of deliberately-overfit fixtures the Risk/Overfitting Critic must flag."""

    # ── Audit & observability ────────────────────────────────────────────────
    audit_enabled: bool = True
    audit_log_path: str = "out/audit.jsonl"

    # ── API access control (simple RBAC; off by default for the demo) ────────
    api_require_auth: bool = False
    api_token: str = ""

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_json: bool = False

    # ── Vertex AI (only used when a real seam is selected) ───────────────────
    vertex_project_id: str = "ford-bbfd8f90a37aa7c3fcd1ded7"
    # Claude (Opus 4.x) is served on the GLOBAL endpoint, not a regional one.
    vertex_region: str = "global"
    # NOTE the two model ids: the AnthropicVertex SDK takes the bare id; the Claude Agent SDK
    # (CLI under the hood) takes the [1m] context-window suffixed id.
    vertex_model: str = "claude-opus-4-8"
    agent_model: str = "claude-opus-4-8[1m]"
    request_timeout_s: int = Field(default=120, ge=1, le=600)
    max_retries: int = Field(default=4, ge=0, le=10)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings (cached). Call ``get_settings.cache_clear()`` in tests."""
    return Settings()
