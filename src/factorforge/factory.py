"""Dependency-injection factory: build the configured seam implementations from settings.

Real implementations are imported lazily so their heavy / credentialed dependencies
(``anthropic``, ``claude-agent-sdk``, ``pandas-datareader``) stay optional — a fresh clone runs
the full mock pipeline without them.
"""

from __future__ import annotations

from factorforge.backends.base import AgentBackend
from factorforge.backtest.data import DataSource
from factorforge.config import BackendKind, DataSourceKind, ProviderKind, Settings
from factorforge.knowledge import KnowledgeBase
from factorforge.providers.base import LLMProvider


def build_provider(settings: Settings) -> LLMProvider:
    """Build the structured-generation provider (extraction / navigation)."""
    if settings.llm_provider == ProviderKind.VERTEX:
        from factorforge.providers.vertex import VertexProvider

        return VertexProvider(settings)
    from factorforge.providers.mock import MockProvider

    return MockProvider()


def build_data_source(settings: Settings) -> DataSource:
    """Build the backtest data source (synthetic by default; real Fama-French on demand)."""
    if settings.data_source == DataSourceKind.FRENCH:
        from factorforge.backtest.data import FrenchDataSource

        return FrenchDataSource()
    from factorforge.backtest.data import SyntheticDataSource

    return SyntheticDataSource(seed=settings.synthetic_seed)


def build_backend(
    settings: Settings, kb: KnowledgeBase, data_source: DataSource, provider: LLMProvider
) -> AgentBackend:
    """Build the multi-agent execution backend (deterministic mock by default; Claude on demand)."""
    if settings.agent_backend == BackendKind.CLAUDE:
        from factorforge.backends.claude import ClaudeAgentBackend

        return ClaudeAgentBackend(settings, data_source, kb)
    from factorforge.backends.mock import MockBackend

    return MockBackend(settings, kb, data_source, provider)
