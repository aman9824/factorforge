"""Cost telemetry: the tracker accumulates real usage, the wiring feeds it, mock reports zero."""

from __future__ import annotations

from factorforge.backends.base import AgentBackend
from factorforge.config import Settings
from factorforge.models import DocTree, Document, NavSelection, RawExtraction
from factorforge.orchestrator import research
from factorforge.providers.base import LLMProvider
from factorforge.telemetry import CostTracker

QUESTION = "Does the value premium depend on the inflation regime?"


def _settings() -> Settings:
    return Settings(audit_enabled=False)  # no disk writes in tests


def test_cost_tracker_accumulates_by_stage() -> None:
    t = CostTracker()
    t.add("extract", input_tokens=100, output_tokens=20)
    t.add("extract", input_tokens=50, output_tokens=10)
    t.add("navigate", input_tokens=30, output_tokens=5)

    assert t.model_calls == 3
    assert t.input_tokens == 180
    assert t.output_tokens == 35
    assert t.total_tokens == 215
    assert t.by_stage == {"extract": 180, "navigate": 30 + 5}


def test_provider_records_usage_only_when_a_tracker_is_attached() -> None:
    """A provider that calls ``_record`` is a no-op until the orchestrator attaches a tracker."""

    class _RecordingProvider(LLMProvider):
        name = "recording"

        def extract(self, doc: Document) -> RawExtraction:
            self._record("extract", input_tokens=7, output_tokens=3)
            return RawExtraction(entities=[], relations=[])

        def navigate(self, query: str, tree: DocTree, max_nodes: int = 5) -> NavSelection:
            self._record("navigate", input_tokens=4, output_tokens=1)
            return NavSelection()

    provider = _RecordingProvider()
    doc = Document(id="d", title="t", text="body")

    provider.extract(doc)  # no tracker yet -> silently dropped
    assert provider.tracker is None

    tracker = CostTracker()
    provider.tracker = tracker
    provider.extract(doc)
    assert tracker.summary() == {
        "model_calls": 1,
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
        "by_stage": {"extract": 10},
    }


def test_agent_backend_exposes_a_tracker_seam() -> None:
    # The base seam exists so the orchestrator can attach one uniformly across backends.
    assert AgentBackend.tracker is None


def test_mock_pipeline_reports_zero_cost() -> None:
    # The mock makes no model calls, so the run's cost telemetry is honestly zero (not absent).
    report = research(QUESTION, settings=_settings())
    assert report.cost.model_calls == 0
    assert report.cost.total_tokens == 0
    assert report.cost.by_stage == {}
