"""CLI commands run cleanly; the API serves research + graph with RBAC-lite."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import factorforge.api as api_mod
from factorforge.api import app as api_app
from factorforge.cli import app as cli_app
from factorforge.config import Settings

runner = CliRunner()
QUESTION = "Does the value premium depend on the inflation regime?"


def test_cli_demo_runs() -> None:
    result = runner.invoke(cli_app, ["demo"])
    assert result.exit_code == 0, result.output


def test_cli_backtest_runs() -> None:
    result = runner.invoke(cli_app, ["backtest", "momentum"])
    assert result.exit_code == 0, result.output


def test_cli_build_graph_runs() -> None:
    result = runner.invoke(cli_app, ["build-graph"])
    assert result.exit_code == 0, result.output


def test_api_health() -> None:
    assert TestClient(api_app).get("/health").json() == {"status": "ok"}


def test_api_research_and_graph() -> None:
    client = TestClient(api_app)
    resp = client.post("/research", json={"question": QUESTION})
    assert resp.status_code == 200
    body = resp.json()
    assert body["hypothesis"]["factor_name"] == "value"
    assert body["numbers_verified"] is True
    assert body["risk"]["recommendation"] == "promising"

    graph = client.get("/graph")
    assert graph.status_code == 200 and graph.json()["entities"] > 0


def test_api_rbac_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_mod, "get_settings", lambda: Settings(api_require_auth=True, api_token="secret"))
    client = TestClient(api_app)

    # Missing token -> rejected.
    assert client.post("/research", json={"question": QUESTION}).status_code == 401
    # Correct token -> allowed.
    ok = client.post("/research", json={"question": QUESTION}, headers={"X-API-Token": "secret"})
    assert ok.status_code == 200
