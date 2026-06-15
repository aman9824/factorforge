# Changelog

All notable changes to FactorForge are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [0.1.0] — 2026-06-14

Initial release — a complete, eval-gated, offline-by-default pipeline.

### Added
- **Scaffold:** typed `pydantic-settings` config with three offline-default seams
  (`llm_provider`, `agent_backend`, `data_source`), `structlog` logging, JSON-serializable domain
  models, CI (Python 3.11/3.12), Makefile, Dockerfile, MIT license.
- **Corpus → PageIndex tree:** frontmatter loader + deterministic hierarchical structure tree per
  document (titles, char-offset spans, summaries) — no embeddings.
- **Extraction + citations:** `LLMProvider` seam (deterministic `mock` + `AnthropicVertex`
  `vertex`) for entity/relation extraction and tree navigation; span-citation **verifier** that
  proves every extracted fact resolves to an exact source span.
- **Knowledge graph:** `GraphStore` interface + in-process `networkx` implementation (traversal,
  neighbors, shortest path, term search, JSON round-trip).
- **Vectorless retrieval:** graph traversal + PageIndex-style tree search + structured filters,
  composed into ranked, traceable `EvidencePath`s with proven citations.
- **MCP server:** standalone `FastMCP` knowledge-graph server (stdio + streamable-HTTP) exposing
  `graph_search`, `get_neighbors`, `list_factors`, `navigate_document`, `get_evidence`,
  `get_section` + `kg://`/`doc://` resources.
- **Backtest:** synthetic (planted-signal) + Fama-French (`make fetch-french`, never redistributed)
  data sources; cross-sectional long/short factor construction; `bt`-powered engine with our own
  stat math; overfitting diagnostics (deflated Sharpe, PBO via CSCV, IS/OOS decay, parameter
  sensitivity).
- **Multi-agent pipeline (Claude Agent SDK):** Researcher → Hypothesizer → Backtester →
  Risk/Overfitting Critic → Reporter as Agent Skills; deterministic mock backend + real Claude
  backend (external KG MCP over stdio; in-process backtest tools). Orchestrator enforces
  **verify-don't-trust** (independent re-run of the backtest tools).
- **Evals:** eval gate (extraction, citations, retrieval paths, factor inference, overfitting
  catch, backtest reproducibility) wired into CI, plus vectorless-retrieval cost telemetry.
- **Interfaces & enterprise:** typer CLI, FastAPI (RBAC-lite), append-only JSONL audit log.

### Honesty
- Research/educational only; **not financial advice**; no claim of real alpha. Backtests use
  synthetic or historical sample data; data provenance and limitations are disclosed.
