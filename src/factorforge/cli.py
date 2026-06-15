"""FactorForge command line — research, backtest, serve the MCP server, fetch data.

    factorforge demo                      # full pipeline on the offline mock + synthetic data
    factorforge research "<question>"     # research your own question
    factorforge backtest value            # single deterministic backtest + diagnostics
    factorforge build-graph               # build the knowledge graph and print its stats
    factorforge serve-mcp [--http]        # run the standalone knowledge-graph MCP server
    factorforge fetch-french              # download real Fama-French data locally (needs [french])
"""

from __future__ import annotations

import sys

# Windows-safe output regardless of the console code page.
_reconfigure = getattr(sys.stdout, "reconfigure", None)
if _reconfigure is not None:
    _reconfigure(encoding="utf-8")

import typer  # noqa: E402
from rich.console import Console  # noqa: E402

from factorforge.config import Settings, get_settings  # noqa: E402
from factorforge.logging import configure_logging  # noqa: E402
from factorforge.models import Report  # noqa: E402

app = typer.Typer(add_completion=False, help="FactorForge — auditable vectorless quant-research engine.")
console = Console()

DEFAULT_QUESTION = "Does the value premium depend on the inflation regime?"

_VERDICT_STYLE = {"promising": "green", "inconclusive": "yellow", "likely_overfit": "red"}


def _run(question: str, settings: Settings) -> Report:
    configure_logging(settings.log_level, settings.log_json)
    from factorforge.orchestrator import research

    return research(question, settings=settings)


def _print_report(report: Report) -> None:
    rec = report.risk.recommendation.value
    style = _VERDICT_STYLE.get(rec, "white")
    console.rule(f"[bold]FactorForge[/] — {report.backend} backend")
    console.print(f"[bold]Question:[/] {report.question}")
    console.print(f"[bold]Verdict:[/] [{style}]{rec}[/]  (overfitting risk: {report.risk.overfitting_risk.value})")
    console.print(f"[bold]Hypothesis:[/] {report.hypothesis.factor_name} "
                  f"({report.hypothesis.direction.value}, {report.hypothesis.quantiles} quantiles)")

    s = report.backtest.stats
    console.print(
        f"[bold]Backtest[/] (data: {report.backtest.data_source}, verified: {report.numbers_verified}): "
        f"Sharpe {s.sharpe:.2f}, CAGR {s.cagr:.1%}, vol {s.ann_vol:.1%}, maxDD {s.max_drawdown:.1%}, "
        f"IC {report.backtest.information_coefficient:+.2f}"
    )
    d = report.diagnostics
    console.print(
        f"[bold]Overfitting[/]: deflated Sharpe {d.deflated_sharpe:.2f}, PBO {d.pbo:.0%}, "
        f"IS/OOS {d.in_sample_sharpe:.2f}/{d.out_sample_sharpe:.2f}, {d.n_trials} trials"
    )

    console.print(f"\n[bold]Evidence[/] ({len(report.evidence_paths)} paths):")
    for path in report.evidence_paths[:5]:
        step = path.steps[0].label if path.steps else ""
        cite = path.citations[0] if path.citations else None
        where = f"{cite.doc_id} > {cite.section}" if cite else ""
        console.print(f"  [{path.kind.value}] {step}  [dim]({where})[/]")

    c = report.cost
    if c.model_calls:
        stages = ", ".join(f"{k} {v:,}" for k, v in c.by_stage.items())
        console.print(
            f"\n[bold]Cost[/]: {c.model_calls} model calls, {c.total_tokens:,} tokens "
            f"(in {c.input_tokens:,} / out {c.output_tokens:,})  [dim]{stages}[/]"
        )
    else:
        console.print("\n[bold]Cost[/]: 0 model calls (mock — no model usage)")

    console.print(f"\n[bold]Report:[/]\n{report.narrative}")
    console.print(f"\n[dim]{report.disclaimer}[/]")


@app.command()
def research(question: str = typer.Argument(..., help="The research question.")) -> None:
    """Run the multi-agent research pipeline on a question."""
    _print_report(_run(question, get_settings()))


@app.command()
def demo() -> None:
    """Run the canonical demo question end-to-end (offline mock + synthetic data)."""
    _print_report(_run(DEFAULT_QUESTION, get_settings()))


@app.command()
def backtest(
    factor: str = typer.Argument(..., help="Factor name (e.g. value, momentum, size)."),
    direction: str = typer.Option("high_minus_low", help="high_minus_low | low_minus_high"),
    quantiles: int = typer.Option(5, help="Number of quantiles."),
) -> None:
    """Run a single deterministic backtest + overfitting diagnostics."""
    from factorforge.backtest.tools import backtest_hypothesis, diagnose_hypothesis
    from factorforge.factory import build_data_source
    from factorforge.models import FactorDirection, FactorHypothesis

    settings = get_settings()
    ds = build_data_source(settings)
    hyp = FactorHypothesis(
        factor_name=factor, thesis="cli", rank_signal="characteristic",
        direction=FactorDirection(direction), quantiles=quantiles,
    )
    bt = backtest_hypothesis(hyp, ds)
    diag = diagnose_hypothesis(hyp, ds)
    s = bt.stats
    console.print(f"[bold]{factor}[/] ({direction}, q={quantiles}, data={bt.data_source})")
    console.print(f"  Sharpe {s.sharpe:.2f}  CAGR {s.cagr:.1%}  vol {s.ann_vol:.1%}  maxDD {s.max_drawdown:.1%}  IC {bt.information_coefficient:+.2f}")
    console.print(f"  deflated Sharpe {diag.deflated_sharpe:.2f}  PBO {diag.pbo:.0%}  IS/OOS {diag.in_sample_sharpe:.2f}/{diag.out_sample_sharpe:.2f}")
    console.print("  [dim]Research/educational only — not financial advice.[/]")


@app.command("build-graph")
def build_graph() -> None:
    """Build the knowledge graph from the corpus and print its statistics."""
    from factorforge.factory import build_provider
    from factorforge.knowledge import build_knowledge_base

    settings = get_settings()
    kb = build_knowledge_base(build_provider(settings))
    console.print(f"Knowledge graph: {kb.graph.stats()}  ({len(kb.docs)} documents)")


@app.command("serve-mcp")
def serve_mcp(http: bool = typer.Option(False, "--http", help="Serve over streamable-HTTP instead of stdio.")) -> None:
    """Run the standalone knowledge-graph MCP server."""
    from factorforge.mcp_server.__main__ import main as mcp_main

    mcp_main(["--http"] if http else [])


@app.command("fetch-french")
def fetch_french() -> None:
    """Download real Fama-French data into the git-ignored cache (requires the [french] extra)."""
    from factorforge.backtest.fetch_french import main as fetch_main

    fetch_main()


if __name__ == "__main__":
    app()
