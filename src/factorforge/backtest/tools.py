"""The agent-callable backtest tools — deterministic numbers, never the model's.

``run_backtest`` and ``run_diagnostics`` are the only way a number enters the pipeline. The
Backtester and Critic agents *invoke* them; the orchestrator independently *re-invokes* them to
verify the agents reported what the tools actually returned (verify, don't trust — P2's stance).

Each tool has a provider-agnostic :class:`ToolSpec` (so any backend can register it) and a plain
dict-in/dict-out callable (for transport), plus a typed convenience for internal use.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.factorforge.backtest.data import DataSource
from src.factorforge.backtest.diagnostics import run_diagnostics
from src.factorforge.backtest.engine import run_backtest
from src.factorforge.models import BacktestResult, FactorHypothesis, OverfittingReport


class ToolSpec(BaseModel):
    """Provider-agnostic tool descriptor so any backend can register the tool its own way."""

    name: str
    description: str
    input_schema: dict[str, Any]


# ── typed convenience (used internally by the orchestrator) ───────────────────

def backtest_hypothesis(hyp: FactorHypothesis, data_source: DataSource) -> BacktestResult:
    return run_backtest(data_source.load(hyp.factor_name), hyp)


def diagnose_hypothesis(
    hyp: FactorHypothesis, data_source: DataSource, n_trials: int | None = None
) -> OverfittingReport:
    return run_diagnostics(data_source.load(hyp.factor_name), hyp, n_trials=n_trials)


# ── dict-in/dict-out callables (tool transport) ───────────────────────────────

def run_backtest_tool(inputs: dict[str, Any], data_source: DataSource) -> dict[str, Any]:
    hyp = FactorHypothesis.model_validate(inputs)
    return backtest_hypothesis(hyp, data_source).model_dump()


def run_diagnostics_tool(inputs: dict[str, Any], data_source: DataSource) -> dict[str, Any]:
    data = dict(inputs)
    n_trials = data.pop("n_trials", None)
    hyp = FactorHypothesis.model_validate(data)
    return diagnose_hypothesis(hyp, data_source, n_trials=n_trials).model_dump()


BACKTEST_TOOL = ToolSpec(
    name="run_backtest",
    description=(
        "Run a deterministic backtest of a cross-sectional long/short factor hypothesis and return "
        "its performance statistics (Sharpe, CAGR, vol, max drawdown, information coefficient, "
        "turnover). This is the ONLY source of backtest numbers — never compute them yourself."
    ),
    input_schema=FactorHypothesis.model_json_schema(),
)

DIAGNOSTICS_TOOL = ToolSpec(
    name="run_diagnostics",
    description=(
        "Compute overfitting diagnostics for a factor hypothesis: in/out-of-sample Sharpe decay, "
        "deflated Sharpe ratio (adjusted for the number of trials), probability of backtest "
        "overfitting (PBO), and parameter sensitivity. Optionally include an integer 'n_trials' to "
        "account for how many candidates were searched."
    ),
    input_schema=FactorHypothesis.model_json_schema(),
)
