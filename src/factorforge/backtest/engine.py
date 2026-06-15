"""The backtest engine — ``bt`` runs the portfolio; we own the statistics.

``bt`` (MIT) handles the genuinely fiddly part: applying target weights, rebalancing monthly, and
producing the strategy equity curve. We then compute the headline statistics ourselves from that
curve with an explicit periods-per-year, so the numbers are deterministic, frequency-correct, and
unit-testable — never produced by a model.
"""

from __future__ import annotations

import math
import os

# ffn (pulled in by bt) imports matplotlib.pyplot at import time — force a headless backend before
# importing bt so this is safe on a server / in CI regardless of how the process was launched.
os.environ.setdefault("MPLBACKEND", "Agg")

from typing import Any  # noqa: E402

import bt  # noqa: E402
import pandas as pd  # noqa: E402

from factorforge.backtest.data import FactorPanel  # noqa: E402
from factorforge.backtest.factor import (  # noqa: E402
    build_target_weights,
    information_coefficient,
    turnover,
)
from factorforge.models import BacktestResult, BacktestStats, FactorHypothesis  # noqa: E402

PERIODS_PER_YEAR = 12  # the bundled panels are monthly

# bt is slow per run and a single pipeline backtests the same (deterministic) configuration many
# times (the diagnostics grid, the verify-don't-trust re-run). Cache equity curves content-keyed so
# repeats are free. The result is a pure function of (panel content, direction, quantiles, capital).
_EQUITY_CACHE: dict[tuple[Any, ...], Any] = {}


def _panel_key(panel: FactorPanel) -> tuple[Any, ...]:
    return (
        panel.source,
        panel.name,
        int(pd.util.hash_pandas_object(panel.returns).sum()),
        int(pd.util.hash_pandas_object(panel.signal).sum()),
    )


def returns_to_prices(returns: pd.DataFrame) -> pd.DataFrame:
    """Convert a returns matrix to a price index (base 100) for ``bt``."""
    return (1.0 + returns).cumprod() * 100.0


def _equity_curve(
    prices: pd.DataFrame, weights: pd.DataFrame, name: str, capital: float
) -> pd.Series:
    strategy = bt.Strategy(
        name,
        [
            bt.algos.RunMonthly(run_on_first_date=True),
            bt.algos.SelectAll(),
            bt.algos.WeighTarget(weights),
            bt.algos.Rebalance(),
        ],
    )
    backtest = bt.Backtest(
        strategy, prices, initial_capital=capital, integer_positions=False, progress_bar=False
    )
    # Run the simulation directly (not bt.run, which also builds an ffn stats Result we don't use —
    # we compute our own stats from the equity curve). This is the equity curve series.
    backtest.run()
    equity: pd.Series = backtest.strategy.prices
    return equity


def _simulate(
    panel: FactorPanel, hyp: FactorHypothesis, capital: float
) -> tuple[pd.Series, pd.DataFrame]:
    weights = build_target_weights(panel.signal, hyp.quantiles, hyp.direction)
    key = (_panel_key(panel), hyp.direction.value, hyp.quantiles, capital)
    equity = _EQUITY_CACHE.get(key)
    if equity is None:
        prices = returns_to_prices(panel.returns)
        equity = _equity_curve(prices, weights, hyp.factor_name, capital)
        _EQUITY_CACHE[key] = equity
    return equity, weights


def compute_stats(equity: pd.Series) -> BacktestStats:
    """Headline performance stats computed from an equity curve (monthly → annualized)."""
    rets = equity.pct_change().dropna()
    n = int(len(rets))
    start_val, end_val = float(equity.iloc[0]), float(equity.iloc[-1])

    total_return = end_val / start_val - 1.0
    years = n / PERIODS_PER_YEAR if n else 1.0
    cagr = (end_val / start_val) ** (1.0 / years) - 1.0 if years > 0 and start_val > 0 else 0.0
    ann_vol = float(rets.std(ddof=1)) * math.sqrt(PERIODS_PER_YEAR) if n > 1 else 0.0
    ann_ret = float(rets.mean()) * PERIODS_PER_YEAR if n else 0.0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min()) if n else 0.0
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    return BacktestStats(
        total_return=total_return, cagr=cagr, ann_vol=ann_vol,
        sharpe=sharpe, max_drawdown=max_dd, calmar=calmar,
    )


def strategy_returns(panel: FactorPanel, hyp: FactorHypothesis, capital: float = 1_000_000.0) -> pd.Series:
    """The strategy's realized periodic returns (used by the diagnostics layer)."""
    equity, _ = _simulate(panel, hyp, capital)
    return equity.pct_change().dropna()


def run_backtest(
    panel: FactorPanel, hyp: FactorHypothesis, capital: float = 1_000_000.0
) -> BacktestResult:
    """Run the full backtest and return deterministic, frequency-correct statistics."""
    equity, weights = _simulate(panel, hyp, capital)
    rets = equity.pct_change().dropna()
    return BacktestResult(
        factor_name=hyp.factor_name,
        stats=compute_stats(equity),
        n_periods=int(len(rets)),
        start=str(equity.index[0].date()),
        end=str(equity.index[-1].date()),
        information_coefficient=information_coefficient(panel.signal, panel.returns),
        turnover=turnover(weights),
        data_source=panel.source,
    )
