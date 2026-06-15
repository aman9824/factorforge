"""The Risk/Overfitting Critic: it must reject overfit factors and pass robust ones."""

from __future__ import annotations

from factorforge.backends.mock import MockBackend
from factorforge.backtest.data import SyntheticDataSource
from factorforge.backtest.tools import diagnose_hypothesis
from factorforge.models import (
    FactorDirection,
    FactorHypothesis,
    OverfittingReport,
    Recommendation,
    Severity,
)


def _hyp(name: str, direction: FactorDirection = FactorDirection.HIGH_MINUS_LOW) -> FactorHypothesis:
    return FactorHypothesis(factor_name=name, thesis="t", rank_signal="char", direction=direction, quantiles=5)


def test_judge_flags_overfit_diagnostics() -> None:
    bad = OverfittingReport(
        in_sample_sharpe=2.0, out_sample_sharpe=0.1, sharpe_decay=1.9,
        deflated_sharpe=0.30, pbo=0.70, n_trials=50, param_sensitivity=0.8,
    )
    risk = MockBackend._judge(bad)
    assert risk.recommendation == Recommendation.LIKELY_OVERFIT
    assert risk.overfitting_risk == Severity.HIGH
    assert any(f.severity == Severity.HIGH for f in risk.risk_flags)


def test_judge_passes_robust_diagnostics() -> None:
    good = OverfittingReport(
        in_sample_sharpe=1.6, out_sample_sharpe=1.5, sharpe_decay=0.1,
        deflated_sharpe=0.99, pbo=0.10, n_trials=5, param_sensitivity=0.1,
    )
    risk = MockBackend._judge(good)
    assert risk.recommendation == Recommendation.PROMISING
    assert risk.risk_flags == []


def test_overfit_fixture_is_caught_under_multiple_testing() -> None:
    ds = SyntheticDataSource(seed=7)
    # 'size' has no planted premium; searched across many trials, it must not survive.
    diag = diagnose_hypothesis(_hyp("size", FactorDirection.LOW_MINUS_HIGH), ds, n_trials=50)
    assert diag.deflated_sharpe < 0.90
    assert MockBackend._judge(diag).recommendation == Recommendation.LIKELY_OVERFIT


def test_real_factor_survives_honest_trials() -> None:
    ds = SyntheticDataSource(seed=7)
    diag = diagnose_hypothesis(_hyp("value"), ds, n_trials=5)
    assert diag.deflated_sharpe >= 0.90
    assert MockBackend._judge(diag).recommendation == Recommendation.PROMISING
