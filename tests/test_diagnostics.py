"""Overfitting diagnostics: PSR, deflated Sharpe, PBO, and value-vs-noise discrimination."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorforge.backtest.data import SyntheticDataSource
from factorforge.backtest.diagnostics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    run_diagnostics,
)
from factorforge.models import FactorDirection, FactorHypothesis


def _hyp(name: str) -> FactorHypothesis:
    return FactorHypothesis(
        factor_name=name, thesis="t", rank_signal="char",
        direction=FactorDirection.HIGH_MINUS_LOW, quantiles=5,
    )


def test_psr_high_for_skilled_low_for_noise() -> None:
    rng = np.random.default_rng(0)
    skilled = pd.Series(rng.normal(0.01, 0.02, 180))   # positive mean, modest vol
    # Demean so the realized Sharpe is exactly 0 -> PSR is exactly 0.5 (otherwise sqrt(T-1)
    # amplifies any tiny sample mean and the PSR drifts far from 0.5).
    raw = pd.Series(rng.normal(0.00, 0.02, 180))
    noise = raw - raw.mean()
    assert probabilistic_sharpe_ratio(skilled, 0.0) > 0.9
    assert probabilistic_sharpe_ratio(noise, 0.0) == pytest.approx(0.5, abs=0.02)


def test_expected_max_sharpe_grows_with_trials() -> None:
    assert expected_max_sharpe(2, 0.01) < expected_max_sharpe(10, 0.01) < expected_max_sharpe(100, 0.01)
    assert expected_max_sharpe(1, 0.01) == 0.0


def test_deflated_sharpe_falls_as_trials_rise() -> None:
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.012, 0.02, 200))
    few = deflated_sharpe_ratio(rets, n_trials=1, sr_variance=0.01)
    many = deflated_sharpe_ratio(rets, n_trials=200, sr_variance=0.01)
    assert few >= many


def test_pbo_low_when_one_config_dominates() -> None:
    idx = pd.date_range("2004-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(2)
    matrix = pd.DataFrame(
        {
            "winner": rng.normal(0.02, 0.02, 120),   # strong in every sub-period
            "n1": rng.normal(0.0, 0.02, 120),
            "n2": rng.normal(0.0, 0.02, 120),
            "n3": rng.normal(0.0, 0.02, 120),
        },
        index=idx,
    )
    assert probability_of_backtest_overfitting(matrix) < 0.5


def test_run_diagnostics_discriminates_real_from_noise() -> None:
    ds = SyntheticDataSource(seed=7)
    value = run_diagnostics(ds.load("value"), _hyp("value"))
    size = run_diagnostics(ds.load("size"), _hyp("size"))

    assert value.deflated_sharpe > 0.9            # robust, real premium
    assert value.deflated_sharpe > size.deflated_sharpe
    assert value.in_sample_sharpe > 0 and value.out_sample_sharpe > 0
    assert 0.0 <= value.pbo <= 1.0 and 0.0 <= size.pbo <= 1.0


def test_run_diagnostics_is_deterministic() -> None:
    ds = SyntheticDataSource(seed=7)
    a = run_diagnostics(ds.load("value"), _hyp("value"))
    b = run_diagnostics(ds.load("value"), _hyp("value"))
    assert a.model_dump() == b.model_dump()
