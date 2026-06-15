"""Backtest data, factor construction, and the bt engine (deterministic numbers)."""

from __future__ import annotations

import pandas as pd
import pytest

from factorforge.backtest.data import FrenchDataSource, SyntheticDataSource
from factorforge.backtest.engine import compute_stats, run_backtest
from factorforge.backtest.factor import build_target_weights
from factorforge.backtest.tools import backtest_hypothesis, run_backtest_tool
from factorforge.models import FactorDirection, FactorHypothesis


def _hyp(name: str, quantiles: int = 5) -> FactorHypothesis:
    return FactorHypothesis(
        factor_name=name, thesis="t", rank_signal="char",
        direction=FactorDirection.HIGH_MINUS_LOW, quantiles=quantiles,
    )


def test_synthetic_source_is_deterministic() -> None:
    a = SyntheticDataSource(seed=7).load("value")
    b = SyntheticDataSource(seed=7).load("value")
    pd.testing.assert_frame_equal(a.returns, b.returns)
    pd.testing.assert_frame_equal(a.signal, b.signal)


def test_build_target_weights_is_dollar_neutral() -> None:
    signal = pd.DataFrame([[10.0, 20.0, 30.0, 40.0]], columns=["A", "B", "C", "D"])
    w = build_target_weights(signal, quantiles=2, direction=FactorDirection.HIGH_MINUS_LOW)
    row = w.iloc[0]
    assert row["A"] == pytest.approx(-0.25) and row["B"] == pytest.approx(-0.25)
    assert row["C"] == pytest.approx(0.25) and row["D"] == pytest.approx(0.25)
    assert row.sum() == pytest.approx(0.0)              # dollar neutral
    assert row.abs().sum() == pytest.approx(1.0)        # gross 1.0


def test_compute_stats_matches_hand_calculation() -> None:
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    equity = pd.Series([100.0, 110.0, 99.0], index=idx)
    stats = compute_stats(equity)
    assert stats.total_return == pytest.approx(-0.01)
    assert stats.max_drawdown == pytest.approx(-0.10)
    # rets = [+0.1, -0.1] -> sample std (ddof=1) = sqrt(0.02); annualized = sqrt(0.02 * 12).
    assert stats.ann_vol == pytest.approx((0.02 * 12) ** 0.5, rel=1e-6)


def test_planted_signal_is_recovered_noise_is_not() -> None:
    ds = SyntheticDataSource(seed=7)
    value = run_backtest(ds.load("value"), _hyp("value"))
    size = run_backtest(ds.load("size"), _hyp("size"))
    assert value.stats.sharpe > 1.0          # real planted premium
    assert value.information_coefficient > 0.03
    assert size.stats.sharpe < 0.6           # no premium planted


def test_backtest_is_deterministic_for_verification() -> None:
    ds = SyntheticDataSource(seed=7)
    first = run_backtest(ds.load("value"), _hyp("value"))
    second = run_backtest(ds.load("value"), _hyp("value"))
    assert first.model_dump() == second.model_dump()


def test_tool_dict_io_matches_typed() -> None:
    ds = SyntheticDataSource(seed=7)
    hyp = _hyp("quality")
    typed = backtest_hypothesis(hyp, ds).model_dump()
    via_tool = run_backtest_tool(hyp.model_dump(), ds)
    assert typed == via_tool


def test_french_source_fails_gracefully_without_cache(tmp_path: object) -> None:
    src = FrenchDataSource(cache_dir=tmp_path)  # type: ignore[arg-type]
    assert src.available_factors() == []
    with pytest.raises(FileNotFoundError):
        src.load("value")
