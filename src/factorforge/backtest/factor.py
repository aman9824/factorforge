"""Cross-sectional factor construction + signal-quality math — our own, unit-tested.

This is the layer that turns a ranking *signal* into a dollar-neutral long/short *portfolio*: rank
the cross-section each period, go long the top quantile and short the bottom (equal-weight, gross
1.0, net 0.0). ``bt`` consumes these weights to run the portfolio; the math here is ours so it can
be tested to the decimal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factorforge.models import FactorDirection


def build_target_weights(
    signal: pd.DataFrame, quantiles: int, direction: FactorDirection
) -> pd.DataFrame:
    """Dollar-neutral long/short target weights (rows sum to 0, gross exposure 1).

    Long the top ``1/quantiles`` of the cross-section by signal, short the bottom ``1/quantiles``,
    equal-weighted within each leg. ``LOW_MINUS_HIGH`` flips the legs.
    """
    n = signal.shape[1]
    k = n // quantiles
    if k < 1:
        raise ValueError(f"quantiles={quantiles} too large for a {n}-asset universe")

    ranks = signal.rank(axis=1, method="first")
    long_leg = (ranks > (n - k)).astype(float)
    short_leg = (ranks <= k).astype(float)
    weights = long_leg * (0.5 / k) - short_leg * (0.5 / k)
    if direction == FactorDirection.LOW_MINUS_HIGH:
        weights = -weights
    return weights


def information_coefficient(signal: pd.DataFrame, returns: pd.DataFrame) -> float:
    """Mean cross-sectional rank (Spearman) correlation between signal_t and next-period returns."""
    forward = returns.shift(-1)
    ics: list[float] = []
    for date in signal.index[:-1]:
        ic = signal.loc[date].corr(forward.loc[date], method="spearman")
        if pd.notna(ic):
            ics.append(float(ic))
    return float(np.mean(ics)) if ics else 0.0


def turnover(weights: pd.DataFrame) -> float:
    """Average one-sided turnover per rebalance (sum of |Δweight| / 2)."""
    changes = weights.diff().abs().sum(axis=1) / 2.0
    tail = changes.iloc[1:]
    return float(tail.mean()) if len(tail) else 0.0
