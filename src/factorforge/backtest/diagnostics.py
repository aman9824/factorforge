"""Overfitting diagnostics — the anti-self-deception layer.

A high in-sample Sharpe means little until it survives honest checks. We compute:

* **IS/OOS decay** — Sharpe on the first vs. second half of the sample.
* **Deflated Sharpe Ratio (DSR)** — the Probabilistic Sharpe Ratio evaluated against the *expected
  maximum* Sharpe under ``n_trials`` searches (Bailey & López de Prado), so more searching raises
  the bar.
* **Probability of Backtest Overfitting (PBO)** — via Combinatorially-Symmetric Cross-Validation
  over a grid of configurations: how often the in-sample-best config underperforms out-of-sample.
* **Parameter sensitivity** — dispersion of Sharpe across the configuration grid.

All closed-form / combinatorial — deterministic and unit-testable. References: Bailey & López de
Prado (2014, Deflated Sharpe Ratio); Bailey et al. (2017, Probability of Backtest Overfitting).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from src.factorforge.backtest.data import FactorPanel
from src.factorforge.backtest.engine import PERIODS_PER_YEAR, strategy_returns
from src.factorforge.models import FactorHypothesis, OverfittingReport

# Configuration grid for PBO / parameter-sensitivity / Sharpe-variance. Kept small (each variant is
# a full bt run, ~3s); 3 quantile cuts is enough signal while keeping the pipeline responsive.
_GRID_QUANTILES = (3, 5, 10)


def _pp_sharpe(rets: pd.Series) -> float:
    """Per-period Sharpe (mean / std)."""
    sd = float(rets.std(ddof=1))
    return float(rets.mean()) / sd if sd > 0 else 0.0


def _ann_sharpe(rets: pd.Series) -> float:
    return _pp_sharpe(rets) * math.sqrt(PERIODS_PER_YEAR)


def probabilistic_sharpe_ratio(rets: pd.Series, benchmark_sr: float = 0.0) -> float:
    """P(true per-period Sharpe > benchmark_sr), correcting for sample length, skew, and kurtosis."""
    n = int(len(rets))
    if n < 3:
        return 0.0
    sr = _pp_sharpe(rets)
    g3 = float(skew(rets, bias=False))
    g4 = float(kurtosis(rets, fisher=False, bias=False))  # normal -> 3
    denom = math.sqrt(max(1e-12, 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr))
    z = (sr - benchmark_sr) * math.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period Sharpe across ``n_trials`` independent null strategies."""
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    gamma = 0.5772156649015329  # Euler–Mascheroni
    z1 = float(norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return math.sqrt(sr_variance) * ((1.0 - gamma) * z1 + gamma * z2)


def deflated_sharpe_ratio(rets: pd.Series, n_trials: int, sr_variance: float) -> float:
    """PSR evaluated against the expected-maximum Sharpe under multiple testing."""
    return probabilistic_sharpe_ratio(rets, benchmark_sr=expected_max_sharpe(n_trials, sr_variance))


def probability_of_backtest_overfitting(matrix: pd.DataFrame, n_splits: int = 4) -> float:
    """CSCV PBO: fraction of IS/OOS partitions where the IS-best config lands below the OOS median."""
    t, n_cfg = matrix.shape
    if n_cfg < 2 or t < n_splits:
        return 0.0
    chunks = np.array_split(np.arange(t), n_splits)
    half = n_splits // 2
    below_median = 0
    total = 0
    for combo in itertools.combinations(range(n_splits), half):
        is_idx = np.concatenate([chunks[c] for c in combo])
        oos_idx = np.concatenate([chunks[c] for c in range(n_splits) if c not in combo])
        is_perf = matrix.iloc[is_idx].apply(_pp_sharpe, axis=0)
        oos_perf = matrix.iloc[oos_idx].apply(_pp_sharpe, axis=0)
        best = int(np.asarray(is_perf.values).argmax())
        oos_rank = float(oos_perf.rank().iloc[best])      # 1 (worst) .. n_cfg (best)
        omega = min(max(oos_rank / (n_cfg + 1), 1e-6), 1 - 1e-6)
        logit = math.log(omega / (1.0 - omega))
        below_median += 1 if logit <= 0 else 0
        total += 1
    return below_median / total if total else 0.0


def run_diagnostics(
    panel: FactorPanel, hyp: FactorHypothesis, n_trials: int | None = None
) -> OverfittingReport:
    base = strategy_returns(panel, hyp)
    mid = len(base) // 2
    is_sr = _ann_sharpe(base.iloc[:mid])
    oos_sr = _ann_sharpe(base.iloc[mid:])

    # Configuration grid (vary the quantile cut) → a returns matrix for PBO + sensitivity.
    n_assets = panel.returns.shape[1]
    quantiles = sorted({hyp.quantiles} | {q for q in _GRID_QUANTILES if 2 <= q <= n_assets and n_assets // q >= 1})
    columns = {f"q{q}": strategy_returns(panel, hyp.model_copy(update={"quantiles": q})) for q in quantiles}
    matrix = pd.DataFrame(columns).dropna()

    grid_pp = [_pp_sharpe(matrix[c]) for c in matrix.columns]
    grid_ann = [_ann_sharpe(matrix[c]) for c in matrix.columns]
    sr_variance = float(np.var(grid_pp, ddof=1)) if len(grid_pp) > 1 else 1.0 / max(len(base), 2)
    param_sensitivity = float(np.std(grid_ann, ddof=1)) if len(grid_ann) > 1 else 0.0

    trials = n_trials if n_trials is not None else len(quantiles)
    dsr = deflated_sharpe_ratio(base, trials, sr_variance)
    pbo = probability_of_backtest_overfitting(matrix)

    notes = [
        f"Searched {trials} configuration(s); Sharpe dispersion across the grid = {param_sensitivity:.2f}.",
        f"In-sample Sharpe {is_sr:.2f} vs out-of-sample {oos_sr:.2f} (decay {is_sr - oos_sr:+.2f}).",
    ]
    if dsr < 0.90:
        notes.append(f"Deflated Sharpe {dsr:.2f} is below 0.90 — not significant after multiple testing.")
    if pbo > 0.6:
        notes.append(f"PBO {pbo:.0%} is elevated (supplementary; underpowered with few configurations).")

    return OverfittingReport(
        in_sample_sharpe=is_sr,
        out_sample_sharpe=oos_sr,
        sharpe_decay=is_sr - oos_sr,
        deflated_sharpe=dsr,
        pbo=pbo,
        n_trials=trials,
        param_sensitivity=param_sensitivity,
        notes=notes,
    )
