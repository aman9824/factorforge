"""Backtest data sources — the third seam (synthetic by default, real Fama-French on demand).

A :class:`FactorPanel` is everything the backtester needs: a cross-section of asset ``returns`` and
a point-in-time ``signal`` (the characteristic to rank on), aligned on the same dates/columns.

* :class:`SyntheticDataSource` — deterministic, seeded, with a *planted* per-factor premium so the
  pipeline can demonstrably recover a real signal (and an overfitting fixture can be constructed).
  This is the committed default → hermetic CI + offline demo.
* :class:`FrenchDataSource` — reads the git-ignored cache populated by ``make fetch-french`` (real
  Fama-French sorted-portfolio returns), never redistributed. Same interface, same pipeline.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# src/factorforge/backtest/data.py -> parents[3] == project root (projects/factorforge)
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"

# Planted per-factor monthly premia for the synthetic universe. Some factors carry a real premium
# (recoverable); ``size`` carries none (a natural "inconclusive / prone-to-overfit" case).
_SYNTHETIC_ALPHAS: dict[str, float] = {
    "value": 0.0050,
    "momentum": 0.0050,
    "quality": 0.0040,
    "low_volatility": 0.0022,
    "size": 0.0000,
}


@dataclass
class FactorPanel:
    """Aligned cross-sectional returns + ranking signal for one factor."""

    name: str
    returns: pd.DataFrame   # dates x assets, simple periodic returns
    signal: pd.DataFrame    # dates x assets, point-in-time ranking characteristic
    source: str
    description: str = ""


class DataSource(ABC):
    name: str = "base"

    @abstractmethod
    def available_factors(self) -> list[str]: ...

    @abstractmethod
    def load(self, factor: str) -> FactorPanel: ...


class SyntheticDataSource(DataSource):
    """Deterministic synthetic panel with a planted factor structure (seeded; CI-safe)."""

    name = "synthetic"

    def __init__(self, seed: int = 7, n_assets: int = 30, n_periods: int = 240) -> None:
        self.seed = seed
        self.n_assets = n_assets
        self.n_periods = n_periods
        self._cache: dict[str, FactorPanel] = {}

    def available_factors(self) -> list[str]:
        return sorted(_SYNTHETIC_ALPHAS)

    def load(self, factor: str) -> FactorPanel:
        if factor not in self._cache:
            self._cache[factor] = self._build(factor)
        return self._cache[factor]

    def _alpha(self, factor: str) -> float:
        # Known factors get their planted premium; anything else is pure noise (alpha 0) — useful
        # for "novel"/overfit candidates the critic should reject.
        return _SYNTHETIC_ALPHAS.get(factor, 0.0)

    def _seed_for(self, factor: str) -> int:
        # Stable across runs (Python's hash() is salted; hashlib is not).
        digest = hashlib.md5(factor.encode("utf-8")).hexdigest()
        return self.seed + int(digest[:6], 16)

    def _build(self, factor: str) -> FactorPanel:
        rng = np.random.default_rng(self._seed_for(factor))
        n, t = self.n_assets, self.n_periods
        dates = pd.date_range("2004-01-31", periods=t, freq="ME")
        assets = [f"A{i:02d}" for i in range(n)]

        # Static per-asset characteristic (standardized) → the true cross-sectional signal.
        char = rng.standard_normal(n)
        char = (char - char.mean()) / char.std(ddof=0)

        market = rng.normal(0.005, 0.04, size=t)          # common market factor
        betas = rng.uniform(0.8, 1.2, size=n)
        idio = rng.normal(0.0, 0.05, size=(t, n))         # idiosyncratic noise
        alpha = self._alpha(factor)

        # returns_{t,i} = market*beta_i + alpha * char_i + noise
        rets = market[:, None] * betas[None, :] + alpha * char[None, :] + idio
        returns = pd.DataFrame(rets, index=dates, columns=assets)

        # Observable signal: the characteristic, with a slow deterministic drift so weights (and
        # thus turnover) are non-trivial. Drift is small relative to the static spread.
        drift = np.cumsum(rng.normal(0.0, 0.02, size=(t, n)), axis=0)
        signal = pd.DataFrame(char[None, :] + drift, index=dates, columns=assets)

        return FactorPanel(
            name=factor,
            returns=returns,
            signal=signal,
            source="synthetic",
            description=f"Synthetic planted-signal panel (alpha={alpha:.4f}/mo, seed={self.seed}).",
        )


class FrenchDataSource(DataSource):
    """Real Fama-French sorted-portfolio returns from the git-ignored fetch cache.

    The cache is produced by ``make fetch-french`` (see :mod:`factorforge.backtest.fetch_french`)
    and is **never committed** (the Ken French Data Library is not redistributable). The pipeline
    is identical to the synthetic path — only the panel differs.
    """

    name = "french"

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR

    def _panel_path(self, factor: str) -> Path:
        return self.cache_dir / f"{factor}.csv"

    def available_factors(self) -> list[str]:
        if not self.cache_dir.exists():
            return []
        return sorted(p.stem for p in self.cache_dir.glob("*.csv"))

    def load(self, factor: str) -> FactorPanel:
        path = self._panel_path(factor)
        if not path.exists():
            raise FileNotFoundError(
                f"No cached French data for '{factor}' at {path}. Run `make fetch-french` first."
            )
        # Cached layout: a returns matrix (dates x portfolios) already scaled to decimals. The
        # ranking signal is the portfolio's ordinal position (1..k), broadcast across dates.
        returns = pd.read_csv(path, index_col=0, parse_dates=True)
        ranks = np.arange(1, returns.shape[1] + 1, dtype=float)
        signal = pd.DataFrame(
            np.tile(ranks, (returns.shape[0], 1)), index=returns.index, columns=returns.columns
        )
        return FactorPanel(
            name=factor, returns=returns, signal=signal, source="french",
            description=f"Ken French sorted-portfolio returns ({factor}); research/educational only.",
        )
