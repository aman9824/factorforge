"""Fetch real Fama-French sorted-portfolio returns into the git-ignored cache (local only).

Run via ``make fetch-french`` / ``factorforge fetch-french``. Requires the optional ``[french]``
extra (``pandas-datareader``) and network access to the Ken French Data Library. The data is
**never committed or redistributed** — it is downloaded on demand into ``data/cache/``.

We own two conversions ``pandas_datareader`` does NOT do: coding ``-99.99`` / ``-999`` as missing,
and dividing percent returns by 100. The signal for ranking is the decile's ordinal position.

Source: Kenneth R. French – Data Library, Tuck School of Business, Dartmouth College
(© Eugene F. Fama & Kenneth R. French). FactorForge claims no ownership of this data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from factorforge.backtest.data import DEFAULT_CACHE_DIR
from factorforge.logging import get_logger

log = get_logger(__name__)

# Friendly factor name -> Ken French dataset id (decile portfolios sorted on the characteristic).
FRENCH_DATASETS: dict[str, str] = {
    "value": "Portfolios_Formed_on_BE-ME",
    "size": "Portfolios_Formed_on_ME",
    "momentum": "10_Portfolios_Prior_12_2",
}
_MISSING = [-99.99, -999.0, -99.0]


def fetch_french(
    factors: dict[str, str] | None = None, cache_dir: Path | None = None
) -> list[str]:
    """Download, clean, and cache the configured French portfolio datasets. Returns written paths."""
    from pandas_datareader.data import DataReader  # optional dep, imported on use

    datasets = factors or FRENCH_DATASETS
    out_dir = cache_dir or DEFAULT_CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for factor, dataset_id in datasets.items():
        log.info("french.fetch", factor=factor, dataset=dataset_id)
        bundle = DataReader(dataset_id, "famafrench")
        df = bundle[0].copy()                       # [0] = monthly value-weighted returns
        df = df.replace(_MISSING, np.nan) / 100.0   # we own the NaN coding + percent scaling
        if hasattr(df.index, "to_timestamp"):       # PeriodIndex -> month-end timestamps
            df.index = df.index.to_timestamp(how="end").normalize()
        if df.shape[1] > 10:                         # keep the trailing decile portfolios
            df = df.iloc[:, -10:]
        df = df.dropna(how="all")
        path = out_dir / f"{factor}.csv"
        df.to_csv(path)
        written.append(str(path))
        log.info("french.cached", factor=factor, rows=int(df.shape[0]), cols=int(df.shape[1]), path=str(path))
    return written


def main() -> None:
    from factorforge.config import get_settings
    from factorforge.logging import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    paths = fetch_french()
    print(f"Cached {len(paths)} French datasets:")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
