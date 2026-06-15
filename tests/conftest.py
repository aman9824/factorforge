"""Shared test fixtures.

Forces a headless matplotlib backend before any test imports ``bt``/``ffn`` (which import
``matplotlib.pyplot`` at module load), and isolates the cached settings singleton so tests that
tweak the environment don't leak into each other.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from collections.abc import Iterator  # noqa: E402

import pytest  # noqa: E402

from factorforge.config import get_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
