"""The eval gate must pass on the deterministic mock backend (this is what CI runs)."""

from __future__ import annotations

from evals.run_evals import main


def test_eval_gate_passes() -> None:
    assert main() == 0
