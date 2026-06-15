"""Append-only audit log — reproducibility + traceability for regulated-style workflows.

Every pipeline run records its stages, the evidence paths retrieved, the hypothesis, the
backtest/diagnostics numbers, and the verification outcome as JSON lines. The log is the paper
trail that makes a run auditable after the fact.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.factorforge.config import Settings


class AuditLog:
    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.audit_enabled
        self.path = Path(settings.audit_log_path)

    def record(self, event: str, data: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": datetime.now(UTC).isoformat(), "event": event, **data}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
