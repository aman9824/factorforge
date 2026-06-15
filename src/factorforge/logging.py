"""Structured logging via ``structlog``.

One ``configure_logging`` call wires structlog to render either human-friendly console lines
(default) or JSON (``FF_LOG_JSON=1``, suitable for log aggregation / observability). Everything
else in the codebase just calls ``get_logger(__name__)`` and logs key/value pairs.
"""

from __future__ import annotations

import logging
from typing import Any

import structlog


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    """Configure structlog + stdlib logging once, at process start."""
    logging.basicConfig(format="%(message)s", level=getattr(logging, level.upper(), logging.INFO))

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    processors.append(
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None, **initial: Any) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, optionally with pre-bound context."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    if initial:
        logger = logger.bind(**initial)
    return logger
