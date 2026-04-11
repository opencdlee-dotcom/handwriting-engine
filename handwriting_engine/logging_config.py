"""
Structured logging configuration for the handwriting engine.

Provides JSON-format logging for production use (API calls, consensus
decisions, enhancement operations) and human-readable format for
development.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class StructuredFormatter(logging.Formatter):
    """JSON log formatter for structured log analysis."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge extra structured fields if present
        if hasattr(record, "structured_data"):
            log_entry.update(record.structured_data)
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


def configure_logging(
    level: str = "INFO",
    format: str = "text",
    stream=None,
) -> None:
    """Configure logging for the handwriting engine.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        format: "json" for structured JSON output, "text" for human-readable.
        stream: Output stream (defaults to stderr).
    """
    root_logger = logging.getLogger("handwriting_engine")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    handler = logging.StreamHandler(stream or sys.stderr)
    if format == "json":
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        ))

    root_logger.addHandler(handler)


def log_api_call(
    logger: logging.Logger,
    provider: str,
    latency_ms: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    success: bool = True,
    error: str = "",
    cache_hit: bool = False,
):
    """Log a structured API call event."""
    data = {
        "event": "api_call",
        "provider": provider,
        "latency_ms": round(latency_ms, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "success": success,
        "cache_hit": cache_hit,
    }
    if error:
        data["error"] = error

    record = logger.makeRecord(
        logger.name, logging.INFO if success else logging.WARNING,
        "", 0, f"API call to {provider} ({'cached' if cache_hit else f'{latency_ms:.0f}ms'})",
        (), None,
    )
    record.structured_data = data
    logger.handle(record)


def log_consensus_decision(
    logger: logging.Logger,
    strategy: str,
    providers: list[str],
    confidence: float,
    confidence_level: str,
    disagreement_count: int = 0,
):
    """Log a structured consensus decision event."""
    data = {
        "event": "consensus",
        "strategy": strategy,
        "providers": providers,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "disagreements": disagreement_count,
    }
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0,
        f"Consensus ({strategy}): confidence={confidence:.2f} ({confidence_level})",
        (), None,
    )
    record.structured_data = data
    logger.handle(record)


class Timer:
    """Context manager for timing operations."""

    def __init__(self):
        self.elapsed_ms: float = 0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
