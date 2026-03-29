"""
Abstract base for vision providers and shared result types.
"""

from __future__ import annotations

import concurrent.futures
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Retry defaults
RETRY_MAX_ATTEMPTS = 3
RETRY_BASE_WAIT = 10  # seconds
RETRY_MAX_WAIT = 90   # seconds


def retry_api_call(fn, *, retries: int = RETRY_MAX_ATTEMPTS, retryable_exceptions: tuple = (), timeout: float | None = None):
    """Retry an API call with exponential backoff and jitter.

    Args:
        fn: Zero-arg callable that makes the API call.
        retries: Max number of attempts.
        retryable_exceptions: Tuple of exception types to retry on.
        timeout: Optional per-attempt timeout in seconds. If the call
                 exceeds this, a TimeoutError is raised (and retried).

    Returns:
        The return value of fn().

    Raises:
        The last exception if all retries are exhausted.
    """
    retries = max(1, retries)  # Ensure at least one attempt
    # Always treat TimeoutError as retryable when a timeout is set
    if timeout is not None:
        retryable_exceptions = (*retryable_exceptions, TimeoutError)
    last_exc = None
    for attempt in range(retries):
        try:
            if timeout is not None:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(fn)
                    return future.result(timeout=timeout)
            else:
                return fn()
        except retryable_exceptions as e:
            last_exc = e
            if attempt < retries - 1:
                wait = min(RETRY_BASE_WAIT * (2 ** attempt) + random.uniform(0, 3), RETRY_MAX_WAIT)
                logger.warning(f"API call failed ({type(e).__name__}), retrying in {wait:.1f}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
    raise last_exc


class CircuitBreaker:
    """Simple circuit breaker -- skips providers after repeated failures."""

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60):
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}

    def record_failure(self, provider: str):
        self._failures[provider] = self._failures.get(provider, 0) + 1
        if self._failures[provider] >= self._failure_threshold:
            self._open_until[provider] = time.time() + self._cooldown_seconds
            logger.warning(f"Circuit breaker OPEN for {provider} (cooldown {self._cooldown_seconds}s)")

    def record_success(self, provider: str):
        self._failures.pop(provider, None)
        self._open_until.pop(provider, None)

    def is_open(self, provider: str) -> bool:
        deadline = self._open_until.get(provider)
        if deadline is None:
            return False
        if time.time() >= deadline:
            # Cooldown expired -- half-open, allow one attempt
            self._open_until.pop(provider, None)
            self._failures[provider] = self._failure_threshold - 1
            return False
        return True


# Module-level circuit breaker instance
circuit_breaker = CircuitBreaker()


@dataclass
class ConsensusResult:
    """Result from a multi-model consensus read."""
    text: str
    confidence: float = 1.0
    confidence_level: str = "HIGH"  # "HIGH" or "LOW" — derived from confidence score
    provider_results: dict[str, str] = field(default_factory=dict)
    disagreements: list[str] = field(default_factory=list)
    strategy_used: str = "single"
    tokens_used: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class VisionProvider(Protocol):
    """Protocol that all vision providers must implement."""

    name: str

    @property
    def usage(self) -> dict:
        """Return token usage dict with 'input_tokens' and 'output_tokens'."""
        ...

    def read_image(
        self,
        image_b64: str,
        media_type: str,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> str:
        """Read a single image. Returns extracted text."""
        ...

    def read_batch(
        self,
        image_blocks: list[dict],
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 8192,
    ) -> str:
        """Read multiple images in one call. Returns extracted text."""
        ...

    @classmethod
    def is_available(cls) -> bool:
        """Check if the provider's SDK is installed and API key is set."""
        ...
