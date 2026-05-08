"""Tests for the token-reduction + accuracy fixes in vision.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Task 2: Gemini context cache dedup
# ---------------------------------------------------------------------------


def test_gemini_context_cache_enabled_once_per_unique_key():
    """`enable_context_cache` is invoked once per unique cache_key but skipped
    on second call with the same key.
    """
    from handwriting_engine import vision

    # Reset module-level dedup set so test is order-independent
    vision._gemini_cache_keys.clear()

    fake_provider = MagicMock()
    fake_provider.enable_context_cache = MagicMock()

    long_prompt = "X" * 5000  # >= 4096-char gate
    cache_key = "abc123"

    with patch.object(vision, "get_provider", return_value=fake_provider):
        vision._maybe_enable_gemini_cache(long_prompt, cache_key, "gemini")
        vision._maybe_enable_gemini_cache(long_prompt, cache_key, "gemini")
        vision._maybe_enable_gemini_cache(long_prompt, cache_key, "gemini")

    # Despite 3 calls, the underlying provider only sees ONE
    assert fake_provider.enable_context_cache.call_count == 1
    fake_provider.enable_context_cache.assert_called_with(long_prompt)


def test_gemini_context_cache_keyed_per_unique_key():
    """Two different cache keys each enable caching once."""
    from handwriting_engine import vision

    vision._gemini_cache_keys.clear()

    fake_provider = MagicMock()
    fake_provider.enable_context_cache = MagicMock()

    long_prompt = "X" * 5000

    with patch.object(vision, "get_provider", return_value=fake_provider):
        vision._maybe_enable_gemini_cache(long_prompt, "key-A", "gemini")
        vision._maybe_enable_gemini_cache(long_prompt, "key-B", "gemini")
        # Repeats — should be no-ops
        vision._maybe_enable_gemini_cache(long_prompt, "key-A", "gemini")
        vision._maybe_enable_gemini_cache(long_prompt, "key-B", "gemini")

    assert fake_provider.enable_context_cache.call_count == 2


def test_gemini_context_cache_skipped_when_provider_not_gemini():
    """Non-gemini providers should not trigger the cache path."""
    from handwriting_engine import vision

    vision._gemini_cache_keys.clear()
    fake_provider = MagicMock()
    fake_provider.enable_context_cache = MagicMock()

    with patch.object(vision, "get_provider", return_value=fake_provider):
        vision._maybe_enable_gemini_cache("X" * 5000, "k", "claude")
        vision._maybe_enable_gemini_cache("X" * 5000, "k", "openai")

    fake_provider.enable_context_cache.assert_not_called()


def test_gemini_context_cache_skipped_for_short_prompts():
    """Below the min-token gate, no cache call is made (Gemini would reject)."""
    from handwriting_engine import vision

    vision._gemini_cache_keys.clear()
    fake_provider = MagicMock()
    fake_provider.enable_context_cache = MagicMock()

    with patch.object(vision, "get_provider", return_value=fake_provider):
        # 100 chars << 4096 gate
        vision._maybe_enable_gemini_cache("short prompt", "k", "gemini")

    fake_provider.enable_context_cache.assert_not_called()


def test_gemini_context_cache_consensus_provider_list():
    """When providers_used is a list (consensus path), gemini-presence triggers caching."""
    from handwriting_engine import vision

    vision._gemini_cache_keys.clear()
    fake_provider = MagicMock()
    fake_provider.enable_context_cache = MagicMock()

    with patch.object(vision, "get_provider", return_value=fake_provider):
        vision._maybe_enable_gemini_cache(
            "X" * 5000, "list-key", ["claude", "gemini", "openai"]
        )

    fake_provider.enable_context_cache.assert_called_once()
