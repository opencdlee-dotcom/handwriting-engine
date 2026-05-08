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


# ---------------------------------------------------------------------------
# Task 3: persistent vocab-priming cache (hash → JSON)
# ---------------------------------------------------------------------------


def test_vocab_hints_cached_to_disk_and_reused(tmp_path, monkeypatch):
    """Second call with the same image b64 must NOT invoke the provider."""
    from handwriting_engine import vision

    # Redirect cache dir to tmp_path so we don't pollute ~/.handwriting-engine
    monkeypatch.setattr(vision, "_VOCAB_CACHE_DIR", tmp_path / "vocab-cache")

    call_counter = {"n": 0}

    def fake_read_image(*args, **kwargs):
        call_counter["n"] += 1
        return "alpha, beta, gamma"

    fake_provider = MagicMock()
    fake_provider.read_image = MagicMock(side_effect=fake_read_image)

    monkeypatch.setattr(vision, "available_providers", lambda: ["gemini"])
    monkeypatch.setattr(vision, "get_provider", lambda name: fake_provider)

    img_b64 = "ZmFrZS1pbWFnZS1ieXRlcw=="  # "fake-image-bytes" base64

    first = vision._extract_vocabulary_hints(img_b64, "image/jpeg")
    assert first == ["alpha", "beta", "gamma"]
    assert call_counter["n"] == 1

    # Second call: should hit disk cache
    second = vision._extract_vocabulary_hints(img_b64, "image/jpeg")
    assert second == ["alpha", "beta", "gamma"]
    assert call_counter["n"] == 1, "provider must not be called on cache hit"

    # Different image: should miss and call provider again
    third = vision._extract_vocabulary_hints("b3RoZXItaW1hZ2U=", "image/jpeg")
    assert third == ["alpha", "beta", "gamma"]
    assert call_counter["n"] == 2


def test_vocab_hints_cache_survives_corrupt_file(tmp_path, monkeypatch):
    """A malformed cache JSON should fall through to a fresh provider call."""
    import hashlib

    from handwriting_engine import vision

    cache_dir = tmp_path / "vocab-cache"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(vision, "_VOCAB_CACHE_DIR", cache_dir)

    img_b64 = "Y29ycnVwdC1jYWNoZQ=="
    digest = hashlib.sha256(img_b64.encode("utf-8")).hexdigest()
    (cache_dir / f"{digest}.json").write_text("{not json")

    fake_provider = MagicMock()
    fake_provider.read_image = MagicMock(return_value="alpha, beta, gamma")
    monkeypatch.setattr(vision, "available_providers", lambda: ["gemini"])
    monkeypatch.setattr(vision, "get_provider", lambda name: fake_provider)

    out = vision._extract_vocabulary_hints(img_b64, "image/jpeg")
    assert out == ["alpha", "beta", "gamma"]
    fake_provider.read_image.assert_called_once()
