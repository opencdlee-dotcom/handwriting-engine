"""Tests for the per-process raw-read cache used by the sweep loop.

The cache lives next to ``_read_single`` / ``_read_consensus`` and is keyed
by ``(image_path, provider, prompt_hash, system_prompt_hash, auto_enhance,
inject_lessons, enhance_strategy, line_level, auto_retry)``. Hits return
the original text/confidence/latency but report 0 tokens (the work was
already paid by an earlier strategy in the sweep). Cache misses fall
through to ``read_page``.

These tests mock ``read_page`` directly so they exercise the cache layer
without hitting the provider stack.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

from handwriting_engine.benchmark.evaluate import _read_single, _read_consensus


@pytest.fixture
def img_path(tmp_path):
    p = tmp_path / "page.png"
    Image.new("RGB", (64, 64), color=(200, 200, 200)).save(p)
    return str(p)


def _patch_read_page(text="hello world"):
    """Patch read_page + provider stack so _read_single returns deterministically."""
    rp = patch("handwriting_engine.vision.read_page", return_value=text)
    # Make the provider lookup yield zero usage deltas — token counts come
    # from the mocked provider stack, not from read_page.
    fake_provider = MagicMock()
    fake_provider.usage = {"input_tokens": 0, "output_tokens": 0}
    gp = patch(
        "handwriting_engine.providers.get_provider",
        return_value=fake_provider,
    )
    return rp, gp


class TestReadCacheSingle:
    def test_second_call_is_cached(self, img_path):
        """Same (image, provider, prompt) twice -> second call is cached."""
        rp_patch, gp_patch = _patch_read_page("the quick brown fox")
        with rp_patch as mock_rp, gp_patch:
            cache: dict = {}
            r1 = _read_single(
                img_path, "gemini", "biology",
                read_cache=cache,
            )
            r2 = _read_single(
                img_path, "gemini", "biology",
                read_cache=cache,
            )
            # First call hit the provider
            assert mock_rp.call_count == 1, (
                f"read_page should be called exactly once for one cache key; "
                f"got {mock_rp.call_count}"
            )
            # First result is uncached
            assert r1.get("cached") is False
            # Second is cached and reports zero tokens
            assert r2.get("cached") is True
            assert r2["input_tokens"] == 0
            assert r2["output_tokens"] == 0
            assert r2["text"] == "the quick brown fox"

    def test_different_prompt_misses(self, img_path):
        """Distinct prompt arg => different cache entry => provider re-called."""
        rp_patch, gp_patch = _patch_read_page("foo")
        with rp_patch as mock_rp, gp_patch:
            cache: dict = {}
            _read_single(
                img_path, "gemini", "biology",
                prompt="prompt A", read_cache=cache,
            )
            _read_single(
                img_path, "gemini", "biology",
                prompt="prompt B", read_cache=cache,
            )
            assert mock_rp.call_count == 2, (
                "Different prompt strings must not collide in the cache"
            )
            # And the cache holds two entries.
            assert len(cache) == 2

    def test_different_flags_miss(self, img_path):
        """auto_enhance / line_level / auto_retry change keys -> miss."""
        rp_patch, gp_patch = _patch_read_page("baz")
        with rp_patch as mock_rp, gp_patch:
            cache: dict = {}
            _read_single(
                img_path, "gemini", "biology",
                line_level=False, read_cache=cache,
            )
            _read_single(
                img_path, "gemini", "biology",
                line_level=True, read_cache=cache,
            )
            assert mock_rp.call_count == 2

    def test_no_cache_means_no_caching(self, img_path):
        """read_cache=None preserves legacy behavior: every call re-reads."""
        rp_patch, gp_patch = _patch_read_page("legacy")
        with rp_patch as mock_rp, gp_patch:
            r1 = _read_single(img_path, "gemini", "biology")
            r2 = _read_single(img_path, "gemini", "biology")
            # Provider hit twice
            assert mock_rp.call_count == 2
            # Neither result claims cached=True (legacy result dicts may
            # carry cached=False or omit the key — both are acceptable as
            # long as no false-positive hit is signaled).
            assert not r1.get("cached", False)
            assert not r2.get("cached", False)

    def test_different_provider_misses(self, img_path):
        """Distinct provider names -> separate cache entries."""
        rp_patch, gp_patch = _patch_read_page("provider sensitive")
        with rp_patch as mock_rp, gp_patch:
            cache: dict = {}
            _read_single(img_path, "gemini", "biology", read_cache=cache)
            _read_single(img_path, "claude", "biology", read_cache=cache)
            assert mock_rp.call_count == 2


class TestReadCacheConsensus:
    """_read_consensus shares the cache; key uses joined provider list."""

    def test_consensus_second_call_cached(self, img_path):
        # Patch the consensus engine, not read_page.
        fake_result = MagicMock()
        fake_result.text = "consensus output"
        fake_result.confidence = 0.8
        fake_result.tokens_used = {"input_tokens": 100, "output_tokens": 50}
        with patch(
            "handwriting_engine.vision.read_with_consensus",
            return_value=fake_result,
        ) as mock_rc:
            cache: dict = {}
            r1 = _read_consensus(
                img_path, ["gemini", "claude"], "vote", "biology",
                read_cache=cache,
            )
            r2 = _read_consensus(
                img_path, ["gemini", "claude"], "vote", "biology",
                read_cache=cache,
            )
            assert mock_rc.call_count == 1
            assert r1.get("cached") is False
            assert r2.get("cached") is True
            assert r2["input_tokens"] == 0
            assert r2["output_tokens"] == 0
            assert r2["text"] == "consensus output"

    def test_consensus_different_strategy_misses(self, img_path):
        fake_result = MagicMock()
        fake_result.text = "x"
        fake_result.confidence = 0.7
        fake_result.tokens_used = {"input_tokens": 0, "output_tokens": 0}
        with patch(
            "handwriting_engine.vision.read_with_consensus",
            return_value=fake_result,
        ) as mock_rc:
            cache: dict = {}
            _read_consensus(
                img_path, ["gemini", "claude"], "vote", "biology",
                read_cache=cache,
            )
            _read_consensus(
                img_path, ["gemini", "claude"], "best_of", "biology",
                read_cache=cache,
            )
            assert mock_rc.call_count == 2
