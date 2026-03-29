"""Tests for the vision API result cache."""

import pytest

from handwriting_engine.providers.cache import VisionCache


@pytest.fixture
def cache(tmp_path):
    return VisionCache(cache_dir=tmp_path / "cache")


class TestVisionCache:
    def test_miss_returns_none(self, cache):
        result = cache.get("claude", "img123", "read this", "system")
        assert result is None

    def test_put_and_get(self, cache):
        cache.put("claude", "img123", "read this", "system", "Hello World")
        result = cache.get("claude", "img123", "read this", "system")
        assert result == "Hello World"

    def test_different_provider_is_different_key(self, cache):
        cache.put("claude", "img123", "read this", "system", "Claude result")
        cache.put("gemini", "img123", "read this", "system", "Gemini result")
        assert cache.get("claude", "img123", "read this", "system") == "Claude result"
        assert cache.get("gemini", "img123", "read this", "system") == "Gemini result"

    def test_different_prompt_is_different_key(self, cache):
        cache.put("claude", "img123", "prompt A", "", "Result A")
        cache.put("claude", "img123", "prompt B", "", "Result B")
        assert cache.get("claude", "img123", "prompt A", "") == "Result A"
        assert cache.get("claude", "img123", "prompt B", "") == "Result B"

    def test_clear(self, cache):
        cache.put("claude", "img123", "p", "s", "result")
        cache.clear()
        assert cache.get("claude", "img123", "p", "s") is None

    def test_stats(self, cache):
        stats = cache.stats()
        assert stats["entries"] == 0
        assert stats["total_hits"] == 0

        cache.put("claude", "img123", "p", "s", "result")
        cache.get("claude", "img123", "p", "s")
        cache.get("claude", "img123", "p", "s")

        stats = cache.stats()
        assert stats["entries"] == 1
        assert stats["total_hits"] == 2

    def test_empty_result_not_cached(self, cache):
        cache.put("claude", "img123", "p", "s", "")
        assert cache.get("claude", "img123", "p", "s") is None

    def test_disabled_cache(self, tmp_path):
        c = VisionCache(cache_dir=tmp_path, enabled=False)
        c.put("claude", "img123", "p", "s", "result")
        assert c.get("claude", "img123", "p", "s") is None
        assert c.stats()["enabled"] is False
