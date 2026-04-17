"""
Vision API result cache — avoids re-reading identical images.

Keys on (provider, image_hash, prompt_hash, system_prompt_hash).
Uses SQLite for persistence across sessions. Thread-safe.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".handwriting-engine" / "cache"
_MAX_ENTRIES = 10_000


def _cache_enabled_by_env() -> bool:
    """Respect HE_CACHE_ENABLED=0/false to disable the cache globally."""
    val = os.getenv("HE_CACHE_ENABLED", "1").lower()
    return val not in ("0", "false", "no", "off")


class VisionCache:
    """Persistent cache for vision API results."""

    def __init__(self, cache_dir: Path | str | None = None, enabled: bool | None = None):
        # When `enabled` is not passed (module-level default instance), consult the
        # HE_CACHE_ENABLED env var so tests and CI can disable the process-wide cache.
        # When `enabled` is passed explicitly, respect the caller's choice.
        if enabled is None:
            self._enabled = _cache_enabled_by_env()
        else:
            self._enabled = enabled
        if not self._enabled:
            return

        self._dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / "vision_cache.db"
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        cache_key TEXT PRIMARY KEY,
                        provider TEXT NOT NULL,
                        result TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        hit_count INTEGER DEFAULT 0
                    )
                """)
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _make_key(
        provider: str,
        image_b64: str,
        prompt: str,
        system_prompt: str,
        config_fp: str = "",
    ) -> str:
        """Deterministic cache key from inputs.

        `config_fp` should capture model, temperature, and max_tokens so that
        two runs with different configs don't collide on the same image+prompt.
        """
        # Hash image data (first 1000 + last 1000 chars for speed on large images)
        img_sample = image_b64[:1000] + image_b64[-1000:] if len(image_b64) > 2000 else image_b64
        raw = f"{provider}|{img_sample}|{len(image_b64)}|{prompt}|{system_prompt}|{config_fp}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(
        self,
        provider: str,
        image_b64: str,
        prompt: str,
        system_prompt: str,
        config_fp: str = "",
    ) -> str | None:
        """Look up a cached result. Returns None on miss."""
        if not self._enabled:
            return None

        key = self._make_key(provider, image_b64, prompt, system_prompt, config_fp)
        conn = None
        with self._lock:
            try:
                conn = sqlite3.connect(str(self._db_path))
                row = conn.execute(
                    "SELECT result FROM cache WHERE cache_key = ?", (key,)
                ).fetchone()
                if row:
                    conn.execute(
                        "UPDATE cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                        (key,),
                    )
                    conn.commit()
                    logger.debug(f"Cache HIT for {provider}")
                    return row[0]
            except sqlite3.Error:
                pass
            finally:
                if conn:
                    conn.close()
        return None

    def put(
        self,
        provider: str,
        image_b64: str,
        prompt: str,
        system_prompt: str,
        result: str,
        config_fp: str = "",
    ):
        """Store a result in the cache."""
        if not self._enabled or not result:
            return

        import time

        key = self._make_key(provider, image_b64, prompt, system_prompt, config_fp)
        with self._lock:
            try:
                conn = sqlite3.connect(str(self._db_path))
                conn.execute(
                    "INSERT OR REPLACE INTO cache (cache_key, provider, result, created_at) VALUES (?, ?, ?, ?)",
                    (key, provider, result, time.time()),
                )
                # Evict oldest entries if over limit
                count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
                if count > _MAX_ENTRIES:
                    conn.execute(
                        "DELETE FROM cache WHERE cache_key IN "
                        "(SELECT cache_key FROM cache ORDER BY created_at ASC LIMIT ?)",
                        (count - _MAX_ENTRIES,),
                    )
                conn.commit()
            except sqlite3.Error as e:
                logger.warning(f"Cache write failed: {e}")
            finally:
                conn.close()

    def clear(self):
        """Clear all cached results."""
        if not self._enabled:
            return
        with self._lock:
            try:
                conn = sqlite3.connect(str(self._db_path))
                conn.execute("DELETE FROM cache")
                conn.commit()
            except sqlite3.Error:
                pass
            finally:
                conn.close()

    def stats(self) -> dict:
        """Return cache statistics."""
        if not self._enabled:
            return {"enabled": False, "entries": 0, "total_hits": 0}
        with self._lock:
            try:
                conn = sqlite3.connect(str(self._db_path))
                count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
                hits = conn.execute("SELECT COALESCE(SUM(hit_count), 0) FROM cache").fetchone()[0]
                return {"enabled": True, "entries": count, "total_hits": hits}
            except sqlite3.Error:
                return {"enabled": True, "entries": 0, "total_hits": 0}
            finally:
                conn.close()


# Module-level cache instance
vision_cache = VisionCache()


def _config_fingerprint(provider_obj, max_tokens: int) -> str:
    """Build a stable fingerprint from provider config that affects outputs."""
    model = getattr(provider_obj, "_model", "") or ""
    temperature = getattr(provider_obj, "_temperature", None)
    thinking_budget = getattr(provider_obj, "_thinking_budget", None)
    parts = [
        f"model={model}",
        f"max_tokens={max_tokens}",
        f"temperature={temperature}",
        f"thinking_budget={thinking_budget}",
    ]
    return "|".join(parts)


def cached_read_image(
    provider_obj,
    image_b64: str,
    media_type: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
) -> str:
    """Provider-agnostic read wrapper that consults VisionCache before calling.

    Returns cached text on hit; otherwise calls `provider_obj.read_image(...)`
    and stores the result. Errors and empty strings are not cached.
    """
    provider_name = getattr(provider_obj, "name", type(provider_obj).__name__)
    config_fp = _config_fingerprint(provider_obj, max_tokens)

    cached = vision_cache.get(provider_name, image_b64, prompt, system_prompt, config_fp)
    if cached is not None:
        return cached

    result = provider_obj.read_image(image_b64, media_type, prompt, system_prompt, max_tokens)
    if result:
        vision_cache.put(provider_name, image_b64, prompt, system_prompt, result, config_fp)
    return result
