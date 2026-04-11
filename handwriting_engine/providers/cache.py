"""
Vision API result cache — avoids re-reading identical images.

Keys on (provider, image_hash, prompt_hash, system_prompt_hash).
Uses SQLite for persistence across sessions. Thread-safe.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".handwriting-engine" / "cache"
_MAX_ENTRIES = 10_000


class VisionCache:
    """Persistent cache for vision API results."""

    def __init__(self, cache_dir: Path | str | None = None, enabled: bool = True):
        self._enabled = enabled
        if not enabled:
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
    def _make_key(provider: str, image_b64: str, prompt: str, system_prompt: str) -> str:
        """Deterministic cache key from inputs."""
        # Hash image data (first 1000 + last 1000 chars for speed on large images)
        img_sample = image_b64[:1000] + image_b64[-1000:] if len(image_b64) > 2000 else image_b64
        raw = f"{provider}|{img_sample}|{len(image_b64)}|{prompt}|{system_prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, provider: str, image_b64: str, prompt: str, system_prompt: str) -> str | None:
        """Look up a cached result. Returns None on miss."""
        if not self._enabled:
            return None

        key = self._make_key(provider, image_b64, prompt, system_prompt)
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

    def put(self, provider: str, image_b64: str, prompt: str, system_prompt: str, result: str):
        """Store a result in the cache."""
        if not self._enabled or not result:
            return

        import time

        key = self._make_key(provider, image_b64, prompt, system_prompt)
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
