"""
Persistent writer profile store for cross-session handwriting calibration.

Saves writer-specific disambiguation observations (how they form 7s, 4s, a vs o,
etc.) to ~/.handwriting-engine/writer-profiles/{writer_id}.json and injects
them as a calibration block into transcription prompts.

Also serves as the home for S2 per-writer few-shot exemplar selection
(``select_exemplars``), since exemplars are conceptually a richer flavor of
the same writer-calibration mechanism.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROFILES_DIR = Path.home() / ".handwriting-engine" / "writer-profiles"


@dataclass(frozen=True)
class Exemplar:
    """One (sample, ground-truth) pair to inject as a few-shot exemplar."""

    sample_id: int
    image_path: str
    ground_truth: str


def select_exemplars(
    writer_id: str,
    *,
    k: int = 3,
    exclude_sample_id: Optional[int] = None,
    db_path: Optional[str | Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> list[Exemplar]:
    """Return up to ``k`` deterministic exemplars for ``writer_id``.

    Strategy v0 per S2-SPEC § Design:
    1. Pull every (sample, ground_truth) pair where ``samples.student =
       writer_id`` (latest GT per sample wins on ties).
    2. Order by ``quality_assessments.score DESC`` when available (joined
       LEFT so writers without quality data still produce results), then by
       ``samples.id ASC`` for determinism.
    3. Slice ``min(k, len)`` off the top.

    Returns ``[]`` for empty writers, ``k <= 0``, or when no eligible
    sample/GT pair exists. Existence of the image on disk is *not* checked
    here — callers that need that guarantee filter post hoc.

    Either ``conn`` or ``db_path`` may be supplied. With neither, defaults
    to ``DEFAULT_DB_PATH`` from ``benchmark.db``.
    """
    if not writer_id or k <= 0:
        return []

    owns_conn = conn is None
    if owns_conn:
        from handwriting_engine.benchmark.db import get_connection

        conn = get_connection(db_path)
    try:
        # Latest GT per sample: ROW_NUMBER would be ideal but SQLite versions
        # bundled with older Pythons lack window functions reliably; use a
        # correlated subquery instead.
        params: list = [writer_id]
        exclude_sql = ""
        if exclude_sample_id is not None:
            exclude_sql = "AND s.id != ?"
            params.append(exclude_sample_id)

        # quality_assessments may not have a row for every sample — LEFT JOIN
        # so missing scores fall to the bottom (NULL sorted last via COALESCE).
        sql = f"""
            SELECT
                s.id          AS sample_id,
                s.image_path  AS image_path,
                gt.text       AS ground_truth
            FROM samples AS s
            JOIN ground_truths AS gt
                ON gt.sample_id = s.id
                AND gt.id = (
                    SELECT MAX(id) FROM ground_truths WHERE sample_id = s.id
                )
            LEFT JOIN quality_assessments AS qa
                ON qa.sample_id = s.id
            WHERE s.student = ?
              {exclude_sql}
            ORDER BY
                COALESCE(qa.contrast_score, -1) DESC,
                s.id ASC
            LIMIT ?
        """
        params.append(k)
        rows = conn.execute(sql, params).fetchall()
    finally:
        if owns_conn:
            conn.close()

    return [
        Exemplar(
            sample_id=row["sample_id"],
            image_path=row["image_path"],
            ground_truth=row["ground_truth"],
        )
        for row in rows
    ]


class WriterProfileStore:
    """Load, save, and inject writer-specific handwriting profiles."""

    def __init__(self, profiles_dir: Path | None = None):
        self._dir = profiles_dir or _PROFILES_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, writer_id: str) -> Path:
        safe_id = "".join(c for c in writer_id if c.isalnum() or c in "-_")
        return self._dir / f"{safe_id}.json"

    def save(self, writer_id: str, profile: dict) -> Path:
        """Save a writer profile dict to disk."""
        path = self._path(writer_id)
        with open(path, "w") as f:
            json.dump({"writer_id": writer_id, **profile}, f, indent=2)
        logger.info("Writer profile saved: %s", path)
        return path

    def load(self, writer_id: str) -> dict | None:
        """Load a writer profile. Returns None if not found."""
        path = self._path(writer_id)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load writer profile %s: %s", writer_id, e)
            return None

    def delete(self, writer_id: str) -> bool:
        """Delete a writer profile. Returns True if deleted."""
        path = self._path(writer_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_writers(self) -> list[str]:
        """List all writer IDs with saved profiles."""
        return [p.stem for p in self._dir.glob("*.json")]

    def build_calibration_block(self, profile: dict) -> str:
        """Convert a writer profile dict into a prompt calibration string."""
        if not profile:
            return ""

        lines = ["=== WRITER-SPECIFIC CALIBRATION (from prior sessions) ==="]

        if profile.get("crosses_sevens") is not None:
            val = "YES — always cross 7s" if profile["crosses_sevens"] else "NO — uncrossed 7s"
            lines.append(f"- 7s: {val}")

        if profile.get("open_four") is not None:
            val = "OPEN top (like an upside-down h)" if profile["open_four"] else "CLOSED top"
            lines.append(f"- 4s: {val}")

        if profile.get("a_style"):
            lines.append(f"- Letter 'a': {profile['a_style']}")

        if profile.get("connects_letters") is not None:
            val = "cursive (connected)" if profile["connects_letters"] else "printed (separate)"
            lines.append(f"- Writing style: {val}")

        if profile.get("zero_vs_oh"):
            lines.append(f"- 0 vs O: {profile['zero_vs_oh']}")

        if profile.get("confusion_resolutions"):
            lines.append("- Known character resolutions for this writer:")
            for pair, resolution in profile["confusion_resolutions"].items():
                lines.append(f"  • {pair}: always reads as '{resolution}'")

        lines.append("Apply these observations consistently to ALL text in this image.")
        return "\n".join(lines)
