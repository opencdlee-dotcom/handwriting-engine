"""
Persistent writer profile store for cross-session handwriting calibration.

Saves writer-specific disambiguation observations (how they form 7s, 4s, a vs o,
etc.) to ~/.handwriting-engine/writer-profiles/{writer_id}.json and injects
them as a calibration block into transcription prompts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROFILES_DIR = Path.home() / ".handwriting-engine" / "writer-profiles"


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
