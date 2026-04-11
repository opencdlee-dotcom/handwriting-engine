<plan phase="4" index="01" requirement="REQ-006,REQ-007">
  <objective>Add Sauvola adaptive binarization to enhance.py and implement WriterProfileStore for persistent writer calibration</objective>

  <files>
    <modify>handwriting_engine/enhance.py</modify>
    <create>handwriting_engine/writer_profile_store.py</create>
    <modify>handwriting_engine/handwriting.py</modify>
    <modify>tests/test_enhance.py</modify>
    <create>tests/test_writer_profile_store.py</create>
  </files>

  <tasks>
    <task type="auto">
      <name>Add sauvola_enhance() to enhance.py</name>
      <action>
In ~/Developer/handwriting-engine/handwriting_engine/enhance.py:

1. Add sauvola_enhance() function after clahe_enhance():

def sauvola_enhance(
    image_path: str,
    output_path: Optional[str] = None,
    window_size: int = 25,
    k: float = 0.2,
) -> str:
    """Sauvola adaptive binarization — best for degraded docs with uneven illumination.

    MDPI Electronics 2024 review: adaptive local methods (Sauvola, Niblack)
    outperform global Otsu for documents with non-uniform backgrounds.

    Args:
        window_size: Local window size for threshold computation (odd number).
        k: Sauvola sensitivity parameter (0.1 = less aggressive, 0.5 = more).

    Falls back to clahe_enhance() if scikit-image is not installed.
    """
    try:
        from skimage.filters import threshold_sauvola
    except ImportError:
        logger.warning("scikit-image not installed, falling back to clahe_enhance")
        return clahe_enhance(image_path, output_path=output_path)

    img = Image.open(image_path)
    img_rgb, alpha = _strip_alpha(img)

    # Convert to grayscale for Sauvola
    gray = np.array(img_rgb.convert("L"))

    # Compute Sauvola threshold map
    thresh = threshold_sauvola(gray, window_size=window_size, k=k)
    binary = (gray > thresh).astype(np.uint8) * 255

    # Convert binary back to RGB
    result_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    result = Image.fromarray(result_rgb)

    result = _restore_alpha(result, alpha)
    dest = output_path or image_path
    _save(result, dest)
    return dest

2. In enhance_image(), add "sauvola" to the strategy dispatch before the final raise ValueError:

    if strategy == "sauvola":
        return sauvola_enhance(image_path, output_path=output_path)

3. Update the ValueError message to include "sauvola" in the list.
      </action>
      <verify>python3 -c "from handwriting_engine.enhance import sauvola_enhance; print('OK')"</verify>
      <done>sauvola_enhance() importable; enhance_image(strategy='sauvola') dispatches to it</done>
    </task>

    <task type="auto">
      <name>Add test_sauvola to tests/test_enhance.py</name>
      <action>
In ~/Developer/handwriting-engine/tests/test_enhance.py, append:

def test_sauvola_enhance_returns_path(tmp_path):
    """sauvola_enhance should return a path without crashing."""
    from handwriting_engine.enhance import sauvola_enhance, enhance_image
    img = Image.new("RGB", (200, 100), color=(200, 200, 200))
    path = str(tmp_path / "test.jpg")
    img.save(path)
    result = sauvola_enhance(path)
    assert result == path

def test_enhance_image_sauvola_strategy(tmp_path):
    from handwriting_engine.enhance import enhance_image
    img = Image.new("RGB", (200, 100), color=(200, 200, 200))
    path = str(tmp_path / "test.jpg")
    img.save(path)
    result = enhance_image(path, strategy="sauvola")
    assert result == path

Make sure the test file imports Image from PIL at the top if not already there.
      </action>
      <verify>python3 -m pytest tests/test_enhance.py -x -q 2>&1 | tail -3</verify>
      <done>All enhance tests pass including new sauvola tests</done>
    </task>

    <task type="auto">
      <name>Create writer_profile_store.py</name>
      <action>
Create ~/Developer/handwriting-engine/handwriting_engine/writer_profile_store.py:

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
        """Save a writer profile dict to disk.

        Args:
            writer_id: Unique identifier for this writer.
            profile: Dict with keys like 'crosses_sevens', 'open_four',
                     'a_style', 'connects_letters', 'confusion_resolutions'.

        Returns:
            Path to the saved profile file.
        """
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
        """Convert a writer profile dict into a prompt calibration string.

        This string can be injected into the WRITER_CALIBRATION section of
        a transcription prompt to give the model writer-specific context.
        """
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
      </action>
      <verify>python3 -c "from handwriting_engine.writer_profile_store import WriterProfileStore; print('OK')"</verify>
      <done>WriterProfileStore importable with save/load/build_calibration_block methods</done>
    </task>

    <task type="auto">
      <name>Create tests/test_writer_profile_store.py and commit</name>
      <action>
Create ~/Developer/handwriting-engine/tests/test_writer_profile_store.py:

"""Tests for WriterProfileStore."""
import pytest
from handwriting_engine.writer_profile_store import WriterProfileStore


def test_save_and_load_round_trip(tmp_path):
    store = WriterProfileStore(profiles_dir=tmp_path)
    profile = {"crosses_sevens": True, "open_four": False, "connects_letters": True}
    store.save("test_writer", profile)
    loaded = store.load("test_writer")
    assert loaded["crosses_sevens"] is True
    assert loaded["open_four"] is False
    assert loaded["writer_id"] == "test_writer"


def test_load_nonexistent_returns_none(tmp_path):
    store = WriterProfileStore(profiles_dir=tmp_path)
    assert store.load("no_such_writer") is None


def test_delete(tmp_path):
    store = WriterProfileStore(profiles_dir=tmp_path)
    store.save("writer_a", {"crosses_sevens": False})
    assert store.delete("writer_a") is True
    assert store.load("writer_a") is None
    assert store.delete("writer_a") is False


def test_list_writers(tmp_path):
    store = WriterProfileStore(profiles_dir=tmp_path)
    store.save("alice", {"crosses_sevens": True})
    store.save("bob", {"crosses_sevens": False})
    writers = store.list_writers()
    assert "alice" in writers
    assert "bob" in writers


def test_build_calibration_block_with_full_profile(tmp_path):
    store = WriterProfileStore(profiles_dir=tmp_path)
    profile = {
        "crosses_sevens": True,
        "open_four": True,
        "connects_letters": False,
        "confusion_resolutions": {"u/v": "v", "1/l": "1"},
    }
    block = store.build_calibration_block(profile)
    assert "7s" in block
    assert "4s" in block
    assert "printed" in block
    assert "u/v" in block
    assert "v" in block


def test_build_calibration_block_empty(tmp_path):
    store = WriterProfileStore(profiles_dir=tmp_path)
    assert store.build_calibration_block({}) == ""


def test_writer_id_sanitization(tmp_path):
    store = WriterProfileStore(profiles_dir=tmp_path)
    store.save("writer/with/slashes", {"crosses_sevens": True})
    # Slashes are stripped — should not create subdirectories
    writers = store.list_writers()
    assert any("writer" in w for w in writers)

Then run: python3 -m pytest tests/test_writer_profile_store.py tests/test_enhance.py -q
Then commit all Phase 4 changes.
      </action>
      <verify>python3 -m pytest tests/test_writer_profile_store.py tests/test_enhance.py -q 2>&1 | tail -4</verify>
      <done>All Phase 4 tests pass</done>
    </task>
  </tasks>

  <dependencies>none</dependencies>
  <commit_message>feat(phase-4): Sauvola binarization and WriterProfileStore (REQ-006, REQ-007)</commit_message>
</plan>
