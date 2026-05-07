"""Phase 9 / RPT-03 — guided lab notebook ingest.

Covers:
- Walks an image directory; new images get inserted with category='lab'.
- prompt_fn output is stored as ground_truth, source='lab-grader'.
- Empty prompt_fn return = user-skip; sample stays without GT.
- Resume: re-running skips images that already have ground truth.
- Counts dict reflects each disposition.
- CLI surface invokes ingest_lab and reports the counts.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
from click.testing import CliRunner

from handwriting_engine.benchmark.db import (
    get_connection,
    get_sample_by_hash,
)
from handwriting_engine.benchmark.ingest import hash_file, ingest_lab
from handwriting_engine.cli import cli


def _make_image_dir(tmp_path: Path, n: int) -> Path:
    """Create n unique images, return the dir path."""
    img_dir = tmp_path / "lab_images"
    img_dir.mkdir()
    for i in range(n):
        # Use varying pixel value to keep file hashes unique.
        Image.new("RGB", (64, 64), (10 + i, 20 + i, 30 + i)).save(img_dir / f"page_{i:03d}.png")
    return img_dir


# --- core ---


class TestIngestLabCore:
    def test_requires_prompt_fn(self, tmp_path):
        d = _make_image_dir(tmp_path, 1)
        with pytest.raises(ValueError, match="prompt_fn"):
            ingest_lab(d, db_path=tmp_path / "db.db")

    def test_inserts_samples_and_ground_truth(self, tmp_path):
        d = _make_image_dir(tmp_path, 3)
        db = tmp_path / "lab.db"

        def prompt(_path, _suggestion):
            return "the cell underwent mitosis"

        counts = ingest_lab(d, prompt_fn=prompt, db_path=db, student="alice")
        assert counts["newly_added"] == 3
        assert counts["annotated"] == 3
        assert counts["skipped_existing_gt"] == 0
        assert counts["skipped_user"] == 0

        # Spot-check storage.
        conn = get_connection(db)
        sample = get_sample_by_hash(conn, hash_file(d / "page_000.png"))
        assert sample is not None
        assert sample.category == "lab"
        assert sample.student == "alice"
        gt_row = conn.execute(
            "SELECT text, source, author FROM ground_truths WHERE sample_id = ?",
            (sample.id,),
        ).fetchone()
        conn.close()
        assert gt_row["text"] == "the cell underwent mitosis"
        assert gt_row["source"] == "lab-grader"
        assert gt_row["author"] == "alice"

    def test_user_skip_leaves_sample_without_ground_truth(self, tmp_path):
        d = _make_image_dir(tmp_path, 2)
        db = tmp_path / "skip.db"

        def prompt(_path, _suggestion):
            return None  # user cancels EDITOR

        counts = ingest_lab(d, prompt_fn=prompt, db_path=db)
        assert counts["newly_added"] == 2
        assert counts["annotated"] == 0
        assert counts["skipped_user"] == 2

        # Samples present, no GT rows.
        conn = get_connection(db)
        gt_count = conn.execute("SELECT COUNT(*) FROM ground_truths").fetchone()[0]
        sample_count = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        conn.close()
        assert sample_count == 2
        assert gt_count == 0

    def test_empty_string_prompt_treated_as_skip(self, tmp_path):
        d = _make_image_dir(tmp_path, 2)
        db = tmp_path / "empty.db"

        def prompt(_path, _suggestion):
            return "   "  # whitespace-only treated as skip

        counts = ingest_lab(d, prompt_fn=prompt, db_path=db)
        assert counts["annotated"] == 0
        assert counts["skipped_user"] == 2

    def test_resume_skips_existing_gt(self, tmp_path):
        d = _make_image_dir(tmp_path, 3)
        db = tmp_path / "resume.db"

        # First pass: annotate page 0 and 1, skip page 2.
        decisions = iter(["text 0", "text 1", None])
        ingest_lab(d, prompt_fn=lambda _p, _s: next(decisions), db_path=db)

        # Second pass: prompt should only be called for page 2.
        called_for: list[str] = []

        def prompt2(path, _s):
            called_for.append(Path(path).name)
            return "text 2"

        counts = ingest_lab(d, prompt_fn=prompt2, db_path=db)

        assert counts["skipped_existing_gt"] == 2
        assert counts["annotated"] == 1
        assert called_for == ["page_002.png"]

    def test_vlm_suggestion_passed_to_prompt(self, tmp_path):
        d = _make_image_dir(tmp_path, 1)
        db = tmp_path / "vlm.db"

        captured: dict = {}

        def prompt(path, suggestion):
            captured["suggestion"] = suggestion
            return "final text"

        # Patch read_with_consensus where ingest_lab will import it from.
        with patch("handwriting_engine.vision.read_with_consensus") as mock_read:
            mock_read.return_value = type("R", (), {"text": "vlm guess"})()
            counts = ingest_lab(
                d, prompt_fn=prompt, db_path=db,
                use_vlm_suggestion=True, vlm_provider="gemini",
            )

        assert captured["suggestion"] == "vlm guess"
        assert counts["annotated"] == 1

    def test_vlm_failure_falls_back_to_empty_suggestion(self, tmp_path):
        d = _make_image_dir(tmp_path, 1)
        db = tmp_path / "vlm_fail.db"

        captured: dict = {}

        def prompt(_path, suggestion):
            captured["suggestion"] = suggestion
            return "manual"

        with patch("handwriting_engine.vision.read_with_consensus",
                   side_effect=RuntimeError("provider down")):
            counts = ingest_lab(
                d, prompt_fn=prompt, db_path=db, use_vlm_suggestion=True,
            )
        # VLM error must not crash the workflow; suggestion is empty,
        # the user can still annotate manually.
        assert captured["suggestion"] == ""
        assert counts["annotated"] == 1

    def test_non_directory_raises(self, tmp_path):
        bogus = tmp_path / "not_a_dir.png"
        bogus.write_bytes(b"not an image")
        with pytest.raises(FileNotFoundError):
            ingest_lab(bogus, prompt_fn=lambda *a: "x", db_path=tmp_path / "db.db")


# --- CLI ---


class TestIngestLabCli:
    def test_cli_invokes_ingest_lab(self, tmp_path):
        d = _make_image_dir(tmp_path, 2)
        db = tmp_path / "cli.db"

        runner = CliRunner()
        # Patch click.edit to simulate the user typing.
        with patch("click.edit", return_value="the cell\n"):
            result = runner.invoke(cli, [
                "benchmark", "ingest-lab", str(d),
                "--student", "alice",
                "--db-path", str(db),
            ])
        assert result.exit_code == 0, result.output
        assert "Lab ingest complete" in result.output
        assert "2 annotated" in result.output

    def test_cli_skip_via_editor_cancel(self, tmp_path):
        d = _make_image_dir(tmp_path, 2)
        db = tmp_path / "cli_skip.db"

        runner = CliRunner()
        # click.edit returning None signals the user closed without saving.
        with patch("click.edit", return_value=None):
            result = runner.invoke(cli, [
                "benchmark", "ingest-lab", str(d),
                "--db-path", str(db),
            ])
        assert result.exit_code == 0, result.output
        assert "0 annotated" in result.output
        assert "2 skipped" in result.output
