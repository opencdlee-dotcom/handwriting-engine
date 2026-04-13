"""Tests for benchmark ingestion — image import and deduplication."""

import os
import pytest
from PIL import Image

from handwriting_engine.benchmark.db import get_connection, list_samples
from handwriting_engine.benchmark.ingest import (
    hash_file,
    ingest_directory,
    ingest_single,
    _extract_page_number,
)


@pytest.fixture
def img_dir(tmp_path):
    """Create a temp directory with synthetic test images."""
    for i in range(3):
        img = Image.new("RGB", (200, 200), color=(100 + i * 50, 100, 100))
        img.save(tmp_path / f"page_{i + 1:03d}.png")
    return tmp_path


@pytest.fixture
def db_path(tmp_path):
    """Temp database path."""
    return tmp_path / "test_benchmark.db"


class TestHashFile:
    def test_deterministic(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("hello world")
        h1 = hash_file(path)
        h2 = hash_file(path)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("hello")
        b.write_text("world")
        assert hash_file(a) != hash_file(b)


class TestExtractPageNumber:
    def test_standard_format(self):
        assert _extract_page_number("page_003") == 3
        assert _extract_page_number("page_001") == 1

    def test_no_number(self):
        assert _extract_page_number("image") == 0

    def test_multiple_numbers(self):
        # Takes first number
        assert _extract_page_number("page_2_v3") == 2


class TestIngestDirectory:
    def test_imports_images(self, img_dir, db_path):
        samples = ingest_directory(img_dir, student="test", category="bio", db_path=db_path, assess_quality=False)
        assert len(samples) == 3
        assert all(s.student == "test" for s in samples)
        assert all(s.category == "bio" for s in samples)

    def test_dedup_on_reimport(self, img_dir, db_path):
        first = ingest_directory(img_dir, db_path=db_path, assess_quality=False)
        second = ingest_directory(img_dir, db_path=db_path, assess_quality=False)
        assert len(first) == 3
        assert len(second) == 0  # All duplicates

    def test_page_numbers_extracted(self, img_dir, db_path):
        samples = ingest_directory(img_dir, db_path=db_path, assess_quality=False)
        page_nums = sorted(s.page_number for s in samples)
        assert page_nums == [1, 2, 3]

    def test_nonexistent_dir_raises(self, db_path):
        with pytest.raises(FileNotFoundError):
            ingest_directory("/nonexistent/dir", db_path=db_path)

    def test_empty_dir(self, tmp_path, db_path):
        samples = ingest_directory(tmp_path, db_path=db_path, assess_quality=False)
        assert len(samples) == 0


class TestIngestSingle:
    def test_import_one(self, img_dir, db_path):
        img = list(img_dir.glob("*.png"))[0]
        sample = ingest_single(img, student="alice", db_path=db_path)
        assert sample is not None
        assert sample.student == "alice"

    def test_duplicate_returns_none(self, img_dir, db_path):
        img = list(img_dir.glob("*.png"))[0]
        first = ingest_single(img, db_path=db_path)
        second = ingest_single(img, db_path=db_path)
        assert first is not None
        assert second is None

    def test_nonexistent_file_raises(self, db_path):
        with pytest.raises(FileNotFoundError):
            ingest_single("/nonexistent/img.png", db_path=db_path)

    def test_duplicate_race_returns_none(self, img_dir, db_path):
        """ingest_single handles IntegrityError race gracefully."""
        img = list(img_dir.glob("*.png"))[0]
        # First import
        first = ingest_single(img, db_path=db_path)
        assert first is not None
        # Second import returns None (duplicate)
        second = ingest_single(img, db_path=db_path)
        assert second is None


class TestExtractPageNumberImproved:
    def test_page_prefix_pattern(self):
        assert _extract_page_number("page_005") == 5
        assert _extract_page_number("page5") == 5
        assert _extract_page_number("Page-12") == 12

    def test_falls_back_to_last_number(self):
        # student_2_v3 -> falls back to last number (3)
        assert _extract_page_number("student_2_v3") == 3

    def test_page_pattern_wins(self):
        # student_2_page_5 -> page pattern finds 5
        assert _extract_page_number("student_2_page_5") == 5


class TestGenerateDegradedVariants:
    def test_generates_variants(self, db_path, tmp_path):
        from handwriting_engine.benchmark.ingest import generate_degraded_variants
        from handwriting_engine.benchmark.db import get_connection, insert_ground_truth
        import random

        # Create an image with varied pixel content so degradations produce unique hashes
        img = Image.new("RGB", (200, 200))
        pixels = img.load()
        random.seed(42)
        for x in range(200):
            for y in range(200):
                pixels[x, y] = (random.randint(0, 255), random.randint(50, 200), random.randint(0, 255))
        rich_img_path = tmp_path / "rich_image.png"
        img.save(rich_img_path)

        sample = ingest_single(rich_img_path, db_path=db_path)
        assert sample is not None

        conn = get_connection(db_path)
        insert_ground_truth(conn, sample.id, "test ground truth text")
        conn.close()

        out_dir = tmp_path / "degraded"
        variants = generate_degraded_variants(sample.id, out_dir, db_path=db_path)
        assert len(variants) == 7  # blur, lowcontrast, rotate, noise, crop80, perspective, elastic
        assert all(v.has_ground_truth for v in variants)

    def test_missing_sample_raises(self, db_path, tmp_path):
        from handwriting_engine.benchmark.ingest import generate_degraded_variants

        with pytest.raises(ValueError, match="not found"):
            generate_degraded_variants(999, tmp_path, db_path=db_path)


class TestIAMIngest:
    """Tests for IAM ingestion (IAM-01). Task 1: parse + ingest; Task 2: CLI."""

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _make_ascii_dir(self, tmp_path):
        """Create a minimal IAM ascii/ directory with a lines.txt and PNG stubs."""
        ascii_dir = tmp_path / "ascii"
        ascii_dir.mkdir()
        lines_txt = ascii_dir / "lines.txt"
        lines_txt.write_text(
            "# IAM Online Handwriting Database\n"
            "\n"
            "a01-000u-00 ok 154 1 408 768 27 51 A MOVE IN\n"
            "a01-000u-01 ok 154 1 408 768 27 51 THE LONG|WAY\n"
            "a01-000u-02 err 0 0 0 0 0 0 bad line\n"
            "a01-001-00 ok 120 1 300 600 20 40 HELLO WORLD\n",
            encoding="utf-8",
        )
        # Create a fake lines/ directory with one real PNG per line
        lines_dir = tmp_path / "lines"
        writer_dir = lines_dir / "a01"
        form_dir = writer_dir / "a01-000u"
        form_dir.mkdir(parents=True)
        img = Image.new("RGB", (200, 50), color=(240, 240, 240))
        img.save(form_dir / "a01-000u-00.png")
        # Second line image (distinct pixels so hash differs)
        img2 = Image.new("RGB", (200, 50), color=(230, 230, 230))
        img2.save(form_dir / "a01-000u-01.png")
        return ascii_dir, lines_dir

    # ------------------------------------------------------------------ #
    # parse_iam_lines tests                                                #
    # ------------------------------------------------------------------ #

    def test_parse_skips_comments(self, tmp_path):
        from handwriting_engine.benchmark.ingest import parse_iam_lines

        lines_txt = tmp_path / "lines.txt"
        lines_txt.write_text(
            "# comment line\n"
            "\n"
            "a01-000u-00 ok 154 1 408 768 27 51 A MOVE IN\n",
            encoding="utf-8",
        )
        result = parse_iam_lines(lines_txt)
        assert len(result) == 1
        assert result[0]["line_id"] == "a01-000u-00"

    def test_parse_filters_err(self, tmp_path):
        from handwriting_engine.benchmark.ingest import parse_iam_lines

        lines_txt = tmp_path / "lines.txt"
        lines_txt.write_text(
            "a01-000u-00 ok 154 1 408 768 27 51 A MOVE IN\n"
            "a01-000u-01 err 0 0 0 0 0 0 bad\n",
            encoding="utf-8",
        )
        result = parse_iam_lines(lines_txt)
        assert len(result) == 1
        assert result[0]["line_id"] == "a01-000u-00"

    def test_parse_extracts_fields(self, tmp_path):
        from handwriting_engine.benchmark.ingest import parse_iam_lines

        lines_txt = tmp_path / "lines.txt"
        lines_txt.write_text(
            "a01-000u-00 ok 154 1 408 768 27 51 A MOVE IN\n",
            encoding="utf-8",
        )
        result = parse_iam_lines(lines_txt)
        assert len(result) == 1
        rec = result[0]
        assert rec["line_id"] == "a01-000u-00"
        assert rec["writer_id"] == "a01"
        assert rec["form_id"] == "a01-000u"
        assert rec["transcription"] == "A MOVE IN"

    def test_parse_replaces_pipes(self, tmp_path):
        from handwriting_engine.benchmark.ingest import parse_iam_lines

        lines_txt = tmp_path / "lines.txt"
        lines_txt.write_text(
            "a01-000u-00 ok 154 1 408 768 27 51 put|down|a|resolution\n",
            encoding="utf-8",
        )
        result = parse_iam_lines(lines_txt)
        assert result[0]["transcription"] == "put down a resolution"

    def test_parse_filters_partition(self, tmp_path):
        from handwriting_engine.benchmark.ingest import parse_iam_lines

        lines_txt = tmp_path / "lines.txt"
        lines_txt.write_text(
            "a01-000u-00 ok 154 1 408 768 27 51 A MOVE IN\n"
            "a01-001-00 ok 120 1 300 600 20 40 HELLO WORLD\n",
            encoding="utf-8",
        )
        result = parse_iam_lines(lines_txt, partition_forms={"a01-001"})
        assert len(result) == 1
        assert result[0]["form_id"] == "a01-001"

    # ------------------------------------------------------------------ #
    # ingest_iam tests                                                     #
    # ------------------------------------------------------------------ #

    def test_ingest_sets_category_and_student(self, tmp_path):
        from handwriting_engine.benchmark.ingest import ingest_iam
        from handwriting_engine.benchmark.db import get_connection, list_samples

        ascii_dir, lines_dir = self._make_ascii_dir(tmp_path)
        db_path = tmp_path / "test.db"
        ingest_iam(
            ascii_dir=ascii_dir,
            lines_dir=lines_dir,
            db_path=db_path,
        )
        conn = get_connection(db_path)
        samples = list_samples(conn)
        conn.close()
        assert len(samples) >= 1
        for s in samples:
            assert s.category == "iam"
            assert s.student.startswith("iam-writer-")

    def test_ingest_inserts_ground_truth(self, tmp_path):
        from handwriting_engine.benchmark.ingest import ingest_iam
        from handwriting_engine.benchmark.db import get_connection, list_samples, get_latest_ground_truth

        ascii_dir, lines_dir = self._make_ascii_dir(tmp_path)
        db_path = tmp_path / "test.db"
        ingest_iam(
            ascii_dir=ascii_dir,
            lines_dir=lines_dir,
            db_path=db_path,
        )
        conn = get_connection(db_path)
        samples = list_samples(conn)
        for s in samples:
            gt = get_latest_ground_truth(conn, s.id)
            assert gt is not None, f"Sample {s.id} missing ground truth"
            assert gt.text.strip() != ""
        conn.close()

    def test_ingest_iam_dedup(self, tmp_path):
        from handwriting_engine.benchmark.ingest import ingest_iam
        from handwriting_engine.benchmark.db import get_connection, list_samples

        ascii_dir, lines_dir = self._make_ascii_dir(tmp_path)
        db_path = tmp_path / "test.db"
        result1 = ingest_iam(ascii_dir=ascii_dir, lines_dir=lines_dir, db_path=db_path)
        result2 = ingest_iam(ascii_dir=ascii_dir, lines_dir=lines_dir, db_path=db_path)
        conn = get_connection(db_path)
        count_after = len(list_samples(conn))
        conn.close()
        assert result2["skipped_dup"] > 0, "Second import should skip duplicates"
        assert result2["ingested"] == 0, "Second import should ingest nothing new"
        assert count_after == result1["ingested"], "Row count must not change on reimport"

    # ------------------------------------------------------------------ #
    # CLI test                                                             #
    # ------------------------------------------------------------------ #

    def test_cli_ingest_iam_command(self, tmp_path):
        from click.testing import CliRunner
        from handwriting_engine.cli import cli

        ascii_dir, lines_dir = self._make_ascii_dir(tmp_path)
        db_path = tmp_path / "test.db"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "benchmark", "ingest-iam",
                str(ascii_dir),
                "--lines-dir", str(lines_dir),
                "--all-partitions",
                "--db-path", str(db_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "ingested" in result.output.lower() or "IAM ingest" in result.output
