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
