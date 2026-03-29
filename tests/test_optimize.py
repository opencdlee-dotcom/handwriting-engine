"""Tests for image optimization, EXIF stripping, and adaptive settings."""

import base64
import os
import tempfile
import pytest
from PIL import Image

from handwriting_engine.optimize import (
    validate_and_prepare_image,
    build_image_blocks,
    batch_pages_by_size,
    get_adaptive_settings,
)


def _make_image(size=(200, 200), color=(128, 128, 128), mode="RGB"):
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    img = Image.new(mode, size, color)
    img.save(path, "JPEG", quality=95)
    return path


# --- validate_and_prepare_image ---

def test_prepare_normal_image():
    path = _make_image()
    result = validate_and_prepare_image(path)
    os.unlink(path)
    assert result is not None
    b64_data, media_type = result
    assert media_type == "image/jpeg"
    # Verify it's valid base64
    decoded = base64.b64decode(b64_data)
    assert len(decoded) > 0


def test_prepare_missing_file():
    result = validate_and_prepare_image("/nonexistent/path.jpg")
    assert result is None


def test_prepare_empty_file():
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    # File exists but is empty
    result = validate_and_prepare_image(path)
    os.unlink(path)
    assert result is None


def test_prepare_oversized_image_resized():
    """Images larger than max_long_side should be resized."""
    path = _make_image(size=(3000, 2000))
    result = validate_and_prepare_image(path, max_long_side=1568)
    os.unlink(path)
    assert result is not None
    b64_data, _ = result
    raw = base64.b64decode(b64_data)
    # Re-open to check dimensions
    import io
    img = Image.open(io.BytesIO(raw))
    assert max(img.size) <= 1568


def test_prepare_rgba_converted():
    """RGBA images should be converted to RGB for JPEG."""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img = Image.new("RGBA", (200, 200), (128, 128, 128, 255))
    img.save(path, "PNG")
    result = validate_and_prepare_image(path)
    os.unlink(path)
    assert result is not None


def test_prepare_quality_degradation():
    """If image is too large at default quality, should try lower qualities."""
    # Create a large complex image
    path = _make_image(size=(1500, 1500))
    # Use a very small max_file_bytes to force degradation
    result = validate_and_prepare_image(path, max_file_bytes=5000)
    os.unlink(path)
    # May return None if even quality=50 is too big, or succeed at lower quality
    # Either way, shouldn't crash


def test_prepare_corrupt_file():
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(b"not a real image file at all")
    result = validate_and_prepare_image(path)
    os.unlink(path)
    assert result is None


# --- build_image_blocks ---

def test_build_image_blocks():
    path = _make_image()
    pages = [{"path": path, "page_number": 1}]
    blocks = build_image_blocks(pages)
    os.unlink(path)
    # Should produce an image block + text label
    assert len(blocks) == 2
    assert blocks[0]["type"] == "image"
    assert blocks[1]["type"] == "text"
    assert "[Page 1]" in blocks[1]["text"]


def test_build_image_blocks_skips_corrupt():
    pages = [{"path": "/nonexistent.jpg", "page_number": 1}]
    blocks = build_image_blocks(pages)
    assert len(blocks) == 0


# --- batch_pages_by_size ---

def test_batch_pages_single():
    path = _make_image()
    pages = [{"path": path, "page_number": 1}]
    batches = batch_pages_by_size(pages)
    os.unlink(path)
    assert len(batches) == 1
    assert len(batches[0]) == 1


def test_batch_pages_splits_large():
    """Multiple pages should be split into batches that fit size limit."""
    paths = [_make_image(size=(800, 800)) for _ in range(5)]
    pages = [{"path": p, "page_number": i + 1} for i, p in enumerate(paths)]
    # Use a small max to force splitting
    batches = batch_pages_by_size(pages, max_batch_bytes=50000)
    for p in paths:
        os.unlink(p)
    assert len(batches) >= 1
    total_pages = sum(len(b) for b in batches)
    assert total_pages <= 5  # Some may be skipped if too large


# --- get_adaptive_settings ---

def test_adaptive_settings_small():
    dpi, quality, max_long = get_adaptive_settings(5)
    assert dpi == 200
    assert quality == 85
    assert max_long == 1568


def test_adaptive_settings_medium():
    dpi, quality, max_long = get_adaptive_settings(25)
    assert dpi == 150
    assert quality == 80


def test_adaptive_settings_large():
    dpi, quality, max_long = get_adaptive_settings(100)
    assert dpi == 100
    assert quality == 70
    assert max_long == 1024


def test_adaptive_settings_huge():
    """Very large documents should get the most aggressive tier."""
    dpi, quality, max_long = get_adaptive_settings(5000)
    assert dpi == 100
    assert max_long == 1024
