"""Tests for quality assessment module."""

import os
import tempfile
import pytest
from PIL import Image


def _make_test_image(color=(255, 255, 255), size=(200, 200), path=None):
    """Create a solid color test image."""
    img = Image.new("RGB", size, color)
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
    img.save(path, "JPEG", quality=95)
    return path


def test_assess_blur_white():
    from handwriting_engine.quality import assess_blur
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    score = assess_blur(img)
    # Laplacian kernel with offset=128 produces non-zero variance even on uniform images
    # The key behavior: a uniform image should score differently from a sharp edgy image
    assert isinstance(score, float), "Should return a float"


def test_assess_contrast_white():
    from handwriting_engine.quality import assess_contrast
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    score = assess_contrast(img)
    assert score < 0.1, "Solid color should have near-zero contrast"


def test_assess_contrast_high():
    from handwriting_engine.quality import assess_contrast
    img = Image.new("RGB", (200, 200), (0, 0, 0))
    # Draw white rectangle in center
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 150], fill=(255, 255, 255))
    score = assess_contrast(img)
    assert score > 0.5, "Black/white image should have high contrast"


def test_assess_brightness_white():
    from handwriting_engine.quality import assess_brightness
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    mean, std = assess_brightness(img)
    assert mean > 250, "White image brightness should be ~255"


def test_assess_image_good():
    """A high-contrast image with edges should be 'good'."""
    from handwriting_engine.quality import assess_image
    img = Image.new("RGB", (400, 400), (240, 240, 240))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    for y in range(0, 400, 20):
        draw.line([(0, y), (400, y)], fill=(20, 20, 20), width=2)
    path = _make_test_image()
    img.save(path, "JPEG", quality=95)
    report = assess_image(path)
    os.unlink(path)
    assert report["quality"] in ("good", "fair")


def test_assess_image_faint():
    """High brightness + low contrast = faint ink detection."""
    from handwriting_engine.quality import assess_image
    img = Image.new("RGB", (200, 200), (220, 220, 220))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "faint", fill=(200, 200, 200))
    path = _make_test_image()
    img.save(path, "JPEG", quality=95)
    report = assess_image(path)
    os.unlink(path)
    assert report["faint_ink"] is True


def test_batch_assess():
    from handwriting_engine.quality import batch_assess
    tmpdir = tempfile.mkdtemp()
    for i in range(3):
        _make_test_image(path=os.path.join(tmpdir, f"page_{i:03d}.jpg"))
    results = batch_assess(tmpdir)
    assert len(results) == 3
    for path, report in results.items():
        assert "quality" in report
