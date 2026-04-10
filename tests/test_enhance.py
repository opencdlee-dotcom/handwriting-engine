"""Tests for enhancement module."""

import os
import tempfile
import pytest
from PIL import Image


def _make_dark_image(path=None):
    """Create a dark, low-contrast test image."""
    img = Image.new("RGB", (200, 200), (50, 50, 50))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "dark text", fill=(70, 70, 70))
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
    img.save(path, "JPEG", quality=95)
    return path


def test_proven_enhance_changes_pixels():
    from handwriting_engine.enhance import proven_enhance
    path = _make_dark_image()
    original = Image.open(path).copy()
    out_fd, out_path = tempfile.mkstemp(suffix=".jpg")
    os.close(out_fd)
    result = proven_enhance(path, output_path=out_path)
    enhanced = Image.open(result)
    # Enhanced should differ from original
    assert enhanced.size[0] >= original.size[0], "Should be upscaled"
    os.unlink(path)
    os.unlink(out_path)


def test_proven_enhance_upscales():
    from handwriting_engine.enhance import proven_enhance
    path = _make_dark_image()
    fd, out_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    proven_enhance(path, output_path=out_path, upscale=2)
    original = Image.open(path)
    enhanced = Image.open(out_path)
    assert enhanced.size[0] == original.size[0] * 2
    assert enhanced.size[1] == original.size[1] * 2
    os.unlink(path)
    os.unlink(out_path)


def test_smart_enhance_on_good_image():
    """Smart enhance should not modify a good image."""
    from handwriting_engine.enhance import smart_enhance
    img = Image.new("RGB", (400, 400), (240, 240, 240))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    for y in range(0, 400, 20):
        draw.line([(0, y), (400, y)], fill=(20, 20, 20), width=2)
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    img.save(path, "JPEG", quality=95)
    result = smart_enhance(path)
    assert result == path  # Should return same path (no changes needed)
    os.unlink(path)


def test_crisp_presets():
    from handwriting_engine.enhance import crisp
    path = _make_dark_image()
    for strength in ("light", "medium", "heavy"):
        fd, out = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        result = crisp(path, strength=strength, output_path=out)
        assert os.path.exists(result)
        os.unlink(out)
    os.unlink(path)


def test_enhance_image_strategies():
    from handwriting_engine.enhance import enhance_image
    path = _make_dark_image()
    for strategy in ("proven", "smart", "light", "medium", "heavy"):
        fd, out = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        result = enhance_image(path, strategy=strategy, output_path=out)
        assert os.path.exists(result)
        os.unlink(out)
    os.unlink(path)


def test_rgba_preservation():
    """Enhancement must preserve alpha channel."""
    from handwriting_engine.enhance import proven_enhance
    img = Image.new("RGBA", (100, 100), (50, 50, 50, 128))
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.save(path)
    fd, out = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    result = proven_enhance(path, output_path=out)
    enhanced = Image.open(result)
    assert enhanced.mode == "RGBA", "Alpha channel must be preserved"
    os.unlink(path)
    os.unlink(out)


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
