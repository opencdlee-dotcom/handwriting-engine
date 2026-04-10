"""Tests for line-level segmentation."""
import io
from unittest.mock import MagicMock
import pytest
from PIL import Image, ImageDraw

from handwriting_engine.line_reader import (
    segment_lines,
    read_page_by_lines,
    _pil_to_b64,
    MIN_LINE_HEIGHT_PX,
)


def _make_lined_image(tmp_path, n_lines=3, line_height=30, gap=20, width=400, page_height=None):
    """Create a synthetic multi-line handwriting image for testing."""
    if page_height is None:
        page_height = n_lines * (line_height + gap) + gap
    img = Image.new("RGB", (width, page_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = gap
    for i in range(n_lines):
        # Draw a "text line" as a black rectangle band
        draw.rectangle([10, y, width - 10, y + line_height], fill=(0, 0, 0))
        y += line_height + gap
    path = str(tmp_path / "test_lines.jpg")
    img.save(path, "JPEG")
    return path


def test_segment_lines_finds_multiple_lines(tmp_path):
    path = _make_lined_image(tmp_path, n_lines=3)
    lines = segment_lines(path)
    assert len(lines) >= 2, f"Expected >= 2 lines, got {len(lines)}"
    for crop, bbox in lines:
        assert isinstance(crop, Image.Image)
        x, y, w, h = bbox
        assert h >= MIN_LINE_HEIGHT_PX


def test_segment_lines_single_line_returns_empty(tmp_path):
    """A single-line image should return [] (caller falls back to whole-page)."""
    path = _make_lined_image(tmp_path, n_lines=1, line_height=40)
    lines = segment_lines(path)
    assert lines == []


def test_segment_lines_bad_path():
    lines = segment_lines("/nonexistent/path.jpg")
    assert lines == []


def test_pil_to_b64_rgb():
    img = Image.new("RGB", (100, 50), color=(200, 200, 200))
    b64, media_type = _pil_to_b64(img)
    assert len(b64) > 0
    assert media_type == "image/jpeg"


def test_pil_to_b64_rgba_converts():
    img = Image.new("RGBA", (100, 50), color=(200, 200, 200, 255))
    b64, media_type = _pil_to_b64(img)
    assert len(b64) > 0


def test_read_page_by_lines_calls_provider_per_line(tmp_path):
    path = _make_lined_image(tmp_path, n_lines=3)
    mock_provider = MagicMock()
    mock_provider.read_image.return_value = "line text"
    result = read_page_by_lines(path, mock_provider, "read this")
    # Should have called read_image once per detected line
    assert mock_provider.read_image.call_count >= 2
    assert "line text" in result


def test_read_page_by_lines_single_line_returns_empty(tmp_path):
    """Single-line image: returns empty string so caller falls back."""
    path = _make_lined_image(tmp_path, n_lines=1, line_height=40)
    mock_provider = MagicMock()
    result = read_page_by_lines(path, mock_provider, "read this")
    assert result == ""
    mock_provider.read_image.assert_not_called()


def test_read_page_by_lines_assembles_in_order(tmp_path):
    path = _make_lined_image(tmp_path, n_lines=3)
    call_count = [0]
    def side_effect(*args, **kwargs):
        call_count[0] += 1
        return f"line_{call_count[0]}"
    mock_provider = MagicMock()
    mock_provider.read_image.side_effect = side_effect
    result = read_page_by_lines(path, mock_provider, "read")
    parts = result.split("\n")
    assert parts[0] == "line_1"
    assert parts[-1] == f"line_{call_count[0]}"
