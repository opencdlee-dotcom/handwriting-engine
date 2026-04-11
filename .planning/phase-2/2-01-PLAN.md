<plan phase="2" index="01" requirement="REQ-003">
  <objective>Create line_reader.py with OpenCV horizontal projection profile line segmentation, reassembly, and integrate line_level=True into vision.py read_image()</objective>

  <files>
    <create>handwriting_engine/line_reader.py</create>
    <modify>handwriting_engine/vision.py</modify>
    <create>tests/test_line_reader.py</create>
  </files>

  <tasks>
    <task type="auto">
      <name>Create handwriting_engine/line_reader.py</name>
      <action>
Create ~/Developer/handwriting-engine/handwriting_engine/line_reader.py:

"""
Line-level segmentation for handwritten page images.

Segments a page into individual text lines using OpenCV horizontal
projection profile, reads each line independently via a vision provider,
and reassembles into a complete transcription. Reduces attention diffusion
on dense multi-line pages.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Minimum line height in pixels to count as a text line (filters ruled lines)
MIN_LINE_HEIGHT_PX = 15
# Minimum gap between lines (pixels) to be considered a separator
MIN_GAP_PX = 5
# Padding added above/below each line crop
LINE_PADDING_PX = 4


def segment_lines(image_path: str) -> list[tuple[Image.Image, tuple[int, int, int, int]]]:
    """Segment a page image into individual text line crops.

    Uses horizontal projection profile (sum of dark pixels per row).
    Gaps between text lines appear as near-zero rows in the profile.

    Args:
        image_path: Path to the input image.

    Returns:
        List of (line_image_pil, bbox) tuples where bbox = (x, y, w, h).
        Returns empty list if segmentation fails or finds < 2 lines.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning("segment_lines: could not read image at %s", image_path)
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Binarize: dark ink on white
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Horizontal projection: sum dark pixels per row
        row_sums = np.sum(binary, axis=1)
        h, w = binary.shape
        # Smooth the projection to avoid micro-gaps splitting lines
        kernel = np.ones(3) / 3
        smoothed = np.convolve(row_sums, kernel, mode="same")
        # Find text regions: rows with significant ink
        threshold = max(smoothed.max() * 0.02, 5)
        in_text = smoothed > threshold
        # Find transitions: text start/end pairs
        regions = []
        start = None
        for i, val in enumerate(in_text):
            if val and start is None:
                start = i
            elif not val and start is not None:
                regions.append((start, i))
                start = None
        if start is not None:
            regions.append((start, h))
        # Merge regions that are too close (gap < MIN_GAP_PX)
        merged = []
        for region in regions:
            if merged and region[0] - merged[-1][1] < MIN_GAP_PX:
                merged[-1] = (merged[-1][0], region[1])
            else:
                merged.append(list(region))
        # Filter out regions that are too short (ruled paper lines)
        lines = [(s, e) for s, e in merged if (e - s) >= MIN_LINE_HEIGHT_PX]
        if len(lines) < 2:
            return []
        # Crop each line with padding
        result = []
        pil_img = Image.open(image_path)
        img_w, img_h = pil_img.size
        for start_row, end_row in lines:
            y1 = max(0, start_row - LINE_PADDING_PX)
            y2 = min(img_h, end_row + LINE_PADDING_PX)
            crop = pil_img.crop((0, y1, img_w, y2))
            bbox = (0, y1, img_w, y2 - y1)
            result.append((crop, bbox))
        pil_img.close()
        return result
    except Exception as e:
        logger.warning("segment_lines failed: %s", e)
        return []


def _pil_to_b64(img: Image.Image, fmt: str = "JPEG") -> tuple[str, str]:
    """Convert PIL image to base64 string and media type."""
    buf = io.BytesIO()
    if img.mode == "RGBA" and fmt == "JPEG":
        img = img.convert("RGB")
    img.save(buf, format=fmt, quality=95)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    media_type = f"image/{fmt.lower()}"
    return b64, media_type


def read_page_by_lines(
    image_path: str,
    provider,
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 1024,
    line_separator: str = "\n",
) -> str:
    """Segment page into lines, read each independently, reassemble.

    Args:
        image_path: Path to the page image.
        provider: VisionProvider instance to use for each line read.
        prompt: Per-line reading prompt.
        system_prompt: System prompt for the provider.
        max_tokens: Max tokens per line read.
        line_separator: Separator between reassembled lines.

    Returns:
        Reassembled transcription string. Falls back to empty string on
        complete failure. Callers should fall back to whole-page read if
        this returns empty.
    """
    lines = segment_lines(image_path)
    if not lines:
        logger.info("read_page_by_lines: segmentation found < 2 lines, caller should fall back")
        return ""

    line_texts = []
    for i, (line_img, bbox) in enumerate(lines):
        try:
            b64, media_type = _pil_to_b64(line_img)
            text = provider.read_image(b64, media_type, prompt, system_prompt, max_tokens)
            line_texts.append(text.strip())
            line_img.close()
        except Exception as e:
            logger.warning("read_page_by_lines: line %d failed: %s", i, e)
            line_texts.append("")

    return line_separator.join(t for t in line_texts if t)
      </action>
      <verify>python -c "from handwriting_engine.line_reader import segment_lines, read_page_by_lines; print('OK')"</verify>
      <done>line_reader.py importable with segment_lines and read_page_by_lines</done>
    </task>

    <task type="auto">
      <name>Add line_level parameter to vision.py read_image()</name>
      <action>
In ~/Developer/handwriting-engine/handwriting_engine/vision.py, find the main read_image() function (it will have image_path, prompt, etc. as parameters).

Add a `line_level: bool = False` parameter to the function signature.

Inside the function, before the main provider read, add:

    if line_level:
        from handwriting_engine.line_reader import read_page_by_lines
        from handwriting_engine.providers import get_provider as _get_provider, available_providers as _avail
        avail = _avail()
        if avail:
            _prov = _get_provider(avail[0])
            line_text = read_page_by_lines(image_path, _prov, prompt, system_prompt or "", max_tokens)
            if line_text.strip():
                return _postprocess_output(line_text)
        # Fall through to whole-page read if line segmentation fails

This should be inserted early in read_image(), after image loading/enhancement but before the provider read call.
      </action>
      <verify>python -c "import inspect; from handwriting_engine.vision import read_image; print('line_level' in str(inspect.signature(read_image)))"</verify>
      <done>read_image() accepts line_level=True parameter</done>
    </task>

    <task type="auto">
      <name>Create tests/test_line_reader.py</name>
      <action>
Create ~/Developer/handwriting-engine/tests/test_line_reader.py:

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
      </action>
      <verify>cd "~/Developer/handwriting-engine" && python -m pytest tests/test_line_reader.py -x -q 2>&1 | tail -5</verify>
      <done>All line_reader tests pass</done>
    </task>
  </tasks>

  <dependencies>none</dependencies>
  <commit_message>feat(phase-2): line-level segmentation pipeline (REQ-003)</commit_message>
</plan>
