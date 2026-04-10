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
