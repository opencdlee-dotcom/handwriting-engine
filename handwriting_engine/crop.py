"""
Crop individual answer regions from scanned answer sheets.

Supports a standard 2-column layout (1-25 left, 26-50 right) with
empirically calibrated underline positions. Each crop is enhanced via
adaptive thresholding (OpenCV when available, Pillow fallback) for
improved handwriting recognition accuracy.
"""

import os
import tempfile
from dataclasses import dataclass

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2
    import numpy as np

    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from handwriting_engine._constants import (
    CROP_DPI,
    CROP_JPEG_QUALITY,
    CROP_MIN_HEIGHT,
    CROP_PADDING,
)


@dataclass
class AnswerSheetLayout:
    """
    Layout parameters for answer sheet cropping.

    All position values are expressed as fractions of the full page
    dimension (width or height). Defaults were calibrated from real
    scanned answer sheets at 300 DPI (US Letter = 2550 x 3300 px).
    """

    # Underline positions (fraction of page height)
    first_underline_frac: float = 0.169
    line_spacing_frac: float = 0.0303

    # Crop region relative to each underline (fraction of page height)
    text_above_frac: float = 0.028
    text_below_frac: float = 0.006

    # Horizontal column boundaries (fraction of page width)
    left_x_start: float = 0.005
    left_x_end: float = 0.47
    right_x_start: float = 0.49
    right_x_end: float = 0.995

    # Grid
    questions_per_column: int = 25


def compute_answer_regions(
    image_width: int,
    image_height: int,
    layout: AnswerSheetLayout,
    num_questions: int,
) -> dict[int, tuple[int, int, int, int]]:
    """
    Compute bounding boxes for each answer region.

    Uses the layout's underline positions and column boundaries to calculate
    pixel-coordinate crop rectangles for each question number.

    Args:
        image_width: Full page image width in pixels.
        image_height: Full page image height in pixels.
        layout: An AnswerSheetLayout defining the geometry.
        num_questions: How many questions to compute regions for.

    Returns:
        {question_number: (x1, y1, x2, y2)} in pixel coordinates.
    """
    regions: dict[int, tuple[int, int, int, int]] = {}
    pad = CROP_PADDING

    for q in range(1, num_questions + 1):
        if q <= layout.questions_per_column:
            col_x1 = int(image_width * layout.left_x_start)
            col_x2 = int(image_width * layout.left_x_end)
            line_idx = q - 1
        else:
            col_x1 = int(image_width * layout.right_x_start)
            col_x2 = int(image_width * layout.right_x_end)
            line_idx = q - layout.questions_per_column - 1

        # Underline y-position for this question
        underline_y = layout.first_underline_frac + line_idx * layout.line_spacing_frac

        # Crop from above the underline (where text sits) to just below it
        y1 = int(image_height * (underline_y - layout.text_above_frac))
        y2 = int(image_height * (underline_y + layout.text_below_frac))

        # Clamp with padding
        x1 = max(0, col_x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image_width, col_x2 + pad)
        y2 = min(image_height, y2 + pad)

        regions[q] = (x1, y1, x2, y2)

    return regions


def preprocess_crop(img: Image.Image) -> Image.Image:
    """
    Enhance a single answer crop for better readability.

    When OpenCV is available, applies CLAHE histogram equalization followed by
    adaptive Gaussian thresholding. Otherwise falls back to Pillow's
    autocontrast, unsharp mask, and contrast enhancement.

    The result is always upscaled to at least CROP_MIN_HEIGHT pixels tall.

    Args:
        img: A PIL Image of a single answer crop.

    Returns:
        Enhanced PIL Image in RGB mode.
    """
    gray = img.convert("L")

    if HAS_OPENCV:
        arr = np.array(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        arr = clahe.apply(arr)
        binary = cv2.adaptiveThreshold(
            arr,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10,
        )
        result = Image.fromarray(binary).convert("RGB")
    else:
        enhanced = ImageOps.autocontrast(gray, cutoff=2)
        enhanced = enhanced.filter(
            ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2)
        )
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.4)
        result = enhanced.convert("RGB")

    # Ensure minimum height for downstream readability. LANCZOS upscale adds no
    # information and inflates image tokens sent to vision providers — pad the
    # canvas with white instead so the vision model sees the original pixels.
    w, h = result.size
    if h < CROP_MIN_HEIGHT:
        padded = Image.new("RGB", (w, CROP_MIN_HEIGHT), color=(255, 255, 255))
        y_offset = (CROP_MIN_HEIGHT - h) // 2
        padded.paste(result, (0, y_offset))
        result = padded

    return result


def crop_and_enhance(
    image_path: str,
    regions: dict[int, tuple[int, int, int, int]],
    output_dir: str,
) -> dict[int, str]:
    """
    Crop each answer region from a full page image and preprocess it.

    Opens the source image once, extracts each numbered region, runs
    preprocess_crop on it, and saves the result as a PNG.

    Args:
        image_path: Path to the full page image.
        regions: Mapping of question number to (x1, y1, x2, y2) bounding box.
        output_dir: Directory to write individual crop files.

    Returns:
        {question_number: cropped_image_path}
    """
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(image_path)
    crops: dict[int, str] = {}

    for q_num, (x1, y1, x2, y2) in sorted(regions.items()):
        crop = img.crop((x1, y1, x2, y2))
        enhanced = preprocess_crop(crop)
        crop_path = os.path.join(output_dir, f"q{q_num:02d}.png")
        enhanced.save(crop_path, "PNG")
        crops[q_num] = crop_path

    img.close()
    return crops


def crop_answer_sheet(
    pdf_path: str,
    page_number: int = 0,
    num_questions: int = 23,
    output_dir: str | None = None,
    layout: AnswerSheetLayout | None = None,
) -> dict[int, str]:
    """
    Full pipeline: render a PDF page at high DPI, then crop all answer regions.

    Renders the specified page at CROP_DPI, computes answer region bounding
    boxes using the provided (or default) layout, then crops and enhances
    each region individually.

    Args:
        pdf_path: Path to the student exam PDF.
        page_number: 0-indexed page number of the answer sheet.
        num_questions: Number of questions to crop (default 23).
        output_dir: Where to save crops (auto temp dir if None).
        layout: Custom AnswerSheetLayout or None for defaults.

    Returns:
        {question_number: cropped_image_path}
    """
    import fitz

    if layout is None:
        layout = AnswerSheetLayout()
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="answer_crops_")

    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    page = doc[page_number]
    pix = page.get_pixmap(dpi=CROP_DPI)
    full_image_path = os.path.join(output_dir, "full_page.png")
    pix.save(full_image_path)
    doc.close()

    regions = compute_answer_regions(pix.width, pix.height, layout, num_questions)
    return crop_and_enhance(full_image_path, regions, output_dir)
