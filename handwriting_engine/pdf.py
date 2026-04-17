"""
Convert PDF pages to images for handwriting recognition.

Handles orientation detection, DPI-controlled rendering, optional enhancement,
and memory-safe page-by-page conversion with progress callback support.
"""

import gc
import os

import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter

from handwriting_engine._constants import (
    JPEG_QUALITY,
    MAX_IMAGE_LONG_SIDE,
    PDF_DPI,
    SURVEY_DPI,
    SURVEY_MAX_LONG_SIDE,
    SURVEY_QUALITY,
)


def detect_and_fix_orientation(image_path: str) -> bool:
    """
    Auto-rotate landscape pages to portrait.

    Detects pages where width significantly exceeds height (ratio > 1.15)
    and rotates them 90 degrees clockwise, overwriting the original file.

    Returns:
        True if the image was rotated, False otherwise.
    """
    img = Image.open(image_path)
    width, height = img.size
    rotated = False

    if width > height and (width / height) > 1.15:
        img = img.rotate(-90, expand=True)
        img.save(image_path)
        rotated = True

    img.close()
    return rotated


def get_page_count(pdf_path: str) -> int:
    """
    Get PDF page count without full rendering.

    Opens the PDF only long enough to read metadata, then closes immediately.
    """
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


def convert_pdf(
    pdf_path: str,
    output_dir: str,
    dpi: int | None = None,
    jpeg_quality: int | None = None,
    max_long_side: int | None = None,
    auto_enhance: bool = False,
    progress_callback=None,
    max_pages: int = 500,
) -> list[dict]:
    """
    Convert PDF to JPEG images.

    Renders each page at the specified DPI, corrects orientation, resizes to
    fit within max_long_side, and optionally applies image enhancement. Uses
    gc.collect() after each page to keep memory usage bounded.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save page images.
        dpi: Rendering DPI (default: PDF_DPI from _constants).
        jpeg_quality: JPEG compression quality (default: JPEG_QUALITY from _constants).
        max_long_side: Max pixel dimension on longest side (default: MAX_IMAGE_LONG_SIDE).
        auto_enhance: Apply light sharpening and contrast enhancement to all pages.
        progress_callback: Optional callable(current, total) for GUI/progress integration.

    Returns:
        List of dicts with keys: page_number, path, original_rotation, auto_corrected.
    """
    if dpi is None:
        dpi = PDF_DPI
    if jpeg_quality is None:
        jpeg_quality = JPEG_QUALITY
    if max_long_side is None:
        max_long_side = MAX_IMAGE_LONG_SIDE

    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    if total_pages > max_pages:
        doc.close()
        raise ValueError(f"PDF has {total_pages} pages (max {max_pages}). Pass max_pages= to override.")
    results = []

    for page_num in range(total_pages):
        page = doc.load_page(page_num)

        # Fix PDF rotation metadata so rendering is upright
        rotation = page.rotation
        if rotation != 0:
            page.set_rotation(0)

        # Render at requested DPI
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Save as temporary PNG, then convert to JPEG with resize
        tmp_path = os.path.join(output_dir, f"page_{page_num + 1:03d}.png")
        output_path = os.path.join(output_dir, f"page_{page_num + 1:03d}.jpg")
        pix.save(tmp_path)
        del pix
        gc.collect()

        # Auto-rotate landscape pages
        was_rotated = detect_and_fix_orientation(tmp_path)

        # Convert to JPEG with optional downscale
        img = Image.open(tmp_path)
        if max_long_side:
            w, h = img.size
            long_side = max(w, h)
            if long_side > max_long_side:
                ratio = max_long_side / long_side
                new_size = (int(w * ratio), int(h * ratio))
                img = img.resize(new_size, Image.LANCZOS)

        # Convert RGBA to RGB for JPEG compatibility (PDF pages with transparency)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(output_path, "JPEG", quality=jpeg_quality)
        img.close()
        os.remove(tmp_path)

        # Optional enhancement pass
        if auto_enhance:
            _enhance_page(output_path, jpeg_quality)

        results.append({
            "page_number": page_num + 1,
            "path": output_path,
            "original_rotation": rotation,
            "auto_corrected": was_rotated,
        })

        if progress_callback is not None:
            progress_callback(page_num + 1, total_pages)

        gc.collect()

    doc.close()
    return results


def convert_pdf_survey(pdf_path: str, output_dir: str) -> list[dict]:
    """
    Low-res survey pass for quick page content identification.

    Uses minimal DPI and quality -- just enough to detect section headers
    and page content types, not to read handwriting detail.
    """
    return convert_pdf(
        pdf_path,
        output_dir,
        dpi=SURVEY_DPI,
        jpeg_quality=SURVEY_QUALITY,
        max_long_side=SURVEY_MAX_LONG_SIDE,
    )


def cleanup_images(output_dir: str):
    """
    Remove all generated page images from the output directory.

    Deletes .jpg and .png files, then attempts to remove the directory itself.
    Silently ignores a non-empty directory (e.g., if non-image files remain).
    """
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith((".jpg", ".png")):
                os.remove(os.path.join(output_dir, f))
        try:
            os.rmdir(output_dir)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _enhance_page(image_path: str, jpeg_quality: int) -> None:
    """Apply light sharpening and contrast boost to a page image in place."""
    from handwriting_engine.enhance import mark_enhanced

    img = Image.open(image_path)
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img.save(image_path, "JPEG", quality=jpeg_quality)
    img.close()
    mark_enhanced(image_path)
