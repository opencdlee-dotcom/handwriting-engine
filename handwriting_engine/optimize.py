"""
Image validation, resizing, and optimization for vision-model APIs.

Ensures images stay within recommended dimension and byte-size limits,
provides adaptive DPI selection for large submissions, and builds
ready-to-send API content blocks with size-aware batching.
"""

import base64
import io
import logging
import os

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

from handwriting_engine._constants import (
    DPI_TIERS,
    JPEG_QUALITY,
    MAX_IMAGE_FILE_BYTES,
    MAX_IMAGE_LONG_SIDE,
    MAX_REQUEST_BYTES,
)


def validate_and_prepare_image(
    image_path: str,
    max_long_side: int = MAX_IMAGE_LONG_SIDE,
    max_file_bytes: int = MAX_IMAGE_FILE_BYTES,
    jpeg_quality: int = JPEG_QUALITY,
) -> tuple[str, str] | None:
    """
    Validate, resize if needed, and base64-encode an image for API consumption.

    Handles missing/empty files, corrupt images, RGBA/palette conversion to RGB,
    dimension capping, and progressive quality reduction if the encoded result
    still exceeds max_file_bytes.

    Args:
        image_path: Path to the image file on disk.
        max_long_side: Maximum pixel dimension on the longest side.
        max_file_bytes: Maximum encoded JPEG size in bytes.
        jpeg_quality: Starting JPEG quality (will reduce to 70, then 50 if needed).

    Returns:
        (base64_data, media_type) on success, or None if the image is unusable.
    """
    if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
        return None

    try:
        img = Image.open(image_path)
        img.load()  # force full decode to catch corruption
    except (IOError, OSError, Image.UnidentifiedImageError, SyntaxError):
        return None

    # Apply EXIF rotation and strip metadata (prevents PII leaking to APIs)
    try:
        img = ImageOps.exif_transpose(img)
    except (AttributeError, ValueError):
        pass  # No EXIF or malformed — continue without

    # Convert RGBA/palette to RGB for JPEG compatibility
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Resize if dimensions exceed limit
    w, h = img.size
    long_side = max(w, h)
    if long_side > max_long_side:
        ratio = max_long_side / long_side
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    # Encode to JPEG in memory, reducing quality if still too large
    for quality in (jpeg_quality, 70, 50):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, subsampling=0)
        raw_bytes = buf.getvalue()
        if len(raw_bytes) <= max_file_bytes:
            break
    else:
        img.close()
        return None

    img.close()
    data = base64.standard_b64encode(raw_bytes).decode("utf-8")
    return data, "image/jpeg"


def build_image_blocks(
    pages: list[dict],
    max_long_side: int = MAX_IMAGE_LONG_SIDE,
    jpeg_quality: int = JPEG_QUALITY,
) -> list[dict]:
    """
    Build API content blocks for a batch of page images.

    Each page produces a validated, optimized image block followed by a
    ``[Page N]`` text label. Corrupted or oversized images are skipped
    gracefully.

    Args:
        pages: List of dicts, each requiring ``path`` and ``page_number`` keys.
        max_long_side: Maximum pixel dimension for each image.
        jpeg_quality: JPEG compression quality.

    Returns:
        List of content-block dicts (image + text pairs) ready for API use.
    """
    image_blocks: list[dict] = []

    for page in pages:
        result = validate_and_prepare_image(
            page["path"],
            max_long_side=max_long_side,
            jpeg_quality=jpeg_quality,
        )
        if result is None:
            continue

        data, media_type = result
        image_blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        })
        image_blocks.append({
            "type": "text",
            "text": f"[Page {page['page_number']}]",
        })

    return image_blocks


def batch_pages_by_size(
    pages: list[dict],
    max_long_side: int = MAX_IMAGE_LONG_SIDE,
    jpeg_quality: int = JPEG_QUALITY,
    max_batch_bytes: int = MAX_REQUEST_BYTES,
) -> list[list[dict]]:
    """
    Split pages into batches that each fit within an API size limit.

    Pre-encodes each image to measure its actual base64 size, then groups
    pages so no single batch exceeds *max_batch_bytes*. If an individual
    image exceeds the budget, it is re-encoded at reduced quality/resolution
    before grouping.

    Args:
        pages: List of page dicts (must have ``path`` and ``page_number``).
        max_long_side: Maximum pixel dimension per image.
        jpeg_quality: JPEG quality for encoding.
        max_batch_bytes: Maximum total base64 bytes per batch.

    Returns:
        List of page groups, each safe to send in one API call.
    """
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_size = 0

    for page in pages:
        result = validate_and_prepare_image(
            page["path"],
            max_long_side=max_long_side,
            jpeg_quality=jpeg_quality,
        )
        if result is None:
            continue

        b64_data, _ = result
        image_size = len(b64_data)

        # If a single image exceeds the budget, shrink it further
        if image_size > max_batch_bytes:
            result = validate_and_prepare_image(
                page["path"],
                max_long_side=min(max_long_side, 800),
                max_file_bytes=1_500_000,
                jpeg_quality=50,
            )
            if result is None:
                continue
            b64_data, _ = result
            image_size = len(b64_data)

        # Start a new batch if adding this image would exceed the limit
        if current_batch and current_size + image_size > max_batch_bytes:
            batches.append(current_batch)
            current_batch = []
            current_size = 0

        current_batch.append(page)
        current_size += image_size

    if current_batch:
        batches.append(current_batch)

    return batches


def is_blank_page(image_path: str, white_threshold: int = 245, white_ratio: float = 0.97) -> bool:
    """Detect blank/near-blank pages to skip before sending to API.

    A page is considered blank if more than `white_ratio` (default 97%) of
    its pixels are above `white_threshold` (default 245/255 brightness).
    This saves API tokens on filler pages common in student submissions.

    Args:
        image_path: Path to the image file.
        white_threshold: Pixel brightness above which a pixel is "white".
        white_ratio: Fraction of white pixels required to declare blank.

    Returns:
        True if the page appears blank.
    """
    try:
        img = Image.open(image_path)
        img.load()
    except (IOError, OSError, Image.UnidentifiedImageError):
        return False

    gray = img.convert("L")
    # Downsample for speed on large images
    if max(gray.size) > 800:
        ratio = 800 / max(gray.size)
        gray = gray.resize((int(gray.width * ratio), int(gray.height * ratio)))

    hist = gray.histogram()
    total_pixels = sum(hist)
    white_pixels = sum(hist[white_threshold:])
    img.close()

    return (white_pixels / total_pixels) >= white_ratio if total_pixels > 0 else False


def get_adaptive_settings(page_count: int) -> tuple[int, int, int]:
    """
    Select rendering parameters based on the number of pages.

    Walks the DPI_TIERS table (from _constants) and returns the first tier
    whose max_pages threshold accommodates *page_count*. Falls back to the
    most aggressive (lowest quality) tier for very large documents.

    Args:
        page_count: Total number of pages in the document.

    Returns:
        (dpi, jpeg_quality, max_long_side)
    """
    for max_pages, dpi, quality, max_long_side in DPI_TIERS:
        if page_count <= max_pages:
            return dpi, quality, max_long_side
    # Fallback to most aggressive tier
    _, dpi, quality, max_long_side = DPI_TIERS[-1]
    return dpi, quality, max_long_side
