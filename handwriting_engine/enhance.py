"""
Image enhancement pipeline for handwritten submissions.

Provides targeted fixes (sharpen, contrast, brighten, denoise) as well as
higher-level strategies:

* **smart_enhance** -- assessment-driven, only touches detected issues.
* **proven_enhance** -- the PIL pipeline from the BIOL 107 grading session
  that delivered a 15-20% accuracy improvement on faded handwriting.
* **crisp** -- image-crisp presets (light / medium / heavy).
* **enhance_image** -- unified entry point that dispatches to all of the above.

All defaults are imported from ``handwriting_engine._constants`` so they
can be tuned in one place.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore
    _HAS_CV2 = False

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat


# Track paths that have already been enhanced in this process so downstream
# passes can skip redundant work (e.g. PDF rasterization enhances, then
# vision.read_page() would otherwise enhance again).
_HE_ENHANCED_PATHS: set[str] = set()


def mark_enhanced(path: str) -> None:
    """Record that `path` has already been enhanced in this process."""
    if path:
        _HE_ENHANCED_PATHS.add(os.path.abspath(path))


def was_enhanced(path: str) -> bool:
    """Check whether `path` has been enhanced in this process."""
    if not path:
        return False
    return os.path.abspath(path) in _HE_ENHANCED_PATHS


def _require_cv2(func_name: str) -> None:
    if not _HAS_CV2:
        raise ImportError(
            f"{func_name} requires opencv-python and numpy. "
            "Install with: pip install opencv-python numpy"
        )

from handwriting_engine._constants import (
    CRISP_PRESETS,
    PROVEN_AUTOCONTRAST_CUTOFF,
    PROVEN_BRIGHTNESS,
    PROVEN_CONTRAST,
    PROVEN_SHARPEN,
    PROVEN_UPSCALE,
)
from handwriting_engine.quality import assess_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_alpha(img: Image.Image) -> tuple[Image.Image, Image.Image | None]:
    """
    If *img* has an alpha channel, split it off and return
    ``(rgb_image, alpha_channel)``.  Otherwise return ``(img, None)``.
    """
    if img.mode == "RGBA":
        alpha = img.getchannel("A")
        return img.convert("RGB"), alpha
    return img if img.mode == "RGB" else img.convert("RGB"), None


def _restore_alpha(img: Image.Image, alpha: Image.Image | None) -> Image.Image:
    """Re-attach *alpha* (if not ``None``) to *img*, resizing alpha to match."""
    if alpha is None:
        return img
    if alpha.size != img.size:
        alpha = alpha.resize(img.size, Image.LANCZOS)
    img = img.convert("RGBA")
    img.putalpha(alpha)
    return img


def _save(img: Image.Image, output_path: str) -> None:
    """Save with format inferred from extension, preserving RGBA for PNG.
    Strips EXIF metadata to prevent PII leakage (GPS, device info)."""
    # Strip EXIF before saving
    img.info.pop("exif", None)
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".png":
        img.save(output_path)
    else:
        # JPEG does not support alpha -- convert if necessary
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(output_path, "JPEG", quality=95, subsampling=0, exif=b"")


# ---------------------------------------------------------------------------
# Targeted enhancement building blocks
# ---------------------------------------------------------------------------


def _enhance_sharpen(img: Image.Image, heavy: bool = False) -> Image.Image:
    """Unsharp mask + DETAIL filter.  Preserves the current mode."""
    if heavy:
        img = img.filter(ImageFilter.UnsharpMask(radius=3.0, percent=200, threshold=0))
        img = img.filter(ImageFilter.DETAIL)
        img = img.filter(ImageFilter.DETAIL)
        img = ImageEnhance.Sharpness(img).enhance(1.3)
    else:
        img = img.filter(ImageFilter.UnsharpMask(radius=2.5, percent=180, threshold=0))
        img = img.filter(ImageFilter.DETAIL)
        img = ImageEnhance.Sharpness(img).enhance(1.2)
    return img


def _enhance_contrast(img: Image.Image, heavy: bool = False) -> Image.Image:
    """Auto-contrast followed by a contrast boost."""
    img = ImageOps.autocontrast(img, cutoff=2)
    factor = 1.25 if heavy else 1.15
    img = ImageEnhance.Contrast(img).enhance(factor)
    return img


def _enhance_brighten(img: Image.Image, heavy: bool = False) -> Image.Image:
    """Normalise brightness toward a white-paper baseline (~180 mean)."""
    gray = img.convert("L")
    current_mean = ImageStat.Stat(gray).mean[0]
    target = 180
    if current_mean < 10:
        current_mean = 10
    factor = target / current_mean
    factor = max(0.5, min(factor, 2.5 if heavy else 2.0))
    img = ImageEnhance.Brightness(img).enhance(factor)
    return img


def _enhance_denoise(img: Image.Image) -> Image.Image:
    """Light median filter to reduce salt-and-pepper noise."""
    return img.filter(ImageFilter.MedianFilter(size=3))


def _deskew(img: Image.Image, angle: float) -> Image.Image:
    """Rotate image to correct detected skew. Fills exposed corners with white."""
    if abs(angle) < 0.1:
        return img
    return img.rotate(angle, expand=True, fillcolor=(255, 255, 255) if img.mode == "RGB" else 255)


# ---------------------------------------------------------------------------
# High-level strategies
# ---------------------------------------------------------------------------


def smart_enhance(
    image_path: str,
    assessment: dict | None = None,
    output_path: str | None = None,
) -> str:
    """
    Apply targeted enhancement based on a quality assessment.

    Only touches the specific issues detected -- does not blindly sharpen
    or boost contrast on images that are already good.

    Parameters
    ----------
    image_path : str
        Path to the source image.
    assessment : dict, optional
        Pre-computed :func:`~handwriting_engine.quality.assess_image` result.
        Will be computed on the fly if not provided.
    output_path : str, optional
        Where to save the result.  Defaults to *image_path* (in-place).

    Returns
    -------
    str
        The path to the (possibly enhanced) output image.
    """
    if assessment is None:
        assessment = assess_image(image_path)

    strategy = assessment.get("recommended_strategy")
    if strategy is None:
        return image_path  # already good

    img = Image.open(image_path)
    img, alpha = _strip_alpha(img)

    # Deskew first — straighten before any other enhancement
    skew_angle = assessment.get("skew_angle", 0.0)
    if skew_angle:
        img = _deskew(img, skew_angle)

    heavy = assessment["quality"] == "poor"
    faint = assessment.get("faint_ink", False)

    if strategy == "full":
        img = _enhance_denoise(img)
        img = _enhance_contrast(img, heavy=heavy)
        if faint:
            img = _enhance_contrast(img, heavy=True)
        img = _enhance_sharpen(img, heavy=heavy)
        img = _enhance_brighten(img, heavy=heavy)
    elif strategy == "sharpen":
        img = _enhance_sharpen(img, heavy=heavy)
    elif strategy == "contrast":
        img = _enhance_contrast(img, heavy=heavy)
        if faint:
            img = _enhance_contrast(img, heavy=True)
    elif strategy == "brighten":
        img = _enhance_brighten(img, heavy=heavy)

    img = _restore_alpha(img, alpha)

    dest = output_path or image_path
    _save(img, dest)
    img.close()
    return dest


def proven_enhance(
    image_path: str,
    output_path: Optional[str] = None,
    sharpen: float = PROVEN_SHARPEN,
    contrast: float = PROVEN_CONTRAST,
    brightness: float = PROVEN_BRIGHTNESS,
    autocontrast_cutoff: int = PROVEN_AUTOCONTRAST_CUTOFF,
    upscale: int = PROVEN_UPSCALE,
    preserve_color: bool = False,
) -> str:
    """
    The proven PIL pipeline from the BIOL 107 grading session (15-20%
    accuracy improvement on faded/light handwriting).

    Pipeline order: grayscale -> autocontrast -> sharpen -> contrast ->
    brightness -> upscale.

    Parameters
    ----------
    image_path : str
        Path to the source image.
    output_path : str, optional
        Where to save the result.  Defaults to *image_path* (in-place).
    sharpen : float
        Sharpness enhancement factor.
    contrast : float
        Contrast enhancement factor.
    brightness : float
        Brightness enhancement factor.
    autocontrast_cutoff : int
        Percentile cutoff for :func:`PIL.ImageOps.autocontrast`.
    upscale : int
        Integer scale factor for the final resize (1 = no upscale).
    preserve_color : bool
        If True, skip grayscale conversion. Use when color carries meaning
        (red-pen corrections, highlighted text, multi-ink annotations).

    Returns
    -------
    str
        The path to the enhanced output image.
    """
    img = Image.open(image_path)
    img, alpha = _strip_alpha(img)

    if preserve_color:
        # Apply the same pipeline on RGB — autocontrast works per-channel
        img = ImageOps.autocontrast(img, cutoff=autocontrast_cutoff)
    else:
        # 1. Grayscale
        img = img.convert("L")
        # 2. Autocontrast
        img = ImageOps.autocontrast(img, cutoff=autocontrast_cutoff)

    # 3. Sharpen
    img = ImageEnhance.Sharpness(img).enhance(sharpen)

    # 4. Contrast
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # 5. Brightness
    img = ImageEnhance.Brightness(img).enhance(brightness)

    # 6. Upscale
    if upscale and upscale > 1:
        if upscale > 8:
            raise ValueError(f"Upscale factor {upscale} too large (max 8)")
        new_size = (img.width * upscale, img.height * upscale)
        img = img.resize(new_size, Image.LANCZOS)

    # Ensure RGB output
    if img.mode == "L":
        img = img.convert("RGB")

    img = _restore_alpha(img, alpha)

    dest = output_path or image_path
    _save(img, dest)
    img.close()
    return dest


def crisp(
    image_path: str,
    strength: str = "medium",
    output_path: Optional[str] = None,
) -> str:
    """
    Apply image-crisp sharpen presets (light / medium / heavy).

    Uses ``CRISP_PRESETS`` from ``_constants.py`` which define tuples of
    ``(unsharp_radius, unsharp_percent, unsharp_threshold, detail_passes,
    contrast_factor)``.

    Parameters
    ----------
    image_path : str
        Path to the source image.
    strength : str
        One of ``"light"``, ``"medium"``, or ``"heavy"``.
    output_path : str, optional
        Where to save the result.  Defaults to *image_path* (in-place).

    Returns
    -------
    str
        The path to the enhanced output image.

    Raises
    ------
    ValueError
        If *strength* is not a recognised preset name.
    """
    if strength not in CRISP_PRESETS:
        raise ValueError(
            f"Unknown crisp strength {strength!r}. "
            f"Choose from: {', '.join(sorted(CRISP_PRESETS))}"
        )

    radius, percent, threshold, detail_passes, contrast_factor = CRISP_PRESETS[strength]

    img = Image.open(image_path)
    img, alpha = _strip_alpha(img)

    # Unsharp mask
    img = img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

    # Detail passes
    for _ in range(detail_passes):
        img = img.filter(ImageFilter.DETAIL)

    # Contrast boost
    if contrast_factor != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    img = _restore_alpha(img, alpha)

    dest = output_path or image_path
    _save(img, dest)
    img.close()
    return dest


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def llm_enhance(
    image_path: str,
    output_path: Optional[str] = None,
) -> str:
    """Light enhancement optimized for LLM vision models.

    Unlike the proven pipeline, this preserves grayscale nuance that LLMs
    use for character interpretation. Research (PreP-OCR, ACL 2025) shows
    heavy preprocessing barely helps and can hurt LLM-based OCR.

    Pipeline:
    - Gentle autocontrast (cutoff=1, not 2)
    - Light sharpen (1.5, not 3.0)
    - NO grayscale conversion (preserves color context)
    - NO heavy contrast boost
    - Upscale only if < 800px long side
    """
    img = Image.open(image_path)
    img, alpha = _strip_alpha(img)

    # Gentle autocontrast — just normalize without crushing
    img = ImageOps.autocontrast(img, cutoff=1)

    # Light sharpen — enough to define strokes, not enough to create artifacts
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)

    # Only upscale if the image is small
    w, h = img.size
    if max(w, h) < 800:
        scale = 1200 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    img = _restore_alpha(img, alpha)
    dest = output_path or image_path
    _save(img, dest)
    img.close()
    return dest


def adaptive_enhance(
    image_path: str,
    assessment: dict | None = None,
    output_path: str | None = None,
) -> str:
    """Quality-adaptive enhancement using per-issue-type parameters.

    Unlike smart_enhance (which maps to fixed strategies), this uses
    fine-grained parameters from recommend_enhancement_params() to
    apply exactly the right treatment for each image's specific issues.
    """
    from handwriting_engine.quality import recommend_enhancement_params

    if assessment is None:
        assessment = assess_image(image_path)

    params = recommend_enhancement_params(assessment)

    if params.get("strategy") == "llm":
        return llm_enhance(image_path, output_path=output_path)

    img = Image.open(image_path)
    img, alpha = _strip_alpha(img)

    # Deskew first — straighten before any other enhancement
    if assessment.get("skew_angle"):
        img = _deskew(img, assessment["skew_angle"])

    if params.get("denoise"):
        img = _enhance_denoise(img)

    if params.get("contrast"):
        cutoff = params.get("autocontrast_cutoff", 2)
        factor = params.get("contrast_factor", 1.15)
        img = ImageOps.autocontrast(img, cutoff=cutoff)
        img = ImageEnhance.Contrast(img).enhance(factor)

    if params.get("sharpen"):
        factor = params.get("sharpen_factor", 2.0)
        img = img.filter(ImageFilter.UnsharpMask(radius=2.0, percent=int(factor * 60), threshold=0))
        img = ImageEnhance.Sharpness(img).enhance(min(factor * 0.5, 1.5))

    if params.get("brighten"):
        img = _enhance_brighten(img, heavy=False)

    img = _restore_alpha(img, alpha)
    dest = output_path or image_path
    _save(img, dest)
    img.close()
    return dest


# ---------------------------------------------------------------------------
# CLAHE + Lined Paper Removal
# ---------------------------------------------------------------------------


def clahe_enhance(
    image_path: str,
    output_path: Optional[str] = None,
    clip_limit: float = 3.0,
    tile_grid: int = 8,
) -> str:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement.

    Handles non-uniform illumination from flatbed scanners far better than
    global contrast stretching.  Primary use: faint ink on non-uniform
    backgrounds.

    Parameters
    ----------
    clip_limit : float
        2.0 for mild faint, 3.0 for clearly faint, 4.0 for very faint pencil.
    tile_grid : int
        Grid size for local histogram regions.  8 for full pages, 4 for
        narrow crops like name fields.
    """
    _require_cv2("clahe_enhance")
    img = Image.open(image_path)
    img_rgb, alpha = _strip_alpha(img)

    arr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid)
    )
    enhanced = clahe.apply(arr)

    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    result = Image.fromarray(enhanced_rgb)

    # Light unsharp mask to define stroke edges without noise amplification
    result = result.filter(
        ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=1)
    )

    result = _restore_alpha(result, alpha)
    dest = output_path or image_path
    _save(result, dest)
    return dest


def sauvola_enhance(
    image_path: str,
    output_path: Optional[str] = None,
    window_size: int = 25,
    k: float = 0.2,
) -> str:
    """Sauvola adaptive binarization — best for degraded docs with uneven illumination.

    MDPI Electronics 2024 review: adaptive local methods (Sauvola, Niblack)
    outperform global Otsu for documents with non-uniform backgrounds.

    Args:
        window_size: Local window size for threshold computation (odd number).
        k: Sauvola sensitivity parameter (0.1 = less aggressive, 0.5 = more).

    Falls back to clahe_enhance() if scikit-image is not installed.
    """
    _require_cv2("sauvola_enhance")
    try:
        from skimage.filters import threshold_sauvola
    except ImportError:
        logger.warning("scikit-image not installed, falling back to clahe_enhance")
        return clahe_enhance(image_path, output_path=output_path)

    img = Image.open(image_path)
    img_rgb, alpha = _strip_alpha(img)

    # Convert to grayscale for Sauvola
    gray = np.array(img_rgb.convert("L"))

    # Compute Sauvola threshold map
    thresh = threshold_sauvola(gray, window_size=window_size, k=k)
    binary = (gray > thresh).astype(np.uint8) * 255

    # Convert binary back to RGB
    result_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    result = Image.fromarray(result_rgb)

    result = _restore_alpha(result, alpha)
    dest = output_path or image_path
    _save(result, dest)
    return dest


def remove_horizontal_lines(
    image_path: str,
    output_path: Optional[str] = None,
    line_length_ratio: float = 0.5,
) -> str:
    """
    Remove ruled paper lines while preserving handwritten strokes.

    Uses morphological opening to detect horizontal structures spanning at
    least *line_length_ratio* of the page width, then subtracts them.
    A small vertical dilation repairs letter strokes cut by line removal.

    Parameters
    ----------
    line_length_ratio : float
        Minimum fraction of page width for a structure to count as a
        ruling line.  Default 0.5 safely preserves letter crossbars.
    """
    _require_cv2("remove_horizontal_lines")
    img = Image.open(image_path)
    img_rgb, alpha = _strip_alpha(img)

    arr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(
        arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    h, w = binary.shape
    kernel_w = max(30, int(w * line_length_ratio))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)

    # Repair strokes cut by line removal
    repair = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    lines = cv2.dilate(lines, repair, iterations=1)

    result_gray = cv2.add(arr, lines)
    result_rgb = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2RGB)
    result = Image.fromarray(result_rgb)

    result = _restore_alpha(result, alpha)
    dest = output_path or image_path
    _save(result, dest)
    return dest


def enhance_image(
    image_path: str,
    strategy: str = "smart",
    output_path: Optional[str] = None,
) -> str:
    """
    Unified enhancement entry point.

    Parameters
    ----------
    image_path : str
        Path to the source image.
    strategy : str
        One of:

        * ``"smart"`` -- assessment-driven targeted enhancement.
        * ``"adaptive"`` -- quality-adaptive with per-issue parameters.
        * ``"proven"`` -- the BIOL 107 proven pipeline.
        * ``"light"`` / ``"medium"`` / ``"heavy"`` -- crisp presets.
        * ``"full"`` -- full enhancement (denoise + contrast + sharpen +
          brighten) regardless of assessment.
    output_path : str, optional
        Where to save the result.  Defaults to *image_path* (in-place).

    Returns
    -------
    str
        The path to the enhanced output image.

    Raises
    ------
    ValueError
        If *strategy* is not recognised.
    """
    if strategy == "smart":
        result = smart_enhance(image_path, output_path=output_path)
    elif strategy == "adaptive":
        result = adaptive_enhance(image_path, output_path=output_path)
    elif strategy == "proven":
        result = proven_enhance(image_path, output_path=output_path)
    elif strategy == "llm":
        result = llm_enhance(image_path, output_path=output_path)
    elif strategy == "clahe":
        result = clahe_enhance(image_path, output_path=output_path)
    elif strategy in ("light", "medium", "heavy"):
        result = crisp(image_path, strength=strategy, output_path=output_path)
    elif strategy == "full":
        forced_assessment = {
            "quality": "poor",
            "recommended_strategy": "full",
            "faint_ink": False,
        }
        result = smart_enhance(image_path, assessment=forced_assessment, output_path=output_path)
    elif strategy == "sauvola":
        result = sauvola_enhance(image_path, output_path=output_path)
    else:
        raise ValueError(
            f"Unknown enhancement strategy {strategy!r}. "
            f"Choose from: smart, adaptive, proven, llm, clahe, sauvola, light, medium, heavy, full"
        )

    mark_enhanced(result)
    return result
