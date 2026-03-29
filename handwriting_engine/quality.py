"""
Image quality assessment for handwritten submissions.

Detects poor-quality page scans (blur, low contrast, bad brightness) and
reports issues with recommended enhancement strategies. Ported from
Exam grader's image_quality.py with batch assessment support.

All thresholds are imported from handwriting_engine._constants so consumers
can override them in one place.
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

from handwriting_engine._constants import (
    BLUR_THRESHOLD,
    BRIGHTNESS_HIGH,
    BRIGHTNESS_LOW,
    CONTRAST_THRESHOLD,
    FAINT_INK_BRIGHTNESS,
    FAINT_INK_CONTRAST,
)


# ---------------------------------------------------------------------------
# Individual quality metrics
# ---------------------------------------------------------------------------


def assess_blur(img: Image.Image) -> float:
    """
    Approximate Laplacian variance using a 3x3 edge-detection kernel.

    Higher variance means a sharper image.  Values below
    ``BLUR_THRESHOLD`` (default 100) indicate a blurry scan.
    """
    gray = img.convert("L")
    laplacian = ImageFilter.Kernel(
        size=(3, 3),
        kernel=[0, 1, 0, 1, -4, 1, 0, 1, 0],
        scale=1,
        offset=128,
    )
    filtered = gray.filter(laplacian)
    stat = ImageStat.Stat(filtered)
    return round(stat.var[0], 2)


def assess_contrast(img: Image.Image) -> float:
    """
    Contrast as the normalised spread between the 5th and 95th percentile
    pixel intensities.

    Returns a value in [0, 1] where higher means more contrast.
    Values below ``CONTRAST_THRESHOLD`` (default 0.4) indicate low contrast.
    """
    gray = img.convert("L")
    hist = gray.histogram()
    total = sum(hist)

    cumulative = 0
    p5 = 0
    for i, count in enumerate(hist):
        cumulative += count
        if cumulative >= total * 0.05:
            p5 = i
            break

    cumulative = 0
    p95 = 255
    for i, count in enumerate(hist):
        cumulative += count
        if cumulative >= total * 0.95:
            p95 = i
            break

    return round((p95 - p5) / 255.0, 3)


def assess_brightness(img: Image.Image) -> tuple[float, float]:
    """Return ``(mean, stddev)`` of pixel brightness."""
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    return round(stat.mean[0], 1), round(stat.stddev[0], 1)


# ---------------------------------------------------------------------------
# Full assessment
# ---------------------------------------------------------------------------


def assess_image(image_path: str) -> dict:
    """
    Run a full quality assessment on a single image.

    Returns a dict with scores, detected issues, an overall quality rating
    (``good`` / ``fair`` / ``poor``), and a recommended enhancement strategy.
    """
    try:
        img = Image.open(image_path)
        img.load()  # Force full decode to catch corruption
    except (IOError, OSError, Image.UnidentifiedImageError, SyntaxError, Image.DecompressionBombError):
        return {
            "path": image_path,
            "blur_score": 0,
            "contrast_score": 0,
            "brightness_mean": 0,
            "brightness_std": 0,
            "issues": ["corrupt_image"],
            "quality": "poor",
            "faint_ink": False,
            "recommended_strategy": None,
            "needs_enhanced_reading": False,
        }

    blur = assess_blur(img)
    contrast = assess_contrast(img)
    bright_mean, bright_std = assess_brightness(img)

    issues: list[str] = []
    if blur < BLUR_THRESHOLD:
        issues.append("blurry")
    if contrast < CONTRAST_THRESHOLD:
        issues.append("low_contrast")
    if bright_mean < BRIGHTNESS_LOW:
        issues.append("too_dark")
    elif bright_mean > BRIGHTNESS_HIGH:
        issues.append("too_bright")

    if not issues:
        quality = "good"
    elif len(issues) == 1 and blur >= (BLUR_THRESHOLD / 2) and contrast >= (CONTRAST_THRESHOLD * 0.625):
        quality = "fair"
    else:
        quality = "poor"

    # Pick a targeted strategy
    if quality == "good":
        strategy = None
    elif set(issues) == {"blurry"}:
        strategy = "sharpen"
    elif set(issues) == {"low_contrast"}:
        strategy = "contrast"
    elif "too_dark" in issues or "too_bright" in issues:
        strategy = "brighten" if len(issues) == 1 else "full"
    else:
        strategy = "full"

    # Faint-ink detection: high brightness + low contrast = washed-out handwriting
    faint = bright_mean > FAINT_INK_BRIGHTNESS and contrast < FAINT_INK_CONTRAST
    if faint and "low_contrast" not in issues:
        issues.append("faint_ink")

    img.close()

    return {
        "path": image_path,
        "blur_score": blur,
        "contrast_score": contrast,
        "brightness_mean": bright_mean,
        "brightness_std": bright_std,
        "issues": issues,
        "quality": quality,
        "faint_ink": faint,
        "recommended_strategy": strategy,
        "needs_enhanced_reading": quality == "poor" or faint,
    }


def recommend_enhancement_params(assessment: dict) -> dict:
    """Return specific enhancement parameters based on quality assessment.

    Different issues need different treatment:
    - Faint ink: heavy autocontrast, NO sharpen (amplifies noise)
    - Blurry: sharpen + light denoise, NO contrast boost (amplifies artifacts)
    - Low contrast only: autocontrast + moderate contrast, preserve color
    - Good quality: llm_enhance only (heavy preprocessing hurts LLM OCR)
    """
    issues = set(assessment.get("issues", []))
    quality = assessment.get("quality", "good")
    faint = assessment.get("faint_ink", False)

    if quality == "good":
        return {"strategy": "llm", "reason": "good quality — light touch only"}

    params = {
        "denoise": False,
        "sharpen": False,
        "sharpen_factor": 1.0,
        "contrast": False,
        "contrast_factor": 1.0,
        "autocontrast_cutoff": 2,
        "brighten": False,
        "preserve_color": True,  # Default to preserving color for LLM reads
        "reason": "",
    }

    if faint or "faint_ink" in issues:
        params["contrast"] = True
        params["contrast_factor"] = 2.0
        params["autocontrast_cutoff"] = 3
        params["brighten"] = True
        # NO sharpen for faint — amplifies noise in washed-out images
        params["reason"] = "faint ink — heavy contrast, no sharpen"
    elif "blurry" in issues and "low_contrast" not in issues:
        params["sharpen"] = True
        params["sharpen_factor"] = 2.5
        params["denoise"] = True
        # NO contrast boost for blur — amplifies blur artifacts
        params["reason"] = "blurry — sharpen + denoise, no contrast"
    elif "low_contrast" in issues and "blurry" not in issues:
        params["contrast"] = True
        params["contrast_factor"] = 1.5
        params["autocontrast_cutoff"] = 2
        params["reason"] = "low contrast — moderate contrast boost"
    elif "too_dark" in issues:
        params["brighten"] = True
        params["contrast"] = True
        params["contrast_factor"] = 1.3
        params["reason"] = "too dark — brighten + light contrast"
    elif "too_bright" in issues:
        params["contrast"] = True
        params["contrast_factor"] = 1.5
        params["autocontrast_cutoff"] = 3
        params["reason"] = "too bright — contrast boost to recover detail"
    else:
        # Multiple issues — full treatment
        params["denoise"] = True
        params["sharpen"] = True
        params["sharpen_factor"] = 2.0
        params["contrast"] = True
        params["contrast_factor"] = 1.5 if quality == "fair" else 2.0
        params["brighten"] = True
        params["reason"] = f"multiple issues ({', '.join(issues)}) — full treatment"

    params["strategy"] = "adaptive"
    return params


# ---------------------------------------------------------------------------
# Batch assessment
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def batch_assess(directory: str) -> dict[str, dict]:
    """
    Assess every image file in *directory* (non-recursive).

    Returns a manifest dict mapping each image path to its
    :func:`assess_image` result::

        {
            "/path/to/page_001.jpg": { ... assessment dict ... },
            ...
        }

    Only files whose extension matches a common image format are included.
    """
    directory = os.path.abspath(directory)
    manifest: dict[str, dict] = {}

    for entry in sorted(Path(directory).iterdir()):
        if entry.is_file() and entry.suffix.lower() in _IMAGE_EXTENSIONS:
            path_str = str(entry)
            manifest[path_str] = assess_image(path_str)

    return manifest
