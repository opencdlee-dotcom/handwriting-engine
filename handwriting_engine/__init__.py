"""
Handwriting Engine — Unified handwriting recognition with multi-model consensus.

Usage:
    from handwriting_engine import proven_enhance, assess_image, convert_pdf, read_page
"""

__version__ = "0.1.0"

# Security: limit decompression bomb threshold
from PIL import Image as _Image
_Image.MAX_IMAGE_PIXELS = 178_956_970  # ~179 megapixels (PIL default)

# Quality assessment
from handwriting_engine.quality import (
    assess_image,
    assess_blur,
    assess_contrast,
    assess_brightness,
    batch_assess,
    recommend_enhancement_params,
    classify_handwriting_style,
)

# Image enhancement
from handwriting_engine.enhance import (
    enhance_image,
    smart_enhance,
    proven_enhance,
    crisp,
    llm_enhance,
    adaptive_enhance,
)

# PDF conversion
from handwriting_engine.pdf import (
    convert_pdf,
    convert_pdf_survey,
    get_page_count,
    detect_and_fix_orientation,
    cleanup_images,
)

# API image optimization
from handwriting_engine.optimize import (
    validate_and_prepare_image,
    build_image_blocks,
    batch_pages_by_size,
    get_adaptive_settings,
    is_blank_page,
)

# Answer cropping
from handwriting_engine.crop import (
    crop_answer_sheet,
    crop_and_enhance,
    AnswerSheetLayout,
)

# Reading strategies
from handwriting_engine.handwriting import (
    get_reading_strategies,
    get_grading_handwriting_rules,
    get_disambiguation_pairs,
    CHARACTER_DISAMBIGUATION,
)

# Data models
from handwriting_engine.models import (
    PageInfo,
    ImageQuality,
    PageOrientation,
    ContentType,
    ConsensusResult,
    HandwritingEngineError,
    ProviderError,
    ImageError,
    ConfigError,
)

# Prompt adaptation
from handwriting_engine.prompt_adapter import adapt_system_prompt, adapt_user_prompt

# Vision (lazy — doesn't fail if no API keys)
def read_page(*args, **kwargs):
    from handwriting_engine.vision import read_page as _rp
    return _rp(*args, **kwargs)

def read_with_consensus(*args, **kwargs):
    from handwriting_engine.vision import read_with_consensus as _rc
    return _rc(*args, **kwargs)

# Writer identification (lazy — needs google-genai)
def identify_writer(*args, **kwargs):
    from handwriting_engine.writer_embeddings import identify_writer as _iw
    return _iw(*args, **kwargs)

def enroll_writer(*args, **kwargs):
    from handwriting_engine.writer_embeddings import enroll_writer as _ew
    return _ew(*args, **kwargs)

__all__ = [
    # Quality
    "assess_image", "assess_blur", "assess_contrast", "assess_brightness", "batch_assess",
    "recommend_enhancement_params", "classify_handwriting_style",
    # Enhancement
    "enhance_image", "smart_enhance", "proven_enhance", "crisp", "llm_enhance", "adaptive_enhance",
    # PDF
    "convert_pdf", "convert_pdf_survey", "get_page_count", "detect_and_fix_orientation", "cleanup_images",
    # Optimization
    "validate_and_prepare_image", "build_image_blocks", "batch_pages_by_size", "get_adaptive_settings", "is_blank_page",
    # Cropping
    "crop_answer_sheet", "crop_and_enhance", "AnswerSheetLayout",
    # Handwriting
    "get_reading_strategies", "get_grading_handwriting_rules", "get_disambiguation_pairs", "CHARACTER_DISAMBIGUATION",
    # Models & Exceptions
    "PageInfo", "ImageQuality", "PageOrientation", "ContentType", "ConsensusResult",
    "HandwritingEngineError", "ProviderError", "ImageError", "ConfigError",
    # Prompt adaptation
    "adapt_system_prompt", "adapt_user_prompt",
    # Vision
    "read_page", "read_with_consensus",
    # Writer identification
    "identify_writer", "enroll_writer",
    # Benchmark (subpackage)
    "benchmark",
]
