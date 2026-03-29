"""
Unified vision API — single entry point for reading handwriting.
Supports single-model reads, multi-model consensus, and two-pass strategies.
Delegates to providers/ for actual API calls.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from handwriting_engine._constants import MAX_IMAGE_LONG_SIDE, JPEG_QUALITY
from handwriting_engine.providers import get_provider, available_providers
from handwriting_engine.providers.base import ConsensusResult

logger = logging.getLogger(__name__)


def read_page(
    image_path: str,
    prompt: str = "",
    domain: str = "biology",
    provider: str = "claude",
    system_prompt: str = "",
    max_tokens: int = 4096,
    inject_strategies: bool = True,
    auto_enhance: bool = False,
    inject_lessons: bool = True,
    writer_id: str | None = None,
) -> str:
    """
    Read a single handwritten page.
    Injects reading strategies and disambiguation rules by default.

    Args:
        auto_enhance: If True, assess image quality and enhance if needed
                      before reading. Uses 'proven' strategy for poor images.
        inject_lessons: If True, prepend learned lessons from past corrections
                        into the system prompt.
    """
    from handwriting_engine.optimize import validate_and_prepare_image
    from handwriting_engine.handwriting import get_reading_strategies

    assessment = None
    if auto_enhance:
        import tempfile
        from handwriting_engine.quality import assess_image
        from handwriting_engine.enhance import enhance_image
        assessment = assess_image(image_path)
        if assessment["quality"] != "good":
            suffix = os.path.splitext(image_path)[1] or ".jpg"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.close()
            image_path = enhance_image(image_path, strategy="adaptive", output_path=tmp.name)

    # Prepare faint content protocol injection if quality assessment detects faint ink
    if assessment and assessment.get("faint_ink"):
        from handwriting_engine.handwriting import FAINT_CONTENT_PROTOCOL
        faint_protocol = FAINT_CONTENT_PROTOCOL
    else:
        faint_protocol = ""

    result = validate_and_prepare_image(image_path)
    if result is None:
        logger.error(f"Could not process image: {image_path}")
        return ""

    b64_data, media_type = result

    if not prompt:
        prompt = (
            "Read all handwritten text in this image carefully. "
            "Preserve original spelling and formatting exactly as written. "
            "For any character you are less than 80% sure about, mark it with [?]. "
            "If a region is completely illegible, write [illegible: ~N chars]."
        )

    # Reading strategies go in system prompt (enables prompt caching, better model attention)
    if inject_strategies:
        strategies = get_reading_strategies(domain=domain, content_type="default")
        system_prompt = strategies + ("\n\n" + system_prompt if system_prompt else "")

    # Inject faint content protocol when image has faint ink
    if faint_protocol:
        system_prompt = (system_prompt + "\n\n" + faint_protocol) if system_prompt else faint_protocol

    # Inject learned lessons from past corrections
    if inject_lessons:
        from handwriting_engine.lessons import build_lessons_prompt
        lessons = build_lessons_prompt(category=domain)
        if lessons:
            system_prompt = system_prompt + "\n\n" + lessons if system_prompt else lessons

    # Inject writer-specific calibration if available
    if writer_id:
        from handwriting_engine.lessons import load_writer_calibration
        writer_cal = load_writer_calibration(writer_id)
        if writer_cal:
            system_prompt = (system_prompt + "\n\n" + writer_cal) if system_prompt else writer_cal

    p = get_provider(provider)
    return p.read_image(b64_data, media_type, prompt, system_prompt, max_tokens)


def read_all_pages(
    page_images: list[str | dict],
    prompt: str = "",
    domain: str = "biology",
    provider: str = "claude",
    system_prompt: str = "",
    delay: float = 2.0,
) -> list[dict]:
    """
    Read multiple pages sequentially.
    Returns [{page_number, path, transcription}, ...]
    """
    import time

    results = []
    for i, page in enumerate(page_images):
        if isinstance(page, dict):
            path = page.get("path", page.get("image_path", ""))
            page_num = page.get("page_number", i + 1)
        else:
            path = page
            page_num = i + 1

        transcription = read_page(path, prompt=prompt, domain=domain, provider=provider, system_prompt=system_prompt)
        results.append({"page_number": page_num, "path": path, "transcription": transcription})

        if i < len(page_images) - 1 and delay > 0:
            time.sleep(delay)

    return results


def read_with_consensus(
    image_path: str,
    prompt: str = "",
    domain: str = "biology",
    providers: list[str] | None = None,
    strategy: str = "vote",
    content_type: str = "default",
    inject_strategies: bool = True,
    inject_lessons: bool = True,
    writer_id: str | None = None,
) -> ConsensusResult:
    """
    Read a page using multi-model consensus.
    """
    from handwriting_engine.optimize import validate_and_prepare_image
    from handwriting_engine.handwriting import get_reading_strategies
    from handwriting_engine.consensus import read_with_consensus as _consensus

    result = validate_and_prepare_image(image_path)
    if result is None:
        return ConsensusResult(text="", confidence=0.0)

    b64_data, media_type = result

    if not prompt:
        prompt = (
            "Read all handwritten text in this image carefully. "
            "Preserve original spelling and formatting exactly as written. "
            "For any character you are less than 80% sure about, mark it with [?]. "
            "If a region is completely illegible, write [illegible: ~N chars]."
        )

    # Reading strategies go in system prompt for better model attention + caching
    system_prompt = ""
    if inject_strategies:
        strategies = get_reading_strategies(domain=domain, content_type=content_type)
        system_prompt = strategies

    # Inject learned lessons from past corrections
    if inject_lessons:
        from handwriting_engine.lessons import build_lessons_prompt
        lessons = build_lessons_prompt(category=domain)
        if lessons:
            system_prompt = system_prompt + "\n\n" + lessons if system_prompt else lessons

    # Inject writer-specific calibration if available
    if writer_id:
        from handwriting_engine.lessons import load_writer_calibration
        writer_cal = load_writer_calibration(writer_id)
        if writer_cal:
            system_prompt = (system_prompt + "\n\n" + writer_cal) if system_prompt else writer_cal

    return _consensus(
        image_b64=b64_data,
        media_type=media_type,
        prompt=prompt,
        system_prompt=system_prompt,
        providers=providers,
        strategy=strategy,
        content_type=content_type,
    )


def read_batch_with_vision(
    page_images: list[str | dict],
    prompt: str,
    provider: str = "claude",
    system_prompt: str = "",
    max_tokens: int = 8192,
) -> str:
    """
    Read multiple pages in a single API call (batched).
    Uses optimize.build_image_blocks for proper sizing.
    """
    from handwriting_engine.optimize import build_image_blocks

    pages = []
    for i, p in enumerate(page_images):
        if isinstance(p, dict):
            pages.append(p)
        else:
            pages.append({"path": p, "page_number": i + 1})

    image_blocks = build_image_blocks(pages)
    if not image_blocks:
        return ""

    p = get_provider(provider)
    return p.read_batch(image_blocks, prompt, system_prompt, max_tokens)


def read_with_enhancement_comparison(
    image_path: str,
    domain: str = "biology",
    provider: str = "gemini",
    enhance_strategy: str = "proven",
) -> str:
    """Read both original and enhanced image, return the better result.

    Runs two reads (original + enhanced) and picks whichever has fewer
    uncertainty markers and more substantive content. Costs 2x but
    catches cases where enhancement helps or hurts.
    """
    from handwriting_engine.enhance import enhance_image
    from handwriting_engine.consensus import _estimate_confidence

    original_text = read_page(image_path, domain=domain, provider=provider)

    import tempfile
    suffix = os.path.splitext(image_path)[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    enhanced_path = enhance_image(image_path, strategy=enhance_strategy, output_path=tmp.name)
    enhanced_text = read_page(enhanced_path, domain=domain, provider=provider)

    orig_confidence = _estimate_confidence(original_text)
    enh_confidence = _estimate_confidence(enhanced_text)

    if enh_confidence > orig_confidence:
        logger.debug(f"Enhanced version chosen (confidence {enh_confidence:.2f} > {orig_confidence:.2f})")
        return enhanced_text
    else:
        logger.debug(f"Original version chosen (confidence {orig_confidence:.2f} >= {enh_confidence:.2f})")
        return original_text


def new_usage_tracker() -> dict:
    """Create a fresh token usage tracker."""
    return {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def print_usage(tracker: dict, label: str = ""):
    """Print token usage summary."""
    prefix = f"[{label}] " if label else ""
    inp = tracker.get("input_tokens", 0)
    out = tracker.get("output_tokens", 0)
    calls = tracker.get("calls", 0)
    print(f"{prefix}Tokens: {inp:,} in + {out:,} out = {inp + out:,} total ({calls} calls)")
