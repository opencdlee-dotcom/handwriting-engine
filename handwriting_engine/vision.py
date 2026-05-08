"""
Unified vision API — single entry point for reading handwriting.
Supports single-model reads, multi-model consensus, and two-pass strategies.
Delegates to providers/ for actual API calls.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import re

from PIL import Image

from handwriting_engine._constants import MAX_IMAGE_LONG_SIDE, JPEG_QUALITY
from handwriting_engine.providers import get_provider, available_providers
from handwriting_engine.providers.base import ConsensusResult

logger = logging.getLogger(__name__)


def _build_system_prompt(
    *,
    domain: str,
    content_type: str,
    writer_profile: dict | None,
    faint_protocol: str,
    inject_lessons: bool,
    writer_id: str | None,
    vocabulary_hints: list[str] | None,
    inject_strategies: bool,
) -> tuple[str, str]:
    """Assemble the full system prompt and a stable cache key.

    Returns (system_prompt_text, cache_key). The cache key is a sha256 hex
    digest of the inputs that determine the prompt content — usable as a
    dedup key for provider-side context caching.
    """
    from handwriting_engine.handwriting import get_reading_strategies

    parts: list[str] = []

    if inject_strategies:
        strategies = get_reading_strategies(
            domain=domain,
            content_type=content_type,
            writer_profile=writer_profile,
        )
        if strategies:
            parts.append(strategies)

    if faint_protocol:
        parts.append(faint_protocol)

    if inject_lessons:
        from handwriting_engine.lessons import build_lessons_prompt
        lessons = build_lessons_prompt(category=domain)
        if lessons:
            parts.append(lessons)

    if writer_id:
        from handwriting_engine.lessons import load_writer_calibration
        writer_cal = load_writer_calibration(writer_id)
        if writer_cal:
            parts.append(writer_cal)

    if vocabulary_hints:
        hint_text = "Expected vocabulary includes: " + ", ".join(vocabulary_hints[:50])
        parts.append(hint_text)

    system_prompt = "\n\n".join(parts)

    # Stable cache key over the inputs that determine the assembled prompt
    key_material = "".join([
        f"d={domain or ''}",
        f"c={content_type or ''}",
        f"wp={'1' if writer_profile else '0'}",
        f"fp={'1' if faint_protocol else '0'}",
        f"il={'1' if inject_lessons else '0'}",
        f"wid={writer_id or ''}",
        f"vh={'|'.join((vocabulary_hints or [])[:50])}",
        f"is={'1' if inject_strategies else '0'}",
        # Hashing the assembled text guards against drift in the upstream
        # builders (e.g. lessons store changes) — same inputs but different
        # content should yield different cache keys.
        f"body={system_prompt}",
    ])
    cache_key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()

    return system_prompt, cache_key


# Module-level dedup set: tracks which Gemini context-cache keys have already
# been registered with the gemini provider in this process.
_gemini_cache_keys: set[str] = set()


def _maybe_enable_gemini_cache(system_prompt: str, cache_key: str, providers_used) -> None:
    """Enable Gemini context cache once per unique cache_key per process.

    `providers_used` is either a single provider name (str) or an iterable of
    provider names (for consensus). Caching is only attempted when "gemini"
    is involved and the prompt is long enough to clear the minimum-token
    threshold (Flash: 1,024 tokens; we approximate with a char budget so we
    don't pay token-counting calls).
    """
    if not system_prompt:
        return
    if isinstance(providers_used, str):
        if providers_used != "gemini":
            return
    else:
        if "gemini" not in (providers_used or []):
            return

    if cache_key in _gemini_cache_keys:
        return

    # Rough char-based gate: ~4 chars/token. Flash minimum is 1,024 tokens,
    # so require >= 4,096 chars before attempting to cache. Below this the
    # provider would reject the cache create anyway and we'd just log.
    if len(system_prompt) < 4096:
        return

    try:
        gp = get_provider("gemini")
    except Exception as e:
        logger.debug("Gemini provider unavailable for context caching: %s", e)
        return

    enable = getattr(gp, "enable_context_cache", None)
    if not callable(enable):
        return

    try:
        enable(system_prompt)
        _gemini_cache_keys.add(cache_key)
    except Exception as e:
        logger.debug("Gemini context cache enable failed: %s", e)

# ---------------------------------------------------------------------------
# Post-processing — strip common LLM artifacts from transcriptions
# ---------------------------------------------------------------------------

_PREAMBLE_RE = re.compile(
    r"^(Here is|The transcribed|The handwritten|I can see|The text reads|"
    r"Transcription:|Below is|The following|The image shows|I'll transcribe)"
    r"[^\n]*\n",
    re.IGNORECASE,
)

_TRAILING_RE = re.compile(
    r"\n(Note:|Please note|Some words|I was unable|Disclaimer|The handwriting|"
    r"I hope|Let me know|If you need|Overall)[^\n]*$",
    re.IGNORECASE,
)


def _postprocess_output(text: str, domain: str | None = None) -> str:
    """Strip common LLM artifacts from transcription output.

    Removes preamble ("Here is the transcribed text:"), trailing commentary
    ("Note: some words were difficult to read"), markdown formatting artifacts,
    and deduplicates repeated sentences.
    """
    if not text or not text.strip():
        return ""

    result = text

    # 1. Strip preamble (first 1-3 lines only)
    for _ in range(3):
        m = _PREAMBLE_RE.match(result)
        if m:
            result = result[m.end():]
        else:
            break

    # 2. Strip trailing meta-commentary (last lines)
    for _ in range(3):
        m = _TRAILING_RE.search(result)
        if m:
            result = result[:m.start()]
        else:
            break

    # 3. Remove markdown artifacts from plain transcription
    result = re.sub(r'\*\*([^*]+)\*\*', r'\1', result)  # **bold** → bold
    result = re.sub(r'^#{1,3}\s+', '', result, flags=re.MULTILINE)  # # headers
    result = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'\1', result)  # _italic_

    # 4. Deduplicate sentences appearing 3+ times
    lines = result.split('\n')
    seen: dict[str, int] = {}
    deduped = []
    for line in lines:
        key = line.strip().lower()
        if not key:
            deduped.append(line)
            continue
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= 2:
            deduped.append(line)
    result = '\n'.join(deduped)

    # Domain spell correction (optional — only when domain is specified)
    if domain:
        try:
            from handwriting_engine.postprocess import correct_domain_terms
            result = correct_domain_terms(result, domain)
        except Exception as e:
            logger.warning("Domain correction failed: %s", e)

    return result.strip()


def _extract_vocabulary_hints(image_b64: str, media_type: str) -> list[str]:
    """Cheap pre-scan to extract clearly legible words for vocabulary priming.

    Uses Gemini Flash (cheapest provider) with minimal tokens to identify
    domain terms that are clearly visible. These are then injected as
    vocabulary_hints for the full read, helping the model resolve ambiguous
    characters in context.
    """
    from handwriting_engine._constants import VOCAB_PRIMING_MAX_TOKENS

    try:
        avail = available_providers()
        # Prefer gemini (cheapest), fall back to whatever is available
        provider_name = "gemini" if "gemini" in avail else avail[0] if avail else None
        if not provider_name:
            return []
        p = get_provider(provider_name)
        result = p.read_image(
            image_b64, media_type,
            "List all clearly legible words from this handwritten image. "
            "Output ONLY the words, comma-separated. Focus on technical or scientific terms.",
            "",  # No system prompt for speed
            VOCAB_PRIMING_MAX_TOKENS,
        )
        # Parse comma-separated words, deduplicate, limit to 50
        words = [w.strip() for w in result.split(",") if w.strip()]
        seen = set()
        unique = []
        for w in words:
            low = w.lower()
            if low not in seen and len(w) > 2:
                seen.add(low)
                unique.append(w)
        return unique[:50]
    except Exception as e:
        logger.warning("Vocabulary priming failed: %s", e)
        return []


def _dual_polarity_read(
    image_path: str,
    b64_data: str,
    media_type: str,
    provider_obj,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
) -> str:
    """Read both original and color-inverted versions for faint ink.

    Faint pencil strokes invisible on white paper pop against a dark background
    in the inverted version. The LLM's vision encoder activates different feature
    channels on inverted images. Returns the cleaner result or a merged version.
    """
    import io
    import base64
    from PIL import ImageOps

    # Normal read
    normal_text = provider_obj.read_image(b64_data, media_type, prompt, system_prompt, max_tokens)

    # Create inverted image
    try:
        img = Image.open(image_path)
        img.load()
        if img.mode == "RGBA":
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        inverted = ImageOps.invert(img)
        buf = io.BytesIO()
        inverted.save(buf, format="JPEG", quality=90)
        inv_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
        inverted.close()
        img.close()
    except Exception as e:
        logger.warning("Dual-polarity inversion failed: %s", e)
        return normal_text

    inv_prompt = (
        "This is a color-inverted version of handwritten text (dark background, light text). "
        "Read all text carefully, focusing on faint characters that may be more visible "
        "in this inverted view. " + prompt
    )
    try:
        inverted_text = provider_obj.read_image(inv_b64, "image/jpeg", inv_prompt, system_prompt, max_tokens)
    except Exception as e:
        logger.warning("Dual-polarity inverted read failed: %s", e)
        return normal_text

    # Pick the cleaner result (fewer uncertainty markers)
    marker_re = re.compile(r"\[\?\]|\?\?\?|\[illegible[^\]]*\]", re.IGNORECASE)
    normal_markers = len(marker_re.findall(normal_text))
    inverted_markers = len(marker_re.findall(inverted_text))

    if inverted_markers < normal_markers:
        logger.info("Dual-polarity: inverted version cleaner (%d vs %d markers)",
                     inverted_markers, normal_markers)
        return inverted_text
    return normal_text


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
    vocabulary_hints: list[str] | None = None,
    postprocess: bool = True,
    auto_retry: bool = False,
    auto_classify: bool = False,
    vocab_priming: bool = False,
    line_level: bool = False,
    auto_identify_writer: bool = False,
) -> str:
    """
    Read a single handwritten page.
    Injects reading strategies and disambiguation rules by default.

    Args:
        auto_enhance: If True, assess image quality and enhance if needed.
                      Uses 'proven' strategy for faint/poor images, 'adaptive' for fair.
        inject_lessons: If True, prepend learned lessons from past corrections.
        postprocess: If True, strip LLM preamble, trailing commentary, markdown artifacts.
        auto_retry: If True, retry with enhancement comparison when confidence is low.
        auto_classify: If True, classify handwriting style (cursive/print) and use
                       for content-type-specific reading strategies.
    """
    from handwriting_engine.optimize import validate_and_prepare_image

    # Classify handwriting style for content-type-aware routing
    detected_content_type = "default"
    if auto_classify:
        from handwriting_engine.quality import classify_handwriting_style
        style = classify_handwriting_style(image_path)
        detected_content_type = {"cursive": "cursive", "print": "printed", "mixed": "handwriting"}.get(style, "default")

    assessment = None
    if auto_enhance:
        import tempfile
        from handwriting_engine.quality import assess_image
        from handwriting_engine.enhance import enhance_image
        assessment = assess_image(image_path)
        if assessment["quality"] != "good":
            # Use proven for faint/poor (15-20% CER improvement), adaptive for fair
            if assessment.get("faint_ink") or assessment["quality"] == "poor":
                strategy = "proven"
            else:
                strategy = "adaptive"
            suffix = os.path.splitext(image_path)[1] or ".jpg"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.close()
            image_path = enhance_image(image_path, strategy=strategy, output_path=tmp.name)

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
            "If a region is completely illegible, write [illegible: ~N chars]. "
            "For names: read letter by letter, flag the whole name with [?] if any letter is uncertain. "
            "Do NOT guess — if unsure, flag it rather than producing a confident wrong reading."
        )

    # Auto-identify writer from image when no writer_id was provided
    if writer_id is None and auto_identify_writer:
        from handwriting_engine.writer_embeddings import identify_writer
        detected_id, score = identify_writer(image_path)
        if detected_id:
            writer_id = detected_id
            logger.debug("Auto-identified writer %s (similarity %.3f)", writer_id, score)

    # Reading strategies go in system prompt (enables prompt caching, better model attention)
    # Load writer profile first so it can replace the generic calibration block
    writer_profile_dict = None
    if writer_id:
        from handwriting_engine.writer_profile_store import WriterProfileStore
        writer_profile_dict = WriterProfileStore().load(writer_id)

    # Auto-extract vocabulary hints via cheap pre-scan when vocab_priming is enabled.
    # Done before assembling the system prompt so hints are folded in as part of
    # the cached/hashed payload.
    if vocab_priming and not vocabulary_hints:
        vocabulary_hints = _extract_vocabulary_hints(b64_data, media_type)

    # Assemble system prompt via shared helper so call sites stay aligned and
    # we get a stable cache key for provider-side context caching.
    assembled, cache_key = _build_system_prompt(
        domain=domain,
        content_type=detected_content_type,
        writer_profile=writer_profile_dict,
        faint_protocol=faint_protocol,
        inject_lessons=inject_lessons,
        writer_id=writer_id,
        vocabulary_hints=vocabulary_hints,
        inject_strategies=inject_strategies,
    )
    if assembled and system_prompt:
        system_prompt = assembled + "\n\n" + system_prompt
    else:
        system_prompt = assembled or system_prompt

    # Wire Gemini context cache (idempotent per cache_key, gated by length)
    _maybe_enable_gemini_cache(system_prompt, cache_key, provider)

    # Adapt prompts for provider-specific optimization
    from handwriting_engine.prompt_adapter import adapt_system_prompt, adapt_user_prompt
    system_prompt = adapt_system_prompt(system_prompt, provider)
    prompt = adapt_user_prompt(prompt, provider)

    if line_level:
        from handwriting_engine.line_reader import read_page_by_lines
        avail = available_providers()
        if avail:
            _prov = get_provider(avail[0])
            line_text = read_page_by_lines(image_path, _prov, prompt, system_prompt or "", max_tokens)
            if line_text.strip():
                return _postprocess_output(line_text, domain=domain) if postprocess else line_text
        # Fall through to whole-page read if line segmentation fails

    # Stage 4: when a writer-specific TrOCR checkpoint exists for this writer,
    # load THAT instead of the base singleton. Skipped silently for non-trocr
    # providers (cloud VLMs don't have per-writer checkpoints) and for writers
    # that haven't been fine-tuned yet (the base model still works).
    p = None
    if provider == "trocr" and writer_id:
        from handwriting_engine.providers.trocr_provider import writer_checkpoint_dir
        if writer_checkpoint_dir(writer_id).exists():
            p = get_provider("trocr", writer_id=writer_id)
    if p is None:
        p = get_provider(provider)

    # S2: per-writer few-shot exemplars. Returns None (fall through) when the
    # writer has <2 GT samples, the provider is OCR-only, or HE_FEW_SHOT_K=0.
    exemplar_blocks = None
    if writer_id:
        from handwriting_engine.few_shot import select_and_build_exemplar_blocks

        exemplar_blocks = select_and_build_exemplar_blocks(
            writer_id=writer_id,
            provider=provider,
            target_image_b64=b64_data,
            target_media_type=media_type,
        )

    # Dual-polarity reading for faint ink: send both normal and inverted images
    from handwriting_engine._constants import DUAL_POLARITY_ENABLED
    if DUAL_POLARITY_ENABLED and assessment and assessment.get("faint_ink"):
        # Faint-ink dual-polarity wins over few-shot — they target different
        # failure modes and combining them inflates per-call token cost
        # without a verified gain.
        raw = _dual_polarity_read(image_path, b64_data, media_type, p, prompt, system_prompt, max_tokens)
    elif exemplar_blocks is not None:
        raw = p.read_batch(exemplar_blocks, prompt=prompt, system_prompt=system_prompt, max_tokens=max_tokens)
    else:
        raw = p.read_image(b64_data, media_type, prompt, system_prompt, max_tokens)

    if postprocess:
        raw = _postprocess_output(raw, domain=domain)

    # Auto-retry with enhancement comparison when confidence is low,
    # or zoomed crop verification when confidence is moderate but has [?] markers
    if auto_retry:
        from handwriting_engine.consensus import _single_text_confidence
        from handwriting_engine._constants import (
            ZOOM_VERIFY_CONFIDENCE_LOW, ZOOM_VERIFY_CONFIDENCE_HIGH,
            ZOOM_VERIFY_MIN_MARKERS,
        )
        confidence = _single_text_confidence(raw)
        marker_count = len(re.findall(r"\[\?\]|\?\?\?|\[illegible[^\]]*\]", raw, re.IGNORECASE))

        if confidence < ZOOM_VERIFY_CONFIDENCE_LOW:
            # Very low confidence — full retry with enhancement comparison
            logger.info("Low confidence (%.2f), retrying with enhancement comparison", confidence)
            retry = read_with_enhancement_comparison(
                image_path, domain=domain, provider=provider, enhance_strategy="proven",
            )
            if _single_text_confidence(retry) > confidence:
                raw = _postprocess_output(retry, domain=domain) if postprocess else retry
        elif confidence < ZOOM_VERIFY_CONFIDENCE_HIGH and marker_count >= ZOOM_VERIFY_MIN_MARKERS:
            # Moderate confidence with uncertainty markers — zoomed crop verification
            logger.info("Moderate confidence (%.2f) with %d markers, trying zoomed verification",
                        confidence, marker_count)
            verified = _zoomed_verify(image_path, raw, provider, system_prompt, max_tokens)
            if postprocess:
                verified = _postprocess_output(verified, domain=domain)
            verified_confidence = _single_text_confidence(verified)
            if verified_confidence > confidence:
                raw = verified

    return raw


def _zoomed_verify(
    image_path: str,
    text: str,
    provider: str,
    system_prompt: str,
    max_tokens: int = 4096,
) -> str:
    """Re-read uncertain regions at higher resolution to resolve [?] markers.

    Splits the page into horizontal strips, identifies which strips contain
    uncertain text (by mapping [?] positions to line thirds), and re-sends
    those strips at 2x resolution with a focused verification prompt.
    """
    from handwriting_engine._constants import ZOOM_VERIFY_STRIPS
    from handwriting_engine.optimize import validate_and_prepare_image

    # Find [?] marker positions within the text (by line number)
    lines = text.split("\n")
    total_lines = len(lines)
    if total_lines == 0:
        return text

    uncertain_lines = set()
    for i, line in enumerate(lines):
        if "[?]" in line or "???" in line or re.search(r"\[illegible[^\]]*\]", line, re.IGNORECASE):
            uncertain_lines.add(i)

    if not uncertain_lines:
        return text

    # Map uncertain lines to image strips
    strip_count = min(ZOOM_VERIFY_STRIPS, total_lines)
    if strip_count < 1:
        return text
    lines_per_strip = total_lines / strip_count
    uncertain_strips = set()
    for line_idx in uncertain_lines:
        strip_idx = min(int(line_idx / lines_per_strip), strip_count - 1)
        uncertain_strips.add(strip_idx)

    # Crop uncertain strips from the original image and re-read
    try:
        img = Image.open(image_path)
        img.load()
    except (IOError, OSError):
        return text

    w, h = img.size
    strip_height = h // strip_count
    p = get_provider(provider)
    replacements = {}

    for strip_idx in sorted(uncertain_strips):
        y_start = strip_idx * strip_height
        y_end = h if strip_idx == strip_count - 1 else (strip_idx + 1) * strip_height
        # Add 10% overlap to avoid cutting through text lines
        overlap = int(strip_height * 0.1)
        y_start = max(0, y_start - overlap)
        y_end = min(h, y_end + overlap)

        strip_img = img.crop((0, y_start, w, y_end))

        # Upscale 2x for better resolution on the cropped region
        strip_img = strip_img.resize((w * 2, (y_end - y_start) * 2), Image.LANCZOS)

        # Convert to base64
        import io, base64
        if strip_img.mode not in ("RGB", "L"):
            strip_img = strip_img.convert("RGB")
        buf = io.BytesIO()
        strip_img.save(buf, format="JPEG", quality=95)
        b64_data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
        strip_img.close()

        # Context: which lines from the original reading correspond to this strip
        line_start = int(strip_idx * lines_per_strip)
        line_end = min(total_lines, int((strip_idx + 1) * lines_per_strip))
        context_text = "\n".join(lines[line_start:line_end])

        verify_prompt = (
            f"This is a zoomed-in section of a handwritten page. "
            f"A previous reading of this section produced:\n\n"
            f"\"{context_text}\"\n\n"
            f"Characters marked with [?] or [illegible] need verification. "
            f"Please read this section carefully and provide the corrected text. "
            f"Only output the text for this section, preserving original spelling."
        )

        try:
            strip_read = p.read_image(b64_data, "image/jpeg", verify_prompt, system_prompt, max_tokens)
            if strip_read.strip():
                replacements[strip_idx] = (line_start, line_end, strip_read.strip())
        except Exception as e:
            logger.warning(f"Zoomed verify failed for strip {strip_idx}: {e}")

    img.close()

    if not replacements:
        return text

    # Rebuild text: replace uncertain strip lines with zoomed readings
    result_lines = list(lines)
    for strip_idx in sorted(replacements.keys(), reverse=True):
        line_start, line_end, new_text = replacements[strip_idx]
        new_lines = new_text.split("\n")
        result_lines[line_start:line_end] = new_lines

    return "\n".join(result_lines)


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
    vocabulary_hints: list[str] | None = None,
    auto_classify: bool = False,
    auto_identify_writer: bool = False,
) -> ConsensusResult:
    """
    Read a page using multi-model consensus.
    """
    from handwriting_engine.optimize import validate_and_prepare_image
    from handwriting_engine.consensus import read_with_consensus as _consensus

    # Auto-classify handwriting style for content-type-aware provider routing
    if auto_classify and content_type == "default":
        from handwriting_engine.quality import classify_handwriting_style
        style = classify_handwriting_style(image_path)
        content_type = {"cursive": "cursive", "print": "printed", "mixed": "handwriting"}.get(style, "default")

    result = validate_and_prepare_image(image_path)
    if result is None:
        return ConsensusResult(text="", confidence=0.0)

    b64_data, media_type = result

    if not prompt:
        prompt = (
            "Read all handwritten text in this image carefully. "
            "Preserve original spelling and formatting exactly as written. "
            "For any character you are less than 80% sure about, mark it with [?]. "
            "If a region is completely illegible, write [illegible: ~N chars]. "
            "For names: read letter by letter, flag the whole name with [?] if any letter is uncertain. "
            "Do NOT guess — if unsure, flag it rather than producing a confident wrong reading."
        )

    # Auto-identify writer from image when no writer_id was provided
    if writer_id is None and auto_identify_writer:
        from handwriting_engine.writer_embeddings import identify_writer
        detected_id, score = identify_writer(image_path)
        if detected_id:
            writer_id = detected_id
            logger.debug("Auto-identified writer %s (similarity %.3f)", writer_id, score)

    # Load writer profile first so it can replace the generic calibration block
    writer_profile_dict = None
    if writer_id:
        from handwriting_engine.writer_profile_store import WriterProfileStore
        writer_profile_dict = WriterProfileStore().load(writer_id)

    # Assemble system prompt via shared helper for parity with read_page().
    # Faint protocol is not assessed here (consensus path doesn't peek at
    # quality before assembly) — pass empty so the cache key reflects that.
    system_prompt, cache_key = _build_system_prompt(
        domain=domain,
        content_type=content_type,
        writer_profile=writer_profile_dict,
        faint_protocol="",
        inject_lessons=inject_lessons,
        writer_id=writer_id,
        vocabulary_hints=vocabulary_hints,
        inject_strategies=inject_strategies,
    )

    # Wire Gemini context cache when gemini is among the providers in this run
    _maybe_enable_gemini_cache(system_prompt, cache_key, providers or available_providers())

    # For smart routing, assess image quality and pass to consensus
    quality_assessment = None
    if strategy == "smart":
        from handwriting_engine.quality import assess_image
        quality_assessment = assess_image(image_path)

    return _consensus(
        image_b64=b64_data,
        media_type=media_type,
        prompt=prompt,
        system_prompt=system_prompt,
        providers=providers,
        strategy=strategy,
        content_type=content_type,
        quality_assessment=quality_assessment,
        writer_profile=writer_profile_dict,
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
