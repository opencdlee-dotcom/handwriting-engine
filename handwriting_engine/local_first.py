"""
Local-first OCR strategy: try TrOCR (free, local) and only escalate to a cloud
VLM when generation confidence falls below a threshold.

The TrOCR confidence is exp(mean_token_logprob), so it sits in (0, 1] with 1.0
meaning "certain" and ~0 meaning "lost". Default threshold 0.6 puts the
escalation knee where TrOCR's CER tends to spike on the IAM test set; tune via
``HE_LOCAL_FIRST_THRESHOLD``.

Returns the same shape as a normal provider read so callers (sweep, vision
pipeline) don't have to special-case it. When TrOCR wins, input/output token
counts are 0 — the cloud is never touched. When the fallback fires, it
propagates the cloud provider's token usage so cost projections stay honest.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from handwriting_engine.optimize import validate_and_prepare_image
from handwriting_engine.providers import get_provider
from handwriting_engine.vision import read_page

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.6
THRESHOLD_ENV_VAR = "HE_LOCAL_FIRST_THRESHOLD"

# Per-token floor: even if mean confidence is high, a single token with
# confidence below this triggers escalation. Catches buried bad chars (one
# wrong letter in a long, otherwise confident line) that the mean averages
# away. 0.3 chosen so common ambiguous-but-correct tokens (e.g. mid-word
# punctuation) don't constantly escalate.
DEFAULT_MIN_TOKEN_THRESHOLD = 0.3
MIN_TOKEN_THRESHOLD_ENV_VAR = "HE_LOCAL_FIRST_MIN_TOKEN_THRESHOLD"


def _resolve_threshold(explicit: float | None) -> float:
    """Threshold precedence: explicit arg > env var > DEFAULT_THRESHOLD."""
    if explicit is not None:
        return float(explicit)
    raw = os.getenv(THRESHOLD_ENV_VAR)
    if raw is None or raw == "":
        return DEFAULT_THRESHOLD
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r, falling back to default %.2f",
            THRESHOLD_ENV_VAR, raw, DEFAULT_THRESHOLD,
        )
        return DEFAULT_THRESHOLD


def _resolve_min_token_threshold(explicit: float | None) -> float:
    """Min-token threshold precedence: explicit > env > DEFAULT_MIN_TOKEN_THRESHOLD."""
    if explicit is not None:
        return float(explicit)
    raw = os.getenv(MIN_TOKEN_THRESHOLD_ENV_VAR)
    if raw is None or raw == "":
        return DEFAULT_MIN_TOKEN_THRESHOLD
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r, falling back to default %.2f",
            MIN_TOKEN_THRESHOLD_ENV_VAR, raw, DEFAULT_MIN_TOKEN_THRESHOLD,
        )
        return DEFAULT_MIN_TOKEN_THRESHOLD


@dataclass
class LocalFirstResult:
    """Mirror of the dict shape produced by ``_read_single`` / ``_read_consensus``.

    Kept dataclass-light so call sites can ``asdict()`` if they need a plain
    dict, or read attributes directly.

    ``local_min_token_confidence`` is exp(min(token_logprobs)) — surfaces the
    worst single token even when the mean is high. 0.0 means we never got a
    confidence reading (no scores from the model, or TrOCR errored).
    """
    text: str
    confidence: float
    input_tokens: int
    output_tokens: int
    escalated: bool
    local_confidence: float
    fallback_provider: str | None
    local_min_token_confidence: float = 0.0


def read_local_first(
    image_path: str,
    fallback_provider: str,
    *,
    threshold: float | None = None,
    min_token_threshold: float | None = None,
    domain: str = "biology",
    inject_lessons: bool = False,
    line_level: bool = False,
    auto_retry: bool = False,
) -> LocalFirstResult:
    """Run TrOCR; escalate to cloud if mean OR min-token confidence is too low.

    Two-gate escalation:
      - ``local_conf < threshold``           — overall sequence looks shaky, OR
      - ``local_min_token_conf < min_token_threshold`` — even one token looks bad

    The min-token gate catches single buried bad chars in otherwise confident
    lines (e.g. one wrong letter mid-word) that the sequence-mean averages away.

    Args:
        image_path: Path to the page image.
        fallback_provider: Cloud provider name (e.g. "gemini", "claude") to
            invoke when TrOCR's confidence is too low.
        threshold: Override the default 0.6 mean-confidence threshold. None
            falls back to ``HE_LOCAL_FIRST_THRESHOLD`` env, then DEFAULT.
        min_token_threshold: Override the default 0.3 min-token threshold.
            None falls back to ``HE_LOCAL_FIRST_MIN_TOKEN_THRESHOLD`` env.
        domain, inject_lessons, line_level, auto_retry: forwarded to
            ``read_page`` when escalation fires.

    Returns:
        ``LocalFirstResult`` with text, confidence, token counts, and an
        ``escalated`` flag. When TrOCR wins, both token counts are zero.
    """
    effective_threshold = _resolve_threshold(threshold)
    effective_min_token_threshold = _resolve_min_token_threshold(min_token_threshold)

    prepared = validate_and_prepare_image(image_path)
    if prepared is None:
        return LocalFirstResult(
            text="",
            confidence=0.0,
            input_tokens=0,
            output_tokens=0,
            escalated=False,
            local_confidence=0.0,
            fallback_provider=None,
            local_min_token_confidence=0.0,
        )
    b64_data, media_type = prepared

    trocr = get_provider("trocr")
    local_min_conf = 0.0
    try:
        # Always call the legacy 2-tuple API so existing VisionProvider mocks
        # and alternative implementations keep working.
        local_text, local_conf = trocr.read_image_with_confidence(
            b64_data, media_type=media_type
        )
    except Exception as exc:
        logger.warning("local_first: TrOCR read failed (%s); escalating", exc)
        local_text, local_conf = "", 0.0

    # If the provider exposes the richer token-confidence API (TrOCRProvider
    # does after the min-token-confidence change), use it to recover min_conf.
    # We tolerate any error here — min_conf defaults to local_conf so the
    # min-token gate can't accidentally escalate-by-default on legacy providers.
    token_conf_fn = getattr(trocr, "read_image_with_token_confidence", None)
    if callable(token_conf_fn):
        try:
            result = token_conf_fn(b64_data, media_type=media_type)
            # Expect (text, mean_conf, min_conf). Anything else: skip silently.
            if isinstance(result, tuple) and len(result) == 3:
                t_text, t_mean, t_min = result
                # Trust the richer reading entirely: it was generated in the
                # same pass as min_conf, so they're guaranteed to align.
                if isinstance(t_text, str):
                    local_text = t_text
                if isinstance(t_mean, (int, float)):
                    local_conf = float(t_mean)
                if isinstance(t_min, (int, float)):
                    local_min_conf = float(t_min)
        except Exception as exc:
            logger.debug(
                "local_first: token-confidence read failed (%s); using mean only",
                exc,
            )

    # If we never got a min_conf reading, fall back to the mean so the
    # min-token gate is a no-op rather than spuriously escalating.
    if local_min_conf == 0.0 and local_conf > 0.0:
        local_min_conf = local_conf

    mean_ok = local_conf >= effective_threshold
    min_ok = local_min_conf >= effective_min_token_threshold
    if mean_ok and min_ok and local_text.strip():
        # Local wins. Zero cloud tokens — that's the whole point.
        return LocalFirstResult(
            text=local_text,
            confidence=local_conf,
            input_tokens=0,
            output_tokens=0,
            escalated=False,
            local_confidence=local_conf,
            fallback_provider=None,
            local_min_token_confidence=local_min_conf,
        )

    # Escalate. Snapshot fallback provider's pre-call usage to compute delta —
    # providers are singletons so cumulative usage isn't safe.
    pre_input = pre_output = 0
    try:
        p = get_provider(fallback_provider)
        pre_usage = p.usage
        pre_input = pre_usage.get("input_tokens", 0)
        pre_output = pre_usage.get("output_tokens", 0)
    except Exception as exc:
        logger.warning(
            "local_first: failed to load fallback provider %r (%s); "
            "returning local result anyway",
            fallback_provider, exc,
        )
        return LocalFirstResult(
            text=local_text,
            confidence=local_conf,
            input_tokens=0,
            output_tokens=0,
            escalated=False,
            local_confidence=local_conf,
            fallback_provider=None,
            local_min_token_confidence=local_min_conf,
        )

    try:
        cloud_text = read_page(
            image_path,
            domain=domain,
            provider=fallback_provider,
            inject_lessons=inject_lessons,
            line_level=line_level,
            auto_retry=auto_retry,
        )
    except Exception as exc:
        logger.warning(
            "local_first: fallback %s failed (%s); returning local result",
            fallback_provider, exc,
        )
        return LocalFirstResult(
            text=local_text,
            confidence=local_conf,
            input_tokens=0,
            output_tokens=0,
            escalated=False,
            local_confidence=local_conf,
            fallback_provider=None,
            local_min_token_confidence=local_min_conf,
        )

    # Compute delta token usage so multiple sweep samples don't double-count.
    input_tokens = output_tokens = 0
    try:
        post_usage = get_provider(fallback_provider).usage
        input_tokens = max(0, post_usage.get("input_tokens", 0) - pre_input)
        output_tokens = max(0, post_usage.get("output_tokens", 0) - pre_output)
    except Exception:
        pass

    # Cloud confidence isn't measured the same way; mark it neutral-high so
    # downstream consumers don't accidentally re-escalate. The local_confidence
    # field still records why we escalated.
    return LocalFirstResult(
        text=cloud_text,
        confidence=0.7,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        escalated=True,
        local_confidence=local_conf,
        fallback_provider=fallback_provider,
        local_min_token_confidence=local_min_conf,
    )
