"""
S2 — per-writer few-shot exemplar plumbing.

Builds the interleaved content blocks (exemplar_image, exemplar_label, ...,
target_image) that vision providers like Claude and Gemini already accept via
their ``read_batch`` methods. Selection lives in
:mod:`handwriting_engine.writer_profile_store`; this module only handles the
"now turn an Exemplar list into provider-ready blocks" step plus the env-var
opt-out per S2-SPEC § Cost & opt-out.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

from handwriting_engine.writer_profile_store import Exemplar

logger = logging.getLogger(__name__)


# Providers whose `read_batch` carries multi-image content lists. TrOCR (and
# any OCR-only provider that lacks in-context learning) is intentionally
# excluded — see S2-SPEC criterion #5.
EXEMPLAR_PROVIDERS: frozenset[str] = frozenset({"claude", "gemini"})

DEFAULT_FEW_SHOT_K = 3
FEW_SHOT_K_ENV = "HE_FEW_SHOT_K"


# Anti-cargo-cult guidance: returning-writer few-shot can prime the model to
# parrot the reference transcription rather than read the new image. The
# label deliberately separates "reference text" from "what to read".
EXEMPLAR_LABEL_TEMPLATE = (
    "The handwriting in the previous image transcribes to: «{gt}». "
    "This is the same writer as the final image but contains DIFFERENT TEXT. "
    "Read what is in the final image — do not repeat the reference text."
)


def env_few_shot_k(env: Optional[dict] = None) -> int:
    """Return the ``HE_FEW_SHOT_K`` cap, or ``DEFAULT_FEW_SHOT_K`` if unset.

    ``0`` disables few-shot entirely (S2-SPEC § Cost & opt-out). Negative or
    unparseable values are treated as the default — the caller is presumed
    to want the spec-default behavior, not silent disablement.
    """
    env = os.environ if env is None else env
    raw = env.get(FEW_SHOT_K_ENV)
    if raw is None:
        return DEFAULT_FEW_SHOT_K
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_FEW_SHOT_K
    return value if value >= 0 else DEFAULT_FEW_SHOT_K


def provider_supports_exemplars(provider: str) -> bool:
    """True iff the provider name belongs to EXEMPLAR_PROVIDERS."""
    return provider in EXEMPLAR_PROVIDERS


def _media_type_for(image_path: str) -> str:
    suffix = os.path.splitext(image_path)[1].lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")


def _load_exemplar_image(image_path: str) -> Optional[tuple[str, str]]:
    """Encode an exemplar image identically to ``optimize.validate_and_prepare_image``.

    Returns ``(base64, media_type)`` or ``None`` if the image is missing or
    fails to decode. Missing exemplars are filtered out rather than raised —
    a stale DB row should not break the whole transcription path.
    """
    try:
        from handwriting_engine.optimize import validate_and_prepare_image

        result = validate_and_prepare_image(image_path)
        if result is not None:
            return result
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("optimize.validate_and_prepare_image failed for %s: %s", image_path, exc)

    # Fallback: raw read for tests / minimal envs without PIL.
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as fh:
            data = base64.standard_b64encode(fh.read()).decode("utf-8")
    except OSError:
        return None
    return data, _media_type_for(image_path)


def build_exemplar_blocks(
    target_image_b64: str,
    target_media_type: str,
    exemplars: list[Exemplar],
) -> list[dict]:
    """Build ``read_batch`` content blocks: exemplars first, target last.

    Layout (per S2-SPEC § 2):
        exemplar_1_image, exemplar_1_label_text,
        exemplar_2_image, exemplar_2_label_text,
        ...,
        target_image

    The user's prompt is appended by the provider's ``read_batch``. Exemplars
    whose image cannot be loaded are skipped silently (logged at DEBUG); a
    fully-failed exemplar list collapses to a single-image read, which is the
    correct fallback.
    """
    blocks: list[dict] = []
    for ex in exemplars:
        loaded = _load_exemplar_image(ex.image_path)
        if loaded is None:
            logger.debug(
                "Skipping exemplar sample_id=%s: image not loadable (%s)",
                ex.sample_id,
                ex.image_path,
            )
            continue
        b64, media_type = loaded
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            }
        )
        blocks.append(
            {
                "type": "text",
                "text": EXEMPLAR_LABEL_TEMPLATE.format(gt=ex.ground_truth),
            }
        )

    blocks.append(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": target_media_type,
                "data": target_image_b64,
            },
        }
    )
    return blocks


def select_and_build_exemplar_blocks(
    *,
    writer_id: Optional[str],
    provider: str,
    target_image_b64: str,
    target_media_type: str,
    k: Optional[int] = None,
    exclude_sample_id: Optional[int] = None,
    env: Optional[dict] = None,
    conn=None,
    db_path=None,
) -> Optional[list[dict]]:
    """High-level orchestrator: gate, select, and build blocks in one call.

    Returns ``None`` whenever the few-shot path is not applicable (caller
    should fall back to a normal single-image read). This consolidates the
    several spec-defined gates (writer_id required, provider allowlist,
    HE_FEW_SHOT_K, ≥2 GT samples) so the integration in ``vision.read_page``
    is one branch.
    """
    if not writer_id:
        return None
    if not provider_supports_exemplars(provider):
        logger.debug("few-shot: provider %s not in EXEMPLAR_PROVIDERS", provider)
        return None

    cap = env_few_shot_k(env) if k is None else k
    if cap <= 0:
        logger.debug("few-shot: HE_FEW_SHOT_K=%d disables exemplars", cap)
        return None

    from handwriting_engine.writer_profile_store import select_exemplars

    exemplars = select_exemplars(
        writer_id,
        k=cap,
        exclude_sample_id=exclude_sample_id,
        conn=conn,
        db_path=db_path,
    )
    # S2-SPEC criterion #4: cold writers (<2 GT samples) fall back cleanly.
    if len(exemplars) < 2:
        logger.debug(
            "few-shot: only %d exemplar(s) for writer_id=%s — falling back to single-image read",
            len(exemplars),
            writer_id,
        )
        return None

    return build_exemplar_blocks(target_image_b64, target_media_type, exemplars)
