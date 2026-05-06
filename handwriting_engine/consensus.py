"""
Multi-model consensus engine for handwriting recognition.
Strategies: vote, best_of, debate, cascade.

Confidence is derived from inter-provider agreement, not text heuristics.
Word-level voting prevents Frankenstein merges. Disagreements are surfaced
explicitly rather than silently blended away.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from handwriting_engine.providers import get_provider, available_providers
from handwriting_engine.providers.base import ConsensusResult, circuit_breaker

logger = logging.getLogger(__name__)

# Content-type to best-provider routing (2026 benchmarks: Gemini 1.67% CER, Claude 4.28%, GPT 16.81%)
BEST_PROVIDER_ROUTING = {
    "handwriting": "gemini",      # Gemini now BEST for handwriting (1.67% CER)
    "cursive": "gemini",          # 100% on cursive benchmark
    "structured": "gemini",       # Gemini leads OCR/structured docs
    "printed": "gemini",
    "multilang": "gemini",
    "layout": "claude",           # Claude still best for complex layouts
    "scientific": "claude",       # Claude best for scientific notation
    "formatting": "claude",       # Claude highest fidelity preservation
    "default": "gemini",          # Gemini is new default (best CER + cheapest)
}

# Content-type-aware provider weights for vote strategy (2026 benchmarks)
PROVIDER_WEIGHTS = {
    "handwriting": {"gemini": 1.5, "claude": 1.0, "openai": 0.7},
    "cursive":     {"gemini": 1.5, "claude": 1.0, "openai": 0.7},
    "structured":  {"gemini": 1.5, "claude": 1.0, "openai": 0.8},
    "printed":     {"gemini": 1.5, "claude": 1.0, "openai": 0.8},
    "layout":      {"claude": 1.5, "gemini": 1.0, "openai": 0.8},
    "scientific":  {"claude": 1.5, "gemini": 1.0, "openai": 0.8},
    "default":     {"gemini": 1.2, "claude": 1.0, "openai": 0.8},
}

# Cascade order: cheapest first (~$0.0005/img → ~$0.003 → ~$0.005)
CASCADE_ORDER = ["gemini", "openai", "claude"]

# ---------------------------------------------------------------------------
# Char-level confusion-pair resolution (S5)
# ---------------------------------------------------------------------------
#
# Each tuple is (alt_form, canonical_form). When two providers disagree on a
# word and the only char-level differences match one of these pairs, we
# resolve to the canonical form by default. A writer profile entry of the
# form `confusion_resolutions["<a>↔<b>"] = "<a>" or "<b>"` overrides per writer.
#
# Pairs without a clear canonical winner (b↔d, p↔q, i↔j) are intentionally
# omitted — without context, defaulting either way introduces error. They
# defer to the existing [?alt: …] fallback.
_CHAR_CONFUSION_PAIRS: list[tuple[str, str]] = [
    # Multi-char ↔ single-char (the dominant residual error class)
    ("rn", "m"),
    ("cl", "d"),
    ("ri", "n"),
    ("uu", "w"),
    # Single-char letter pairs with a canonical form
    ("u", "v"),
    ("a", "o"),
    ("n", "h"),
    ("e", "c"),
    ("f", "t"),
    ("q", "g"),
    # Case-sensitive letter/digit pairs (canonical = letter form for in-word context)
    ("0", "O"),
    ("1", "l"),
    ("1", "I"),
    ("I", "l"),
    ("5", "S"),
    ("2", "Z"),
    ("6", "G"),
    ("8", "B"),
]


def _confusion_pair_label(a: str, b: str) -> str:
    """Canonical pair-name string used as writer_resolutions dict key."""
    return f"{a}↔{b}"


def _match_confusion_pair(left: str, right: str) -> tuple[str, str] | None:
    """Return the canonical (alt, canonical) pair tuple if (left, right) match
    one of the known confusion pairs in either order; else None.

    Whole-segment match (e.g. ``("rn", "m")``) is checked first so multi-char
    pairs win over the per-char fallback below.
    """
    for alt, canon in _CHAR_CONFUSION_PAIRS:
        if (left, right) == (alt, canon) or (left, right) == (canon, alt):
            return (alt, canon)
    return None


def _decompose_diff_into_pairs(
    seg_a: str, seg_b: str
) -> list[tuple[str, str]] | None:
    """Try to express the (seg_a, seg_b) diff as a sequence of per-position
    confusion-pair swaps.

    Returns a list of (left_char, right_char) sub-pairs, or None if the diff
    isn't decomposable into known confusion pairs.

    Whole-segment match wins (so ``("ll", "II")`` is decomposed into two
    `("l", "I")` pairs only because no entry like ``("ll", "II")`` exists).
    """
    direct = _match_confusion_pair(seg_a, seg_b)
    if direct is not None:
        return [(seg_a, seg_b)]

    # Per-character fallback only works for equal-length segments.
    if len(seg_a) != len(seg_b):
        return None
    parts: list[tuple[str, str]] = []
    for ca, cb in zip(seg_a, seg_b):
        if ca == cb:
            parts.append((ca, cb))
            continue
        if _match_confusion_pair(ca, cb) is None:
            return None
        parts.append((ca, cb))
    return parts


def resolve_char_level(
    candidates: list[str],
    weights: list[float] | None = None,
    *,
    writer_resolutions: dict[str, str] | None = None,
) -> str | None:
    """Resolve a no-majority word-level disagreement at character level.

    When two candidates differ only by known confusion-pair substrings
    (`rn↔m`, `cl↔d`, `0↔O`, …), pick the resolution. The writer profile's
    `confusion_resolutions` map (if provided) overrides the global canonical
    default per pair.

    Returns the resolved word, or `None` when the disagreement is not a
    pure confusion-pair case — caller must fall back to its existing
    no-majority handling (e.g. `[?alt: …]`).

    v0 scope: handles 1-2 unique candidates. With 3+ distinct candidates,
    returns None (defer to existing fallback).
    """
    if not candidates:
        return None

    unique = list(dict.fromkeys(candidates))  # preserve first-seen order
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 2:
        return None

    a, b = unique[0], unique[1]
    matcher = SequenceMatcher(None, a, b)
    opcodes = matcher.get_opcodes()

    # All differing segments must decompose into known confusion-pair swaps.
    decomposed: list[tuple[str, list[tuple[str, str]]]] = []
    has_diff = False
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            decomposed.append(("equal", [(a[i1:i2], a[i1:i2])]))
            continue
        has_diff = True
        seg_a = a[i1:i2]
        seg_b = b[j1:j2]
        parts = _decompose_diff_into_pairs(seg_a, seg_b)
        if parts is None:
            return None
        decomposed.append(("diff", parts))

    if not has_diff:
        return a

    writer_resolutions = writer_resolutions or {}

    out_chars: list[str] = []
    for tag, parts in decomposed:
        if tag == "equal":
            out_chars.append(parts[0][0])
            continue
        for left, right in parts:
            if left == right:
                out_chars.append(left)
                continue
            pair = _match_confusion_pair(left, right)
            if pair is None:
                return None
            alt, canon = pair
            label = _confusion_pair_label(alt, canon)
            choice = writer_resolutions.get(label)
            if choice in (alt, canon):
                out_chars.append(choice)
            else:
                out_chars.append(canon)

    return "".join(out_chars)

# Uncertainty marker pattern — must be defined here (before _self_correct and confidence helpers)
_UNCERTAINTY_RE = re.compile(
    r"\[\?\]|\?\?\?|\[illegible[^\]]*\]|\[unclear\]|unable to read",
    re.IGNORECASE,
)


def _count_uncertainty_markers(text: str) -> int:
    """Count [?] and similar uncertainty markers in transcription output."""
    return len(_UNCERTAINTY_RE.findall(text))


# Single-provider confidence cap — one opinion can never be "certain"
_SINGLE_PROVIDER_MAX_CONFIDENCE = 0.70


def read_with_consensus(
    image_b64: str,
    media_type: str,
    prompt: str,
    system_prompt: str = "",
    providers: list[str] | None = None,
    strategy: str = "vote",
    confidence_threshold: float = 0.8,
    content_type: str = "default",
    max_tokens: int = 4096,
    max_debate_rounds: int = 2,
    quality_assessment: dict | None = None,
    max_self_correct_rounds: int = 1,
    uncertainty_threshold: int = 3,
) -> ConsensusResult:
    """
    Read an image using multiple models and combine results.

    Strategies:
    - vote: Send to all providers, character-level majority wins
    - best_of: Route to single best provider for content type (free, no extra cost)
    - debate: Model A reads, Model B critiques, Model A/C resolves
    - cascade: Cheapest-first, stops when confident
    - smart: Adaptive routing — single provider for easy images, escalating for hard
    - self_correct: Single provider reads, then reviews and corrects its own output

    Args:
        max_debate_rounds: Maximum resolution passes for debate strategy (default 2).
                          Set to 3 for Iterative Consensus Ensemble (ICE) — adds a
                          third pass when uncertainty markers remain after resolution.
        quality_assessment: Pre-computed assess_image() dict for smart routing.
        max_self_correct_rounds: Correction passes for self_correct strategy (default 1, max 3).
        uncertainty_threshold: [?] marker count that triggers auto-escalation in smart strategy.
    """
    if strategy == "best_of":
        return _best_of(image_b64, media_type, prompt, system_prompt, content_type, max_tokens)
    elif strategy == "vote":
        return _vote(image_b64, media_type, prompt, system_prompt, providers, confidence_threshold, content_type, max_tokens)
    elif strategy == "debate":
        return _debate(image_b64, media_type, prompt, system_prompt, providers, max_tokens, max_debate_rounds)
    elif strategy == "cascade":
        return _cascade(image_b64, media_type, prompt, system_prompt, confidence_threshold, max_tokens)
    elif strategy == "self_correct":
        return _self_correct(image_b64, media_type, prompt, system_prompt, content_type, max_tokens, max_self_correct_rounds)
    elif strategy == "smart":
        if quality_assessment is None:
            # No quality data — fall back to vote
            return _vote(image_b64, media_type, prompt, system_prompt, providers, confidence_threshold, content_type, max_tokens)
        return _smart_route(image_b64, media_type, prompt, system_prompt, quality_assessment, content_type, max_tokens, uncertainty_threshold)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use: vote, best_of, debate, cascade, smart, self_correct")


def _best_of(
    image_b64: str, media_type: str, prompt: str, system_prompt: str,
    content_type: str, max_tokens: int,
) -> ConsensusResult:
    """Route to the single best provider for this content type.

    Confidence is capped at _SINGLE_PROVIDER_MAX_CONFIDENCE because we have
    no cross-validation.  Uncertainty markers in the output reduce it further.
    """
    provider_name = BEST_PROVIDER_ROUTING.get(content_type, "claude")

    avail = available_providers()
    if not avail:
        raise ValueError(
            "No vision providers available. Install at least one: "
            "pip install handwriting-engine[all]"
        )
    if provider_name not in avail:
        provider_name = avail[0]

    # Try preferred, fall back to others on failure
    text = ""
    used_provider = provider_name
    provider = None
    for candidate_name in [provider_name] + [p for p in avail if p != provider_name]:
        if circuit_breaker.is_open(candidate_name):
            logger.warning(f"best_of: circuit breaker open for {candidate_name}, skipping")
            continue
        try:
            provider = get_provider(candidate_name)
            text = provider.read_image(image_b64, media_type, prompt, system_prompt, max_tokens)
            used_provider = candidate_name
            circuit_breaker.record_success(candidate_name)
            break
        except Exception as e:
            logger.warning(f"best_of: {candidate_name} failed: {e}")
            circuit_breaker.record_failure(candidate_name)
            text = ""

    if not text:
        return ConsensusResult(text="", confidence=0.0, strategy_used="best_of_failed")

    confidence = _single_text_confidence(text)

    return ConsensusResult(
        text=text,
        confidence=confidence,
        confidence_level=_derive_confidence_level(confidence, text),
        provider_results={used_provider: text},
        strategy_used="best_of",
        tokens_used=provider.usage if provider else {},
    )


def _self_correct(
    image_b64: str, media_type: str, prompt: str, system_prompt: str,
    content_type: str, max_tokens: int,
    max_rounds: int = 1,
) -> ConsensusResult:
    """Two-pass self-correction: read → review own output → correct.

    Based on Journal of Documentation 2025 (peer-reviewed): GPT-4o
    self-correction drops CER from 1.75% to 1.39% on IAM (~20% reduction).
    Same model reads, then reviews its own transcription against the image.

    Args:
        max_rounds: Number of correction passes (default 1, max 3).
                   Extra passes only trigger if [?] markers remain.
    """
    from handwriting_engine.handwriting import SELF_CORRECTION_PROMPT

    max_rounds = min(max(1, max_rounds), 3)

    provider_name = BEST_PROVIDER_ROUTING.get(content_type, BEST_PROVIDER_ROUTING["default"])
    avail = available_providers()
    if not avail:
        raise ValueError(
            "No vision providers available. Install at least one: "
            "pip install handwriting-engine[all]"
        )
    if provider_name not in avail:
        provider_name = avail[0]

    if circuit_breaker.is_open(provider_name):
        # Fall back to best_of if primary provider circuit is open
        return _best_of(image_b64, media_type, prompt, system_prompt, content_type, max_tokens)

    try:
        provider = get_provider(provider_name)
        initial_text = provider.read_image(image_b64, media_type, prompt, system_prompt, max_tokens)
        circuit_breaker.record_success(provider_name)
    except Exception as e:
        logger.warning(f"self_correct: initial read failed for {provider_name}: {e}")
        circuit_breaker.record_failure(provider_name)
        return ConsensusResult(text="", confidence=0.0, strategy_used="self_correct_failed")

    if not initial_text.strip():
        return ConsensusResult(text="", confidence=0.0, strategy_used="self_correct_failed")

    corrected_text = initial_text
    rounds_done = 0

    for round_num in range(max_rounds):
        # Build the correction prompt with the current transcription
        correction_prompt = SELF_CORRECTION_PROMPT.format(
            initial_transcription=corrected_text
        )

        try:
            corrected_text = provider.read_image(
                image_b64, media_type, correction_prompt, system_prompt, max_tokens
            )
            rounds_done += 1
            circuit_breaker.record_success(provider_name)
        except Exception as e:
            logger.warning(f"self_correct: correction pass {round_num + 1} failed: {e}")
            break

        # Stop early if no uncertainty markers remain
        if _count_uncertainty_markers(corrected_text) == 0:
            break

    if not corrected_text.strip():
        corrected_text = initial_text

    confidence = _single_text_confidence(corrected_text)

    return ConsensusResult(
        text=corrected_text,
        confidence=confidence,
        confidence_level=_derive_confidence_level(confidence, corrected_text),
        provider_results={
            "initial": initial_text,
            "corrected": corrected_text,
        },
        strategy_used=f"self_correct_{rounds_done}pass",
        tokens_used=provider.usage,
    )


def _vote(
    image_b64: str, media_type: str, prompt: str, system_prompt: str,
    providers: list[str] | None, confidence_threshold: float,
    content_type: str = "default", max_tokens: int = 4096,
) -> ConsensusResult:
    """Send to N providers, word-level majority vote.

    Instead of character-level merging (which creates Frankenstein text no
    provider actually produced), this:
    1. Aligns provider outputs at the word level
    2. For each position, picks the word with the most (weighted) votes
    3. Marks positions where providers disagree with [?alt: ...]
    4. Derives confidence from the fraction of words that agreed
    """
    if providers is None:
        providers = available_providers()[:2]
    if len(providers) < 2:
        providers = available_providers()[:2]
    if not providers:
        raise ValueError(
            "No vision providers available. Install at least one: "
            "pip install handwriting-engine[all]"
        )
    if len(providers) < 2:
        p = get_provider(providers[0])
        text = p.read_image(image_b64, media_type, prompt, system_prompt, max_tokens)
        conf = _single_text_confidence(text)
        return ConsensusResult(
            text=text,
            confidence=conf,
            confidence_level=_derive_confidence_level(conf, text),
            provider_results={providers[0]: text},
            strategy_used="vote_single_fallback",
            tokens_used=p.usage,
        )

    import concurrent.futures

    def _read_from_provider(name):
        if circuit_breaker.is_open(name):
            logger.warning(f"Circuit breaker open for {name}, skipping")
            return name, "", {}, None
        try:
            p = get_provider(name)
            text = p.read_image(image_b64, media_type, prompt, system_prompt, max_tokens)
            circuit_breaker.record_success(name)
            return name, text, p.usage, p
        except Exception as e:
            logger.warning(f"Provider {name} failed: {e}")
            circuit_breaker.record_failure(name)
            return name, "", {}, None

    results = {}
    total_tokens = {}
    provider_objects = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(providers)) as pool:
        futures = {pool.submit(_read_from_provider, name): name for name in providers}
        for future in concurrent.futures.as_completed(futures):
            name, text, usage, provider_obj = future.result()
            results[name] = text
            if usage:
                total_tokens[name] = usage
            if provider_obj:
                provider_objects[name] = provider_obj

    provider_names = [name for name, v in results.items() if v]
    texts = [results[name] for name in provider_names]
    if not texts:
        return ConsensusResult(text="", confidence=0.0, provider_results=results, strategy_used="vote")

    if len(texts) == 1:
        conf = _single_text_confidence(texts[0])
        return ConsensusResult(
            text=texts[0],
            confidence=conf,
            confidence_level=_derive_confidence_level(conf, texts[0]),
            provider_results=results, strategy_used="vote_partial",
            tokens_used=total_tokens,
        )

    # Hallucination rejection: with 3+ providers, zero-weight any provider whose
    # best agreement with ANY other provider is below 60%. This catches outputs
    # that are wildly different from all others (hallucinated text).
    # Does not exclude — zero-weights so the output still appears in provider_results.
    hallucination_flagged = set()
    if len(texts) >= 3:
        for i, name in enumerate(provider_names):
            max_agreement = max(
                _word_agreement_ratio(texts[i], texts[j])
                for j in range(len(texts)) if j != i
            )
            if max_agreement < 0.60:
                hallucination_flagged.add(name)
                logger.warning(f"Vote: {name} flagged as potential hallucination "
                               f"(max agreement {max_agreement:.2f} < 0.60)")

    # Outlier detection: exclude providers with <20% agreement with ALL others.
    # Only with 4+ providers — with exactly 3, excluding one removes tiebreaking.
    # Threshold lowered from 30% to 20%: at 30%, legitimate minority readings
    # get caught when 2 providers agree on wrong text.
    excluded = []
    if len(texts) >= 4:
        filtered_names = []
        filtered_texts = []
        for i, name in enumerate(provider_names):
            agreements = []
            for j, other_name in enumerate(provider_names):
                if i != j:
                    agreements.append(_word_agreement_ratio(texts[i], texts[j]))
            if all(a < 0.20 for a in agreements):
                excluded.append(f"'{name}' excluded as outlier (agreement < 20% with all others)")
                logger.warning(f"Vote: excluding {name} as outlier")
            else:
                filtered_names.append(name)
                filtered_texts.append(texts[i])

        # Safety net: never reduce below 2 providers
        if len(filtered_texts) >= 2 and len(filtered_texts) < len(texts):
            provider_names = filtered_names
            texts = filtered_texts

    # Confidence-weighted voting: combine static provider weights with
    # per-read quality signals. A provider that returns many [?] markers
    # or very short output gets its weight reduced for this specific vote.
    # Hallucination-flagged providers get zero weight (effectively excluded from vote).
    static_weights = PROVIDER_WEIGHTS.get(content_type, PROVIDER_WEIGHTS["default"])
    provider_weights = []
    for i, name in enumerate(provider_names):
        if name in hallucination_flagged:
            provider_weights.append((name, 0.0))
            excluded.append(f"'{name}' zero-weighted: potential hallucination (agreement < 60%)")
            continue
        base_weight = static_weights.get(name, 1.0)
        # Prefer real logprob confidence when available (0.0-1.0 scale),
        # fall back to heuristic marker-counting confidence (0.0-0.7 scale)
        p_obj = provider_objects.get(name)
        try:
            logprob_conf = p_obj.get_mean_confidence() if p_obj and hasattr(p_obj, 'get_mean_confidence') else 0.0
        except (TypeError, AttributeError):
            logprob_conf = 0.0
        if isinstance(logprob_conf, (int, float)) and logprob_conf > 0:
            read_confidence = logprob_conf
            confidence_scale = 0.4 + read_confidence * 0.6
        else:
            read_confidence = _single_text_confidence(texts[i])
            # Scale: confidence 0.7 (max) -> 1.0x, confidence 0.3 -> 0.6x
            confidence_scale = 0.4 + (read_confidence / _SINGLE_PROVIDER_MAX_CONFIDENCE) * 0.6
        provider_weights.append((name, base_weight * confidence_scale))

    best_text, disagreements, confidence = _word_level_vote(
        texts, provider_weights,
    )

    if excluded:
        disagreements = disagreements + excluded

    return ConsensusResult(
        text=best_text,
        confidence=round(confidence, 3),
        confidence_level=_derive_confidence_level(confidence, best_text),
        provider_results=results,
        disagreements=disagreements,
        strategy_used="vote",
        tokens_used=total_tokens,
    )


def _debate(
    image_b64: str, media_type: str, prompt: str, system_prompt: str,
    providers: list[str] | None, max_tokens: int,
    max_rounds: int = 2,
) -> ConsensusResult:
    """Two-pass propose / independent-read / word-diff / resolve debate.

    Phase 1: Provider A reads the image (proposal).
    Phase 2: Provider B reads the SAME image independently (no bias from A).
    Phase 3: Word-diff the two reads to find disagreements.
    Phase 4: If disagreements exist, Provider B re-reads with the diff as
             context, resolving each conflict against the actual image.

    Confidence is derived from the word-level agreement ratio between
    the independent reads (phases 1-2), NOT hardcoded.
    """
    if providers is None:
        providers = available_providers()[:2]
    if not providers:
        raise ValueError(
            "No vision providers available. Install at least one: "
            "pip install handwriting-engine[all]"
        )

    if len(providers) < 2:
        # Single provider — no debate possible
        candidate = providers[0]
        if circuit_breaker.is_open(candidate):
            return ConsensusResult(text="", confidence=0.0, strategy_used="debate_failed")
        try:
            proposer = get_provider(candidate)
            initial_read = proposer.read_image(image_b64, media_type, prompt, system_prompt, max_tokens)
            circuit_breaker.record_success(candidate)
            conf = _single_text_confidence(initial_read)
            return ConsensusResult(
                text=initial_read,
                confidence=conf,
                confidence_level=_derive_confidence_level(conf, initial_read),
                provider_results={candidate: initial_read},
                strategy_used="debate_single_fallback",
            )
        except Exception as e:
            logger.warning(f"debate: {candidate} failed: {e}")
            circuit_breaker.record_failure(candidate)
            return ConsensusResult(text="", confidence=0.0, strategy_used="debate_failed")

    # Phase 1 & 2: Run proposer and critic reads in parallel (independent)
    import concurrent.futures

    def _debate_read(candidate):
        if circuit_breaker.is_open(candidate):
            logger.warning(f"debate: circuit breaker open for {candidate}, skipping")
            return candidate, "", None
        try:
            p = get_provider(candidate)
            text = p.read_image(image_b64, media_type, prompt, system_prompt, max_tokens)
            circuit_breaker.record_success(candidate)
            return candidate, text, p
        except Exception as e:
            logger.warning(f"debate: {candidate} failed: {e}")
            circuit_breaker.record_failure(candidate)
            return candidate, "", None

    proposer_name, proposer, initial_read = None, None, ""
    critic_name, critic, independent_read = None, None, ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(providers)) as pool:
        futures = {pool.submit(_debate_read, name): name for name in providers}
        results_list = []
        for future in concurrent.futures.as_completed(futures):
            name, text, provider_obj = future.result()
            if text:
                results_list.append((name, text, provider_obj))

    # Assign first two successful reads as proposer and critic
    if len(results_list) >= 2:
        proposer_name, initial_read, proposer = results_list[0]
        critic_name, independent_read, critic = results_list[1]
    elif len(results_list) == 1:
        proposer_name, initial_read, proposer = results_list[0]

    if not initial_read:
        return ConsensusResult(text="", confidence=0.0, strategy_used="debate_failed")

    if not independent_read:
        # Only proposer succeeded — return single-provider result
        conf = _single_text_confidence(initial_read)
        return ConsensusResult(
            text=initial_read,
            confidence=conf,
            confidence_level=_derive_confidence_level(conf, initial_read),
            provider_results={proposer_name: initial_read},
            strategy_used="debate_single_fallback",
        )

    # Phase 3: Word-level diff to find real disagreements
    word_diffs = _word_diff(initial_read, independent_read)
    agreement_ratio = _word_agreement_ratio(initial_read, independent_read)

    if not word_diffs:
        # Both providers agree on every word — high confidence
        conf = min(0.95, agreement_ratio)
        return ConsensusResult(
            text=initial_read,
            confidence=conf,
            confidence_level=_derive_confidence_level(conf, initial_read),
            provider_results={
                proposer_name: initial_read,
                critic_name: independent_read,
            },
            strategy_used="debate_agreed",
            tokens_used={
                key: proposer.usage.get(key, 0) + critic.usage.get(key, 0)
                for key in set(list(proposer.usage.keys()) + list(critic.usage.keys()))
            },
        )

    # Phase 4: Targeted resolution — a DIFFERENT model verifies disputed words.
    # Research: LLMs cannot reliably self-correct OCR. Cross-model resolution
    # gives a fresh perspective on ambiguous characters.
    diff_summary = "\n".join(
        f"- Area {i+1}: \"{d['word_a']}\" or \"{d['word_b']}\"?"
        for i, d in enumerate(word_diffs[:30])
    )
    resolve_prompt = (
        f"Two preliminary readings of this handwritten image identified areas "
        f"that need careful verification. They agreed on most text but these "
        f"specific words need your attention:\n\n"
        f"{diff_summary}\n\n"
        f"Please look carefully at each flagged area in the image and report "
        f"what you actually see written. Base your reading on the image only — "
        f"the preliminary readings are provided as areas to pay extra attention to, "
        f"not as options to choose between.\n"
        f"Then produce the complete final text with all areas resolved. "
        f"Mark any word you are still uncertain about with [?]. "
        f"Preserve original spelling exactly as written."
    )

    # Select resolver: prefer a third provider, fall back to proposer (fresh perspective
    # since it hasn't seen the critic's output or the disagreement summary before)
    resolver_name = proposer_name
    resolver = proposer
    avail = available_providers()
    for candidate in avail:
        if candidate != proposer_name and candidate != critic_name and not circuit_breaker.is_open(candidate):
            try:
                resolver = get_provider(candidate)
                resolver_name = candidate
                break
            except Exception:
                continue

    final = resolver.read_image(image_b64, media_type, resolve_prompt, system_prompt, max_tokens)
    used_providers = {proposer_name, critic_name, resolver_name}
    strategy_used = "debate_resolved"

    # Phase 5 (ICE): Optional third resolution pass when uncertainty markers remain.
    # Only triggers when max_rounds >= 3, [?] markers exist, and a fresh provider is available.
    if max_rounds >= 3 and _UNCERTAINTY_RE.search(final):
        avail = available_providers()
        ice_candidates = [c for c in avail if c not in used_providers and not circuit_breaker.is_open(c)]
        if ice_candidates:
            remaining_markers = _UNCERTAINTY_RE.findall(final)
            ice_prompt = (
                f"A previous multi-model reading of this handwritten image still has "
                f"{len(remaining_markers)} uncertain area(s). The current best reading is:\n\n"
                f"\"{final}\"\n\n"
                f"Please look at the image carefully and resolve each [?], [illegible], "
                f"or ??? marker. Produce the complete final text. "
                f"Mark any word you are still uncertain about with [?]. "
                f"Preserve original spelling exactly as written."
            )
            try:
                ice_provider = get_provider(ice_candidates[0])
                ice_result = ice_provider.read_image(image_b64, media_type, ice_prompt, system_prompt, max_tokens)
                if ice_result.strip():
                    # Only accept if it reduced uncertainty markers
                    new_markers = len(_UNCERTAINTY_RE.findall(ice_result))
                    if new_markers < len(remaining_markers):
                        final = ice_result
                        used_providers.add(ice_candidates[0])
                        strategy_used = "debate_ice_resolved"
                        logger.info(f"ICE pass reduced markers from {len(remaining_markers)} to {new_markers}")
            except Exception as e:
                logger.warning(f"debate ICE pass failed: {e}")

    # Confidence: base is agreement ratio, boosted slightly by resolution pass(es)
    resolution_boost = min(0.1, (1.0 - agreement_ratio) * 0.5)
    confidence = min(0.92, agreement_ratio + resolution_boost)

    disagreement_strs = [
        f"'{d['word_a']}' vs '{d['word_b']}'" for d in word_diffs[:20]
    ]

    return ConsensusResult(
        text=final,
        confidence=round(confidence, 3),
        confidence_level=_derive_confidence_level(confidence, final),
        provider_results={
            proposer_name: initial_read,
            critic_name: independent_read,
            f"{resolver_name}_resolved": final,
        },
        disagreements=disagreement_strs,
        strategy_used=strategy_used,
        tokens_used={
            key: proposer.usage.get(key, 0) + critic.usage.get(key, 0)
            for key in set(list(proposer.usage.keys()) + list(critic.usage.keys()))
        },
    )


def _cascade(
    image_b64: str, media_type: str, prompt: str, system_prompt: str,
    confidence_threshold: float, max_tokens: int,
) -> ConsensusResult:
    """Start with cheapest provider, escalate and cross-validate.

    Instead of trusting a single provider's heuristic confidence:
    1. Get a read from the cheapest provider
    2. If the text has few uncertainty markers, accept at capped confidence
    3. Otherwise escalate to the next provider and cross-validate —
       confidence comes from word-level agreement between the two reads
    """
    avail = available_providers()
    if not avail:
        raise ValueError(
            "No vision providers available. Install at least one: "
            "pip install handwriting-engine[all]"
        )

    order = [p for p in CASCADE_ORDER if p in avail]
    if not order:
        order = avail

    all_results = {}
    prev_text = None
    prev_name = None

    for provider_name in order:
        if circuit_breaker.is_open(provider_name):
            logger.warning(f"Cascade: circuit breaker open for {provider_name}, skipping")
            continue
        try:
            provider = get_provider(provider_name)
            text = provider.read_image(image_b64, media_type, prompt, system_prompt, max_tokens)
            all_results[provider_name] = text
            circuit_breaker.record_success(provider_name)

            if prev_text is not None:
                # Cross-validate against previous tier
                agreement = _word_agreement_ratio(prev_text, text)
                if agreement >= confidence_threshold:
                    # Both tiers agree — use the higher-tier read, confidence from agreement
                    conf = round(min(0.95, agreement), 3)
                    return ConsensusResult(
                        text=text,
                        confidence=conf,
                        confidence_level=_derive_confidence_level(conf, text),
                        provider_results=all_results,
                        strategy_used=f"cascade_{provider_name}_validated",
                        tokens_used=provider.usage,
                    )
                # Disagreement — continue escalating
                logger.info(
                    f"Cascade: {prev_name} and {provider_name} disagree "
                    f"(agreement={agreement:.2f}), escalating"
                )
            else:
                # First provider — check if text looks clean enough to stop
                single_conf = _single_text_confidence(text)
                if single_conf >= confidence_threshold:
                    return ConsensusResult(
                        text=text,
                        confidence=round(single_conf, 3),
                        confidence_level=_derive_confidence_level(single_conf, text),
                        provider_results=all_results,
                        strategy_used=f"cascade_{provider_name}",
                        tokens_used=provider.usage,
                    )

            prev_text = text
            prev_name = provider_name

        except Exception as e:
            logger.warning(f"Cascade: {provider_name} failed: {e}")
            circuit_breaker.record_failure(provider_name)

    # Exhausted all providers — return last result with honest confidence
    if all_results:
        last_name = list(all_results.keys())[-1]
        last_text = all_results[last_name]
        # If we have 2+ results, use agreement; otherwise single-text confidence
        if len(all_results) >= 2:
            texts = list(all_results.values())
            agreement = _word_agreement_ratio(texts[-2], texts[-1])
            confidence = round(agreement, 3)
        else:
            confidence = round(_single_text_confidence(last_text), 3)
        return ConsensusResult(
            text=last_text,
            confidence=confidence,
            confidence_level=_derive_confidence_level(confidence, last_text),
            provider_results=all_results,
            strategy_used=f"cascade_best_effort_{last_name}",
        )

    return ConsensusResult(text="", confidence=0.0, provider_results=all_results, strategy_used="cascade_failed")


def _smart_route(
    image_b64: str, media_type: str, prompt: str, system_prompt: str,
    quality_assessment: dict, content_type: str, max_tokens: int,
    uncertainty_threshold: int = 3,
) -> ConsensusResult:
    """Adaptive routing based on image quality — spend API calls where they matter.

    Easy (sharp, high contrast): Single Gemini read — cheapest, best CER.
    Medium (fair quality, one issue): Gemini + cross-validator.
    Hard (poor, faint, multiple issues): Full vote with all providers.

    After easy/medium reads, if [?] marker count exceeds uncertainty_threshold,
    automatically escalates to self_correct for a correction pass.
    """
    from handwriting_engine._constants import (
        SMART_ROUTE_EASY_BLUR, SMART_ROUTE_EASY_CONTRAST,
        SMART_ROUTE_HARD_BLUR, SMART_ROUTE_HARD_CONTRAST,
    )

    quality = quality_assessment.get("quality", "fair")
    blur = quality_assessment.get("blur_score", 100)
    contrast = quality_assessment.get("contrast_score", 0.4)
    faint = quality_assessment.get("faint_ink", False)
    issues = quality_assessment.get("issues", [])

    # Classify difficulty
    if quality == "good" and blur >= SMART_ROUTE_EASY_BLUR and contrast >= SMART_ROUTE_EASY_CONTRAST:
        difficulty = "easy"
    elif quality == "poor" or faint or len(issues) >= 2 or blur < SMART_ROUTE_HARD_BLUR or contrast < SMART_ROUTE_HARD_CONTRAST:
        difficulty = "hard"
    else:
        difficulty = "medium"

    logger.info(f"Smart route: difficulty={difficulty} (quality={quality}, blur={blur}, contrast={contrast})")

    avail = available_providers()
    if not avail:
        raise ValueError("No vision providers available.")

    if difficulty == "easy":
        # Single best provider (Gemini) — cheapest and best CER
        result = _best_of(image_b64, media_type, prompt, system_prompt, content_type, max_tokens)
        marker_count = _count_uncertainty_markers(result.text)
        if marker_count > uncertainty_threshold:
            logger.info(f"Smart route: escalating to self_correct ({marker_count} markers > threshold {uncertainty_threshold})")
            corrected = _self_correct(image_b64, media_type, prompt, system_prompt, content_type, max_tokens)
            corrected.strategy_used = "smart→self_correct"
            return corrected
        return result

    elif difficulty == "medium":
        # Gemini + one cross-validator via cascade
        result = _cascade(image_b64, media_type, prompt, system_prompt, 0.7, max_tokens)
        marker_count = _count_uncertainty_markers(result.text)
        if marker_count > uncertainty_threshold:
            logger.info(f"Smart route medium: escalating to self_correct ({marker_count} markers)")
            corrected = _self_correct(image_b64, media_type, prompt, system_prompt, content_type, max_tokens)
            corrected.strategy_used = "smart→self_correct"
            return corrected
        return result

    else:
        # Hard: full vote with all available providers
        return _vote(image_b64, media_type, prompt, system_prompt, None, 0.8, content_type, max_tokens)


# ---------------------------------------------------------------------------
# Confidence scoring — derived from actual signals, not vibes
# ---------------------------------------------------------------------------


def _derive_confidence_level(confidence: float, text: str) -> str:
    """Derive HIGH/LOW confidence level from score and text signals.

    LOW when: confidence < 0.5, multiple [?] markers, very short output,
    or faint content detected. HIGH otherwise.
    """
    if confidence < 0.5:
        return "LOW"

    marker_count = len(_UNCERTAINTY_RE.findall(text))
    words = text.split()

    # Multiple uncertainty markers = LOW
    if marker_count >= 3:
        return "LOW"

    # Very short output for what should be a page = LOW
    if len(words) < 5 and not text.strip().startswith("["):
        return "LOW"

    # Single-character answers: always LOW without cross-validation
    stripped = text.strip()
    if len(stripped) <= 2 and stripped.isalpha():
        return "LOW"

    # High marker ratio = LOW
    if words and marker_count / len(words) > 0.1:
        return "LOW"

    # Check for faint content flags
    if "[illegible" in text.lower() or "page faint" in text.lower():
        return "LOW"

    return "HIGH"


def _single_text_confidence(text: str) -> float:
    """Confidence for a single-provider read (no cross-validation).

    Capped at _SINGLE_PROVIDER_MAX_CONFIDENCE because one opinion can't
    be certain. Reduced by uncertainty markers — each [?] is a real flag,
    not a minor penalty.

    Short, specific text (names, single words) gets an extra penalty:
    high specificity from a single provider without cross-validation is
    the hallucination pattern ("Sheldon" → "Steffon Espericueta").
    """
    if not text or not text.strip():
        return 0.0

    words = text.split()
    if not words:
        return 0.0

    marker_count = len(_UNCERTAINTY_RE.findall(text))
    marker_ratio = marker_count / max(len(words), 1)

    # Start at cap, subtract based on marker density
    score = _SINGLE_PROVIDER_MAX_CONFIDENCE - (marker_ratio * 1.5)

    # Penalize very short output (< 3 words) — suspicious for page reads
    if len(words) < 3:
        score -= 0.15

    # Short text with NO uncertainty markers is the hallucination pattern:
    # model confidently produces specific text from ambiguous input.
    # 1-5 words with zero [?] flags and no illegible markers = overconfident.
    if len(words) <= 5 and marker_count == 0:
        score -= 0.10

    # Single-character answers: cap even lower — one character can never
    # have enough visual evidence for moderate confidence without cross-validation
    stripped = text.strip()
    if len(stripped) <= 2 and stripped.isalpha():
        score = min(score, 0.45)

    # Penalize highly repetitive text
    unique_words = len(set(w.lower() for w in words))
    if len(words) > 5 and unique_words / len(words) < 0.15:
        score -= 0.2

    return round(max(0.0, min(_SINGLE_PROVIDER_MAX_CONFIDENCE, score)), 3)


def _estimate_confidence(text: str) -> float:
    """Legacy alias — routes to _single_text_confidence."""
    return _single_text_confidence(text)


# ---------------------------------------------------------------------------
# Word-level alignment and voting
# ---------------------------------------------------------------------------

def _tokenize_preserving_newlines(text: str) -> list[str]:
    """Split text into words while preserving line structure.

    Newlines become explicit '\\n' tokens so line breaks survive voting.
    """
    tokens = []
    for line in text.split("\n"):
        words = line.split()
        tokens.extend(words)
        tokens.append("\n")
    # Remove trailing newline token
    if tokens and tokens[-1] == "\n":
        tokens.pop()
    return tokens


def _word_diff(text_a: str, text_b: str) -> list[dict]:
    """Find word-level disagreements between two reads.

    Returns list of {pos, word_a, word_b} dicts for each disagreement.
    """
    words_a = _tokenize_preserving_newlines(text_a)
    words_b = _tokenize_preserving_newlines(text_b)
    matcher = SequenceMatcher(None, words_a, words_b)
    diffs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        chunk_a = " ".join(w for w in words_a[i1:i2] if w != "\n") or "(empty)"
        chunk_b = " ".join(w for w in words_b[j1:j2] if w != "\n") or "(empty)"
        if chunk_a.strip() or chunk_b.strip():
            diffs.append({"pos": i1, "word_a": chunk_a, "word_b": chunk_b})
    return diffs[:30]


def _word_agreement_ratio(text_a: str, text_b: str) -> float:
    """Fraction of words that match between two texts (0.0–1.0)."""
    words_a = _tokenize_preserving_newlines(text_a)
    words_b = _tokenize_preserving_newlines(text_b)
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    matcher = SequenceMatcher(None, words_a, words_b)
    return matcher.ratio()


def _word_level_vote(
    texts: list[str],
    provider_weights: list[tuple[str, float]],
    writer_profile: dict | None = None,
) -> tuple[str, list[str], float]:
    """Word-level majority vote across N provider outputs.

    For each word position:
    - If all providers agree → keep the word
    - If majority agrees → keep majority, record disagreement
    - If no majority → try char-level confusion-pair resolution (S5);
      if that defers, keep highest-weighted provider's word, mark with [?alt: ...]

    Returns (final_text, disagreements, confidence).
    Confidence = fraction of word positions where providers agreed.
    """
    writer_resolutions = (
        (writer_profile or {}).get("confusion_resolutions") or {}
    )
    all_words = [_tokenize_preserving_newlines(t) for t in texts]

    # Use highest-weighted provider as alignment anchor
    anchor_idx = 0
    best_weight = 0.0
    for i, (_, w) in enumerate(provider_weights):
        if w > best_weight:
            best_weight = w
            anchor_idx = i

    anchor_words = all_words[anchor_idx]

    # Align each other text to the anchor at word level
    alignments = []
    for i, words in enumerate(all_words):
        if i == anchor_idx:
            alignments.append(list(enumerate(range(len(anchor_words)))))
            continue
        matcher = SequenceMatcher(None, anchor_words, words)
        alignments.append(matcher.get_opcodes())

    # Build per-position vote map: position -> {word: total_weight}
    max_pos = len(anchor_words)
    votes: dict[int, dict[str, float]] = {}

    # Anchor votes
    anchor_name, anchor_weight = provider_weights[anchor_idx]
    for pos, word in enumerate(anchor_words):
        votes[pos] = {word: anchor_weight}

    # Other provider votes, aligned to anchor positions
    for i, words in enumerate(all_words):
        if i == anchor_idx:
            continue
        pname, pweight = provider_weights[i]
        matcher = SequenceMatcher(None, anchor_words, words)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for offset, pos in enumerate(range(i1, i2)):
                    word = anchor_words[pos]
                    votes.setdefault(pos, {})
                    votes[pos][word] = votes[pos].get(word, 0) + pweight
            elif tag == "replace":
                # Map replaced words position-by-position where possible
                a_span = list(range(i1, i2))
                b_span = list(range(j1, j2))
                for k in range(max(len(a_span), len(b_span))):
                    pos = a_span[min(k, len(a_span) - 1)]
                    word = words[b_span[min(k, len(b_span) - 1)]]
                    votes.setdefault(pos, {})
                    votes[pos][word] = votes[pos].get(word, 0) + pweight

    # Resolve each position
    result_words = []
    disagreements = []
    agreed = 0
    total = 0

    for pos in range(max_pos):
        word_votes = votes.get(pos, {})
        if not word_votes:
            result_words.append(anchor_words[pos])
            agreed += 1
            total += 1
            continue

        # Skip newline tokens from disagreement counting
        if list(word_votes.keys()) == ["\n"]:
            result_words.append("\n")
            continue

        total += 1
        sorted_votes = sorted(word_votes.items(), key=lambda x: x[1], reverse=True)
        winner = sorted_votes[0][0]
        winner_weight = sorted_votes[0][1]

        if len(sorted_votes) == 1:
            # Unanimous
            result_words.append(winner)
            agreed += 1
        else:
            # Check if winner has strict majority of total weight
            total_weight = sum(w for _, w in sorted_votes)
            runner_up = sorted_votes[1][0]

            if winner_weight > total_weight * 0.5:
                # Majority — use winner but record the disagreement
                result_words.append(winner)
                if winner != "\n" and runner_up != "\n":
                    disagreements.append(f"'{runner_up}' vs '{winner}' (majority)")
            else:
                # No majority — try char-level confusion-pair resolution
                # before falling back to the [?alt: …] marker.
                candidates = [w for w, _ in sorted_votes if w != "\n"]
                weights = [v for w, v in sorted_votes if w != "\n"]
                resolved = resolve_char_level(
                    candidates,
                    weights,
                    writer_resolutions=writer_resolutions,
                )
                if resolved is not None:
                    result_words.append(resolved)
                else:
                    # No clean confusion-pair match — emit ambiguous marker.
                    result_words.append(winner)
                    alts = [w for w, _ in sorted_votes[1:] if w != "\n"]
                    if alts and winner != "\n":
                        alt_str = "/".join(alts[:2])
                        result_words.append(f"[?alt: {alt_str}]")
                        disagreements.append(f"'{winner}' vs '{alt_str}' (no majority)")

    # Reconstruct text from tokens
    final_parts = []
    for w in result_words:
        if w == "\n":
            final_parts.append("\n")
        else:
            if final_parts and final_parts[-1] != "\n":
                final_parts.append(" ")
            final_parts.append(w)
    final_text = "".join(final_parts).strip()

    confidence = agreed / total if total > 0 else 0.0
    # Cap: even with full agreement, multi-provider max is 0.95
    confidence = min(0.95, confidence)

    return final_text, disagreements[:20], confidence


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_strategy_cost(
    strategy: str,
    providers: list[str] | None = None,
    image_count: int = 1,
    avg_input_tokens: int = 2000,
    avg_output_tokens: int = 1000,
) -> dict:
    """Estimate USD cost for a consensus strategy before running.

    Returns dict with per-provider costs and total.
    """
    from handwriting_engine._constants import COST_PER_1M_TOKENS

    if providers is None:
        providers = available_providers()

    # How many provider calls per image for each strategy
    calls_per_image = {
        "best_of": 1,
        "vote": len(providers),
        "debate": 3,  # 2 independent reads + 1 resolution
        "cascade": 1.5,  # Average: sometimes 1, sometimes 2-3
    }

    num_calls = calls_per_image.get(strategy, len(providers))

    costs = {}
    for p in providers:
        rates = COST_PER_1M_TOKENS.get(p, {"input": 3.0, "output": 15.0})
        per_call = (
            (avg_input_tokens / 1_000_000) * rates["input"]
            + (avg_output_tokens / 1_000_000) * rates["output"]
        )
        provider_calls = num_calls if strategy == "vote" else (num_calls / len(providers) if providers else 0)
        costs[p] = round(per_call * provider_calls * image_count, 4)

    total = sum(costs.values())
    return {"per_provider": costs, "total": round(total, 4), "strategy": strategy, "image_count": image_count}
