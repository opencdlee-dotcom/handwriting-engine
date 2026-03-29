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
) -> ConsensusResult:
    """
    Read an image using multiple models and combine results.

    Strategies:
    - vote: Send to all providers, character-level majority wins
    - best_of: Route to single best provider for content type (free, no extra cost)
    - debate: Model A reads, Model B critiques, Model A/C resolves
    """
    if strategy == "best_of":
        return _best_of(image_b64, media_type, prompt, system_prompt, content_type, max_tokens)
    elif strategy == "vote":
        return _vote(image_b64, media_type, prompt, system_prompt, providers, confidence_threshold, content_type, max_tokens)
    elif strategy == "debate":
        return _debate(image_b64, media_type, prompt, system_prompt, providers, max_tokens)
    elif strategy == "cascade":
        return _cascade(image_b64, media_type, prompt, system_prompt, confidence_threshold, max_tokens)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use: vote, best_of, debate, cascade")


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
    from handwriting_engine._constants import API_TIMEOUT_SECONDS

    def _read_from_provider(name):
        if circuit_breaker.is_open(name):
            logger.warning(f"Circuit breaker open for {name}, skipping")
            return name, "", {}
        try:
            p = get_provider(name)
            text = p.read_image(image_b64, media_type, prompt, system_prompt, max_tokens)
            circuit_breaker.record_success(name)
            return name, text, p.usage
        except Exception as e:
            logger.warning(f"Provider {name} failed: {e}")
            circuit_breaker.record_failure(name)
            return name, "", {}

    results = {}
    total_tokens = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(providers)) as pool:
        futures = {pool.submit(_read_from_provider, name): name for name in providers}
        for future in concurrent.futures.as_completed(futures):
            name, text, usage = future.result()
            results[name] = text
            if usage:
                total_tokens[name] = usage

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

    # Outlier detection: exclude providers with <30% agreement with ALL others
    excluded = []
    if len(texts) >= 3:
        filtered_names = []
        filtered_texts = []
        for i, name in enumerate(provider_names):
            agreements = []
            for j, other_name in enumerate(provider_names):
                if i != j:
                    agreements.append(_word_agreement_ratio(texts[i], texts[j]))
            if all(a < 0.30 for a in agreements):
                excluded.append(f"'{name}' excluded as outlier (agreement < 30% with all others)")
                logger.warning(f"Vote: excluding {name} as outlier")
            else:
                filtered_names.append(name)
                filtered_texts.append(texts[i])

        if filtered_texts and len(filtered_texts) < len(texts):
            provider_names = filtered_names
            texts = filtered_texts

    weights = PROVIDER_WEIGHTS.get(content_type, PROVIDER_WEIGHTS["default"])
    provider_weights = [(provider_names[i], weights.get(provider_names[i], 1.0)) for i in range(len(texts))]

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

    # Phase 1: Propose — try preferred proposer, fall back to others
    initial_read = ""
    proposer = None
    proposer_name = None
    for candidate in providers:
        if circuit_breaker.is_open(candidate):
            logger.warning(f"debate: circuit breaker open for {candidate}, skipping")
            continue
        try:
            proposer = get_provider(candidate)
            initial_read = proposer.read_image(image_b64, media_type, prompt, system_prompt, max_tokens)
            proposer_name = candidate
            circuit_breaker.record_success(candidate)
            break
        except Exception as e:
            logger.warning(f"debate: proposer {candidate} failed: {e}")
            circuit_breaker.record_failure(candidate)

    if not initial_read:
        return ConsensusResult(text="", confidence=0.0, strategy_used="debate_failed")

    if len(providers) < 2:
        conf = _single_text_confidence(initial_read)
        return ConsensusResult(
            text=initial_read,
            confidence=conf,
            confidence_level=_derive_confidence_level(conf, initial_read),
            provider_results={proposer_name: initial_read},
            strategy_used="debate_single_fallback",
        )

    # Phase 2: Independent read — try critic candidates (skip proposer used)
    independent_read = ""
    critic = None
    critic_name = None
    critic_candidates = [p for p in providers if p != proposer_name]
    for candidate in critic_candidates:
        if circuit_breaker.is_open(candidate):
            logger.warning(f"debate: circuit breaker open for {candidate}, skipping")
            continue
        try:
            critic = get_provider(candidate)
            independent_read = critic.read_image(image_b64, media_type, prompt, system_prompt, max_tokens)
            critic_name = candidate
            circuit_breaker.record_success(candidate)
            break
        except Exception as e:
            logger.warning(f"debate: critic {candidate} failed: {e}")
            circuit_breaker.record_failure(candidate)

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
            tokens_used={**proposer.usage, **critic.usage},
        )

    # Phase 4: Targeted resolution — critic re-reads ONLY disputed words
    diff_summary = "\n".join(
        f"- Position {d['pos']}: Provider A read \"{d['word_a']}\" vs Provider B read \"{d['word_b']}\""
        for d in word_diffs[:30]
    )
    resolve_prompt = (
        f"Two AI models read this handwritten image independently. "
        f"They agreed on most text but disagreed on these specific words:\n\n"
        f"{diff_summary}\n\n"
        f"Look at the image carefully and for EACH disagreement above, "
        f"state which reading is correct (or provide a third reading if both are wrong). "
        f"Then produce the complete final text with all disagreements resolved. "
        f"Mark any word you are still uncertain about with [?]. "
        f"Preserve original spelling exactly as written."
    )
    final = critic.read_image(image_b64, media_type, resolve_prompt, system_prompt, max_tokens)

    # Confidence: base is agreement ratio, boosted slightly by resolution pass
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
            f"{critic_name}_resolved": final,
        },
        disagreements=disagreement_strs,
        strategy_used="debate_resolved",
        tokens_used={**proposer.usage, **critic.usage},
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


# ---------------------------------------------------------------------------
# Confidence scoring — derived from actual signals, not vibes
# ---------------------------------------------------------------------------

_UNCERTAINTY_RE = re.compile(
    r"\[\?\]|\?\?\?|\[illegible[^\]]*\]|\[unclear\]|unable to read",
    re.IGNORECASE,
)


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
) -> tuple[str, list[str], float]:
    """Word-level majority vote across N provider outputs.

    For each word position:
    - If all providers agree → keep the word
    - If majority agrees → keep majority, record disagreement
    - If no majority → keep highest-weighted provider's word, mark with [?alt: ...]

    Returns (final_text, disagreements, confidence).
    Confidence = fraction of word positions where providers agreed.
    """
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
                # No majority — use highest-weighted provider, mark ambiguous
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
