"""Synthetic OCR corruption — apply realistic error patterns to clean text.

The patterns here are drawn from three sources:
1. Classical OCR confusion tables (rn↔m, cl↔d, ii↔u, oo↔co, etc.)
2. LLM-VLM specific failure modes observed in handwriting (doubled letters,
   smushed adjacent words, capitalization slips, missed diacritics, terminal
   punctuation drops)
3. Visual character similarity in handwriting (l↔1↔I, S↔5, B↔8, Z↔2, G↔6, o↔0)

All patterns are applied stochastically — a single string passes through every
mutator in sequence with low per-event probability, so the resulting corruption
mixes multiple error types per example. This matches what we see in real VLM
output (a page rarely has only one error type).

Determinism: every function takes a `random.Random` instance — pass a seeded
one for reproducible corpora.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


# =====================================================================
# Confusion tables
# =====================================================================

# Bidirectional letter-pair confusions (input → output, weight)
# Higher weight = more frequent in real OCR/HTR output.
_PAIR_CONFUSIONS: list[tuple[str, str, int]] = [
    ("rn", "m", 8),
    ("m", "rn", 4),
    ("cl", "d", 7),
    ("d", "cl", 3),
    ("ii", "u", 6),
    ("u", "ii", 2),
    ("oo", "co", 4),
    ("co", "oo", 2),
    ("ri", "n", 5),
    ("n", "ri", 2),
    ("vv", "w", 5),
    ("w", "vv", 2),
    ("ni", "m", 4),
    ("nn", "m", 4),
    ("ll", "h", 3),
    ("h", "ll", 1),
    ("ee", "ce", 3),
    ("le", "te", 3),
    ("te", "le", 3),
    ("a", "o", 5),
    ("o", "a", 5),
    ("e", "c", 5),
    ("c", "e", 4),
    ("i", "l", 5),
    ("l", "i", 4),
    ("u", "v", 4),
    ("v", "u", 3),
    ("s", "z", 2),
    ("z", "s", 2),
    ("g", "q", 3),
    ("q", "g", 2),
    ("h", "n", 3),
    ("n", "h", 2),
    ("f", "t", 3),
    ("t", "f", 2),
]

# Single-character substitutions (digit-letter visual similarity)
_DIGIT_LETTER_CONFUSIONS: list[tuple[str, str, int]] = [
    ("0", "o", 4),
    ("o", "0", 2),
    ("1", "l", 4),
    ("l", "1", 2),
    ("1", "I", 3),
    ("I", "1", 2),
    ("5", "S", 3),
    ("S", "5", 2),
    ("8", "B", 3),
    ("B", "8", 2),
    ("2", "Z", 2),
    ("Z", "2", 2),
    ("6", "G", 2),
    ("G", "6", 2),
    ("g", "9", 2),
    ("9", "g", 2),
]


def _weighted_choice(rng: random.Random, options: list[tuple[str, str, int]]) -> tuple[str, str]:
    """Sample (input_pattern, output_pattern) by weight."""
    total = sum(w for _, _, w in options)
    r = rng.uniform(0, total)
    upto = 0
    for inp, out, w in options:
        upto += w
        if upto >= r:
            return inp, out
    return options[-1][0], options[-1][1]


# =====================================================================
# Mutators — each takes (text, rng) → mutated text
# =====================================================================

def apply_pair_confusion(text: str, rng: random.Random, prob: float = 0.04) -> str:
    """Per-character pass: occasionally swap a 1-2 char pattern with a confusion.

    `prob` is per-position probability of attempting a swap. Most positions
    will not match any pattern; the actual swap rate is much lower.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        if rng.random() < prob:
            # Try a 2-char pattern at this position
            two = text[i:i + 2]
            applicable = [(inp, op, w) for inp, op, w in _PAIR_CONFUSIONS if inp == two]
            if applicable:
                _, op = _weighted_choice(rng, applicable)
                out.append(op)
                i += 2
                continue
            # Try single-char (letter-letter) confusion
            one = text[i:i + 1]
            applicable = [(inp, op, w) for inp, op, w in _PAIR_CONFUSIONS if inp == one]
            if applicable:
                _, op = _weighted_choice(rng, applicable)
                out.append(op)
                i += 1
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def apply_digit_letter_confusion(text: str, rng: random.Random, prob: float = 0.03) -> str:
    """Swap visually-similar digits and letters."""
    out = []
    for ch in text:
        if rng.random() < prob:
            applicable = [(inp, op, w) for inp, op, w in _DIGIT_LETTER_CONFUSIONS if inp == ch]
            if applicable:
                _, op = _weighted_choice(rng, applicable)
                out.append(op)
                continue
        out.append(ch)
    return "".join(out)


def apply_doubling(text: str, rng: random.Random, prob: float = 0.012) -> str:
    """Occasionally double a letter (common HTR error: 'celll' for 'cell')."""
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and rng.random() < prob:
            out.append(ch)
    return "".join(out)


def apply_dropping(text: str, rng: random.Random, prob: float = 0.012) -> str:
    """Drop a letter occasionally ('mitcondria' for 'mitochondria')."""
    out = []
    for i, ch in enumerate(text):
        if ch.isalpha() and rng.random() < prob and 0 < i < len(text) - 1:
            continue
        out.append(ch)
    return "".join(out)


def apply_transposition(text: str, rng: random.Random, prob: float = 0.008) -> str:
    """Swap adjacent letters ('mitochondira' for 'mitochondria')."""
    chars = list(text)
    i = 0
    while i < len(chars) - 1:
        if chars[i].isalpha() and chars[i + 1].isalpha() and rng.random() < prob:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            i += 2
        else:
            i += 1
    return "".join(chars)


def apply_smush_split(text: str, rng: random.Random, smush_prob: float = 0.015, split_prob: float = 0.008) -> str:
    """Smush adjacent words (drop space) or split a word (insert space)."""
    # Smush: drop occasional spaces
    parts = text.split(" ")
    if len(parts) <= 1:
        return text
    smushed = [parts[0]]
    for p in parts[1:]:
        if rng.random() < smush_prob:
            smushed[-1] += p
        else:
            smushed.append(p)

    # Split: occasionally insert a space inside a long-enough word
    out = []
    for word in smushed:
        if len(word) > 6 and rng.random() < split_prob:
            cut = rng.randint(2, len(word) - 2)
            out.append(word[:cut] + " " + word[cut:])
        else:
            out.append(word)
    return " ".join(out)


def apply_capitalization_slip(text: str, rng: random.Random, prob: float = 0.01) -> str:
    """Occasionally flip case of a letter (HTR mis-reads sentence-initial caps)."""
    out = []
    for ch in text:
        if ch.isalpha() and rng.random() < prob:
            out.append(ch.lower() if ch.isupper() else ch.upper())
        else:
            out.append(ch)
    return "".join(out)


def apply_punctuation_drop(text: str, rng: random.Random, prob: float = 0.10) -> str:
    """Occasionally drop terminal punctuation. HTR loses these often."""
    if not text:
        return text
    if text[-1] in ".,;:!?" and rng.random() < prob:
        return text[:-1]
    return text


def apply_diacritic_strip(text: str, rng: random.Random, prob: float = 0.6) -> str:
    """Strip diacritics — VLM/HTR routinely drops them (résumé→resume)."""
    if rng.random() > prob:
        return text
    table = str.maketrans(
        "áàâäãåéèêëíìîïóòôöõúùûüñç",
        "aaaaaaeeeeiiiiooooouuuunc",
    )
    return text.translate(table)


# =====================================================================
# Pipeline
# =====================================================================

@dataclass
class CorruptionConfig:
    """Per-mutator probabilities. Defaults tuned to roughly match observed
    Gemini Flash error rates (~1-2% CER) at a moderate setting and ~5-10%
    at an aggressive setting, so the model sees a range of difficulties."""
    pair_confusion_prob: float = 0.04
    digit_letter_prob: float = 0.03
    doubling_prob: float = 0.012
    dropping_prob: float = 0.012
    transposition_prob: float = 0.008
    smush_prob: float = 0.015
    split_prob: float = 0.008
    capitalization_prob: float = 0.01
    punctuation_drop_prob: float = 0.10
    diacritic_strip_prob: float = 0.6

    @classmethod
    def light(cls) -> "CorruptionConfig":
        return cls(
            pair_confusion_prob=0.015,
            digit_letter_prob=0.01,
            doubling_prob=0.004,
            dropping_prob=0.004,
            transposition_prob=0.003,
            smush_prob=0.005,
            split_prob=0.003,
            capitalization_prob=0.005,
            punctuation_drop_prob=0.05,
        )

    @classmethod
    def aggressive(cls) -> "CorruptionConfig":
        return cls(
            pair_confusion_prob=0.08,
            digit_letter_prob=0.06,
            doubling_prob=0.025,
            dropping_prob=0.025,
            transposition_prob=0.018,
            smush_prob=0.030,
            split_prob=0.015,
            capitalization_prob=0.020,
            punctuation_drop_prob=0.20,
        )


def corrupt(text: str, rng: random.Random, config: CorruptionConfig | None = None) -> str:
    """Apply the full corruption pipeline to clean text. Returns corrupted text."""
    if config is None:
        config = CorruptionConfig()
    t = text
    t = apply_diacritic_strip(t, rng, config.diacritic_strip_prob)
    t = apply_pair_confusion(t, rng, config.pair_confusion_prob)
    t = apply_digit_letter_confusion(t, rng, config.digit_letter_prob)
    t = apply_doubling(t, rng, config.doubling_prob)
    t = apply_dropping(t, rng, config.dropping_prob)
    t = apply_transposition(t, rng, config.transposition_prob)
    t = apply_smush_split(t, rng, config.smush_prob, config.split_prob)
    t = apply_capitalization_slip(t, rng, config.capitalization_prob)
    t = apply_punctuation_drop(t, rng, config.punctuation_drop_prob)
    return t


def make_pair(
    clean: str,
    rng: random.Random,
    config: CorruptionConfig | None = None,
    ensure_corrupted: bool = True,
    max_retries: int = 3,
) -> tuple[str, str]:
    """Produce a (corrupted, clean) pair from a clean source string.

    `ensure_corrupted=True` retries up to `max_retries` if corruption produced
    an identical string (avoids degenerate identity examples in the corpus).
    """
    cfg = config or CorruptionConfig()
    for _ in range(max_retries):
        corrupted = corrupt(clean, rng, cfg)
        if not ensure_corrupted or corrupted != clean:
            return corrupted, clean
    # Last resort: force at least one mutation
    if clean and len(clean) > 2:
        idx = rng.randint(0, len(clean) - 2)
        chars = list(clean)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars), clean
    return clean, clean


# =====================================================================
# Mixed-difficulty sampler
# =====================================================================

def sample_difficulty(rng: random.Random) -> CorruptionConfig:
    """Return a corruption config sampled from a difficulty distribution.

    60% default (≈1-3% CER), 25% light (≈0.5% CER — clean cases the corrector
    must learn to leave alone), 15% aggressive (≈8% CER — pathological cases).
    """
    r = rng.random()
    if r < 0.25:
        return CorruptionConfig.light()
    if r < 0.85:
        return CorruptionConfig()
    return CorruptionConfig.aggressive()
