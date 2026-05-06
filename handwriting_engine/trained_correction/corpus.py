"""Clean reference text builder.

Produces a stream of clean sentences / short paragraphs that mimic the
distribution of text the engine actually encounters: lab notebook entries,
science explanations, observational notes, methodology snippets.

Sources:
1. Domain vocabulary from handwriting_engine.postprocess (biology, chemistry, general)
2. Multi-word phrases from the same module
3. Lab notebook sentence templates (this file)
4. Optional: system word list for general English coverage

The corpus is generated, not curated, by design — the corrector learns to
preserve clean English given a generative process the user controls.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

from handwriting_engine.postprocess import (
    _BIOLOGY_TERMS,
    _CHEMISTRY_TERMS,
    _GENERAL_TERMS,
    _BIOLOGY_PHRASES,
    _CHEMISTRY_PHRASES,
)


# =====================================================================
# Lab notebook style templates
# =====================================================================

_OBSERVATION_TEMPLATES = [
    "The {term} was observed under the microscope.",
    "We measured the {term} at {value} {unit}.",
    "{term_cap} appears to {action} when exposed to {term2}.",
    "After {duration} minutes, the {term} began to {action}.",
    "The reaction produced a {color} {term}.",
    "Note: {term_cap} is {adjective} compared to {term2}.",
    "The {term} concentration was approximately {value} {unit}.",
    "Initial {term} reading: {value} {unit}.",
    "Final {term} reading: {value} {unit} — a difference of {delta}.",
    "{term_cap} was added in excess to drive the reaction.",
    "The control sample contained no {term}.",
    "We hypothesize that {term} influences the rate of {term2}.",
    "Results suggest a positive correlation between {term} and {term2}.",
    "The {term} sample showed {adjective} activity.",
    "{term_cap} acts as a catalyst in this reaction.",
    "We added {value} mL of {term} solution to the flask.",
    "The {term} membrane became permeable after heating.",
    "Cells in {term} phase showed visible {term2} structures.",
]

_METHODOLOGY_TEMPLATES = [
    "First, prepare a {value} {unit} solution of {term}.",
    "Add the {term} to the test tube and mix gently.",
    "Heat the {term} sample to {value} degrees Celsius.",
    "Filter the {term} mixture using filter paper.",
    "Centrifuge the {term} sample at {value} rpm for {duration} minutes.",
    "Stain the {term} with {term2} for visualization.",
    "Place the {term} on a glass slide and add a cover slip.",
    "Wash the {term} three times with distilled water.",
    "Incubate the {term} at room temperature for {duration} hours.",
    "Repeat the procedure with a fresh {term} sample.",
]

_REASONING_TEMPLATES = [
    "Therefore, the {term} must be present in higher concentrations.",
    "This indicates that {term} is responsible for the change.",
    "The data supports the hypothesis that {term} affects {term2}.",
    "A larger sample of {term} would reduce experimental error.",
    "However, the {term} reading varies significantly across trials.",
    "The {term} response was {adjective} in all three replicates.",
    "Because of this, we conclude that {term} drives the process.",
    "If the {term} concentration were higher, the reaction would proceed faster.",
    "Although {term} typically shows {action}, in this case it did not.",
    "Compare the {term} of group A with the {term} of group B.",
]

_PHRASE_TEMPLATES = [
    "{phrase_cap} is fundamental to understanding {term}.",
    "Students should learn about {phrase} before {term}.",
    "The role of {phrase} in {term} cannot be overstated.",
    "{phrase_cap} differs from {term} in several ways.",
    "We observed {phrase} during the experiment.",
    "{phrase_cap} requires {term} to function properly.",
    "Diagram showing {phrase} and the associated {term}.",
    "The {phrase} pathway involves multiple steps.",
    "Without {phrase}, {term} would not occur.",
    "{phrase_cap} produces {term} as a byproduct.",
]

_EQUATIONS_AND_VALUES = [
    "pH = {value}",
    "OD600 = {value_dec}",
    "T = {value} K",
    "n = {value} samples",
    "rate = {value_dec} M/s",
    "yield = {value}%",
    "Vmax = {value} units",
    "Km = {value_dec} mM",
    "lambda max = {value} nm",
]

_VALUES = ["1.5", "2.0", "3.7", "4.2", "5.0", "5.5", "6.8", "7.0", "7.4", "8.5", "10", "12", "15", "20", "25", "37", "60", "100", "250", "500"]
_VALUE_DECIMALS = ["0.05", "0.1", "0.25", "0.45", "0.5", "0.75", "1.0", "1.2", "1.5", "2.5"]
_UNITS = ["mL", "L", "g", "mg", "kg", "mol", "mmol", "M", "mM", "uM", "nm", "C", "K", "min", "hr"]
_DURATIONS = ["5", "10", "15", "30", "45", "60", "90", "120"]
_DELTAS = ["0.05", "0.1", "0.5", "1.0", "1.5", "2.5", "5", "10"]
_COLORS = ["clear", "yellow", "blue", "red", "green", "brown", "white", "purple", "pink", "colorless"]
_ACTIONS = ["dissolve", "precipitate", "react", "absorb", "expand", "contract", "denature", "polymerize", "crystallize", "evaporate"]
_ADJECTIVES = ["significant", "minimal", "rapid", "slow", "consistent", "variable", "stable", "unstable", "uniform", "non-uniform"]


def _all_single_terms() -> list[str]:
    return sorted(_BIOLOGY_TERMS | _CHEMISTRY_TERMS | _GENERAL_TERMS)


def _all_phrases() -> list[str]:
    return sorted(" ".join(p) for p in (_BIOLOGY_PHRASES | _CHEMISTRY_PHRASES))


def _capitalize(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _fill_template(tmpl: str, rng: random.Random, terms: list[str], phrases: list[str]) -> str:
    """Fill placeholders in a template with sampled vocabulary / values."""
    term = rng.choice(terms)
    term2 = rng.choice(terms)
    phrase = rng.choice(phrases) if phrases else term
    return tmpl.format(
        term=term,
        term_cap=_capitalize(term),
        term2=term2,
        phrase=phrase,
        phrase_cap=_capitalize(phrase),
        value=rng.choice(_VALUES),
        value_dec=rng.choice(_VALUE_DECIMALS),
        unit=rng.choice(_UNITS),
        duration=rng.choice(_DURATIONS),
        delta=rng.choice(_DELTAS),
        color=rng.choice(_COLORS),
        action=rng.choice(_ACTIONS),
        adjective=rng.choice(_ADJECTIVES),
    )


# =====================================================================
# General English coverage from system word list (optional)
# =====================================================================

def _load_system_wordlist(max_words: int = 5000) -> list[str]:
    """Load /usr/share/dict/words on macOS / common Linux. Returns [] if missing.

    We keep only short, common-looking words (length ≤ 12, all-lowercase).
    """
    candidates = [Path("/usr/share/dict/words"), Path("/usr/dict/words")]
    for p in candidates:
        if p.is_file():
            try:
                with p.open() as f:
                    words = [
                        w.strip() for w in f
                        if w.strip().isalpha()
                        and w.strip().islower()
                        and 3 <= len(w.strip()) <= 12
                    ]
                if len(words) > max_words:
                    return words[:max_words]
                return words
            except OSError:
                continue
    return []


_SIMPLE_SENTENCE_TEMPLATES = [
    "The {w1} was carefully placed near the {w2}.",
    "{w1_cap} affects how the {w2} behaves.",
    "{w1_cap} and {w2} are related but distinct.",
    "We compared the {w1} to the {w2}.",
    "The {w1} did not show any {w2} activity.",
    "Both samples contained {w1} and {w2}.",
    "Note the difference between {w1} and {w2}.",
    "The {w1} appeared shortly after the {w2}.",
]


# =====================================================================
# Public API
# =====================================================================

def generate_sentences(
    n: int,
    rng: random.Random,
    use_system_wordlist: bool = True,
) -> Iterator[str]:
    """Yield `n` clean sentences sampled from templates + vocabulary."""
    terms = _all_single_terms()
    phrases = _all_phrases()
    general_words = _load_system_wordlist() if use_system_wordlist else []

    all_templates: list[tuple[str, str]] = (
        [(t, "domain") for t in _OBSERVATION_TEMPLATES]
        + [(t, "domain") for t in _METHODOLOGY_TEMPLATES]
        + [(t, "domain") for t in _REASONING_TEMPLATES]
        + [(t, "phrase") for t in _PHRASE_TEMPLATES]
        + [(t, "equation") for t in _EQUATIONS_AND_VALUES]
    )
    # Only include general-English templates if we actually have a wordlist for them
    if general_words:
        all_templates += [(t, "general") for t in _SIMPLE_SENTENCE_TEMPLATES]

    for _ in range(n):
        tmpl, kind = rng.choice(all_templates)
        if kind == "general":
            w1 = rng.choice(general_words)
            w2 = rng.choice(general_words)
            sentence = tmpl.format(w1=w1, w1_cap=_capitalize(w1), w2=w2)
        else:
            sentence = _fill_template(tmpl, rng, terms, phrases)
        yield sentence


def generate_paragraphs(
    n: int,
    rng: random.Random,
    sentences_per_paragraph: tuple[int, int] = (1, 4),
    use_system_wordlist: bool = True,
) -> Iterator[str]:
    """Yield `n` short paragraphs (1-4 sentences each)."""
    sentence_iter = generate_sentences(
        n * sentences_per_paragraph[1],
        rng,
        use_system_wordlist=use_system_wordlist,
    )
    sentences = list(sentence_iter)
    cursor = 0
    for _ in range(n):
        k = rng.randint(*sentences_per_paragraph)
        chunk = sentences[cursor : cursor + k]
        cursor += k
        if not chunk:
            break
        yield " ".join(chunk)
