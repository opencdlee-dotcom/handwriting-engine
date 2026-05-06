"""
Domain-specific spell correction for handwriting transcription output.

Uses edit-distance-1 lookup against curated domain word lists. Only corrects
when there is exactly ONE candidate within edit distance 1 (prevents overcorrection).
Springer 2022: BERT achieves 71.4% HTR spell correction — but for constrained
biology/chemistry vocabulary, simple edit-distance lookup is faster and adequate.

Does NOT correct proper nouns, numbers, or abbreviations.
"""

from __future__ import annotations

import re
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Biology domain word list — common lab terms that OCR confuses
_BIOLOGY_TERMS = {
    "mitosis", "meiosis", "mitochondria", "chloroplast", "photosynthesis",
    "respiration", "glycolysis", "ribosome", "chromosome", "chromatid",
    "centromere", "telomere", "prophase", "metaphase", "anaphase", "telophase",
    "interphase", "cytokinesis", "deoxyribose", "nucleotide", "adenine",
    "thymine", "guanine", "cytosine", "uracil", "transcription", "translation",
    "replication", "polymerase", "helicase", "ligase", "primase", "topoisomerase",
    "nucleosome", "histone", "chromatin", "plasmid", "prokaryote", "eukaryote",
    "organelle", "cytoplasm", "nucleus", "vacuole", "lysosome", "peroxisome",
    "endoplasmic", "reticulum", "golgi", "apparatus", "membrane", "phospholipid",
    "bilayer", "osmosis", "diffusion", "concentration", "gradient", "enzyme",
    "substrate", "catalysis", "inhibitor", "allosteric", "denaturation",
    "protein", "peptide", "amino", "acid", "carbohydrate", "glucose", "fructose",
    "sucrose", "lactose", "starch", "cellulose", "lipid", "fatty", "glycerol",
    "absorption", "digestion", "circulation", "respiration", "excretion",
    "homeostasis", "population", "ecosystem", "biodiversity", "evolution",
    "natural", "selection", "mutation", "allele", "genotype", "phenotype",
    "dominant", "recessive", "heterozygous", "homozygous", "gamete", "zygote",
    "embryo", "placenta", "differentiation", "stem", "cell", "apoptosis",
    "receptor", "hormone", "neurotransmitter", "synapse", "axon", "dendrite",
    "antibody", "antigen", "lymphocyte", "phagocyte", "pathogen", "bacteria",
    "virus", "fungus", "protist", "kingdom", "species", "taxonomy", "phylogeny",
}

_CHEMISTRY_TERMS = {
    "hydrogen", "oxygen", "nitrogen", "carbon", "sulfur", "phosphorus",
    "sodium", "potassium", "calcium", "magnesium", "chlorine", "fluorine",
    "molecule", "compound", "element", "isotope", "electron", "proton",
    "neutron", "orbital", "valence", "covalent", "ionic", "metallic",
    "electronegativity", "oxidation", "reduction", "titration", "molarity",
    "molality", "solution", "solvent", "solute", "precipitation", "equilibrium",
    "catalyst", "activation", "enthalpy", "entropy", "gibbs", "exothermic",
    "endothermic", "reaction", "reactant", "product", "stoichiometry",
    "mole", "avogadro", "bohr", "quantum", "spectrum", "wavelength",
    "absorbance", "transmittance", "spectrophotometer", "chromatography",
    "electrophoresis", "centrifuge", "distillation", "filtration",
    "crystallization", "sublimation", "evaporation", "condensation",
}

_GENERAL_TERMS = {
    "approximately", "therefore", "however", "although", "because",
    "similar", "different", "compare", "contrast", "describe", "explain",
    "analyze", "evaluate", "calculate", "determine", "identify", "define",
    "significant", "negligible", "proportional", "inversely", "directly",
    "increase", "decrease", "constant", "variable", "hypothesis", "experiment",
    "observation", "conclusion", "evidence", "data", "result", "average",
    "standard", "deviation", "percentage", "measurement", "temperature",
    "pressure", "volume", "mass", "weight", "density", "velocity",
    "acceleration", "frequency", "amplitude", "wavelength", "intensity",
}

_DOMAIN_WORDLISTS = {
    "biology": _BIOLOGY_TERMS | _GENERAL_TERMS,
    "chemistry": _CHEMISTRY_TERMS | _GENERAL_TERMS,
    "general": _GENERAL_TERMS,
    "science": _BIOLOGY_TERMS | _CHEMISTRY_TERMS | _GENERAL_TERMS,
}

# Multi-word domain phrases. Word-by-word edit-distance lookup misses these:
# "natural selecton" → neither word is far from a single-word target, but the
# phrase is unambiguously "natural selection." We snap when both tokens of a
# bigram are close (ED ≤ 1) to a known phrase.
_BIOLOGY_PHRASES = {
    ("natural", "selection"),
    ("stem", "cell"),
    ("amino", "acid"),
    ("amino", "acids"),
    ("fatty", "acid"),
    ("fatty", "acids"),
    ("nucleic", "acid"),
    ("nucleic", "acids"),
    ("cell", "membrane"),
    ("cell", "wall"),
    ("cell", "cycle"),
    ("cell", "division"),
    ("active", "transport"),
    ("passive", "transport"),
    ("facilitated", "diffusion"),
    ("endoplasmic", "reticulum"),
    ("golgi", "apparatus"),
    ("krebs", "cycle"),
    ("citric", "acid"),
    ("electron", "transport"),
    ("light", "reactions"),
    ("dark", "reactions"),
    ("calvin", "cycle"),
    ("genetic", "code"),
    ("messenger", "rna"),
    ("transfer", "rna"),
    ("ribosomal", "rna"),
    ("double", "helix"),
    ("base", "pair"),
    ("base", "pairs"),
    ("punnett", "square"),
    ("phenotypic", "ratio"),
    ("genotypic", "ratio"),
    ("food", "chain"),
    ("food", "web"),
    ("trophic", "level"),
    ("biotic", "factor"),
    ("biotic", "factors"),
    ("abiotic", "factor"),
    ("abiotic", "factors"),
    ("carrying", "capacity"),
    ("limiting", "factor"),
    ("limiting", "factors"),
    ("white", "blood"),
    ("red", "blood"),
    ("blood", "cell"),
    ("blood", "cells"),
    ("immune", "system"),
    ("nervous", "system"),
    ("digestive", "system"),
    ("respiratory", "system"),
    ("circulatory", "system"),
}

_CHEMISTRY_PHRASES = {
    ("periodic", "table"),
    ("atomic", "number"),
    ("atomic", "mass"),
    ("atomic", "weight"),
    ("mass", "number"),
    ("ionic", "bond"),
    ("ionic", "bonds"),
    ("covalent", "bond"),
    ("covalent", "bonds"),
    ("hydrogen", "bond"),
    ("hydrogen", "bonds"),
    ("chemical", "equation"),
    ("balanced", "equation"),
    ("limiting", "reagent"),
    ("limiting", "reactant"),
    ("activation", "energy"),
    ("reaction", "rate"),
    ("equilibrium", "constant"),
    ("specific", "heat"),
    ("heat", "capacity"),
    ("boiling", "point"),
    ("melting", "point"),
    ("freezing", "point"),
    ("phase", "change"),
    ("phase", "transition"),
    ("ideal", "gas"),
    ("partial", "pressure"),
    ("vapor", "pressure"),
    ("electron", "configuration"),
    ("oxidation", "state"),
    ("oxidation", "number"),
    ("redox", "reaction"),
    ("acid", "base"),
    ("strong", "acid"),
    ("weak", "acid"),
    ("strong", "base"),
    ("weak", "base"),
}

_DOMAIN_PHRASELISTS = {
    "biology": _BIOLOGY_PHRASES,
    "chemistry": _CHEMISTRY_PHRASES,
    "general": set(),
    "science": _BIOLOGY_PHRASES | _CHEMISTRY_PHRASES,
}

# Pattern: skip numbers, abbreviations (all caps <= 4 chars), [?] markers
_SKIP_RE = re.compile(r'^\d|^\[|^[A-Z]{1,4}$|[0-9]')


def _edit_distance_1_candidates(word: str, wordlist: set[str]) -> list[str]:
    """Return all words in wordlist within edit distance 1 of word."""
    word_lower = word.lower()
    candidates = []
    w = word_lower

    # Check exact match first
    if w in wordlist:
        return []  # Already correct

    # Generate all strings at edit distance 1
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]

    edits = set()
    # Deletions
    edits.update(a + b[1:] for a, b in splits if b)
    # Transpositions
    edits.update(a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1)
    # Replacements
    edits.update(a + c + b[1:] for a, b in splits if b for c in letters)
    # Insertions
    edits.update(a + c + b for a, b in splits for c in letters)

    candidates = [e for e in edits if e in wordlist]
    return candidates


def _within_edit_distance_1(word: str, target: str) -> bool:
    """Return True iff edit distance(word, target) ≤ 1 (insert / delete / replace / transpose).

    Cheaper than building the full edit-1 set when we only need a yes/no
    against a fixed target.
    """
    if word == target:
        return True
    lw, lt = len(word), len(target)
    if abs(lw - lt) > 1:
        return False
    if lw == lt:
        diffs = [(a, b) for a, b in zip(word, target) if a != b]
        if len(diffs) == 1:
            return True
        # Single transposition: exactly two adjacent mismatched pairs that swap
        if len(diffs) == 2:
            i = next(idx for idx, (a, b) in enumerate(zip(word, target)) if a != b)
            if i + 1 < lw and word[i] == target[i + 1] and word[i + 1] == target[i]:
                return True
        return False
    # Length differs by 1 — one insertion or deletion
    short, long_ = (word, target) if lw < lt else (target, word)
    i = j = 0
    skipped = False
    while i < len(short) and j < len(long_):
        if short[i] != long_[j]:
            if skipped:
                return False
            skipped = True
            j += 1
        else:
            i += 1
            j += 1
    return True


def _correct_phrases(
    words: list[str],
    phrases: set[tuple[str, str]],
    wordlist: set[str],
) -> tuple[list[str], set[int]]:
    """Snap bigram (w_i, w_{i+1}) to a known phrase when both words are within ED1
    of exactly one phrase pair.

    Returns the corrected word list plus the set of indices that were rewritten —
    callers use that set to skip those tokens during single-word correction so
    we don't double-correct.

    Conservative: matches only when *exactly one* phrase pair fits, and at least
    one of the two tokens is genuinely wrong (not already in the wordlist). The
    second condition prevents rewriting valid bigrams that happen to look like
    phrases.
    """
    if not phrases:
        return words, set()

    out = list(words)
    rewritten: set[int] = set()

    for i in range(len(words) - 1):
        if i in rewritten or (i + 1) in rewritten:
            continue

        w1_raw, w2_raw = words[i], words[i + 1]
        w1_core = _strip_word(w1_raw)
        w2_core = _strip_word(w2_raw)
        if not w1_core or not w2_core:
            continue
        if len(w1_core) < 3 or len(w2_core) < 3:
            continue
        if _SKIP_RE.match(w1_core) or _SKIP_RE.match(w2_core):
            continue

        w1_lower = w1_core.lower()
        w2_lower = w2_core.lower()

        # At least one word must be "wrong" (not already in vocabulary) — else
        # the bigram is already plausible and we shouldn't touch it.
        if w1_lower in wordlist and w2_lower in wordlist:
            continue

        candidates = [
            (a, b) for a, b in phrases
            if _within_edit_distance_1(w1_lower, a) and _within_edit_distance_1(w2_lower, b)
        ]

        if len(candidates) != 1:
            continue
        a, b = candidates[0]
        # Skip no-op snaps (already canonical)
        if w1_lower == a and w2_lower == b:
            continue

        out[i] = _restore_capitalization(w1_raw, w1_core, a)
        out[i + 1] = _restore_capitalization(w2_raw, w2_core, b)
        rewritten.add(i)
        rewritten.add(i + 1)
        logger.debug("Phrase corrected ('%s', '%s') -> ('%s', '%s')", w1_raw, w2_raw, a, b)

    return out, rewritten


def _strip_word(word: str) -> str:
    """Extract alphabetic core of a token, dropping leading / trailing punctuation."""
    stripped = word.rstrip(".,;:!?")
    i = 0
    while i < len(stripped) and not stripped[i].isalpha():
        i += 1
    core = stripped[i:]
    j = len(core)
    while j > 0 and not core[j - 1].isalpha():
        j -= 1
    return core[:j]


def _restore_capitalization(original_token: str, original_core: str, replacement: str) -> str:
    """Rebuild a token: original prefix + replacement (matching case) + original suffix."""
    # Reconstruct prefix / suffix from original_token using original_core
    idx = original_token.find(original_core)
    prefix = original_token[:idx] if idx >= 0 else ""
    suffix = original_token[idx + len(original_core):] if idx >= 0 else ""
    if original_core.isupper():
        replacement = replacement.upper()
    elif original_core[:1].isupper():
        replacement = replacement.capitalize()
    return prefix + replacement + suffix


def correct_domain_terms(text: str, domain: str = "biology") -> str:
    """Apply domain-specific spell correction to HTR output.

    Two-pass: phrase-level (multi-word) snap first, then single-word edit-distance-1.
    Phrase pass catches multi-word terms ("natural selection", "amino acid") where
    word-level lookup misses or gets ambiguous; single-word pass handles the rest.

    Only corrects a word when:
    1. It is NOT in the domain wordlist (potential error)
    2. Exactly ONE candidate exists at edit distance 1 (unambiguous)
    3. The word doesn't look like a number, abbreviation, or [?] marker

    Args:
        text: Transcription text to correct.
        domain: One of 'biology', 'chemistry', 'general', 'science'.

    Returns:
        Corrected text (may be unchanged if no corrections made).
    """
    wordlist = _DOMAIN_WORDLISTS.get(domain, _GENERAL_TERMS)
    phrases = _DOMAIN_PHRASELISTS.get(domain, set())

    words = text.split()

    # Pass 1: phrase-level correction
    words, phrase_rewritten = _correct_phrases(words, phrases, wordlist)

    corrected = []
    corrections_made = 0

    for idx, word in enumerate(words):
        if idx in phrase_rewritten:
            corrected.append(word)
            continue
        # Strip punctuation for lookup but preserve it in output
        stripped = word.rstrip(".,;:!?")
        suffix = word[len(stripped):]
        prefix = ""

        # Handle leading punctuation
        i = 0
        while i < len(stripped) and not stripped[i].isalpha():
            prefix += stripped[i]
            i += 1
        core_raw = stripped[i:]

        # Also strip trailing non-alpha chars (closing brackets, parens, etc.)
        # that rstrip(".,;:!?") misses, appending them to suffix
        j = len(core_raw)
        while j > 0 and not core_raw[j - 1].isalpha():
            j -= 1
        core = core_raw[:j]
        suffix = core_raw[j:] + suffix

        # Skip short words, numbers, [?] markers, abbreviations
        if len(core) < 4 or _SKIP_RE.match(core):
            corrected.append(word)
            continue

        candidates = _edit_distance_1_candidates(core, wordlist)
        if len(candidates) == 1:
            # Guard: for short words (< 6 chars), only allow insertion-type corrections
            # (candidate is longer = a missing letter was found). Replacement corrections
            # on short words produce false positives: "bell"→"cell", "sell"→"cell", etc.
            if len(core) < 6 and len(candidates[0]) <= len(core):
                corrected.append(word)
                continue
            # Preserve original capitalization
            candidate = candidates[0]
            if core.isupper():
                candidate = candidate.upper()
            elif core[0].isupper():
                candidate = candidate.capitalize()
            corrected_word = prefix + candidate + suffix
            corrected.append(corrected_word)
            corrections_made += 1
            logger.debug("Corrected '%s' -> '%s'", word, corrected_word)
        else:
            corrected.append(word)

    if corrections_made:
        logger.info("Domain correction (%s): %d word(s) corrected", domain, corrections_made)

    return " ".join(corrected)
