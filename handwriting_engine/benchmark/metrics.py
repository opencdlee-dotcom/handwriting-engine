"""
Accuracy metrics for handwriting recognition evaluation.

Pure functions — no DB, no I/O, no side effects.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize text for fair CER/WER comparison.

    - Unicode NFC normalization (cafe\\u0301 → café)
    - Lowercase
    - Strip ALL engine uncertainty markers (from consensus.py and handwriting.py)
    - Collapse whitespace
    """
    # Unicode normalization first — compose decomposed characters
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    # Engine markers from consensus.py _UNCERTAINTY_RE
    text = re.sub(r"\[\?\]", "", text)                     # [?]
    text = re.sub(r"\[\?alt:\s*[^\]]*\]", "", text)        # [?alt: B/D]
    text = re.sub(r"\?\?\?", "", text)                     # ???
    text = re.sub(r"\[illegible[^\]]*\]", "", text)        # [illegible: ~3 chars]
    text = re.sub(r"\[unclear\]", "", text)                # [unclear]
    text = re.sub(r"unable to read", "", text)             # unable to read
    # Engine markers from handwriting.py / formats.py
    text = re.sub(r"\[diagram:\s*[^\]]*\]", "", text)      # [diagram: description]
    text = re.sub(r"\[graph:\s*[^\]]*\]", "", text)        # [graph: description]
    text = re.sub(r"\[margin:\s*[^\]]*\]", "", text)       # [margin: text]
    text = re.sub(r"\[inserted:\s*[^\]]*\]", "", text)     # [inserted: text]
    text = re.sub(r"\[empty\]", "", text)                  # [empty]
    text = re.sub(r"\[crossed out[^\]]*\]", "", text)      # [crossed out: ...]
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Optional jiwer integration for faster C++ CER/WER
try:
    import jiwer as _jiwer
    _HAS_JIWER = True
except ImportError:
    _HAS_JIWER = False


def levenshtein_distance(s, t) -> int:
    """Levenshtein edit distance between two sequences.

    Space-optimized two-row DP — O(min(m, n)) memory.
    Works on both strings and lists of strings (for WER).
    """
    if len(s) < len(t):
        s, t = t, s
    if not t:
        return len(s)

    prev = list(range(len(t) + 1))
    curr = [0] * (len(t) + 1)

    for i in range(1, len(s) + 1):
        curr[0] = i
        for j in range(1, len(t) + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev

    return prev[len(t)]


def character_error_rate(hypothesis: str, reference: str) -> tuple[float, int, int]:
    """Character Error Rate = edit_distance(hyp, ref) / len(ref).

    Both strings are normalized before comparison.
    Uses jiwer (C++ backend) if available, falls back to pure Python.

    Returns:
        (cer, char_edits, ref_chars). CER can exceed 1.0 if hypothesis
        is much longer than reference (hallucination).
    """
    hyp = normalize_text(hypothesis)
    ref = normalize_text(reference)
    if not ref:
        return (0.0 if not hyp else float("inf"), len(hyp), 0)

    if _HAS_JIWER:
        cer_val = _jiwer.cer(ref, hyp)
        edits = round(cer_val * len(ref))
        return (cer_val, edits, len(ref))

    edits = levenshtein_distance(hyp, ref)
    return (edits / len(ref), edits, len(ref))


def word_error_rate(hypothesis: str, reference: str) -> tuple[float, int, int]:
    """Word Error Rate = edit_distance(hyp_words, ref_words) / len(ref_words).

    Both strings are normalized before comparison.
    Uses jiwer (C++ backend) if available, falls back to pure Python.

    Returns:
        (wer, word_edits, ref_words). WER can exceed 1.0.
    """
    hyp = normalize_text(hypothesis)
    ref = normalize_text(reference)
    hyp_words = hyp.split()
    ref_words = ref.split()
    if not ref_words:
        return (0.0 if not hyp_words else float("inf"), len(hyp_words), 0)

    if _HAS_JIWER:
        wer_val = _jiwer.wer(ref, hyp)
        edits = round(wer_val * len(ref_words))
        return (wer_val, edits, len(ref_words))

    edits = levenshtein_distance(hyp_words, ref_words)
    return (edits / len(ref_words), edits, len(ref_words))


# --- Domain Term Accuracy ---

# Biology domain terms — single words
_BIOLOGY_TERMS_SINGLE = {
    "mitosis", "meiosis", "membrane", "atp", "dna", "rna", "enzyme", "protein",
    "glucose", "photosynthesis", "respiration", "ribosome", "chromosome", "gene",
    "allele", "phenotype", "genotype", "nucleus", "cytoplasm", "organelle",
    "mitochondria", "chloroplast", "golgi", "lysosome",
    "vacuole", "cellulose", "phospholipid", "osmosis", "diffusion", "homeostasis",
    "transcription", "translation", "codon", "anticodon", "plasmid", "vector",
    "pcr", "elisa", "electrophoresis", "centrifuge", "pipette", "incubation",
    "absorbance", "od600", "concentration", "dilution", "buffer", "agar",
    "colony", "streak", "inoculate", "autoclave", "sterile", "aseptic",
    "hypothesis", "control", "variable", "dependent", "independent", "replicate",
    "lysine", "leucine", "alanine", "glycine", "proline", "serine", "threonine",
    "catalase", "amylase", "lipase", "protease", "kinase", "synthase", "polymerase",
}

# Multi-word biology terms
_BIOLOGY_TERMS_MULTI = {
    "endoplasmic reticulum", "golgi apparatus", "cell membrane", "cell wall",
    "krebs cycle", "citric acid cycle", "electron transport chain",
    "atp synthase", "dna polymerase", "rna polymerase", "amino acid",
    "nucleic acid", "fatty acid", "stem cell", "base pair",
    "restriction enzyme", "gel electrophoresis", "serial dilution",
    "aseptic technique", "negative control", "positive control",
    "dependent variable", "independent variable", "standard deviation",
}


def domain_term_accuracy(
    hypothesis: str,
    reference: str,
    domain_terms: set[str] | None = None,
) -> tuple[float, int, int, list[str]]:
    """Compute accuracy on domain-specific terms only.

    Handles both single-word and multi-word terms. Multi-word terms
    (e.g. "endoplasmic reticulum") are checked via substring matching.

    Args:
        hypothesis: Model output text.
        reference: Ground truth text.
        domain_terms: Set of domain terms (single or multi-word). Default: biology terms.

    Returns:
        (accuracy, correct_count, total_terms, missed_terms).
        accuracy is 0.0-1.0, or -1 if no domain terms in reference.
    """
    if domain_terms is not None:
        single_terms = {t for t in domain_terms if " " not in t}
        multi_terms = {t for t in domain_terms if " " in t}
    else:
        single_terms = _BIOLOGY_TERMS_SINGLE
        multi_terms = _BIOLOGY_TERMS_MULTI

    hyp = normalize_text(hypothesis)
    ref = normalize_text(reference)

    found_terms: list[str] = []
    correct_terms: list[str] = []

    # Check multi-word terms first (substring match)
    for term in multi_terms:
        if term in ref:
            found_terms.append(term)
            if term in hyp:
                correct_terms.append(term)

    # Check single-word terms
    ref_words = set(ref.split())
    hyp_words = set(hyp.split())
    matched_single = ref_words & single_terms
    for term in matched_single:
        found_terms.append(term)
        if term in hyp_words:
            correct_terms.append(term)

    if not found_terms:
        return (-1.0, 0, 0, [])

    missed = sorted(set(found_terms) - set(correct_terms))
    accuracy = len(correct_terms) / len(found_terms)
    return (accuracy, len(correct_terms), len(found_terms), missed)


# --- Error Taxonomy ---

# Confusion pairs from handwriting_engine/handwriting.py
_CONFUSION_PAIRS = {
    ("1", "l"), ("1", "i"), ("0", "o"), ("5", "s"), ("2", "z"),
    ("6", "g"), ("8", "b"), ("9", "q"), ("7", "1"), ("4", "9"), ("3", "8"),
    ("r", "n"), ("c", "l"), ("u", "v"), ("a", "o"), ("n", "h"),
    ("e", "c"), ("b", "d"), ("p", "r"), ("h", "k"), ("m", "n"),
    ("g", "c"), ("e", "f"), ("p", "f"), ("t", "+"), ("x", "*"),
}
# Make bidirectional
_CONFUSION_SET = _CONFUSION_PAIRS | {(b, a) for a, b in _CONFUSION_PAIRS}


def classify_errors(
    hypothesis: str, reference: str,
) -> dict[str, int]:
    """Classify character-level errors into categories.

    Categories:
        confusion_pair: Maps to known handwriting confusion pairs (a/o, 1/l, etc.)
        substitution: Character replaced but not a known confusion pair
        insertion: Extra characters in hypothesis (hallucination)
        deletion: Missing characters from hypothesis (omission)

    Returns:
        Dict of {category: count}.
    """
    hyp = normalize_text(hypothesis)
    ref = normalize_text(reference)

    counts = {"confusion_pair": 0, "substitution": 0, "insertion": 0, "deletion": 0}

    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, ref, hyp)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "replace":
            ref_chars = ref[i1:i2]
            hyp_chars = hyp[j1:j2]
            for k in range(min(len(ref_chars), len(hyp_chars))):
                pair = (ref_chars[k], hyp_chars[k])
                if pair in _CONFUSION_SET:
                    counts["confusion_pair"] += 1
                else:
                    counts["substitution"] += 1
            diff = len(hyp_chars) - len(ref_chars)
            if diff > 0:
                counts["insertion"] += diff
            elif diff < 0:
                counts["deletion"] += abs(diff)
        elif tag == "insert":
            counts["insertion"] += j2 - j1
        elif tag == "delete":
            counts["deletion"] += i2 - i1

    return counts
