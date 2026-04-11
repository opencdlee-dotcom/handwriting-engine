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


def correct_domain_terms(text: str, domain: str = "biology") -> str:
    """Apply domain-specific spell correction to HTR output.

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

    words = text.split()
    corrected = []
    corrections_made = 0

    for word in words:
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
