<plan phase="5" index="01" requirement="REQ-008,REQ-009">
  <objective>Add domain spell correction postprocessor and extend benchmark CLI with --compare-strategies and regression alerting</objective>

  <files>
    <create>handwriting_engine/postprocess.py</create>
    <modify>handwriting_engine/vision.py</modify>
    <modify>handwriting_engine/benchmark/evaluate.py</modify>
    <create>tests/test_postprocess.py</create>
  </files>

  <tasks>
    <task type="auto">
      <name>Create handwriting_engine/postprocess.py with domain spell correction</name>
      <action>
Create /Users/user/Documents/VSCode Projects/handwriting-engine/handwriting_engine/postprocess.py:

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
        core = stripped[i:]

        # Skip short words, numbers, [?] markers, abbreviations
        if len(core) < 4 or _SKIP_RE.match(core):
            corrected.append(word)
            continue

        candidates = _edit_distance_1_candidates(core, wordlist)
        if len(candidates) == 1:
            # Preserve original capitalization
            candidate = candidates[0]
            if core[0].isupper():
                candidate = candidate.capitalize()
            corrected_word = prefix + candidate + suffix
            corrected.append(corrected_word)
            corrections_made += 1
            logger.debug("Corrected '%s' → '%s'", word, corrected_word)
        else:
            corrected.append(word)

    if corrections_made:
        logger.info("Domain correction (%s): %d word(s) corrected", domain, corrections_made)

    return " ".join(corrected)
      </action>
      <verify>python3 -c "from handwriting_engine.postprocess import correct_domain_terms; r = correct_domain_terms('mitocondria is an organele', 'biology'); print(r)"</verify>
      <done>correct_domain_terms importable; corrects 'mitocondria' → 'mitochondria' and 'organele' → 'organelle'</done>
    </task>

    <task type="auto">
      <name>Wire domain correction into vision.py _postprocess_output()</name>
      <action>
In /Users/user/Documents/VSCode Projects/handwriting-engine/handwriting_engine/vision.py:

1. Find the _postprocess_output() function and add an optional domain parameter:
   def _postprocess_output(text: str, domain: str | None = None) -> str:

2. At the end of _postprocess_output(), before `return result.strip()`, add:
    # Domain spell correction (optional — only when domain is specified)
    if domain:
        try:
            from handwriting_engine.postprocess import correct_domain_terms
            result = correct_domain_terms(result, domain)
        except Exception as e:
            logger.warning("Domain correction failed: %s", e)

3. Find the read_page() function and add `domain: str | None = None` parameter.
   Pass it through to _postprocess_output().
      </action>
      <verify>python3 -c "import inspect; from handwriting_engine.vision import read_page; print('domain' in str(inspect.signature(read_page)))"</verify>
      <done>read_page() accepts domain parameter; _postprocess_output() applies correction when domain given</done>
    </task>

    <task type="auto">
      <name>Create tests/test_postprocess.py</name>
      <action>
Create /Users/user/Documents/VSCode Projects/handwriting-engine/tests/test_postprocess.py (or append if it exists):

"""Tests for domain spell correction."""
from handwriting_engine.postprocess import correct_domain_terms, _edit_distance_1_candidates, _BIOLOGY_TERMS


def test_corrects_mitochondria_typo():
    result = correct_domain_terms("the mitocondria is important", "biology")
    assert "mitochondria" in result


def test_no_correction_when_ambiguous():
    # "cat" has many edit-distance-1 candidates — should not be corrected
    result = correct_domain_terms("the cat sat", "biology")
    assert result == "the cat sat"


def test_preserves_numbers():
    result = correct_domain_terms("pH is 7.4 and temp is 37C", "biology")
    assert "7.4" in result
    assert "37C" in result


def test_preserves_abbreviations():
    result = correct_domain_terms("ATP ADP NAD", "biology")
    assert "ATP" in result
    assert "ADP" in result


def test_preserves_question_markers():
    result = correct_domain_terms("the [?] is [illegible]", "biology")
    assert "[?]" in result
    assert "[illegible]" in result


def test_correct_domain_terms_exact_match_no_change():
    result = correct_domain_terms("mitosis is a type of cell division", "biology")
    assert "mitosis" in result


def test_edit_distance_1_exact_match_returns_empty():
    candidates = _edit_distance_1_candidates("mitosis", _BIOLOGY_TERMS)
    assert candidates == []  # Already correct — no candidates needed


def test_unknown_domain_falls_back_to_general():
    # Should not raise even for unknown domain
    result = correct_domain_terms("the experiment was successful", "unknown_domain")
    assert isinstance(result, str)


def test_empty_string():
    assert correct_domain_terms("", "biology") == ""
      </action>
      <verify>python3 -m pytest tests/test_postprocess.py -q 2>&1 | tail -3</verify>
      <done>All postprocess tests pass</done>
    </task>

    <task type="auto">
      <name>Add --compare-strategies flag to benchmark CLI and regression alerting</name>
      <action>
In /Users/user/Documents/VSCode Projects/handwriting-engine/handwriting_engine/benchmark/evaluate.py:

Read the file first to understand the existing CLI structure. Then:

1. Find the `benchmark run` command (likely a click command). Add a `--compare-strategies` option:
   @click.option("--compare-strategies", default=None, help="Comma-separated strategies to compare, e.g. vote,best_of,self_correct")

2. When --compare-strategies is provided:
   - Parse the comma-separated list
   - Run the benchmark for each strategy
   - Output a comparison table: strategy | CER | WER | samples
   - Example output:
     Strategy Comparison:
     strategy      | CER    | WER
     --------------|--------|-------
     best_of       | 1.67%  | 3.21%
     self_correct  | 1.34%  | 2.87%

3. Add regression alerting: after each benchmark run, compare CER to the previous run stored in the DB. If CER increased by > 0.5 percentage points, print:
   WARNING: CER regression detected: {prev:.2f}% → {current:.2f}% (>{threshold:.1f}% increase)

If the evaluate.py file is complex, only add the --compare-strategies option and a simple comparison loop. Do not refactor existing code.

Note: If evaluate.py doesn't have a straightforward click command structure, add the comparison as a new subcommand: `benchmark compare-strategies --strategies vote,best_of,self_correct`
      </action>
      <verify>python3 -m pytest tests/test_benchmark_evaluate.py -q 2>&1 | tail -3</verify>
      <done>Existing benchmark evaluate tests still pass after changes</done>
    </task>

    <task type="auto">
      <name>Run full test suite and commit Phase 5</name>
      <action>
Run: python3 -m pytest tests/ -q --ignore=tests/test_optimize.py 2>&1 | tail -8
(test_optimize.py has a pre-existing failure unrelated to our changes)

Fix any failures in new test files. Then commit:
git add handwriting_engine/postprocess.py handwriting_engine/vision.py handwriting_engine/benchmark/evaluate.py tests/test_postprocess.py
git commit -m "feat(phase-5): domain spell correction + benchmark compare-strategies (REQ-008, REQ-009)"
      </action>
      <verify>python3 -m pytest tests/ -q --ignore=tests/test_optimize.py 2>&1 | tail -4</verify>
      <done>Full test suite passes (excluding pre-existing test_optimize failure)</done>
    </task>
  </tasks>

  <dependencies>none</dependencies>
  <commit_message>feat(phase-5): domain spell correction + benchmark compare-strategies (REQ-008, REQ-009)</commit_message>
</plan>
