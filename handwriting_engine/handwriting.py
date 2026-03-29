"""
Handwriting reading strategies for vision-model prompts.

Merged from exam-grader's grader/handwriting.py and the handwriting-reader
skill's reading-strategies.  Provides string constants, domain rules, and
helper functions that callers inject into transcription / grading prompts.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Core reading protocol -- injected into read_page() prompts
# ---------------------------------------------------------------------------

TRANSCRIPTION_FIDELITY_RULE = """
=== CRITICAL TRANSCRIPTION RULE — READ THIS FIRST ===
You are a TRANSCRIPTION engine, not a proofreader. Your job is to reproduce
EXACTLY what is written, including all spelling errors, grammar mistakes,
unusual capitalization, and abbreviations. DO NOT:
- Fix spelling (if they wrote "mitocondria", output "mitocondria")
- Normalize capitalization (if they wrote "dna", output "dna", not "DNA")
- Add missing punctuation
- Expand abbreviations (w/ stays w/, b/c stays b/c)
- "Clean up" formatting or whitespace
The ONLY modification allowed is resolving genuinely ambiguous characters
using the disambiguation rules below.
""".strip()

WRITER_CALIBRATION = """
=== WRITER CALIBRATION ===
Before reading content, study this writer's handwriting style:
- How do they form their 7s? (crossed or uncrossed?)
- How do they write 4? (open top or closed?)
- Is their 'a' open-topped (looks like 'u') or closed?
- Do they connect letters (cursive) or print separately?
- How do they distinguish 0 from O? 1 from l?
- What is their typical letter size and slant?
Apply these observations CONSISTENTLY across ALL text in this image.
If you resolve a character ambiguity on one line, apply the same resolution
everywhere for this writer.
""".strip()

MULTI_PASS_READING_INSTRUCTIONS = """
=== HANDWRITING READING PROTOCOL (MULTI-PASS) ===

PASS 1 -- STRUCTURE: Before reading any text, map the spatial layout.
- Identify tables, columns, section headers, diagrams, graphs
- Note non-text elements: [diagram: description], [graph: description]
- Identify crossed-out regions (grade FINAL version only)
- Note margin annotations: [margin: text]
- Note insertions with carets or arrows: [inserted: text]
- Detect page orientation -- rotate mentally if sideways or inverted
- Identify color changes (pen switches) that may indicate edits or emphasis

PASS 2 -- CONTENT: Read text within each structural element.
- Read numbers DIGIT BY DIGIT -- never guess a whole number at once
- For tables: read column headers first, then row by row left to right
- Flag anything ambiguous with [?] markers (e.g., "1[?]3.5 mg")
- Overwritten characters: read the final version, note as [?overwritten, was X]
- Faint/smudged text: attempt to read, flag with [?]
- Completely illegible: [illegible: ~N characters]
- Preserve abbreviations as-written (w/, b/c, approx., temp., conc.)
- Preserve original spelling errors -- do NOT silently correct
- For bullet / numbered lists, maintain hierarchy and numbering

PASS 3 -- CROSS-VALIDATION: Review all [?] markers using context.
- Apply character disambiguation rules (see below)
- Check internal consistency: same word appearing multiple times should resolve the same way
- If 2+ instances of a word agree and 1 differs, resolve the outlier to the majority
- Validate word boundaries: if a reading doesn't produce a real word, try confusion pair swaps
- Cross-check numbers against any totals, averages, or expected ranges on the page
""".strip()

# ---------------------------------------------------------------------------
# Number reading rules -- critical for data tables and calculations
# ---------------------------------------------------------------------------

NUMBER_READING_RULES = """
=== NUMBER READING RULES ===
1. Read each digit individually -- do not guess the whole number
2. Cross-reference with neighbors: if a column has 0.45, 0.52, 0.48, then 4.51 is likely 0.51
3. Check decimal alignment: numbers in a column usually share the same decimal places
4. Verify magnitude against expected ranges:
   - pH: 0-14, typically 1 decimal place
   - Temperature: usually degrees C, common values: 4, 25, 37, 42, 65, 72, 95
   - OD/Absorbance: 0-4.0, usually 2-3 decimal places
   - Concentrations: watch for micro (u) vs milli (m) prefix -- verify magnitude
   - Volumes: uL (1-1000), mL (0.1-1000), L (0.1-10)
   - Masses: mg (0.1-999), g (0.1-999), kg (0.1-100)
   - Time: seconds (0-60), minutes (0-60), hours (0-72)
5. Negative signs vs dashes: context determines if '-' is minus or separator
6. Superscripts in scientific notation: 10^3, 10^-4 -- verify exponent sign
7. Comma vs period: in US convention, comma is thousands separator, period is decimal
8. Fraction bars vs division signs: check surrounding context
""".strip()

# ---------------------------------------------------------------------------
# Character disambiguation -- 42+ confusion pairs
# ---------------------------------------------------------------------------

CHARACTER_DISAMBIGUATION = """
=== CHARACTER DISAMBIGUATION ===
When a character is ambiguous, use context to resolve:

| #  | Confused Pair     | Resolution Rule |
|----|-------------------|-----------------|
| 1  | 1 / l / I         | In numbers/data -> 1. In words -> l or I. Start of sentence -> I |
| 2  | 0 / O             | In numbers/measurements -> 0. In words -> O |
| 3  | 5 / S             | In numbers -> 5. In words -> S. Check if chemical element symbol |
| 4  | 2 / Z             | In numbers -> 2. In words -> Z |
| 5  | 6 / G             | In numbers -> 6. In words -> G |
| 6  | 8 / B             | In numbers -> 8. In words -> B |
| 7  | rn / m            | Read surrounding word -- does 'rn' or 'm' make a valid word? |
| 8  | cl / d            | Read surrounding word -- does 'cl' or 'd' make a valid word? |
| 9  | u / v             | Rounded bottom = u, pointed = v |
| 10 | a / o             | Has tail = a, fully closed = o |
| 11 | n / h             | Tall ascender = h, short = n |
| 12 | e / c             | Loop closed = e, open = c |
| 13 | t / +             | In text -> t. In math/data -> + |
| 14 | x / multiply      | In text/variables -> x. Between numbers -> multiply sign |
| 15 | H / K             | H has two verticals + horizontal bar. K has one vertical + diagonal strokes. Check stroke count |
| 16 | B / N / R         | B has bumps on right side. N has diagonal between verticals. R has bump + leg. Check if closed loops present |
| 17 | P / R             | P has closed loop, no leg. R has loop + diagonal leg. Check bottom-right stroke |
| 18 | P / F             | P has closed loop at top. F has open horizontal strokes, no loop. Check if top-right curves back |
| 19 | M / N             | M has two humps or pointed top. N has one diagonal. Count vertical strokes |
| 20 | D / O             | D has flat left side (vertical stroke). O is fully rounded. Check left side |
| 21 | B / D             | B has TWO bumps/loops stacked on right side. D has ONE smooth curve. Count loops: 2=B, 1=D |
| 22 | O / Q             | O is fully closed oval. Q has a tail or slash extending from bottom-right. Check for tail stroke |
| 23 | G / C             | G has an inward horizontal bar at mid-right. C is open (no bar). Check if right side closes inward |
| 24 | H / A             | H has two full-height verticals + horizontal bar at middle. A has two diagonals meeting at top + lower bar. Check if top is pointed (A) or two parallel verticals (H) |
| 25 | L / I             | L has a vertical + horizontal foot at bottom. I is a single vertical stroke (may have serifs top and bottom). Check for bottom-right horizontal extension |
| 26 | J / I             | J curves left at the bottom. I is straight. Check bottom stroke direction |
| 27 | E / F             | E has three horizontal strokes (top, middle, bottom). F has two (top, middle, no bottom). Check for bottom horizontal |
| 28 | K / R             | K has diagonal strokes from mid-vertical. R has a loop at top-right + diagonal leg. Check for loop |
| 29 | U / V             | U has a rounded bottom. V has a sharp pointed bottom. Check curve vs angle |
| 30 | W / M             | W points down at bottom. M points up at top. Check orientation of peaks/valleys |
| 31 | q / g             | q has straight descender. g has looped descender (in many styles). Check descender shape |
| 32 | b / d (lower)     | b has ascender on left + bump on right. d has bump on left + ascender on right. Mirror pair |
| 33 | p / q (lower)     | p has descender on left + bump on right. q has bump on left + descender on right. Mirror pair |
| 34 | f / t (lower)     | f has a descender and top curve. t is shorter with a horizontal cross. Check height and descender |
| 35 | i / j (lower)     | i has a dot and no descender. j has a dot and curves left at bottom |
| 36 | w / uu            | w is a single character. uu is two separate u strokes. Check spacing between humps |
| 37 | ri / n            | ri has a dot above. n does not. If no dot visible, check stroke connection |
| 38 | 9 / q             | In numbers -> 9. In words -> q. Check surrounding context |
| 39 | 7 / 1             | 7 has a horizontal top stroke. 1 is a single vertical (sometimes with serif). Check for crossed 7 |
| 40 | 4 / 9             | 4 has an open top with angular strokes. 9 has a closed loop at top. Check if top is open or closed |
| 41 | 3 / 8             | 3 is open on the left. 8 is fully closed. Check if left side connects |
| 42 | S / 5 / $         | S is smooth curves. 5 has angular top. $ has vertical line through S. Check for straight edges and vertical bar |
| 43 | C / (             | C is a letter-sized curve. ( is taller and narrower. Check height relative to surrounding text |
| 44 | . / ,             | Period sits on baseline. Comma has a descending tail. Check for downward stroke |

WORD-BOUNDARY VALIDATION: After reading each word, check if it produces a
real English or scientific term. If not, identify characters from confusion
pairs above and try swapping the most likely candidate.

MULTI-PAGE CONSISTENCY: Track the writer's handwriting style across pages.
If you resolve a character on page 1, apply the same resolution consistently.
Note: do they cross their 7s? How do they form their 4s? Open or closed?
""".strip()

# ---------------------------------------------------------------------------
# Scientific notation & symbols
# ---------------------------------------------------------------------------

SCIENTIFIC_NOTATION = """
=== SCIENTIFIC NOTATION & SYMBOLS ===
| Pattern                          | Interpretation |
|----------------------------------|---------------|
| Small raised character           | Superscript: x squared, Ca2+, 10^3 |
| Small lowered character          | Subscript: H2O, CO2, O2 |
| Single Greek-like character      | Greek letter: alpha, beta, gamma, delta, mu, lambda, sigma, pi |
| Bar over character               | Mean/average |
| Arrow over character             | Vector |
| Subscript/superscript combos     | H2O may appear as H2O or H20 -- all equivalent |
| Arrow between terms              | Reaction, conversion, or process direction |
| Double arrow                     | Equilibrium or reversible reaction |
| Wavy line                        | Approximately equal |
| Plus/minus stacked               | Plus-or-minus (uncertainty) |
""".strip()

# ---------------------------------------------------------------------------
# Table detection
# ---------------------------------------------------------------------------

TABLE_DETECTION = """
=== TABLE DETECTION IN HANDWRITING ===
Signs of a handwritten table:
- Vertical/horizontal lines (even partial or wobbly)
- Consistent column spacing across rows
- Header row with different styling (underlined, boxed, bolder)
- Repeated measurement patterns (number, unit, number, unit)
- Grid-like alignment of content

When a table is detected:
1. Identify column headers first
2. Read row by row, left to right
3. Verify column alignment -- characters should line up vertically
4. Apply number reading rules within each column
5. Cross-reference values against column context for disambiguation
6. If a cell appears empty, mark as [empty] -- do not fabricate values
7. Check for merged cells or spanning headers
""".strip()

# ---------------------------------------------------------------------------
# Single-letter answer protocol (matching / fill-in questions)
# ---------------------------------------------------------------------------

SINGLE_LETTER_PROTOCOL = """
=== SINGLE-LETTER ANSWER PROTOCOL (FILL-IN / MATCHING) ===
When the expected answer is a SINGLE CAPITAL LETTER (matching questions,
fill-in-the-blank with letter choices), apply EXTRA scrutiny:

DUAL-READ PROCEDURE:
1. First read: identify the letter based on raw stroke structure alone
2. Second read: identify the letter considering answer-key context
3. Compare: if both reads agree, confidence = HIGH
4. If they disagree, report BOTH and set confidence = LOW

RULES:
1. NEVER guess a single letter -- examine stroke structure methodically
2. For each letter, report your TWO most likely readings: "Primary: B, Alt: D"
3. Check the student's OTHER single-letter answers on the same page to calibrate:
   - How does THIS student form B vs D? H vs K? G vs C?
   - Use consistent interpretation across all their answers
4. If the primary reading does NOT match the answer key but the alternate
   reading DOES, award the point (benefit of the doubt) BUT mark confidence
   as LOW
5. HIGH-RISK PAIRS to watch for specifically:
   B/D, O/Q, G/C, M/N, H/A, L/I, P/F, K/R, E/F, J/I
""".strip()

# ---------------------------------------------------------------------------
# Faint content protocol -- injected when quality assessment detects faint ink
# ---------------------------------------------------------------------------

FAINT_CONTENT_PROTOCOL = """
=== FAINT CONTENT PROTOCOL ===
This image has been flagged as containing faint or washed-out handwriting.
Apply these rules:

1. ATTEMPT TO READ: Even faint text may be partially legible. Try harder than
   usual to resolve characters — adjust your visual sensitivity.

2. PARTIAL READS: If you can read some words but not others:
   - Report all readable content normally
   - Mark unreadable sections with [faint: ~N chars] (not [illegible])
   - Never return an empty string — always report what IS visible

3. MULTI-PART ANSWERS: If parts of an answer are clear and parts are faint:
   - Report the clear parts fully
   - Flag faint parts separately: [faint: partial text visible: "..."]

4. COMPLETELY UNREADABLE: Only if the ENTIRE page is too faint to read anything:
   - Return: "[PAGE FAINT — requires human review]"
   - Do NOT return empty string or generic "unable to read"

5. CONFIDENCE: All readings from faint content should be treated as LOW confidence.
   Flag with [?] more liberally than usual when reading faint text.

6. DO NOT OVER-COMPENSATE: Do not guess or fabricate text that isn't visible.
   Better to flag as [faint] than to hallucinate content.
""".strip()

# ---------------------------------------------------------------------------
# Biology domain rules
# ---------------------------------------------------------------------------

BIOLOGY_DOMAIN_RULES = """
=== BIOLOGY DOMAIN RULES ===
When grading biology content, apply these additional disambiguation rules:

TERM BANK (prefer these when disambiguation is ambiguous):
- Core: mitosis, meiosis, membrane, metabolism, enzyme, chromosome, ribosome,
  lysosome, cytoplasm, nucleotide, phospholipid, polypeptide, homeostasis,
  endoplasmic reticulum, mitochondria, photosynthesis, transcription,
  translation, replication, fermentation, centrifugation, electrophoresis,
  spectrophotometry, chromatography, denaturation, annealing, elongation
- Bases: adenine, thymine, guanine, cytosine, uracil
- Amino acids: alanine, arginine, asparagine, aspartate, cysteine, glutamate,
  glutamine, glycine, histidine, isoleucine, leucine, lysine, methionine,
  phenylalanine, proline, serine, threonine, tryptophan, tyrosine, valine
- Acronyms: ATP, ADP, DNA, RNA, mRNA, tRNA, rRNA, NADH, NADPH, FADH2, PCR,
  CRISPR, SDS-PAGE, ELISA, BLAST, OD600, CFU, MOI, GOI, MCS, ori, amp, kan

PREFIX CLUES (helps identify garbled words):
- bio- = life (biology, biosynthesis, biofilm)
- cyto- = cell (cytoplasm, cytoskeleton, cytokine)
- endo- = within (endoplasmic, endocytosis, endospore)
- exo- = outside (exocytosis, exonuclease, exothermic)
- hyper- = above/excess (hypertonic, hyperthermia)
- hypo- = below/under (hypotonic, hypothermia)
- inter- = between (interphase, intergenic)
- intra- = within (intracellular, intronic)
- micro- = small (microscope, microliter, microcentrifuge)
- mono- = one (monomer, monosaccharide, monoclonal)
- poly- = many (polymer, polysaccharide, polymerase)
- proto- = first (prokaryote, protoplast, proto-oncogene)

SUFFIX CLUES (helps identify garbled words):
- -ase = enzyme (lipase, protease, polymerase, helicase, ligase)
- -ose = sugar (glucose, fructose, sucrose, lactose, maltose)
- -ine = amino acid/base (adenine, lysine, glycine, leucine)
- -sis = process (mitosis, osmosis, apoptosis, lysis, stasis)
- -lysis = breakdown (hydrolysis, glycolysis, hemolysis, proteolysis)
- -genesis = origin/creation (biogenesis, mutagenesis, oogenesis)
- -phage = eating (bacteriophage, macrophage)
- -philic / -phobic = loving / fearing (hydrophilic, hydrophobic)
- -plasm = formed material (cytoplasm, protoplasm, neoplasm)
- -some = body (ribosome, lysosome, chromosome, centrosome)

CHARACTER PREFERENCES IN BIO CONTEXT:
- 'rn' -> 'm' in: enzyme, membrane, chromosome, fermentation, ribosome
- 'cl' -> 'd' in: acid, lipid, nucleotide
- prefer 'u' in: nucleus, uracil, glucose, tubule
- 'l' vs '1': '1' before units (1 uL), 'l' in words (cell)

MEASUREMENT RANGES (flag readings outside these):
- OD600: 0-2.0 (typical growth curves), up to 4.0 if spectrophotometer allows
- pH: 0-14
- Absorbance: 0-4.0
- Temperature: usually Celsius (common: 4, 25, 37, 42, 55, 65, 72, 95)
- Volumes: uL (1-1000), mL (0.1-50), L (rare in lab context)
- Common units: uL, mL, nM, uM, mM, M, kDa, bp, kb, Mb, rpm, xg, CFU/mL
""".strip()

# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

CHEMISTRY_DOMAIN_RULES = """
=== CHEMISTRY DOMAIN RULES ===

ELEMENT SYMBOL DISAMBIGUATION:
- Single capital letter = element (C, N, O, H, S, P, K, I, B, F, U, V, W, Y)
- Capital + lowercase = element (Ca, Fe, Na, Mg, Cl, Br, Mn, Cu, Zn, Al, Si, Ni, Co, Cr, Li, Be)
- 'Cl' vs 'CI': Cl is chlorine (very common), CI is unlikely in chemistry context
- 'Co' vs 'CO': Co = cobalt (element), CO = carbon monoxide (compound) — check context
- 'NO' vs 'No': NO = nitric oxide, No = nobelium (rare) — default to NO
- 'l' vs '1' in formulas: HCl not HC1, NaCl not NaC1

REACTION NOTATION:
- Arrow (→) = reaction direction, yield
- Double arrow (⇌) = equilibrium
- Delta (Δ) above arrow = heat required
- Catalyst written above or below arrow
- State symbols: (s) solid, (l) liquid, (g) gas, (aq) aqueous

COMMON UNITS AND ABBREVIATIONS:
- Concentration: M (molar), mM, μM, nM, mol/L
- Mass: g, mg, μg, ng, kg, amu/Da
- Volume: L, mL, μL
- Pressure: atm, kPa, mmHg, torr
- Energy: J, kJ, cal, kcal, eV
- Temperature: °C, K (never °K)

IUPAC NOMENCLATURE CLUES:
- -ane = single bonds (methane, ethane, propane, butane)
- -ene = double bond (ethene, propene)
- -yne = triple bond (ethyne, propyne)
- -ol = alcohol (methanol, ethanol)
- -al = aldehyde (methanal, ethanal)
- -one = ketone (propanone, butanone)
- -oic acid = carboxylic acid (ethanoic, propanoic)
- -amine = amine group
- -ester = ester linkage

GREEK LETTERS IN CHEMISTRY:
- α (alpha): alpha particle, alpha carbon, alpha helix
- β (beta): beta particle, beta sheet, beta decay
- γ (gamma): gamma radiation
- δ (delta): partial charge (δ+, δ-)
- λ (lambda): wavelength
- μ (mu): micro- prefix, chemical potential
- σ (sigma): sigma bond
- π (pi): pi bond
""".strip()

MATH_DOMAIN_RULES = """
=== MATHEMATICS DOMAIN RULES ===

VARIABLE vs NUMBER DISAMBIGUATION:
- 'x' in equations = variable, NOT multiply sign (use · or × for multiplication)
- 'l' (ell) = often a variable or length — distinguish from '1' by context
- 'O' (capital O) = rarely used; prefer '0' (zero) in numeric context
- 'I' = identity matrix or imaginary unit — distinguish from '1' by context
- 'e' = Euler's number (2.718...) when standalone in exponentials
- 'i' = imaginary unit when in complex number context (a + bi)
- 'n' = common variable for integers, 'u' = common for sequences

OPERATOR DISAMBIGUATION:
- '−' (minus/dash) vs '–' (en dash) vs '—' (em dash): in math, always read as minus
- '·' (dot) vs '.' (period): centered dot = multiplication, baseline = decimal
- '/' vs '÷': both = division, but '/' also used in fractions
- '|' (vertical bar): absolute value |x|, set builder {x | ...}, divisibility
- '∫' vs 'f': integral sign has specific shape (elongated S), 'f' is a letter

COMMON SYMBOL CONFUSIONS:
- Σ (sigma) vs S: Σ = summation, S = variable or set
- Δ (delta) vs triangle: Δx = change in x, △ = geometric triangle
- ∈ (epsilon variant) vs E: ∈ = set membership, E = variable
- ∅ (empty set) vs 0 vs O vs φ (phi): context determines
- ∞ (infinity) vs 8: ∞ is horizontal, 8 is vertical
- √ (square root) vs 'v': √ has horizontal bar extending right
- ≤ and ≥ vs < and >: check for underline stroke
- ≠ vs =: check for slash through equals sign
- ≈ vs =: check for wavy vs straight lines

EQUATION FORMATTING:
- Fractions: numerator above line, denominator below
- Exponents/superscripts: smaller, raised characters (x², aⁿ)
- Subscripts: smaller, lowered characters (xₙ, a₁)
- Matrices: rectangular arrays in brackets [ ] or parentheses ( )
- Vectors: arrow above (v⃗) or bold (v)
""".strip()

PHYSICS_DOMAIN_RULES = """
=== PHYSICS DOMAIN RULES ===

UNIT DISAMBIGUATION:
- 'm' = meters (distance) vs 'M' = mega- prefix or molar
- 'g' = grams vs 'G' = gravitational constant (6.674×10⁻¹¹) vs g = acceleration (9.8 m/s²)
- 'N' = Newtons (force) vs 'n' = variable (often count or index)
- 'W' = Watts (power) vs 'w' = angular velocity or width
- 'J' = Joules (energy) vs 'j' = current density or index
- 'A' = Amperes (current) vs 'a' = acceleration
- 'V' = Volts (potential) vs 'v' = velocity
- 'C' = Coulombs (charge) vs 'c' = speed of light (3×10⁸ m/s) vs °C (temperature)
- 'T' = Tesla (magnetic field) vs 't' = time vs 'T' = period or temperature
- 'Hz' = Hertz, NOT 'H2' — check for lowercase z
- 'Pa' = Pascals, NOT 'pa' or 'PA'

SI PREFIXES (critical for magnitude):
- p (pico) = 10⁻¹², n (nano) = 10⁻⁹, μ (micro) = 10⁻⁶
- m (milli) = 10⁻³, c (centi) = 10⁻², k (kilo) = 10³
- M (mega) = 10⁶, G (giga) = 10⁹, T (tera) = 10¹²
- μ vs u: both used for micro-, but μ is standard

VECTOR NOTATION:
- Arrow above letter (F⃗, v⃗) = vector quantity
- Bold letter (F, v) = vector in typed context
- Hat/caret above (x̂, ŷ, ẑ) = unit vector
- Dot above (ẋ) = time derivative (velocity from position)
- Double dot (ẍ) = second derivative (acceleration)

COMMON PHYSICS CONSTANTS:
- c = 3.00 × 10⁸ m/s (speed of light)
- g = 9.8 or 9.81 m/s² (gravitational acceleration)
- G = 6.674 × 10⁻¹¹ N·m²/kg²
- h = 6.626 × 10⁻³⁴ J·s (Planck's constant)
- k_B = 1.381 × 10⁻²³ J/K (Boltzmann)
- e = 1.602 × 10⁻¹⁹ C (elementary charge)
- ε₀ = 8.854 × 10⁻¹² F/m (permittivity)

CIRCUIT DIAGRAM SYMBOLS:
- Resistor: zigzag line or rectangular box
- Capacitor: two parallel lines (gap between them)
- Battery: alternating long/short parallel lines
- Ground: decreasing horizontal lines or triangle
""".strip()

GENERAL_DOMAIN_RULES = """
=== GENERAL DOMAIN RULES ===
These rules apply across all academic domains. Use character disambiguation
and number reading rules above. No domain-specific term bank is loaded.

COMMON ACADEMIC ABBREVIATIONS:
- e.g. = for example, i.e. = that is, etc. = and so on
- fig. = figure, eq. = equation, ch. = chapter
- vs. = versus, approx. = approximately
- w/ = with, w/o = without, b/c = because
""".strip()

DOMAIN_RULES: dict[str, str] = {
    "biology": BIOLOGY_DOMAIN_RULES,
    "chemistry": CHEMISTRY_DOMAIN_RULES,
    "math": MATH_DOMAIN_RULES,
    "physics": PHYSICS_DOMAIN_RULES,
    "general": GENERAL_DOMAIN_RULES,
}

# ---------------------------------------------------------------------------
# Structured disambiguation pair access
# ---------------------------------------------------------------------------

_DISAMBIGUATION_PAIRS: list[tuple[str, str, str]] = [
    ("1", "l / I", "In numbers/data -> 1. In words -> l or I. Start of sentence -> I"),
    ("0", "O", "In numbers/measurements -> 0. In words -> O"),
    ("5", "S", "In numbers -> 5. In words -> S. Check if chemical element symbol"),
    ("2", "Z", "In numbers -> 2. In words -> Z"),
    ("6", "G", "In numbers -> 6. In words -> G"),
    ("8", "B", "In numbers -> 8. In words -> B"),
    ("rn", "m", "Read surrounding word -- does 'rn' or 'm' make a valid word?"),
    ("cl", "d", "Read surrounding word -- does 'cl' or 'd' make a valid word?"),
    ("u", "v", "Rounded bottom = u, pointed = v"),
    ("a", "o", "Has tail = a, fully closed = o"),
    ("n", "h", "Tall ascender = h, short = n"),
    ("e", "c", "Loop closed = e, open = c"),
    ("t", "+", "In text -> t. In math/data -> +"),
    ("x", "multiply", "In text/variables -> x. Between numbers -> multiply sign"),
    ("H", "K", "H has two verticals + horizontal bar. K has one vertical + diagonals"),
    ("B", "N / R", "B has bumps. N has diagonal. R has bump + leg. Check closed loops"),
    ("P", "R", "P has closed loop, no leg. R has loop + diagonal leg"),
    ("P", "F", "P has closed loop at top. F has open horizontals, no loop"),
    ("M", "N", "M has two humps. N has one diagonal. Count vertical strokes"),
    ("D", "O", "D has flat left side. O is fully rounded"),
    ("B", "D", "B has TWO loops on right. D has ONE smooth curve"),
    ("O", "Q", "O is closed. Q has tail extending from bottom-right"),
    ("G", "C", "G has inward bar at mid-right. C is open"),
    ("H", "A", "H has parallel verticals. A has pointed top"),
    ("L", "I", "L has horizontal foot. I is single vertical"),
    ("J", "I", "J curves left at bottom. I is straight"),
    ("E", "F", "E has three horizontals. F has two (no bottom)"),
    ("K", "R", "K has diagonals from mid-vertical. R has loop + leg"),
    ("U", "V", "U has rounded bottom. V has sharp point"),
    ("W", "M", "W points down. M points up. Check peak/valley orientation"),
    ("q", "g", "q has straight descender. g has looped descender"),
    ("b", "d", "b: ascender left + bump right. d: bump left + ascender right"),
    ("p", "q", "p: descender left + bump right. q: bump left + descender right"),
    ("f", "t", "f has descender and top curve. t is shorter with cross"),
    ("i", "j", "i has dot, no descender. j has dot, curves left at bottom"),
    ("w", "uu", "w is single char. uu is two strokes. Check spacing"),
    ("ri", "n", "ri has dot above. n does not. Check for dot"),
    ("9", "q", "In numbers -> 9. In words -> q"),
    ("7", "1", "7 has horizontal top. 1 is single vertical. Check for crossed 7"),
    ("4", "9", "4 has open angular top. 9 has closed loop top"),
    ("3", "8", "3 is open on left. 8 is fully closed"),
    ("S", "5 / $", "S is smooth. 5 has angular top. $ has vertical through S"),
    ("C", "(", "C is letter-sized. ( is taller and narrower"),
    (".", ",", "Period on baseline. Comma has descending tail"),
]


def get_disambiguation_pairs() -> list[tuple[str, str, str]]:
    """Return structured list of confusion pairs.

    Each tuple is (char_a, char_b, resolution_rule).
    """
    return list(_DISAMBIGUATION_PAIRS)


# ---------------------------------------------------------------------------
# Helper to build composite prompt sections
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Content-type → strategy subset mapping
# ---------------------------------------------------------------------------

# Which rule blocks matter most for each content type.  Avoids dumping
# ~2,500 tokens of rules that don't apply (noise hurts model attention).
_CONTENT_TYPE_RULES: dict[str, list[str]] = {
    "handwriting": ["fidelity", "calibration", "multipass", "disambiguation"],
    "cursive":     ["fidelity", "calibration", "multipass", "disambiguation"],
    "structured":  ["fidelity", "multipass", "numbers", "tables"],
    "printed":     ["fidelity", "multipass"],
    "layout":      ["fidelity", "multipass", "tables", "scientific"],
    "scientific":  ["fidelity", "multipass", "numbers", "scientific", "disambiguation"],
    "data_table":  ["fidelity", "numbers", "tables"],
    "single_letter": ["fidelity", "calibration", "single_letter", "disambiguation"],
    "default":     ["fidelity", "calibration", "multipass", "numbers", "disambiguation"],
    "full":        ["fidelity", "calibration", "multipass", "numbers",
                    "disambiguation", "scientific", "tables", "single_letter"],
}

_RULE_BLOCKS = {
    "fidelity":      lambda: TRANSCRIPTION_FIDELITY_RULE,
    "calibration":   lambda: WRITER_CALIBRATION,
    "multipass":     lambda: MULTI_PASS_READING_INSTRUCTIONS,
    "numbers":       lambda: NUMBER_READING_RULES,
    "disambiguation": lambda: CHARACTER_DISAMBIGUATION,
    "scientific":    lambda: SCIENTIFIC_NOTATION,
    "tables":        lambda: TABLE_DETECTION,
    "single_letter": lambda: SINGLE_LETTER_PROTOCOL,
    "faint":         lambda: FAINT_CONTENT_PROTOCOL,
}


def get_reading_strategies(
    domain: str = "biology",
    content_type: str = "default",
) -> str:
    """
    Build a handwriting reading strategy block tailored to content type.

    Instead of injecting ALL rules every time (~2,500 tokens), selects
    only the rule subsets relevant to the content type.  Use
    content_type="full" to get everything (backwards-compatible).

    Args:
        domain: Subject domain ('biology', 'general', or any key in DOMAIN_RULES)
        content_type: Type of content being read (see _CONTENT_TYPE_RULES keys)

    Returns:
        Multi-line string with applicable reading strategies
    """
    rule_keys = _CONTENT_TYPE_RULES.get(content_type, _CONTENT_TYPE_RULES["default"])
    sections = [_RULE_BLOCKS[k]() for k in rule_keys if k in _RULE_BLOCKS]

    if domain in DOMAIN_RULES:
        sections.append(DOMAIN_RULES[domain])

    return "\n\n".join(sections)


def get_grading_handwriting_rules(domain: str = "biology") -> str:
    """
    Compact version of handwriting rules for grading prompts.
    Focused on disambiguation and tolerance rather than full
    transcription protocol.
    """
    sections = [
        CHARACTER_DISAMBIGUATION,
        NUMBER_READING_RULES,
        SINGLE_LETTER_PROTOCOL,
    ]

    if domain in DOMAIN_RULES:
        sections.append(DOMAIN_RULES[domain])

    return "\n\n".join(sections)
