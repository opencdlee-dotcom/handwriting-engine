# Domain Pitfalls: HTR Benchmarking

**Domain:** Adding rigorous accuracy benchmarking to an existing HTR engine
**Researched:** 2026-04-11
**Milestone:** v3.0 — Verified Accuracy

---

## Critical Pitfalls

Mistakes that cause measurement invalidation, requiring re-runs or discarding data.

---

### Pitfall 1: IAM Test/Train Split Contamination

**What goes wrong:** Evaluating on IAM images that were used (directly or indirectly) to tune the system — through prompt engineering, confusion-pair selection, or vocabulary hint lists that were derived from observing IAM errors. The reported CER is then a training-set number, not a generalization number.

**Why it happens:** The confusion pairs in `handwriting.py` and `_CONFUSION_PAIRS` in `metrics.py` were almost certainly refined by looking at IAM failures. The biology vocabulary hints were chosen partly by observing IAM biology pages. Any prompt iteration that used IAM feedback is contamination.

**Consequences:** The 1.67% baseline may be partially contaminated. If this system is reported as "SOTA" and contamination is discovered, the claim collapses. More immediately: strategy improvements measured on contaminated test data are not reliable predictors of real-world gains.

**Prevention:**
- Use the **official IAM test split** (Aachen partition: 2,915 test lines) without exception. Never look at test-split images when tuning prompts, confusion pairs, or vocabulary.
- Maintain a hard `BENCHMARK_DATA_DIR/iam/test/` with a README stating "do not inspect these to tune the engine."
- For the v3.0 run: treat the current baseline of 1.67% as **suspect** until you confirm which IAM partition it was measured on. The CLAUDE.md says "internal benchmark" — investigate whether those images overlap with any images used during development.
- If contamination is confirmed, report two numbers: "development set CER" and "held-out test set CER."

**Detection:** Run `benchmark ingest` on official IAM test split, then compare CER — if it's materially higher than 1.67%, the original set was likely easier or contaminated.

**Phase:** Address in the setup phase of v3.0 before any comparison runs.

---

### Pitfall 2: Strategy Comparison Without Controlling Model Version

**What goes wrong:** Two benchmark runs executed days apart use different model versions for the same provider label (e.g., `gemini-1.5-flash` vs `gemini-1.5-flash-002`), making the CER delta attributable to the model change rather than the strategy change.

**Why it happens:** The system currently pins GPT (`gpt-4.1-2025-04-14`) but the Gemini and Claude provider code uses non-date-pinned aliases. The `compare_strategies()` function runs independent benchmark calls sequentially — any API update between runs is invisible to the comparison logic.

**Consequences:** A self_correct vs best_of comparison that spans a Gemini model update will produce a false CER delta. The confusion-pair improvement from self_correct could be entirely explained by the model version change, or masked by it.

**Prevention:**
- Pin all model versions in `_constants.py` with date suffixes before any v3.0 comparison run. Verify by logging the model string returned in API metadata (not just what was requested — some APIs silently reroute).
- Record the exact model version in the `runs` table. Add a `model_versions` JSON column to the `runs` schema so comparisons can assert version parity.
- Run all strategy comparisons in a **single `compare_strategies()` call** rather than separate CLI invocations on different days.

**Detection:** Check that `runs.providers` JSON in the DB contains version-pinned strings, not aliases.

**Phase:** Model pinning must happen in Phase 1 of v3.0, before any benchmark runs.

---

### Pitfall 3: Naive Full-Dataset API Cost Explosion

**What goes wrong:** Running `--compare-strategies vote,best_of,self_correct,line_level` with all three cloud providers on 500 IAM test samples burns through $50-200 in a single command because the cost is multiplicative: `strategies × providers × samples × passes_per_strategy`.

**Why it happens:** `compare_strategies()` runs a full `run_benchmark()` per strategy. Self_correct sends 2 API calls per sample per provider (primary read + correction pass). Line_level sends N API calls per sample where N = number of detected lines. With 3 providers × 4 strategies × 500 samples × 2 passes = 12,000 API calls.

**Consequences:** Budget exceeded, partial runs with corrupt DB state, rate limit throttling mid-run causing inflated latency numbers.

**Prevention:**
- Always run `--smoke` first (3 hardest samples). Use smoke results to extrapolate full cost before committing.
- Add a `--dry-run` / cost estimator that prints predicted cost before executing: `(providers × strategies × samples × avg_tokens) × rate`.
- Use `--sample-ids` to run on 20-30 representative samples for strategy development, then full IAM test set for final publication numbers only.
- OpenAI Batch API (already in `providers/batch_openai.py`) gives 50% discount — use it for the final full-suite run.
- Never run full sweep at temperature 0 as a "quick sanity check" — always smoke first.

**Detection:** Cost estimates before execution; token counters in `_read_single()` already track per-sample token usage — add a pre-run cost projection.

**Phase:** Cost management infrastructure before any full-IAM comparison run.

---

### Pitfall 4: CER Normalization Asymmetry Between Strategies

**What goes wrong:** `normalize_text()` strips `[?]` and `[illegible]` markers before computing CER. This means a strategy that emits many uncertainty markers gets artificially low CER compared to one that guesses (possibly wrong) characters. The comparison appears to favor the conservative strategy even when the guessing strategy is more useful.

**Why it happens:** The current `normalize_text()` removes all engine markers. Self_correct tends to resolve ambiguities and emit fewer `[?]` — so it gets credit for both error reduction AND marker reduction in the same CER number. Line_level may produce more confident reads (fewer markers) without being more accurate character-for-character.

**Consequences:** `compare_strategies()` output will show self_correct as better than best_of even if the actual transcription of ambiguous characters is identical — the difference is just in marker use, not character accuracy.

**Prevention:**
- Report **two CER numbers** per strategy: `cer_stripped` (current behavior, markers removed) and `cer_markers_as_errors` (markers counted as deletions/substitutions of the underlying character count).
- Also report `[?]_rate` (markers per 100 characters) as a separate column in the comparison table — this separates "accurate on clear text" from "confident on unclear text."
- The `classify_errors()` function already distinguishes insertion/deletion/substitution — add a `marker_count` field to `EvalMetric`.

**Detection:** Look for strategy comparisons where CER drops but `[?]_rate` also drops — this suggests the improvement is partly definitional, not real.

**Phase:** Fix normalization reporting before any strategy comparison results are committed as regression baselines.

---

### Pitfall 5: Benchmark Data Contamination via System Prompt

**What goes wrong:** The system prompt includes vocabulary hints, confusion pair warnings, and writer calibration that was derived from observing errors on the same images used as benchmark samples. Evaluating on those images reports the engine performing better than it would on unseen data.

**Why it happens:** The biology vocabulary list in `postprocess.py` and `metrics.py` was built from observed lab notebook content. If the IAM benchmark set includes biology content and those exact term patterns informed the vocabulary hints, the hints function as a lookup table for that specific content. Similarly, `WriterProfileStore` calibration data derived from benchmark images and then injected during benchmark evaluation is circular.

**Consequences:** Domain term accuracy metrics (already computed by `domain_term_accuracy()`) will be inflated by vocabulary hints that essentially memorize the test vocabulary.

**Prevention:**
- Run baseline benchmarks **without** vocabulary hints and **without** domain spell correction first. Record this as the "no-assist" CER.
- Run with hints/correction enabled as a separate named run. The delta shows the value of those features, not their presence in the baseline.
- Never use benchmark images to calibrate `WriterProfileStore` entries that are then used during benchmark evaluation.
- Add a `--no-hints` flag to `benchmark run` that disables vocabulary_hints injection and domain correction for clean baseline measurements.

**Phase:** Protocol discipline — establish before the first v3.0 benchmark run, document in the benchmark run label.

---

## Moderate Pitfalls

---

### Pitfall 6: Statistical Significance Theater on Small IAM Samples

**What goes wrong:** Reporting a CER improvement from 1.67% to 1.34% as "statistically significant (p < 0.05)" based on a paired t-test over 50 samples, without reporting effect size or confidence intervals. The p-value is real but the interval is wide — the true improvement could be anywhere from 0.1% to 0.9%.

**Why it happens:** Paired t-tests on CER proportions with n=50 will achieve p < 0.05 for differences as small as 0.2 CER percentage points. Reviewers and downstream consumers treat p < 0.05 as "real improvement" without checking whether the improvement is meaningful.

**Consequences:** A 0.1% CER improvement gets treated as equivalent to a 0.6% improvement. Worse, if you later measure on a different 50 samples, the ordering may reverse.

**Prevention:**
- Report **95% confidence intervals** on CER differences, not just point estimates. Bootstrap resampling (already half-built with `bootstrap-gt` command) can generate these.
- Report **effect size** (Cohen's d for paired differences) alongside p-values.
- For CER comparisons at the 1-2% absolute level, you need approximately **200-500 samples** to reliably detect 0.2 percentage point differences at 80% power. With fewer samples, report the comparison as "directional" not "confirmed."
- Use **Wilcoxon signed-rank test** rather than paired t-test — CER values per sample are not normally distributed (heavy right tail from hard images), so the t-test assumption is violated.
- The McNemar test is appropriate for comparing per-character binary correct/incorrect decisions across strategies.

**Detection:** Check whether confidence intervals on reported CER deltas overlap zero. If they do, the comparison is not reliable.

**Phase:** Statistics methodology before any final v3.0 numbers are documented.

---

### Pitfall 7: IAM vs Lab Notebook Distribution Mismatch

**What goes wrong:** The engine achieves excellent CER on IAM (clean, scanned, English prose from 1990s writers) but the actual use case is student lab notebooks (biology jargon, variable scan quality, student handwriting from 2020s, pencil on ruled paper, inline diagrams). Optimizing for IAM CER produces a system tuned for the wrong distribution.

**Why it happens:** IAM is the standard academic benchmark, so it's tempting to chase IAM numbers. But the PROJECT.md explicitly states "best configuration for lab notebook grading use case" as a target outcome. These are different optimization problems.

**Consequences:** Self_correct may improve IAM CER but not improve (or even degrade) performance on lab notebooks if the correction prompts are tuned toward IAM-style errors (cursive lowercase confusion) rather than lab notebook errors (chemical symbol misread, equation notation, pencil fading).

**Prevention:**
- Maintain **two separate benchmark sets**: `category="iam"` and `category="lab_notebook"` in the samples table. The DB schema already has a `category` column.
- Report CER separately for each category. Never average them together.
- The primary optimization target for production decisions should be lab notebook CER, with IAM as a secondary validation check.
- Collect at least 30 real lab notebook pages with manual ground truth before drawing any production conclusions from v3.0 results.
- The `[?]` rate reduction metric (from PROJECT.md) is a lab-notebook-specific proxy — track it separately from CER.

**Detection:** If CER improves on IAM but `[?]_rate` on lab notebooks doesn't change, the improvement is academically valid but operationally meaningless.

**Phase:** Data collection (real lab notebook GT) must precede or run in parallel with IAM benchmarking.

---

### Pitfall 8: Temperature Non-Determinism Inflating Variance

**What goes wrong:** The engine uses `temperature=0.5` for Gemini (documented in CLAUDE.md as intentional — "temp 0 causes degenerate sampling"). At temperature 0.5, re-running the same benchmark on the same images will produce different CER values on each run. A difference between strategies could be within the noise band of single-temperature variance.

**Why it happens:** Temperature 0.5 introduces stochastic variation. For a single 50-sample run, the standard deviation of the CER estimate is not just from sampling variance (image difficulty) but also from generation variance (temperature noise). Two runs of the same strategy can differ by 0.1-0.2 CER percentage points purely from temperature.

**Consequences:** The regression threshold of 0.5 percentage points in `compare_strategies()` may be smaller than the temperature noise floor on small sample sets, triggering false regression alerts or missing real regressions.

**Prevention:**
- Run each strategy **3 times** on the same sample set and report mean ± std CER across runs. A strategy comparison is valid only when the difference exceeds 3× the within-strategy run-to-run std.
- For regression detection, calibrate the threshold empirically: run the same strategy twice on the same samples and measure the natural variance. Set the regression threshold above this floor.
- Consider a **temperature-0 run in parallel** for IAM benchmarking even if production uses 0.5 — this gives a noise-free comparison baseline, with the caveat that the documented degenerate sampling issue should be confirmed empirically rather than assumed.

**Detection:** Run the same strategy twice on the same 20 samples. If CER differs by more than 0.15 percentage points, temperature noise is a material confounder.

**Phase:** Calibration run before any strategy comparisons are committed.

---

### Pitfall 9: CER Metric Blindness to Domain Term Errors

**What goes wrong:** CER treats all characters equally. Getting "mitochondria" wrong counts as 11 edit operations in the same pool as getting "the" wrong (3 operations). In biology lab notebooks, a single domain term error ("mitocondria" → "mitochondria") has a disproportionate impact on student grading correctness — it's a substantively worse error than misreading a filler word.

**Why it happens:** Standard CER is a document-level metric borrowed from ASR evaluation where all characters carry equal weight. Lab notebook grading needs term-level correctness for specific vocabulary items.

**Consequences:** Self_correct might improve overall CER by correctly reading small common words while actually degrading domain term accuracy. The aggregate CER improvement would mask the operationally important regression.

**Prevention:**
- `domain_term_accuracy()` is already implemented in `metrics.py` and should be reported alongside CER for every strategy comparison. Never report CER alone for lab notebook evaluation.
- Add domain term accuracy to the comparison table output in `compare_strategies()`.
- Flag any run where CER improves but domain term accuracy decreases — this is an inversion that matters for the use case.

**Detection:** Compare `cer` and `domain_term_accuracy` columns across strategies. They should move in the same direction; if they diverge, investigate.

**Phase:** Reporting layer — ensure domain term accuracy appears in all comparison outputs before v3.0 results are committed.

---

## Minor Pitfalls

---

### Pitfall 10: Benchmark Infrastructure Over-Engineering

**What goes wrong:** The benchmarking milestone turns into building elaborate dashboards, interactive drill-downs, error taxonomy visualizations, and multi-dimensional comparison matrices — all before getting a single valid CER number on the held-out test set.

**Why it happens:** The benchmark subpackage is already substantial (7 modules, multiple CLI subcommands). There's temptation to keep extending it with quality-vs-accuracy correlation charts, per-writer breakdowns, and export pipelines rather than running the actual benchmark and reading the numbers.

**Consequences:** The v3.0 milestone ships without confirming whether self_correct actually achieves the target <1.3% CER. Infrastructure exists but the measurement hasn't happened.

**Prevention:**
- The v3.0 success criterion is specific numbers: CER for each strategy. Every infrastructure task should be gated on "does this unblock getting the number, or does it make the number prettier?"
- The existing `report.py`, `evaluate.py`, and `compare_strategies()` are already sufficient. Do not extend them before running at least one complete comparison.
- Timebox infrastructure work: if a reporting enhancement takes more than 2 hours, defer it until after the core CER numbers are committed.

**Phase:** Ongoing — apply this discipline at every planning step.

---

### Pitfall 11: Baseline CER Recorded Without Normalization Documentation

**What goes wrong:** The 1.67% baseline is committed to the regression DB and to CLAUDE.md without recording exactly what normalization was applied, which IAM partition was used, what model version was running, and whether vocabulary hints were enabled. Future comparisons are then made against an underspecified baseline.

**Why it happens:** The 1.67% number appears in CLAUDE.md as a fact but its provenance is opaque — "internal benchmark" without partition, model version, or normalization details.

**Consequences:** A future run that produces 1.8% CER might represent a real regression, a different model version, a different IAM partition, or different normalization — but there is no way to distinguish these causes.

**Prevention:**
- The first v3.0 task should be to **reproduce the 1.67% baseline** with fully documented parameters and commit the run record to the DB with all metadata.
- The `runs` table already stores provider/strategy/domain. Add a `notes` field entry that captures: IAM partition identifier, model version string, normalization version, hints on/off.
- Treat the original 1.67% as a "claimed baseline" and the first v3.0 run as the "verified baseline" — they may differ, and that's fine.

**Phase:** First task of v3.0 before any comparison work.

---

### Pitfall 12: PaddleOCR Benchmark on Wrong IAM Preprocessing

**What goes wrong:** PaddleOCR (PP-OCRv5) was designed and tested with its own internal preprocessing pipeline. Running it through the handwriting engine's `enhance_image()` → PaddleOCR path may produce worse CER than running PaddleOCR on raw IAM images, because the enhancement was tuned for LLM vision inputs, not for the PaddleOCR CNN frontend.

**Why it happens:** The enhancement pipeline (grayscale → autocontrast → sharpen → contrast → brightness → 2x upscale) was validated for Gemini/Claude/GPT inputs. PaddleOCR's CNN may prefer different preprocessing or may have internal preprocessing that conflicts with the engine's upscaling.

**Consequences:** PaddleOCR will appear worse than its true capability in comparisons, leading to incorrect ensemble weight assignments.

**Prevention:**
- Run PaddleOCR benchmarks in two modes: `enhance_strategy=None` (raw input) and `enhance_strategy="proven"`. Report both CER values.
- The brainiac research notes PP-OCRv5 at "~5.8% CER on IAM (pre-v3.0)" — verify this is with or without engine preprocessing.
- Do not set PaddleOCR ensemble weights until preprocessing interaction is measured.

**Phase:** PaddleOCR benchmark setup.

---

### Pitfall 13: Lessons Bridge Circular Feedback in Benchmarks

**What goes wrong:** The `--feed-lessons` flag feeds high-error benchmark outputs back into the lessons system. If that same lessons data is then injected during a subsequent benchmark run (`inject_lessons=True`), the benchmark is measuring the system's ability to memorize its own past errors on that dataset, not its generalization ability.

**Why it happens:** The lessons bridge (`lessons_bridge.py`) and the `inject_lessons` flag in `run_benchmark()` make it easy to create a training loop that runs through the benchmark infrastructure. This is useful for production adaptation but invalid for accuracy reporting.

**Consequences:** CER numbers generated with `inject_lessons=True` on data that previously had `--feed-lessons` applied are not comparable to cold-start measurements.

**Prevention:**
- All benchmark runs used for accuracy reporting must use `inject_lessons=False`.
- The lessons system is a production feature for writer adaptation, not an evaluation tool. These two workflows should never be mixed in the same benchmark run.
- Add a warning to the CLI: if `inject_lessons=True` is passed with a dataset that has previous lesson entries, print a warning that results are not cold-start valid.

**Phase:** Protocol documentation before any v3.0 runs.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Baseline reproduction | Unverified 1.67% provenance (Pitfall 11) | Reproduce with full documentation first |
| Strategy comparison setup | Model version drift between runs (Pitfall 2) | Pin all models before first comparison |
| IAM data ingestion | Test/train contamination (Pitfall 1) | Confirm official Aachen partition usage |
| Self_correct vs best_of comparison | Temperature noise floor (Pitfall 8) | Run each strategy 3x, report mean ± std |
| Normalization reporting | Marker stripping hides [?] behavior (Pitfall 4) | Report both cer_stripped and [?]_rate |
| Domain term accuracy | CER blindness to biology terms (Pitfall 9) | Always co-report domain_term_accuracy |
| Lab notebook evaluation | IAM/lab distribution mismatch (Pitfall 7) | Separate category="iam" vs "lab_notebook" |
| Full-IAM sweep | API cost explosion (Pitfall 3) | Smoke first, cost-estimate before full run |
| PaddleOCR benchmarking | Enhancement pipeline conflict (Pitfall 12) | Test raw vs enhanced inputs separately |
| Statistical reporting | p-value theater (Pitfall 6) | Report 95% CI and effect size |
| Lessons feedback | Circular benchmark contamination (Pitfall 13) | inject_lessons=False for all accuracy runs |
| Infrastructure extensions | Over-engineering before measurement (Pitfall 10) | Get numbers first, visualize second |

## Sources

- Internal: `handwriting_engine/benchmark/metrics.py` — `normalize_text()`, `domain_term_accuracy()`, `classify_errors()`
- Internal: `handwriting_engine/benchmark/evaluate.py` — `compare_strategies()`, `_read_single()`, temperature/version tracking
- Internal: `CLAUDE.md` — temperature=0.5 rationale, model version pinning history, 1.67% baseline claim
- Internal: `brainiac-htr-sota.md` — Journal of Documentation 2025 self-correction findings, PaddleOCR IAM CER estimates
- Domain knowledge: IAM Aachen partition (standard academic split: 6,482 train / 976 validation / 2,915 test lines)
- Domain knowledge: Wilcoxon signed-rank test appropriateness for non-normal per-sample CER distributions
- Domain knowledge: McNemar test for per-character binary comparisons across strategies
- Domain knowledge: Bootstrap resampling for CER confidence intervals (200+ samples recommended for 0.2pp sensitivity)
