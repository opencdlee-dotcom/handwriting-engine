# Feature Landscape

**Domain:** HTR accuracy validation — benchmarking milestone for handwriting-engine v3.0
**Researched:** 2026-04-11

---

## What Already Exists (Do Not Rebuild)

The benchmark subsystem is substantially built. Before assessing what to add, map what is already in place:

| Capability | Location | Status |
|------------|----------|--------|
| CER/WER (Levenshtein, jiwer fallback) | `benchmark/metrics.py` | Complete |
| Error taxonomy (confusion_pair / substitution / insertion / deletion) | `benchmark/metrics.py:classify_errors()` | Complete |
| Domain term accuracy (biology terms, single + multi-word) | `benchmark/metrics.py:domain_term_accuracy()` | Complete |
| SQLite run/sample/ground_truth/eval_metrics schema | `benchmark/db.py` | Complete (schema v3) |
| Per-run aggregate report (table / JSON / CSV) | `benchmark/report.py:generate_report()` | Complete |
| Run-vs-run comparison with regression delta | `benchmark/report.py:compare_runs()` | Complete |
| Regression detection (threshold-based, run-over-run) | `benchmark/report.py:detect_regressions()` | Complete |
| Per-sample drill-down (all providers, CER, output preview) | `benchmark/report.py:sample_drill_down()` | Complete |
| Quality-vs-accuracy correlation | `benchmark/report.py:quality_correlation()` | Complete |
| Confidence calibration (Pearson, per-bucket) | `benchmark/report.py:confidence_calibration()` | Complete |
| `--compare-strategies` CLI flag | `cli.py` → `evaluate.py:compare_strategies()` | Complete |
| `--preprocessing` / `enhance_strategy` flag | `cli.py` → `run_benchmark()` | Complete |
| Smoke mode (3 hardest samples, history-ranked) | `evaluate.py:_select_smoke_samples()` | Complete |
| Image ingestion with SHA-256 dedup | `benchmark/ingest.py:ingest_directory()` | Complete |
| Synthetic degradation (7 variants per sample) | `benchmark/ingest.py:generate_degraded_variants()` | Complete |
| Bootstrap ground truth (high-agreement auto-GT) | `benchmark/ingest.py:bootstrap_ground_truth()` | Complete |
| `_count_uncertainty_markers()` in consensus engine | `consensus.py` | Complete |
| Mean CER, median CER, stdev CER per (provider, strategy) | `report.py:_aggregate_results()` | Complete |
| Cost estimation (USD) per run | `evaluate.py:estimate_cost()` | Complete |

**Key gap:** Everything above operates on whatever samples are in the SQLite DB. There is no IAM-specific ingest path, no per-writer CER breakdown query, no [?]-marker rate metric in the report layer, no statistical significance test, and no best-configuration recommendation output. Those are the delta for v3.0.

---

## Table Stakes

Features whose absence makes the v3.0 milestone goal ("turning claimed improvements into proven numbers") incomplete.

| Feature | Why Expected | Complexity | Depends On |
|---------|--------------|------------|------------|
| IAM test set ingest script | All CER claims are against IAM — without real IAM images in the DB the headline number is unverifiable | Low | Existing `ingest_directory()` — needs a script that fetches IAM via the Unofficial IAM downloader or assumes user has the dataset |
| IAM ground truth loading | IAM ships per-line `.gt.txt` files; the benchmark expects ground truth in DB — need a loader that reads those files and calls `insert_ground_truth()` | Low-Med | `benchmark/db.py:insert_ground_truth()` |
| Per-strategy CER comparison with stdev | Already partially exists via `compare_strategies()`, but stdev is computed in `_aggregate_results()` yet NOT rendered in the `compare_strategies()` output table (only in `generate_report()`) | Low | `evaluate.py:compare_strategies()` — add stdev column |
| [?] marker rate metric | The `_count_uncertainty_markers()` function exists in `consensus.py` but is NOT in the metrics layer and NOT stored in eval_metrics or provider_outputs — the report has no [?] rate column | Med | `benchmark/metrics.py` — new `uncertainty_marker_rate()` function; DB schema — add `marker_count` column to `provider_outputs` |
| Per-writer CER breakdown | The `samples` table has a `student` column. No report function groups by student and computes per-student mean/median CER. Essential for lab notebook validation because accuracy varies sharply by handwriting style | Med | `benchmark/db.py` — new query; `benchmark/report.py` — new `per_writer_report()` function |
| Regression baseline commit | A specific run designated as the "locked baseline" — future CI smoke tests compare against it. Currently there is no concept of a pinned baseline run vs. an ad-hoc run | Low | `benchmark/db.py` — add `is_baseline` flag to `runs` table; CLI `benchmark baseline set <run_id>` |
| Baseline CI smoke test | `benchmark run --smoke` already picks 3 hardest samples, but does not compare against a pinned baseline — it just runs and shows a table. Need `benchmark smoke-check` that exits non-zero if any strategy regresses beyond threshold vs baseline | Low | Baseline run ID; existing `detect_regressions()` |

---

## Differentiators

Features that go beyond minimum verification and make the benchmark report publishable / defensible.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Statistical significance reporting (bootstrap CI or McNemar's test) | CER difference of 0.37pp (1.67% → 1.30%) on a small IAM test set may not be significant at p<0.05. Without a p-value or confidence interval, the claim is not peer-review quality | Med | scipy not currently a dependency; bootstrap resampling can be done in stdlib. For paired CER, bootstrap CI on delta is more appropriate than McNemar (which is for binary outcomes). No new dependencies needed if implemented with stdlib `random`. |
| Error taxonomy in comparison table | `classify_errors()` exists but is never called during benchmark runs — errors are not stored by type. Adding a call in `run_benchmark()` and a column in `eval_metrics` for confusion_pair / substitution / insertion / deletion counts would let the report show "self_correct cuts confusion_pair errors by X%" | Med | `benchmark/db.py` schema add; `evaluate.py` call `classify_errors()` and store result |
| Best-configuration recommendation output | A `benchmark recommend` CLI command that queries all runs, finds the (provider, strategy) combination with lowest mean CER, verifies it is statistically distinguishable from baseline, and prints "RECOMMENDED: gemini / self_correct — 1.31% CER (baseline: 1.67%, delta: -0.36pp, p<0.05)" | Med | Depends on statistical significance feature; per-strategy CER comparison |
| Per-writer CER heatmap (text table) | Extend per-writer breakdown to show a matrix: rows = writers, columns = strategies, cells = CER. Reveals whether self_correct helps uniformly or only for certain handwriting styles | Med | Per-writer CER breakdown (table stakes) |
| [?] marker rate trend across strategies | Show how marker rate drops: baseline=4.2 markers/page, line_level=2.1, self_correct=0.8. Validates the "line_level reduces [?] markers" claim specifically | Low | [?] marker rate metric (table stakes) |
| Cost-per-accuracy-point summary | Extend the existing cost column into a derived metric: `(baseline_cer - strategy_cer) / strategy_cost_usd`. Frames accuracy gains in economic terms for lab notebook grading budget decisions | Low | Existing cost estimation; per-strategy CER |

---

## Anti-Features

Features to explicitly NOT build in this milestone.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Full IAM dataset download automation | IAM requires registration at https://fki.iam.unibe.ch — cannot auto-download without credentials. Building an automation that fails on first run wastes time | Document the manual download step; write the ingest script assuming the user has already placed images in a directory |
| TrOCR fine-tuning within the benchmark run | `fine_tune_for_writer()` exists but requires GPU or long CPU time. Including it in benchmark runs makes them non-reproducible (model weights change each run) | Benchmark TrOCR zero-shot only in v3.0; fine-tuning is a separate evaluation that can be documented but not automated in CI |
| Per-character confusion matrix visualization | Computing a full 26x26 confusion matrix (a→b, a→c, etc.) is interesting but not actionable for the v3.0 goal. The existing `classify_errors()` categories (confusion_pair / substitution / insertion / deletion) are sufficient | Use existing classify_errors output in the report |
| Multiple IAM test splits (lines vs. words vs. forms) | IAM has three granularities. Lines is the standard for comparing against published CER numbers (TrOCR, DTrOCR, etc.). Words and forms add complexity without comparability | Use IAM lines split only (consistent with all published benchmarks in brainiac-htr-sota.md) |
| Real-time benchmark dashboard / web UI | Out of scope per PROJECT.md. The benchmark is a CLI tool for a single developer | Keep CLI-only |
| Word-level CER breakdown | WER is already computed. Word-level CER (CER restricted to word boundaries) is not standard and not what published benchmarks report | Report WER as-is |

---

## Feature Dependencies

```
IAM ground truth loading → Per-strategy CER comparison (needs real labeled data)
IAM ground truth loading → Per-writer CER breakdown (needs student metadata in IAM)
IAM ground truth loading → [?] marker rate metric (needs real runs against real text)

[?] marker rate metric → [?] marker rate trend across strategies

Per-strategy CER comparison (with stdev) → Statistical significance reporting
Statistical significance reporting → Best-configuration recommendation output

Regression baseline commit → Baseline CI smoke test

Error taxonomy in comparison table → (standalone, no new dependencies)
Per-writer CER breakdown → Per-writer CER heatmap
```

---

## What a Complete Benchmark Report Looks Like for This Use Case

A complete v3.0 benchmark report covers three audiences and three data sources:

### Report Section 1 — IAM Headline Numbers
Standard evaluation on IAM lines test split (same split as TrOCR, DTrOCR published results). Shows:
- Baseline CER (gemini/single): target ~1.67% (already measured)
- self_correct CER: target ~1.30% (unverified)
- line_level CER: measured
- prompt_adapter CER: measured
- PaddleOCR CER: measured (expect ~5.8% from brainiac research)
- Bootstrap 95% CI for each strategy delta vs. baseline
- Error taxonomy breakdown per strategy

This answers: "Did the v2.0 features actually improve IAM accuracy?"

### Report Section 2 — Lab Notebook [?] Marker Rate
Evaluation on real student lab notebook pages (existing DB samples with `category='biology'`). Shows:
- Mean [?] markers per page per strategy
- [?] rate reduction: baseline → line_level → self_correct → line_level+self_correct
- Per-writer [?] rates (some writers consistently trigger more uncertainty markers)
- Domain term accuracy per strategy

This answers: "Did the v2.0 features reduce uncertainty in real grading workflows?"

### Report Section 3 — Best Configuration Recommendation
A single-page summary intended to be committed to the repo as the official recommendation:
- RECOMMENDED CONFIG: `gemini / self_correct` (or whatever wins)
- CER: X.XX% (IAM lines)
- [?] rate: X.X per page (lab notebooks)
- Cost: $X.XXXX per page
- Statistical confidence: p < 0.05 vs baseline
- CLI command to reproduce: `handwriting-engine benchmark run --strategies self_correct --providers gemini`

This answers: "What should callers of handwriting_engine use?"

### Report Section 4 — Regression Baseline
The winning run is tagged as `is_baseline=True` in the DB. Future CI runs `benchmark smoke-check` against it. The baseline is committed as a JSON export alongside the code.

---

## MVP Recommendation

For the v3.0 milestone, prioritize in this order:

1. **IAM ingest + ground truth loader** — unblocks everything; without real IAM data none of the other features produce valid numbers
2. **[?] marker rate metric** — add `uncertainty_marker_rate()` to metrics, add `marker_count` column to `provider_outputs`, render in `generate_report()`
3. **Per-writer CER breakdown** — new `per_writer_report()` in report.py; query groups by `samples.student`
4. **Statistical significance (bootstrap CI)** — stdlib-only implementation; add to `compare_strategies()` output
5. **Regression baseline commit** — add `is_baseline` to runs table, `benchmark baseline set` CLI command
6. **Best-configuration recommendation output** — `benchmark recommend` command; synthesizes sections 1-3

Defer:
- **Per-writer CER heatmap**: nice-to-have; plain per-writer table is sufficient for v3.0
- **Cost-per-accuracy-point**: low effort but secondary to getting accurate numbers first
- **Error taxonomy in comparison table**: `classify_errors()` exists but storing per-error-type counts requires a schema migration; defer to v3.1 unless the IAM data shows a specific error type dominating

---

## Sources

- Codebase inspection: `handwriting_engine/benchmark/` (metrics.py, report.py, evaluate.py, db.py, ingest.py, models.py) — HIGH confidence
- `~/Developer/handwriting-engine/.planning/PROJECT.md` — HIGH confidence
- `.planning/research/brainiac-htr-sota.md` — MEDIUM confidence (PaddleOCR IAM CER ~5.8% is from pre-v3.0 research, not yet measured against PP-OCRv5 specifically)
- `.planning/milestones/v2.0-MILESTONE-AUDIT.md` — HIGH confidence (tech debt items confirmed)
- IAM dataset evaluation protocol: consistent with published TrOCR / DTrOCR benchmarks using IAM lines split — MEDIUM confidence (standard practice, but exact split selection needs to be confirmed against the IAM paper splits when downloading)
- Journal of Documentation 2025 (GPT-4o self-correction: 1.75% → 1.39%) — MEDIUM confidence (peer-reviewed; applied to Gemini projection is an estimate, not measured)
