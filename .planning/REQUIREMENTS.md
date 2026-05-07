# Requirements: handwriting-engine

**Defined:** 2026-04-11
**Core Value:** Highest accuracy handwriting transcription available — every accuracy claim backed by measured, reproducible numbers.

## v3.0 Requirements — Verified Accuracy

Requirements for the benchmarking milestone. Each maps to a roadmap phase.

### Measurement Foundation

- [x] **FOUND-01**: Developer can reproduce the 1.67% CER baseline with a documented provenance record (model version, IAM partition ID, normalization flags, vocabulary hints off) so all future comparisons have a valid anchor.
- [x] **FOUND-02**: Benchmark runs store `[?]_marker_rate` as a separate column alongside CER, so strategies that resolve ambiguity are not conflated with those that improve character accuracy.
- [x] **FOUND-03**: Developer can run a 20-sample noise floor calibration that measures CER variance at temperature 0.5 and reports the minimum detectable CER difference for this test set.
- [x] **FOUND-04**: CLI warns with an API cost projection (strategies × providers × samples × passes) before executing any sweep run, preventing unintended cost explosions.

### IAM Benchmarking

- [x] **IAM-01**: Developer can ingest the IAM Handwriting Database line images and ground-truth transcriptions into the existing benchmark DB using `benchmark ingest-iam` (parses IAM ascii/ GT format, tags `category="iam"`, `student="iam-writer-XXX"`).
- [ ] **IAM-02**: Developer can run a full strategy sweep with `benchmark sweep` that executes all strategies (baseline, self_correct, line_level, prompt_adapted, zoomed_verify) against the IAM test set and stores one run_id per strategy.
- [ ] **IAM-03**: `benchmark report` can group and display per-writer CER breakdown, showing variance across writers to distinguish systematic gains from writer-specific noise.

### Statistics

- [ ] **STAT-01**: `benchmark compare` automatically appends Wilcoxon signed-rank p-value and Cohen's r effect size when comparing two runs with n >= 10 samples, so CER differences are statistically defensible.
- [ ] **STAT-02**: `benchmark compare` reports 95% bootstrap confidence intervals on CER estimates, distinguishing real improvements from sampling noise.

### Reporting

- [x] **RPT-01**: Schema v6 adds `is_baseline` flag to runs table; `benchmark set-baseline RUN_ID` pins a run as the regression anchor; `detect_regressions()` compares against the pinned baseline, not runs[-2]. (shipped 2026-05-06)
- [x] **RPT-02**: `benchmark recommend` outputs the best strategy+provider configuration with a weighted composite score (70% CER / 15% cost / 15% stability across runs). (shipped 2026-05-06; end-to-end on multi-strategy sweep gated on IAM data)
- [x] **RPT-03**: Developer can collect and store ground-truth transcriptions from real student lab notebooks using `benchmark ingest-lab` with a guided annotation workflow, enabling production-distribution benchmarks distinct from IAM. (shipped 2026-05-06)

## v4.0 Requirements (Deferred)

### Per-provider prompt adaptation measurement
- **ADAPT-01**: Benchmark prompt_adapter per provider — measure CER delta from Gemini concise vs. full disambiguation table
- **ADAPT-02**: Benchmark writer_embeddings auto-identification accuracy vs. manual writer_id

### Advanced statistics
- **STAT-03**: McNemar's test for comparing error patterns (not just aggregate CER) between strategies
- **STAT-04**: Per-character confusion matrix from error taxonomy data

## Out of Scope

| Feature | Reason |
|---------|--------|
| MLflow / W&B experiment tracking | Overkill for single-developer project; SQLite benchmark DB is sufficient |
| Pandas for aggregation | Existing stdlib statistics + SQLite already handles all aggregation needs |
| Automated IAM download | IAM is registration-gated at HEIA-FR; must be downloaded manually |
| Training set evaluation | Test/train discipline: only IAM test partition (never lines used during prompt development) |
| Multi-language IAM subsets | English only; scope matches production use case |
| Real-time benchmark dashboard | Not needed; CLI report output is sufficient for this use case |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUND-01 | Phase 6 | Complete |
| FOUND-02 | Phase 6 | Complete |
| FOUND-03 | Phase 6 | Complete |
| FOUND-04 | Phase 6 | Complete |
| IAM-01 | Phase 7 | Complete |
| IAM-02 | Phase 7 | Pending |
| IAM-03 | Phase 7 | Pending |
| STAT-01 | Phase 8 | Implemented (verification gated on IAM data) |
| STAT-02 | Phase 8 | Implemented (verification gated on IAM data) |
| RPT-01 | Phase 9 | Implemented |
| RPT-02 | Phase 9 | Implemented (verification gated on IAM sweep) |
| RPT-03 | Phase 9 | Implemented |

**Coverage:**
- v3.0 requirements: 12 total
- Mapped to phases: 12
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-11*
*Last updated: 2026-04-11 after v3.0 roadmap creation*
