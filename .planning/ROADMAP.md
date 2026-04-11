# Roadmap: Handwriting Engine

## Milestones

- ✅ **v2.0 Maximum Accuracy** — Phases 1-5 (shipped 2026-04-11)
- 📋 **v3.0 Verified Accuracy** — Phases 6-9 (active)

## Phases

<details>
<summary>✅ v2.0 Maximum Accuracy (Phases 1-5) — SHIPPED 2026-04-11</summary>

- [x] Phase 1: Self-Correction + Smart Escalation (2/2 plans) — REQ-001, REQ-002
- [x] Phase 2: Line-Level Segmentation (1/1 plan) — REQ-003
- [x] Phase 3: New Local Model Providers (1/1 plan) — REQ-004, REQ-005
- [x] Phase 4: Preprocessing + Writer Adaptation (1/1 plan) — REQ-006, REQ-007
- [x] Phase 5: Post-Processing + Benchmark Suite Completion (1/1 plan) — REQ-008, REQ-009

Full details: `.planning/milestones/v2.0-ROADMAP.md`

</details>

### v3.0 — Verified Accuracy

- [ ] **Phase 6: Measurement Foundation** — Reproducible baseline + variance floor + cost guardrails
- [ ] **Phase 7: IAM Data Ingestion + Sweep Infrastructure** — Full IAM benchmark pipeline
- [ ] **Phase 8: Statistics Layer** — Statistical defensibility for all comparisons
- [ ] **Phase 9: Final Sweep, Recommendation, and Baseline Lock** — Best config identified, regression anchor committed

## Phase Details

### Phase 6: Measurement Foundation
**Goal**: The developer can run reproducible CER benchmarks with documented provenance, a known noise floor, and protection against runaway API cost before any strategy sweep begins.
**Depends on**: Nothing (first v3.0 phase)
**Requirements**: FOUND-01, FOUND-02, FOUND-03, FOUND-04
**Success Criteria** (what must be TRUE when this phase completes):
  1. Developer can re-run the baseline IAM benchmark and reproduce 1.67% CER; the run record shows model version, IAM partition ID, normalization flags, and vocabulary-hints=off so any future delta is unambiguous.
  2. Every benchmark run record includes a `[?]_marker_rate` column separate from CER, so the developer can tell whether a strategy reduced ambiguity markers vs. improved character accuracy.
  3. Developer can execute a 20-sample noise calibration run that prints the CER variance and minimum detectable difference at temperature 0.5 — so the developer knows whether a measured CER delta is real or noise.
  4. Attempting to launch any sweep run first prints an API cost projection (strategies × providers × samples × passes) and requires confirmation before proceeding; no blind cost explosion is possible.
**Plans**: 4 plans

Plans:
- [ ] 06-01-PLAN.md — Wave 0 test stubs: failing tests for all Phase 6 behaviors (FOUND-01 through FOUND-04)
- [ ] 06-02-PLAN.md — v4 schema migration + dataclass extensions (db.py, models.py)
- [ ] 06-03-PLAN.md — Provenance capture + marker rate computation + report display (evaluate.py, report.py)
- [ ] 06-04-PLAN.md — CLI surface: benchmark calibrate subcommand + cost guardrail + provenance flags (cli.py)

### Phase 7: IAM Data Ingestion + Sweep Infrastructure
**Goal**: The developer can load the IAM Handwriting Database into the benchmark system and execute a full multi-strategy sweep against it, with per-writer variance visible in reports.
**Depends on**: Phase 6 (measurement foundation must exist before sweep runs are meaningful)
**Requirements**: IAM-01, IAM-02, IAM-03
**Success Criteria** (what must be TRUE when this phase completes):
  1. Developer runs `benchmark ingest-iam <path>` against the downloaded IAM ascii/ directory and the benchmark DB is populated with line images tagged `category="iam"` and `student="iam-writer-XXX"`, with no manual data wrangling needed.
  2. Developer runs `benchmark sweep` and the system executes all five strategies (baseline, self_correct, line_level, prompt_adapted, zoomed_verify) against the IAM test set, storing one run_id per strategy — without the developer manually invoking each strategy.
  3. `benchmark report` shows a per-writer CER breakdown table, allowing the developer to see whether a strategy's CER gain is consistent across writers or driven by a few easy writers.
**Plans**: TBD

### Phase 8: Statistics Layer
**Goal**: CER comparisons between strategies are statistically defensible — not just raw delta numbers — so the developer can assert with confidence that a measured improvement is real.
**Depends on**: Phase 7 (needs populated multi-strategy runs with n >= 10 samples)
**Requirements**: STAT-01, STAT-02
**Success Criteria** (what must be TRUE when this phase completes):
  1. Running `benchmark compare RUN_A RUN_B` on any two runs with n >= 10 samples automatically appends a Wilcoxon signed-rank p-value and Cohen's r effect size to the output, with no extra flags needed.
  2. The same `benchmark compare` output includes 95% bootstrap confidence intervals on both CER estimates, so the developer can see whether the CI bands overlap and judge whether the difference is distinguishable from sampling noise.
**Plans**: TBD

### Phase 9: Final Sweep, Recommendation, and Baseline Lock
**Goal**: The developer knows which strategy+provider configuration is best for lab notebook grading, and a regression baseline is pinned so any future code change that silently degrades accuracy is immediately detectable.
**Depends on**: Phase 8 (statistically-grounded run comparisons must be available before a recommendation is trustworthy)
**Requirements**: RPT-01, RPT-02, RPT-03
**Success Criteria** (what must be TRUE when this phase completes):
  1. Developer runs `benchmark set-baseline RUN_ID` to pin any run as the regression anchor; `detect_regressions()` then compares future runs against that pinned run (not the penultimate run), and the schema tracks the `is_baseline` flag durably across sessions.
  2. `benchmark recommend` outputs a single ranked recommendation with a composite score (70% CER / 15% cost / 15% stability) and the winning strategy+provider combination is unambiguous.
  3. Developer can run `benchmark ingest-lab` against real student lab notebook images and store ground-truth transcriptions via a guided annotation workflow, producing a production-distribution test set that is separate from IAM.
**Plans**: TBD

## Progress

| Phase | Milestone | Plans | Status | Shipped |
|-------|-----------|-------|--------|---------|
| 1. Self-Correction + Smart Escalation | v2.0 | 2/2 | ✅ Complete | 2026-04-09 |
| 2. Line-Level Segmentation | v2.0 | 1/1 | ✅ Complete | 2026-04-09 |
| 3. New Local Model Providers | v2.0 | 1/1 | ✅ Complete | 2026-04-09 |
| 4. Preprocessing + Writer Adaptation | v2.0 | 1/1 | ✅ Complete | 2026-04-09 |
| 5. Post-Processing + Benchmark Suite | v2.0 | 1/1 | ✅ Complete | 2026-04-09 |
| 6. Measurement Foundation | v3.0 | 4/4 | In progress | - |
| 7. IAM Data Ingestion + Sweep Infrastructure | v3.0 | 0/? | Not started | - |
| 8. Statistics Layer | v3.0 | 0/? | Not started | - |
| 9. Final Sweep, Recommendation, and Baseline Lock | v3.0 | 0/? | Not started | - |
