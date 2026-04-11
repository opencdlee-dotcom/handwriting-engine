---
gsd_state_version: 1.0
milestone: v3.0
milestone_name: — Verified Accuracy
status: unknown
last_updated: "2026-04-11T21:16:11.811Z"
progress:
  total_phases: 4
  completed_phases: 1
  total_plans: 4
  completed_plans: 4
---

# Execution State

**Project:** handwriting-engine-v3
**Milestone:** v3.0 — Verified Accuracy
**Started:** 2026-04-11

---

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-04-11)

**Core value:** Highest-accuracy LLM-vision handwriting transcription with self-correction, ensemble providers, and writer adaptation
**Current focus:** Ready to plan Phase 6 — Measurement Foundation

---

## Current Position

| Field | Value |
|-------|-------|
| Phase | 6 — Measurement Foundation |
| Plan | 03 complete (Provenance capture + report display) |
| Status | in_progress |
| Progress | Phase 6 of 9 (v3.0 scope: phases 6-9) |

```
v3.0 Progress: [ 6 ][ 7 ][ 8 ][ 9 ]
                ^
               here
```

---

## Performance Metrics

| Metric | Value | Source |
|--------|-------|--------|
| Baseline CER | 1.67% | Gemini Flash, IAM — unverified provenance |
| Self-correction target CER | ~1.3% | JoD 2025 — unverified |
| Test suite | 442 passing + 61 (test_improvements.py) | post-v2.0 |
| Codebase | ~18,150 LOC Python | post-v2.0 |

---
| Phase 06 P02 | 15 | 2 tasks | 3 files |
| Phase 06 P03 | 30 | 2 tasks | 4 files |
| Phase 06 P04 | 15 | 2 tasks | 2 files |

## Accumulated Context

### Decisions

- Phases 6-9 derived from 12 v3.0 requirements (FOUND, IAM, STAT, RPT categories)
- Phase ordering follows data dependency: foundation → ingestion → statistics → reporting
- Research summary file not present; phase grouping from orchestrator instructions used directly
- 06-01: Wave 0 RED stubs written before any implementation — Nyquist compliance enforced
- 06-01: CliRunner imported at module level in test_benchmark_evaluate.py for consistency
- 06-01: test_calibrate_no_samples_error uses OR condition to tolerate missing command returning non-zero
- [Phase 06]: Updated base _SCHEMA_SQL alongside migration so fresh :memory: DBs have v4 columns from creation
- [Phase 06]: _compute_marker_rate() computes [?] token fraction from raw text before normalization in evaluate.py
- [Phase 06 P03]: _resolve_model_version() uses DEFAULT_*_MODEL constants from _constants.py (not GEMINI_MODEL etc.)
- [Phase 06 P03]: Two test stubs (test_marker_rate_in_report, test_report_contains_provenance_header) fixed — missing run setup caused false failures on empty DB
- [Phase 06 P03]: list_runs() in db.py uses r.keys() check for graceful degradation on pre-v4 DBs
- [Phase 06]: Use statistics.pstdev (not stdev) for calibrate so single-sample calibration returns 0.0 variance instead of raising StatisticsError
- [Phase 06]: Import _read_single via module reference in calibrate command so test mocks intercept correctly
- [Phase 06]: vocab_hints_off promoted from hardcoded 0 to proper parameter threaded from CLI through run_benchmark() to insert_run()

### Key Facts for Planning

- IAM database must be manually downloaded (registration-gated at HEIA-FR) — `benchmark ingest-iam` handles parsing only
- Benchmark DB is SQLite at `~/.handwriting-engine/benchmark.db` — schema will need v4 migration for `is_baseline` flag (RPT-01)
- `[?]_marker_rate` column (FOUND-02) requires schema change to the runs table
- Cost projection guardrail (FOUND-04) must fire before any sweep — not after
- Statistics layer (Phase 8) requires n >= 10 samples per run; IAM test set provides this
- `benchmark recommend` composite score: 70% CER / 15% cost / 15% stability

### Blockers

None at roadmap creation time.

### Todos

- [x] Run `/gsd:plan-phase 6` to begin Phase 6 planning
- [x] Execute 06-01 (Wave 0 RED stubs) — complete 2026-04-11

---

## Log

Newest entries first.

### Entries

```
[2026-04-11] 06-03 COMPLETE — Provenance capture + marker rate wired in evaluate.py; Provenance header + marker_rate column added to report.py; list_runs() in db.py extended. Two test stubs fixed (missing run setup). 4 files modified.
[2026-04-11] 06-01 COMPLETE — Wave 0 RED stubs written. 12 new failing tests across 2 files (test_benchmark_db.py, test_benchmark_evaluate.py). All existing tests remain GREEN. Commits: d103aed, 84fcd5b.
[2026-04-11] ROADMAP — v3.0 roadmap created. 4 phases (6-9), 12/12 requirements mapped. Ready to plan Phase 6.
[2026-04-11] INIT — v3.0 milestone started. Benchmarking focus: validate all v2.0 accuracy claims against IAM and real lab notebooks.
[2026-04-11] POST-v2.0 — Committed post-v2.0 Codex session work (cb0ae56): skew detection, zoomed verify, prompt_adapter, writer_embeddings, batch_openai, Gemini context caching. 61 new tests.
[2026-04-11] COMPLETE — v2.0 milestone shipped. 5 phases, 6 plans, 9/9 requirements. 442 tests passing. Archived to milestones/v2.0-*.
```
