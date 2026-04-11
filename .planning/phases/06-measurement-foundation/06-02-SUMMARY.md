---
phase: 06-measurement-foundation
plan: 02
subsystem: benchmark
tags: [schema-migration, sqlite, dataclasses, marker-rate, provenance]
dependency_graph:
  requires: [06-01]
  provides: [v4-schema, ProviderOutput.question_marker_rate, StrategyResult.mean_marker_rate, RunSummary-provenance-fields]
  affects: [06-03, 06-04, 06-05]
tech_stack:
  added: []
  patterns: [SQLite ALTER TABLE migration, dataclass field extension with safe defaults]
key_files:
  created: []
  modified:
    - handwriting_engine/benchmark/db.py
    - handwriting_engine/benchmark/models.py
    - handwriting_engine/benchmark/evaluate.py
decisions:
  - "Updated base _SCHEMA_SQL (not just migration) so fresh :memory: DBs have v4 columns without needing to run migration"
  - "_compute_marker_rate() counts [?] tokens from raw text before normalization to avoid false 0.0 rates"
metrics:
  duration: ~15min
  completed: 2026-04-11
  tasks_completed: 2
  files_modified: 3
---

# Phase 6 Plan 02: v4 Schema Migration and Model Extension Summary

SQLite v4 migration with 5 new columns (runs: model_version, iam_partition, norm_flags, vocab_hints_off; provider_outputs: question_marker_rate) plus dataclass field additions to ProviderOutput, StrategyResult, and RunSummary.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add _MIGRATIONS[4] and bump CURRENT_SCHEMA_VERSION in db.py | daf0940 | handwriting_engine/benchmark/db.py |
| 2 | Extend ProviderOutput and StrategyResult dataclasses in models.py | f11f376 | handwriting_engine/benchmark/models.py, evaluate.py |

## Verification

- `pytest tests/test_benchmark_db.py -x -q` — 18 passed (includes test_v4_migration_columns GREEN)
- `pytest tests/test_benchmark_db.py tests/test_benchmark_evaluate.py::TestEstimateCost tests/test_benchmark_evaluate.py::TestRunBenchmark -x -q` — 25 passed
- All 32 pre-06-01 benchmark tests pass (18 db + 14 evaluate)
- 3 of 4 TestMarkerRate tests now GREEN (test_marker_rate_in_report deferred to report.py plan)

## Decisions Made

1. **Base schema updated alongside migration**: For fresh `:memory:` DBs, `ensure_schema()` seeds `schema_version` at `CURRENT_SCHEMA_VERSION=4`. Since `_apply_migrations()` only runs `version > current`, migration 4 would never run on fresh DBs. Solution: added the new columns to `_SCHEMA_SQL` directly so fresh DBs have them from creation. Migration 4 still runs for existing v3 DBs.

2. **_compute_marker_rate in evaluate.py**: The `question_marker_rate` field in db/models is only useful if evaluate.py actually populates it. The TestMarkerRate tests verify this end-to-end. The helper counts `[?]` tokens from raw text BEFORE normalization (critical: normalization strips `[?]` markers).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical Functionality] Added _compute_marker_rate() to evaluate.py**
- **Found during:** Task 2 verification
- **Issue:** `insert_provider_output()` now accepts `question_marker_rate` but `evaluate.py` never computed or passed it, leaving the column NULL for all rows. Three TestMarkerRate tests would fail.
- **Fix:** Added `_compute_marker_rate()` helper that splits raw text on whitespace and counts tokens matching `\[\?\]`. Called from `_run_benchmark_inner()` on the raw result text before `insert_provider_output()`.
- **Files modified:** handwriting_engine/benchmark/evaluate.py
- **Commit:** f11f376

## Self-Check

Files exist:
- [x] handwriting_engine/benchmark/db.py — FOUND
- [x] handwriting_engine/benchmark/models.py — FOUND
- [x] handwriting_engine/benchmark/evaluate.py — FOUND

Commits exist:
- [x] daf0940 — FOUND (feat(06-02): add _MIGRATIONS[4] and bump CURRENT_SCHEMA_VERSION to 4)
- [x] f11f376 — FOUND (feat(06-02): extend data models and add marker rate computation in evaluate.py)

Content verified:
- [x] CURRENT_SCHEMA_VERSION = 4 in db.py
- [x] _MIGRATIONS[4] contains 5 ALTER TABLE + UPDATE schema_version
- [x] ProviderOutput.question_marker_rate: float | None = None
- [x] StrategyResult.mean_marker_rate: float = 0.0
- [x] RunSummary provenance fields with defaults
- [x] _compute_marker_rate() in evaluate.py

## Self-Check: PASSED
