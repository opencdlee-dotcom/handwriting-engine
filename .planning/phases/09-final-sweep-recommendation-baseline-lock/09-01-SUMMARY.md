---
phase: 09-final-sweep-recommendation-baseline-lock
plan: "01"
subsystem: benchmark
tags: [schema-migration, regression-detection, baseline, cli, tdd]

requires:
  - phase: 06
    provides: schema_version migration plumbing reused for v6
  - phase: 07
    provides: detect_regressions's run-history primitives (list_runs, get_run_results)

provides:
  - Schema v6: ALTER TABLE runs ADD COLUMN is_baseline INTEGER DEFAULT 0
  - benchmark/db.py:set_baseline(conn, run_id) — at-most-one invariant
  - benchmark/db.py:get_baseline_run_id(conn) -> int | None
  - benchmark/models.py:RunSummary.is_baseline field
  - benchmark/report.py:detect_regressions retargeted to pinned baseline
  - CLI: benchmark set-baseline RUN_ID

affects:
  - All subsequent regression checks (Phase 9 onward) — anchor is now the pinned run, not the immediately-preceding run
  - list_runs() consumers — RunSummary now carries is_baseline (0 or 1)

tech-stack:
  added: []
  patterns:
    - At-most-one invariant via UPDATE-clear + UPDATE-set (no partial-unique index needed for sqlite portability)
    - Fallback to runs[-2] when no baseline pinned, preserving pre-Phase-9 behavior on fresh DBs
    - Self-compare guard: when current run IS the baseline, fall through to runs[-2] to avoid no-op

key-files:
  created:
    - tests/test_benchmark_baseline.py
  modified:
    - handwriting_engine/benchmark/db.py
    - handwriting_engine/benchmark/models.py
    - handwriting_engine/benchmark/report.py
    - handwriting_engine/cli.py

key-decisions:
  - "Fallback to runs[-2] when no baseline pinned. Preserves pre-Phase-9 behavior so existing test fixtures keep working without retroactive baseline pins."
  - "Self-compare guard. If `detect_regressions(run_id=X)` is called and X is the pinned baseline, comparing against itself is a no-op — fall through to the prior run instead."
  - "Atomic UPDATE pattern, not partial unique index. SQLite supports the index but sqlite3 module behavior across versions is uneven; the UPDATE-clear + UPDATE-set pattern is portable and explicit."
  - "ValueError on unknown run_id. set_baseline refuses to silently no-op when the user passes a typo'd ID."

verification:
  unit_coverage:
    - 12 tests in tests/test_benchmark_baseline.py
  criterion_status:
    - "RPT-01: IMPLEMENTED. Schema v6 column durable across reopen verified."
  pre_existing_failures:
    - tests/test_enhance.py (cv2 missing)
    - tests/test_trained_correction.py::TestConfidenceGate (transformers/torch missing)
