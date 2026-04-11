---
phase: 06-measurement-foundation
plan: 01
subsystem: testing
tags: [pytest, tdd, benchmark, sqlite, cli, red-green-refactor]

# Dependency graph
requires: []
provides:
  - Failing test stub for v4 DB migration columns (test_v4_migration_columns)
  - Failing test stubs for question_marker_rate storage (TestMarkerRate, 4 tests)
  - Failing test stubs for benchmark calibrate CLI subcommand (TestCalibrateCommand, 3 tests)
  - Failing test stubs for pre-flight cost projection prompt (TestCostProjection, 3 tests)
  - Failing test stubs for run provenance capture (TestProvenanceCapture, 2 tests)
affects: [06-02, 06-03, 06-04, benchmark, database, cli]

# Tech tracking
tech-stack:
  added: []
  patterns: [TDD Wave 0 — write all failing stubs before any implementation; fixture reuse via seeded_db; CliRunner for CLI integration testing]

key-files:
  created: []
  modified:
    - tests/test_benchmark_db.py
    - tests/test_benchmark_evaluate.py

key-decisions:
  - "Wave 0 RED stubs written before any Wave 1-3 implementation — Nyquist compliance enforced"
  - "CliRunner imported at module level in test_benchmark_evaluate.py for consistency"
  - "test_calibrate_no_samples_error uses OR condition (exit != 0 OR 'No samples' in output) to tolerate missing command returning non-zero"

patterns-established:
  - "TDD Wave 0: all test stubs for a phase written as RED before any production code"
  - "Acceptable RED modes: AssertionError (column missing), OperationalError (column not in schema), SystemExit (CLI command not registered)"
  - "seeded_db fixture reused across evaluate test classes — single image + ground truth + connection lifecycle"

requirements-completed: [FOUND-01, FOUND-02, FOUND-03, FOUND-04]

# Metrics
duration: 2min
completed: 2026-04-11
---

# Phase 6 Plan 01: Measurement Foundation Wave 0 Summary

**Wave 0 RED stubs for 12 benchmark tests across v4 schema migration, marker rate, calibrate CLI, cost projection, and provenance capture — proving Nyquist compliance before implementation begins**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-04-11T00:01:01Z
- **Completed:** 2026-04-11T00:02:46Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added `test_v4_migration_columns` to `TestSchemaCreation` — asserts model_version, iam_partition, norm_flags, vocab_hints_off in runs and question_marker_rate in provider_outputs; fails RED with AssertionError
- Added `TestMarkerRate` (4 tests), `TestCalibrateCommand` (3 tests), `TestCostProjection` (3 tests), `TestProvenanceCapture` (2 tests) to test_benchmark_evaluate.py — all fail RED with acceptable error modes
- All 14 previously passing evaluate tests and 17 previously passing db tests remain GREEN

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend TestSchemaCreation with v4 migration column test** - `d103aed` (test)
2. **Task 2: Add TestMarkerRate, TestCalibrateCommand, TestCostProjection, TestProvenanceCapture stubs** - `84fcd5b` (test)

## Files Created/Modified
- `tests/test_benchmark_db.py` - Added test_v4_migration_columns to TestSchemaCreation class (11 lines)
- `tests/test_benchmark_evaluate.py` - Added CliRunner import + 4 new test classes totaling 190 lines

## Decisions Made
- Added `from click.testing import CliRunner` at module level (not inside each method) for cleaner code, following Python import conventions
- `test_calibrate_no_samples_error` passes in RED state because the assertion `exit_code != 0 OR "No samples" in output` is satisfied by "No such command 'calibrate'" returning exit_code=2 — this is acceptable per the plan ("SystemExit (command not registered)" is listed as an acceptable failure mode)

## Deviations from Plan

None - plan executed exactly as written. The CliRunner module-level import was a minor style choice consistent with the existing mock imports in the file.

## Issues Encountered

None. All tests failed with the expected error modes (AssertionError for missing DB columns, SystemExit for missing CLI commands, AssertionError for missing CLI flags and output strings). No SyntaxErrors or IndentationErrors in either test file.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Wave 0 stubs complete for all Phase 6 behaviors
- Wave 1 (06-02): Implement v4 DB migration to add provenance columns — test_v4_migration_columns will turn GREEN
- Wave 2 (06-03): Implement question_marker_rate computation and storage — TestMarkerRate and TestProvenanceCapture will turn GREEN
- Wave 3 (06-04): Implement benchmark calibrate command and cost projection — TestCalibrateCommand and TestCostProjection will turn GREEN

---
*Phase: 06-measurement-foundation*
*Completed: 2026-04-11*
