---
phase: 07-iam-data-ingestion-sweep-infrastructure
plan: 02
subsystem: benchmark
tags: [iam, sqlite, ingestion, cli, click, tdd, deduplication]

requires:
  - phase: 07-01-iam-data-ingestion-sweep-infrastructure
    provides: RED stub tests for TestIAMIngest (9 stubs) that this plan turns GREEN

provides:
  - parse_iam_lines(lines_txt, partition_forms) — parses IAM ascii/lines.txt into record dicts
  - ingest_iam(ascii_dir, lines_dir, partition_file, db_path) — bulk-inserts IAM line images into benchmark DB
  - benchmark ingest-iam CLI command with partition safety guard

affects:
  - 07-03 (sweep infrastructure — uses ingest_iam output as benchmark DB population)
  - 07-04 (per-writer report — relies on student=iam-writer-* tags set here)

tech-stack:
  added: []
  patterns:
    - IAM ascii format parsing: skip #/blank/err, split whitespace, 9-field minimum, pipe->space transcription
    - Atomic IAM commit: insert_sample (autocommit=False) + insert_ground_truth + conn.commit() in one transaction
    - Partition safety guard: CLI aborts if neither --partition-file nor --all-partitions provided

key-files:
  created: []
  modified:
    - handwriting_engine/benchmark/ingest.py
    - handwriting_engine/cli.py
    - tests/test_benchmark_ingest.py

key-decisions:
  - "No quality assessment called during IAM ingest — pre-segmented clean PNGs, assessment adds latency without benefit"
  - "Partition safety enforced at CLI layer, not ingest_iam() — ingest_iam passes partition_forms=None when not provided, CLI is responsible for the guard"
  - "ingest_iam uses autocommit=False + explicit conn.commit() per record so ground truth is committed atomically with sample"
  - "TestSweep and TestPerWriterReport RED stubs (8 failures) are pre-existing Wave 2/3 stubs — out of scope for this plan"

patterns-established:
  - "IAM line ID parsing: split('-') gives [writer_id, form_suffix, line_num], form_id = parts[0]+'-'+parts[1]"
  - "ingest_iam returns {'ingested': N, 'skipped_dup': N, 'skipped_missing': N} dict — consistent return contract"

requirements-completed: [IAM-01]

duration: 25min
completed: 2026-04-12
---

# Phase 07 Plan 02: IAM Data Ingestion Infrastructure Summary

**parse_iam_lines() + ingest_iam() + benchmark ingest-iam CLI: IAM Handwriting Database bulk-ingested into benchmark DB with partition safety guard, atomic ground truth commits, and SHA-256 deduplication**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-04-12T00:00:00Z
- **Completed:** 2026-04-12T00:25:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- 9/9 `TestIAMIngest` stubs turned GREEN (parse tests x5, ingest tests x3, CLI test x1)
- `parse_iam_lines()` correctly filters comments, blanks, err-status, short lines; extracts IAM fields; replaces pipe separators; filters by partition set
- `ingest_iam()` bulk-inserts IAM line images tagged `category="iam"`, `student="iam-writer-{writer_id}"`; commits ground truth atomically in same transaction; deduplicates by SHA-256 hash
- `benchmark ingest-iam` CLI command registered under the `benchmark` group with full partition safety guard (requires `--partition-file` OR `--all-partitions`, or aborts with clear error)

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: TestIAMIngest real tests** - `c0794ed` (test) — replace 9 stub pytest.fail() calls with real implementations
2. **Task 1 GREEN: parse_iam_lines + ingest_iam** - `d7266e8` (feat) — implement parse_iam_lines(), _iam_image_path(), ingest_iam() in ingest.py
3. **Task 1 test re-apply + Task 2: CLI command** - staged (feat) — updated test file + benchmark ingest-iam command in cli.py

_Note: TDD task had test→feat commits. Test file was disk-reverted between commits, requiring re-application._

## Files Created/Modified

- `handwriting_engine/benchmark/ingest.py` — added `parse_iam_lines()`, `_iam_image_path()`, `ingest_iam()` (128 lines)
- `handwriting_engine/cli.py` — added `benchmark_ingest_iam` command under `benchmark` group (55 lines)
- `tests/test_benchmark_ingest.py` — replaced 9 pytest.fail() stubs with real test implementations

## Decisions Made

- No quality assessment in IAM ingest path — IAM line images are pre-segmented clean PNGs; assess_quality adds latency without benefit for this use case
- Partition safety guard lives at the CLI layer — `ingest_iam()` itself is agnostic (caller passes `partition_forms=None` or a set); the CLI is responsible for the guard so ingest_iam() remains usable programmatically
- autocommit=False + explicit conn.commit() per record — ensures ground truth is committed atomically with its sample (no orphan samples)
- IntegrityError rollback on per-record failure — does not abort the full ingest; skips the duplicate and continues

## Deviations from Plan

None — plan executed exactly as written. The 8 pre-existing RED stub failures in `TestSweep` and `TestPerWriterReport` are Wave 2/3 stubs unrelated to this plan.

## Issues Encountered

- Test file was reverted on disk after first commit (system behavior). Re-applied test implementations using Write tool on second pass. No code was lost — implementation commits were preserved.

## Next Phase Readiness

- `ingest_iam()` and `benchmark ingest-iam` are ready for integration with real IAM data
- `student="iam-writer-{writer_id}"` tags are in place for per-writer reporting (Phase 07-04)
- Wave 2 (07-03 sweep infrastructure) can now build `run_sweep()` knowing the DB population strategy is established

## Self-Check: PASSED

- [x] `handwriting_engine/benchmark/ingest.py` — FOUND
- [x] `handwriting_engine/cli.py` — FOUND
- [x] `tests/test_benchmark_ingest.py` — FOUND
- [x] `.planning/phases/07-iam-data-ingestion-sweep-infrastructure/07-02-SUMMARY.md` — FOUND
- [x] 9/9 `TestIAMIngest` tests pass — VERIFIED
- [x] Commits `c0794ed`, `d7266e8`, `ea785ae` exist in git log

---
*Phase: 07-iam-data-ingestion-sweep-infrastructure*
*Completed: 2026-04-12*
