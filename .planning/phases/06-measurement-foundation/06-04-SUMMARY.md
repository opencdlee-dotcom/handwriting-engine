---
phase: 06-measurement-foundation
plan: "04"
subsystem: benchmark
tags: [cli, calibrate, cost-projection, provenance, tdd]
dependency_graph:
  requires: [06-03]
  provides: [benchmark-calibrate-cmd, cost-projection-guard, iam-partition-flag, vocab-hints-off-flag, yes-flag]
  affects: [cli.py, evaluate.py, test_benchmark_evaluate.py]
tech_stack:
  added: []
  patterns: [pre-flight-cost-guardrail, pstdev-for-single-sample-calibration, module-ref-import-for-mock-interception]
key_files:
  created: []
  modified:
    - handwriting_engine/cli.py
    - handwriting_engine/benchmark/evaluate.py
decisions:
  - "Use statistics.pstdev (not stdev) so calibrate works with a single sample — pstdev([x]) = 0.0 rather than raising StatisticsError"
  - "Import _read_single via module reference (import handwriting_engine.benchmark.evaluate as _evaluate; call _evaluate._read_single) so test @patch on the module namespace intercepts correctly"
  - "Add --db-path hidden option to benchmark run so tests can pass seeded_db without affecting production usage"
  - "vocab_hints_off promoted from hardcoded 0 in _run_benchmark_inner() to a proper parameter threaded from CLI through run_benchmark() to insert_run()"
metrics:
  duration_minutes: 15
  completed_date: "2026-04-11"
  tasks_completed: 2
  files_modified: 2
requirements: [FOUND-01, FOUND-03, FOUND-04]
---

# Phase 6 Plan 04: CLI Surface — Calibrate + Cost Projection + Provenance Flags Summary

**One-liner:** Add benchmark calibrate subcommand (CER variance + min detectable delta using pstdev), pre-flight cost projection guardrail with --yes bypass and graceful decline (exit 0), and --iam-partition / --vocab-hints-off flags wired through to run_benchmark() and the DB.

## What Was Built

### cli.py changes

1. **`benchmark calibrate` subcommand** — registered as `@benchmark.command("calibrate")` with:
   - `--samples N` (default 20): number of random samples to evaluate
   - `--provider P` (default "gemini"): which provider to use for calibration reads
   - `--db-path` (hidden): override DB path for testing
   - Uses `statistics.pstdev` to compute standard deviation (works with 1+ values, returns 0.0 for single sample)
   - Fetches samples via `samples_with_ground_truth()`, randomly selects N with `random.sample()`
   - Calls `_evaluate._read_single()` (via module reference for mock interception)
   - Computes CER against ground truth via `character_error_rate()`
   - Prints: `"CER variance: ±{sd*100:.2f}%  |  Min detectable delta: {mdd*100:.2f}% (2σ)"`
   - Warns when fewer samples available than requested (continues with available count)
   - Exits non-zero with error message when DB has no ground truth samples

2. **`_get_avg_tokens_per_read(conn)` helper** — queries most recent run's average input/output tokens from provider_outputs table. Falls back to (2000.0, 500.0) if no prior runs or any exception.

3. **`benchmark run` extended with new options:**
   - `--yes / -y`: skip cost confirmation prompt (CI-friendly)
   - `--iam-partition TEXT`: IAM partition label for provenance
   - `--vocab-hints-off`: flag to record vocabulary hints disabled
   - `--db-path` (hidden): override DB path for testing

4. **Pre-flight cost projection block** in `benchmark_run_cmd`:
   - Runs BEFORE `run_benchmark()`, after `compare_strategies` early-return
   - Fetches sample count and avg token rates from DB
   - Computes total cost per provider and sums across all providers
   - Always prints: `"Estimated cost: $X.XXX"` followed by provider/strategy/sample breakdown
   - Shows `"Proceed? [y/N]"` prompt unless `--yes` is set
   - Declining exits with `sys.exit(0)` — graceful, no error code

5. **`run_benchmark()` call updated** — passes `iam_partition=iam_partition`, `vocab_hints_off=int(vocab_hints_off)`, `db_path=db_path`.

6. **`generate_report()` call updated** — passes `db_path=db_path` for consistency with test isolation.

### evaluate.py changes

1. **`run_benchmark()` signature extended** — added `vocab_hints_off: int = 0` parameter (backward-compatible default).

2. **`_run_benchmark_inner()` signature extended** — same `vocab_hints_off: int = 0` parameter added and threaded through to `insert_run()`.

3. **`insert_run()` call updated** — replaces the previously hardcoded `vocab_hints_off = 0 if vocabulary_hints else 0` with the actual parameter value.

## Decisions Made

1. **`statistics.pstdev` not `stdev`** — `test_calibrate_output_format` uses `--samples 1` with 1 sample in the seeded DB, expecting the format output to succeed. `stdev` raises `StatisticsError` with < 2 values. `pstdev` returns 0.0 for a single value, producing valid output `"CER variance: ±0.00%  |  Min detectable delta: 0.00% (2σ)"` which matches the regex pattern.

2. **Module-reference import for `_read_single`** — The test patches `handwriting_engine.benchmark.evaluate._read_single`. If the calibrate command imports with `from handwriting_engine.benchmark.evaluate import _read_single`, the local binding isn't patched. Using `import handwriting_engine.benchmark.evaluate as _evaluate` and calling `_evaluate._read_single()` ensures the mock intercepts correctly.

3. **`--db-path` on `benchmark run`** — Tests pass `--db-path str(seeded_db)` to `benchmark run` invocations. This flag was previously absent from the command. Added as a hidden option to avoid cluttering `--help` output while enabling test isolation.

4. **`vocab_hints_off` promoted from hardcoded** — Previously `_run_benchmark_inner()` hardcoded `vocab_hints_off = 0 if vocabulary_hints else 0`. Replaced with a proper parameter so the CLI's `--vocab-hints-off` flag actually propagates to the DB.

## Deviations from Plan

None — plan executed exactly as written. The only implementation-level decision was using `pstdev` instead of `stdev` (implied by the test's `--samples 1` assertion), which was the correct interpretation of the plan's intent.

## Success Criteria Verification

- [x] `benchmark calibrate` subcommand registered in the CLI
- [x] Calibrate output format: `"CER variance: ±X.XX%  |  Min detectable delta: Y.YY% (2σ)"` — verified by TestCalibrateCommand::test_calibrate_output_format (PASS)
- [x] Calibrate warns when fewer samples available than requested; does not abort — verified by test_calibrate_undersample_warning (PASS)
- [x] Calibrate exits non-zero with error message when DB has no ground truth samples — verified by test_calibrate_no_samples_error (PASS)
- [x] `"Estimated cost: $X.XXX"` appears in benchmark run output before any API calls — verified by TestCostProjection::test_cost_always_shown (PASS)
- [x] `"Proceed? [y/N]"` prompt shown unless --yes is passed — verified by test_yes_bypasses_prompt (PASS)
- [x] Declining prompt exits 0 without running benchmark — verified by test_decline_exits_cleanly (PASS)
- [x] `pytest tests/test_benchmark_db.py tests/test_benchmark_evaluate.py -q` — 44 passed, 0 failed

## Self-Check: PASSED

Files verified present:
- FOUND: handwriting_engine/cli.py (benchmark_calibrate_cmd function present)
- FOUND: handwriting_engine/benchmark/evaluate.py (vocab_hints_off parameter present)

Commit verified: 937c24c — feat(06-04): add benchmark calibrate subcommand and cost projection guard
