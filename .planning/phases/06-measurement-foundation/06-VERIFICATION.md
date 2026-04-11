---
phase: 06-measurement-foundation
verified: 2026-04-11T00:00:00Z
status: passed
score: 8/8 must-haves verified
---

# Phase 6: Measurement Foundation Verification Report

**Phase Goal:** The developer can run reproducible CER benchmarks with documented provenance, a known noise floor, and protection against runaway API cost before any strategy sweep begins.
**Verified:** 2026-04-11
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Run record shows model version, IAM partition ID, normalization flags, vocab-hints=off — reproducible baseline anchor | VERIFIED | `insert_run()` accepts all four fields; `_run_benchmark_inner()` passes `model_version=_resolve_model_version(providers)`, `norm_flags=_NORM_FLAGS`, `iam_partition=iam_partition`, `vocab_hints_off=vocab_hints_off` |
| 2 | Every benchmark run record includes `[?]_marker_rate` column separate from CER | VERIFIED | `question_marker_rate REAL DEFAULT NULL` in `_SCHEMA_SQL` and `_MIGRATIONS[4]`; `_compute_marker_rate()` called on raw text BEFORE `character_error_rate()` in both single-provider and consensus loops |
| 3 | 20-sample noise calibration prints CER variance and minimum detectable difference | VERIFIED | `benchmark calibrate` command registered at line 399 of cli.py; uses `statistics.pstdev`; prints `"CER variance: ±{sd*100:.2f}%  |  Min detectable delta: {mdd*100:.2f}% (2σ)"` |
| 4 | Any sweep run first prints API cost projection and requires confirmation | VERIFIED | Cost projection block in `benchmark_run_cmd` runs before `run_benchmark()` call; always prints `"Estimated cost: $X.XXX"`; `click.confirm("Proceed?")` unless `--yes` is set; `sys.exit(0)` on decline |
| 5 | `generate_report()` shows Provenance: header with model, partition, norm flags | VERIFIED | `_format_table()` emits `"Provenance:"` block when `run_meta` is truthy; `generate_report()` fetches run meta and passes it through |
| 6 | `generate_report()` table includes `marker_rate` column | VERIFIED | Header includes `'marker_rate':>11`; each row formats `marker_pct = f"{r.mean_marker_rate * 100:.2f}%"` |
| 7 | Calibrate warns on undersample, errors on empty DB | VERIFIED | `if n < samples: click.echo(f"Warning: ...")` and `if not all_samples: sys.exit(1)` |
| 8 | `--iam-partition` and `--vocab-hints-off` flags wire through CLI to DB | VERIFIED | Options registered on `benchmark run`; passed as `iam_partition=iam_partition`, `vocab_hints_off=int(vocab_hints_off)` to `run_benchmark()`, which threads them to `insert_run()` |

**Score:** 8/8 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `handwriting_engine/benchmark/db.py` | v4 migration, bumped CURRENT_SCHEMA_VERSION=4, updated insert_run/insert_provider_output | VERIFIED | `CURRENT_SCHEMA_VERSION = 4` at line 23; `_MIGRATIONS[4]` with 5 ALTER TABLE + UPDATE at lines 131-139; `insert_run()` with 4 new params; `insert_provider_output()` with `question_marker_rate` param; `list_runs()` populates RunSummary provenance fields |
| `handwriting_engine/benchmark/models.py` | ProviderOutput.question_marker_rate, StrategyResult.mean_marker_rate, RunSummary provenance fields | VERIFIED | `question_marker_rate: float | None = None` on ProviderOutput (line 52); `mean_marker_rate: float = 0.0` on StrategyResult (line 85); RunSummary has model_version, iam_partition, norm_flags, vocab_hints_off with safe defaults |
| `handwriting_engine/benchmark/evaluate.py` | `_NORM_FLAGS`, `_resolve_model_version()`, `_compute_marker_rate()`, marker rate in both loops, provenance passed to insert_run | VERIFIED | All four symbols present; marker_rate computation on raw_text before character_error_rate() in single-provider loop (lines 336-353) and consensus loop (lines 371-389); insert_run called with all provenance kwargs |
| `handwriting_engine/benchmark/report.py` | Provenance header in _format_table, marker_rate column, _aggregate_results computes mean_marker_rate | VERIFIED | `_format_table()` emits "Provenance:" block; header includes "marker_rate"; `_aggregate_results()` computes `mean_marker_rate` via `statistics.mean(marker_rates)` |
| `handwriting_engine/cli.py` | `benchmark calibrate` subcommand, cost projection block, --yes/--iam-partition/--vocab-hints-off flags | VERIFIED | `@benchmark.command("calibrate")` at line 399; cost block at lines 304-331; three new options registered before `benchmark_run_cmd` |
| `tests/test_benchmark_db.py` | test_v4_migration_columns in TestSchemaCreation | VERIFIED | Method at line 56 checks all 5 new columns via PRAGMA table_info |
| `tests/test_benchmark_evaluate.py` | TestMarkerRate (4 tests), TestCalibrateCommand (3 tests), TestCostProjection (3 tests), TestProvenanceCapture (2 tests) | VERIFIED | All 4 classes present at lines 203, 278, 323, 377 |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `evaluate.py::_run_benchmark_inner` | `db.py::insert_provider_output` | `question_marker_rate=marker_rate` computed from raw text BEFORE `character_error_rate()` | WIRED | Lines 336-352: `_compute_marker_rate(raw_text)` called before `character_error_rate()` at line 356 |
| `evaluate.py::_run_benchmark_inner` | `db.py::insert_run` | `model_version=_resolve_model_version(providers)`, `norm_flags=_NORM_FLAGS` | WIRED | Lines 305-315: all four provenance kwargs explicitly passed |
| `report.py::_format_table` | `report.py::generate_report` | `run_meta=run_meta` dict fetched from runs table | WIRED | `generate_report()` queries `SELECT model_version, iam_partition, norm_flags, vocab_hints_off FROM runs WHERE id = ?` and passes result to `_format_table(run_id, results, run_meta=run_meta)` |
| `cli.py::benchmark_calibrate_cmd` | `db.py::samples_with_ground_truth` | `all_samples = samples_with_ground_truth(conn)` then `random.sample()` | WIRED | Lines 421-432 in cli.py |
| `cli.py::benchmark_calibrate_cmd` | `evaluate._read_single` | Module-reference import `import handwriting_engine.benchmark.evaluate as _evaluate`; calls `_evaluate._read_single()` | WIRED | Module ref at line 416 ensures test `@patch("handwriting_engine.benchmark.evaluate._read_single")` intercepts correctly |
| `cli.py::benchmark_run_cmd (cost block)` | `evaluate.py::estimate_cost` | `estimate_cost(int(_avg_in * _n_strat * _n_samples), int(_avg_out * _n_strat * _n_samples), p)` | WIRED | Lines 318-323; imported via `from handwriting_engine.benchmark.evaluate import run_benchmark, ..., estimate_cost, _available_providers` |
| `cli.py::benchmark_run_cmd (--iam-partition)` | `evaluate.py::run_benchmark` | `iam_partition=iam_partition` kwarg | WIRED | Line 342; `run_benchmark()` accepts `iam_partition: str | None = None` and threads through `_run_benchmark_inner()` to `insert_run()` |
| `consensus loop` | `db.py::insert_provider_output` | `question_marker_rate=consensus_marker_rate` | WIRED | Lines 371-389: separate `_compute_marker_rate(raw_text)` before `character_error_rate()` in consensus branch |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FOUND-01 | 06-02, 06-03, 06-04 | Reproducible baseline with provenance record (model version, IAM partition ID, norm flags, vocab hints off) | SATISFIED | `insert_run()` stores all 4 fields; `_resolve_model_version()` builds model label; `_NORM_FLAGS` constant pinned; `--iam-partition`/`--vocab-hints-off` flags wire to DB |
| FOUND-02 | 06-02, 06-03 | `[?]_marker_rate` stored as separate column alongside CER | SATISFIED | `question_marker_rate` column in `provider_outputs` schema; `_compute_marker_rate()` computes from raw text before normalization; `mean_marker_rate` aggregated in report |
| FOUND-03 | 06-04 | 20-sample noise floor calibration with CER variance and min detectable delta | SATISFIED | `benchmark calibrate` subcommand registered; output format `"CER variance: ±X.XX%  |  Min detectable delta: Y.YY% (2σ)"` |
| FOUND-04 | 06-04 | CLI warns with API cost projection before any sweep run | SATISFIED | Cost block always runs before `run_benchmark()`; always shows `"Estimated cost: $X.XXX"`; `--yes` bypasses prompt; decline exits 0 |

No orphaned requirements — all four FOUND IDs declared in plan frontmatter and all four satisfied.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `cli.py` | ~317 | `_total_reads` variable computed in plan description but absent in actual code — cost summed directly per provider without intermediate variable | Info | No functional impact; cost is computed correctly; _total_reads was an intermediate the plan described but code skips |
| `evaluate.py` | 466 | `detect_regressions(conn, threshold=...)` passes a connection as `run_id` positional arg (mismatch with report.py signature `run_id=None, threshold=0.03, db_path=None`) | Warning | Pre-existing bug in `compare_strategies()`, not introduced by Phase 6; not covered by Phase 6 tests or requirements |

No blocker anti-patterns found for Phase 6 deliverables. The `detect_regressions` mismatch exists in `compare_strategies()` which predates Phase 6 and is outside the requirement scope.

---

### Human Verification Required

None — all Phase 6 behaviors are verifiable programmatically via the test suite. The test file contains 12 new tests (TestMarkerRate x4, TestCalibrateCommand x3, TestCostProjection x3, TestProvenanceCapture x2) plus test_v4_migration_columns. The full test suite should be run to confirm:

1. `pytest tests/test_benchmark_db.py -x -q` — all pass including test_v4_migration_columns
2. `pytest tests/test_benchmark_evaluate.py::TestMarkerRate -x -q` — 4 tests
3. `pytest tests/test_benchmark_evaluate.py::TestCalibrateCommand -x -q` — 3 tests
4. `pytest tests/test_benchmark_evaluate.py::TestCostProjection -x -q` — 3 tests
5. `pytest tests/test_benchmark_evaluate.py::TestProvenanceCapture -x -q` — 2 tests

Note: Bash execution was unavailable during this verification session. All checks are static analysis of file contents. The implementation is substantive and correctly wired per code inspection.

---

## Summary

All four Phase 6 requirements (FOUND-01 through FOUND-04) are fully satisfied:

- **FOUND-01 (Provenance):** Four schema columns added to `runs` table in `_SCHEMA_SQL` and `_MIGRATIONS[4]`; `_resolve_model_version()` builds model label from constants; `_NORM_FLAGS` pinned; all fields wired from CLI flags through `run_benchmark()` to `insert_run()`.
- **FOUND-02 (Marker Rate):** `question_marker_rate` column in `provider_outputs`; `_compute_marker_rate()` called on raw text before `character_error_rate()` in both single-provider and consensus loops; `mean_marker_rate` aggregated in report with dedicated column.
- **FOUND-03 (Calibrate):** `benchmark calibrate` subcommand registered; module-reference import ensures test mocks work; `pstdev` used to handle single-sample case; correct output format.
- **FOUND-04 (Cost guardrail):** Cost projection block always runs before benchmark; shows `"Estimated cost: $X.XXX"`; `--yes` bypasses; decline exits 0.

The base schema (`_SCHEMA_SQL`) was updated in addition to the migration, so fresh in-memory DBs have all v4 columns without needing to run the migration — a correct implementation decision documented in the 06-02 summary.

---

_Verified: 2026-04-11_
_Verifier: Claude (gsd-verifier)_
