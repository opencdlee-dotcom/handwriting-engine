---
phase: 07-iam-data-ingestion-sweep-infrastructure
plan: 04
subsystem: benchmark
tags: [iam, report, per-writer, click, tdd]

requires:
  - phase: 07-01-iam-data-ingestion-sweep-infrastructure
    provides: RED stub tests for TestPerWriterReport (3 stubs) that this plan turns GREEN
  - phase: 07-02-iam-data-ingestion-sweep-infrastructure
    provides: ingest_iam() populating samples.student='iam-writer-XXX' rows that the SQL groups by

provides:
  - generate_per_writer_report(run_id, db_path) — formatted per-writer CER table (Writer, Mean CER, Min CER, Max CER, N)
  - benchmark report --per-writer flag wiring

affects:
  - Sweep interpretation — reveals whether a strategy CER gain is consistent across writers or driven by a few easy ones
  - 07-03 sweep run_ids — now reportable per writer

tech-stack:
  added: []
  patterns:
    - SQL aggregation: AVG/MIN/MAX(em.cer) GROUP BY s.student (excluding empty student)
    - Empty-state handling: explicit message string rather than empty table when no writer-tagged rows exist

key-files:
  created: []
  modified:
    - handwriting_engine/benchmark/report.py
    - handwriting_engine/cli.py
    - tests/test_benchmark_evaluate.py

key-decisions:
  - "SQL filter excludes student IS NULL OR student='' — non-IAM samples have no writer data and would pollute the table"
  - "Empty-state message includes 'No writer data' (test contract) and a tip pointing to `benchmark ingest-iam`"
  - "Sorted by mean_cer DESC — hardest writers first (most actionable view)"
  - "--per-writer branches early in benchmark_report_cmd; existing report logic untouched when flag absent"

patterns-established:
  - "Per-writer SQL: provider_outputs JOIN eval_metrics JOIN samples, GROUP BY s.student"

requirements-completed: [IAM-03]

duration: ~10min
completed: 2026-05-06
---

# Phase 07 Plan 04: Per-Writer Report Summary

**generate_per_writer_report() + benchmark report --per-writer: per-writer CER breakdown for any run, revealing whether a strategy's gain is consistent across writers. Turns 3 TestPerWriterReport RED stubs GREEN.**

## Accomplishments

- 3/3 `TestPerWriterReport` stubs turned GREEN
- `generate_per_writer_report()` exported from `handwriting_engine.benchmark.report`
- Per-writer table: `Writer | Mean CER | Min CER | Max CER | N`, sorted hardest-first
- Empty-writer case returns explanatory message (not crash, not empty)
- `benchmark report --per-writer` flag wired up; visible in `--help`
- Hidden `--db-path` option added to report command for testability

## Verification

```bash
pytest tests/test_benchmark_evaluate.py::TestPerWriterReport -q   # 3 passed
pytest tests/test_benchmark_evaluate.py -q                         # 34 passed
pytest tests/ -q --ignore=tests/test_iam_real_data.py              # 525 passed, 2 skipped, 1 xfailed
python3 -c "from handwriting_engine.benchmark.report import generate_per_writer_report; print('OK')"
python3 -m handwriting_engine.cli benchmark report --help | grep per-writer
```

## Files Modified

- `handwriting_engine/benchmark/report.py` — added `generate_per_writer_report()` (~60 lines)
- `handwriting_engine/cli.py` — added `--per-writer` flag + early-return branch on `benchmark_report_cmd`
- `tests/test_benchmark_evaluate.py` — replaced 3 `pytest.fail()` stubs with real assertions (writer-grouping, empty-state, CLI flag)

## Phase 07 Now Complete

All 4 plans landed. Phase 07 (IAM Data Ingestion + Sweep Infrastructure) ships IAM-01, IAM-02, IAM-03.

**What this unlocks:**
1. User can download IAM, run `benchmark ingest-iam`, then `benchmark sweep` to populate the DB with one run_id per strategy.
2. `benchmark report --per-writer` immediately shows whether a strategy's gain is consistent across writers.
3. The sweep run outputs become real-data training pairs for `trained_correction.dataset.from_benchmark_db()` — the v2 corrector retrain that resolves the synthetic-only hallucination failure mode.

**Next phase:** Phase 08 (Statistics Layer) — Wilcoxon p-values + bootstrap CIs on `benchmark compare`. Out of scope for this session.
