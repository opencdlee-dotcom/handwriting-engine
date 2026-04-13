---
phase: 7
slug: iam-data-ingestion-sweep-infrastructure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-12
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >= 8.0.0 |
| **Config file** | none — discovered via `tests/` directory |
| **Quick run command** | `pytest tests/test_benchmark_ingest.py tests/test_benchmark_evaluate.py -x -q` |
| **Full suite command** | `pytest tests/ -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_benchmark_ingest.py tests/test_benchmark_evaluate.py -x -q`
- **After every plan wave:** Run `pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| wave0-stubs | 01 | 0 | IAM-01, IAM-02, IAM-03 | unit | `pytest tests/test_benchmark_ingest.py tests/test_benchmark_evaluate.py -x -q` | ❌ W0 | ⬜ pending |
| parse-iam-lines | 02 | 1 | IAM-01 | unit | `pytest tests/test_benchmark_ingest.py::TestIAMIngest -x -q` | ❌ W0 | ⬜ pending |
| ingest-iam-fn | 02 | 1 | IAM-01 | unit | `pytest tests/test_benchmark_ingest.py::TestIAMIngest::test_ingest_sets_category_and_student tests/test_benchmark_ingest.py::TestIAMIngest::test_ingest_inserts_ground_truth tests/test_benchmark_ingest.py::TestIAMIngest::test_ingest_iam_dedup -x -q` | ❌ W0 | ⬜ pending |
| ingest-iam-cli | 02 | 1 | IAM-01 | unit | `pytest tests/test_benchmark_ingest.py::TestIAMIngest::test_cli_ingest_iam_command -x -q` | ❌ W0 | ⬜ pending |
| thread-line-level | 03 | 2 | IAM-02 | unit | `pytest tests/test_benchmark_evaluate.py::TestSweep::test_run_benchmark_accepts_line_level -x -q` | ❌ W0 | ⬜ pending |
| thread-auto-retry | 03 | 2 | IAM-02 | unit | `pytest tests/test_benchmark_evaluate.py::TestSweep::test_run_benchmark_accepts_auto_retry -x -q` | ❌ W0 | ⬜ pending |
| run-sweep-fn | 03 | 2 | IAM-02 | unit | `pytest tests/test_benchmark_evaluate.py::TestSweep::test_run_sweep_returns_five_run_ids -x -q` | ❌ W0 | ⬜ pending |
| sweep-cli | 03 | 2 | IAM-02 | unit | `pytest tests/test_benchmark_evaluate.py::TestSweep::test_sweep_cli_shows_cost tests/test_benchmark_evaluate.py::TestSweep::test_sweep_cli_yes_executes -x -q` | ❌ W0 | ⬜ pending |
| per-writer-report | 04 | 2 | IAM-03 | unit | `pytest tests/test_benchmark_evaluate.py::TestPerWriterReport -x -q` | ❌ W0 | ⬜ pending |
| report-cli-flag | 04 | 2 | IAM-03 | unit | `pytest tests/test_benchmark_evaluate.py::TestPerWriterReport::test_report_cli_per_writer_flag -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_benchmark_ingest.py` — add `TestIAMIngest` class with 9 failing stubs covering IAM-01
  - `test_parse_skips_comments`
  - `test_parse_filters_err`
  - `test_parse_extracts_fields`
  - `test_parse_replaces_pipes`
  - `test_parse_filters_partition`
  - `test_ingest_sets_category_and_student`
  - `test_ingest_inserts_ground_truth`
  - `test_ingest_iam_dedup`
  - `test_cli_ingest_iam_command`
- [ ] `tests/test_benchmark_evaluate.py` — add `TestSweep` class with 5 failing stubs covering IAM-02
  - `test_run_benchmark_accepts_line_level`
  - `test_run_benchmark_accepts_auto_retry`
  - `test_run_sweep_returns_five_run_ids`
  - `test_sweep_cli_shows_cost`
  - `test_sweep_cli_yes_executes`
- [ ] `tests/test_benchmark_evaluate.py` — add `TestPerWriterReport` class with 3 failing stubs covering IAM-03
  - `test_per_writer_report_groups_by_student`
  - `test_per_writer_report_no_writers`
  - `test_report_cli_per_writer_flag`

All stubs use `pytest.fail("not implemented")` or `assert False` as the body.
Existing test infrastructure (`:memory:` SQLite, `CliRunner`) applies — no new fixtures needed.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `benchmark ingest-iam /path/to/ascii` against real IAM download | IAM-01 | Requires registration-gated IAM database (HEIA-FR); cannot be automated in CI | Download IAM from HEIA-FR, run `handwriting-engine benchmark ingest-iam ./ascii --partition-file testset.txt`, confirm summary shows ingested count > 0 and DB has rows with `category='iam'` |
| `benchmark sweep` end-to-end with real API calls | IAM-02 | Makes live API calls (Gemini); cost guardrail must fire in real TTY | Run `handwriting-engine benchmark sweep --provider gemini` (without `--yes`), confirm cost projection shown, then `--yes` executes all 5 strategies |
| Per-writer report on real IAM run | IAM-03 | Requires real IAM data in DB | After sweep, run `handwriting-engine benchmark report --per-writer` and confirm multiple writers (iam-writer-a01, etc.) appear with distinct CER values |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
