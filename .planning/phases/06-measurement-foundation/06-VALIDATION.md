---
phase: 6
slug: measurement-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-11
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >= 8.0.0 |
| **Config file** | none — discovered via `tests/` directory |
| **Quick run command** | `pytest tests/test_benchmark_db.py tests/test_benchmark_evaluate.py -x -q` |
| **Full suite command** | `pytest tests/ -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_benchmark_db.py tests/test_benchmark_evaluate.py -x -q`
- **After every plan wave:** Run `pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| v4-migration | 01 | 1 | FOUND-01, FOUND-02 | unit | `pytest tests/test_benchmark_db.py::TestSchemaCreation -x -q` | ❌ W0 | ⬜ pending |
| provenance-insert | 01 | 1 | FOUND-01 | unit | `pytest tests/test_benchmark_db.py -k "provenance" -x -q` | ❌ W0 | ⬜ pending |
| list-runs-provenance | 01 | 1 | FOUND-01 | unit | `pytest tests/test_benchmark_db.py -k "list_runs_provenance" -x -q` | ❌ W0 | ⬜ pending |
| report-provenance | 01 | 1 | FOUND-01 | unit | `pytest tests/test_benchmark_evaluate.py -k "report_provenance" -x -q` | ❌ W0 | ⬜ pending |
| marker-rate-store | 02 | 1 | FOUND-02 | unit | `pytest tests/test_benchmark_evaluate.py -k "marker_rate" -x -q` | ❌ W0 | ⬜ pending |
| marker-rate-clean | 02 | 1 | FOUND-02 | unit | `pytest tests/test_benchmark_evaluate.py -k "marker_rate_clean" -x -q` | ❌ W0 | ⬜ pending |
| marker-rate-before-norm | 02 | 1 | FOUND-02 | unit | `pytest tests/test_benchmark_evaluate.py -k "marker_rate_before_norm" -x -q` | ❌ W0 | ⬜ pending |
| report-marker-col | 02 | 1 | FOUND-02 | unit | `pytest tests/test_benchmark_evaluate.py -k "report_marker_rate" -x -q` | ❌ W0 | ⬜ pending |
| calibrate-cmd | 03 | 2 | FOUND-03 | unit | `pytest tests/test_benchmark_evaluate.py -k "calibrate" -x -q` | ❌ W0 | ⬜ pending |
| calibrate-format | 03 | 2 | FOUND-03 | unit | `pytest tests/test_benchmark_evaluate.py -k "calibrate_format" -x -q` | ❌ W0 | ⬜ pending |
| calibrate-undersample | 03 | 2 | FOUND-03 | unit | `pytest tests/test_benchmark_evaluate.py -k "calibrate_undersample" -x -q` | ❌ W0 | ⬜ pending |
| cost-projection | 04 | 2 | FOUND-04 | unit | `pytest tests/test_benchmark_evaluate.py -k "cost_projection" -x -q` | ❌ W0 | ⬜ pending |
| cost-yes-bypass | 04 | 2 | FOUND-04 | unit | `pytest tests/test_benchmark_evaluate.py -k "cost_yes_bypass" -x -q` | ❌ W0 | ⬜ pending |
| cost-decline | 04 | 2 | FOUND-04 | unit | `pytest tests/test_benchmark_evaluate.py -k "cost_decline" -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_benchmark_db.py` — extend `TestSchemaCreation` with `test_v4_migration_columns` covering both `runs` (4 provenance cols) and `provider_outputs` (`question_marker_rate`) new columns
- [ ] `tests/test_benchmark_evaluate.py` — add `TestMarkerRate` class: rate from raw text, rate=0 for clean output, rate stored in DB, rate in report output
- [ ] `tests/test_benchmark_evaluate.py` — add `TestCalibrateCommand` class (Click `CliRunner`): output format, undersample warning, no-samples error
- [ ] `tests/test_benchmark_evaluate.py` — add `TestCostProjection` class: always-shown, `--yes` bypass, decline exits cleanly
- [ ] `tests/test_benchmark_evaluate.py` — add `TestProvenanceCapture` class: provenance columns in DB after run, report header contains provenance block

All use `CliRunner` from Click for CLI tests and `:memory:` SQLite for DB tests — both patterns already established in `tests/conftest.py`.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Provenance columns visible in `benchmark list --runs` terminal output | FOUND-01 | Requires a real terminal TTY to verify table formatting | Run `handwriting-engine benchmark list --runs` after a real run and confirm model_version + iam_partition columns present |
| Cost prompt interactive confirmation | FOUND-04 | stdin interaction can't be fully automated with CliRunner | Run `handwriting-engine benchmark run --providers gemini --strategies single` and confirm prompt appears; then test `--yes` skips it |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
