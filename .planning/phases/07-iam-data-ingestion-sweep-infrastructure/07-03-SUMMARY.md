---
phase: 07-iam-data-ingestion-sweep-infrastructure
plan: 03
subsystem: benchmark
tags: [iam, sweep, click, tdd, line-level, auto-retry]

requires:
  - phase: 07-01-iam-data-ingestion-sweep-infrastructure
    provides: RED stub tests for TestSweep (5 stubs) that this plan turns GREEN
  - phase: 07-02-iam-data-ingestion-sweep-infrastructure
    provides: ingest_iam() populating samples.category='iam' rows that run_sweep() filters on

provides:
  - SWEEP_STRATEGIES — 5-strategy config list (baseline, self_correct, line_level, prompt_adapted, zoomed_verify)
  - run_sweep(provider, db_path, yes, on_progress) — executes all 5 strategies, returns {name: run_id} dict
  - run_benchmark line_level/auto_retry parameters (threaded through _run_benchmark_inner -> _read_single -> read_page)
  - benchmark sweep CLI command with cost projection guardrail

affects:
  - 07-04 (per-writer report — consumes sweep run_ids to break down CER per writer)
  - trained_correction (real-data retrain — uses sweep outputs as (vlm_text, ground_truth) training pairs)

tech-stack:
  added: []
  patterns:
    - Strategy config table with kwargs dict — keeps run_sweep() body small, easy to extend
    - IAM filter via samples.category='iam' (not a new param) — matches plan's must_have truth
    - Cost projection at CLI layer; run_sweep() itself does not prompt — keeps library callable from non-CLI contexts

key-files:
  created: []
  modified:
    - handwriting_engine/benchmark/evaluate.py
    - handwriting_engine/cli.py
    - tests/test_benchmark_evaluate.py

key-decisions:
  - "line_level and auto_retry threaded through the full call chain (_read_single signature, _run_benchmark_inner signature, run_benchmark signature) — backward-compatible defaults of False everywhere"
  - "prompt_adapted strategy distinguished from baseline by leaving vocab_hints_off at default (0=hints ON), since prompt_adapter.py runs by default in read_page() — no special flag needed"
  - "run_sweep() forwards on_progress to run_benchmark for per-strategy progress reporting in CLI"
  - "Cost projection uses ~2000 input + ~200 output token estimate per sample per strategy (rough; matches benchmark run pattern)"
  - "Empty-DB path: cost line still printed even when n_samples=0 (test contract — `test_sweep_cli_shows_cost`)"

patterns-established:
  - "Strategy table pattern: list of {name, label, kwargs} dicts — caller spreads kwargs into run_benchmark()"
  - "IAM-only sweep: SQL filter `WHERE s.category='iam'` joined to ground_truths to skip un-transcribed IAM samples"

requirements-completed: [IAM-02]

duration: ~30min
completed: 2026-05-06
---

# Phase 07 Plan 03: Sweep Infrastructure Summary

**run_sweep() + benchmark sweep CLI: all 5 strategies executable in one command against IAM samples, with line_level/auto_retry threaded through run_benchmark for sweep parity. Turns 5 TestSweep RED stubs GREEN.**

## Accomplishments

- 5/5 `TestSweep` stubs turned GREEN
- `run_benchmark()` accepts `line_level=True` and `auto_retry=True`, threading both through to `_read_single()` -> `read_page()`
- `_run_benchmark_inner()` signature extended (kw-only) — backward-compatible
- `SWEEP_STRATEGIES` exported from `evaluate.py`: 5 entries (baseline, self_correct, line_level, prompt_adapted, zoomed_verify)
- `run_sweep()` filters IAM samples via `samples.category='iam'` and runs each strategy via `run_benchmark()`, returning `{strategy_name: run_id}`
- `benchmark sweep` CLI command registered with cost projection (`Estimated cost ~$X/strategy x 5 = ~$Y total`), warning on empty IAM DB, `--yes` to bypass confirmation, exit 0 on success listing all 5 run_ids

## Verification

```bash
# All 5 TestSweep tests pass:
pytest tests/test_benchmark_evaluate.py::TestSweep -q   # 5 passed

# Existing tests untouched (TestRunBenchmark, TestEstimateCost, TestReport, TestSmokeMode, TestProgressCallback, TestDrillDown, TestRegressionDetect):
pytest tests/test_benchmark_evaluate.py -q              # 31 passed, 3 failed (TestPerWriterReport — 07-04 territory)

# Imports clean:
python3 -c "from handwriting_engine.benchmark.evaluate import run_sweep, SWEEP_STRATEGIES; print(len(SWEEP_STRATEGIES))"  # -> 5

# CLI registered:
python3 -m handwriting_engine.cli benchmark sweep --help                  # exit 0
```

## Files Modified

- `handwriting_engine/benchmark/evaluate.py` — added `line_level`/`auto_retry` parameters to `_read_single()`, `run_benchmark()`, `_run_benchmark_inner()`; threaded through to `read_page()`; appended `SWEEP_STRATEGIES` constant + `run_sweep()` function (~95 lines added)
- `handwriting_engine/cli.py` — added `benchmark_sweep` command with cost projection guardrail (~70 lines)
- `tests/test_benchmark_evaluate.py` — replaced 5 `pytest.fail()` stubs in `TestSweep` with real assertions using `_read_single`/`_available_providers` mocks and `CliRunner`

## Decisions Made

- **Backward-compatible threading.** `line_level=False` and `auto_retry=False` defaults everywhere; existing tests untouched.
- **Strategy table.** A list of `{name, label, kwargs}` dicts keeps `run_sweep()` to ~15 lines and lets future strategies be added by appending one entry.
- **CLI prompts cost; library does not.** `run_sweep()` is callable from notebooks/scripts without prompting; the CLI command owns the confirm flow.
- **IAM filter at SQL.** `WHERE s.category='iam'` joined to `ground_truths` skips IAM samples without transcriptions — no separate param.

## Out of Scope (handled by 07-04)

- `TestPerWriterReport` (3 stubs) remains RED — that's IAM-03's plan.

## Unblocks

- **07-04 (per-writer report).** Sweep run_ids are now produced; per-writer breakdown can group by `samples.student`.
- **Trained corrector real-data retrain.** Once user runs `benchmark ingest-iam` + `benchmark sweep`, the resulting (provider_outputs.output_text, ground_truths.text) pairs feed `trained_correction.dataset.from_benchmark_db()` for the v2 fine-tune that resolves the synthetic-only hallucination failure mode documented in `trained_correction/EVAL-RESULTS.md`.
