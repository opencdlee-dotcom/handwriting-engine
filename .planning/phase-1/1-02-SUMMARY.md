---
phase: 1
plan: "02"
subsystem: consensus
tags: [smart-routing, uncertainty-escalation, tests, self-correction]
dependency_graph:
  requires: [phase-1-plan-01]
  provides: [smart escalation to self_correct, _count_uncertainty_markers tests, TestSelfCorrectStrategy, TestSmartEscalation]
  affects: [handwriting_engine/consensus.py, tests/test_consensus.py]
tech_stack:
  added: []
  patterns: [uncertainty-gated escalation, TDD fixture pattern, class-based test grouping]
key_files:
  created: []
  modified:
    - handwriting_engine/consensus.py
    - tests/test_consensus.py
decisions:
  - Escalation applied to both "easy" (_best_of) and "medium" (_cascade) paths in _smart_route()
  - "hard" path uses full vote — already maximum effort, no escalation needed
  - uncertainty_threshold threaded from read_with_consensus() → _smart_route() as parameter (not global)
  - Tests use class-based grouping (TestSelfCorrectStrategy, TestSmartEscalation) for isolation
metrics:
  duration: "~10 minutes"
  completed: "2026-04-09"
  tasks_completed: 2
  files_modified: 2
  tests_added: 10
  tests_total: 37
---

# Phase 1 Plan 02: Smart Escalation and Self-Correction Tests Summary

**One-liner:** Uncertainty-gated auto-escalation in _smart_route() plus 10 new tests covering _count_uncertainty_markers, self_correct strategy, and smart escalation paths.

## What Was Built

Extended `_smart_route()` with uncertainty-gated self-correction escalation, and added comprehensive test coverage for all new Phase 1 functionality.

### Changes Made

**`handwriting_engine/consensus.py`**:
- Updated `_smart_route()` signature: added `uncertainty_threshold: int = 3` parameter
- "easy" path: after `_best_of()` result, counts `[?]` markers; if `> uncertainty_threshold`, escalates to `_self_correct()` with `strategy_used = "smart→self_correct"`
- "medium" path: same escalation logic after `_cascade()` result
- "hard" path: unchanged — full vote is already maximum effort
- `read_with_consensus()` now passes `uncertainty_threshold` to `_smart_route()` call

**`tests/test_consensus.py`**:
- Added import: `_count_uncertainty_markers`
- Added 3 standalone test functions for `_count_uncertainty_markers`:
  - `test_count_uncertainty_markers_none` — clean text returns 0
  - `test_count_uncertainty_markers_question` — `[?]` and `[illegible: ...]` counted correctly
  - `test_count_uncertainty_markers_multiple` — `[?] [?] [unclear] ??? [illegible]` returns >= 3
- Added `TestSelfCorrectStrategy` class (5 tests):
  - `test_self_correct_calls_provider_twice` — initial + correction = 2 calls
  - `test_self_correct_no_extra_pass_when_clean` — still does 1 correction pass even when clean
  - `test_self_correct_max_rounds_respected` — max_self_correct_rounds=2 → 3 total calls
  - `test_self_correct_strategy_used_field` — strategy_used contains "self_correct"
  - `test_self_correct_provider_results_has_initial_and_corrected` — verifies dict keys
- Added `TestSmartEscalation` class (2 tests):
  - `test_smart_escalates_on_high_uncertainty` — 4 `[?]` markers > threshold 3 → escalates
  - `test_smart_no_escalation_on_clean_output` — clean output → no escalation, 1 call only

## Test Results

```
37 passed, 5 warnings in 0.52s
```

All 37 tests pass (27 pre-existing + 10 new).

## Commit

- `38b6c5e` — feat(phase-1-02): smart escalation + self-correction tests (REQ-001, REQ-002)

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- `_smart_route` importable with new signature: VERIFIED
- All 37 tests pass: VERIFIED (`python3 -m pytest tests/test_consensus.py -x -q`)
- Commit `38b6c5e` exists: VERIFIED
- `TestSelfCorrectStrategy` in test file: VERIFIED
- `TestSmartEscalation` in test file: VERIFIED
