---
phase: 1
plan: "01"
subsystem: consensus
tags: [self-correction, consensus, handwriting, ocr]
dependency_graph:
  requires: []
  provides: [_self_correct, _count_uncertainty_markers, SELF_CORRECTION_PROMPT]
  affects: [handwriting_engine/consensus.py, handwriting_engine/handwriting.py]
tech_stack:
  added: []
  patterns: [two-pass self-correction, forward-reference avoidance, module-level regex]
key_files:
  created: []
  modified:
    - handwriting_engine/handwriting.py
    - handwriting_engine/consensus.py
decisions:
  - Moved _UNCERTAINTY_RE to module top to avoid forward reference from _self_correct()
  - _self_correct() placed between _best_of() and _vote() for logical grouping
  - max_rounds clamped to [1, 3] — never 0 passes, never runaway loops
metrics:
  duration: "~10 minutes"
  completed: "2026-04-09"
  tasks_completed: 3
  files_modified: 2
---

# Phase 1 Plan 01: Self-Correction Consensus Strategy Summary

**One-liner:** Two-pass GPT-4o-style self-correction strategy using SELF_CORRECTION_PROMPT (JoD 2025: CER 1.75% → 1.39% on IAM).

## What Was Built

Added the `self_correct` consensus strategy to the handwriting engine, implementing the Journal of Documentation 2025 finding that model self-correction reduces Character Error Rate by ~20% on IAM handwriting datasets.

### Changes Made

**`handwriting_engine/handwriting.py`** (line 276):
- Added `SELF_CORRECTION_PROMPT` constant with `{initial_transcription}` format placeholder
- Placed after `ANTI_HALLUCINATION_PROTOCOL` block
- Covers: character confusion pairs, [?] marker resolution, number re-reads, word boundary checks

**`handwriting_engine/consensus.py`**:
- Moved `_UNCERTAINTY_RE` regex from line ~680 to module top (after `CASCADE_ORDER`, before all functions) — required to avoid forward reference from `_self_correct()`
- Added `_count_uncertainty_markers(text: str) -> int` helper immediately after `_UNCERTAINTY_RE`
- Added `_self_correct()` function between `_best_of()` and `_vote()`:
  - Performs initial read using best provider for content type
  - Runs up to `max_rounds` correction passes (clamped [1, 3])
  - Stops early if no uncertainty markers remain after a pass
  - Falls back to `_best_of()` if circuit breaker is open
  - Returns `ConsensusResult` with `provider_results={"initial": ..., "corrected": ...}` and `strategy_used="self_correct_{N}pass"`
- Updated `read_with_consensus()` signature: added `max_self_correct_rounds: int = 1` and `uncertainty_threshold: int = 3`
- Added `elif strategy == "self_correct"` dispatch branch
- Updated ValueError message to include `"self_correct"` in valid strategy list

## Commit

- `ad6fe35` — feat(phase-1-01): add self-correction consensus strategy (REQ-001)

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- `SELF_CORRECTION_PROMPT` in handwriting.py: FOUND (line 276)
- `_self_correct` importable: VERIFIED (`python3 -c "from handwriting_engine.consensus import _self_correct; print('OK')"`)
- `_count_uncertainty_markers` importable: VERIFIED
- `max_self_correct_rounds` in `read_with_consensus` signature: VERIFIED
- Commit `ad6fe35` exists: VERIFIED
