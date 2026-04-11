# Phase 1 Context — Self-Correction + Smart Escalation

## Implementation Decisions

### Self-Correction Strategy (REQ-001)

**Where to add it:** `handwriting_engine/consensus.py` — add `_self_correct()` function, wire into `read_with_consensus()` dispatch.

**How self-correction works:**
1. Call primary provider's `read_image()` to get initial transcription
2. Build a correction prompt that includes: the original image + the initial transcription + instructions to find errors
3. Call the same provider again with this correction prompt
4. Return the corrected output as `ConsensusResult`

**Self-correction prompt design:**
- System prompt: instructs the model it is reviewing its OWN previous transcription
- Include the top 15 character disambiguation pairs (from `handwriting.py CHARACTER_DISAMBIGUATION`)
- Ask it to flag any suspicious readings and correct them
- Explicitly say: "If you are confident the original is correct, reproduce it unchanged"
- Prevents overcorrection — only change what is genuinely suspicious

**The correction prompt constant:** Add `SELF_CORRECTION_PROMPT` to `handwriting_engine/handwriting.py` alongside existing prompt constants.

**`max_self_correct_rounds` parameter:** Default 1. Loop: if corrected output still contains [?] markers AND rounds > 1, do another pass. Cap at 3.

**Provider for self-correction:** Same provider that did the initial read. Do NOT switch providers — that would be debate strategy, not self-correction.

**ConsensusResult fields:**
- `strategy_used = "self_correct"`
- `confidence` = based on [?] marker count in corrected output (same formula as existing)
- `provider_results = {"initial": initial_text, "corrected": corrected_text}`

### Smart Escalation (REQ-002)

**Where to add it:** `_smart_route()` in `consensus.py` — after primary provider read, count [?] markers. If count > threshold, call `_self_correct()`.

**[?] marker counting:**
```python
import re
def _count_uncertainty_markers(text: str) -> int:
    return len(re.findall(r'\[\?[^\]]*\]', text))
```

**Threshold:** `uncertainty_threshold=3` default. Configurable via `read_with_consensus(uncertainty_threshold=N)`.

**Strategy label:** `"smart→self_correct"` when escalation triggers.

### Tests

**File:** `tests/test_consensus.py` — add new test class `TestSelfCorrectStrategy`.

**Test approach:** Use `unittest.mock.patch` to mock provider `read_image`. First call returns text with [?] markers. Second call (correction) returns clean text. Verify:
- Correction was called
- strategy_used = "self_correct"
- Output is the corrected text

**Smart escalation test:** Mock returns > 3 [?] markers on first read → verify self_correct triggered. Mock returns 1 [?] marker → verify no escalation.

## Files to Change

- `handwriting_engine/consensus.py` — add `_self_correct()`, update `read_with_consensus()`
- `handwriting_engine/handwriting.py` — add `SELF_CORRECTION_PROMPT` constant
- `tests/test_consensus.py` — add `TestSelfCorrectStrategy` tests

## Files NOT to Change

- `providers/base.py` — ConsensusResult already has `provider_results` dict and `strategy_used` field
- `providers/gemini.py`, `providers/claude.py`, `providers/openai.py` — no provider changes needed
- `vision.py` — no changes needed for Phase 1
