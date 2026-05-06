# S5 — Char-Level Consensus + Confusion-Pair-Aware Postprocess

**Status:** SPEC (not yet a phase). Implementation unblocked — no data dependency for char-consensus; confusion-postprocess wants populated DB but degrades gracefully.
**Authored:** 2026-05-06
**Source:** `.planning/NEXT-STEPS.md` § Out-of-band side projects.

---

## Goal

Two complementary additions to the engine's transcription pipeline:

1. **Char-level consensus.** When ≥2 providers run, in addition to today's word-level voting ([`consensus.py:1018`](../handwriting_engine/consensus.py#L1018)), compute character-level alignment within disagreed-on words. Producers disagreeing on `rn` vs `m`, `cl` vs `d`, `0` vs `O` get resolved at the char level using the global confusion-pair map plus per-writer profile, instead of arbitrarily picking one provider's word.

2. **Confusion-pair-aware postprocess.** Extend [`postprocess.correct()`](../handwriting_engine/postprocess.py) with a pass that consumes the running confusion-pair stats already tracked by [`benchmark/metrics.py:classify_errors`](../handwriting_engine/benchmark/metrics.py#L224) — when a candidate word is one confusion-pair-swap away from a known-domain term, prefer the swap.

After this phase, single-character substitution errors (the dominant residual error class per Phase 7 drill-down reports) drop measurably without regressing easy cases.

## Why this is high-leverage

- The drill-down report already tracks confusion pairs per [`benchmark/report.py:301`](../handwriting_engine/benchmark/report.py#L301) `sample_drill_down`. The data exists; it's read-only today. Putting it back into the pipeline closes the observability → action loop.
- Word-level voting silently mis-resolves when *all* providers see different things (no majority). Char-level voting catches the dominant subset of these: single-char swaps within otherwise-aligned words.
- Phase 5 already shipped postprocess + multi-word phrase correction (commits `9b8c464`, `f0ebdea`). S5 is an additive postprocess pass, not a redesign.
- Compounds with S2 and S4: per-writer confusion-pair history (from S4 corrections) makes the postprocess pass per-writer adaptive without needing a new model.

## Non-goals

- Replacing word-level voting. Char-level runs *within* word-level disagreements, not instead of them.
- Building a language model. We use existing wordlists and known confusion pairs ([`handwriting.py:594`](../handwriting_engine/handwriting.py#L594) `get_disambiguation_pairs`).
- Trained-corrector territory. The postprocess pass is rule-based; the trained corrector is its own (separately-tracked) project.

---

## Design

### 1. Char-level consensus ([`consensus.py`](../handwriting_engine/consensus.py))

Today's flow at [`consensus.py:1018`](../handwriting_engine/consensus.py#L1018):

```python
sorted_votes = sorted(word_votes.items(), key=lambda x: x[1], reverse=True)
winner = sorted_votes[0][0]
# ...if no majority, pick highest-weighted, mark [?alt: …]
```

S5 inserts a step *before* the no-majority fallback:

```python
if not has_majority(sorted_votes):
    # Try to resolve at char level
    resolved = resolve_char_level(
        candidates=[w for w, _ in sorted_votes],
        weights=[v for _, v in sorted_votes],
        confusion_pairs=GLOBAL_CONFUSION_PAIRS,
        writer_confusion_resolutions=writer_profile.get("confusion_resolutions", {}) if writer_profile else {},
    )
    if resolved is not None:
        result_words.append(resolved)
        # Don't emit [?alt: …] — char-level resolved it
        continue
    # else fall through to existing no-majority handling
```

`resolve_char_level()` algorithm v0:

1. Align candidate strings via `difflib.SequenceMatcher` char-by-char.
2. For each diff position, gather the chars each provider voted for.
3. If all chars at that position map to the *same confusion-pair group* (e.g. `{r,n}` vs `{m}` is the known `rn↔m` pair), apply the writer's preference if set, otherwise the global default for that pair.
4. If no confusion-pair match, return `None` (defer to existing fallback).

Determinism: same inputs ⇒ same output. Ordering of providers must not affect result.

### 2. Confusion-pair-aware postprocess ([`postprocess.py`](../handwriting_engine/postprocess.py))

New function:

```python
def correct_confusion_pairs(
    text: str,
    *,
    domain: str = "biology",
    writer_id: str | None = None,
    db_path: Path | None = None,
) -> tuple[str, list[Correction]]:
    """
    For each word in text, if a single confusion-pair swap produces a known
    domain term, prefer the swap. Returns corrected text + list of corrections
    applied (for audit logging).
    """
```

Logic:

1. Tokenize.
2. For each word that's NOT in the domain wordlist:
   - Generate all candidates 1 confusion-pair swap away (`rn↔m`, `cl↔d`, `0↔O`, `l↔1`, `5↔S`, full list from `get_disambiguation_pairs()`).
   - If exactly one candidate is in the wordlist, swap.
   - If multiple candidates are in the wordlist, prefer the one matching writer-specific resolutions. If still ambiguous, no swap (don't introduce error).
3. Return the corrected text + audit list.

This pass runs **after** `correct()`'s existing edit-distance-1 wordlist correction at [`postprocess.py:200-228`](../handwriting_engine/postprocess.py#L200-L228) — the new pass is restricted to confusion-pair-shaped edits, which are higher-precision than generic ED1.

### 3. DB integration (optional, graceful)

If `db_path` is provided and `corrections` table exists (from S4), pull writer-specific historical confusion-pair resolutions to bias the postprocess. If no DB or table absent, fall back to global pairs. Hard-fail-free.

### 4. Wiring

- `consensus.py` calls `resolve_char_level` inline; no top-level API change.
- `postprocess.correct()` gains a `correct_confusion_pairs` step in its existing pipeline. New env flag `HE_CONFUSION_POSTPROCESS=1` (default ON; flip to 0 to disable for A/B).

---

## Falsifiable success criteria

When this phase is complete, all of the following are TRUE:

1. **Char-level resolves a known confusion case.** Given fixture providers returning `["modern", "rnodern", "modern"]` (with weights), word-level voting resolves to `modern` (majority). Given `["modern", "rnodern"]` (no majority), char-level resolution returns `modern` *with no `[?alt: …]` marker* because the only difference is the `m↔rn` confusion pair. Verified by unit test.
2. **Defers cleanly when not a confusion case.** Given `["apple", "orange"]` (unrelated words), char-level returns `None` and the existing `[?alt: orange]` marker emits as today.
3. **Writer-specific bias works.** With `writer_profile = {"confusion_resolutions": {"rn↔m": "rn"}}`, char-level returns `rnodern` instead of `modern` from input `["modern", "rnodern"]`. Verified by unit test.
4. **Postprocess corrects a real confusion case.** Input: `"the celI underwent mitosis"` (capital I after cell, common `l↔I` confusion). Output: `"the cell underwent mitosis"`. Audit log records the correction.
5. **Postprocess does not over-correct.** Input: `"the apple is red"` — no swap suggests a domain term exists, so output is bit-identical to input. Verified across a 50-sample non-confusion fixture.
6. **End-to-end CER win.** On the IAM test set's `prompt_adapted` strategy, CER with `HE_CONFUSION_POSTPROCESS=1` is lower than with `HE_CONFUSION_POSTPROCESS=0` by a margin where the Phase 8 Wilcoxon test reports `p < 0.05`. (Like S2, this requires Phase 8 to be testable.)
7. **No regression on the LabNoteBookGrader fixture.** Pre-S5 vs. post-S5 CER on the grader's existing test corpus is ≥0 (improvement or unchanged). Verified by `cd professor/LabNoteBookGrader && pytest tests/test_grading_accuracy.py` or equivalent.
8. **Audit trail.** Every confusion-pair correction applied is logged at INFO with original word, corrected word, and pair name, so silent over-correction is detectable.

## Out of scope (queued for follow-up)

- **Multi-swap candidates.** v0 only considers 1-confusion-pair-swap-away candidates. 2-swap candidates explode the candidate space; revisit only if v0 ships and the residual error analysis says 2-swap matters.
- **Position-weighted edit costs.** Treating confusion-pair swaps as cheaper than generic ED1 — already implicit in this design (separate pass), but a unified weighted-edit-distance reformulation is a v2 concern.
- **Char-level consensus across full sentences (not just disagreement words).** Compute cost is high; v0 restricts to no-majority words.
- **Learning new confusion pairs from data.** v0's pairs are from `get_disambiguation_pairs()`. Discovering new pairs from corrections data is a separable analytics project.

---

## Risks & open questions

| Risk | Mitigation / decision needed |
|------|------------------------------|
| Postprocess pass over-corrects on non-domain text (poetry, names, code in lab notes). | The "exactly one candidate in wordlist" rule already prevents most. Add a per-domain wordlist; when `domain="general"`, postprocess runs at lower aggressiveness (require ≥2 matching domain wordlists or skip). |
| Char-level alignment fails on string-length disagreements (`"colour"` vs `"color"` — different length). | `SequenceMatcher` handles this. Test with a length-mismatch fixture in criterion #2. |
| Determinism across provider order. | The `resolve_char_level` algorithm sorts candidates by weight; ties broken by provider name (alpha). Documented; tested. |
| Trained-corrector overlap. The trained corrector also fixes confusion pairs. Stacking both could double-correct or fight. | The trained corrector is gated OFF today (`HE_USE_TRAINED_CORRECTOR=0`). When it eventually flips on, run both with the rule-based pass *first* and trained pass *second* — rule-based handles obvious cases, trained handles residuals. Document the order. |
| Per-writer confusion data is sparse early on (S4 just shipped). | Pass falls back to global confusion pairs. As S4 accumulates data, per-writer biasing kicks in automatically. No phase ordering blocker. |
| Wordlist coverage gaps — biology terms missing produce false negatives. | Existing biology wordlist is the input; gaps are pre-existing. Document, don't expand wordlist as part of S5. |

## Touchpoints (preliminary)

- **Edit:** [`handwriting_engine/consensus.py`](../handwriting_engine/consensus.py) — `resolve_char_level()`, hook into existing voting fallback.
- **Edit:** [`handwriting_engine/postprocess.py`](../handwriting_engine/postprocess.py) — `correct_confusion_pairs()`, integrate into `correct()` pipeline.
- **Edit:** [`handwriting_engine/handwriting.py`](../handwriting_engine/handwriting.py) — expose confusion-pair list in a more-machine-consumable shape if not already.
- **New:** tests for both directions: char-consensus + postprocess.
- **Doc:** README env-flag table.

## Estimated scope

- 1 plan, ~250 LOC engine-side, ~300 LOC tests.
- Falsification of criterion #6 needs Phase 8 + IAM data; criteria #1–5, #7–8 are testable today.

---

## Promotion path

`/gsd-add-phase`. Order suggestion: ship S5 *before* S4 if you want immediate engine-internal CER wins; ship S5 *after* S4 if you want per-writer postprocess biasing on day one. Spec is order-agnostic.
