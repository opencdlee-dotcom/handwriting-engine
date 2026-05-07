# S2 — Per-Writer Few-Shot Exemplars

**Status:** SPEC (not yet a phase). Implementation blocked on benchmark DB population (same blocker as Phase 8).
**Authored:** 2026-05-06
**Source:** `.planning/NEXT-STEPS.md` § Out-of-band side projects.

---

## Goal

When transcribing an image whose `writer_id` has ≥2 already-labeled samples in the benchmark DB, the engine prepends 2–3 of those (image, ground-truth) pairs to the target prompt as multi-image exemplars. On the same writer's held-out images, this reduces CER by a margin distinguishable from noise (per Phase 8 stats), without regressing CER on writers who have <2 stored samples.

## Why this is high-leverage

- Lab-notebook grading is the dominant downstream consumer (`professor/LabNoteBookGrader/`). A semester has the *same* student writing weekly — the returning-writer scenario is the steady-state, not the edge case.
- Today's adaptation is a **text** calibration block (`writer_profile_store.build_calibration_block()` at [writer_profile_store.py:63](../handwriting_engine/writer_profile_store.py#L63)) that lists discrete observations ("crosses 7s: YES"). That's brittle — it depends on a human entering observations.
- Few-shot exemplars are the standard VLM technique for in-context style adaptation. Both Gemini and Claude support multi-image content lists today (Claude via [`providers/claude.py:56-65`](../handwriting_engine/providers/claude.py#L56-L65) `read_batch`; Gemini's SDK accepts `contents=[image_part_1, ..., image_part_N, prompt]`).

## Non-goals

- Training a model. This is purely in-context.
- Cross-writer transfer. Exemplars are sourced strictly within the same `writer_id`.
- Replacing the text calibration block. The text block stays as a fallback for writers with <2 GT samples and is composable with exemplars when both exist.

---

## Design

### 1. Exemplar selection (`writer_profile_store.py`)

Add `select_exemplars(writer_id, *, k=3, exclude_sample_id=None) -> list[Exemplar]`:

```python
@dataclass(frozen=True)
class Exemplar:
    sample_id: int
    image_path: str          # absolute, on local disk
    ground_truth: str        # the canonical transcription
```

Selection strategy v0 (cheapest defensible default — refine in v1 if benchmarks justify):

1. Pull all `(samples × ground_truths)` rows for `student = writer_id`, excluding `exclude_sample_id` (the target itself if it happens to have a GT).
2. Order by `quality_assessments.score DESC` if quality scores exist for that writer; else by `sample_id ASC` (stable, deterministic).
3. Return the first `min(k, available)` — caller decides how many to actually inject.

Determinism matters: same target image + same DB state ⇒ same exemplars selected. Don't randomize without a seed.

### 2. Provider-side multi-image plumbing

- **Claude** ([`providers/claude.py`](../handwriting_engine/providers/claude.py)) — already supports it via `read_batch`. Add a thin `read_with_exemplars(target_image_b64, exemplar_blocks, prompt, ...)` wrapper that constructs `[exemplar_1_image, exemplar_1_label_text, exemplar_2_image, exemplar_2_label_text, ..., target_image, target_prompt]`. Exemplar label text wraps the GT clearly: `"The handwriting in the previous image transcribes to: «{gt}»"`.
- **Gemini** ([`providers/gemini.py:143`](../handwriting_engine/providers/gemini.py#L143)) — currently `contents=[image_part, prompt]`. Extend to accept a list. Same labeled-exemplar interleaving.
- **TrOCR** — out of scope. It's a fixed-vocab encoder, no in-context learning. Skip silently if exemplars are passed.

### 3. Prompt-construction integration (`handwriting.py`)

`get_reading_strategies()` is the wrong layer (it returns text only). Few-shot injection happens one level up, where the provider call is assembled. Touchpoints:

- The transcription entrypoint(s) that already accept a `writer_profile` argument. When they detect `writer_id` and the DB has ≥2 GT samples for that writer, they call `select_exemplars()` and pass the result into the provider via the new `read_with_exemplars` path.
- The text calibration block is **still injected** alongside, in the prompt text — exemplars and text observations compose, they don't substitute. Reasoning: text observations encode binary facts ("crosses 7s") that exemplars may not visually demonstrate in a 3-image sample.

### 4. Cost & opt-out

- Adding 3 images to a Gemini Flash call ~quadruples that call's image-token cost. Document this in the docstring and surface a `HE_FEW_SHOT_K` env var (default 3, 0 = disabled).
- For batch / sweep contexts, default `k=2` if more than 50 samples are queued for the same writer in one batch — Claude's prompt cache amortizes the exemplar tokens across the batch but Gemini doesn't cache identically.

---

## Falsifiable success criteria

When this phase is complete, all of the following are TRUE:

1. **Eligibility gate works.** `select_exemplars("writer-with-1-sample")` returns `[]`. `select_exemplars("writer-with-5-samples", k=3)` returns 3 deterministic Exemplar rows. Verified by unit tests against a fixture DB.
2. **Provider calls carry exemplars end-to-end.** A live transcription against a writer with ≥3 GT samples produces a request payload that the provider mock asserts contains the exemplar images **before** the target image, in order, each followed by its labeled GT text. Verified for both Claude and Gemini via recorded HTTP/SDK fixtures.
3. **CER improves on a held-out per-writer split.** On the IAM test set restricted to writers with ≥4 GT samples (sourcing 3 exemplars + 1+ held-out target per writer), running the `prompt_adapted` strategy with exemplars enabled vs. disabled produces a CER delta where the Phase 8 Wilcoxon test reports `p < 0.05` and the bootstrap CIs do not overlap.
4. **No regression on cold writers.** On writers with <2 GT samples, with `HE_FEW_SHOT_K=3`, the codepath falls back cleanly to text-only calibration and CER is statistically indistinguishable from the pre-S2 baseline (Wilcoxon `p > 0.10`).
5. **TrOCR passthrough.** Calling the transcription entrypoint with `writer_id` set but `provider=trocr` does not error and does not include exemplars in the call. Logged at DEBUG, not WARNING.
6. **Cost guardrail surfaces it.** Existing sweep cost projection (Phase 6 / IAM-02) accounts for the exemplar image tokens when `HE_FEW_SHOT_K > 0`.

## Out of scope (queued for follow-up)

- **Smart exemplar selection** — diversity-by-character-coverage, hardness-aware ("show the model the writer's *messy* samples"). v0 is recency/quality-ordered. Revisit only if v0 ships and CER gain is below the IAM-test-set theoretical ceiling estimated below.
- **Dynamic `k`** — adapting `k` based on target image difficulty. Out of scope until v0 establishes the baseline.
- **Cross-session exemplar caching** — provider-side prompt cache hits for repeated exemplar sets across calls. Worth doing but separable; track as S2.1.

---

## Risks & open questions

| Risk | Mitigation / decision needed |
|------|------------------------------|
| Benchmark DB is empty today (`~/.handwriting-engine/benchmark.db` does not exist). | Block implementation start until IAM ingest+sweep run completes (Phase 7 prerequisite). Spec stays valid in the meantime. |
| Exemplar GT may itself be wrong (human transcription error in IAM is non-zero). | Already mitigated by `quality_assessments` table — gate exemplars to score ≥ threshold once that's populated. v0: trust GT. |
| 3 high-res IAM line images in one Gemini call may exceed Flash input-token budget on long lines. | Concrete number needed. Pre-implementation: measure max-image-token-count in IAM lines/ to confirm headroom. If tight, downsample exemplars to 1024px width before encoding. |
| Returning-writer few-shot may bias the model into copying the *previous transcription style* rather than reading the *new image*. (Cargo-culting from in-context examples is a known VLM failure mode.) | Address in the prompt: "The reference samples are from the same writer but contain DIFFERENT TEXT. Read what is in the final image — do not repeat the reference text." Verify in success criterion #4 — if cold-writer regression is significant, this is the suspect. |
| Phase 8 stats infrastructure must exist before criterion #3 is testable. | This phase depends on Phase 8. Order: Phase 8 → S2. Specifying it now is fine; implementation order is enforced by the criterion. |

## Touchpoints (preliminary)

- **New:** [`writer_profile_store.py`](../handwriting_engine/writer_profile_store.py) — `select_exemplars()`, `Exemplar` dataclass.
- **Edit:** [`providers/claude.py`](../handwriting_engine/providers/claude.py) — add `read_with_exemplars()` wrapper.
- **Edit:** [`providers/gemini.py`](../handwriting_engine/providers/gemini.py) — extend `contents` list construction for multi-image.
- **Edit:** [`handwriting.py`](../handwriting_engine/handwriting.py) — wire the writer_id branch into the new provider path.
- **New:** tests under `tests/` mirroring the success-criteria numbering (1–6).
- **Doc:** README provider section + `HE_FEW_SHOT_K` env var.

## Estimated scope

- ~2 plans (provider plumbing + integration & tests). ~250–400 LOC engine-side, ~300 LOC tests.
- Falsification requires Phase 8 + a populated IAM DB; budget one sweep run (~$3–5) for criterion #3.

---

## Promotion path

When ready to start: `/gsd-add-phase` (or `/gsd-insert-phase` to slot between 8 and 9). This SPEC.md graduates into the new phase's directory as the seed for `/gsd-discuss-phase`.
