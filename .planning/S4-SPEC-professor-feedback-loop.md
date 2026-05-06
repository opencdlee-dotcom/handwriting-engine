# S4 — Professor OS ↔ Engine Feedback Loop

**Status:** SPEC (not yet a phase). Implementation depends on S3 (skill bridge) for full payoff. Engine-side write API can ship independently.
**Authored:** 2026-05-06
**Source:** `.planning/NEXT-STEPS.md` § Out-of-band side projects.

---

## Goal

When `professor/LabNoteBookGrader/` grades a student's lab notebook and the engine reports a low-confidence read on a region, the grader surfaces it to the user (instructor), captures the corrected transcription, and writes the (image_region, ground_truth) pair into the engine's `benchmark.db` as a per-writer ground truth. Over a semester, the same student's per-writer accuracy compounds — every grading session improves the next one.

## Why this is high-leverage

- The grader already imports the engine ([`grader/handwriting_reader.py`](../../professor/LabNoteBookGrader/grader/handwriting_reader.py), [`grader/pdf_processor.py`](../../professor/LabNoteBookGrader/grader/pdf_processor.py)). Engine integration is in place; only the *write-back* direction is missing.
- Lab-notebook semesters generate ~12 weeks × N students of GT-quality data for free, as a byproduct of work the instructor was already doing.
- This is the data source S2 (per-writer few-shot exemplars) needs in order to deliver value on returning students. Without S4 the benchmark DB stays IAM-only and S2's per-writer exemplars only help on IAM samples — not on actual classroom workload.
- The engine already has the write primitives: [`benchmark/db.py:289`](../handwriting_engine/benchmark/db.py#L289) `insert_ground_truth()`, [`db.py:218`](../handwriting_engine/benchmark/db.py#L218) sample insert. S4 is mostly wiring + UI in the grader, not new engine plumbing.

## Non-goals

- Replacing the human grading workflow. Corrections happen as a side effect of grading, not as a separate "annotation session" the instructor must run.
- Round-tripping every transcription. Only **low-confidence** reads (engine-reported `[?alt: …]` markers, consensus disagreement, or `confidence < threshold`) ask for confirmation.
- Auto-correcting from prior corrections. Compounding happens through S2 (few-shot) and S5 (confusion-pair postprocess), not by mutating the prompt directly with prior corrections (that's brittle).
- Cross-instructor data sharing. Profile + GT data stays in the local instructor's `~/.handwriting-engine/`. Multi-instructor sync is a separate problem.

---

## Design

### 1. Engine-side: a stable `record_correction()` API

In [`handwriting_engine/benchmark/db.py`](../handwriting_engine/benchmark/db.py), add a high-level helper that wraps the existing primitives:

```python
def record_correction(
    *,
    image_path: str,
    writer_id: str,
    corrected_text: str,
    original_vlm_text: str,
    confidence: float,
    source: str = "labgrader",
) -> int:
    """
    Idempotently record an instructor-corrected transcription.

    - If a sample with this image_hash exists, attach a new ground_truth row
      (don't dup the sample). Otherwise insert sample + ground_truth.
    - Stores the (original_vlm_text, corrected_text) pair in a new
      `corrections` table for trained-corrector training data.
    - Returns the ground_truth id.
    """
```

New table:

```sql
CREATE TABLE IF NOT EXISTS corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id       INTEGER NOT NULL REFERENCES samples(id),
    ground_truth_id INTEGER NOT NULL REFERENCES ground_truths(id),
    original_text   TEXT NOT NULL,
    confidence      REAL,
    source          TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_corrections_sample ON corrections(sample_id);
```

Why a separate table from `ground_truths`: GTs from IAM are authoritative; instructor corrections are *also* GT but may be lower-quality (instructor might typo) — keeping them in their own table lets the trained corrector train on `(original_text, corrected_text)` directly, and lets future quality-assessment runs flag suspect rows without polluting `ground_truths`.

### 2. Grader-side: low-confidence detection

In [`professor/LabNoteBookGrader/grader/handwriting_reader.py`](../../professor/LabNoteBookGrader/grader/handwriting_reader.py), after the engine call:

```python
result = read_with_consensus(...)
low_confidence_regions = extract_review_targets(result, threshold=0.7)
# Each region carries: image crop path, original VLM text, confidence, marker context
```

`extract_review_targets` is grader-side; it consumes the engine's `ConsensusResult` (already a public model — [`__init__.py:62`](../handwriting_engine/__init__.py#L62)) and returns reviewable regions.

### 3. Grader-side: correction capture

The grader already has a GUI ([`LabNoteBookGrader/gui/`](../../professor/LabNoteBookGrader/gui/)). Add a "Review low-confidence reads" panel that, per region:

1. Shows the image crop alongside the engine's transcription with `[?alt: X/Y]` markers visible.
2. Lets the instructor either accept (= confirm engine output is correct) or correct (type the right text).
3. On submit, calls `handwriting_engine.benchmark.db.record_correction()` with the writer_id derived from the active student.

Engine writer_id ↔ Professor OS student ID mapping: instructor sets `student_id` per assignment; grader maps `student_id → writer_id="prof-{course}-{student_id}"`. Documented mapping rule, not hardcoded magic.

### 4. Trained-corrector training-data feed

The trained corrector ([`handwriting_engine/trained_correction/`](../handwriting_engine/trained_correction/)) currently can ingest from `--from-benchmark-db` per [NEXT-STEPS.md:90](NEXT-STEPS.md#L90). After S4, the same flag picks up real instructor corrections — no separate training pipeline needed.

---

## Falsifiable success criteria

When this phase is complete, all of the following are TRUE:

1. **Engine API is stable and idempotent.** Calling `record_correction()` twice with identical args writes one sample + one GT + two correction rows (the correction history grows; the canonical sample doesn't dup). Verified by unit test.
2. **Schema migration is reversible.** The `corrections` table can be dropped without breaking any other engine functionality. Verified by running the test suite with the table absent.
3. **Grader surfaces low-confidence reads.** Running the grader on a fixture lab notebook with deliberately ambiguous handwriting produces a non-empty review queue. Engine confidence threshold is configurable (env var or grader config), default 0.7.
4. **Corrections persist to the engine's DB.** After running a grading session and submitting 5 corrections, `sqlite3 ~/.handwriting-engine/benchmark.db "SELECT COUNT(*) FROM corrections"` returns ≥5, all linked to valid samples and GTs.
5. **Per-writer accumulation works.** Grading the same student twice in two sessions adds rows to the *same* `student` value in `samples`. `SELECT COUNT(*) FROM samples WHERE student='prof-bio101-jdoe'` increases monotonically across sessions.
6. **Trained corrector ingests real data.** `python3 -m handwriting_engine.trained_correction.train --from-benchmark-db` produces a training file whose row count matches `(SELECT COUNT(*) FROM corrections)`. Manual spot-check of 20 random rows confirms the (original, corrected) pairs are sane.
7. **No regression on existing grader workflow.** A grader run with the review panel disabled (env flag) produces output bit-identical to pre-S4. Reviewability is opt-in for any user not yet ready for the workflow.

## Out of scope (queued for follow-up)

- **Inter-instructor data sharing / sync.**
- **Auto-derived writer profiles from corrections.** (Could compute "this student crosses 7s 80% of the time" from accumulated corrections — separate project.)
- **Confidence calibration on the grader side.** (The 0.7 threshold is heuristic. Phase 8 stats could inform a better cutoff later.)
- **Web UI / cloud upload.** Local-only.

---

## Risks & open questions

| Risk | Mitigation / decision needed |
|------|------------------------------|
| Instructor may rush and submit wrong corrections (typos, misreadings of their own student's hand). Garbage-in poisons S2/trained-corrector. | Track `source="labgrader"` on every row. Add a `quality` column or use the existing `quality_assessments` table to let a periodic review pass downgrade suspect rows. **Don't** auto-trust corrections for trained-corrector training without a quality gate. |
| Image crops sent to engine.db must be persisted somewhere — the original PDF page is not a stable identifier. | The grader already extracts page images for grading; persist crops to `~/.handwriting-engine/student-corpus/{course}/{student_id}/{date}-p{page}-r{region}.png` and pass that absolute path to `record_correction`. |
| Privacy: student handwriting samples are FERPA-protected. Storing them outside the grader's directory crosses a boundary. | Document local-only storage. Engine `~/.handwriting-engine/` is on the same machine as the grader. Add a `--purge-student-data <student_id>` engine CLI command for end-of-semester cleanup. |
| The grader's GUI is a separate codebase; spec creep risk. | Scope clarification: S4 ships the *engine API* + the *grader-side detection logic*. The GUI panel is a follow-up plan; an interim CLI-prompt fallback in the grader is acceptable for v0. |
| Schema migration on a populated benchmark DB. | Use `CREATE TABLE IF NOT EXISTS` (existing pattern in [`db.py`](../handwriting_engine/benchmark/db.py)). No data migration needed — it's an additive table. |
| S2 needs this data to be useful, but S2 is also blocked on IAM. Risk of waiting for S4 before shipping S2 even on IAM-only. | Order: S4 ships independently. S2 implementation can use IAM data alone for falsification (criterion #3 in S2-SPEC); S4 then makes S2 useful for the actual classroom use case. |

## Touchpoints (preliminary)

- **New:** [`handwriting_engine/benchmark/db.py`](../handwriting_engine/benchmark/db.py) — `record_correction()`, `corrections` table.
- **New:** engine CLI `benchmark record-correction` for ad-hoc / scripting use.
- **New:** engine CLI `benchmark purge-writer <writer_id>` for cleanup.
- **Edit:** [`professor/LabNoteBookGrader/grader/handwriting_reader.py`](../../professor/LabNoteBookGrader/grader/handwriting_reader.py) — add `extract_review_targets()`.
- **Edit:** grader workflow / GUI to show review panel.
- **Tests:** engine-side unit tests + grader integration test against fixture notebook.

## Estimated scope

- ~2 plans (engine schema/API; grader detection + UI). Engine: ~150 LOC + tests. Grader: ~250 LOC + tests + GUI panel.
- No external data dependency. Can ship engine-side immediately; grader integration in parallel.

---

## Promotion path

`/gsd-add-phase`. This phase straddles two repos — keep the engine-side and grader-side as **separate plans within one phase** so the engine API is reviewable and lockable before grader work begins.
