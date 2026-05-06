# NEXT-STEPS — What to do after Phase 7

**Status:** Phase 7 (IAM Data Ingestion + Sweep Infrastructure) shipped 2026-05-06.
The infrastructure is ready. The remaining unblockers are **manual data acquisition** and **one sweep run**, after which Phases 8-9 + the trained-corrector real-data retrain are all unblocked.

---

## Why this exists

Phase 7 ended with all 17 RED stubs GREEN, all 4 plans landed, and `benchmark sweep` / `benchmark report --per-writer` callable from the CLI. But two downstream goals remain blocked on **user-side work that can't be automated:**

1. **Phase 8 (Statistics Layer)** needs ≥10-sample multi-strategy runs in the benchmark DB. That requires running the sweep against real IAM data.
2. **Trained-corrector real-data retrain** needs (vlm_text, ground_truth) pairs from real VLM runs. Today the corrector is gated OFF (`HE_USE_TRAINED_CORRECTOR=0`) because synthetic-only training causes hallucinations on hard cases — see `handwriting_engine/trained_correction/EVAL-RESULTS.md`. The sweep produces exactly those pairs.

Both unblock from a single sweep run.

---

## Step 1 — Download the IAM Handwriting Database

The IAM dataset is registration-gated, so this is manual.

1. Register and download from one of:
   - [HEIA-FR mirror](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
   - Original Univ. Bern site (legacy)
2. Extract `lines.tgz` (line-level images) and `ascii.tgz` (transcriptions). Layout expected:
   ```
   <iam-root>/
   ├── ascii/
   │   └── lines.txt
   └── lines/
       └── <writer>/<form>/<form>-<line>.png
   ```
3. **Optional but recommended:** also grab the `largeWriterIndependentTextLineRecognitionTask/` partition files. They split IAM into `trainset.txt`, `validationset1.txt`, `testset.txt`. The sweep should run against `testset.txt` only — never on training data, or the baseline isn't a true generalization measure.

---

## Step 2 — Ingest IAM into the benchmark DB

```bash
cd "/Users/user/Documents/Work & Projects/VSCode Projects/handwriting-engine"

# Test partition only (safer, ~2k samples):
python3 -m handwriting_engine.cli benchmark ingest-iam \
    --ascii-dir <iam-root>/ascii \
    --lines-dir <iam-root>/lines \
    --partition-file <iam-root>/largeWriterIndependentTextLineRecognitionTask/testset.txt

# Verify ingestion:
python3 -m handwriting_engine.cli benchmark list --show-samples | head -20
```

Expected: rows with `category='iam'`, `student='iam-writer-XXX'`. The CLI prints `{ingested, skipped_dup, skipped_missing}` counts.

> **Cost note:** The next step runs ALL FIVE strategies against every ingested sample. If you ingest the full test set (~2k lines), one sweep can run ~$3-5 in API calls (Gemini Flash is the cheap default). If unsure, ingest ~50 samples first using a partition subset, do a smoke sweep, then scale up.

---

## Step 3 — Run the multi-strategy sweep

```bash
python3 -m handwriting_engine.cli benchmark sweep --provider gemini
# Confirms cost projection, then executes all 5 strategies:
#   baseline, self_correct, line_level, prompt_adapted, zoomed_verify
# Returns one run_id per strategy.
```

Add `--yes` to skip the confirmation prompt (useful in CI / headless runs).

---

## Step 4 — Inspect per-writer breakdown

```bash
# Replace <run_id> with one of the run_ids the sweep printed:
python3 -m handwriting_engine.cli benchmark report --run-id <run_id> --per-writer
```

This is the IAM-03 deliverable: shows whether a strategy's CER gain is consistent across writers or driven by a few easy ones.

---

## Step 5 — Retrain the trained corrector on real data

Now the (vlm_output, ground_truth) pairs from the sweep can fine-tune the FLAN-T5 corrector that's currently gated off:

```bash
# Continue from the v1 synthetic checkpoint (don't start from scratch — preserves
# the easy-error fixes the synthetic data already taught it):
python3 -m handwriting_engine.trained_correction.train \
    --from-benchmark-db ~/.handwriting-engine/benchmark.db \
    --continue-from ~/.handwriting-engine/models/trained-corrector-v1 \
    --output-dir ~/.handwriting-engine/models/trained-corrector-v2 \
    --epochs 3

# A/B eval: v2 vs v1 vs heuristic-only
python3 -m handwriting_engine.cli trained-correction eval --n-pairs 200 --seed 9999
```

Expected: hallucinations on hard cases (the 3/10 spot-check failures documented in `trained_correction/EVAL-RESULTS.md`) drop substantially — because the model now sees the actual VLM error distribution rather than a guessed-at synthetic one. Once the gated A/B passes, flip `HE_USE_TRAINED_CORRECTOR=1` to default-on the combined heuristic→trained pipeline.

---

## Step 6 — Plan Phase 8

Once the sweep run lives in the DB, Phase 8 (Statistics Layer) is unblocked:
- Wilcoxon signed-rank p-values on `benchmark compare`
- 95% bootstrap CIs on CER estimates
- Cohen's r effect size

Run `/gsd:plan-phase 8` from inside the engine directory when ready.

---

## Out-of-band side projects flagged in the broader strategy

These do not block Phase 8 but are worth queuing for follow-up sessions:

- **S2 — Per-writer few-shot exemplars.** Currently `writer_profile_store.build_calibration_block()` injects writer-specific text hints into prompts. Stronger: pull 2-3 already-labeled images of the same writer from the benchmark DB and pass them as multi-image prompts (Gemini and Claude both support it). Likely the biggest single CER gain on returning writers (lab notebook semester scenarios). **Spec drafted: `.planning/S2-SPEC-per-writer-few-shot.md` (2026-05-06).**
- **S3 — Wire `~/.claude/skills/handwriting-reader/` skill to call the engine library directly.** The skill currently does its own multi-pass workflow. One source of truth = engine improvements propagate immediately. **Spec drafted: `.planning/S3-SPEC-skill-engine-bridge.md` (2026-05-06).**
- **S4 — Professor OS feedback loop.** `professor/LabNoteBookGrader/` graders should surface low-confidence reads, capture corrections, and write them back to the benchmark DB as per-writer ground truth. Per-writer accuracy then compounds over a semester. **Spec drafted: `.planning/S4-SPEC-professor-feedback-loop.md` (2026-05-06).**
- **S5 — Char-level consensus + confusion-pair-aware postprocess.** Word-level voting catches obvious disagreements; char-level catches single-character swaps (`rn↔m`, `cl↔d`). The drill-down report already tracks confusion pairs; postprocess can consume them. **Spec drafted: `.planning/S5-SPEC-char-consensus-confusion-postprocess.md` (2026-05-06).**
