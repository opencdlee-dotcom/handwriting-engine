# trained_correction

Optional learned post-correction layer for the handwriting engine. Stacks on
top of the existing heuristic post-correction (`handwriting_engine.postprocess`).

## When to use this

Heuristic post-correction (`correct_domain_terms`) catches edit-distance-1
single-word errors and bigram phrase errors against curated wordlists. It's
high-precision and cheap. Past that, multi-character OCR confusions
(`rn`↔`m`, `cl`↔`d`), doubled letters, smushed words, dropped punctuation,
and context-dependent errors need a learned model.

This subpackage trains a small seq2seq model (default: `flan-t5-small`, ~80M
params) on synthetic OCR-error pairs to fix what the heuristic can't.

## Quick start

```bash
# Install the optional deps
pip install -e ".[trained-correction]"

# Train (default: ~45 min on CPU, less on MPS / CUDA)
handwriting-engine trained-correction train \
  --output-dir ~/.handwriting-engine/models/trained-corrector-v1 \
  --num-pairs 50000 --num-epochs 2

# A/B evaluate against heuristic
handwriting-engine trained-correction eval --n-pairs 1000

# Enable in production (off by default)
export HE_USE_TRAINED_CORRECTOR=1
```

Or pass `use_trained=True` directly:

```python
from handwriting_engine.postprocess import correct
out = correct(vlm_output, domain="biology", use_trained=True)
```

## Architecture

```
synthetic_data.py  ─┐
                    ├─→ dataset.py ─→ train.py ─→ checkpoint
corpus.py          ─┘                                │
                                                     ▼
                                              corrector.py
                                                     │
                                                     ▼
                                  postprocess.correct() ─→ caller
                                              ▲
                                              │
                                  correct_domain_terms (heuristic)
```

Order matters at inference: heuristic runs first (high precision, cheap),
trained model runs second on the heuristic output (fixes what's left).
Reversed order tends to let the trained model introduce errors the
heuristic then can't undo.

## Synthetic data

The corruption pipeline simulates realistic VLM/HTR error patterns:

| Pattern | Example | Source |
|---------|---------|--------|
| Pair confusion | `rn`↔`m`, `cl`↔`d`, `ii`↔`u`, `oo`↔`co`, `vv`↔`w` | Classical OCR confusion tables |
| Letter substitution | `a`↔`o`, `e`↔`c`, `i`↔`l`, `u`↔`v` | HTR shape similarity |
| Digit/letter | `0`↔`o`, `1`↔`l`/`I`, `5`↔`S`, `8`↔`B` | Visual similarity |
| Doubling | `cell` → `celll` | HTR repeated stroke |
| Dropping | `mitochondria` → `mitcondria` | HTR missed letter |
| Transposition | `mitochondria` → `mitochondira` | HTR ordering error |
| Smush/split | `the cell` → `thecell`, `cell` → `ce ll` | Word boundary ambiguity |
| Capitalization slip | `Hello` → `hello` | HTR sentence-initial caps |
| Punctuation drop | `seen.` → `seen` | HTR terminal mark loss |
| Diacritic strip | `résumé` → `resume` | VLM standard behavior |

Three difficulty configs (light / default / aggressive) sampled per-example
so the model sees a CER spread from ~0.5% to ~10%.

## Caveats

**Synthetic-only training has a sim-to-real gap.** The real OCR error
distribution from Gemini / Claude / GPT-4 vision is not perfectly captured by
the synthetic corruption pipeline. A small real-data fine-tune (a few hundred
to a few thousand `(VLM_output, ground_truth)` pairs) lifts production
quality significantly.

The handwriting engine's Phase 7 (IAM ingestion) will produce exactly that
data. Plan: train v0 on synthetic, fine-tune v1 on synthetic + IAM real pairs.

## File map

- `synthetic_data.py` — corruption patterns + pipeline
- `corpus.py` — clean reference text generator (lab notebook templates +
  domain vocab + optional system wordlist)
- `dataset.py` — `(corrupted, clean)` pair builder + torch Dataset wrapper
- `train.py` — manual PyTorch training loop (no `accelerate` dep)
- `corrector.py` — inference singleton (lazy load, beam search, chunking)
- `eval.py` — A/B harness (CER for input vs heuristic vs trained vs combined)

## Testing

22 unit tests live in `tests/test_trained_correction.py`. The trained-model
integration test is gated on `HE_TRAINED_CORRECTOR_PATH` so the suite runs
without a checkpoint.
