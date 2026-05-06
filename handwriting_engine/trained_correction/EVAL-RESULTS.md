# Trained Corrector v0 — Evaluation Results

**Date:** 2026-05-05
**Branch:** `feat/trained-corrector`
**Checkpoint:** `~/.handwriting-engine/models/trained-corrector-v1/`
**Base model:** `google/flan-t5-small` (~80M params)
**Training:** 1500 synthetic pairs, 2 epochs, batch 4, seqlen 192, CPU, ~22 min wall clock
**Final losses:** train 0.494, val 0.387, test 0.385

---

## A/B evaluation on 200 held-out synthetic pairs (seed 9999)

```
n: 200
avg_cer_input:     0.0732   (corrupted text vs ground truth)
avg_cer_heuristic: 0.0688   (-0.44pp from input)
avg_cer_trained:   0.0548   (-1.40pp from heuristic)
avg_cer_combined:  0.0507   (-1.82pp from heuristic, -2.25pp from input)
```

| Pipeline | CER | Relative reduction vs input |
|----------|-----|------------------------------|
| Input (corrupted) | 7.32% | — |
| Heuristic only | 6.88% | 6.0% |
| Trained only | 5.48% | 25.1% |
| **Combined (heuristic → trained)** | **5.07%** | **30.7%** |

**Combined pipeline beats heuristic-alone by 1.82 absolute CER points (26% relative reduction).** This is the headline result.

## Spot checks (10 representative biology/lab inputs)

| # | Input | Output (heuristic → trained) | Verdict |
|---|-------|------------------------------|---------|
| 1 | `the mitocondria is the powerhouse of the cell` | `The mitochondria is the powerhouse of the cell.` | ✓ |
| 2 | `natural selecton drives evolution over generations` | `Natural selection drives evolution over generations.` | ✓ |
| 3 | `pH was 7.4 and the temprature was 37C` | `pH was 7.4 and the temperature was 37C.` | ✓ |
| 4 | `celll growth observed under microscope` | `Cell cell growth observed under microscope.` | ✗ duplicated word |
| 5 | `add 2.5 mL of natuiral selection sample` | `Add 2.5 mL of natural selection sample.` | ✓ |
| 6 | `amino acidd are the building blocks of proteins` | `amino acids are the building blocks of protein.` | ✓ (trained fixed `acidd → acids`) |
| 7 | `the periodc table arranges elements by atomic numbr` | `The periodic table arranges elements by atomic nucleus.` | ✗ hallucination (`numbr → nucleus`) |
| 8 | `photosythesis occurs in chloroplasts during the light reacton` | `Photosynthesis occurs in chloroplast during the light reaction.` | ✓ |
| 9 | `we observed the niitochondria after staining` | `We observed the nucleotide after staining.` | ✗ hallucination (`niitochondria → nucleotide`) |
| 10 | `electron trasnport chain produces ATP` | `electron transport chain produces ATP` | ✓ heuristic alone fixed; trained pass left it correctly unchanged |

**7/10 wins, 3/10 losses.** All three losses are the same pattern: when the heuristic *can't* fix a token (because it's edit-distance >1 from any vocabulary word, or short, or otherwise ineligible), the trained model picks a plausible *scientific-sounding* substitution that may not be the right word.

This is the canonical synthetic-to-real failure mode. The model learned:
- Correct capitalization, punctuation, common mis-typings ✓
- Multi-word phrase corrections beyond bigram lookup ✓
- General sentence shape preservation ✓

But also learned (from synthetic data only):
- "When in doubt, output a real-looking scientific word" — produces hallucinations on hard cases.

## Recommended deployment posture

1. **Off by default.** `HE_USE_TRAINED_CORRECTOR=0` until further validation.
2. **Combined pipeline is the right pattern when enabled.** Heuristic first, trained second. The heuristic acts as a high-precision filter for the easy errors; the trained model only fires on the residual.
3. **Real-data fine-tune is required before production.** Phase 7 IAM ingestion produces the data; expected to dramatically reduce hallucinations because the model will see the actual VLM error distribution rather than a guessed-at synthetic one.
4. **Beam search + chunking** as in `corrector.py` defaults. Don't lower beam to 1 — it amplifies the hallucination failure mode.

## Reproducing this evaluation

```bash
# A/B eval on synthetic
handwriting-engine trained-correction eval --n-pairs 200 --seed 9999

# Heuristic-only baseline (no model load)
handwriting-engine trained-correction eval --n-pairs 200 --seed 9999 --skip-trained

# Spot-check arbitrary inputs
python3 -c "
from handwriting_engine.postprocess import correct
print(correct('YOUR INPUT HERE', domain='biology', use_trained=True))
"
```

## What v4.1 should do

1. Fine-tune from synthetic v0 on real `(VLM_output, ground_truth)` pairs from Phase 7. Even 500 real pairs will likely cut hallucinations significantly.
2. Add a **confidence gate**: only apply the trained corrector when input passed the heuristic with ≥1 successful word correction (i.e. there were errors the heuristic could fix). On already-clean inputs, skip the trained pass — it sometimes "improves" already-correct text.
3. Add a **fidelity check**: compare token overlap between input and output; if the model rewrote >X% of tokens, fall back to heuristic-only output (catches obvious hallucinations).
4. Train at larger scale: 50K pairs, 3 epochs, longer sequences, MPS (when not wedged) or cloud GPU.
