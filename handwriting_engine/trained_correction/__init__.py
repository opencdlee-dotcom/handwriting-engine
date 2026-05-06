"""Trained post-correction layer.

Heuristic post-correction (handwriting_engine.postprocess) tops out at
edit-distance-1 word + bigram lookups against curated wordlists. Past that,
multi-character OCR confusions, doubled letters, smushed words, and context-
sensitive corrections need a learned model.

This subpackage provides:
- synthetic_data: realistic OCR corruption patterns for generating training pairs
- corpus: clean reference text builder (lab/science/general)
- dataset: torch Dataset wrappers for (corrupted, clean) pairs
- train: ByT5-small fine-tuning entrypoint
- corrector: inference-time load+predict interface

Optional dependency: install with `pip install handwriting-engine[trained-correction]`.

Caveats:
- Synthetic-only training has a known sim-to-real gap. Plan for a small real-data
  fine-tune once Phase 7 (IAM ingestion) lands and (VLM_output, ground_truth)
  pairs are available.
- The trained corrector is OFF BY DEFAULT in the engine. Enable per-call via
  `correct(text, use_trained=True)` or via env var HE_USE_TRAINED_CORRECTOR=1.
"""

from __future__ import annotations
