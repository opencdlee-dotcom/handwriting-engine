"""Stage 3 retraining pipeline: fine-tune the trained corrector on real
(TrOCR_output, ground_truth) pairs harvested from the benchmark DB.

This is a one-shot orchestrator wrapping three things that already exist:

1. `dataset.from_benchmark_db(providers=['trocr'])` — pulls real pairs.
2. `train.main(argv)` — fine-tunes the v1 synthetic checkpoint into v3.
3. `eval.evaluate_synthetic(...)` — measures v1-vs-v3 CER delta.

The pipeline gates on a minimum pair count: synthetic-only training is what
produced hallucinations on hard cases (see `EVAL-RESULTS.md`), so retraining
on a tiny real set isn't worth the GPU time and would just shift the bias.
Default minimum is 50 pairs; tests can drop this via `pairs_min`.

Wired into the CLI as `trained-correction retrain-from-trocr`. Users run
this once after `bench-my-pages.py` or `benchmark sweep` populates the DB
with TrOCR outputs against ground-truthed samples.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


DEFAULT_V1_PATH = Path.home() / ".handwriting-engine" / "models" / "trained-corrector-v1"
DEFAULT_V3_PATH = Path.home() / ".handwriting-engine" / "models" / "trained-corrector-v3"
DEFAULT_PAIRS_MIN = 50


def _remediation_message(n_pairs: int) -> str:
    """User-facing remediation when pair count falls short.

    Names both data-acquisition entry points so the user can pick whichever
    matches their setup (one-off pages vs full IAM sweep).
    """
    return (
        f"Only {n_pairs} TrOCR (output, ground_truth) pair(s) in the benchmark DB — "
        f"need at least {DEFAULT_PAIRS_MIN} to retrain meaningfully.\n"
        f"  Remediation: run `bench-my-pages.py <dir>` to add ground-truthed lab pages, "
        f"OR run `python3 -m handwriting_engine.cli benchmark sweep --provider gemini` "
        f"with TrOCR enabled (`pip install -e '.[trained-correction]'`) so the "
        f"`trocr_local` strategy populates real pairs.\n"
        f"  Once the DB has enough TrOCR pairs, re-run this command."
    )


def _count_trocr_pairs(db_path: str | None = None) -> int:
    """Count distinct (TrOCR_output, ground_truth) pairs in the benchmark DB.

    Reuses the existing `from_benchmark_db` ingestion path so the count
    matches what training would actually see (deduped, identical-pair
    skipped, length-filtered).
    """
    from handwriting_engine.trained_correction.dataset import from_benchmark_db
    pairs = from_benchmark_db(db_path=db_path, providers=["trocr"])
    return len(pairs)


def _build_train_argv(
    db_path: str | None,
    v1_path: Path,
    v3_path: Path,
    num_epochs: int,
) -> list[str]:
    """Construct the argv for `train.main()` — continuing from v1 into v3."""
    argv: list[str] = [
        "--from-benchmark-db",
        "--benchmark-providers", "trocr",
        "--continue-from", str(v1_path),
        "--output-dir", str(v3_path),
        "--num-epochs", str(num_epochs),
    ]
    if db_path is not None:
        argv.extend(["--benchmark-db-path", db_path])
    return argv


def _eval_with_checkpoint(
    checkpoint_path: Path,
    n_pairs: int,
    seed: int,
) -> float | None:
    """Run the synthetic eval harness with `checkpoint_path` as the active
    trained corrector, return `avg_cer_combined_gated` (or None on failure).

    Sets `HE_TRAINED_CORRECTOR_PATH` for the duration of the call and busts
    the `get_default_corrector` lru_cache so the new path is picked up.
    """
    if not checkpoint_path.is_dir():
        logger.warning("Skipping eval: checkpoint not found at %s", checkpoint_path)
        return None

    from handwriting_engine.trained_correction import corrector as _corrector_mod
    from handwriting_engine.trained_correction.eval import evaluate_synthetic

    prior = os.environ.get("HE_TRAINED_CORRECTOR_PATH")
    os.environ["HE_TRAINED_CORRECTOR_PATH"] = str(checkpoint_path)
    try:
        # Bust the lru_cache so the corrector reloads the new checkpoint.
        _corrector_mod.get_default_corrector.cache_clear()
        result = evaluate_synthetic(n_pairs=n_pairs, seed=seed)
    finally:
        if prior is None:
            os.environ.pop("HE_TRAINED_CORRECTOR_PATH", None)
        else:
            os.environ["HE_TRAINED_CORRECTOR_PATH"] = prior
        _corrector_mod.get_default_corrector.cache_clear()
    return result.avg_cer_combined_gated


def retrain_from_trocr(
    db_path: str | None = None,
    v1_path: Path | str | None = None,
    v3_path: Path | str | None = None,
    num_epochs: int = 3,
    pairs_min: int = DEFAULT_PAIRS_MIN,
    run_eval: bool = True,
    eval_n_pairs: int = 200,
    eval_seed: int = 9999,
    train_main: Callable[[list[str]], int] | None = None,
    out_stream=None,
    err_stream=None,
) -> int:
    """Run Stage 3: retrain v1 → v3 on real TrOCR pairs, then A/B eval.

    Returns 0 on success, 1 on insufficient data, 2 on training failure.

    Args:
        db_path: Override the benchmark DB path.
        v1_path: Synthetic-trained checkpoint to continue from. Defaults to
            ~/.handwriting-engine/models/trained-corrector-v1.
        v3_path: Output directory for the retrained checkpoint. Defaults to
            ~/.handwriting-engine/models/trained-corrector-v3.
        num_epochs: Training epochs.
        pairs_min: Minimum (TrOCR_output, ground_truth) pair count required
            to proceed. Tests pass a small number; production keeps the
            default (50).
        run_eval: Run the v1-vs-v3 CER comparison after training.
        eval_n_pairs: Synthetic pairs for the post-training eval. The eval
            measures generalization, not memorization, so synthetic pairs
            are appropriate even though training used real pairs.
        eval_seed: RNG seed for the synthetic eval set.
        train_main: Optional dependency-injected `train.main` (test hook).
            Defaults to the real `train.main` from `train.py`.
        out_stream: Optional stdout writer (test hook). Defaults to sys.stdout.
        err_stream: Optional stderr writer (test hook). Defaults to sys.stderr.
    """
    out = out_stream if out_stream is not None else sys.stdout
    err = err_stream if err_stream is not None else sys.stderr

    v1 = Path(v1_path) if v1_path else DEFAULT_V1_PATH
    v3 = Path(v3_path) if v3_path else DEFAULT_V3_PATH

    n_pairs = _count_trocr_pairs(db_path=db_path)
    if n_pairs < pairs_min:
        print(_remediation_message(n_pairs), file=err)
        return 1

    print(
        f"Found {n_pairs} TrOCR (output, ground_truth) pair(s) — proceeding with "
        f"retrain (continue from {v1}, output to {v3}, {num_epochs} epochs).",
        file=out,
    )

    if train_main is None:
        from handwriting_engine.trained_correction.train import main as train_main

    argv = _build_train_argv(db_path=db_path, v1_path=v1, v3_path=v3, num_epochs=num_epochs)
    rc = train_main(argv)
    if rc != 0:
        print(f"ERROR: training exited with code {rc}", file=err)
        return 2

    if not run_eval:
        return 0

    print("\nRunning v1-vs-v3 CER comparison on synthetic eval set...", file=out)
    v1_cer = _eval_with_checkpoint(v1, n_pairs=eval_n_pairs, seed=eval_seed)
    v3_cer = _eval_with_checkpoint(v3, n_pairs=eval_n_pairs, seed=eval_seed)

    if v1_cer is None or v3_cer is None:
        print(
            "WARNING: could not run full A/B eval (one or both checkpoints unavailable). "
            "Run `trained-correction eval` manually after pointing "
            "HE_TRAINED_CORRECTOR_PATH at each checkpoint.",
            file=err,
        )
        return 0

    delta = v3_cer - v1_cer
    sign = "-" if delta < 0 else "+"
    print(
        f"\nCER (combined+gated) — v1: {v1_cer:.4f}  v3: {v3_cer:.4f}  "
        f"delta: {sign}{abs(delta):.4f}  ({'v3 better' if delta < 0 else 'v3 worse or unchanged'})",
        file=out,
    )
    return 0
