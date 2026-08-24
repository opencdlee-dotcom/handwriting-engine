"""Fine-tune ByT5-small on synthetic OCR-error → clean-text pairs.

Default model: google/byt5-small (~300MB, ~300M params). Byte-level tokenizer
makes this a natural fit for handwriting correction — there's no tokenizer
drift from misspellings, and byte-level avoids subword-vocabulary issues with
unusual scientific terms.

Usage (from the engine repo):

    python -m handwriting_engine.trained_correction.train \\
        --output-dir ./ckpt/corrector-v1 \\
        --num-pairs 50000 \\
        --num-epochs 2 \\
        --batch-size 8

On Apple Silicon MPS the model fits comfortably; bf16 is unsupported on MPS
so we run fp32 by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    """Seed Python, numpy, torch (CPU + MPS / CUDA if available)."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass


def _resolve_device(prefer: str = "auto") -> str:
    import torch
    if prefer == "cpu":
        return "cpu"
    if prefer in ("mps", "auto") and torch.backends.mps.is_available():
        return "mps"
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="google/flan-t5-small",
                        help="HF model id to fine-tune. Default flan-t5-small (~80M params, "
                             "instruction-tuned — follows task prefixes naturally and avoids "
                             "the t5-small 'correct: → translate to German' confusion). "
                             "Alternatives: google-t5/t5-small (60M, lightest), "
                             "google/byt5-small (300M, byte-level — robust to misspellings "
                             "but slow on MPS).")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for checkpoints + manifest")
    parser.add_argument("--num-pairs", type=int, default=50000,
                        help="Total synthetic training pairs to generate")
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Higher than typical T5 default — synthetic data is forgiving")
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--no-system-wordlist", action="store_true",
                        help="Skip /usr/share/dict/words for general English")
    parser.add_argument("--from-benchmark-db", action="store_true",
                        help="Mix real (VLM_output, ground_truth) pairs from "
                             "~/.handwriting-engine/benchmark.db into the training corpus. "
                             "Real pairs are duplicated 3x relative to synthetic to up-weight "
                             "the real distribution. No-op if the DB doesn't exist.")
    parser.add_argument("--benchmark-db-path", default=None,
                        help="Override the default benchmark.db path")
    parser.add_argument("--benchmark-providers", default=None,
                        help="Comma-separated list of providers to include "
                             "(e.g. 'gemini,claude'). Default: all providers.")
    parser.add_argument("--real-data-weight", type=int, default=3,
                        help="How many times each real pair appears in the corpus "
                             "relative to synthetic. Default 3.")
    parser.add_argument("--continue-from", default=None,
                        help="Path to an existing checkpoint to continue fine-tuning from. "
                             "Overrides --model-name. Use to fine-tune the synthetic v0 "
                             "model on real Phase 7 IAM data.")
    parser.add_argument("--quick", action="store_true",
                        help="Tiny run for smoke-testing the pipeline")
    args = parser.parse_args(argv)

    if args.quick:
        # --quick caps things to a tiny smoke run regardless of other flags
        args.num_pairs = min(args.num_pairs, 200)
        args.num_epochs = 1
        args.max_input_length = min(args.max_input_length, 128)
        args.max_target_length = min(args.max_target_length, 128)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports — these are optional deps
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
    except ImportError as e:
        logger.error(
            "Missing optional deps. Install with: "
            "pip install handwriting-engine[trained-correction]\n  Underlying: %s", e
        )
        return 2

    from handwriting_engine.trained_correction.dataset import (
        build_pairs,
        from_benchmark_db,
        split_pairs,
        make_torch_dataset,
    )

    device = _resolve_device(args.device)
    logger.info("Using device: %s", device)

    # ---- Data ----
    logger.info("Generating %d synthetic (corrupted, clean) pairs...", args.num_pairs)
    pairs = build_pairs(
        n=args.num_pairs,
        seed=args.seed,
        use_paragraphs=True,
        use_system_wordlist=not args.no_system_wordlist,
    )

    real_pairs_count = 0
    if args.from_benchmark_db:
        providers_filter = (
            [p.strip() for p in args.benchmark_providers.split(",") if p.strip()]
            if args.benchmark_providers else None
        )
        real_pairs = from_benchmark_db(
            db_path=args.benchmark_db_path,
            providers=providers_filter,
        )
        if real_pairs:
            real_pairs_count = len(real_pairs)
            # Up-weight real pairs by replication; mixing into synthetic at higher
            # weight encourages the model to track real-world distribution.
            pairs = pairs + real_pairs * args.real_data_weight
            logger.info(
                "Mixed in %d real pairs (replicated %dx → %d effective real examples)",
                real_pairs_count, args.real_data_weight, real_pairs_count * args.real_data_weight,
            )
        else:
            logger.info("No real pairs available (DB empty or missing); using synthetic only")

    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    logger.info("Train: %d  Val: %d  Test: %d", len(train_pairs), len(val_pairs), len(test_pairs))

    # ---- Tokenizer + model ----
    model_source = args.continue_from if args.continue_from else args.model_name
    if args.continue_from:
        logger.info("Continuing from checkpoint: %s", args.continue_from)
    else:
        logger.info("Loading tokenizer + model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_source)
    model.to(device)
    model.train()

    # ---- Datasets ----
    train_ds = make_torch_dataset(
        train_pairs, tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )
    val_ds = make_torch_dataset(
        val_pairs, tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    ) if val_pairs else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0) if val_ds else None

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = max(1, int(0.06 * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ---- Train loop ----
    logger.info("Starting training: %d total steps (warmup=%d)", total_steps, warmup_steps)
    best_val_loss = float("inf")
    global_step = 0
    train_losses = []

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        running_count = 0
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item() * args.gradient_accumulation_steps
            running_count += 1

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg = running_loss / max(1, running_count)
                    logger.info(
                        "epoch %d  step %d/%d  loss %.4f  lr %.2e",
                        epoch, global_step, total_steps, avg, scheduler.get_last_lr()[0],
                    )
                    train_losses.append(avg)

                if val_loader is not None and global_step % args.save_steps == 0:
                    val_loss = _eval_loss(model, val_loader, device)
                    logger.info("epoch %d  step %d  val_loss %.4f", epoch, global_step, val_loss)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        _save_checkpoint(model, tokenizer, output_dir)
                        logger.info("Saved best checkpoint (val_loss=%.4f)", val_loss)
                    model.train()

        # End-of-epoch eval + save
        if val_loader is not None:
            val_loss = _eval_loss(model, val_loader, device)
            logger.info("end of epoch %d  val_loss %.4f", epoch, val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(model, tokenizer, output_dir)
                logger.info("Saved best checkpoint (val_loss=%.4f)", val_loss)
            model.train()
        else:
            _save_checkpoint(model, tokenizer, output_dir)

    # Always save final state if no val loader (otherwise best is already saved)
    if val_loader is None:
        _save_checkpoint(model, tokenizer, output_dir)

    final_train_loss = train_losses[-1] if train_losses else float("nan")

    # ---- Manifest ----
    has_real = real_pairs_count > 0
    manifest = {
        "schema_version": 1,
        "base_model": args.model_name,
        "continued_from": args.continue_from,
        "training": {
            "num_pairs_synthetic": args.num_pairs,
            "num_pairs_real": real_pairs_count,
            "real_data_weight": args.real_data_weight if has_real else 0,
            "num_train": len(train_pairs),
            "num_val": len(val_pairs),
            "num_test": len(test_pairs),
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "device": device,
            "max_input_length": args.max_input_length,
            "max_target_length": args.max_target_length,
            "total_steps": total_steps,
        },
        "metrics": {
            "final_train_loss": final_train_loss,
            "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        },
        "data_provenance": {
            "source": "synthetic+real" if has_real else "synthetic",
            "corpus_version": "v0",
            "system_wordlist": not args.no_system_wordlist,
            "real_pairs_from": args.benchmark_db_path or "~/.handwriting-engine/benchmark.db" if has_real else None,
        },
        "caveats": (
            ["Synthetic+real training — sim-to-real gap reduced by mixing real Phase 7 pairs."]
            if has_real else
            ["Synthetic-only training — has known sim-to-real gap.",
             "Plan a small real-data fine-tune (Phase 7 IAM data) before claiming production parity."]
        ),
    }

    # ---- Hold-out test loss ----
    if test_pairs:
        test_ds = make_torch_dataset(
            test_pairs, tokenizer,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loss = _eval_loss(model, test_loader, device)
        logger.info("Held-out test loss: %.4f", test_loss)
        manifest["metrics"]["test_loss"] = test_loss

    with (output_dir / "training_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Training complete. Checkpoint at: %s", output_dir)
    return 0


def _eval_loss(model, loader, device) -> float:
    """Average loss on a DataLoader. Returns finite float."""
    import torch
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            total += float(out.loss.item())
            count += 1
    return total / max(1, count)


def _save_checkpoint(model, tokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    sys.exit(main())
