"""torch Dataset for (corrupted, clean) training pairs.

Pairs are generated lazily via the corpus + synthetic_data modules. Each
example is tokenized at __getitem__ time using the supplied tokenizer (HF
tokenizers are deterministic given input, so this stays reproducible under
seeded RNG).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from handwriting_engine.trained_correction.corpus import generate_sentences, generate_paragraphs
from handwriting_engine.trained_correction.synthetic_data import (
    CorruptionConfig,
    make_pair,
    sample_difficulty,
)


@dataclass(frozen=True)
class CorrectionExample:
    """One training example: corrupted input, clean target."""
    corrupted: str
    clean: str


def build_pairs(
    n: int,
    seed: int = 42,
    use_paragraphs: bool = True,
    sample_difficulty_per_example: bool = True,
    use_system_wordlist: bool = True,
) -> list[CorrectionExample]:
    """Generate `n` (corrupted, clean) pairs.

    `sample_difficulty_per_example=True` mixes light / medium / aggressive
    corruption configs so the corrector sees a difficulty spread.
    """
    rng = random.Random(seed)
    if use_paragraphs:
        clean_iter = generate_paragraphs(n, rng, use_system_wordlist=use_system_wordlist)
    else:
        clean_iter = generate_sentences(n, rng, use_system_wordlist=use_system_wordlist)

    out: list[CorrectionExample] = []
    for clean in clean_iter:
        cfg = sample_difficulty(rng) if sample_difficulty_per_example else CorruptionConfig()
        corrupted, clean_out = make_pair(clean, rng, cfg)
        out.append(CorrectionExample(corrupted=corrupted, clean=clean_out))
    return out


def from_benchmark_db(
    db_path: str | None = None,
    providers: list[str] | None = None,
    strategies: list[str] | None = None,
    min_text_chars: int = 10,
) -> list[CorrectionExample]:
    """Load real `(VLM_output, ground_truth)` pairs from the engine's benchmark DB.

    Joins `provider_outputs` and `ground_truths` on `sample_id`. Each row produces
    one (corrupted=VLM output, clean=ground truth) pair.

    Returns [] (with a logged debug message) if the DB doesn't exist yet — this is
    the common case before Phase 7 ingestion lands. Callers can mix this with
    synthetic pairs from `build_pairs()` to fine-tune.

    Args:
        db_path: Override the default SQLite path (~/.handwriting-engine/benchmark.db).
        providers: Filter to specific providers (e.g. ["gemini", "claude"]). None = all.
        strategies: Filter to specific consensus strategies. None = all.
        min_text_chars: Drop pairs where either text is shorter than this. Filters out
            degenerate / corrupt rows that would teach the corrector wrong patterns.
    """
    import logging
    import sqlite3
    from pathlib import Path

    log = logging.getLogger(__name__)
    if db_path is None:
        db_path = str(Path.home() / ".handwriting-engine" / "benchmark.db")
    if not Path(db_path).is_file():
        log.debug("Benchmark DB not found at %s — returning empty pair list", db_path)
        return []

    sql_parts = [
        "SELECT po.output_text AS corrupted, gt.text AS clean",
        "FROM provider_outputs po",
        "JOIN ground_truths gt ON po.sample_id = gt.sample_id",
        "WHERE po.error IS NULL",
        "  AND length(po.output_text) >= ?",
        "  AND length(gt.text) >= ?",
    ]
    params: list = [min_text_chars, min_text_chars]
    if providers:
        placeholders = ",".join("?" * len(providers))
        sql_parts.append(f"  AND po.provider IN ({placeholders})")
        params.extend(providers)
    if strategies:
        placeholders = ",".join("?" * len(strategies))
        sql_parts.append(f"  AND po.strategy IN ({placeholders})")
        params.extend(strategies)
    sql = "\n".join(sql_parts)

    out: list[CorrectionExample] = []
    seen: set[tuple[str, str]] = set()
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(sql, params)
            for corrupted, clean in cur:
                # Dedupe identical (corrupted, clean) pairs — multiple providers can
                # produce the same output; we don't want to weight a pair more heavily
                # just because three providers agreed on the wrong answer.
                key = (corrupted, clean)
                if key in seen:
                    continue
                if corrupted == clean:
                    continue  # No correction signal — skip
                seen.add(key)
                out.append(CorrectionExample(corrupted=corrupted, clean=clean))
    except sqlite3.DatabaseError as e:
        log.warning("Failed reading benchmark DB %s: %s", db_path, e)
        return []

    log.info("Loaded %d real (corrupted, clean) pairs from %s", len(out), db_path)
    return out


def split_pairs(
    pairs: list[CorrectionExample],
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42,
) -> tuple[list[CorrectionExample], list[CorrectionExample], list[CorrectionExample]]:
    """Deterministic train/val/test split of generated pairs."""
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    n_val = int(len(pairs) * val_frac)
    n_test = int(len(pairs) * test_frac)
    val_idx = set(indices[:n_val])
    test_idx = set(indices[n_val : n_val + n_test])
    train, val, test = [], [], []
    for i, p in enumerate(pairs):
        if i in val_idx:
            val.append(p)
        elif i in test_idx:
            test.append(p)
        else:
            train.append(p)
    return train, val, test


# =====================================================================
# torch Dataset (lazy import — torch is an optional dep)
# =====================================================================

def make_torch_dataset(
    pairs: list[CorrectionExample],
    tokenizer: "PreTrainedTokenizerBase",
    max_input_length: int = 512,
    max_target_length: int = 512,
):
    """Return a torch Dataset that tokenizes (corrupted, clean) on access.

    Lazy torch import so the corpus / synthetic_data modules stay import-
    safe without the optional dep.
    """
    import torch
    from torch.utils.data import Dataset

    class _CorrectionDataset(Dataset):
        def __init__(self, pairs_, tokenizer_, max_in_, max_out_):
            self.pairs = pairs_
            self.tokenizer = tokenizer_
            self.max_in = max_in_
            self.max_out = max_out_

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx: int):
            ex = self.pairs[idx]
            # Instruction-style prefix. Specifically NOT "correct: " — vanilla
            # t5-small was pretrained with prefixes like "translate English to
            # German: " and treats "correct: ..." as a translation task. flan-t5
            # follows arbitrary instructions, but a directive prefix avoids any
            # residual pretraining bias and helps both bases learn faster.
            prompt = f"Fix OCR errors in this text: {ex.corrupted}"
            model_inputs = self.tokenizer(
                prompt,
                max_length=self.max_in,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            labels = self.tokenizer(
                ex.clean,
                max_length=self.max_out,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).input_ids[0]
            # HF convention: -100 in labels = ignore in loss
            labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
            return {
                "input_ids": model_inputs.input_ids[0],
                "attention_mask": model_inputs.attention_mask[0],
                "labels": labels,
            }

    return _CorrectionDataset(pairs, tokenizer, max_input_length, max_target_length)
