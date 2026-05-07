"""Inference-time interface for the trained corrector.

Singleton-style cached loader so repeated calls to `correct(text)` reuse the
same model + tokenizer; we never load weights more than once per process.
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Default checkpoint search order. First hit wins.
_DEFAULT_CHECKPOINT_PATHS = [
    Path.home() / ".handwriting-engine" / "models" / "trained-corrector-v1",
    Path.cwd() / "ckpt" / "trained-corrector-v1",
]


def _find_default_checkpoint() -> Path | None:
    for p in _DEFAULT_CHECKPOINT_PATHS:
        if p.is_dir() and (p / "config.json").is_file():
            return p
    env_path = os.environ.get("HE_TRAINED_CORRECTOR_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_dir():
            return p
    return None


class TrainedCorrector:
    """Wrapper around a fine-tuned ByT5-small (or compatible seq2seq) checkpoint.

    Lazy load on first `correct()` call. Subsequent calls reuse the model.
    Thread-safe via a single load lock.
    """

    _instance_lock = threading.Lock()

    def __init__(self, checkpoint_path: Path | str, device: str = "auto"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._resolved_device: str | None = None

    def _resolve_device(self) -> str:
        if self._resolved_device is not None:
            return self._resolved_device
        import torch
        if self.device == "cpu":
            d = "cpu"
        elif self.device in ("mps", "auto") and torch.backends.mps.is_available():
            d = "mps"
        elif self.device in ("cuda", "auto") and torch.cuda.is_available():
            d = "cuda"
        else:
            d = "cpu"
        self._resolved_device = d
        return d

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._instance_lock:
            if self._model is not None:
                return
            try:
                import torch  # noqa: F401
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            except ImportError as e:
                raise ImportError(
                    "Trained corrector requires optional deps. Install with: "
                    "pip install handwriting-engine[trained-correction]"
                ) from e

            logger.info("Loading trained corrector from %s", self.checkpoint_path)
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_path))
            self._model = AutoModelForSeq2SeqLM.from_pretrained(str(self.checkpoint_path))
            device = self._resolve_device()
            self._model.to(device)
            self._model.eval()
            logger.info("Trained corrector ready (device=%s)", device)

    def correct(
        self,
        text: str,
        max_length: int = 512,
        num_beams: int = 4,
        chunk_length: int = 240,
    ) -> str:
        """Correct `text`, returning a (hopefully) cleaner version.

        Long inputs are split into chunks (`chunk_length` characters at sentence
        boundaries when possible) — ByT5-small was trained at max_length=256
        and degrades on longer inputs.
        """
        if not text or not text.strip():
            return text
        self._ensure_loaded()
        chunks = self._split_for_inference(text, chunk_length)
        outputs = [self._correct_chunk(c, max_length, num_beams) for c in chunks]
        return " ".join(outputs).strip()

    def _correct_chunk(self, text: str, max_length: int, num_beams: int) -> str:
        import torch
        device = self._resolve_device()
        prompt = f"Fix OCR errors in this text: {text}"
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=0,  # don't suppress n-grams (we want exact text)
            )
        decoded = self._tokenizer.decode(generated[0], skip_special_tokens=True)
        return decoded

    @staticmethod
    def _split_for_inference(text: str, target_len: int) -> list[str]:
        """Split text at sentence boundaries when possible, falling back to
        whitespace, falling back to fixed-length cuts. Each chunk ≤ target_len."""
        if len(text) <= target_len:
            return [text]
        # Try sentence boundary splits
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        cur = ""
        for sent in sentences:
            if not sent:
                continue
            if len(cur) + len(sent) + 1 <= target_len:
                cur = (cur + " " + sent).strip() if cur else sent
            else:
                if cur:
                    chunks.append(cur)
                if len(sent) <= target_len:
                    cur = sent
                else:
                    # Hard chunk an over-long sentence
                    for i in range(0, len(sent), target_len):
                        chunks.append(sent[i : i + target_len])
                    cur = ""
        if cur:
            chunks.append(cur)
        return chunks


@lru_cache(maxsize=1)
def get_default_corrector() -> TrainedCorrector | None:
    """Return the process-wide default corrector, or None if no checkpoint found."""
    ckpt = _find_default_checkpoint()
    if ckpt is None:
        return None
    return TrainedCorrector(ckpt)


def correct(text: str, **kwargs: Any) -> str:
    """Module-level convenience: correct text with the default corrector.

    Returns input unchanged if no checkpoint is available.
    """
    corrector = get_default_corrector()
    if corrector is None:
        logger.debug("No trained corrector checkpoint found; returning text unchanged")
        return text
    return corrector.correct(text, **kwargs)


def is_available() -> bool:
    """True iff a default checkpoint can be located on disk."""
    return _find_default_checkpoint() is not None
