"""
TrOCR vision provider using HuggingFace transformers.

microsoft/trocr-base-handwritten achieves ~3.42% CER on IAM.
Fine-tuning with as few as 5 lines per writer makes it very effective
for single-writer manuscript transcription.

Install: pip install transformers torch torchvision

Apple Silicon: Uses MPS if available, falls back to CPU.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TROCR_MODEL = "microsoft/trocr-base-handwritten"

# HF's `model.generate()` defaults to max_length=21 for VisionEncoderDecoder
# models, which truncates anything past ~20 tokens (one short line). Lab
# notebooks routinely exceed that. 256 tokens covers a long single line plus
# headroom for short multi-line crops; tunable via env for power users.
DEFAULT_TROCR_MAX_NEW_TOKENS = 256
TROCR_MAX_NEW_TOKENS_ENV = "HE_TROCR_MAX_NEW_TOKENS"

# Beam search reduces CER vs greedy on TrOCR with no extra training. 4 beams
# is the IAM sweet spot in the original paper; tunable via env so power users
# can dial up/down for cost/accuracy.
DEFAULT_TROCR_NUM_BEAMS = 4
TROCR_NUM_BEAMS_ENV = "HE_TROCR_NUM_BEAMS"

# Stage 4: writer-specific fine-tuned checkpoints live here (one dir per writer).
WRITER_MODELS_DIR = Path.home() / ".handwriting-engine" / "writer-models"


def _resolve_max_new_tokens(explicit: int | None = None) -> int:
    """Resolve max_new_tokens for TrOCR generate(). Precedence: explicit > env > default."""
    if explicit is not None and explicit > 0:
        return explicit
    raw = os.getenv(TROCR_MAX_NEW_TOKENS_ENV)
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return DEFAULT_TROCR_MAX_NEW_TOKENS


def _resolve_num_beams(explicit: int | None = None) -> int:
    """Resolve num_beams for TrOCR generate(). Precedence: explicit > env > default.

    A value of 1 disables beam search (greedy). Must be >= 1.
    """
    if explicit is not None and explicit > 0:
        return explicit
    raw = os.getenv(TROCR_NUM_BEAMS_ENV)
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return DEFAULT_TROCR_NUM_BEAMS


def writer_checkpoint_dir(writer_id: str) -> Path:
    """Return the on-disk directory for a writer's fine-tuned checkpoint.

    Existence is NOT checked here — callers decide what to do when missing.
    """
    return WRITER_MODELS_DIR / writer_id


class TrOCRProvider:
    name = "trocr"

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        writer_id: str | None = None,
    ):
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch torchvision")

        import torch

        # Stage 4: prefer a writer-specific checkpoint when one exists for this
        # writer. Falls through to the configured base model otherwise so the
        # call site doesn't need to pre-check existence.
        self._writer_id = writer_id
        resolved_name: str | None = None
        if writer_id:
            ckpt = writer_checkpoint_dir(writer_id)
            if ckpt.exists() and ckpt.is_dir():
                resolved_name = str(ckpt)
                logger.info(
                    "Loading writer-specific TrOCR checkpoint for %s from %s",
                    writer_id, ckpt,
                )
            else:
                logger.info(
                    "No writer-specific TrOCR checkpoint for %s at %s; using base model",
                    writer_id, ckpt,
                )

        self._model_name = (
            resolved_name
            or model_name
            or os.getenv("TROCR_MODEL", DEFAULT_TROCR_MODEL)
        )

        if device is not None:
            self._device = device
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        logger.info("Loading TrOCR model %s on %s", self._model_name, self._device)
        self._processor = TrOCRProcessor.from_pretrained(self._model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(self._model_name)
        self._model = self._model.to(self._device)
        self._model.eval()
        self._usage = {"input_tokens": 0, "output_tokens": 0}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def usage(self) -> dict:
        return dict(self._usage)

    def _b64_to_pil(self, image_b64: str):
        from PIL import Image
        data = base64.standard_b64decode(image_b64)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img

    def _run_trocr(self, image_b64: str, max_new_tokens: int | None = None) -> str:
        import torch
        img = self._b64_to_pil(image_b64)
        pixel_values = self._processor(images=img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)
        with torch.no_grad():
            generated_ids = self._model.generate(
                pixel_values,
                max_new_tokens=_resolve_max_new_tokens(max_new_tokens),
                num_beams=_resolve_num_beams(),
                length_penalty=1.0,
                early_stopping=True,
            )
        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self._usage["output_tokens"] += len(text.split())
        img.close()
        return text

    def _run_trocr_with_confidence(
        self, image_b64: str, max_new_tokens: int | None = None,
    ) -> tuple[str, float, float]:
        """Generate text + (mean_conf, min_conf) derived from token logprobs.

        Returns ``(text, mean_conf, min_conf)`` where:
          - mean_conf = exp(mean(logprobs))   — overall sequence confidence
          - min_conf  = exp(min(logprobs))    — worst-token confidence

        The min_conf surfaces single buried bad chars that the mean averages
        away on otherwise confident lines (e.g. a wrong letter in a long word).

        With ``num_beams > 1`` the raw ``output.scores`` are beam-level scores
        whose alignment with the chosen sequence is non-trivial. We use HF's
        ``compute_transition_scores`` (post-2022 API) which returns per-token
        logprobs aligned with ``output.sequences`` regardless of beam count.

        Falls back to (text, 0.0, 0.0) when scores aren't available.
        """
        import math
        import torch

        img = self._b64_to_pil(image_b64)
        pixel_values = self._processor(images=img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)
        num_beams = _resolve_num_beams()
        with torch.no_grad():
            output = self._model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=_resolve_max_new_tokens(max_new_tokens),
                num_beams=num_beams,
                length_penalty=1.0,
                early_stopping=True,
            )

        sequences = output.sequences
        text = self._processor.batch_decode(sequences, skip_special_tokens=True)[0]
        self._usage["output_tokens"] += len(text.split())
        img.close()

        scores = getattr(output, "scores", None)
        if not scores:
            return text, 0.0, 0.0

        # compute_transition_scores aligns per-token logprobs with the chosen
        # sequence even under beam search. Shape: (batch, gen_len).
        logprobs: list[float] = []
        try:
            transition_scores = self._model.compute_transition_scores(
                sequences,
                scores,
                getattr(output, "beam_indices", None),
                normalize_logits=True,
            )
            row = transition_scores[0]
            # Filter out -inf padding that shows up at sequence end for finished beams.
            for v in row.tolist():
                if v == float("-inf") or math.isnan(v):
                    continue
                logprobs.append(float(v))
        except Exception:
            # Fallback: greedy-style alignment (only correct when num_beams == 1).
            seq = sequences[0]
            gen_ids = seq[seq.shape[0] - len(scores):]
            for step_logits, tok_id in zip(scores, gen_ids):
                log_probs_step = torch.log_softmax(step_logits[0], dim=-1)
                logprobs.append(float(log_probs_step[int(tok_id)].item()))

        if not logprobs:
            return text, 0.0, 0.0

        mean_logprob = sum(logprobs) / len(logprobs)
        min_logprob = min(logprobs)
        mean_conf = max(0.0, min(1.0, math.exp(mean_logprob)))
        min_conf = max(0.0, min(1.0, math.exp(min_logprob)))
        return text, mean_conf, min_conf

    def read_image(
        self,
        image_b64: str,
        media_type: str,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> str:
        """Run TrOCR on a single image. Ignores prompt (local model)."""
        return self._run_trocr(image_b64)

    def read_image_with_confidence(
        self,
        image_b64: str,
        media_type: str = "image/png",
        prompt: str = "",
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> tuple[str, float]:
        """Run TrOCR and return (text, mean_confidence) where confidence in (0, 1].

        confidence = exp(mean_token_logprob). 1.0 = certain, ~0 = uncertain.
        Used by callers that only need a single scalar. For the min-token
        confidence (catches single buried bad chars), use
        ``read_image_with_token_confidence``.
        """
        text, mean_conf, _ = self._run_trocr_with_confidence(image_b64)
        return text, mean_conf

    def read_image_with_token_confidence(
        self,
        image_b64: str,
        media_type: str = "image/png",
        prompt: str = "",
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> tuple[str, float, float]:
        """Run TrOCR and return (text, mean_confidence, min_token_confidence).

        ``min_token_confidence = exp(min(logprobs))`` exposes the worst single
        token, which the mean averages away on otherwise confident lines. Used
        by the local_first strategy to escalate when even one token looks bad.
        Ignores prompt/system_prompt — TrOCR is a local model.
        """
        return self._run_trocr_with_confidence(image_b64)

    def read_batch(
        self,
        image_blocks: list[dict],
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 8192,
    ) -> str:
        results = []
        for block in image_blocks:
            b64 = block.get("data", "")
            if b64:
                results.append(self._run_trocr(b64))
        return "\n\n".join(r for r in results if r)

    def fine_tune_for_writer(
        self,
        writer_id: str,
        line_images_b64: list[str],
        transcriptions: list[str],
        epochs: int = 3,
        lr: float = 5e-5,
    ) -> Path:
        """Fine-tune model for a specific writer. Saves to ~/.handwriting-engine/writer-models/.

        Research: 5 lines per writer makes TrOCR 'very effective' for single-writer HTR.
        (arXiv 2305.02593)

        Returns:
            Path to the saved fine-tuned model directory.
        """
        import torch
        from torch.optim import AdamW
        from PIL import Image

        save_dir = Path.home() / ".handwriting-engine" / "writer-models" / writer_id
        save_dir.mkdir(parents=True, exist_ok=True)

        if len(line_images_b64) != len(transcriptions):
            raise ValueError("line_images_b64 and transcriptions must have the same length")

        self._model.train()
        optimizer = AdamW(self._model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            for b64, text in zip(line_images_b64, transcriptions):
                img = self._b64_to_pil(b64)
                pixel_values = self._processor(images=img, return_tensors="pt").pixel_values.to(self._device)
                labels = self._processor.tokenizer(
                    text, return_tensors="pt", padding=True
                ).input_ids.to(self._device)
                labels[labels == self._processor.tokenizer.pad_token_id] = -100
                outputs = self._model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                img.close()
            logger.info("Writer %s fine-tune epoch %d/%d, loss=%.4f", writer_id, epoch + 1, epochs, total_loss)

        self._model.eval()
        self._model.save_pretrained(str(save_dir))
        self._processor.save_pretrained(str(save_dir))
        logger.info("Fine-tuned model saved to %s", save_dir)
        return save_dir


# Register on import
from handwriting_engine.providers import register  # noqa: E402
register("trocr", TrOCRProvider)
