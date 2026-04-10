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


class TrOCRProvider:
    name = "trocr"

    def __init__(self, model_name: str | None = None, device: str | None = None):
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch torchvision")

        import torch

        self._model_name = model_name or os.getenv("TROCR_MODEL", DEFAULT_TROCR_MODEL)

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

    def _run_trocr(self, image_b64: str) -> str:
        import torch
        img = self._b64_to_pil(image_b64)
        pixel_values = self._processor(images=img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)
        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values)
        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self._usage["output_tokens"] += len(text.split())
        img.close()
        return text

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
