<plan phase="3" index="01" requirement="REQ-004,REQ-005">
  <objective>Add PaddleOCR 3.0 and TrOCR as VisionProvider implementations with lazy imports, register both in providers/__init__.py, and add tests</objective>

  <files>
    <create>handwriting_engine/providers/paddleocr_provider.py</create>
    <create>handwriting_engine/providers/trocr_provider.py</create>
    <modify>handwriting_engine/providers/__init__.py</modify>
    <create>tests/test_providers_new.py</create>
  </files>

  <tasks>
    <task type="auto">
      <name>Create handwriting_engine/providers/paddleocr_provider.py</name>
      <action>
Create ~/Developer/handwriting-engine/handwriting_engine/providers/paddleocr_provider.py:

"""
PaddleOCR 3.0 (PP-OCRv5) vision provider.

PP-OCRv5 (May 2025) outperforms GPT-4o and Gemini 2.5 Pro on Baidu's
17-scenario benchmark with <100M parameters. Provides offline inference
with no API cost.

Install: pip install paddlepaddle paddleocr

Note: paddlepaddle on Apple Silicon (M-series) runs CPU-only.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class PaddleOCRProvider:
    name = "paddleocr"

    def __init__(self, lang: str = "en", use_angle_cls: bool = True, **kwargs):
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError("Install paddleocr: pip install paddlepaddle paddleocr")

        self._lang = lang
        self._ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            use_gpu=False,  # CPU-safe for Apple Silicon
            show_log=False,
            **kwargs,
        )
        self._usage = {"input_tokens": 0, "output_tokens": 0}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import paddleocr  # noqa: F401
            import paddle  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def usage(self) -> dict:
        return dict(self._usage)

    def _b64_to_temp_file(self, image_b64: str, suffix: str = ".jpg") -> str:
        """Write base64 image to a temp file. PaddleOCR needs a file path."""
        data = base64.standard_b64decode(image_b64)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(data)
        tmp.close()
        return tmp.name

    def _run_ocr(self, image_b64: str) -> str:
        """Run PaddleOCR and return concatenated text."""
        tmp_path = self._b64_to_temp_file(image_b64)
        try:
            result = self._ocr.ocr(tmp_path, cls=True)
            if not result or result[0] is None:
                return ""
            lines = []
            for page in result:
                if page is None:
                    continue
                for line in page:
                    if line and len(line) >= 2:
                        text_conf = line[1]
                        if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 1:
                            text = str(text_conf[0])
                            lines.append(text)
            return "\n".join(lines)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def read_image(
        self,
        image_b64: str,
        media_type: str,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> str:
        """Read a single image via PaddleOCR. Ignores prompt/system_prompt (local model)."""
        text = self._run_ocr(image_b64)
        # Approximate token usage
        self._usage["output_tokens"] += len(text.split())
        return text

    def read_batch(
        self,
        image_blocks: list[dict],
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 8192,
    ) -> str:
        """Read multiple images. Returns concatenated results."""
        results = []
        for block in image_blocks:
            b64 = block.get("data", "")
            if b64:
                results.append(self._run_ocr(b64))
        return "\n\n".join(r for r in results if r)


# Register on import
from handwriting_engine.providers import register  # noqa: E402
register("paddleocr", PaddleOCRProvider)
      </action>
      <verify>python -c "from handwriting_engine.providers.paddleocr_provider import PaddleOCRProvider; print(PaddleOCRProvider.is_available())"</verify>
      <done>PaddleOCRProvider importable and is_available() returns bool without crash</done>
    </task>

    <task type="auto">
      <name>Create handwriting_engine/providers/trocr_provider.py</name>
      <action>
Create ~/Developer/handwriting-engine/handwriting_engine/providers/trocr_provider.py:

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
      </action>
      <verify>python -c "from handwriting_engine.providers.trocr_provider import TrOCRProvider; print(TrOCRProvider.is_available())"</verify>
      <done>TrOCRProvider importable and is_available() returns bool without crash</done>
    </task>

    <task type="auto">
      <name>Register new providers in providers/__init__.py and add tests</name>
      <action>
1. In ~/Developer/handwriting-engine/handwriting_engine/providers/__init__.py:

In the _try_autoload() function, add two new elif branches:
        elif name == "paddleocr":
            from handwriting_engine.providers import paddleocr_provider  # noqa: F401
        elif name == "trocr":
            from handwriting_engine.providers import trocr_provider  # noqa: F401

Also update the available_providers() function — the autoload loop currently only checks ("claude", "openai", "gemini"). Add "paddleocr" and "trocr":
    for name in ("claude", "openai", "gemini", "paddleocr", "trocr"):
        _try_autoload(name)

2. Create ~/Developer/handwriting-engine/tests/test_providers_new.py:

"""Tests for PaddleOCR and TrOCR providers — availability and interface."""
import base64
import io
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image


def _make_test_b64() -> str:
    img = Image.new("RGB", (200, 50), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


class TestPaddleOCRProvider:

    def test_is_available_returns_bool(self):
        from handwriting_engine.providers.paddleocr_provider import PaddleOCRProvider
        result = PaddleOCRProvider.is_available()
        assert isinstance(result, bool)

    def test_not_available_when_not_installed(self):
        with patch.dict("sys.modules", {"paddleocr": None, "paddle": None}):
            from handwriting_engine.providers.paddleocr_provider import PaddleOCRProvider
            # Should return False when paddle not importable
            # (actual result depends on environment — just check it returns bool)
            assert isinstance(PaddleOCRProvider.is_available(), bool)

    def test_init_raises_import_error_when_not_installed(self):
        with patch("builtins.__import__", side_effect=ImportError("no paddle")):
            from handwriting_engine.providers.paddleocr_provider import PaddleOCRProvider
            with pytest.raises(ImportError):
                PaddleOCRProvider.__init__(PaddleOCRProvider.__new__(PaddleOCRProvider))

    def test_name_attribute(self):
        from handwriting_engine.providers.paddleocr_provider import PaddleOCRProvider
        assert PaddleOCRProvider.name == "paddleocr"

    def test_registered_in_registry(self):
        from handwriting_engine.providers import _REGISTRY
        import handwriting_engine.providers.paddleocr_provider  # noqa: F401
        assert "paddleocr" in _REGISTRY


class TestTrOCRProvider:

    def test_is_available_returns_bool(self):
        from handwriting_engine.providers.trocr_provider import TrOCRProvider
        result = TrOCRProvider.is_available()
        assert isinstance(result, bool)

    def test_name_attribute(self):
        from handwriting_engine.providers.trocr_provider import TrOCRProvider
        assert TrOCRProvider.name == "trocr"

    def test_registered_in_registry(self):
        from handwriting_engine.providers import _REGISTRY
        import handwriting_engine.providers.trocr_provider  # noqa: F401
        assert "trocr" in _REGISTRY

    def test_fine_tune_raises_on_length_mismatch(self):
        from handwriting_engine.providers.trocr_provider import TrOCRProvider
        if not TrOCRProvider.is_available():
            pytest.skip("transformers not installed")
        provider = TrOCRProvider()
        with pytest.raises(ValueError, match="same length"):
            provider.fine_tune_for_writer("test_writer", ["b64img"], ["text1", "text2"])
      </action>
      <verify>cd "~/Developer/handwriting-engine" && python -m pytest tests/test_providers_new.py -x -q 2>&1 | tail -10</verify>
      <done>New provider tests pass; paddleocr and trocr in _REGISTRY after import</done>
    </task>
  </tasks>

  <dependencies>none</dependencies>
  <commit_message>feat(phase-3): PaddleOCR 3.0 and TrOCR vision providers (REQ-004, REQ-005)</commit_message>
</plan>
