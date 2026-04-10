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
