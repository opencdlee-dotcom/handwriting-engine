"""
Google Gemini vision provider — best for OCR, structured documents, multi-language.

Key optimizations from research:
- High resolution mode for fine detail
- Flash model for cost-effective bulk processing (95%+ accuracy)
- Structured output via response_mime_type + response_schema
- Retry with exponential backoff on rate limits and server errors
"""

from __future__ import annotations

import json
import os
import base64
import logging

from handwriting_engine._constants import DEFAULT_GEMINI_MODEL
from handwriting_engine.providers.base import retry_api_call

logger = logging.getLogger(__name__)


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None, thinking_budget: int | None = None):
        try:
            from google import genai
        except ImportError:
            raise ImportError("Install google-genai: pip install google-genai")

        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self._api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        self._model = model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self._client = genai.Client(api_key=self._api_key)
        self._usage = {"input_tokens": 0, "output_tokens": 0}

        # Thinking budget: 0 for Flash (thinking hurts OCR accuracy),
        # 128 (minimum) for Pro. More thinking = more "corrections" = worse OCR.
        if thinking_budget is not None:
            self._thinking_budget = thinking_budget
        elif "flash" in self._model.lower():
            self._thinking_budget = 0
        else:
            self._thinking_budget = 128

    def _retryable_exceptions(self) -> tuple:
        from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError
        return (ResourceExhausted, ServiceUnavailable, InternalServerError)

    def _build_config(self, max_tokens: int, **extra) -> object:
        """Build GenerateContentConfig with thinking budget applied."""
        from google.genai import types
        kwargs = {"max_output_tokens": max_tokens, "temperature": 0, **extra}
        if self._thinking_budget is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self._thinking_budget,
            )
        return types.GenerateContentConfig(**kwargs)

    def read_image(
        self,
        image_b64: str,
        media_type: str = "image/jpeg",
        prompt: str = "Read all handwritten text in this image.",
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> str:
        from google.genai import types

        image_bytes = base64.b64decode(image_b64)
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=media_type)

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        config = self._build_config(max_tokens)

        def _call():
            return self._client.models.generate_content(
                model=self._model,
                contents=[image_part, full_prompt],
                config=config,
            )

        response = retry_api_call(_call, retryable_exceptions=self._retryable_exceptions())
        self._accumulate(response)
        # Check for blocked/empty responses (safety filters)
        if not response.candidates:
            logger.warning("Gemini response blocked (no candidates) — possibly safety-filtered")
            return ""
        return response.text or ""

    def read_batch(
        self,
        image_blocks: list[dict],
        prompt: str = "Read all handwritten text in these images.",
        system_prompt: str = "",
        max_tokens: int = 8192,
    ) -> str:
        """Read multiple images. Converts from Claude format to Gemini format."""
        from google.genai import types

        parts = []
        for block in image_blocks:
            if isinstance(block, dict):
                if block.get("type") == "image":
                    source = block.get("source", {})
                    media_type = source.get("media_type", "image/jpeg")
                    data = source.get("data", "")
                    image_bytes = base64.b64decode(data)
                    parts.append(types.Part.from_bytes(data=image_bytes, mime_type=media_type))
                elif block.get("type") == "text":
                    parts.append(block["text"])

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        parts.append(full_prompt)
        config = self._build_config(max_tokens)

        def _call():
            return self._client.models.generate_content(
                model=self._model,
                contents=parts,
                config=config,
            )

        response = retry_api_call(_call, retryable_exceptions=self._retryable_exceptions())
        self._accumulate(response)
        if not response.candidates:
            logger.warning("Gemini batch response blocked (no candidates)")
            return ""
        return response.text or ""

    def read_structured(
        self,
        image_blocks: list[dict],
        prompt: str,
        json_schema: dict,
        system_prompt: str = "",
        max_tokens: int = 8192,
    ) -> dict:
        """Use Gemini's structured output for reliable JSON responses."""
        from google.genai import types

        parts = []
        for block in image_blocks:
            if isinstance(block, dict):
                if block.get("type") == "image":
                    source = block.get("source", {})
                    media_type = source.get("media_type", "image/jpeg")
                    data = source.get("data", "")
                    image_bytes = base64.b64decode(data)
                    parts.append(types.Part.from_bytes(data=image_bytes, mime_type=media_type))
                elif block.get("type") == "text":
                    parts.append(block["text"])

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        parts.append(full_prompt)
        config = self._build_config(
            max_tokens,
            response_mime_type="application/json",
            response_schema=json_schema,
        )

        def _call():
            return self._client.models.generate_content(
                model=self._model,
                contents=parts,
                config=config,
            )

        response = retry_api_call(_call, retryable_exceptions=self._retryable_exceptions())
        self._accumulate(response)
        text = response.text or "{}"
        return json.loads(text)

    def _accumulate(self, response):
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            self._usage["input_tokens"] += getattr(um, "prompt_token_count", 0) or 0
            self._usage["output_tokens"] += getattr(um, "candidates_token_count", 0) or 0

    @property
    def usage(self) -> dict:
        return dict(self._usage)

    @classmethod
    def is_available(cls) -> bool:
        try:
            from google import genai  # noqa: F401
            return bool(os.getenv("GOOGLE_API_KEY"))
        except ImportError:
            return False


from handwriting_engine.providers import register
register("gemini", GeminiProvider)
