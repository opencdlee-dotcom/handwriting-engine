"""
OpenAI GPT vision provider — best for messy handwriting, cursive, single-letter disambiguation.

Key optimizations from research:
- detail="original" for handwriting (NOT "high" — critical for low-contrast scans)
- Structured output with JSON schema for consistent formatting
- Retry with exponential backoff on rate limits and server errors
"""

from __future__ import annotations

import json
import os
import logging

from handwriting_engine._constants import DEFAULT_OPENAI_MODEL
from handwriting_engine.providers.base import retry_api_call

logger = logging.getLogger(__name__)


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self._model = model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        self._client = OpenAI(api_key=self._api_key)
        self._usage = {"input_tokens": 0, "output_tokens": 0}

    def _retryable_exceptions(self) -> tuple:
        from openai import RateLimitError, APIStatusError
        return (RateLimitError, APIStatusError)

    def read_image(
        self,
        image_b64: str,
        media_type: str = "image/jpeg",
        prompt: str = "Read all handwritten text in this image.",
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_b64}",
                        "detail": "original",  # Critical for handwriting
                    },
                },
                {"type": "text", "text": prompt},
            ],
        })

        def _call():
            return self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )

        response = retry_api_call(_call, retryable_exceptions=self._retryable_exceptions())
        self._accumulate(response)
        return response.choices[0].message.content or ""

    def read_batch(
        self,
        image_blocks: list[dict],
        prompt: str = "Read all handwritten text in these images.",
        system_prompt: str = "",
        max_tokens: int = 8192,
    ) -> str:
        """Read multiple images. Converts from Claude format to OpenAI format."""
        content = []
        for block in image_blocks:
            if isinstance(block, dict):
                if block.get("type") == "image":
                    source = block.get("source", {})
                    media_type = source.get("media_type", "image/jpeg")
                    data = source.get("data", "")
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{data}",
                            "detail": "original",
                        },
                    })
                elif block.get("type") == "text":
                    content.append({"type": "text", "text": block["text"]})

        content.append({"type": "text", "text": prompt})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        def _call():
            return self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )

        response = retry_api_call(_call, retryable_exceptions=self._retryable_exceptions())
        self._accumulate(response)
        return response.choices[0].message.content or ""

    def read_structured(
        self,
        image_blocks: list[dict],
        prompt: str,
        json_schema: dict,
        system_prompt: str = "",
        max_tokens: int = 8192,
    ) -> dict:
        """Use OpenAI's structured output for reliable JSON responses."""
        content = []
        for block in image_blocks:
            if isinstance(block, dict):
                if block.get("type") == "image":
                    source = block.get("source", {})
                    media_type = source.get("media_type", "image/jpeg")
                    data = source.get("data", "")
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{data}",
                            "detail": "original",
                        },
                    })
                elif block.get("type") == "text":
                    content.append({"type": "text", "text": block["text"]})

        content.append({"type": "text", "text": prompt})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        def _call():
            return self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": json_schema,
                },
            )

        response = retry_api_call(_call, retryable_exceptions=self._retryable_exceptions())
        self._accumulate(response)
        text = response.choices[0].message.content or "{}"
        return json.loads(text)

    def _accumulate(self, response):
        if hasattr(response, "usage") and response.usage:
            self._usage["input_tokens"] += response.usage.prompt_tokens or 0
            self._usage["output_tokens"] += response.usage.completion_tokens or 0

    @property
    def usage(self) -> dict:
        return dict(self._usage)

    @classmethod
    def is_available(cls) -> bool:
        try:
            from openai import OpenAI  # noqa: F401
            return bool(os.getenv("OPENAI_API_KEY"))
        except ImportError:
            return False


from handwriting_engine.providers import register
register("openai", OpenAIProvider)
