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
import math
import os
import base64
import logging

from handwriting_engine._constants import DEFAULT_GEMINI_MODEL, DEFAULT_GEMINI_TEMPERATURE
from handwriting_engine.providers.base import retry_api_call

logger = logging.getLogger(__name__)


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None, thinking_budget: int | None = None, temperature: float | None = None):
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
        self.token_confidences: list[float] = []

        # Thinking budget: 0 for Flash (thinking hurts OCR accuracy),
        # 128 (minimum) for Pro. More thinking = more "corrections" = worse OCR.
        if thinking_budget is not None:
            self._thinking_budget = thinking_budget
        elif "flash" in self._model.lower():
            self._thinking_budget = 0
        else:
            self._thinking_budget = 128

        # Temperature: Gemini Flash performs better at 0.5 for OCR (Southbridge.AI study).
        # Temp 0 triggers degenerate sampling behavior on recognition tasks.
        if temperature is not None:
            self._temperature = temperature
        else:
            self._temperature = DEFAULT_GEMINI_TEMPERATURE

        # Context cache for batch workflows (90% discount on cached tokens)
        self._cached_content_name: str | None = None
        self._cached_system_prompt: str | None = None
        # Track per-prompt call count so we can auto-enable context caching
        # on the 2nd+ call with the same system prompt (first call has no cache to amortize).
        self._system_prompt_calls: dict[str, int] = {}

    def _retryable_exceptions(self) -> tuple:
        from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError
        return (ResourceExhausted, ServiceUnavailable, InternalServerError)

    def enable_context_cache(self, system_prompt: str) -> None:
        """Create an explicit context cache for a system prompt.

        Gemini's CachedContent API gives 90% discount on cached input tokens.
        Call this before a batch of reads that share the same system prompt.
        Minimum: 1,024 tokens (Flash), 4,096 tokens (Pro).
        """
        from google.genai import types

        if self._cached_system_prompt == system_prompt and self._cached_content_name:
            return  # Already cached

        try:
            cached = self._client.caches.create(
                model=self._model,
                config=types.CreateCachedContentConfig(
                    system_instruction=system_prompt,
                    ttl="3600s",  # 1 hour
                ),
            )
            self._cached_content_name = cached.name
            self._cached_system_prompt = system_prompt
            logger.info("Gemini context cache created: %s", cached.name)
        except Exception as e:
            logger.warning("Context caching failed (will use inline): %s", e)
            self._cached_content_name = None
            self._cached_system_prompt = None

    def _maybe_auto_enable_context_cache(self, system_prompt: str) -> None:
        """Auto-enable Gemini context cache on the 2nd+ call with the same long system prompt.

        Gemini Flash requires 1024+ cached tokens (~4096 chars), Pro requires 4096+ tokens
        (~16384 chars). Creating a cache has fixed overhead, so we wait until the 2nd call
        with the same prompt to amortize the setup cost. Cached calls get a 90% input-token
        discount.
        """
        if not system_prompt:
            return
        # Skip if already cached for this exact prompt
        if self._cached_content_name and self._cached_system_prompt == system_prompt:
            return
        is_flash = "flash" in self._model.lower()
        min_chars = 4096 if is_flash else 16384
        if len(system_prompt) < min_chars:
            return
        # Count calls per prompt; enable cache starting on the 2nd identical call
        count = self._system_prompt_calls.get(system_prompt, 0) + 1
        self._system_prompt_calls[system_prompt] = count
        # Prune the counter to avoid unbounded growth
        if len(self._system_prompt_calls) > 8:
            self._system_prompt_calls = {system_prompt: count}
        if count >= 2:
            self.enable_context_cache(system_prompt)

    def invalidate_cache(self) -> None:
        """Clear the context cache."""
        if self._cached_content_name:
            try:
                self._client.caches.delete(self._cached_content_name)
            except Exception:
                pass
            self._cached_content_name = None
            self._cached_system_prompt = None

    def _build_config(self, max_tokens: int, system_instruction: str = "", **extra) -> object:
        """Build GenerateContentConfig with thinking budget, system instruction, and media resolution."""
        from google.genai import types
        kwargs = {"max_output_tokens": max_tokens, "temperature": self._temperature, "response_logprobs": True, **extra}
        # Use cached content if available and system prompt matches
        if (self._cached_content_name
                and system_instruction
                and self._cached_system_prompt == system_instruction):
            kwargs["cached_content"] = self._cached_content_name
            # Don't send system_instruction when using cache — it's already cached
        elif system_instruction:
            kwargs["system_instruction"] = system_instruction
        if self._thinking_budget is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self._thinking_budget,
            )
        # HIGH resolution ensures max detail for handwriting OCR
        kwargs["media_resolution"] = "MEDIA_RESOLUTION_HIGH"
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

        self._maybe_auto_enable_context_cache(system_prompt)
        config = self._build_config(max_tokens, system_instruction=system_prompt)

        def _call():
            return self._client.models.generate_content(
                model=self._model,
                contents=[image_part, prompt],
                config=config,
            )

        response = retry_api_call(_call, retryable_exceptions=self._retryable_exceptions())
        self._accumulate(response)
        self._parse_logprobs(response)
        # Check for blocked/empty responses (safety filters — OFF by default on 2.5+)
        if not response.candidates:
            logger.warning("Gemini response blocked (no candidates) — check input format")
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

        parts.append(prompt)
        self._maybe_auto_enable_context_cache(system_prompt)
        config = self._build_config(max_tokens, system_instruction=system_prompt)

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

        parts.append(prompt)
        self._maybe_auto_enable_context_cache(system_prompt)
        config = self._build_config(
            max_tokens,
            system_instruction=system_prompt,
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
        if not response.candidates:
            logger.warning("Gemini structured response blocked (no candidates)")
            return {}
        text = response.text or "{}"
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse structured response: {e}")
            return None

    def _accumulate(self, response):
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            self._usage["input_tokens"] += getattr(um, "prompt_token_count", 0) or 0
            self._usage["output_tokens"] += getattr(um, "candidates_token_count", 0) or 0

    def _parse_logprobs(self, response):
        """Extract per-token confidence scores from Gemini logprobs response."""
        self.token_confidences = []
        try:
            for candidate in response.candidates:
                logprobs_result = getattr(candidate, "logprobs_result", None)
                if logprobs_result and hasattr(logprobs_result, "chosen_candidates"):
                    for token_info in logprobs_result.chosen_candidates:
                        lp = getattr(token_info, "log_probability", None)
                        if lp is not None:
                            self.token_confidences.append(math.exp(lp))
        except (AttributeError, TypeError):
            pass

    def get_mean_confidence(self) -> float:
        """Return average per-token confidence (0.0-1.0) from last read's logprobs."""
        if not self.token_confidences:
            return 0.0
        return sum(self.token_confidences) / len(self.token_confidences)

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
