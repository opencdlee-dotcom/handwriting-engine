"""
Claude vision provider — best for complex layouts, scientific notation, formatting.
Uses tool-use for structured output (from LabNoteBookGrader lesson).
Includes quality degradation retry (from LabNoteBookGrader lesson).
"""

from __future__ import annotations

import os
import time
import logging

from handwriting_engine._constants import (
    DEFAULT_CLAUDE_MODEL,
    RETRY_MAX_LONG_SIDE,
    RETRY_JPEG_QUALITY,
)

logger = logging.getLogger(__name__)


class ClaudeProvider:
    name = "claude"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self._model = model or os.getenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
        self._client = anthropic.Anthropic(api_key=self._api_key)
        self._usage = {"input_tokens": 0, "output_tokens": 0}

    def read_image(
        self,
        image_b64: str,
        media_type: str = "image/jpeg",
        prompt: str = "Read all handwritten text in this image.",
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> str:
        content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": image_b64},
            },
            {"type": "text", "text": prompt},
        ]
        return self._call(content, system_prompt, max_tokens)

    def read_batch(
        self,
        image_blocks: list[dict],
        prompt: str = "Read all handwritten text in these images.",
        system_prompt: str = "",
        max_tokens: int = 8192,
    ) -> str:
        content = image_blocks + [{"type": "text", "text": prompt}]
        return self._call(content, system_prompt, max_tokens)

    def read_structured(
        self,
        image_blocks: list[dict],
        prompt: str,
        tool_schema: dict,
        system_prompt: str = "",
        max_tokens: int = 8192,
    ) -> dict:
        """Use Claude's tool-use for 100% reliable structured output (LabNoteBookGrader pattern)."""
        import anthropic

        content = image_blocks + [{"type": "text", "text": prompt}]
        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": content}],
            "tools": [tool_schema],
            "tool_choice": {"type": "any"},
        }
        if system_prompt:
            # Enable prompt caching for long system prompts (same as _call)
            if len(system_prompt) > 4096:
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)
        self._accumulate(response)

        for block in response.content:
            if block.type == "tool_use":
                return block.input
        return {}

    def _call(self, content: list, system_prompt: str, max_tokens: int, retries: int = 3) -> str:
        import anthropic

        for attempt in range(retries):
            try:
                kwargs = {
                    "model": self._model,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": content}],
                }
                if system_prompt:
                    # Enable prompt caching for long system prompts (>1024 tokens ≈ 4096 chars)
                    # Saves up to 90% on repeated calls with same handwriting rules
                    if len(system_prompt) > 4096:
                        kwargs["system"] = [
                            {
                                "type": "text",
                                "text": system_prompt,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ]
                    else:
                        kwargs["system"] = system_prompt

                response = self._client.messages.create(**kwargs)
                self._accumulate(response)
                if not response.content:
                    logger.warning("Claude returned empty response (possible safety refusal)")
                    return ""
                return response.content[0].text

            except anthropic.BadRequestError as e:
                if "Could not process image" in str(e) and attempt < retries - 1:
                    logger.warning(f"Image rejected, retrying with smaller images (attempt {attempt + 1})")
                    content = self._shrink_images(content)
                    continue
                raise

            except anthropic.RateLimitError:
                wait = 30 * (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                continue

        logger.error("All retry attempts exhausted for Claude API call")
        raise RuntimeError("Claude API call failed: all retry attempts exhausted (rate limited)")

    def _shrink_images(self, content: list) -> list:
        """Rebuild content with smaller images for retry."""
        from handwriting_engine.optimize import validate_and_prepare_image
        import tempfile
        import base64

        new_content = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                # Re-encode at smaller size
                source = block["source"]
                if source.get("type") == "base64":
                    img_data = base64.b64decode(source["data"])
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                            f.write(img_data)
                            tmp_path = f.name
                        result = validate_and_prepare_image(
                            tmp_path,
                            max_long_side=RETRY_MAX_LONG_SIDE,
                            jpeg_quality=RETRY_JPEG_QUALITY,
                        )
                        if result:
                            new_content.append({
                                "type": "image",
                                "source": {"type": "base64", "media_type": result[1], "data": result[0]},
                            })
                        else:
                            # Shrink failed — keep original block rather than dropping it
                            new_content.append(block)
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                    continue
            new_content.append(block)
        return new_content

    def _accumulate(self, response):
        if hasattr(response, "usage"):
            self._usage["input_tokens"] += response.usage.input_tokens
            self._usage["output_tokens"] += response.usage.output_tokens

    @property
    def usage(self) -> dict:
        return dict(self._usage)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import anthropic  # noqa: F401
            return bool(os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            return False


# Auto-register
from handwriting_engine.providers import register
register("claude", ClaudeProvider)
