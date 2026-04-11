"""
OpenAI Batch API support for benchmark runs at 50% cost.

The Batch API processes requests asynchronously (up to 24h) at half the
normal per-token price. Ideal for benchmark runs where latency doesn't matter.

Usage:
    batch = OpenAIBatchRunner()
    batch.add_request(image_b64, media_type, prompt, system_prompt, custom_id="sample_42")
    batch_id = batch.submit()
    # ... wait ...
    results = batch.retrieve(batch_id)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import logging
from pathlib import Path

from handwriting_engine._constants import DEFAULT_OPENAI_MODEL

logger = logging.getLogger(__name__)


class OpenAIBatchRunner:
    """Manages OpenAI Batch API requests for benchmark evaluation."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self._model = model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        self._client = OpenAI(api_key=self._api_key)
        self._requests: list[dict] = []

    def add_request(
        self,
        image_b64: str,
        media_type: str = "image/jpeg",
        prompt: str = "Read all handwritten text in this image.",
        system_prompt: str = "",
        max_tokens: int = 4096,
        custom_id: str = "",
    ) -> None:
        """Queue a single image read request for batch processing."""
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
                        "detail": "high",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        })

        self._requests.append({
            "custom_id": custom_id or f"req_{len(self._requests)}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
        })

    def submit(self, metadata: dict | None = None) -> str:
        """Submit all queued requests as a batch. Returns batch ID.

        Writes requests to a JSONL temp file, uploads it, and creates the batch.
        """
        if not self._requests:
            raise ValueError("No requests queued. Call add_request() first.")

        # Write JSONL file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, prefix="he_batch_"
        ) as f:
            for req in self._requests:
                f.write(json.dumps(req) + "\n")
            jsonl_path = f.name

        try:
            # Upload the file
            with open(jsonl_path, "rb") as f:
                file_obj = self._client.files.create(file=f, purpose="batch")

            # Create the batch
            batch = self._client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata=metadata or {"source": "handwriting-engine-benchmark"},
            )

            logger.info(
                "Batch %s submitted: %d requests, file %s",
                batch.id, len(self._requests), file_obj.id,
            )
            self._requests.clear()
            return batch.id

        finally:
            Path(jsonl_path).unlink(missing_ok=True)

    def check_status(self, batch_id: str) -> dict:
        """Check batch processing status.

        Returns dict with: status, total, completed, failed.
        """
        batch = self._client.batches.retrieve(batch_id)
        return {
            "id": batch.id,
            "status": batch.status,
            "total": batch.request_counts.total if batch.request_counts else 0,
            "completed": batch.request_counts.completed if batch.request_counts else 0,
            "failed": batch.request_counts.failed if batch.request_counts else 0,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
        }

    def wait_for_completion(
        self, batch_id: str, poll_interval: int = 60, timeout: int = 86400,
    ) -> dict:
        """Poll until batch completes or times out.

        Args:
            batch_id: The batch ID from submit()
            poll_interval: Seconds between status checks (default 60)
            timeout: Max seconds to wait (default 24h)

        Returns:
            Final status dict
        """
        start = time.monotonic()
        while True:
            status = self.check_status(batch_id)
            logger.info(
                "Batch %s: %s (%d/%d completed, %d failed)",
                batch_id, status["status"],
                status["completed"], status["total"], status["failed"],
            )

            if status["status"] in ("completed", "failed", "expired", "cancelled"):
                return status

            elapsed = time.monotonic() - start
            if elapsed > timeout:
                logger.warning("Batch %s timed out after %ds", batch_id, timeout)
                return status

            time.sleep(poll_interval)

    def retrieve(self, batch_id: str) -> dict[str, dict]:
        """Retrieve results from a completed batch.

        Returns dict mapping custom_id -> {text, input_tokens, output_tokens, error}.
        """
        status = self.check_status(batch_id)

        results = {}

        # Process successful outputs
        if status.get("output_file_id"):
            content = self._client.files.content(status["output_file_id"])
            for line in content.text.strip().split("\n"):
                if not line:
                    continue
                entry = json.loads(line)
                custom_id = entry.get("custom_id", "")
                response = entry.get("response", {})
                body = response.get("body", {})

                if response.get("status_code") == 200:
                    choices = body.get("choices", [])
                    text = choices[0]["message"]["content"] if choices else ""
                    usage = body.get("usage", {})
                    results[custom_id] = {
                        "text": text,
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "error": None,
                    }
                else:
                    error_msg = body.get("error", {}).get("message", "Unknown error")
                    results[custom_id] = {
                        "text": "",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "error": error_msg,
                    }

        # Process errors
        if status.get("error_file_id"):
            content = self._client.files.content(status["error_file_id"])
            for line in content.text.strip().split("\n"):
                if not line:
                    continue
                entry = json.loads(line)
                custom_id = entry.get("custom_id", "")
                error = entry.get("error", {})
                if custom_id not in results:
                    results[custom_id] = {
                        "text": "",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "error": error.get("message", "Batch error"),
                    }

        return results

    def cancel(self, batch_id: str) -> dict:
        """Cancel a pending or in-progress batch."""
        self._client.batches.cancel(batch_id)
        return self.check_status(batch_id)
