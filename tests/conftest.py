"""Shared test fixtures for handwriting engine tests."""

import os
import tempfile
import pytest
from PIL import Image


@pytest.fixture
def tmp_image():
    """Create a temporary test image and clean up after."""
    paths = []

    def _make(color=(128, 128, 128), size=(200, 200)):
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        img = Image.new("RGB", size, color)
        # Draw some edges so it's not completely uniform
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        for y in range(0, size[1], 20):
            draw.line([(0, y), (size[0], y)], fill=(20, 20, 20), width=1)
        img.save(path, "JPEG", quality=95)
        paths.append(path)
        return path

    yield _make

    for p in paths:
        if os.path.exists(p):
            os.unlink(p)


class MockProvider:
    """Mock vision provider for testing consensus without real API calls."""

    name = "mock"

    def __init__(self, response_text="mock response", fail=False):
        self._response = response_text
        self._fail = fail
        self._usage = {"input_tokens": 100, "output_tokens": 50}

    def read_image(self, image_b64, media_type="image/jpeg", prompt="", system_prompt="", max_tokens=4096):
        if self._fail:
            raise RuntimeError("Mock provider failure")
        return self._response

    def read_batch(self, image_blocks, prompt="", system_prompt="", max_tokens=8192):
        if self._fail:
            raise RuntimeError("Mock provider failure")
        return self._response

    @property
    def usage(self):
        return dict(self._usage)

    @classmethod
    def is_available(cls):
        return True
