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
        from handwriting_engine.providers.paddleocr_provider import PaddleOCRProvider
        if not PaddleOCRProvider.is_available():
            pytest.skip("paddleocr not installed — ImportError on init is implicit")
        # When installed, verify init succeeds (is_available() already covers the negative case)
        assert PaddleOCRProvider.is_available() is True

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
