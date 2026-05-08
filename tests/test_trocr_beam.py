"""Tests for TrOCR beam search wiring.

These tests don't require torch/transformers to be installed — they instantiate
a TrOCRProvider with a fully mocked model/processor and assert that
``model.generate`` receives the expected ``num_beams`` / ``length_penalty`` /
``early_stopping`` kwargs on both the plain and confidence paths.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from handwriting_engine.providers import trocr_provider as tp
from handwriting_engine.providers.trocr_provider import (
    DEFAULT_TROCR_NUM_BEAMS,
    TROCR_NUM_BEAMS_ENV,
    TrOCRProvider,
    _resolve_num_beams,
)


# ---------------------------------------------------------------------------
# _resolve_num_beams
# ---------------------------------------------------------------------------


class TestResolveNumBeams:
    def test_default_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv(TROCR_NUM_BEAMS_ENV, raising=False)
        assert _resolve_num_beams(None) == DEFAULT_TROCR_NUM_BEAMS

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv(TROCR_NUM_BEAMS_ENV, "8")
        assert _resolve_num_beams(2) == 2

    def test_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv(TROCR_NUM_BEAMS_ENV, "6")
        assert _resolve_num_beams(None) == 6

    def test_malformed_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(TROCR_NUM_BEAMS_ENV, "not-an-int")
        assert _resolve_num_beams(None) == DEFAULT_TROCR_NUM_BEAMS


# ---------------------------------------------------------------------------
# generate() kwargs
# ---------------------------------------------------------------------------


def _make_provider_with_mocks():
    """Build a TrOCRProvider whose model/processor are fully mocked.

    Skips __init__ so we don't need transformers/torch installed.
    """
    p = TrOCRProvider.__new__(TrOCRProvider)
    p._model_name = "mock"
    p._device = "cpu"
    p._writer_id = None
    p._usage = {"input_tokens": 0, "output_tokens": 0}

    # Processor: takes images -> object with .pixel_values; batch_decode -> list[str]
    processor = MagicMock()
    pv = MagicMock()
    pv.to.return_value = pv  # .to(device) chain
    processor.return_value.pixel_values = pv
    processor.batch_decode.return_value = ["hello world"]
    p._processor = processor

    # Model: generate() returns a tensor stand-in for greedy path; eval() noop
    model = MagicMock()
    # Use a plain object so we can index it for batch_decode without torch.
    model.generate.return_value = MagicMock()
    p._model = model

    return p


def test_run_trocr_passes_beam_kwargs(monkeypatch):
    """Greedy path: generate() must receive num_beams=4 (default), length_penalty=1.0,
    early_stopping=True."""
    monkeypatch.delenv(TROCR_NUM_BEAMS_ENV, raising=False)
    p = _make_provider_with_mocks()

    # Patch _b64_to_pil to avoid actual base64/PIL work.
    img = MagicMock()
    with patch.object(p, "_b64_to_pil", return_value=img):
        # Bypass the `with torch.no_grad()` context — patch torch lookup.
        fake_torch = MagicMock()
        fake_torch.no_grad.return_value.__enter__ = lambda self: None
        fake_torch.no_grad.return_value.__exit__ = lambda self, *a: None
        with patch.dict("sys.modules", {"torch": fake_torch}):
            p._run_trocr("FAKE_B64")

    assert p._model.generate.called
    _, kwargs = p._model.generate.call_args
    assert kwargs.get("num_beams") == DEFAULT_TROCR_NUM_BEAMS
    assert kwargs.get("length_penalty") == 1.0
    assert kwargs.get("early_stopping") is True


def test_run_trocr_with_confidence_passes_beam_kwargs(monkeypatch):
    """Confidence path: generate() must also receive beam kwargs."""
    monkeypatch.delenv(TROCR_NUM_BEAMS_ENV, raising=False)
    p = _make_provider_with_mocks()

    # Confidence path reads output.sequences and output.scores; the mocked
    # model.generate returns a MagicMock so attribute access auto-mocks. We
    # force scores to None so the function early-returns (text, 0.0, 0.0)
    # without exercising torch-dependent log_softmax.
    fake_output = MagicMock()
    fake_output.sequences = MagicMock()
    fake_output.scores = None
    p._model.generate.return_value = fake_output

    img = MagicMock()
    with patch.object(p, "_b64_to_pil", return_value=img):
        fake_torch = MagicMock()
        fake_torch.no_grad.return_value.__enter__ = lambda self: None
        fake_torch.no_grad.return_value.__exit__ = lambda self, *a: None
        with patch.dict("sys.modules", {"torch": fake_torch}):
            text, mean_conf, min_conf = p._run_trocr_with_confidence("FAKE_B64")

    assert p._model.generate.called
    _, kwargs = p._model.generate.call_args
    assert kwargs.get("num_beams") == DEFAULT_TROCR_NUM_BEAMS
    assert kwargs.get("length_penalty") == 1.0
    assert kwargs.get("early_stopping") is True
    # No scores -> both confidences are 0.0
    assert mean_conf == 0.0
    assert min_conf == 0.0


def test_env_override_propagates_to_generate(monkeypatch):
    """HE_TROCR_NUM_BEAMS overrides default."""
    monkeypatch.setenv(TROCR_NUM_BEAMS_ENV, "2")
    p = _make_provider_with_mocks()

    img = MagicMock()
    with patch.object(p, "_b64_to_pil", return_value=img):
        fake_torch = MagicMock()
        fake_torch.no_grad.return_value.__enter__ = lambda self: None
        fake_torch.no_grad.return_value.__exit__ = lambda self, *a: None
        with patch.dict("sys.modules", {"torch": fake_torch}):
            p._run_trocr("FAKE_B64")

    _, kwargs = p._model.generate.call_args
    assert kwargs.get("num_beams") == 2
