"""Tests for the local_first strategy: TrOCR with confidence-gated cloud fallback.

Stage 2 of the local-first OCR rollout. All providers are mocked so these
tests run without torch/transformers or cloud API keys installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from handwriting_engine.benchmark.db import (
    get_connection,
    insert_ground_truth,
    insert_sample,
)
from handwriting_engine.benchmark.evaluate import SWEEP_STRATEGIES, run_sweep
from handwriting_engine.local_first import (
    DEFAULT_THRESHOLD,
    THRESHOLD_ENV_VAR,
    LocalFirstResult,
    _resolve_threshold,
    read_local_first,
)


_GT = "the mitochondria is the powerhouse of the cell"


@pytest.fixture
def image_path(tmp_path):
    p = tmp_path / "page.png"
    Image.new("RGB", (200, 60), color=(240, 240, 240)).save(p)
    return str(p)


@pytest.fixture
def iam_seeded_db(tmp_path):
    db_path = tmp_path / "local_first_sweep.db"
    conn = get_connection(db_path)
    try:
        for i in range(3):
            img_path = tmp_path / f"iam_{i:02d}.png"
            Image.new("RGB", (200, 60), color=(240, 240, 240)).save(img_path)
            sid = insert_sample(
                conn,
                str(img_path),
                f"hash_iam_{i:02d}",
                student=f"iam-writer-{i:03d}",
                category="iam",
            )
            insert_ground_truth(conn, sid, _GT)
    finally:
        conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Threshold resolution
# ---------------------------------------------------------------------------


class TestThresholdResolution:
    def test_default_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv(THRESHOLD_ENV_VAR, raising=False)
        assert _resolve_threshold(None) == DEFAULT_THRESHOLD

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv(THRESHOLD_ENV_VAR, "0.99")
        assert _resolve_threshold(0.4) == 0.4

    def test_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv(THRESHOLD_ENV_VAR, "0.85")
        assert _resolve_threshold(None) == 0.85

    def test_malformed_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(THRESHOLD_ENV_VAR, "not-a-float")
        assert _resolve_threshold(None) == DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# read_local_first behaviour
# ---------------------------------------------------------------------------


def _make_trocr_mock(text: str, confidence: float):
    m = MagicMock()
    m.read_image_with_confidence.return_value = (text, confidence)
    return m


@patch("handwriting_engine.local_first.validate_and_prepare_image")
@patch("handwriting_engine.local_first.get_provider")
def test_high_confidence_skips_fallback(mock_get_provider, mock_prepare, image_path):
    """When TrOCR confidence >= threshold, fallback must NOT be called and
    token counts must be 0."""
    mock_prepare.return_value = ("FAKE_B64", "image/png")
    trocr = _make_trocr_mock(_GT, 0.92)
    mock_get_provider.return_value = trocr

    res = read_local_first(image_path, fallback_provider="gemini", threshold=0.6)

    assert isinstance(res, LocalFirstResult)
    assert res.text == _GT
    assert res.escalated is False
    assert res.input_tokens == 0
    assert res.output_tokens == 0
    assert res.local_confidence == pytest.approx(0.92)
    # Only the trocr provider should ever have been requested.
    assert mock_get_provider.call_count == 1
    assert mock_get_provider.call_args[0][0] == "trocr"


@patch("handwriting_engine.local_first.read_page")
@patch("handwriting_engine.local_first.validate_and_prepare_image")
@patch("handwriting_engine.local_first.get_provider")
def test_low_confidence_escalates(mock_get_provider, mock_prepare, mock_read_page, image_path):
    """When TrOCR confidence < threshold, fallback fires and propagates tokens."""
    mock_prepare.return_value = ("FAKE_B64", "image/png")

    trocr = _make_trocr_mock("garbled output", 0.10)
    # Cloud provider exposes pre/post .usage so the delta computation works.
    cloud = MagicMock()
    # Two reads of .usage: one before read_page, one after.
    cloud.usage = {"input_tokens": 0, "output_tokens": 0}

    def get_provider_side_effect(name, **kwargs):
        if name == "trocr":
            return trocr
        if name == "gemini":
            return cloud
        raise ValueError(f"unexpected provider {name}")

    mock_get_provider.side_effect = get_provider_side_effect

    def read_page_side_effect(*args, **kwargs):
        # Simulate the cloud provider accumulating tokens during the call.
        cloud.usage = {"input_tokens": 1500, "output_tokens": 300}
        return _GT

    mock_read_page.side_effect = read_page_side_effect

    res = read_local_first(image_path, fallback_provider="gemini", threshold=0.6)

    assert res.escalated is True
    assert res.text == _GT
    assert res.input_tokens == 1500
    assert res.output_tokens == 300
    assert res.local_confidence == pytest.approx(0.10)
    assert res.fallback_provider == "gemini"
    mock_read_page.assert_called_once()


@patch("handwriting_engine.local_first.read_page")
@patch("handwriting_engine.local_first.validate_and_prepare_image")
@patch("handwriting_engine.local_first.get_provider")
def test_env_var_changes_threshold(mock_get_provider, mock_prepare, mock_read_page,
                                    image_path, monkeypatch):
    """HE_LOCAL_FIRST_THRESHOLD raises the bar so a 0.7 read now escalates."""
    monkeypatch.setenv(THRESHOLD_ENV_VAR, "0.85")

    mock_prepare.return_value = ("FAKE_B64", "image/png")
    trocr = _make_trocr_mock("borderline read", 0.70)
    cloud = MagicMock()
    cloud.usage = {"input_tokens": 0, "output_tokens": 0}

    def get_provider_side_effect(name, **kwargs):
        return trocr if name == "trocr" else cloud
    mock_get_provider.side_effect = get_provider_side_effect

    def read_page_side_effect(*args, **kwargs):
        cloud.usage = {"input_tokens": 100, "output_tokens": 50}
        return _GT
    mock_read_page.side_effect = read_page_side_effect

    res = read_local_first(image_path, fallback_provider="gemini")
    assert res.escalated is True, (
        "0.70 < 0.85 (env-set threshold) so escalation should fire"
    )
    assert res.text == _GT
    assert res.input_tokens == 100


@patch("handwriting_engine.local_first.read_page")
@patch("handwriting_engine.local_first.validate_and_prepare_image")
@patch("handwriting_engine.local_first.get_provider")
def test_env_var_lowering_threshold_keeps_local(mock_get_provider, mock_prepare,
                                                  mock_read_page, image_path, monkeypatch):
    """Lowering the threshold via env var means a borderline read stays local."""
    monkeypatch.setenv(THRESHOLD_ENV_VAR, "0.30")

    mock_prepare.return_value = ("FAKE_B64", "image/png")
    trocr = _make_trocr_mock("borderline read", 0.50)
    mock_get_provider.return_value = trocr

    res = read_local_first(image_path, fallback_provider="gemini")
    assert res.escalated is False
    assert res.input_tokens == 0
    assert res.output_tokens == 0
    mock_read_page.assert_not_called()


# ---------------------------------------------------------------------------
# Sweep wiring
# ---------------------------------------------------------------------------


class TestSweepEntries:
    def test_local_first_gemini_present(self):
        names = [s["name"] for s in SWEEP_STRATEGIES]
        assert "local_first_gemini" in names

    def test_local_first_marked_mixed_cost(self):
        s = next(c for c in SWEEP_STRATEGIES if c["name"] == "local_first_gemini")
        assert s.get("cost_class") == "mixed", (
            "local_first variants must declare cost_class=mixed so the CLI "
            "doesn't lie about projected cost (escalation IS billed)."
        )
        assert s.get("providers") == ["trocr", "gemini"]

    def test_trocr_local_still_pure_local(self):
        s = next(c for c in SWEEP_STRATEGIES if c["name"] == "trocr_local")
        # Stage 1 contract preserved: trocr_local stays zero-cost.
        assert s.get("cost_class", "local") == "local"


@patch("handwriting_engine.benchmark.evaluate._read_local_first")
@patch("handwriting_engine.benchmark.evaluate._available_providers")
@patch("handwriting_engine.benchmark.evaluate._read_single")
def test_sweep_runs_local_first_when_both_providers_present(
    mock_read_single, mock_providers, mock_local_first, iam_seeded_db,
):
    """When trocr + gemini are available, local_first_gemini produces a run_id."""
    mock_providers.return_value = ["gemini", "trocr"]
    mock_read_single.return_value = {
        "text": _GT, "confidence": 0.7, "latency_ms": 100,
        "input_tokens": 0, "output_tokens": 0, "error": None,
    }
    mock_local_first.return_value = {
        "text": _GT, "confidence": 0.85, "latency_ms": 50,
        "input_tokens": 0, "output_tokens": 0, "error": None,
    }

    run_ids = run_sweep(provider="gemini", db_path=iam_seeded_db, yes=True)

    assert "local_first_gemini" in run_ids
    # _read_local_first must have actually been called (one per sample).
    assert mock_local_first.call_count >= 1


@patch("handwriting_engine.benchmark.evaluate._read_local_first")
@patch("handwriting_engine.benchmark.evaluate._available_providers")
@patch("handwriting_engine.benchmark.evaluate._read_single")
def test_sweep_skips_local_first_when_trocr_missing(
    mock_read_single, mock_providers, mock_local_first, iam_seeded_db,
):
    """Without TrOCR installed, the local_first rows are skipped, not crashed."""
    mock_providers.return_value = ["gemini"]  # no trocr
    mock_read_single.return_value = {
        "text": _GT, "confidence": 0.7, "latency_ms": 100,
        "input_tokens": 100, "output_tokens": 50, "error": None,
    }

    run_ids = run_sweep(provider="gemini", db_path=iam_seeded_db, yes=True)
    assert "local_first_gemini" not in run_ids
    assert "local_first_claude" not in run_ids
    # Cloud strategies still ran — local_first absence isn't fatal.
    assert "baseline" in run_ids
    mock_local_first.assert_not_called()
