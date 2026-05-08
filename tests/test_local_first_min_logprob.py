"""Tests for the min-token-logprob escalation gate in local_first.

Covers:
  - high mean + low min triggers escalation (the new behaviour)
  - high mean + high min skips escalation (preserves the local-wins path)
  - low mean still escalates (legacy gate still works)
  - HE_LOCAL_FIRST_MIN_TOKEN_THRESHOLD env override
  - LocalFirstResult exposes both mean and min confidences
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from handwriting_engine.local_first import (
    DEFAULT_MIN_TOKEN_THRESHOLD,
    MIN_TOKEN_THRESHOLD_ENV_VAR,
    LocalFirstResult,
    _resolve_min_token_threshold,
    read_local_first,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def image_path(tmp_path):
    p = tmp_path / "page.png"
    Image.new("RGB", (200, 60), color=(240, 240, 240)).save(p)
    return str(p)


def _make_trocr_with_token_conf(text: str, mean_conf: float, min_conf: float):
    """Mock that exposes BOTH legacy and new token-confidence APIs.

    The token-confidence path is the one the new code path consumes when
    available, so we configure it with the (text, mean, min) triple.
    """
    m = MagicMock()
    # Legacy 2-tuple — must still be supplied because read_local_first calls
    # it first as the baseline path.
    m.read_image_with_confidence.return_value = (text, mean_conf)
    m.read_image_with_token_confidence.return_value = (text, mean_conf, min_conf)
    return m


# ---------------------------------------------------------------------------
# _resolve_min_token_threshold
# ---------------------------------------------------------------------------


class TestResolveMinTokenThreshold:
    def test_default_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv(MIN_TOKEN_THRESHOLD_ENV_VAR, raising=False)
        assert _resolve_min_token_threshold(None) == DEFAULT_MIN_TOKEN_THRESHOLD

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv(MIN_TOKEN_THRESHOLD_ENV_VAR, "0.99")
        assert _resolve_min_token_threshold(0.4) == 0.4

    def test_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv(MIN_TOKEN_THRESHOLD_ENV_VAR, "0.5")
        assert _resolve_min_token_threshold(None) == 0.5

    def test_malformed_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(MIN_TOKEN_THRESHOLD_ENV_VAR, "not-a-float")
        assert _resolve_min_token_threshold(None) == DEFAULT_MIN_TOKEN_THRESHOLD


# ---------------------------------------------------------------------------
# Two-gate escalation
# ---------------------------------------------------------------------------


@patch("handwriting_engine.local_first.read_page")
@patch("handwriting_engine.local_first.validate_and_prepare_image")
@patch("handwriting_engine.local_first.get_provider")
def test_high_mean_low_min_triggers_escalation(
    mock_get_provider, mock_prepare, mock_read_page, image_path,
):
    """A confident-on-average line with one bad token should escalate."""
    mock_prepare.return_value = ("FAKE_B64", "image/png")

    # Mean way above threshold, but min below DEFAULT_MIN_TOKEN_THRESHOLD (0.3).
    trocr = _make_trocr_with_token_conf("hello vvorld", 0.85, 0.10)
    cloud = MagicMock()
    cloud.usage = {"input_tokens": 0, "output_tokens": 0}

    def get_provider_side_effect(name, **kwargs):
        return trocr if name == "trocr" else cloud
    mock_get_provider.side_effect = get_provider_side_effect

    def read_page_side_effect(*args, **kwargs):
        cloud.usage = {"input_tokens": 200, "output_tokens": 80}
        return "hello world"
    mock_read_page.side_effect = read_page_side_effect

    res = read_local_first(image_path, fallback_provider="gemini", threshold=0.6)
    assert res.escalated is True, (
        "high-mean low-min should trigger escalation via the min-token gate"
    )
    assert res.text == "hello world"
    assert res.local_confidence == pytest.approx(0.85)
    assert res.local_min_token_confidence == pytest.approx(0.10)
    assert res.fallback_provider == "gemini"
    mock_read_page.assert_called_once()


@patch("handwriting_engine.local_first.read_page")
@patch("handwriting_engine.local_first.validate_and_prepare_image")
@patch("handwriting_engine.local_first.get_provider")
def test_high_mean_high_min_skips_escalation(
    mock_get_provider, mock_prepare, mock_read_page, image_path,
):
    """Both mean and min above thresholds — local wins, no cloud call."""
    mock_prepare.return_value = ("FAKE_B64", "image/png")
    trocr = _make_trocr_with_token_conf("hello world", 0.92, 0.7)
    mock_get_provider.return_value = trocr

    res = read_local_first(image_path, fallback_provider="gemini", threshold=0.6)
    assert res.escalated is False
    assert res.text == "hello world"
    assert res.input_tokens == 0
    assert res.output_tokens == 0
    assert res.local_confidence == pytest.approx(0.92)
    assert res.local_min_token_confidence == pytest.approx(0.7)
    mock_read_page.assert_not_called()


@patch("handwriting_engine.local_first.read_page")
@patch("handwriting_engine.local_first.validate_and_prepare_image")
@patch("handwriting_engine.local_first.get_provider")
def test_low_mean_still_escalates(
    mock_get_provider, mock_prepare, mock_read_page, image_path,
):
    """Legacy gate still fires when mean is below threshold (regardless of min)."""
    mock_prepare.return_value = ("FAKE_B64", "image/png")
    # Mean below threshold but min above min-threshold — old gate still wins.
    trocr = _make_trocr_with_token_conf("messy line", 0.20, 0.50)
    cloud = MagicMock()
    cloud.usage = {"input_tokens": 0, "output_tokens": 0}

    def get_provider_side_effect(name, **kwargs):
        return trocr if name == "trocr" else cloud
    mock_get_provider.side_effect = get_provider_side_effect

    def read_page_side_effect(*args, **kwargs):
        cloud.usage = {"input_tokens": 100, "output_tokens": 40}
        return "messy line cleaned"
    mock_read_page.side_effect = read_page_side_effect

    res = read_local_first(image_path, fallback_provider="gemini", threshold=0.6)
    assert res.escalated is True
    assert res.local_confidence == pytest.approx(0.20)
    assert res.local_min_token_confidence == pytest.approx(0.50)


@patch("handwriting_engine.local_first.read_page")
@patch("handwriting_engine.local_first.validate_and_prepare_image")
@patch("handwriting_engine.local_first.get_provider")
def test_min_token_env_override_lowers_bar(
    mock_get_provider, mock_prepare, mock_read_page, image_path, monkeypatch,
):
    """Lowering HE_LOCAL_FIRST_MIN_TOKEN_THRESHOLD lets a low-min stay local."""
    monkeypatch.setenv(MIN_TOKEN_THRESHOLD_ENV_VAR, "0.05")

    mock_prepare.return_value = ("FAKE_B64", "image/png")
    # Min 0.10 — below default 0.30 (would escalate), but above env-set 0.05.
    trocr = _make_trocr_with_token_conf("borderline word", 0.85, 0.10)
    mock_get_provider.return_value = trocr

    res = read_local_first(image_path, fallback_provider="gemini", threshold=0.6)
    assert res.escalated is False, (
        "0.10 min-conf >= 0.05 env-set min-threshold should keep local"
    )
    mock_read_page.assert_not_called()


@patch("handwriting_engine.local_first.read_page")
@patch("handwriting_engine.local_first.validate_and_prepare_image")
@patch("handwriting_engine.local_first.get_provider")
def test_explicit_min_token_threshold_overrides_env(
    mock_get_provider, mock_prepare, mock_read_page, image_path, monkeypatch,
):
    """Explicit min_token_threshold beats env."""
    monkeypatch.setenv(MIN_TOKEN_THRESHOLD_ENV_VAR, "0.05")

    mock_prepare.return_value = ("FAKE_B64", "image/png")
    trocr = _make_trocr_with_token_conf("borderline word", 0.85, 0.10)
    cloud = MagicMock()
    cloud.usage = {"input_tokens": 0, "output_tokens": 0}

    def get_provider_side_effect(name, **kwargs):
        return trocr if name == "trocr" else cloud
    mock_get_provider.side_effect = get_provider_side_effect

    def read_page_side_effect(*args, **kwargs):
        cloud.usage = {"input_tokens": 50, "output_tokens": 20}
        return "borderline word fixed"
    mock_read_page.side_effect = read_page_side_effect

    # Explicit 0.5 > env 0.05 > min_conf 0.10 — should escalate via explicit gate.
    res = read_local_first(
        image_path, fallback_provider="gemini",
        threshold=0.6, min_token_threshold=0.5,
    )
    assert res.escalated is True


def test_local_first_result_exposes_min_token_conf():
    """LocalFirstResult dataclass must expose local_min_token_confidence."""
    r = LocalFirstResult(
        text="x", confidence=0.5, input_tokens=0, output_tokens=0,
        escalated=False, local_confidence=0.5, fallback_provider=None,
        local_min_token_confidence=0.2,
    )
    assert r.local_min_token_confidence == 0.2

    # Also assert default value works (back-compat with callers that don't pass it).
    r2 = LocalFirstResult(
        text="x", confidence=0.5, input_tokens=0, output_tokens=0,
        escalated=False, local_confidence=0.5, fallback_provider=None,
    )
    assert r2.local_min_token_confidence == 0.0
