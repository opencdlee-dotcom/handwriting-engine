"""Tests for the `cli read` command surface (S3 — skill-engine bridge).

These tests pin the contract that the handwriting-reader skill depends on:
- --format=md|json|txt
- --writer=<id> for WriterProfileStore profile binding
- Default --domain=general (was 'biology')
- JSON output exposes alt-markers extracted from consensus text
"""

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from handwriting_engine.cli import cli
from handwriting_engine.providers.base import ConsensusResult


@pytest.fixture
def runner():
    return CliRunner()


def _fake_consensus(*args, **kwargs):
    return ConsensusResult(
        text="The mitochondria [?alt: mitochondrion] is the powerhouse [?alt: power-house]",
        confidence=0.72,
        confidence_level="LOW",
        provider_results={"claude": "mitochondria", "gemini": "mitochondrion"},
        disagreements=["'mitochondria' vs 'mitochondrion' (no majority)"],
        strategy_used="vote",
        tokens_used={"input_tokens": 100, "output_tokens": 50},
    )


def _fake_read_page(*args, **kwargs):
    return "plain single-provider read"


class TestCliReadFormat:
    """`--format` flag controls output shape."""

    def test_format_txt_default_matches_legacy_echo(self, tmp_image, runner):
        path = tmp_image()
        with patch("handwriting_engine.vision.read_page", side_effect=_fake_read_page):
            result = runner.invoke(cli, ["read", path])
        assert result.exit_code == 0, result.output
        assert "plain single-provider read" in result.output

    def test_format_json_returns_parseable_payload(self, tmp_image, runner):
        path = tmp_image()
        with patch("handwriting_engine.vision.read_with_consensus", side_effect=_fake_consensus):
            result = runner.invoke(cli, ["read", path, "--provider", "consensus", "--format", "json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["path"] == path
        assert len(payload["pages"]) == 1
        page = payload["pages"][0]
        assert "mitochondria" in page["text"]
        assert page["confidence"] == pytest.approx(0.72)
        assert page["confidence_level"] == "LOW"
        assert page["strategy_used"] == "vote"
        assert page["disagreements"]

    def test_format_json_extracts_alt_markers(self, tmp_image, runner):
        path = tmp_image()
        with patch("handwriting_engine.vision.read_with_consensus", side_effect=_fake_consensus):
            result = runner.invoke(cli, ["read", path, "--provider", "consensus", "--format", "json"])
        payload = json.loads(result.output)
        markers = payload["pages"][0]["alt_markers"]
        assert len(markers) == 2
        assert any("mitochondrion" in m["alternatives"] for m in markers)
        assert any(m["raw"].startswith("[?alt:") for m in markers)

    def test_format_md_includes_header_and_text(self, tmp_image, runner):
        path = tmp_image()
        with patch("handwriting_engine.vision.read_with_consensus", side_effect=_fake_consensus):
            result = runner.invoke(cli, ["read", path, "--provider", "consensus", "--format", "md"])
        assert result.exit_code == 0, result.output
        assert "## " in result.output  # markdown heading
        assert "mitochondria" in result.output


class TestCliReadDomain:
    """Default domain is 'general', not 'biology'."""

    def test_default_domain_is_general(self, tmp_image, runner):
        path = tmp_image()
        captured: dict = {}

        def _capture(*args, **kwargs):
            captured.update(kwargs)
            return _fake_read_page(*args, **kwargs)

        with patch("handwriting_engine.vision.read_page", side_effect=_capture):
            result = runner.invoke(cli, ["read", path])
        assert result.exit_code == 0
        assert captured.get("domain") == "general"


class TestCliReadWriter:
    """`--writer=<id>` is accepted and forwarded to the consensus call."""

    def test_writer_flag_forwarded_to_consensus(self, tmp_image, runner):
        path = tmp_image()
        captured: dict = {}

        def _capture(*args, **kwargs):
            captured.update(kwargs)
            return _fake_consensus(*args, **kwargs)

        with patch("handwriting_engine.vision.read_with_consensus", side_effect=_capture):
            result = runner.invoke(
                cli, ["read", path, "--provider", "consensus", "--writer", "ada-001", "--format", "json"]
            )
        assert result.exit_code == 0, result.output
        assert captured.get("writer_id") == "ada-001"

    def test_writer_flag_absent_does_not_pass_writer_id(self, tmp_image, runner):
        path = tmp_image()
        captured: dict = {}

        def _capture(*args, **kwargs):
            captured.update(kwargs)
            return _fake_consensus(*args, **kwargs)

        with patch("handwriting_engine.vision.read_with_consensus", side_effect=_capture):
            result = runner.invoke(
                cli, ["read", path, "--provider", "consensus", "--format", "json"]
            )
        assert result.exit_code == 0, result.output
        assert captured.get("writer_id") in (None, "")
