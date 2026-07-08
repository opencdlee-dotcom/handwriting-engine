"""API-key preflight on the `analyze` / `ask` commands.

A missing or placeholder key should produce a one-line ClickException, not a
30-line 401 traceback — and never reach the provider.
"""

import os
import tempfile

import pytest
from click.testing import CliRunner
from PIL import Image

from handwriting_engine.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def img_path():
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    Image.new("RGB", (64, 64), (128, 128, 128)).save(path, "JPEG")
    yield path
    os.unlink(path)


@pytest.fixture(autouse=True)
def _clear_keys(monkeypatch):
    for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    yield


def test_ask_missing_key_is_clean_error(runner, img_path):
    result = runner.invoke(cli, ["ask", img_path, "what is this?", "-p", "claude"])
    assert result.exit_code != 0
    assert "ANTHROPIC_API_KEY is not set" in result.output
    assert "Traceback" not in result.output  # no stack trace


def test_analyze_placeholder_key_is_clean_error(runner, img_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-YOUR-REAL-KEY")
    result = runner.invoke(cli, ["analyze", img_path, "-p", "claude"])
    assert result.exit_code != 0
    assert "looks like a placeholder" in result.output
    assert "Traceback" not in result.output


def test_ask_placeholder_ellipsis_key_caught(runner, img_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-...")
    result = runner.invoke(cli, ["ask", img_path, "q?", "-p", "claude"])
    assert result.exit_code != 0
    assert "placeholder" in result.output


def test_preflight_names_right_env_per_provider(runner, img_path):
    result = runner.invoke(cli, ["ask", img_path, "q?", "-p", "gemini"])
    assert "GOOGLE_API_KEY is not set" in result.output


def test_preflight_accepts_real_looking_key(monkeypatch):
    """A plausible key must clear preflight (no false positive). Pure function, no network."""
    from handwriting_engine.cli import _preflight_api_key
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-abcdef0123456789ABCDEF")
    _preflight_api_key("claude")  # must not raise


def test_friendly_error_maps_401_to_key_hint():
    from handwriting_engine.cli import _friendly_provider_error
    msg = _friendly_provider_error("claude", Exception("Error code: 401 - invalid x-api-key"))
    assert "ANTHROPIC_API_KEY was rejected" in msg
    assert "Traceback" not in msg
