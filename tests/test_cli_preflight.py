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
def _clear_keys(monkeypatch, tmp_path):
    for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(env, raising=False)
    # Point the OAuth-profile lookup at an empty dir so tests never see a real
    # `ant auth login` on the host machine.
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "no-anthropic-config"))
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


# --- read (primary command) ---------------------------------------------------

def test_read_missing_key_is_clean_error(runner, img_path):
    result = runner.invoke(cli, ["read", img_path, "-p", "claude"])
    assert result.exit_code != 0
    assert "ANTHROPIC_API_KEY is not set" in result.output
    assert "Traceback" not in result.output


def test_read_placeholder_key_is_clean_error(runner, img_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-changeme")
    result = runner.invoke(cli, ["read", img_path, "-p", "claude"])
    assert result.exit_code != 0
    assert "placeholder" in result.output
    assert "Traceback" not in result.output


def test_preflight_noop_for_consensus():
    """Consensus is multi-provider — preflight must not block it on one key."""
    from handwriting_engine.cli import _preflight_api_key
    _preflight_api_key("consensus")  # must not raise


# --- batch --read -------------------------------------------------------------

def test_batch_read_missing_key_is_clean_error(runner, tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(img_dir / "a.jpg", "JPEG")
    result = runner.invoke(cli, ["batch", str(img_dir), "--read", "-p", "claude"])
    assert result.exit_code != 0
    assert "ANTHROPIC_API_KEY is not set" in result.output
    assert "Traceback" not in result.output


def test_batch_without_read_needs_no_key(runner, tmp_path):
    """No --read => local-only assessment, no key required (preflight must not fire)."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(img_dir / "a.jpg", "JPEG")
    result = runner.invoke(cli, ["batch", str(img_dir)])
    assert "is not set" not in result.output
    assert "Assessed" in result.output


# --- OAuth credential acceptance (claude only) --------------------------------

def test_preflight_accepts_auth_token_for_claude(monkeypatch):
    """ANTHROPIC_AUTH_TOKEN satisfies the claude preflight with no API key set."""
    from handwriting_engine.cli import _preflight_api_key
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-ant-oat01-xxx")
    _preflight_api_key("claude")  # must not raise


def test_preflight_accepts_ant_login_profile(monkeypatch, tmp_path):
    """An `ant auth login` credentials profile on disk satisfies the check."""
    from handwriting_engine.cli import _preflight_api_key
    creds = tmp_path / "credentials"
    creds.mkdir()
    (creds / "default.json").write_text("{}")
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
    _preflight_api_key("claude")  # must not raise


def test_preflight_no_credential_mentions_oauth(runner, img_path):
    result = runner.invoke(cli, ["ask", img_path, "q?", "-p", "claude"])
    assert result.exit_code != 0
    assert "ant auth login" in result.output  # OAuth path surfaced


def test_gemini_has_no_oauth_path(monkeypatch, tmp_path):
    """The OAuth fallback is claude-only — gemini still hard-requires its key."""
    from handwriting_engine.cli import _preflight_api_key
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-ant-oat01-xxx")  # irrelevant to gemini
    with pytest.raises(Exception, match="GOOGLE_API_KEY is not set"):
        _preflight_api_key("gemini")
