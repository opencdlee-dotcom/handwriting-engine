"""ClaudeProvider.__init__ credential resolution (offline, fake anthropic SDK).

Metered API key → x-api-key path; no key → bare client so the SDK resolves an
OAuth credential (ANTHROPIC_AUTH_TOKEN / `ant auth login` profile).
"""

import sys
import types

import pytest


class FakeAnthropic:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs


@pytest.fixture(autouse=True)
def fake_anthropic(monkeypatch):
    mod = types.ModuleType("anthropic")
    mod.Anthropic = FakeAnthropic
    mod.AnthropicError = type("AnthropicError", (Exception,), {})
    FakeAnthropic.last_kwargs = None
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_MODEL"):
        monkeypatch.delenv(k, raising=False)
    return mod


def _provider(**kwargs):
    from handwriting_engine.providers.claude import ClaudeProvider
    return ClaudeProvider(**kwargs)


def test_explicit_api_key_uses_x_api_key_path():
    _provider(api_key="sk-ant-real")
    assert FakeAnthropic.last_kwargs == {"api_key": "sk-ant-real"}


def test_env_api_key_used(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    _provider()
    assert FakeAnthropic.last_kwargs == {"api_key": "sk-ant-env"}


def test_no_key_falls_back_to_bare_client_for_oauth(monkeypatch):
    # ANTHROPIC_AUTH_TOKEN present but no ANTHROPIC_API_KEY → bare client, so the
    # SDK resolves the Bearer token / profile itself (no api_key kwarg).
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-ant-oat01-xxx")
    _provider()
    assert FakeAnthropic.last_kwargs == {}


def test_no_credentials_raises_clear_error():
    # Emulate the real SDK raising when nothing resolves (older SDKs raise at ctor).
    err = sys.modules["anthropic"].AnthropicError

    class Raising(FakeAnthropic):
        def __init__(self, **kwargs):
            if not kwargs:
                raise err("could not resolve authentication method")
            super().__init__(**kwargs)

    sys.modules["anthropic"].Anthropic = Raising
    from handwriting_engine.providers.claude import ClaudeProvider
    with pytest.raises(ValueError, match="No Anthropic credentials"):
        ClaudeProvider()
