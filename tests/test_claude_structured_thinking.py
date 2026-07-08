"""ClaudeProvider.read_structured thinking pass + fallback chain (offline).

A fake ``anthropic`` module is injected into sys.modules so these run without
the SDK installed — same philosophy as the torch-free trained-correction tests.
"""

import sys
import types
from types import SimpleNamespace

import pytest


class FakeBadRequestError(Exception):
    pass


class FakeInternalServerError(Exception):
    pass


@pytest.fixture(autouse=True)
def fake_anthropic(monkeypatch):
    mod = types.ModuleType("anthropic")
    mod.BadRequestError = FakeBadRequestError
    mod.RateLimitError = type("RateLimitError", (Exception,), {})
    mod.APIConnectionError = type("APIConnectionError", (Exception,), {})
    mod.InternalServerError = FakeInternalServerError
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    # Don't actually sleep between retries in tests.
    monkeypatch.setattr("handwriting_engine.providers.base.time.sleep", lambda *_: None)
    return mod


class FakeMessages:
    """Scripted messages.create: pops one canned response (or exception) per call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _tool_response(payload):
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", input=payload)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _text_only_response():
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking"),
            SimpleNamespace(type="text", text="prose instead of a tool call"),
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _provider(responses):
    from handwriting_engine.providers.claude import ClaudeProvider

    p = ClaudeProvider.__new__(ClaudeProvider)  # skip __init__ (needs API key + SDK)
    p._model = "claude-test"
    p._client = SimpleNamespace(messages=FakeMessages(responses))
    p._usage = {"input_tokens": 0, "output_tokens": 0}
    return p


IMAGE_BLOCK = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "x"}}
SCHEMA = {"name": "doc", "input_schema": {"type": "object"}}


def test_no_thinking_budget_keeps_forced_tool_call():
    p = _provider([_tool_response({"ok": 1})])
    out = p.read_structured([IMAGE_BLOCK], "prompt", SCHEMA)
    assert out == {"ok": 1}
    calls = p._client.messages.calls
    assert len(calls) == 1
    assert calls[0]["tool_choice"] == {"type": "any"}
    assert calls[0]["temperature"] == 0
    assert "thinking" not in calls[0]


def test_thinking_pass_uses_auto_and_grows_max_tokens():
    p = _provider([_tool_response({"ok": 2})])
    out = p.read_structured([IMAGE_BLOCK], "prompt", SCHEMA, max_tokens=8192, thinking_budget=4096)
    assert out == {"ok": 2}
    calls = p._client.messages.calls
    assert len(calls) == 1
    assert calls[0]["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert calls[0]["tool_choice"] == {"type": "auto"}
    assert calls[0]["max_tokens"] == 8192 + 4096
    # Thinking forbids a forced temperature — it must be absent, not 0.
    assert "temperature" not in calls[0]


def test_no_tool_use_falls_back_to_forced_call():
    p = _provider([_text_only_response(), _tool_response({"ok": 3})])
    out = p.read_structured([IMAGE_BLOCK], "prompt", SCHEMA, thinking_budget=2048)
    assert out == {"ok": 3}
    calls = p._client.messages.calls
    assert len(calls) == 2
    assert calls[1]["tool_choice"] == {"type": "any"}
    assert "thinking" not in calls[1]
    assert calls[1]["temperature"] == 0


def test_bad_request_on_thinking_falls_back():
    p = _provider([FakeBadRequestError("thinking is not supported"), _tool_response({"ok": 4})])
    out = p.read_structured([IMAGE_BLOCK], "prompt", SCHEMA, thinking_budget=2048)
    assert out == {"ok": 4}
    assert len(p._client.messages.calls) == 2


def test_fallback_no_tool_use_returns_empty():
    p = _provider([_text_only_response(), _text_only_response()])
    out = p.read_structured([IMAGE_BLOCK], "prompt", SCHEMA, thinking_budget=1024)
    assert out == {}


# --- transient-error retry (the 502 fix) --------------------------------------

def test_transient_500_is_retried_then_succeeds():
    # Forced path (no thinking): a 502 then a good tool call -> retried, succeeds.
    p = _provider([FakeInternalServerError("502"), _tool_response({"ok": 5})])
    out = p.read_structured([IMAGE_BLOCK], "prompt", SCHEMA)
    assert out == {"ok": 5}
    assert len(p._client.messages.calls) == 2


def test_transient_error_retried_on_thinking_pass():
    p = _provider([FakeInternalServerError("502"), _tool_response({"ok": 6})])
    out = p.read_structured([IMAGE_BLOCK], "prompt", SCHEMA, thinking_budget=2048)
    assert out == {"ok": 6}
    # Both attempts keep the thinking params (retry, not fallback).
    calls = p._client.messages.calls
    assert len(calls) == 2
    assert all("thinking" in c for c in calls)


def test_persistent_500_exhausts_and_raises():
    p = _provider([FakeInternalServerError("502")] * 3)
    with pytest.raises(FakeInternalServerError):
        p.read_structured([IMAGE_BLOCK], "prompt", SCHEMA)
    assert len(p._client.messages.calls) == 3  # RETRY_MAX_ATTEMPTS
