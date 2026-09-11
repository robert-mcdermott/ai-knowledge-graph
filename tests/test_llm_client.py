import json

import pytest
import requests

from src.knowledge_graph.llm import (
    LLMClient,
    LLMEmptyResponseError,
    LLMRequestError,
    LLMTruncatedError,
    call_llm,
)


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=None, headers=None):
        self.status_code = status_code
        self._body = body
        self.text = text if text is not None else (json.dumps(body) if body is not None else "")
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    """Records requests and replays a scripted list of responses/exceptions."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def completion(content, finish_reason="stop", reasoning=None, completion_tokens=10):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    return FakeResponse(200, {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"completion_tokens": completion_tokens},
    })


def make_client(script, **kwargs):
    session = FakeSession(script)
    client = LLMClient(model="m", base_url="http://x/v1/chat/completions", api_key="k",
                       _session=session, _sleep=lambda s: None, **kwargs)
    return client, session


def test_happy_path_returns_content_and_sends_expected_payload():
    client, session = make_client([completion("[]")])
    assert client.complete("user", "system") == "[]"
    call = session.calls[0]
    assert call["headers"]["Authorization"] == "Bearer k"
    assert call["json"]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert call["json"]["max_tokens"] == client.max_tokens
    assert call["timeout"] == client.timeout


def test_reasoning_exhausted_budget_raises_truncated_with_helpful_message():
    client, _ = make_client([completion("", finish_reason="length", reasoning="x" * 40000,
                                        completion_tokens=8192)])
    with pytest.raises(LLMTruncatedError) as exc:
        client.complete("u")
    msg = str(exc.value)
    assert "finish_reason='length'" in msg
    assert "hidden reasoning" in msg
    assert "max_tokens" in msg


def test_partial_content_with_length_is_truncated_unless_allowed():
    client, _ = make_client([completion('[{"subject": "a"', finish_reason="length"),
                             completion('[{"subject": "a"', finish_reason="length")])
    with pytest.raises(LLMTruncatedError):
        client.complete("u")
    assert client.complete("u", allow_truncated=True) == '[{"subject": "a"'


def test_empty_content_raises_empty_error():
    client, _ = make_client([completion("   ")])
    with pytest.raises(LLMEmptyResponseError):
        client.complete("u")


def test_switches_to_max_completion_tokens_on_400_and_remembers():
    err = FakeResponse(400, {"error": {"message": "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead."}})
    client, session = make_client([err, completion("ok"), completion("ok2")])
    assert client.complete("u") == "ok"
    assert "max_completion_tokens" in session.calls[1]["json"]
    assert "max_tokens" not in session.calls[1]["json"]
    assert client.complete("u") == "ok2"
    assert "max_completion_tokens" in session.calls[2]["json"]
    assert len(session.calls) == 3


def test_retries_on_5xx_then_succeeds():
    client, session = make_client([FakeResponse(503, text="busy"), FakeResponse(429, text="slow", headers={"Retry-After": "0"}),
                                   completion("fine")])
    assert client.complete("u") == "fine"
    assert len(session.calls) == 3


def test_gives_up_after_max_retries():
    client, session = make_client([FakeResponse(500, text="boom")] * 3, max_retries=2)
    with pytest.raises(LLMRequestError) as exc:
        client.complete("u")
    assert "HTTP 500" in str(exc.value)
    assert len(session.calls) == 3


def test_retries_on_connection_error():
    client, session = make_client([requests.ConnectionError("refused"), completion("ok")])
    assert client.complete("u") == "ok"
    assert len(session.calls) == 2


def test_non_retryable_4xx_raises_immediately():
    client, session = make_client([FakeResponse(401, text="bad key")])
    with pytest.raises(LLMRequestError):
        client.complete("u")
    assert len(session.calls) == 1


def test_content_parts_are_joined():
    body = {"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                         "finish_reason": "stop"}]}
    client, _ = make_client([FakeResponse(200, body)])
    assert client.complete("u") == "ab"


def test_optional_knobs_are_sent():
    client, session = make_client([completion("ok")], json_mode=True, reasoning_effort="low",
                                  extra_body={"think": False}, temperature=None,
                                  extra_headers={"X-Test": "1"})
    client.complete("u")
    payload = session.calls[0]["json"]
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["reasoning_effort"] == "low"
    assert payload["think"] is False
    assert "temperature" not in payload
    assert session.calls[0]["headers"]["X-Test"] == "1"


def test_from_config_reads_llm_table_and_env_key(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret")
    client = LLMClient.from_config({"llm": {"model": "m", "base_url": "http://x", "api_key": "env:MY_KEY",
                                            "max_tokens": 123, "timeout": 5, "token_param": "max_completion_tokens"}})
    assert client.api_key == "secret"
    assert client.max_tokens == 123
    assert client.timeout == 5
    assert client._resolved_token_param == "max_completion_tokens"


def test_from_config_requires_model_and_base_url():
    with pytest.raises(ValueError):
        LLMClient.from_config({"llm": {"model": "m"}})


def test_call_llm_compat_wrapper(monkeypatch):
    monkeypatch.setattr(requests, "Session", lambda: FakeSession([completion("legacy")]))
    assert call_llm("m", "u", "k", "s", 10, 0.1, "http://x") == "legacy"
