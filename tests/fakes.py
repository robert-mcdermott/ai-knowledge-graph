"""Shared test doubles for the LLM client (importable via conftest's sys.path)."""
import json


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
