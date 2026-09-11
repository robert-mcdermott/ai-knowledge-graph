"""LLM interaction utilities for knowledge graph generation.

This module owns everything that talks to an OpenAI-compatible chat completions
endpoint (Ollama, LM Studio, vLLM, LiteLLM, OpenAI, OpenRouter, Gemini, ...) and
everything that turns a model's free-form reply into parsed JSON.

Design goals:

* Never fail silently. A truncated or empty completion raises a descriptive
  ``LLMError`` subclass instead of returning an empty string.
* Be tolerant of servers. Retries with back-off on transient failures, automatic
  fallback from ``max_tokens`` to ``max_completion_tokens`` for newer OpenAI models,
  and a generic ``extra_body`` escape hatch for provider-specific knobs.
* Be tolerant of models. ``extract_json_from_text`` strips ``<think>`` blocks, handles
  code fences, prose around the JSON, unquoted keys, trailing commas, JSON-mode
  wrappers such as ``{"triples": [...]}`` and truncated arrays.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

from src.knowledge_graph.config import resolve_secret

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 32768
DEFAULT_TIMEOUT = 300.0
DEFAULT_MAX_RETRIES = 3
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
TOKEN_PARAMS = ("auto", "max_tokens", "max_completion_tokens")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class LLMError(Exception):
    """Base class for every failure raised by this module."""


class LLMRequestError(LLMError):
    """The HTTP request failed (after retries) or returned an unexpected body."""


class LLMTruncatedError(LLMError):
    """The model stopped because it hit the token budget (``finish_reason == "length"``)."""


class LLMEmptyResponseError(LLMError):
    """The model returned no usable content."""


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
@dataclass
class LLMResponse:
    """A normalized chat completion."""

    content: str
    finish_reason: str | None
    reasoning: str | None
    usage: dict[str, Any]
    raw: dict[str, Any]

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


@dataclass
class LLMClient:
    """Thin client for OpenAI-compatible ``/chat/completions`` endpoints.

    Build one from the loaded configuration with :meth:`from_config` and reuse it
    for every call in a run; it keeps an HTTP session and remembers which token
    parameter name the server accepts.
    """

    model: str
    base_url: str
    api_key: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float | None = 0.2
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    token_param: str = "auto"
    json_mode: bool = False
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)

    # Internal collaborators; overridable in tests.
    _session: Any = field(default_factory=lambda: requests.Session(), repr=False)
    _sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    _resolved_token_param: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("LLMClient requires a model name")
        if not self.base_url:
            raise ValueError("LLMClient requires a base_url")
        if self.token_param not in TOKEN_PARAMS:
            raise ValueError(f"token_param must be one of {TOKEN_PARAMS}, got {self.token_param!r}")
        self.api_key = resolve_secret(self.api_key)
        if self.token_param != "auto":
            self._resolved_token_param = self.token_param

    # ---- construction ----------------------------------------------------- #
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LLMClient":
        """Create a client from the ``[llm]`` table of a loaded config dict."""
        llm = config.get("llm", {})
        if "model" not in llm or "base_url" not in llm:
            raise ValueError("config [llm] must define 'model' and 'base_url'")
        return cls(
            model=llm["model"],
            base_url=llm["base_url"],
            api_key=llm.get("api_key"),
            max_tokens=int(llm.get("max_tokens", DEFAULT_MAX_TOKENS)),
            temperature=llm.get("temperature", 0.2),
            timeout=float(llm.get("timeout", DEFAULT_TIMEOUT)),
            max_retries=int(llm.get("max_retries", DEFAULT_MAX_RETRIES)),
            token_param=llm.get("token_param", "auto"),
            json_mode=bool(llm.get("json_mode", False)),
            reasoning_effort=llm.get("reasoning_effort"),
            extra_body=dict(llm.get("extra_body", {}) or {}),
            extra_headers=dict(llm.get("extra_headers", {}) or {}),
        )

    # ---- public API ------------------------------------------------------- #
    def complete(self, user_prompt: str, system_prompt: str | None = None, *,
                 allow_truncated: bool = False) -> str:
        """Return the model's text reply, raising on truncation or empty output."""
        response = self.complete_raw(user_prompt, system_prompt)
        if response.truncated and not allow_truncated:
            raise LLMTruncatedError(self._truncation_message(response))
        if not response.content.strip():
            raise LLMEmptyResponseError(self._empty_message(response))
        return response.content

    def complete_raw(self, user_prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """Perform the request with retries and return the normalized response."""
        token_param = self._resolved_token_param or "max_tokens"
        attempt = 0
        while True:
            payload = self._build_payload(user_prompt, system_prompt, token_param)
            try:
                http = self._session.post(
                    self.base_url, headers=self._headers(), json=payload, timeout=self.timeout
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= self.max_retries:
                    raise LLMRequestError(
                        f"Could not reach {self.base_url} after {attempt + 1} attempts: {exc}"
                    ) from exc
                self._backoff(attempt, reason=f"{type(exc).__name__}")
                attempt += 1
                continue

            status = http.status_code
            body = http.text or ""

            # Newer OpenAI models reject max_tokens in favour of max_completion_tokens.
            if (status == 400 and token_param == "max_tokens" and self.token_param == "auto"
                    and "max_completion_tokens" in body):
                logger.info("Server wants max_completion_tokens; switching token parameter")
                token_param = self._resolved_token_param = "max_completion_tokens"
                continue

            if status in RETRYABLE_STATUS and attempt < self.max_retries:
                self._backoff(attempt, reason=f"HTTP {status}", retry_after=http.headers.get("Retry-After"))
                attempt += 1
                continue

            if status != 200:
                raise LLMRequestError(f"API request failed (HTTP {status}) for model {self.model!r}: {body[:600]}")

            if self._resolved_token_param is None:
                self._resolved_token_param = token_param
            return self._parse(http)

    # ---- helpers ---------------------------------------------------------- #
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _build_payload(self, user_prompt: str, system_prompt: str | None, token_param: str) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        payload: dict[str, Any] = {"model": self.model, "messages": messages, token_param: self.max_tokens}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        payload.update(self.extra_body)
        return payload

    def _backoff(self, attempt: int, *, reason: str, retry_after: str | None = None) -> None:
        delay = min(2 ** attempt, 30)
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        logger.warning("LLM request failed (%s); retrying in %.0fs (attempt %d/%d)",
                       reason, delay, attempt + 1, self.max_retries)
        self._sleep(delay)

    def _parse(self, http: Any) -> LLMResponse:
        try:
            data = http.json()
        except ValueError as exc:
            raise LLMRequestError(f"Response from {self.base_url} was not JSON: {http.text[:300]}") from exc
        try:
            choice = data["choices"][0]
            message = choice.get("message") or {}
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError(f"Unexpected response shape: {json.dumps(data)[:600]}") from exc

        content = message.get("content")
        if isinstance(content, list):  # some servers return content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        return LLMResponse(
            content=content or "",
            finish_reason=choice.get("finish_reason"),
            reasoning=reasoning if isinstance(reasoning, str) else None,
            usage=data.get("usage") or {},
            raw=data,
        )

    def _truncation_message(self, response: LLMResponse) -> str:
        used = response.usage.get("completion_tokens")
        used_txt = f"{used} completion tokens used, " if used is not None else ""
        msg = (f"Response from model {self.model!r} was cut off (finish_reason='length', "
               f"{used_txt}max_tokens={self.max_tokens}).")
        if response.reasoning and not response.content.strip():
            msg += (f" The model spent the entire budget on hidden reasoning "
                    f"({len(response.reasoning):,} characters) and returned no answer.")
        elif response.content.strip():
            msg += " The answer is incomplete."
        msg += (" Fix: raise llm.max_tokens (reasoning models usually need 16000-32000), "
                "lower llm.reasoning_effort, use smaller chunks, or switch to a non-reasoning model.")
        return msg

    def _empty_message(self, response: LLMResponse) -> str:
        msg = f"Model {self.model!r} returned an empty response (finish_reason={response.finish_reason!r})."
        if response.reasoning:
            msg += f" It produced {len(response.reasoning):,} characters of reasoning but no answer."
        return msg


def call_llm(model, user_prompt, api_key, system_prompt=None, max_tokens=1000,
             temperature=0.2, base_url=None, **kwargs) -> str:
    """Backward-compatible one-shot helper. Prefer :class:`LLMClient`."""
    client = LLMClient(model=model, base_url=base_url, api_key=api_key,
                       max_tokens=max_tokens, temperature=temperature, **kwargs)
    return client.complete(user_prompt, system_prompt)


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)
_CODE_FENCE = re.compile(r"```[a-zA-Z0-9_-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[\]}])")
_UNQUOTED_KEY = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*):')


def strip_reasoning(text: str) -> str:
    """Remove ``<think>...</think>`` blocks (and stray tags) emitted by reasoning models."""
    text = _THINK_BLOCK.sub("", text)
    return _THINK_TAG.sub("", text)


def extract_json_from_text(text: str | None, expect: str = "array"):
    """Extract a JSON array (default) or object from a model reply.

    Args:
        text: The raw model output.
        expect: ``"array"`` or ``"object"``; the container type the caller needs.

    Returns:
        The parsed list/dict, or ``None`` when nothing usable was found.
    """
    if expect not in ("array", "object"):
        raise ValueError("expect must be 'array' or 'object'")
    if not text or not text.strip():
        return None

    text = strip_reasoning(text)
    candidates = [m.group(1).strip() for m in _CODE_FENCE.finditer(text)]
    candidates.append(text.strip())

    for candidate in candidates:
        if not candidate:
            continue
        result = _parse_candidate(candidate, expect)
        if result is not None:
            return result

    logger.warning("No JSON %s could be extracted from the model response", expect)
    return None


def _parse_candidate(text: str, expect: str):
    # 1. The whole candidate is JSON (possibly needing light repair).
    parsed = _loads(text)
    if parsed is not None:
        return _coerce(parsed, expect)

    # 2. JSON embedded in prose: find the first opener and its matching closer.
    opener, closer = ("[", "]") if expect == "array" else ("{", "}")
    start = text.find(opener)
    if start == -1:
        return None
    end = _find_matching(text, start, opener, closer)
    if end != -1:
        parsed = _loads(text[start:end + 1])
        if parsed is not None:
            coerced = _coerce(parsed, expect)
            if coerced is not None:
                return coerced

    # 3. Truncated array: salvage every complete object inside it.
    if expect == "array":
        objects = _salvage_objects(text, start)
        if objects:
            logger.warning("JSON array was incomplete; salvaged %d complete objects", len(objects))
            return objects
    return None


def _coerce(parsed, expect: str):
    """Fit a parsed value to the expected container type, unwrapping JSON-mode wrappers."""
    if expect == "array":
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            if "subject" in parsed and "object" in parsed:
                return [parsed]  # a single triple returned bare
            lists = [v for v in parsed.values() if isinstance(v, list)]
            if len(lists) == 1:
                return lists[0]  # e.g. {"triples": [...]} from JSON mode
        return None
    return parsed if isinstance(parsed, dict) else None


def _loads(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_repair(text))
    except json.JSONDecodeError:
        return None


def _repair(text: str) -> str:
    text = _TRAILING_COMMA.sub(r"\1", text)
    return _UNQUOTED_KEY.sub(r'\1"\2"\3:', text)


def _find_matching(text: str, start: int, opener: str, closer: str) -> int:
    """Index of the bracket matching ``text[start]``, ignoring brackets inside strings."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _salvage_objects(text: str, array_start: int) -> list:
    """Collect every complete top-level ``{...}`` object after ``array_start``."""
    objects = []
    i = array_start + 1
    while True:
        obj_start = text.find("{", i)
        if obj_start == -1:
            break
        obj_end = _find_matching(text, obj_start, "{", "}")
        if obj_end == -1:
            break
        parsed = _loads(text[obj_start:obj_end + 1])
        if isinstance(parsed, dict):
            objects.append(parsed)
        i = obj_end + 1
    return objects
