"""LLM adapter for OpenAI-compatible Chat Completions servers (Ollama, LM Studio, mlx-lm, ...).

Speaks the streaming `/chat/completions` API over `httpx`. Every provider quirk
stays in here (ADR 0009): partial tool-call deltas are assembled, reasoning
output (`reasoning` / `reasoning_content` fields, `<think>` spans in content) is
dropped, and finish reasons are mapped onto the core's `FinishReason`.

Schema-constrained output is sent as `response_format` of type `json_schema`.
Probed on Ollama 0.40.16 with `gemma4:12b-mlx` and `qwen3.5:2b-mlx` (MLX builds):
decoding is constrained (a prompt asking for prose still yields schema-shaped
JSON), with and without streaming and with `reasoning_effort = "none"`;
pydantic's schemas (`$defs`/`$ref`, `anyOf` with null, enums, arrays of
objects) are accepted as they are, so no schema rewriting is done here.

One quirk of constrained decoding is guarded against: when the prompt pulls
against the schema, `gemma4:12b-mlx` can emit whitespace (which the grammar
always allows) indefinitely, and the stream never ends. With a response schema,
a run of more than `MAX_SCHEMA_WHITESPACE_RUN` whitespace characters ends the
response with `FinishReason.LENGTH`, so the caller sees truncated output.

Another is guarded against on request (`hidden_loop_repeats`): with thinking
off and tools offered, `gemma4:12b-mlx` on Ollama 0.40.16 sometimes repeats an
empty thought block (`<|channel>thought\n<channel|>`, four tokens) without end.
Ollama strips those tokens, so the stream carries nothing until the read
timeout. With `logprobs` requested, every generated token still arrives in a
chunk (with empty `content`), which makes the loop visible within half a
second: when the tokens before any visible output are one short block repeated
`hidden_loop_repeats` times, the response is ended with `FinishReason.LENGTH`
and no output, so the gateway can retry it on a fallback role. Measured: normal
replies start with at most two empty blocks; looping ones repeated it over 100
times in 15 s and never recovered.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Final

import httpx

from lucy.core import registry
from lucy.core.types import FinishReason, LLMCaps, LLMChunk, Message, Role, ToolCall, ToolSpec

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL: Final = "http://localhost:11434/v1"
"""Default endpoint of Ollama's OpenAI-compatible API."""

MAX_SCHEMA_WHITESPACE_RUN: Final = 256
"""Consecutive whitespace characters after which a schema-constrained response is cut off."""

MAX_LOOP_PERIOD: Final = 8
"""Longest block of hidden tokens whose repetition `hidden_loop_repeats` detects."""

_FINISH_REASONS: Final[dict[str, FinishReason]] = {
    "stop": FinishReason.STOP,
    "end_turn": FinishReason.STOP,
    "content_filter": FinishReason.STOP,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "length": FinishReason.LENGTH,
    "max_tokens": FinishReason.LENGTH,
}


_SCHEMA_NAME: Final = re.compile(r"[^A-Za-z0-9_-]")


class LLMError(RuntimeError):
    """The model server could not be reached, refused the request or sent a broken stream."""


# --- Thinking filter ----------------------------------------------------------


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest proper prefix of `tag` that `text` ends with."""
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


class _ThinkFilter:
    """Removes `<think>...</think>` spans from streamed text, even when tags span chunks."""

    OPEN: Final = "<think>"
    CLOSE: Final = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False
        self._after_close = False
        self._emitted = False

    def feed(self, text: str) -> str:
        """Add streamed text; return the part that is safe to emit now."""
        self._buffer += text
        out: list[str] = []
        while True:
            if self._inside:
                end = self._buffer.find(self.CLOSE)
                if end == -1:
                    keep = _partial_suffix(self._buffer, self.CLOSE)
                    self._buffer = self._buffer[len(self._buffer) - keep :]
                    break
                self._buffer = self._buffer[end + len(self.CLOSE) :]
                self._inside = False
                self._after_close = True
            else:
                start = self._buffer.find(self.OPEN)
                if start == -1:
                    keep = _partial_suffix(self._buffer, self.OPEN)
                    out.append(self._emit(self._buffer[: len(self._buffer) - keep]))
                    self._buffer = self._buffer[len(self._buffer) - keep :]
                    break
                out.append(self._emit(self._buffer[:start]))
                self._buffer = self._buffer[start + len(self.OPEN) :]
                self._inside = True
        return "".join(out)

    def flush(self) -> str:
        """Return held-back text at the end of the stream; an unclosed think span is dropped."""
        text = "" if self._inside else self._emit(self._buffer)
        self._buffer = ""
        return text

    def _emit(self, text: str) -> str:
        # Whitespace after a leading think span ("<think>...</think>\n\nHi") is dropped.
        if self._after_close and not self._emitted:
            text = text.lstrip()
        if text:
            self._emitted = True
        return text


# --- Hidden-loop detection ----------------------------------------------------


class _HiddenLoopDetector:
    """Detects a response whose hidden tokens (before any visible output) are one short
    block repeated over and over.

    Fed the tokens reported in `logprobs` while nothing visible has been produced.
    `add` returns True once all tokens so far are a block of at most
    `MAX_LOOP_PERIOD` tokens repeated `repeats` times. Once the tokens are no such
    repetition (a tool call starting, say) it never fires.
    """

    def __init__(self, repeats: int) -> None:
        self._repeats = repeats
        self._tokens: list[str] = []
        self._periods = set(range(1, MAX_LOOP_PERIOD + 1))

    def add(self, token: str) -> bool:
        if not self._periods:
            return False
        self._tokens.append(token)
        size = len(self._tokens)
        self._periods = {
            p for p in self._periods if size <= p or self._tokens[size - 1 - p] == token
        }
        return any(size >= p * self._repeats for p in self._periods)


def _logprob_tokens(choice: dict[str, Any]) -> list[str]:
    """The tokens a chunk's `logprobs` reports (empty without logprobs)."""
    logprobs = choice.get("logprobs")
    entries = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(entries, list):
        return []
    return [e["token"] for e in entries if isinstance(e, dict) and isinstance(e.get("token"), str)]


# --- Tool-call assembly -------------------------------------------------------


@dataclass
class _PartialCall:
    id: str = ""
    name: str = ""
    arguments: str = ""
    parsed: dict[str, Any] | None = None


class _ToolCallAssembler:
    """Accumulates tool-call deltas keyed by index into complete `ToolCall`s."""

    def __init__(self) -> None:
        self._calls: dict[int, _PartialCall] = {}
        self._last: int | None = None

    def __bool__(self) -> bool:
        return bool(self._calls)

    def add(self, delta: dict[str, Any]) -> None:
        index = delta.get("index")
        call_id = delta.get("id") or ""
        if not isinstance(index, int):
            # Tolerate servers that omit `index`: a new id starts a new call.
            known_ids = {c.id for c in self._calls.values()}
            if self._last is None or (call_id and call_id not in known_ids):
                index = len(self._calls)
            else:
                index = self._last
        self._last = index
        call = self._calls.setdefault(index, _PartialCall())
        if call_id and not call.id:
            call.id = call_id
        function = delta.get("function") or {}
        name = function.get("name")
        if isinstance(name, str) and name and name != call.name:
            call.name += name
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            call.parsed = arguments
        elif isinstance(arguments, str):
            call.arguments += arguments

    def complete(self) -> list[ToolCall]:
        """Validate and return the assembled calls in index order."""
        calls = []
        for index in sorted(self._calls):
            partial = self._calls[index]
            if not partial.name:
                raise LLMError(f"model returned a tool call without a name (index {index})")
            arguments = partial.parsed
            if arguments is None:
                raw = partial.arguments.strip() or "{}"
                try:
                    arguments = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise LLMError(
                        f"model returned invalid JSON arguments for tool {partial.name!r}: "
                        f"{partial.arguments!r}"
                    ) from exc
            if not isinstance(arguments, dict):
                raise LLMError(
                    f"model returned non-object arguments for tool {partial.name!r}: {arguments!r}"
                )
            call_id = partial.id or f"call_{uuid.uuid4().hex[:12]}"
            calls.append(ToolCall(id=call_id, name=partial.name, arguments=arguments))
        return calls


# --- Wire format --------------------------------------------------------------


def _message_to_wire(message: Message) -> dict[str, Any]:
    wire: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.role is Role.ASSISTANT and message.tool_calls:
        if not message.content:
            wire["content"] = None
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    if message.role is Role.TOOL:
        if message.tool_call_id is None:
            raise ValueError("a tool message needs a tool_call_id")
        wire["tool_call_id"] = message.tool_call_id
    return wire


def _tool_to_wire(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _response_format(schema: dict[str, Any]) -> dict[str, Any]:
    """The OpenAI `response_format` asking for JSON conforming to `schema`."""
    title = schema.get("title")
    name = _SCHEMA_NAME.sub("_", title)[:64] if isinstance(title, str) and title else ""
    return {
        "type": "json_schema",
        "json_schema": {"name": name or "response", "schema": schema, "strict": True},
    }


def _error_detail(response: httpx.Response) -> str:
    """The server's error message from a non-200 response body."""
    try:
        body = response.json()
    except ValueError:
        return response.text.strip()[:500] or response.reason_phrase
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    if isinstance(error, str):
        return error
    return json.dumps(body)[:500]


async def _sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Yield the `data` payload of each server-sent event."""
    data: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
        # Comments (":...") and other fields (event, id, retry) are ignored.
    if data:
        yield "\n".join(data)


# --- Adapter ------------------------------------------------------------------


@registry.register("llm", "openai_compat")
class OpenAICompatLLM:
    """An LLM served over the OpenAI-compatible Chat Completions streaming API.

    Options come from config. `extra` is merged into every request body (it wins
    over the adapter's own keys) for server-specific parameters, e.g.
    `{"reasoning_effort": "none"}` to switch thinking off on Ollama.
    `json_schema` declares that the server enforces `response_format` JSON
    schemas; it is off by default here because many servers silently ignore
    the parameter. `hidden_loop_repeats` (off by default) requests `logprobs`
    and ends a response with no output when its tokens before any visible
    output are one short block repeated that many times (see the module
    docstring); use it only with thinking off, since reasoning tokens are
    hidden too. `transport` is for tests.
    """

    server_name: str = "the model server"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        context_window: int = 8192,
        native_tools: bool = True,
        json_schema: bool = False,
        temperature: float | None = None,
        timeout: float = 120,
        hidden_loop_repeats: int | None = None,
        extra: dict[str, Any] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if hidden_loop_repeats is not None and hidden_loop_repeats < 2:
            raise ValueError("hidden_loop_repeats must be at least 2")
        self.model = model
        self._hidden_loop_repeats = hidden_loop_repeats
        self.base_url = base_url.rstrip("/")
        self.capabilities = LLMCaps(
            native_tools=native_tools,
            json_schema=json_schema,
            streaming=True,
            context_window=context_window,
        )
        self._temperature = temperature
        self._extra = dict(extra or {})
        self._timeout = timeout
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
            transport=transport,
        )

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    def _request_body(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_wire(m) for m in messages],
            "stream": True,
        }
        if tools:
            body["tools"] = [_tool_to_wire(t) for t in tools]
        if response_schema is not None:
            body["response_format"] = _response_format(response_schema)
        if self._temperature is not None:
            body["temperature"] = self._temperature
        if self._hidden_loop_repeats is not None:
            body["logprobs"] = True
        body.update(self._extra)
        return body

    def _where(self) -> str:
        return f"{self.server_name} at {self.base_url}"

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """Stream the response to `messages` as core `LLMChunk`s; see the `LLM` port."""
        if self._client.is_closed:
            raise LLMError(f"the LLM adapter for {self.model!r} has been closed")
        if tools and not self.capabilities.native_tools:
            raise LLMError(
                "tools were passed but native_tools is off; "
                "prompt-based tool calling is not implemented yet"
            )
        if response_schema is not None and not self.capabilities.json_schema:
            raise LLMError(
                f"a response schema was passed but json_schema is off for model {self.model!r}; "
                "use LLMGateway.generate, which falls back to prompting, or enable json_schema "
                "if the server enforces response_format"
            )
        body = self._request_body(messages, tools, response_schema)
        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise LLMError(
                        f"{self._where()} returned HTTP {response.status_code} for model "
                        f"{self.model!r}: {_error_detail(response)}"
                    )
                async for chunk in self._parse(response, response_schema is not None):
                    yield chunk
        except httpx.ConnectError as exc:
            raise LLMError(f"cannot reach {self._where()}; is it running? ({exc})") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"{self._where()} did not respond within {self._timeout:g}s "
                f"(model {self.model!r}); it may still be loading the model"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"connection to {self._where()} failed: {exc!r}") from exc

    async def _parse(
        self, response: httpx.Response, guard_whitespace: bool = False
    ) -> AsyncIterator[LLMChunk]:
        whitespace_run = 0
        think = _ThinkFilter()
        calls = _ToolCallAssembler()
        repeats = self._hidden_loop_repeats
        loop = _HiddenLoopDetector(repeats) if repeats is not None else None
        visible = False
        finish: str | None = None
        done = False
        async for data in _sse_data(response):
            if data.strip() == "[DONE]":
                done = True
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError as exc:
                raise LLMError(f"malformed stream from {self._where()}: {data[:200]!r}") from exc
            if not isinstance(event, dict):
                raise LLMError(f"malformed stream from {self._where()}: {data[:200]!r}")
            if "error" in event:
                error = event["error"]
                message = error.get("message", error) if isinstance(error, dict) else error
                raise LLMError(f"{self._where()} reported an error: {message}")
            for choice in event.get("choices") or []:
                if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta") or {}
                # `reasoning` / `reasoning_content` carry thinking output and are ignored.
                content = delta.get("content")
                if isinstance(content, str) and content:
                    text = think.feed(content)
                    if text:
                        visible = True
                        yield LLMChunk(text=text)
                        if guard_whitespace:
                            stripped = text.rstrip()
                            trailing = len(text) - len(stripped)
                            whitespace_run = trailing if stripped else whitespace_run + trailing
                            if whitespace_run > MAX_SCHEMA_WHITESPACE_RUN:
                                logger.warning(
                                    "model %r emitted over %d whitespace characters in a row "
                                    "in structured output; cutting the response off",
                                    self.model,
                                    MAX_SCHEMA_WHITESPACE_RUN,
                                )
                                yield LLMChunk(finish_reason=FinishReason.LENGTH)
                                return
                for call_delta in delta.get("tool_calls") or []:
                    visible = True
                    calls.add(call_delta)
                if loop is not None and not visible:
                    if any(loop.add(token) for token in _logprob_tokens(choice)):
                        logger.warning(
                            "model %r repeated the same hidden tokens %d times without any "
                            "output; ending the response",
                            self.model,
                            repeats,
                        )
                        yield LLMChunk(finish_reason=FinishReason.LENGTH)
                        return
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        if not done and finish is None:
            raise LLMError(f"stream from {self._where()} ended before the response was complete")
        text = think.flush()
        if text:
            yield LLMChunk(text=text)
        tool_calls = calls.complete()
        for call in tool_calls:
            yield LLMChunk(tool_call=call)
        reason = _FINISH_REASONS.get(finish or "", FinishReason.STOP)
        if tool_calls and reason is FinishReason.STOP:
            reason = FinishReason.TOOL_CALLS
        yield LLMChunk(finish_reason=reason)


@registry.register("llm", "ollama")
class OllamaLLM(OpenAICompatLLM):
    """`OpenAICompatLLM` defaulting to a local Ollama server.

    `json_schema` defaults to on: Ollama 0.40.16 enforces `response_format`
    JSON schemas for the installed MLX models (see the module docstring).
    """

    server_name = "Ollama"

    def __init__(
        self, *, base_url: str = OLLAMA_BASE_URL, json_schema: bool = True, **options: Any
    ) -> None:
        super().__init__(base_url=base_url, json_schema=json_schema, **options)
