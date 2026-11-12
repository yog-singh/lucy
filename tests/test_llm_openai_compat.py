"""Tests for the OpenAI-compatible LLM adapter.

The canned streams mirror what Ollama 0.40.16 sends (captured from gemma4 and
qwen3.5 models): one `data:` line per chunk, a final chunk with an empty delta
and the finish reason, then `data: [DONE]`.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from lucy.adapters.llm_openai_compat import (
    MAX_SCHEMA_WHITESPACE_RUN,
    OLLAMA_BASE_URL,
    LLMError,
    OllamaLLM,
    OpenAICompatLLM,
)
from lucy.core import registry
from lucy.core.ports import LLM
from lucy.core.types import FinishReason, LLMChunk, Message, Role, ToolCall, ToolSpec

USER = [Message(role=Role.USER, content="hi")]
WEATHER = ToolSpec(
    name="get_weather",
    description="Get current weather for a city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    """One streamed chunk in Ollama's envelope."""
    return {
        "id": "chatcmpl-78",
        "object": "chat.completion.chunk",
        "created": 1794421010,
        "model": "gemma4:e4b-mlx",
        "system_fingerprint": "fp_ollama",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def text(content: str, **extra: Any) -> dict[str, Any]:
    return chunk({"content": content, **extra})


def call_delta(
    index: int, *, id: str | None = None, name: str | None = None, args: str | None = None
) -> dict[str, Any]:
    function: dict[str, Any] = {}
    if name is not None:
        function["name"] = name
    if args is not None:
        function["arguments"] = args
    tc: dict[str, Any] = {"index": index, "type": "function", "function": function}
    if id is not None:
        tc["id"] = id
    return chunk({"content": "", "tool_calls": [tc]})


def finish(reason: str = "stop") -> dict[str, Any]:
    return chunk({}, reason)


def sse(*events: dict[str, Any] | str, done: bool = True) -> bytes:
    lines = [f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


class Body(httpx.AsyncByteStream):
    """A response body delivered in the given pieces; records whether it was closed."""

    def __init__(self, pieces: Iterable[bytes], hang: bool = False) -> None:
        self.pieces = list(pieces)
        self.hang = hang
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for piece in self.pieces:
            yield piece
        if self.hang:
            await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


Handler = Callable[[httpx.Request], httpx.Response]


def make_llm(
    handler: Handler | bytes, cls: type[OpenAICompatLLM] = OllamaLLM, **options: Any
) -> tuple[OpenAICompatLLM, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if isinstance(handler, bytes):
            return httpx.Response(200, content=handler)
        return handler(request)

    options.setdefault("model", "gemma4:12b-mlx")
    return cls(transport=httpx.MockTransport(record), **options), requests


async def collect(llm: OpenAICompatLLM, **kwargs: Any) -> list[LLMChunk]:
    kwargs.setdefault("messages", USER)
    return [c async for c in llm.stream(**kwargs)]


def texts(chunks: list[LLMChunk]) -> str:
    return "".join(c.text for c in chunks if c.text)


# --- Registry and protocol ----------------------------------------------------


def test_classes_satisfy_protocol_and_are_registered() -> None:
    assert isinstance(OllamaLLM(model="m"), LLM)
    assert isinstance(OpenAICompatLLM(model="m", base_url="http://x/v1"), LLM)
    assert registry.get("llm", "ollama") is OllamaLLM
    assert registry.get("llm", "openai_compat") is OpenAICompatLLM


def test_reimport_does_not_fail_registration() -> None:
    registry.load_adapters(["lucy.adapters.llm_openai_compat"] * 2)
    # Registering the same class object again is a no-op.
    registry.register("llm", "ollama")(OllamaLLM)
    assert registry.get("llm", "ollama") is OllamaLLM


def test_constructor_options() -> None:
    llm = registry.create("llm", "ollama", model="m", context_window=262144, native_tools=False)
    assert llm.base_url == OLLAMA_BASE_URL
    assert llm.capabilities.context_window == 262144
    assert llm.capabilities.native_tools is False and llm.capabilities.streaming is True
    with pytest.raises(TypeError):
        registry.create("llm", "openai_compat", model="m")  # base_url is required
    with pytest.raises(TypeError):
        registry.create("llm", "ollama", model="m", bogus=1)


# --- Text ---------------------------------------------------------------------


async def test_plain_text_streaming() -> None:
    llm, _ = make_llm(
        sse(
            text("Hello", role="assistant"),
            text(""),
            text(", how"),
            text(" are you?"),
            finish("stop"),
        )
    )
    chunks = await collect(llm)
    assert chunks == [
        LLMChunk(text="Hello"),
        LLMChunk(text=", how"),
        LLMChunk(text=" are you?"),
        LLMChunk(finish_reason=FinishReason.STOP),
    ]


async def test_stream_split_into_arbitrary_byte_pieces() -> None:
    body = sse(text("Hel"), text("lo"), finish())
    pieces = [body[i : i + 7] for i in range(0, len(body), 7)]
    llm, _ = make_llm(lambda r: httpx.Response(200, stream=Body(pieces)))
    assert texts(await collect(llm)) == "Hello"


async def test_reasoning_fields_are_ignored() -> None:
    # Ollama: thinking arrives in `reasoning` with empty `content`.
    llm, _ = make_llm(
        sse(
            text("", role="assistant", reasoning="Thinking"),
            text("", reasoning=" hard"),
            chunk({"reasoning_content": "other servers use this"}),
            text("4"),
            finish(),
        )
    )
    assert await collect(llm) == [LLMChunk(text="4"), LLMChunk(finish_reason=FinishReason.STOP)]


@pytest.mark.parametrize(
    ("pieces", "expected"),
    [
        (["<think>plan</think>Hi"], "Hi"),
        (["<think>plan</think>\n\nHi there"], "Hi there"),
        (["<thi", "nk>pl", "an</th", "ink>Hi"], "Hi"),
        (["A<", "think>x</think", "> B"], "A B"),
        (["<", "t", "h", "i", "n", "k", ">", "x", "<", "/", "think>", "ok"], "ok"),
        (["2 < 3 and <b>"], "2 < 3 and <b>"),
        (["ends with <thi"], "ends with <thi"),
        (["ok<think>never closed"], "ok"),
    ],
)
async def test_think_spans_are_stripped(pieces: list[str], expected: str) -> None:
    llm, _ = make_llm(sse(*(text(p) for p in pieces), finish()))
    chunks = await collect(llm)
    assert texts(chunks) == expected
    assert all(c.text != "" for c in chunks)


# --- Tool calls ---------------------------------------------------------------


async def test_tool_call_split_across_deltas() -> None:
    llm, _ = make_llm(
        sse(
            text("Let me check."),
            call_delta(0, id="call_1", name="get_weather", args=""),
            call_delta(0, args='{"ci'),
            call_delta(0, args='ty": "Par'),
            call_delta(0, args='is"}'),
            finish("tool_calls"),
        )
    )
    chunks = await collect(llm, tools=[WEATHER])
    assert chunks == [
        LLMChunk(text="Let me check."),
        LLMChunk(tool_call=ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})),
        LLMChunk(finish_reason=FinishReason.TOOL_CALLS),
    ]


async def test_multiple_tool_calls_as_ollama_sends_them() -> None:
    # Captured shape: each call complete in one delta, with its own index and id.
    llm, _ = make_llm(
        sse(
            call_delta(0, id="call_jg832m4h", name="get_weather", args='{"city":"Paris"}'),
            call_delta(1, id="call_lcxllrso", name="get_weather", args='{"city":"Tokyo"}'),
            finish("tool_calls"),
        )
    )
    chunks = await collect(llm, tools=[WEATHER])
    assert [c.tool_call for c in chunks if c.tool_call] == [
        ToolCall(id="call_jg832m4h", name="get_weather", arguments={"city": "Paris"}),
        ToolCall(id="call_lcxllrso", name="get_weather", arguments={"city": "Tokyo"}),
    ]
    assert chunks[-1] == LLMChunk(finish_reason=FinishReason.TOOL_CALLS)
    assert sum(c.finish_reason is not None for c in chunks) == 1


async def test_interleaved_tool_call_deltas() -> None:
    llm, _ = make_llm(
        sse(
            call_delta(0, id="a", name="get_weather", args='{"city":'),
            call_delta(1, id="b", name="get_weather", args='{"city":'),
            call_delta(1, args='"Tokyo"}'),
            call_delta(0, args='"Paris"}'),
            finish("tool_calls"),
        )
    )
    calls = [c.tool_call for c in await collect(llm, tools=[WEATHER]) if c.tool_call]
    assert [(c.id, c.arguments["city"]) for c in calls] == [("a", "Paris"), ("b", "Tokyo")]


async def test_missing_tool_call_id_is_generated() -> None:
    llm, _ = make_llm(
        sse(
            call_delta(0, name="get_weather", args='{"city":"Paris"}'),
            call_delta(1, name="get_weather", args="{}"),
            finish("tool_calls"),
        )
    )
    calls = [c.tool_call for c in await collect(llm, tools=[WEATHER]) if c.tool_call]
    assert len(calls) == 2
    assert all(c.id for c in calls) and calls[0].id != calls[1].id
    assert calls[1].arguments == {}


async def test_tool_call_without_index() -> None:
    delta = {
        "content": "",
        "tool_calls": [{"id": "x", "function": {"name": "f", "arguments": "{}"}}],
    }
    llm, _ = make_llm(sse(chunk(delta), finish("tool_calls")))
    calls = [c.tool_call for c in await collect(llm, tools=[WEATHER]) if c.tool_call]
    assert calls == [ToolCall(id="x", name="f", arguments={})]


async def test_invalid_json_arguments_raise() -> None:
    llm, _ = make_llm(
        sse(call_delta(0, id="c", name="get_weather", args='{"city": Par'), finish("tool_calls"))
    )
    with pytest.raises(LLMError, match="invalid JSON arguments for tool 'get_weather'"):
        await collect(llm, tools=[WEATHER])


async def test_non_object_arguments_raise() -> None:
    llm, _ = make_llm(sse(call_delta(0, id="c", name="f", args="[1]"), finish("tool_calls")))
    with pytest.raises(LLMError, match="non-object"):
        await collect(llm, tools=[WEATHER])


async def test_tools_without_native_support_raise() -> None:
    llm, requests = make_llm(sse(finish()), native_tools=False)
    with pytest.raises(LLMError, match="prompt-based tool calling is not implemented"):
        await collect(llm, tools=[WEATHER])
    assert requests == []
    assert texts(await collect(llm)) == ""  # without tools it still works


# --- Finish reasons -----------------------------------------------------------


@pytest.mark.parametrize(
    ("wire", "expected"),
    [
        ("stop", FinishReason.STOP),
        ("length", FinishReason.LENGTH),
        ("tool_calls", FinishReason.TOOL_CALLS),
        ("function_call", FinishReason.TOOL_CALLS),
        ("content_filter", FinishReason.STOP),
        ("something_new", FinishReason.STOP),
    ],
)
async def test_finish_reason_mapping(wire: str, expected: FinishReason) -> None:
    llm, _ = make_llm(sse(text("x"), finish(wire)))
    assert (await collect(llm))[-1] == LLMChunk(finish_reason=expected)


async def test_tool_calls_with_stop_reason_report_tool_calls() -> None:
    llm, _ = make_llm(sse(call_delta(0, id="c", name="f", args="{}"), finish("stop")))
    assert (await collect(llm, tools=[WEATHER]))[-1].finish_reason is FinishReason.TOOL_CALLS


async def test_missing_finish_reason_before_done_defaults_to_stop() -> None:
    llm, _ = make_llm(sse(text("x")))
    assert (await collect(llm))[-1] == LLMChunk(finish_reason=FinishReason.STOP)


async def test_usage_chunk_after_finish_is_tolerated() -> None:
    usage = {"id": "x", "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
    llm, _ = make_llm(sse(text("x"), finish("length"), usage))
    assert await collect(llm) == [
        LLMChunk(text="x"),
        LLMChunk(finish_reason=FinishReason.LENGTH),
    ]


# --- Errors -------------------------------------------------------------------


async def test_http_error_includes_server_message() -> None:
    # Captured from Ollama for an unknown model.
    body = {
        "error": {
            "message": "model 'nope:1b' not found",
            "type": "not_found_error",
            "param": None,
            "code": None,
        }
    }
    llm, _ = make_llm(lambda r: httpx.Response(404, json=body), model="nope:1b")
    with pytest.raises(LLMError, match=r"Ollama .* HTTP 404 .*model 'nope:1b' not found"):
        await collect(llm)


async def test_http_error_with_plain_text_body() -> None:
    llm, _ = make_llm(
        lambda r: httpx.Response(500, text="boom"), OpenAICompatLLM, base_url="http://h/v1"
    )
    with pytest.raises(LLMError, match="HTTP 500.*boom"):
        await collect(llm)


async def test_connection_error_is_actionable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 61] Connection refused", request=request)

    llm, _ = make_llm(refuse)
    with pytest.raises(LLMError, match=r"cannot reach Ollama at http://localhost:11434/v1; is it"):
        await collect(llm)


async def test_real_connection_refused() -> None:
    llm = OllamaLLM(model="m", base_url="http://127.0.0.1:9/v1")
    with pytest.raises(LLMError, match="cannot reach Ollama"):
        await collect(llm)
    await llm.aclose()


async def test_timeout_is_reported() -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    llm, _ = make_llm(slow, timeout=5)
    with pytest.raises(LLMError, match="did not respond within 5s"):
        await collect(llm)


@pytest.mark.parametrize(
    "body",
    [
        b"data: {not json\n\n",
        b'data: "a string"\n\n',
        sse(text("partial"), done=False),
    ],
)
async def test_malformed_stream_raises(body: bytes) -> None:
    llm, _ = make_llm(body)
    with pytest.raises(LLMError, match="malformed|ended before"):
        await collect(llm)


async def test_error_event_in_stream_raises() -> None:
    llm, _ = make_llm(sse(text("a"), {"error": {"message": "model crashed"}}))
    with pytest.raises(LLMError, match="model crashed"):
        await collect(llm)


# --- Request ------------------------------------------------------------------


async def test_request_body() -> None:
    llm, requests = make_llm(
        sse(finish()),
        OpenAICompatLLM,
        base_url="http://studio:1234/v1/",
        model="qwen3.5:2b-mlx",
        api_key="secret",
        temperature=0.2,
        extra={"reasoning_effort": "none", "top_k": 20},
    )
    call = ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})
    messages = [
        Message(role=Role.SYSTEM, content="You are Lucy."),
        Message(role=Role.USER, content="Weather in Paris?"),
        Message(role=Role.ASSISTANT, tool_calls=[call]),
        Message(role=Role.TOOL, content="14C, light rain", tool_call_id="call_1"),
        Message(role=Role.ASSISTANT, content="It is 14C.", tool_calls=[call]),
    ]
    await collect(llm, messages=messages, tools=[WEATHER])
    (request,) = requests
    assert request.method == "POST"
    assert str(request.url) == "http://studio:1234/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret"
    wire_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
    }
    assert json.loads(request.content) == {
        "model": "qwen3.5:2b-mlx",
        "stream": True,
        "temperature": 0.2,
        "reasoning_effort": "none",
        "top_k": 20,
        "messages": [
            {"role": "system", "content": "You are Lucy."},
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": None, "tool_calls": [wire_call]},
            {"role": "tool", "content": "14C, light rain", "tool_call_id": "call_1"},
            {"role": "assistant", "content": "It is 14C.", "tool_calls": [wire_call]},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": WEATHER.parameters,
                },
            }
        ],
    }


async def test_minimal_request_body_has_no_optional_keys() -> None:
    llm, requests = make_llm(sse(finish()))
    await collect(llm)
    body = json.loads(requests[0].content)
    assert set(body) == {"model", "messages", "stream"}
    assert "authorization" not in requests[0].headers


async def test_extra_overrides_adapter_keys() -> None:
    llm, requests = make_llm(sse(finish()), temperature=0.5, extra={"temperature": 0.1})
    await collect(llm)
    assert json.loads(requests[0].content)["temperature"] == 0.1


# --- Structured output --------------------------------------------------------


class Entity(BaseModel):
    name: str
    type: str


class Extraction(BaseModel):
    """A pydantic schema with $defs/$ref and an anyOf-null optional, sent as is."""

    entities: list[Entity]
    note: str | None = None


def test_json_schema_capability_defaults() -> None:
    assert OllamaLLM(model="m").capabilities.json_schema is True
    assert OllamaLLM(model="m", json_schema=False).capabilities.json_schema is False
    generic = OpenAICompatLLM(model="m", base_url="http://x/v1")
    assert generic.capabilities.json_schema is False
    assert OpenAICompatLLM(
        model="m", base_url="http://x/v1", json_schema=True
    ).capabilities.json_schema


async def test_response_schema_is_sent_as_response_format() -> None:
    llm, requests = make_llm(sse(text('{"entities": []}'), finish()))
    schema = Extraction.model_json_schema()
    chunks = await collect(llm, response_schema=schema)
    assert texts(chunks) == '{"entities": []}'
    body = json.loads(requests[0].content)
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Extraction", "schema": schema, "strict": True},
    }
    assert "$defs" in body["response_format"]["json_schema"]["schema"]  # not flattened


@pytest.mark.parametrize(
    ("schema", "name"),
    [
        ({"type": "object"}, "response"),
        ({"type": "object", "title": "My Schema: v2"}, "My_Schema__v2"),
        ({"type": "object", "title": "x" * 80}, "x" * 64),
    ],
)
async def test_response_format_name(schema: dict[str, Any], name: str) -> None:
    llm, requests = make_llm(sse(text("{}"), finish()))
    await collect(llm, response_schema=schema)
    assert json.loads(requests[0].content)["response_format"]["json_schema"]["name"] == name


async def test_no_response_format_without_schema() -> None:
    llm, requests = make_llm(sse(finish()))
    await collect(llm)
    assert "response_format" not in json.loads(requests[0].content)


async def test_response_schema_without_capability_raises() -> None:
    llm, requests = make_llm(sse(finish()), OpenAICompatLLM, base_url="http://h/v1")
    with pytest.raises(LLMError, match="json_schema is off"):
        await collect(llm, response_schema={"type": "object"})
    assert requests == []


async def test_runaway_whitespace_in_structured_output_is_cut_off() -> None:
    # Captured behaviour: gemma4:12b-mlx emitting '\n  ' forever after '{"n"'.
    runaway = [text("\n  ") for _ in range(MAX_SCHEMA_WHITESPACE_RUN)]
    body = Body([sse(text("{"), text('\n  "n"'), *runaway, done=False)], hang=True)
    llm, _ = make_llm(lambda r: httpx.Response(200, stream=body))
    async with asyncio.timeout(1):
        chunks = await collect(llm, response_schema={"type": "object"})
    assert chunks[-1] == LLMChunk(finish_reason=FinishReason.LENGTH)
    assert texts(chunks).startswith('{\n  "n"')
    assert body.closed


async def test_long_whitespace_is_kept_without_schema_and_in_short_runs() -> None:
    pieces = [text(" " * 100), text("x")] * 4
    llm, _ = make_llm(sse(*pieces, finish()))
    assert len(texts(await collect(llm, response_schema={"type": "object"}))) == 404
    llm, _ = make_llm(sse(text(" " * 1000), finish()))
    assert len(texts(await collect(llm))) == 1000


# --- Early exit ---------------------------------------------------------------


async def test_early_close_closes_response() -> None:
    body = Body([sse(text("first"), done=False)], hang=True)
    llm, _ = make_llm(lambda r: httpx.Response(200, stream=body))
    gen = llm.stream(USER)
    assert await anext(gen) == LLMChunk(text="first")
    assert not body.closed
    await gen.aclose()
    assert body.closed


async def test_break_closes_response() -> None:
    body = Body([sse(text("first"), done=False)], hang=True)
    llm, _ = make_llm(lambda r: httpx.Response(200, stream=body))
    async for _ in llm.stream(USER):
        break
    for _ in range(5):  # the event loop finalises the abandoned generator
        await asyncio.sleep(0)
    assert body.closed


async def test_cancellation_closes_response() -> None:
    body = Body([sse(text("first"), done=False)], hang=True)
    llm, _ = make_llm(lambda r: httpx.Response(200, stream=body))
    received: list[LLMChunk] = []

    async def consume() -> None:
        async for c in llm.stream(USER):
            received.append(c)

    task = asyncio.create_task(consume())
    while not received:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert body.closed


async def test_aclose() -> None:
    llm, _ = make_llm(sse(finish()))
    await llm.aclose()
    with pytest.raises(LLMError):
        await collect(llm)


# --- Live ---------------------------------------------------------------------


@pytest.mark.live
async def test_live_ollama_streams_a_reply() -> None:
    llm = OllamaLLM(model="gemma4:e4b-mlx", extra={"reasoning_effort": "none"}, timeout=300)
    try:
        chunks = await collect(
            llm, messages=[Message(role=Role.USER, content="Reply with just the word: hello")]
        )
    finally:
        await llm.aclose()
    assert "hello" in texts(chunks).lower()
    assert chunks[-1].finish_reason is FinishReason.STOP
    assert all(c.text != "" for c in chunks)


@pytest.mark.live
@pytest.mark.parametrize("model", ["qwen3.5:2b-mlx", "gemma4:12b-mlx"])
async def test_live_ollama_structured_output(model: str) -> None:
    llm = OllamaLLM(model=model, extra={"reasoning_effort": "none"}, timeout=300)
    prompt = "Write a short poem about the sea."  # prose request: only the schema makes it JSON
    try:
        chunks = await collect(
            llm,
            messages=[Message(role=Role.USER, content=prompt)],
            response_schema=Extraction.model_json_schema(),
        )
    finally:
        await llm.aclose()
    Extraction.model_validate_json(texts(chunks))
