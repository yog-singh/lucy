import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import BaseModel

from lucy.core import gateway as gateway_module
from lucy.core.config import LLMSettings
from lucy.core.gateway import (
    GatewayError,
    LLMGateway,
    Preempted,
    Priority,
    SilentStreamError,
    StructuredOutputError,
    add_fallbacks,
    fallbacks_in_context,
)
from lucy.core.registry import Registry
from lucy.core.trace import StepFallback, StepTrace, TurnTrace
from lucy.core.types import FinishReason, LLMCaps, LLMChunk, Message, Role, ToolCall, ToolSpec
from tests.fakes import FakeLLM, Pause, Raise

HELLO = [Message(role=Role.USER, content="hi")]


class RegisteredLLM:
    def __init__(self, model: str) -> None:
        self.model = model
        self.capabilities = LLMCaps(
            native_tools=False, json_schema=False, streaming=True, context_window=4096
        )

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[LLMChunk]:
        yield LLMChunk(text=self.model)


async def wait_until(predicate: object) -> None:
    async with asyncio.timeout(1):
        while not predicate():  # type: ignore[operator]
            await asyncio.sleep(0)


async def consume(gateway: LLMGateway, role: str, priority: Priority) -> list[str]:
    return [c.text or "" async for c in gateway.stream(role, HELLO, priority=priority)]


def test_role_lookup() -> None:
    chat, extraction = FakeLLM([]), FakeLLM([])
    gateway = LLMGateway({"chat": chat, "extraction": extraction})
    assert gateway.for_role("chat") is chat
    assert gateway.roles() == ["chat", "extraction"]
    with pytest.raises(GatewayError, match="summary"):
        gateway.for_role("summary")


async def test_unknown_role_in_stream_raises() -> None:
    gateway = LLMGateway({})
    with pytest.raises(GatewayError):
        await consume(gateway, "chat", Priority.FOREGROUND)


async def test_from_settings_shares_instances_for_identical_configs() -> None:
    registry = Registry()
    registry.register("llm", "fake")(RegisteredLLM)
    settings = LLMSettings.model_validate(
        {
            "roles": {
                "chat": {"adapter": "fake", "model": "big"},
                "extraction": {"adapter": "fake", "model": "small"},
                "summary": {"adapter": "fake", "model": "small"},
            }
        }
    )
    gateway = LLMGateway.from_settings(settings, registry=registry)
    assert gateway.roles() == ["chat", "extraction", "summary"]
    assert gateway.for_role("extraction") is gateway.for_role("summary")
    assert gateway.for_role("chat") is not gateway.for_role("summary")
    assert await consume(gateway, "chat", Priority.FOREGROUND) == ["big"]


def test_from_settings_rejects_non_llm_adapter() -> None:
    registry = Registry()
    registry.register("llm", "bad")(dict)
    settings = LLMSettings.model_validate({"roles": {"chat": {"adapter": "bad"}}})
    with pytest.raises(GatewayError, match="not an LLM"):
        LLMGateway.from_settings(settings, registry=registry)


async def test_background_waits_for_foreground() -> None:
    pause = Pause()
    fg_llm = FakeLLM([[LLMChunk(text="a"), pause, LLMChunk(text="b")]])
    bg_llm = FakeLLM([[LLMChunk(text="x")]])
    gateway = LLMGateway({"chat": fg_llm, "extraction": bg_llm})
    async with asyncio.timeout(1):
        fg = asyncio.create_task(consume(gateway, "chat", Priority.FOREGROUND))
        await pause.reached.wait()
        bg = asyncio.create_task(consume(gateway, "extraction", Priority.BACKGROUND))
        await wait_until(lambda: gateway.background_waiting == 1)
        assert bg_llm.calls == []  # not started while foreground is active
        pause.release.set()
        assert await fg == ["a", "b"]
        assert await bg == ["x"]
    assert gateway.foreground_active == 0 and gateway.background_waiting == 0


async def test_foreground_never_waits() -> None:
    pauses = [Pause(), Pause()]
    llm = FakeLLM([[pauses[0], LLMChunk(text="1")], [pauses[1], LLMChunk(text="2")]])
    gateway = LLMGateway({"chat": llm})
    async with asyncio.timeout(1):
        first = asyncio.create_task(consume(gateway, "chat", Priority.FOREGROUND))
        await pauses[0].reached.wait()
        second = asyncio.create_task(consume(gateway, "chat", Priority.FOREGROUND))
        await pauses[1].reached.wait()  # started while the first is still active
        assert gateway.foreground_active == 2
        for pause in pauses:
            pause.release.set()
        assert await first == ["1"] and await second == ["2"]
    assert gateway.foreground_active == 0


async def test_release_when_stream_finishes() -> None:
    gateway = LLMGateway({"chat": FakeLLM([[LLMChunk(text="a")]])})
    assert await consume(gateway, "chat", Priority.FOREGROUND) == ["a"]
    assert gateway.foreground_active == 0


async def test_release_when_stream_raises() -> None:
    gateway = LLMGateway({"chat": FakeLLM([[LLMChunk(text="a"), Raise(RuntimeError("boom"))]])})
    with pytest.raises(RuntimeError, match="boom"):
        await consume(gateway, "chat", Priority.FOREGROUND)
    assert gateway.foreground_active == 0


async def test_release_when_consumer_is_cancelled() -> None:
    pause = Pause()
    gateway = LLMGateway({"chat": FakeLLM([[LLMChunk(text="a"), pause]])})
    async with asyncio.timeout(1):
        task = asyncio.create_task(consume(gateway, "chat", Priority.FOREGROUND))
        await pause.reached.wait()
        assert gateway.foreground_active == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert gateway.foreground_active == 0


async def test_release_when_consumer_closes_early() -> None:
    gateway = LLMGateway({"chat": FakeLLM([[LLMChunk(text="a"), LLMChunk(text="b")]])})
    async with aclosing(gateway.stream("chat", HELLO)) as stream:
        async for _ in stream:
            assert gateway.foreground_active == 1
            break
    assert gateway.foreground_active == 0


async def test_background_proceeds_after_foreground_fails() -> None:
    pause = Pause()
    fg_llm = FakeLLM([[pause, Raise(RuntimeError("boom"))]])
    bg_llm = FakeLLM([[LLMChunk(text="x")]])
    gateway = LLMGateway({"chat": fg_llm, "extraction": bg_llm})
    async with asyncio.timeout(1):
        fg = asyncio.create_task(consume(gateway, "chat", Priority.FOREGROUND))
        await pause.reached.wait()
        bg = asyncio.create_task(consume(gateway, "extraction", Priority.BACKGROUND))
        await wait_until(lambda: gateway.background_waiting == 1)
        pause.release.set()
        with pytest.raises(RuntimeError):
            await fg
        assert await bg == ["x"]


# --- response_schema pass-through ---------------------------------------------


async def test_stream_passes_response_schema() -> None:
    llm = FakeLLM([[LLMChunk(text="{}")], [LLMChunk(text="x")]])
    gateway = LLMGateway({"chat": llm})
    schema = {"type": "object"}
    assert [c.text async for c in gateway.stream("chat", HELLO, response_schema=schema)] == ["{}"]
    assert await consume(gateway, "chat", Priority.FOREGROUND) == ["x"]
    assert llm.schemas_seen == [schema, None]


async def test_plain_stream_works_with_adapters_lacking_response_schema() -> None:
    # RegisteredLLM.stream has the older signature without `response_schema`.
    gateway = LLMGateway({"chat": RegisteredLLM("old"), "bg": RegisteredLLM("older")})
    assert await consume(gateway, "chat", Priority.FOREGROUND) == ["old"]
    assert await consume(gateway, "bg", Priority.BACKGROUND) == ["older"]


# --- generate -----------------------------------------------------------------


class Person(BaseModel):
    name: str
    age: int | None = None


class People(BaseModel):
    people: list[Person]


def reply(*texts: str) -> list[LLMChunk]:
    return [LLMChunk(text=t) for t in texts] + [LLMChunk(finish_reason=FinishReason.STOP)]


async def test_generate_with_native_schema() -> None:
    llm = FakeLLM([reply('{"people": [{"name": ', '"Priya", "age": 31}]}')])
    gateway = LLMGateway({"extraction": llm})
    result = await gateway.generate("extraction", HELLO, People)
    assert result == People(people=[Person(name="Priya", age=31)])
    assert llm.schemas_seen == [People.model_json_schema()]
    assert llm.calls == [HELLO]  # no instruction added when the schema is enforced


@pytest.mark.parametrize(
    "raw",
    [
        '  {"people": []}\n',
        '```json\n{"people": []}\n```',
        '\n```\n{"people": []}\n```\n',
        '```JSON {"people": []}```',
    ],
)
async def test_generate_tolerates_whitespace_and_code_fences(raw: str) -> None:
    gateway = LLMGateway({"extraction": FakeLLM([reply(raw)])})
    assert await gateway.generate("extraction", HELLO, People) == People(people=[])


async def test_generate_falls_back_to_prompting_without_json_schema() -> None:
    llm = FakeLLM([reply('```json\n{"name": "Ana"}\n```')], json_schema=False)
    gateway = LLMGateway({"extraction": llm})
    assert await gateway.generate("extraction", HELLO, Person) == Person(name="Ana")
    assert llm.schemas_seen == [None]
    (messages,) = llm.calls
    assert messages[:-1] == HELLO
    instruction = messages[-1]
    assert instruction.role is Role.USER
    assert "JSON Schema" in instruction.content
    assert json.dumps(Person.model_json_schema(), indent=2) in instruction.content


async def test_generate_retries_with_the_validation_error() -> None:
    llm = FakeLLM([reply('{"name": 5}'), reply('{"name": "Ana"}')])
    gateway = LLMGateway({"extraction": llm})
    assert await gateway.generate("extraction", HELLO, Person) == Person(name="Ana")
    retry = llm.calls[1]
    assert retry[: len(HELLO)] == HELLO
    assert retry[-2] == Message(role=Role.ASSISTANT, content='{"name": 5}')
    assert retry[-1].role is Role.USER
    assert "name: Input should be a valid string" in retry[-1].content
    assert "errors.pydantic.dev" not in retry[-1].content
    assert llm.schemas_seen == [Person.model_json_schema()] * 2


async def test_generate_fallback_retry_keeps_the_schema_instruction() -> None:
    llm = FakeLLM([reply("Sure! Here it is."), reply('{"name": "Ana"}')], json_schema=False)
    gateway = LLMGateway({"extraction": llm})
    assert await gateway.generate("extraction", HELLO, Person) == Person(name="Ana")
    retry = llm.calls[1]
    assert "JSON Schema" in retry[len(HELLO)].content
    assert retry[-2].content == "Sure! Here it is."
    assert "Invalid JSON" in retry[-1].content


async def test_generate_raises_after_retries_with_last_raw_output() -> None:
    llm = FakeLLM([reply("nope"), reply('{"age": 3}'), reply('{"age": "x"}')])
    gateway = LLMGateway({"extraction": llm})
    with pytest.raises(StructuredOutputError, match="Person after 3 attempt") as info:
        await gateway.generate("extraction", HELLO, Person, retries=2)
    assert info.value.raw == '{"age": "x"}'
    assert "name: Field required" in info.value.errors
    assert len(llm.calls) == 3


async def test_generate_without_retries() -> None:
    llm = FakeLLM([reply("{")])
    gateway = LLMGateway({"extraction": llm})
    with pytest.raises(StructuredOutputError) as info:
        await gateway.generate("extraction", HELLO, Person, retries=0)
    assert info.value.raw == "{" and len(llm.calls) == 1
    with pytest.raises(ValueError, match="retries"):
        await gateway.generate("extraction", HELLO, Person, retries=-1)


async def test_generate_does_not_retry_adapter_errors() -> None:
    llm = FakeLLM([[Raise(RuntimeError("server down"))]])
    gateway = LLMGateway({"extraction": llm})
    with pytest.raises(RuntimeError, match="server down"):
        await gateway.generate("extraction", HELLO, Person)
    assert len(llm.calls) == 1
    with pytest.raises(GatewayError):
        await gateway.generate("nope", HELLO, Person)


async def test_generate_runs_in_the_background_lane_by_default() -> None:
    pause = Pause()
    fg_llm = FakeLLM([[pause, LLMChunk(text="hi")]])
    bg_llm = FakeLLM([reply('{"name": "Ana"}')])
    gateway = LLMGateway({"chat": fg_llm, "extraction": bg_llm})
    async with asyncio.timeout(1):
        fg = asyncio.create_task(consume(gateway, "chat", Priority.FOREGROUND))
        await pause.reached.wait()
        bg = asyncio.create_task(gateway.generate("extraction", HELLO, Person))
        await wait_until(lambda: gateway.background_waiting == 1)
        assert bg_llm.calls == []
        pause.release.set()
        await fg
        assert await bg == Person(name="Ana")


# --- Pre-emption ---------------------------------------------------------------


def test_preempted_is_not_a_cancellation() -> None:
    assert not issubclass(Preempted, asyncio.CancelledError)
    assert issubclass(Preempted, Exception)


async def start_background(
    gateway: LLMGateway, role: str, received: list[str]
) -> asyncio.Task[None]:
    async def run() -> None:
        async for chunk in gateway.stream(role, HELLO, priority=Priority.BACKGROUND):
            received.append(chunk.text or "")

    return asyncio.create_task(run())


async def test_foreground_preempts_running_background_mid_stream() -> None:
    bg_pause, fg_pause = Pause(), Pause()
    bg_llm = FakeLLM([[LLMChunk(text="x"), bg_pause, LLMChunk(text="never")]])
    fg_llm = FakeLLM([[fg_pause, LLMChunk(text="a")]])
    gateway = LLMGateway({"chat": fg_llm, "extraction": bg_llm})
    received: list[str] = []
    async with asyncio.timeout(1):
        bg = await start_background(gateway, "extraction", received)
        await bg_pause.reached.wait()
        assert gateway.background_running == 1
        fg = asyncio.create_task(consume(gateway, "chat", Priority.FOREGROUND))
        await fg_pause.reached.wait()
        with pytest.raises(Preempted):
            await bg
        # The pre-empted adapter stream was stopped and closed while the
        # foreground stream is still active.
        assert received == ["x"]
        assert bg_llm.exits == ["cancelled"] and bg_llm.open_streams == 0
        assert gateway.background_running == 0 and gateway.foreground_active == 1
        fg_pause.release.set()
        assert await fg == ["a"]
    assert gateway.foreground_active == 0


async def test_preemption_between_reads_closes_the_adapter_stream() -> None:
    bg_llm = FakeLLM([[LLMChunk(text="x"), LLMChunk(text="y")]])
    fg_pause = Pause()
    fg_llm = FakeLLM([[fg_pause, LLMChunk(text="a")]])
    gateway = LLMGateway({"chat": fg_llm, "extraction": bg_llm})
    async with asyncio.timeout(1):
        async with aclosing(
            gateway.stream("extraction", HELLO, priority=Priority.BACKGROUND)
        ) as bg:
            assert await anext(bg) == LLMChunk(text="x")  # the consumer now does other work
            fg = asyncio.create_task(consume(gateway, "chat", Priority.FOREGROUND))
            await fg_pause.reached.wait()
            await wait_until(lambda: bg_llm.open_streams == 0)
            assert bg_llm.exits == ["closed"]  # closed without the consumer's help
            with pytest.raises(Preempted):
                await anext(bg)
        assert gateway.background_running == 0
        fg_pause.release.set()
        assert await fg == ["a"]


async def test_preempted_background_resumes_after_foreground_when_retried() -> None:
    bg_pause, fg_pause = Pause(), Pause()
    bg_llm = FakeLLM([[bg_pause, LLMChunk(text="lost")], [LLMChunk(text="done")]])
    fg_llm = FakeLLM([[fg_pause, LLMChunk(text="a")]])
    gateway = LLMGateway({"chat": fg_llm, "extraction": bg_llm})
    async with asyncio.timeout(1):
        first = await start_background(gateway, "extraction", [])
        await bg_pause.reached.wait()
        fg = asyncio.create_task(consume(gateway, "chat", Priority.FOREGROUND))
        await fg_pause.reached.wait()
        with pytest.raises(Preempted):
            await first
        retry = asyncio.create_task(consume(gateway, "extraction", Priority.BACKGROUND))
        await wait_until(lambda: gateway.background_waiting == 1)
        assert len(bg_llm.calls) == 1  # the retry waits for the foreground lane
        fg_pause.release.set()
        assert await fg == ["a"]
        assert await retry == ["done"]
    assert bg_llm.exits == ["cancelled", "done"]


async def test_generate_propagates_preempted_without_retrying() -> None:
    bg_pause, fg_pause = Pause(), Pause()
    bg_llm = FakeLLM([[LLMChunk(text='{"name": '), bg_pause, LLMChunk(text='"Ana"}')]])
    fg_llm = FakeLLM([[fg_pause]])
    gateway = LLMGateway({"chat": fg_llm, "extraction": bg_llm})
    async with asyncio.timeout(1):
        job = asyncio.create_task(gateway.generate("extraction", HELLO, Person, retries=3))
        await bg_pause.reached.wait()
        fg = asyncio.create_task(consume(gateway, "chat", Priority.FOREGROUND))
        with pytest.raises(Preempted):
            await job
        assert len(bg_llm.calls) == 1
        fg_pause.release.set()
        await fg


async def test_several_background_streams_preempted_at_once() -> None:
    pauses = [Pause() for _ in range(3)]
    llms = {f"bg{i}": FakeLLM([[LLMChunk(text="x"), pause]]) for i, pause in enumerate(pauses)}
    fg_llm = FakeLLM([[LLMChunk(text="a")]])
    gateway = LLMGateway({"chat": fg_llm, **llms})
    async with asyncio.timeout(1):
        tasks = [await start_background(gateway, role, []) for role in llms]
        for pause in pauses:
            await pause.reached.wait()
        assert gateway.background_running == 3
        assert await consume(gateway, "chat", Priority.FOREGROUND) == ["a"]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(r, Preempted) for r in results)
    assert all(llm.exits == ["cancelled"] for llm in llms.values())
    assert gateway.background_running == 0


class SlowToCloseLLM:
    """Streams one chunk, then hangs; its cleanup blocks until `cleanup` is set."""

    def __init__(self) -> None:
        self.capabilities = LLMCaps(
            native_tools=False, json_schema=False, streaming=True, context_window=4096
        )
        self.reached = asyncio.Event()
        self.cleanup = asyncio.Event()
        self.cleaned = False

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[LLMChunk]:
        try:
            yield LLMChunk(text="x")
            self.reached.set()
            await asyncio.Event().wait()
        finally:
            await self.cleanup.wait()
            self.cleaned = True


async def test_foreground_does_not_wait_for_background_cleanup() -> None:
    slow = SlowToCloseLLM()
    gateway = LLMGateway({"chat": FakeLLM([[LLMChunk(text="a")]]), "extraction": slow})
    async with asyncio.timeout(1):
        bg = await start_background(gateway, "extraction", [])
        await slow.reached.wait()
        # The foreground stream completes while the background cleanup is still blocked.
        assert await consume(gateway, "chat", Priority.FOREGROUND) == ["a"]
        assert not slow.cleaned and not bg.done()
        slow.cleanup.set()
        with pytest.raises(Preempted):
            await bg
    assert slow.cleaned and gateway.background_running == 0


async def test_cancelling_a_background_consumer_is_not_preemption() -> None:
    pause = Pause()
    llm = FakeLLM([[LLMChunk(text="x"), pause]])
    gateway = LLMGateway({"extraction": llm})
    async with asyncio.timeout(1):
        task = await start_background(gateway, "extraction", [])
        await pause.reached.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert llm.exits == ["cancelled"] and llm.open_streams == 0
    assert gateway.background_running == 0


async def test_cancelling_a_consumer_during_preemption_stays_a_cancellation() -> None:
    slow = SlowToCloseLLM()
    gateway = LLMGateway({"chat": FakeLLM([[LLMChunk(text="a")]]), "extraction": slow})
    async with asyncio.timeout(1):
        bg = await start_background(gateway, "extraction", [])
        await slow.reached.wait()
        assert await consume(gateway, "chat", Priority.FOREGROUND) == ["a"]
        bg.cancel()  # while the pre-empted stream is still cleaning up
        await asyncio.sleep(0)
        slow.cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await bg
    assert slow.cleaned and gateway.background_running == 0


async def test_waiting_background_is_not_preempted() -> None:
    fg_pause, second_pause = Pause(), Pause()
    fg_llm = FakeLLM([[fg_pause, LLMChunk(text="a")], [second_pause, LLMChunk(text="b")]])
    bg_llm = FakeLLM([[LLMChunk(text="x")]])
    gateway = LLMGateway({"chat": fg_llm, "extraction": bg_llm})
    async with asyncio.timeout(1):
        fg = asyncio.create_task(consume(gateway, "chat", Priority.FOREGROUND))
        await fg_pause.reached.wait()
        bg = asyncio.create_task(consume(gateway, "extraction", Priority.BACKGROUND))
        await wait_until(lambda: gateway.background_waiting == 1)
        second = asyncio.create_task(consume(gateway, "chat", Priority.FOREGROUND))
        await second_pause.reached.wait()
        fg_pause.release.set()
        second_pause.release.set()
        assert await fg == ["a"] and await second == ["b"]
        assert await bg == ["x"]


# --- Silent streams and fallback ------------------------------------------------

SILENT = [LLMChunk(finish_reason=FinishReason.LENGTH)]
"""What the adapter sends when it detects a hidden loop: a finish reason and no output."""


def fallback_gateway(
    chat: FakeLLM, deep: FakeLLM, *, timeout: float | None = None, **extra: FakeLLM
) -> LLMGateway:
    return LLMGateway(
        {"chat": chat, "chat_deep": deep, **extra},
        fallbacks={"chat": "chat_deep"},
        first_output_timeouts={"chat": timeout} if timeout is not None else None,
    )


async def chunks_of(gateway: LLMGateway, role: str = "chat") -> list[LLMChunk]:
    return [c async for c in gateway.stream(role, HELLO, priority=Priority.FOREGROUND)]


async def test_silent_primary_is_served_by_the_fallback_role(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chat = FakeLLM([SILENT])
    deep = FakeLLM([[LLMChunk(text="Deep."), LLMChunk(finish_reason=FinishReason.STOP)]])
    gateway = fallback_gateway(chat, deep)
    assert await chunks_of(gateway) == [
        LLMChunk(text="Deep."),
        LLMChunk(finish_reason=FinishReason.STOP),
    ]
    assert deep.calls == chat.calls == [HELLO]
    assert chat.exits == deep.exits == ["done"]
    assert gateway.foreground_active == 0
    [record] = fallbacks_in_context()
    assert (record.role, record.served_by) == ("chat", "chat_deep")
    assert record.reason.startswith("ended without output") and record.lost_seconds >= 0
    assert "asking role 'chat_deep' instead" in caplog.text


async def test_stalled_primary_times_out_into_the_fallback() -> None:
    stall = Pause()
    chat = FakeLLM([[stall]])
    deep = FakeLLM([[LLMChunk(tool_call=ToolCall(id="c1", name="wikipedia"))]])
    gateway = fallback_gateway(chat, deep, timeout=0.05)
    async with asyncio.timeout(1):
        chunks = await chunks_of(gateway)
    assert [c.tool_call.name for c in chunks if c.tool_call] == ["wikipedia"]
    assert chat.exits == ["cancelled"] and chat.open_streams == 0
    assert fallbacks_in_context()[0].reason == "no output within 0.05s"


async def test_no_retry_once_text_was_yielded() -> None:
    stall = Pause()
    chat = FakeLLM([[LLMChunk(text="Partly"), stall, LLMChunk(text=" done.")]])
    deep = FakeLLM([])
    gateway = fallback_gateway(chat, deep, timeout=0.01)
    stream = gateway.stream("chat", HELLO)
    async with aclosing(stream):
        assert await anext(stream) == LLMChunk(text="Partly")
        read = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.05)  # well past the first-output timeout
        assert not read.done()
        stall.release.set()
        assert await read == LLMChunk(text=" done.")
    assert deep.calls == [] and fallbacks_in_context() == ()


async def test_held_finish_reason_is_dropped_on_fallback_and_kept_otherwise() -> None:
    # A bare finish chunk before any output is passed on once output follows.
    chat = FakeLLM([[LLMChunk(finish_reason=FinishReason.STOP), LLMChunk(text="Late.")]])
    gateway = fallback_gateway(chat, FakeLLM([]))
    assert await chunks_of(gateway) == [
        LLMChunk(finish_reason=FinishReason.STOP),
        LLMChunk(text="Late."),
    ]


async def test_both_silent_raises_a_readable_error() -> None:
    chat, deep = FakeLLM([SILENT]), FakeLLM([[Pause()]])
    gateway = LLMGateway(
        {"chat": chat, "chat_deep": deep},
        fallbacks={"chat": "chat_deep"},
        first_output_timeouts={"chat_deep": 0.05},
    )
    with pytest.raises(SilentStreamError, match="neither did its fallback role 'chat_deep'"):
        async with asyncio.timeout(1):
            await chunks_of(gateway)
    assert isinstance(SilentStreamError("x"), GatewayError)
    assert chat.open_streams == deep.open_streams == gateway.foreground_active == 0
    assert len(deep.calls) == 1  # the fallback is not retried


async def test_role_without_fallback_is_unchanged() -> None:
    chat = FakeLLM([SILENT, [LLMChunk(text="x")]])
    gateway = LLMGateway({"chat": chat})
    assert await chunks_of(gateway) == SILENT  # an empty reply, as before
    assert await chunks_of(gateway) == [LLMChunk(text="x")]
    assert fallbacks_in_context() == ()


async def test_timeout_without_fallback_raises() -> None:
    chat = FakeLLM([[Pause()]])
    gateway = LLMGateway({"chat": chat}, first_output_timeouts={"chat": 0.05})
    with pytest.raises(SilentStreamError, match="no output within 0.05s"):
        async with asyncio.timeout(1):
            await chunks_of(gateway)
    assert chat.exits == ["cancelled"]


async def test_background_lane_ignores_fallbacks() -> None:
    chat = FakeLLM([SILENT])
    gateway = fallback_gateway(chat, FakeLLM([]), timeout=0.01)
    assert await consume(gateway, "chat", Priority.BACKGROUND) == [""]


async def test_preemption_and_lanes_during_a_fallback() -> None:
    running = Pause()
    worker = FakeLLM(
        [[LLMChunk(text="bg"), running, LLMChunk(text="never")], [LLMChunk(text="ok")]]
    )
    deep_pause = Pause()
    chat = FakeLLM([SILENT])
    deep = FakeLLM([[deep_pause, LLMChunk(text="Deep.")]])
    gateway = fallback_gateway(chat, deep, worker=worker)
    received: list[str] = []
    async with asyncio.timeout(1):
        background = await start_background(gateway, "worker", received)
        await running.reached.wait()

        foreground = asyncio.create_task(chunks_of(gateway))
        with pytest.raises(Preempted):
            await background
        assert received == ["bg"]
        await deep_pause.reached.wait()  # the fallback attempt is running
        assert gateway.foreground_active == 1 and gateway.background_running == 0
        waiting = asyncio.create_task(consume(gateway, "worker", Priority.BACKGROUND))
        await wait_until(lambda: gateway.background_waiting == 1)
        assert len(worker.calls) == 1  # still waiting for the foreground lane
        deep_pause.release.set()
        assert [c.text for c in await foreground] == ["Deep."]
        assert await waiting == ["ok"]


async def test_closing_early_during_either_attempt_releases_everything() -> None:
    # During the first attempt (the consumer is cancelled while it stalls).
    chat, deep = FakeLLM([[Pause()]]), FakeLLM([])
    gateway = fallback_gateway(chat, deep, timeout=10)
    task = asyncio.create_task(chunks_of(gateway))
    await wait_until(lambda: chat.open_streams == 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert chat.exits == ["cancelled"] and gateway.foreground_active == 0 and deep.calls == []

    # During the fallback attempt (the consumer closes the stream after a chunk).
    chat = FakeLLM([SILENT])
    deep = FakeLLM([[LLMChunk(text="first"), Pause(), LLMChunk(text="second")]])
    gateway = fallback_gateway(chat, deep)
    stream = gateway.stream("chat", HELLO)
    assert await anext(stream) == LLMChunk(text="first")
    await stream.aclose()
    assert deep.exits == ["closed"] and deep.open_streams == chat.open_streams == 0
    assert gateway.foreground_active == 0


def test_invalid_fallback_role_is_rejected() -> None:
    with pytest.raises(GatewayError, match="invalid fallback"):
        LLMGateway({"chat": FakeLLM([])}, fallbacks={"chat": "chat_deep"})


async def test_from_settings_takes_fallbacks_and_shares_adapters() -> None:
    registry = Registry()
    registry.register("llm", "fake")(RegisteredLLM)
    settings = LLMSettings.model_validate(
        {
            "roles": {
                "chat": {
                    "adapter": "fake",
                    "model": "m",
                    "fallback": "chat_deep",
                    "first_output_timeout": 20,
                },
                "summary": {"adapter": "fake", "model": "m", "after_tool": "chat_deep"},
                "chat_deep": {"adapter": "fake", "model": "deep"},
            }
        }
    )
    assert settings.roles["chat"].options == {"model": "m"}
    gateway = LLMGateway.from_settings(settings, registry=registry)
    # Same adapter and options: one instance, whatever the fallback settings.
    assert gateway.for_role("chat") is gateway.for_role("summary")
    assert gateway._fallbacks == {"chat": "chat_deep"}
    assert gateway._first_output_timeouts == {"chat": 20}
    assert gateway._after_tool == {"summary": "chat_deep"}


def test_add_fallbacks_annotates_the_step_it_happened_in() -> None:
    t0 = datetime(2026, 11, 12, 10, 0, tzinfo=UTC)
    step = StepTrace(role="chat", messages=HELLO, started_at=t0, ended_at=t0 + timedelta(seconds=2))
    later = step.model_copy(
        update={"started_at": t0 + timedelta(seconds=3), "ended_at": t0 + timedelta(seconds=5)}
    )
    trace = TurnTrace(
        turn_id="t",
        session_id="s",
        started_at=t0,
        ended_at=t0 + timedelta(seconds=5),
        steps=[step, later],
        outcome="completed",
    )
    record = StepFallback(
        role="chat",
        served_by="chat_deep",
        reason="r",
        lost_seconds=0.5,
        at=t0 + timedelta(seconds=3.5),
    )
    assert add_fallbacks(trace) is trace  # nothing recorded in this context
    token = gateway_module._fallbacks.set((record,))
    try:
        annotated = add_fallbacks(trace)
    finally:
        gateway_module._fallbacks.reset(token)
    assert [s.fallback for s in annotated.steps] == [None, record]


# --- Answer steps after a tool result ----------------------------------------------

TOOL_RESULT = [
    *HELLO,
    Message(role=Role.ASSISTANT, tool_calls=[ToolCall(id="c1", name="wikipedia")]),
    Message(role=Role.TOOL, content="Article: ...", tool_call_id="c1"),
]


async def test_after_tool_role_serves_the_answer_step() -> None:
    chat = FakeLLM([[LLMChunk(text="fast")]])
    deep = FakeLLM([[LLMChunk(text="careful")]])
    gateway = LLMGateway({"chat": chat, "chat_deep": deep}, after_tool={"chat": "chat_deep"})
    answer = [c.text async for c in gateway.stream("chat", TOOL_RESULT)]
    plain = [c.text async for c in gateway.stream("chat", HELLO)]
    assert (answer, plain) == (["careful"], ["fast"])
    assert deep.calls == [TOOL_RESULT] and chat.calls == [HELLO]
    # Background streams are not rerouted.
    chat_bg = FakeLLM([[LLMChunk(text="bg")]])
    gateway = LLMGateway({"chat": chat_bg, "chat_deep": deep}, after_tool={"chat": "chat_deep"})
    assert await consume_messages(gateway, TOOL_RESULT, Priority.BACKGROUND) == ["bg"]


async def test_after_tool_role_uses_its_own_fallback_settings() -> None:
    chat, deep, spare = FakeLLM([]), FakeLLM([SILENT]), FakeLLM([[LLMChunk(text="spare")]])
    gateway = LLMGateway(
        {"chat": chat, "chat_deep": deep, "spare": spare},
        after_tool={"chat": "chat_deep"},
        fallbacks={"chat_deep": "spare"},
    )
    assert [c.text async for c in gateway.stream("chat", TOOL_RESULT)] == ["spare"]
    assert fallbacks_in_context()[-1].role == "chat_deep"


def test_invalid_after_tool_role_is_rejected() -> None:
    with pytest.raises(GatewayError, match="invalid after_tool"):
        LLMGateway({"chat": FakeLLM([])}, after_tool={"chat": "chat"})


async def consume_messages(
    gateway: LLMGateway, messages: list[Message], priority: Priority
) -> list[str]:
    return [c.text or "" async for c in gateway.stream("chat", messages, priority=priority)]
