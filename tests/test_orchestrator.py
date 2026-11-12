import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from lucy.core.bus import EventBus
from lucy.core.context import ContextBuilder
from lucy.core.events import (
    OUTBOUND_EVENTS,
    AssistantMessage,
    Event,
    Interrupt,
    SentenceReady,
    TextDelta,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    UserMessage,
)
from lucy.core.gateway import LLMGateway
from lucy.core.orchestrator import Orchestrator
from lucy.core.ports import Tool, tool_spec
from lucy.core.trace import TurnTrace
from lucy.core.types import (
    FinishReason,
    LLMChunk,
    MemoryContext,
    Message,
    Risk,
    Role,
    ToolCall,
    ToolOutput,
)
from tests.fakes import (
    EchoTool,
    FakeLLM,
    InMemoryMemoryStore,
    InMemoryTraceSink,
    Pause,
    Raise,
    Script,
)

PERSONA = "You are Lucy."
SYSTEM = Message(role=Role.SYSTEM, content=PERSONA)


def text(*parts: str) -> list[LLMChunk]:
    chunks = [LLMChunk(text=p) for p in parts]
    return [*chunks, LLMChunk(finish_reason=FinishReason.STOP)]


def calls(*tool_calls: ToolCall) -> list[LLMChunk]:
    chunks = [LLMChunk(tool_call=c) for c in tool_calls]
    return [*chunks, LLMChunk(finish_reason=FinishReason.TOOL_CALLS)]


class RecordingTool:
    def __init__(self, name: str, risk: Risk, error: Exception | None = None) -> None:
        self.name = name
        self.description = f"The {name} tool."
        self.parameters: dict[str, Any] = {"type": "object", "properties": {}}
        self.risk = risk
        self.error = error
        self.runs: list[dict[str, Any]] = []

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        self.runs.append(args)
        if self.error is not None:
            raise self.error
        return ToolOutput(content=f"{self.name} done")


class Harness:
    """An orchestrator on a bus with a recorder of every outbound event."""

    def __init__(
        self,
        scripts: Sequence[Script],
        *,
        tools: Sequence[Tool] = (),
        max_steps: int = 8,
        memory: InMemoryMemoryStore | None = None,
        trace: Any = None,
    ) -> None:
        self.bus = EventBus()
        self.out = self.bus.subscribe(*OUTBOUND_EVENTS)
        self.llm = FakeLLM(scripts)
        self.memory = memory if memory is not None else InMemoryMemoryStore()
        self.trace = trace if trace is not None else InMemoryTraceSink()
        self.orchestrator = Orchestrator(
            bus=self.bus,
            gateway=LLMGateway({"chat": self.llm}),
            memory=self.memory,
            context=ContextBuilder(PERSONA),
            tools=tools,
            trace=self.trace,
            max_steps=max_steps,
        )
        self.task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "Harness":
        self.task = asyncio.create_task(self.orchestrator.run())
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.bus.close()
        assert self.task is not None
        async with asyncio.timeout(1):
            await self.task

    async def say(self, text: str, session: str = "s") -> UserMessage:
        message = UserMessage(session_id=session, text=text)
        await self.bus.publish(message)
        return message

    async def until_turns_end(self, count: int = 1) -> list[Event]:
        """Outbound events up to and including the `count`th TurnEnded."""
        events: list[Event] = []
        async with asyncio.timeout(1):
            async for event in self.out:
                events.append(event)
                if sum(isinstance(e, TurnEnded) for e in events) == count:
                    return events
        raise AssertionError(f"bus closed after {events}")

    def persisted(self) -> list[tuple[str, ...]]:
        return [summary(e) for e in self.memory.episodes]


def summary(event: Event) -> tuple[str, ...]:
    """A compact, comparable view of an event."""
    match event:
        case UserMessage(text=t) | TextDelta(text=t) | SentenceReady(text=t):
            return (event.type, t)
        case AssistantMessage(text=t, tool_calls=c, interrupted=i):
            return (event.type, t, ",".join(x.name for x in c), "interrupted" if i else "")
        case ToolStarted(call=c):
            return (event.type, c.name)
        case ToolFinished(result=r):
            return (event.type, r.content, "error" if r.is_error else "ok")
        case TurnEnded(reason=r, error=e):
            return (event.type, r, e or "")
    raise AssertionError(event)


async def test_simple_turn_publishes_and_persists_in_order() -> None:
    async with Harness([text("Hello there. ", "How are you?")]) as h:
        message = await h.say("hi")
        events = await h.until_turns_end()

    assert [summary(e) for e in events] == [
        ("text_delta", "Hello there. "),
        ("sentence_ready", "Hello there."),
        ("text_delta", "How are you?"),
        ("sentence_ready", "How are you?"),
        ("assistant_message", "Hello there. How are you?", "", ""),
        ("turn_ended", "completed", ""),
    ]
    assert h.persisted() == [
        ("user_message", "hi"),
        ("assistant_message", "Hello there. How are you?", "", ""),
        ("turn_ended", "completed", ""),
    ]
    turn_ids = {e.turn_id for e in [*events, *h.memory.episodes]}
    assert len(turn_ids) == 1 and None not in turn_ids
    assert {e.session_id for e in [*events, *h.memory.episodes]} == {"s"}
    assert h.memory.episodes[0].id == message.id  # a copy of the inbound message
    assert h.llm.calls == [[SYSTEM, Message(role=Role.USER, content="hi")]]


async def test_tool_result_is_fed_back_to_the_llm() -> None:
    call = ToolCall(id="c1", name="echo", arguments={"text": "ping"})
    async with Harness([calls(call), text("It said ping.")], tools=[EchoTool()]) as h:
        await h.say("echo ping")
        events = await h.until_turns_end()

    assert [summary(e) for e in events] == [
        ("assistant_message", "", "echo", ""),
        ("tool_started", "echo"),
        ("tool_finished", "ping", "ok"),
        ("text_delta", "It said ping."),
        ("sentence_ready", "It said ping."),
        ("assistant_message", "It said ping.", "", ""),
        ("turn_ended", "completed", ""),
    ]
    assert h.persisted() == [
        ("user_message", "echo ping"),
        ("assistant_message", "", "echo", ""),
        ("tool_started", "echo"),
        ("tool_finished", "ping", "ok"),
        ("assistant_message", "It said ping.", "", ""),
        ("turn_ended", "completed", ""),
    ]
    assert h.llm.tools_seen[0] == [tool_spec(EchoTool())]
    assert h.llm.calls[1] == [
        SYSTEM,
        Message(role=Role.USER, content="echo ping"),
        Message(role=Role.ASSISTANT, content="", tool_calls=[call]),
        Message(role=Role.TOOL, content="ping", tool_call_id="c1"),
    ]


async def test_unknown_tool_gives_error_result() -> None:
    call = ToolCall(id="c1", name="missing")
    async with Harness([calls(call), text("Sorry.")]) as h:
        await h.say("do it")
        events = await h.until_turns_end()
    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert finished[0].result.is_error and "Unknown tool 'missing'" in finished[0].result.content
    assert h.llm.calls[1][-1].content == finished[0].result.content
    assert isinstance(events[-1], TurnEnded) and events[-1].reason == "completed"


async def test_raising_tool_gives_error_result() -> None:
    tool = RecordingTool("flaky", Risk.ACT, error=ValueError("nope"))
    async with Harness(
        [calls(ToolCall(id="c1", name="flaky")), text("It broke.")], tools=[tool]
    ) as h:
        await h.say("try it")
        events = await h.until_turns_end()
    result = next(e for e in events if isinstance(e, ToolFinished)).result
    assert result.is_error and "ValueError: nope" in result.content
    assert tool.runs == [{}]
    assert isinstance(events[-1], TurnEnded) and events[-1].reason == "completed"


async def test_act_tool_runs_and_dangerous_tool_does_not() -> None:
    act = RecordingTool("lights", Risk.ACT)
    danger = RecordingTool("delete_all", Risk.DANGEROUS)
    script = [
        calls(ToolCall(id="c1", name="lights"), ToolCall(id="c2", name="delete_all")),
        text("Done."),
    ]
    async with Harness(script, tools=[act, danger]) as h:
        await h.say("lights off and delete everything")
        events = await h.until_turns_end()
    results = [e.result for e in events if isinstance(e, ToolFinished)]
    assert act.runs == [{}] and danger.runs == []
    assert results[0].content == "lights done" and not results[0].is_error
    assert results[1].is_error and "confirmation" in results[1].content
    assert results[1].call_id == "c2"


async def test_step_limit_ends_turn_with_error() -> None:
    script = [calls(ToolCall(id="c1", name="echo")), calls(ToolCall(id="c2", name="echo"))]
    async with Harness(script, tools=[EchoTool()], max_steps=2) as h:
        await h.say("loop")
        events = await h.until_turns_end()
    ended = events[-1]
    assert isinstance(ended, TurnEnded) and ended.reason == "error"
    assert ended.error is not None and "step limit" in ended.error
    assert len(h.llm.calls) == 2
    # The last step's tool calls are not run once the limit is reached.
    assert [summary(e) for e in events if isinstance(e, ToolStarted)] == [("tool_started", "echo")]
    assert h.persisted()[-1] == summary(ended)


async def test_interrupt_mid_stream_records_delivered_text_then_queued_turn_runs() -> None:
    pause = Pause()
    script = [[LLMChunk(text="Once upon "), pause, LLMChunk(text="a time.")], text("Okay.")]
    async with Harness(script) as h:
        await h.say("tell me a story")
        await h.say("never mind")  # queued behind the first turn
        async with asyncio.timeout(1):
            await pause.reached.wait()
        await h.bus.publish(Interrupt(session_id="s"))
        events = await h.until_turns_end(2)

    assert [summary(e) for e in events] == [
        ("text_delta", "Once upon "),
        ("assistant_message", "Once upon ", "", "interrupted"),
        ("turn_ended", "interrupted", ""),
        ("text_delta", "Okay."),
        ("sentence_ready", "Okay."),
        ("assistant_message", "Okay.", "", ""),
        ("turn_ended", "completed", ""),
    ]
    assert h.persisted() == [
        ("user_message", "tell me a story"),
        ("assistant_message", "Once upon ", "", "interrupted"),
        ("turn_ended", "interrupted", ""),
        ("user_message", "never mind"),
        ("assistant_message", "Okay.", "", ""),
        ("turn_ended", "completed", ""),
    ]
    # The second turn sees the delivered part of the interrupted reply.
    assert h.llm.calls[1][1:] == [
        Message(role=Role.USER, content="tell me a story"),
        Message(role=Role.ASSISTANT, content="Once upon "),
        Message(role=Role.USER, content="never mind"),
    ]
    assert [t.outcome for t in h.trace.traces] == ["interrupted", "completed"]


async def test_interrupt_before_any_text_records_only_turn_end() -> None:
    pause = Pause()
    async with Harness([[pause, LLMChunk(text="late")]]) as h:
        await h.say("hi")
        async with asyncio.timeout(1):
            await pause.reached.wait()
        await h.bus.publish(Interrupt(session_id="s"))
        events = await h.until_turns_end()
    assert [summary(e) for e in events] == [("turn_ended", "interrupted", "")]
    assert h.persisted() == [("user_message", "hi"), ("turn_ended", "interrupted", "")]


async def test_interrupt_without_running_turn_is_ignored() -> None:
    async with Harness([text("Hi.")]) as h:
        await h.bus.publish(Interrupt(session_id="s"))
        await h.say("hello")
        events = await h.until_turns_end()
    assert isinstance(events[-1], TurnEnded) and events[-1].reason == "completed"


async def test_llm_error_ends_turn_and_next_turn_still_works() -> None:
    script = [[LLMChunk(text="Hm"), Raise(RuntimeError("model down"))], text("Back.")]
    async with Harness(script) as h:
        await h.say("first")
        first = await h.until_turns_end()
        await h.say("second")
        second = await h.until_turns_end()
    assert summary(first[-1]) == ("turn_ended", "error", "RuntimeError: model down")
    assert summary(second[-1]) == ("turn_ended", "completed", "")
    assert h.persisted()[:2] == [
        ("user_message", "first"),
        ("turn_ended", "error", "RuntimeError: model down"),
    ]
    assert [t.outcome for t in h.trace.traces] == ["error", "completed"]
    assert h.trace.traces[0].error == "RuntimeError: model down"


async def test_memory_error_ends_turn_with_error() -> None:
    class BrokenRetrieval(InMemoryMemoryStore):
        async def retrieve(self, query: str, budget: int) -> MemoryContext:
            raise OSError("disk gone")

    async with Harness([], memory=BrokenRetrieval()) as h:
        await h.say("hi")
        events = await h.until_turns_end()
    assert summary(events[-1]) == ("turn_ended", "error", "OSError: disk gone")


async def test_turns_in_one_session_run_in_order_and_see_history() -> None:
    pause = Pause()
    script = [[pause, *text("First answer.")], text("Second answer.")]
    async with Harness(script) as h:
        await h.say("one")
        await h.say("two")
        async with asyncio.timeout(1):
            await pause.reached.wait()
        assert len(h.llm.calls) == 1
        pause.release.set()
        await h.until_turns_end(2)
    assert h.llm.calls[1] == [
        SYSTEM,
        Message(role=Role.USER, content="one"),
        Message(role=Role.ASSISTANT, content="First answer."),
        Message(role=Role.USER, content="two"),
    ]


async def test_sessions_run_concurrently() -> None:
    pause = Pause()
    async with Harness([[pause, *text("A.")], text("B.")]) as h:
        await h.say("a", session="a")
        async with asyncio.timeout(1):
            await pause.reached.wait()
        await h.say("b", session="b")
        events = await h.until_turns_end()  # session b finishes while a is paused
        assert events[-1].session_id == "b"
        pause.release.set()
        events = await h.until_turns_end()
        assert events[-1].session_id == "a"


async def test_trace_records_exact_messages_and_context_report() -> None:
    call = ToolCall(id="c1", name="echo", arguments={"text": "x"})
    async with Harness([calls(call), text("Done.")], tools=[EchoTool()]) as h:
        await h.say("go")
        await h.until_turns_end()
    [trace] = h.trace.traces
    assert isinstance(trace, TurnTrace)
    assert trace.outcome == "completed" and trace.error is None
    assert trace.turn_id == h.memory.episodes[0].turn_id and trace.session_id == "s"
    assert [step.messages for step in trace.steps] == h.llm.calls
    assert [step.finish_reason for step in trace.steps] == [
        FinishReason.TOOL_CALLS,
        FinishReason.STOP,
    ]
    assert all(step.role == "chat" and step.tools == ["echo"] for step in trace.steps)
    for step in trace.steps:
        assert step.first_token_at is not None
        assert step.started_at <= step.first_token_at <= step.ended_at
    assert (
        trace.started_at <= trace.steps[0].started_at and trace.ended_at >= trace.steps[-1].ended_at
    )
    assert [s.name for s in trace.context] == [
        "persona",
        "profile",
        "instructions",
        "memories",
        "history",
    ]


async def test_failing_trace_sink_does_not_break_turns() -> None:
    class BrokenSink:
        async def record(self, trace: TurnTrace) -> None:
            raise RuntimeError("sink down")

    async with Harness([text("One."), text("Two.")], trace=BrokenSink()) as h:
        await h.say("a")
        first = await h.until_turns_end()
        await h.say("b")
        second = await h.until_turns_end()
    assert summary(first[-1]) == summary(second[-1]) == ("turn_ended", "completed", "")


async def test_run_exits_cleanly_when_bus_closes() -> None:
    h = Harness([])
    await h.__aenter__()
    await h.__aexit__()
    assert h.task is not None and h.task.done() and h.task.exception() is None


async def test_closing_bus_mid_turn_stops_the_turn_and_records_it() -> None:
    pause = Pause()
    h = Harness([[LLMChunk(text="Partial"), pause]])
    async with h:
        await h.say("hi")
        async with asyncio.timeout(1):
            await pause.reached.wait()
    assert h.persisted() == [
        ("user_message", "hi"),
        ("assistant_message", "Partial", "", "interrupted"),
        ("turn_ended", "interrupted", ""),
    ]


async def test_cancelling_run_cancels_turns_and_propagates() -> None:
    pause = Pause()
    h = Harness([[LLMChunk(text="Partial"), pause]])
    await h.__aenter__()
    assert h.task is not None
    await h.say("hi")
    async with asyncio.timeout(1):
        await pause.reached.wait()
        h.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await h.task
    assert h.persisted()[-1] == ("turn_ended", "interrupted", "")
    # Interrupted events were still published, since the bus is open.
    events = await h.until_turns_end()
    assert summary(events[-1]) == ("turn_ended", "interrupted", "")


async def test_event_published_right_after_start_is_not_lost() -> None:
    h = Harness([text("Hi.")])
    await h.say("before run is scheduled")
    async with h:
        events = await h.until_turns_end()
    assert summary(events[-1]) == ("turn_ended", "completed", "")
