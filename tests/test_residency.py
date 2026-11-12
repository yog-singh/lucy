"""The model residency policy (`lucy.core.residency`), with fakes only."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import aclosing
from typing import Any

import pytest

from lucy.core.bus import EventBus
from lucy.core.context import ContextBuilder
from lucy.core.events import SpeechActivity, TurnEnded, UserMessage
from lucy.core.gateway import LLMGateway, Priority
from lucy.core.orchestrator import Orchestrator
from lucy.core.residency import Residency, resident
from lucy.core.types import (
    FinishReason,
    LLMChunk,
    MemoryContext,
    Message,
    PromptPrefix,
    Role,
    ToolCall,
    ToolSpec,
)
from tests.fakes import EchoTool, FakeLLM, InMemoryMemoryStore, ModelServer, Pause, ResidentFake

GB = 1_000_000_000


def reply(text: str) -> list[LLMChunk]:
    return [LLMChunk(text=text), LLMChunk(finish_reason=FinishReason.STOP)]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Jobs:
    def __init__(self) -> None:
        self.running = False
        self.due = False

    async def has_due(self) -> bool:
        return self.due


PREFIX = PromptPrefix(
    role="chat",
    messages=[
        Message(role=Role.SYSTEM, content="You are Lucy."),
        Message(role=Role.USER, content="hi"),
    ],
    tools=[ToolSpec(name="t", description="d")],
)


class Rig:
    def __init__(
        self,
        server: ModelServer | None = None,
        *,
        llms: dict[str, Any] | None = None,
        prefix: PromptPrefix | None | Exception = PREFIX,
        **options: Any,
    ) -> None:
        self.server = server or ModelServer()
        self.llm = ResidentFake(self.server, [reply("hello")] * 5)
        self.gateway = LLMGateway(llms or {"chat": self.llm})
        self.bus = EventBus()
        self.clock = Clock()
        self.jobs = Jobs()
        self.prefix_asked: list[str] = []

        async def next_prefix(session: str) -> PromptPrefix | None:
            self.prefix_asked.append(session)
            if isinstance(prefix, Exception):
                raise prefix
            return prefix

        options.setdefault("text_quiet_seconds", 20.0)
        options.setdefault("clock", self.clock)
        self.residency = Residency(
            bus=self.bus, gateway=self.gateway, jobs=self.jobs, prefix=next_prefix, **options
        )

    def turn(self, session: str = "s") -> None:
        """A turn starts, makes a model request, and ends."""
        self.residency.handle(UserMessage(session_id=session, text="hi"))
        self.used()
        self.residency.handle(TurnEnded(session_id=session, reason="completed"))

    def used(self, role: str = "chat") -> None:
        self.residency._on_request(role, self.gateway.for_role(role))

    async def idle_for(self, seconds: float) -> None:
        """Tick at the start of an idle period and again `seconds` later."""
        await self.residency.tick()
        self.clock.now += seconds
        await self.residency.tick()

    async def settle(self) -> None:
        """Let a started cycle run to the end."""
        cycle = self.residency._cycle
        if cycle is not None and cycle.task is not None:
            await asyncio.wait({cycle.task}, timeout=2)

    def voice(self, state: str = "idle", session: str = "v") -> None:
        self.residency.handle(SpeechActivity(session_id=session, state=state))  # type: ignore[arg-type]


# --- Busy guard ------------------------------------------------------------------------


async def test_idle_after_model_work_unloads_and_rewarms() -> None:
    rig = Rig()
    rig.turn()
    rig.clock.now += 30
    await rig.idle_for(1.0)
    await rig.settle()
    assert rig.server.calls == ["size", "unload", "prefill:", "prefill:x", "size"]
    record = rig.residency.records[-1]
    assert record.unloaded == ["srv|m"] and record.size_before == 14 * GB
    assert record.size_after == 9 * GB and record.outcome == "done"


async def test_a_turn_in_flight_is_busy_even_between_tool_steps() -> None:
    """From UserMessage to TurnEnded nothing unloads, though no stream is open."""
    rig = Rig()
    rig.used()
    rig.residency.handle(UserMessage(session_id="s", text="look it up"))
    rig.clock.now += 60
    assert await rig.residency.busy_reason() == "turn"
    await rig.idle_for(60)
    assert rig.server.calls == []
    rig.residency.handle(TurnEnded(session_id="s", reason="completed"))
    assert await rig.residency.busy_reason() is None


@pytest.mark.parametrize("state", ["speaking", "transcribing"])
async def test_speech_in_progress_is_busy(state: str) -> None:
    rig = Rig()
    rig.used()  # loaded, so speech starting does not load it
    rig.voice(state)
    assert await rig.residency.busy_reason() == "speech"
    rig.voice("idle")
    assert await rig.residency.busy_reason() is None


async def test_open_streams_are_busy() -> None:
    server = ModelServer()
    pause = Pause()
    llm = ResidentFake(server, [[LLMChunk(text="a"), pause, *reply("b")]])
    rig = Rig(server, llms={"chat": llm})
    stream = rig.gateway.stream("chat", [Message(role=Role.USER, content="x")])
    async with aclosing(stream):
        task = asyncio.create_task(anext(stream))
        await task
        assert await rig.residency.busy_reason() == "foreground stream"


async def test_background_streams_running_or_waiting_are_busy() -> None:
    server = ModelServer()
    fg, bg = Pause(), Pause()
    llm = ResidentFake(server, [[fg, *reply("f")], [bg, *reply("b")]])
    rig = Rig(server, llms={"chat": llm})

    async def consume(priority: Priority) -> None:
        async with aclosing(rig.gateway.stream("chat", [], priority=priority)) as chunks:
            async for _ in chunks:
                pass

    foreground = asyncio.create_task(consume(Priority.FOREGROUND))
    await fg.reached.wait()
    background = asyncio.create_task(consume(Priority.BACKGROUND))
    await asyncio.sleep(0)
    assert rig.gateway.background_waiting == 1
    fg.release.set()
    await foreground
    await bg.reached.wait()
    assert rig.gateway.background_running == 1
    assert await rig.residency.busy_reason() == "background stream"
    bg.release.set()
    await background
    assert await rig.residency.busy_reason() is None


async def test_waiting_background_stream_alone_is_busy() -> None:
    rig = Rig()
    rig.gateway._background_waiting = 1  # as while a background stream waits for the lane
    assert await rig.residency.busy_reason() == "background stream"


async def test_a_running_job_and_a_due_job_are_busy() -> None:
    rig = Rig()
    rig.jobs.running = True
    assert await rig.residency.busy_reason() == "job"
    rig.jobs.running = False
    rig.jobs.due = True  # due but not started yet
    assert await rig.residency.busy_reason() == "job due"
    rig.turn()
    rig.clock.now += 60
    await rig.idle_for(60)
    assert rig.server.calls == []


async def test_a_failing_job_store_counts_as_busy() -> None:
    rig = Rig()

    async def broken() -> bool:
        raise OSError("disk")

    rig.jobs.has_due = broken  # type: ignore[method-assign]
    assert await rig.residency.busy_reason() == "job store unavailable"


async def test_its_own_warm_up_is_busy() -> None:
    rig = Rig()
    rig.server.hold["prefill:"] = pause = Pause()
    rig.turn()
    rig.clock.now += 30
    await rig.idle_for(1)
    await pause.reached.wait()
    assert await rig.residency.busy_reason() == "warm-up"
    pause.release.set()
    await rig.settle()
    assert await rig.residency.busy_reason() is None


# --- Idle timing -----------------------------------------------------------------------


async def test_voice_unloads_after_the_short_idle_period() -> None:
    rig = Rig(idle_seconds=1.0)
    rig.voice("idle")
    rig.turn("v")
    await rig.idle_for(0.9)
    assert rig.server.calls == []
    rig.clock.now += 0.2
    await rig.residency.tick()
    await rig.settle()
    assert "unload" in rig.server.calls
    assert rig.residency.records[-1].reason == "voice idle"


async def test_text_waits_for_the_quiet_period_after_the_last_turn() -> None:
    rig = Rig(idle_seconds=1.0, text_quiet_seconds=20.0)
    rig.turn()
    await rig.idle_for(5)  # idle, but the user may still be typing a reply
    rig.clock.now += 14.9
    await rig.residency.tick()
    assert rig.server.calls == []
    rig.clock.now += 0.2
    await rig.residency.tick()
    await rig.settle()
    assert "unload" in rig.server.calls
    assert rig.residency.records[-1].reason == "idle"


async def test_busy_restarts_the_idle_period() -> None:
    rig = Rig(idle_seconds=1.0, text_quiet_seconds=0)
    rig.turn()
    await rig.idle_for(0.8)
    rig.jobs.running = True
    await rig.residency.tick()
    rig.jobs.running = False
    await rig.idle_for(0.8)
    assert rig.server.calls == []
    rig.clock.now += 0.3
    await rig.residency.tick()
    await rig.settle()
    assert "unload" in rig.server.calls


async def test_the_voice_session_ending_returns_to_the_text_rule() -> None:
    rig = Rig(idle_seconds=1.0, text_quiet_seconds=20.0)
    rig.voice("idle")
    assert rig.residency.voice_active
    rig.voice("ended")
    assert not rig.residency.voice_active
    rig.turn()
    await rig.idle_for(2)
    assert rig.server.calls == []


async def test_no_unload_when_nothing_ran_since_the_last_one() -> None:
    rig = Rig(text_quiet_seconds=0)
    await rig.idle_for(5)
    assert rig.server.calls == []
    rig.turn()
    await rig.idle_for(5)
    await rig.settle()
    assert rig.server.calls.count("unload") == 1
    await rig.idle_for(5)
    await rig.idle_for(5)
    await rig.settle()
    assert rig.server.calls.count("unload") == 1  # the warm-up requests do not count
    rig.used(), await rig.idle_for(5), await rig.settle()
    assert rig.server.calls.count("unload") == 2


async def test_size_threshold() -> None:
    rig = Rig(text_quiet_seconds=0, min_size_bytes=12 * GB)
    rig.server.size = 10 * GB
    rig.turn()
    await rig.idle_for(5)
    await rig.settle()
    assert rig.server.calls == ["size"]
    assert rig.residency.records[-1].outcome == "below the size limit"
    await rig.idle_for(5)  # no new activity: not even asked again
    assert rig.server.calls == ["size"]
    rig.server.size = 13 * GB
    rig.used()
    await rig.idle_for(5)
    await rig.settle()
    assert rig.server.calls[:3] == ["size", "size", "unload"]


async def test_a_model_already_gone_is_not_unloaded_or_rewarmed() -> None:
    rig = Rig(text_quiet_seconds=0)
    rig.server.size = None  # e.g. the server's own keep-alive expired
    rig.turn()
    await rig.idle_for(5)
    await rig.settle()
    assert rig.server.calls == ["size"]
    assert rig.residency.records[-1].outcome == "not loaded"


# --- Re-warm ---------------------------------------------------------------------------


async def test_rewarm_sends_the_prefix_twice_with_different_endings() -> None:
    rig = Rig(text_quiet_seconds=0)
    rig.turn("chat-1")
    await rig.idle_for(5)
    await rig.settle()
    assert rig.prefix_asked == ["chat-1"]
    first, second = rig.server.prefills
    assert first == [*PREFIX.messages, Message(role=Role.USER, content="")]
    assert second == [*PREFIX.messages, Message(role=Role.USER, content="x")]
    assert rig.residency.records[-1].warm_steps == ["prefill_1", "prefill_2"]


async def test_the_most_recently_active_session_is_warmed() -> None:
    rig = Rig(text_quiet_seconds=0)
    rig.turn("a")
    rig.turn("b")
    await rig.idle_for(5)
    await rig.settle()
    assert rig.prefix_asked == ["b"]


async def test_without_a_prefix_the_model_is_only_loaded() -> None:
    rig = Rig(text_quiet_seconds=0, prefix=None)
    rig.turn()
    await rig.idle_for(5)
    await rig.settle()
    assert rig.server.calls == ["size", "unload", "load", "size"]


async def test_a_failing_prefix_falls_back_to_a_load(caplog: pytest.LogCaptureFixture) -> None:
    rig = Rig(text_quiet_seconds=0, prefix=RuntimeError("db"))
    rig.turn()
    await rig.idle_for(5)
    await rig.settle()
    assert rig.server.calls == ["size", "unload", "load", "size"]
    assert "cannot build the next prompt prefix" in caplog.text


async def test_the_second_prefill_is_skipped_when_a_turn_begins() -> None:
    rig = Rig(text_quiet_seconds=0)
    rig.server.hold["prefill:"] = pause = Pause()
    rig.turn()
    await rig.idle_for(5)
    await pause.reached.wait()
    rig.residency.handle(UserMessage(session_id="s", text="next"))
    pause.release.set()
    await rig.settle()
    assert "prefill:x" not in rig.server.calls
    assert rig.residency.records[-1].warm_steps == ["prefill_1", "skipped prefill_2"]


async def test_the_second_prefill_is_skipped_when_speech_begins() -> None:
    rig = Rig()
    rig.server.hold["prefill:"] = pause = Pause()
    rig.voice("idle")
    rig.turn("v")
    await rig.idle_for(5)
    await pause.reached.wait()
    rig.voice("speaking")
    pause.release.set()
    await rig.settle()
    assert "prefill:x" not in rig.server.calls


async def test_speech_after_the_unload_loads_instead_of_prefilling() -> None:
    rig = Rig()
    rig.server.hold["unload"] = pause = Pause()
    rig.voice("idle")
    rig.turn("v")
    await rig.idle_for(5)
    await pause.reached.wait()
    rig.voice("speaking")
    pause.release.set()
    await rig.settle()
    assert rig.server.calls == ["size", "unload", "load", "size"]


# --- A turn during a warm-up --------------------------------------------------------------


async def test_a_turn_waits_for_the_second_prefill_then_runs() -> None:
    rig = Rig(text_quiet_seconds=0)
    rig.server.hold["prefill:x"] = pause = Pause()
    rig.turn()
    await rig.idle_for(5)
    await pause.reached.wait()
    gate = asyncio.create_task(rig.residency.before_foreground())
    await asyncio.sleep(0.01)
    assert not gate.done()  # waiting: once it finishes the turn finds its prefix cached
    pause.release.set()
    await asyncio.wait_for(gate, 1)
    assert rig.residency._cycle is None
    assert rig.residency.records[-1].outcome == "done"


async def test_a_turn_cancels_the_first_prefill() -> None:
    rig = Rig(text_quiet_seconds=0)
    rig.server.hold["prefill:"] = pause = Pause()
    rig.turn()
    await rig.idle_for(5)
    await pause.reached.wait()
    await asyncio.wait_for(rig.residency.before_foreground(), 1)
    assert rig.residency.records[-1].outcome == "cancelled during prefill_1"
    assert "prefill:x" not in rig.server.calls
    assert await rig.residency.busy_reason() is None


async def test_a_turn_waits_for_an_unload_then_skips_the_rewarm() -> None:
    rig = Rig(text_quiet_seconds=0)
    rig.server.hold["unload"] = pause = Pause()
    rig.turn()
    await rig.idle_for(5)
    await pause.reached.wait()
    gate = asyncio.create_task(rig.residency.before_foreground())
    await asyncio.sleep(0.01)
    assert not gate.done()
    pause.release.set()
    await asyncio.wait_for(gate, 1)
    assert rig.server.calls == ["size", "unload"]
    assert rig.residency.records[-1].outcome == "stopped for a turn"


async def test_a_turn_through_the_gateway_goes_after_the_warm_up() -> None:
    """The real gateway path: the foreground stream starts only after the step in flight."""
    rig = Rig(text_quiet_seconds=0)
    rig.server.hold["prefill:x"] = pause = Pause()
    rig.turn()
    await rig.idle_for(5)
    await pause.reached.wait()
    calls_before = len(rig.llm.calls)

    async def chat() -> str:
        out = ""
        async with aclosing(rig.gateway.stream("chat", [])) as chunks:
            async for chunk in chunks:
                out += chunk.text or ""
        return out

    task = asyncio.create_task(chat())
    await asyncio.sleep(0.01)
    assert len(rig.llm.calls) == calls_before  # not sent yet
    pause.release.set()
    assert await asyncio.wait_for(task, 1) == "hello"


async def test_a_warm_up_step_that_hangs_is_cancelled_for_a_turn() -> None:
    rig = Rig(text_quiet_seconds=0, max_turn_wait=0.05)
    rig.server.hold["prefill:x"] = Pause()  # never released
    rig.turn()
    await rig.idle_for(5)
    await asyncio.sleep(0.01)
    await asyncio.wait_for(rig.residency.before_foreground(), 2)
    assert rig.residency._cycle is None
    assert rig.residency.records[-1].outcome == "cancelled during prefill_2"


async def test_warm_ups_are_not_gateway_requests() -> None:
    """No lane, no fallback, no request listener: a warm-up is invisible to the gateway."""
    rig = Rig(text_quiet_seconds=0)
    rig.turn()
    await rig.idle_for(5)
    await rig.settle()
    assert rig.llm.calls == []
    assert rig.gateway.foreground_active == 0 and rig.gateway.background_running == 0
    assert not rig.residency._chat.used


# --- Failures and shutdown -------------------------------------------------------------------


@pytest.mark.parametrize("step", ["size", "unload", "load", "prefill:", "prefill:x"])
async def test_an_error_at_any_step_is_logged_and_ends_the_cycle(
    step: str, caplog: pytest.LogCaptureFixture
) -> None:
    rig = Rig(text_quiet_seconds=0, prefix=None if step == "load" else PREFIX)
    rig.server.fail[step] = RuntimeError("server down")
    rig.turn()
    with caplog.at_level(logging.WARNING):
        await rig.idle_for(5)
        await rig.settle()
    assert "server down" in caplog.text
    assert rig.residency._cycle is None
    assert await rig.residency.busy_reason() is None
    assert rig.residency.records[-1].outcome.startswith("failed during")
    # Not retried until the model is used again.
    calls = list(rig.server.calls)
    await rig.idle_for(5)
    await rig.settle()
    assert rig.server.calls == calls


async def test_a_step_that_times_out_ends_the_cycle() -> None:
    rig = Rig(text_quiet_seconds=0, step_timeout=0.05)
    rig.server.hold["unload"] = Pause()
    rig.turn()
    await rig.idle_for(5)
    await rig.settle()
    assert rig.residency.records[-1].outcome.startswith("failed during unload")
    assert await rig.residency.busy_reason() is None


async def test_aclose_cancels_a_warm_up_and_detaches() -> None:
    rig = Rig(text_quiet_seconds=0)
    rig.server.hold["prefill:"] = pause = Pause()
    rig.turn()
    await rig.idle_for(5)
    await pause.reached.wait()
    task = rig.residency._cycle.task  # type: ignore[union-attr]
    await asyncio.wait_for(rig.residency.aclose(), 1)
    assert task is not None and task.cancelled()
    assert rig.gateway._gate is None
    rig.used()
    await rig.idle_for(5)
    assert rig.server.calls.count("unload") == 1


async def test_run_follows_the_bus_and_ticks() -> None:
    rig = Rig(text_quiet_seconds=0, idle_seconds=0.1, poll_interval=0.01, clock=time.monotonic)
    task = asyncio.create_task(rig.residency.run())
    await rig.bus.publish(UserMessage(session_id="s", text="hi"))
    await asyncio.sleep(0.02)
    assert await rig.residency.busy_reason() == "turn"
    rig.used()
    await rig.bus.publish(TurnEnded(session_id="s", reason="completed"))
    for _ in range(100):
        if rig.residency.records:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await rig.residency.aclose()
    assert "unload" in rig.server.calls


# --- Capability and shared models --------------------------------------------------------------


def test_adapters_without_the_capability_take_no_part() -> None:
    plain = FakeLLM([])
    gateway = LLMGateway({"chat": plain})
    assert resident(plain) is None
    assert not Residency.supported(gateway)
    with pytest.raises(ValueError):
        Residency(bus=EventBus(), gateway=gateway)


def test_declaring_without_implementing_is_not_enough() -> None:
    llm = FakeLLM([])
    llm.capabilities = llm.capabilities.model_copy(update={"residency": True})
    assert resident(llm) is None


async def test_roles_sharing_a_model_share_one_state() -> None:
    server = ModelServer()
    chat = ResidentFake(server, [reply("a")], key="srv|gemma")
    extraction = ResidentFake(server, [reply("b")], key="srv|gemma")  # other options
    other = ResidentFake(ModelServer(size=3 * GB), [reply("c")], key="srv|small")
    plain = FakeLLM([reply("d")])
    llms = {"chat": chat, "extraction": extraction, "summary": other, "embed_like": plain}
    rig = Rig(server, llms=llms, text_quiet_seconds=0)
    assert sorted(rig.residency.models) == ["srv|gemma", "srv|small"]
    rig.used("extraction")  # background work on the shared model only
    await rig.idle_for(5)
    await rig.settle()
    assert server.calls.count("unload") == 1
    assert other.server.calls == []  # not used: left alone
    assert rig.residency.records[-1].unloaded == ["srv|gemma"]


async def test_a_used_second_model_is_unloaded_but_not_rewarmed() -> None:
    server, small = ModelServer(), ModelServer(size=3 * GB)
    chat = ResidentFake(server, [reply("a")], key="srv|gemma")
    other = ResidentFake(small, [reply("c")], key="srv|small")
    rig = Rig(server, llms={"chat": chat, "summary": other}, text_quiet_seconds=0)
    rig.turn()
    rig.used("summary")
    await rig.idle_for(5)
    await rig.settle()
    assert rig.residency.records[-1].unloaded == ["srv|gemma", "srv|small"]
    assert small.calls == ["size", "unload"]
    assert "prefill:x" in server.calls


# --- Voice --------------------------------------------------------------------------------------


async def test_speech_start_loads_an_unloaded_model() -> None:
    rig = Rig(prefix=None)
    rig.voice("idle")
    rig.turn("v")
    await rig.idle_for(5)
    await rig.settle()
    rig.server.calls.clear()
    rig.server.size = None  # say the re-warm did not happen
    rig.residency._chat.state = "unloaded"
    rig.voice("speaking")
    await rig.settle()
    assert rig.server.calls == ["load", "size"]
    assert rig.residency.records[-1].reason == "speech"


async def test_speech_start_leaves_a_loaded_model_alone() -> None:
    rig = Rig()
    rig.voice("idle")
    rig.used()
    rig.voice("speaking")
    await rig.settle()
    assert rig.server.calls == []


async def test_speech_start_loads_a_model_last_used_long_ago() -> None:
    rig = Rig()
    rig.voice("idle")
    rig.used()
    rig.clock.now += 600  # beyond the server's keep-alive
    rig.voice("speaking")
    await rig.settle()
    assert rig.server.calls[0] == "load"


async def test_speech_during_a_warm_up_starts_no_second_cycle() -> None:
    rig = Rig()
    rig.server.hold["size"] = pause = Pause()
    rig.voice("idle")
    rig.turn("v")
    await rig.idle_for(5)
    await pause.reached.wait()
    cycle = rig.residency._cycle
    rig.voice("speaking")
    assert rig.residency._cycle is cycle
    pause.release.set()
    await rig.settle()


# --- The prefix the re-warm uses ------------------------------------------------------------


class Memory(InMemoryMemoryStore):
    def __init__(self, context: MemoryContext | None = None) -> None:
        super().__init__()
        self.context = context or MemoryContext()

    async def retrieve(self, query: str, budget: int) -> MemoryContext:
        return self.context


async def test_next_turn_prefix_matches_the_next_turn_after_a_tool_turn() -> None:
    """The re-warm prefix equals what the next real turn sends, up to its user message,
    including the shortened tool results of the turn before."""
    long_result = "Title line\n" + "word " * 200
    call = ToolCall(id="c1", name="echo", arguments={"text": long_result})
    llm = FakeLLM(
        [
            [LLMChunk(tool_call=call), LLMChunk(finish_reason=FinishReason.TOOL_CALLS)],
            reply("Found it."),
            reply("Sure."),
        ]
    )
    memory = Memory(MemoryContext(profile="Lives in Pune.", instructions=["Be brief."]))
    bus = EventBus()
    context = ContextBuilder("You are Lucy.", layout="stable_prefix", clock=None)
    orchestrator = Orchestrator(
        bus=bus,
        gateway=LLMGateway({"chat": llm}),
        memory=memory,
        context=context,
        tools=[EchoTool()],
    )
    task = asyncio.create_task(orchestrator.run())
    try:
        async with bus.subscribe(TurnEnded) as ends:
            await bus.publish(UserMessage(session_id="s", text="look it up"))
            await anext(ends)
            prefix = await orchestrator.next_turn_prefix("s")
            await bus.publish(UserMessage(session_id="s", text="thanks"))
            await anext(ends)
    finally:
        bus.close()
        await task
    assert prefix is not None
    sent = llm.calls[-1]
    assert sent[: len(prefix.messages)] == prefix.messages
    assert sent[len(prefix.messages)] == Message(role=Role.USER, content="thanks")
    shortened = [m for m in prefix.messages if m.role is Role.TOOL]
    assert len(shortened) == 1 and len(shortened[0].content) < len(long_result)
    assert prefix.tools == [t for t in llm.tools_seen[-1]]


async def test_next_turn_prefix_is_none_for_the_system_layout_or_no_history() -> None:
    memory = Memory()
    orchestrator = Orchestrator(
        bus=EventBus(),
        gateway=LLMGateway({"chat": FakeLLM([])}),
        memory=memory,
        context=ContextBuilder("p", layout="system"),
    )
    assert await orchestrator.next_turn_prefix("s") is None
    stable = Orchestrator(
        bus=EventBus(),
        gateway=LLMGateway({"chat": FakeLLM([])}),
        memory=memory,
        context=ContextBuilder("p", layout="stable_prefix"),
    )
    assert await stable.next_turn_prefix("s") is None


async def test_next_turn_prefix_keeps_to_the_episode_window() -> None:
    """With more stored events than the window, the prefix drops the same oldest one
    the next turn will drop once its own message is stored."""
    llm = FakeLLM([reply("one"), reply("two"), reply("three")])
    memory = Memory()
    bus = EventBus()
    orchestrator = Orchestrator(
        bus=bus,
        gateway=LLMGateway({"chat": llm}),
        memory=memory,
        context=ContextBuilder("p", layout="stable_prefix"),
        history_limit=5,
    )
    task = asyncio.create_task(orchestrator.run())
    try:
        async with bus.subscribe(TurnEnded) as ends:
            for text in ("a", "b"):
                await bus.publish(UserMessage(session_id="s", text=text))
                await anext(ends)
            prefix = await orchestrator.next_turn_prefix("s")
            await bus.publish(UserMessage(session_id="s", text="c"))
            await anext(ends)
    finally:
        bus.close()
        await task
    assert prefix is not None
    sent = llm.calls[-1]
    assert sent[: len(prefix.messages)] == prefix.messages
    assert sent[len(prefix.messages)] == Message(role=Role.USER, content="c")


async def test_a_turn_message_after_the_unload_skips_the_rewarm() -> None:
    rig = Rig(text_quiet_seconds=0)
    rig.server.hold["unload"] = pause = Pause()
    rig.turn()
    await rig.idle_for(5)
    await pause.reached.wait()
    rig.residency.handle(UserMessage(session_id="s", text="next"))  # its request will load it
    pause.release.set()
    await rig.settle()
    assert rig.server.calls == ["size", "unload"]
    assert rig.residency._chat.state == "unloaded"


async def test_the_text_quiet_period_backs_off_after_a_collision_and_resets() -> None:
    rig = Rig(idle_seconds=1.0, text_quiet_seconds=10.0)
    rig.server.hold["prefill:"] = pause = Pause()
    rig.turn()
    await rig.idle_for(11)
    await pause.reached.wait()
    rig.residency.handle(UserMessage(session_id="s", text="quick reply"))  # mid re-warm
    await asyncio.wait_for(rig.residency.before_foreground(), 1)
    assert rig.residency.text_quiet_seconds == 20.0
    rig.residency.handle(TurnEnded(session_id="s", reason="completed"))
    rig.used()
    await rig.idle_for(15)  # past the setting, short of the backed-off period
    assert rig.server.calls.count("unload") == 1
    for _ in range(3):  # more collisions: capped at four times the setting
        rig.residency._adapt_text_quiet(interrupted=True)
    assert rig.residency.text_quiet_seconds == 40.0
    del rig.server.hold["prefill:"]
    rig.server.size = 12 * GB  # the turn's own request loaded it again
    rig.residency._text_quiet_now = 20.0
    rig.clock.now += 10
    await rig.residency.tick()
    await rig.settle()
    assert rig.server.calls.count("unload") == 2
    assert rig.residency.text_quiet_seconds == 10.0  # undisturbed: back to the setting


async def test_voice_cycles_do_not_change_the_text_quiet_period() -> None:
    rig = Rig(text_quiet_seconds=10.0)
    rig.server.hold["prefill:"] = pause = Pause()
    rig.voice("idle")
    rig.turn("v")
    await rig.idle_for(5)
    await pause.reached.wait()
    rig.residency.handle(UserMessage(session_id="v", text="hi"))
    await asyncio.wait_for(rig.residency.before_foreground(), 1)
    assert rig.residency.text_quiet_seconds == 10.0
