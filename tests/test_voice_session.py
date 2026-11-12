"""VoiceSession end to end with the real orchestrator, a FakeLLM, fake STT/TTS, a
level-based fake VAD and FileAudioIO on simulated time."""

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from lucy.core.bus import EventBus
from lucy.core.context import ContextBuilder
from lucy.core.events import (
    AssistantMessage,
    Event,
    Interrupt,
    PlaybackInterrupted,
    SpeechActivity,
    TextDelta,
    TurnEnded,
    UserMessage,
)
from lucy.core.gateway import LLMGateway
from lucy.core.history import events_to_messages
from lucy.core.orchestrator import Orchestrator
from lucy.core.types import AudioChunk, FinishReason, LLMChunk, Role, ToolCall
from lucy.voice.audio import FileAudioIO
from lucy.voice.session import NullUI, VoiceConfig, VoiceSession, VoiceState
from lucy.voice.vad import ScriptedVAD
from tests.fakes import EchoTool, FakeLLM, InMemoryMemoryStore, Pause, Raise, Script
from tests.voice_fakes import FakeSTT, FakeTTS, concat, silence, tone

TIMEOUT = 10


def reply(*texts: str) -> list[LLMChunk]:
    return [LLMChunk(text=t) for t in texts] + [LLMChunk(finish_reason=FinishReason.STOP)]


class RecordingUI(NullUI):
    def __init__(self) -> None:
        self.states: list[VoiceState] = []
        self.users: list[str] = []
        self.replies: list[str] = [""]
        self.ends: list[tuple[str, str | None]] = []
        self.notices: list[str] = []

    def state(self, state: VoiceState) -> None:
        if not self.states or self.states[-1] != state:
            self.states.append(state)

    def user(self, text: str) -> None:
        self.users.append(text)

    def reply_delta(self, text: str) -> None:
        self.replies[-1] += text

    def reply_end(self, outcome: str, error: str | None) -> None:
        self.ends.append((outcome, error))
        self.replies.append("")

    def notice(self, text: str) -> None:
        self.notices.append(text)


class Harness:
    def __init__(
        self,
        audio: Sequence[AudioChunk],
        scripts: Sequence[Script],
        transcripts: Sequence[str | Exception],
        *,
        tts: FakeTTS | None = None,
        config: VoiceConfig | None = None,
        timings_path: Path | None = None,
        tools: Sequence[Any] = (),
        orchestrate: bool = True,
    ) -> None:
        self.bus = EventBus()
        self.llm = FakeLLM(scripts)
        self.memory = InMemoryMemoryStore()
        self.orchestrator = (
            Orchestrator(
                bus=self.bus,
                gateway=LLMGateway({"chat": self.llm}),
                memory=self.memory,
                context=ContextBuilder("You are Lucy."),
                tools=tools,
            )
            if orchestrate
            else None
        )
        self.io = FileAudioIO(audio)
        self.vad = ScriptedVAD(level=0.1)
        self.stt = FakeSTT(transcripts)
        self.tts = tts or FakeTTS()
        self.ui = RecordingUI()
        self.session = VoiceSession(
            bus=self.bus,
            session_id="cli",
            io=self.io,
            vad=self.vad,
            stt=self.stt,
            tts=self.tts,
            config=config or VoiceConfig(),
            ui=self.ui,
            timings_path=timings_path,
        )
        self._orchestrator_task: asyncio.Task[None] | None = None
        self.run_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "Harness":
        if self.orchestrator is not None:
            self._orchestrator_task = asyncio.create_task(self.orchestrator.run())
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self.run_task is not None and not self.run_task.done():
            self.session.stop()
            await asyncio.gather(self.run_task, return_exceptions=True)
        self.bus.close()
        if self._orchestrator_task is not None:
            await self._orchestrator_task

    def start(self) -> asyncio.Task[None]:
        self.run_task = asyncio.create_task(self.session.run())
        return self.run_task

    async def run(self) -> None:
        async with asyncio.timeout(TIMEOUT):
            await self.start()

    def events(self, kind: type[Event]) -> list[Any]:
        return [e for e in self.memory.episodes if isinstance(e, kind)]


async def until(condition, timeout: float = TIMEOUT) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0)


def drain(subscription) -> list[Event]:
    events = []
    while not subscription._queue.empty():
        item = subscription._queue.get_nowait()
        if isinstance(item, Event):
            events.append(item)
    return events


# --- A whole turn -----------------------------------------------------------------


async def test_a_full_turn_end_to_end(tmp_path: Path) -> None:
    timings_file = tmp_path / "voice_timings.jsonl"
    audio = [concat(silence(0.5), tone(1.0), silence(3.0))]
    async with Harness(
        audio,
        [reply("Hello there. ", "How are you today?")],
        ["hi lucy"],
        timings_path=timings_file,
    ) as h:
        await h.run()
    users = h.events(UserMessage)
    assert [(u.text, u.source) for u in users] == [("hi lucy", "voice")]
    assert h.tts.started == ["Hello there.", "How are you today?"]
    expected = h.tts.samples_for("Hello there.") + h.tts.samples_for("How are you today?")
    assert h.io.played_samples == expected
    assert h.ui.users == ["hi lucy"]
    assert h.ui.replies[0] == "Hello there. How are you today?"
    assert h.ui.ends[0] == ("completed", None)
    assert h.ui.states == [
        VoiceState.LISTENING,
        VoiceState.TRANSCRIBING,
        VoiceState.THINKING,
        VoiceState.SPEAKING,
        VoiceState.LISTENING,
        VoiceState.STOPPED,
    ]
    # The utterance reached STT with its pre-roll and post-roll.
    assert h.stt.calls[0].duration == pytest.approx(1.0 + 0.3 + 0.2, abs=0.07)
    (t,) = h.session.timings
    assert t.outcome == "completed" and t.transcript == "hi lucy"
    moments = [
        t.speech_end,
        t.endpoint,
        t.transcript_ready,
        t.first_text_delta,
        t.first_sentence,
        t.first_tts_audio,
        t.first_audio_played,
        t.playback_end,
    ]
    assert None not in moments and moments == sorted(moments)
    assert t.speech_end == pytest.approx(1.5, abs=0.04)
    assert t.stages()["endpointing"] == pytest.approx(0.608, abs=0.001)
    assert t.reply_audio_s == pytest.approx(expected / 24000)
    assert t.turn_id == users[0].turn_id
    (line,) = timings_file.read_text().splitlines()
    record = json.loads(line)
    assert record["outcome"] == "completed"
    assert record["stages"]["total"] == pytest.approx(0.608, abs=0.001)
    assert record["since_speech_end"]["first_audio_played"] == pytest.approx(0.608, abs=0.001)
    assert record["utterance_s"] == pytest.approx(1.5, abs=0.07)


async def test_two_turns_continue_one_conversation() -> None:
    audio = [concat(silence(0.5), tone(1.0), silence(4.0), tone(0.8), silence(3.0))]
    async with Harness(audio, [reply("First."), reply("Second.")], ["one", "two"]) as h:
        await h.run()
    assert [u.text for u in h.events(UserMessage)] == ["one", "two"]
    second_call = h.llm.calls[1]
    assert [(m.role, m.content) for m in second_call[1:]] == [
        (Role.USER, "one"),
        (Role.ASSISTANT, "First."),
        (Role.USER, "two"),
    ]
    assert len({t.turn_id for t in h.session.timings}) == 2


# --- Half duplex ----------------------------------------------------------------------


async def test_speech_while_lucy_speaks_and_during_the_tail_is_ignored() -> None:
    # Lucy answers from about 2.1 s for 3 s (60 characters at 0.05 s each), then a
    # 1 s tail. Speech at 3.0 s (while she speaks) and 5.3 s (in the tail) is ignored;
    # speech at 7.5 s is answered.
    long_reply = "This reply is exactly long enough to fill three seconds now."
    assert len(long_reply) == 60
    audio = [
        concat(
            silence(0.5),
            tone(1.0),  # 0.5 - 1.5
            silence(1.5),
            tone(0.8),  # 3.0 - 3.8, while Lucy speaks
            silence(1.5),
            tone(0.6),  # 5.3 - 5.9, in the tail
            silence(1.6),
            tone(1.0),  # 7.5 - 8.5
            silence(2.0),
        )
    ]
    config = VoiceConfig(tail_ms=1000)
    async with Harness(
        audio, [reply(long_reply), reply("Yes.")], ["first", "last"], config=config
    ) as h:
        await h.run()
    assert len(h.stt.calls) == 2
    assert [u.text for u in h.events(UserMessage)] == ["first", "last"]
    first, last = h.session.timings
    assert first.first_audio_played == pytest.approx(2.11, abs=0.05)
    assert first.playback_end == pytest.approx(first.first_audio_played + 3.0, abs=0.01)
    assert last.speech_start == pytest.approx(7.5, abs=0.05)
    assert h.vad.resets >= 2  # reset after each reply


# --- Interrupting ---------------------------------------------------------------------


async def test_interrupt_while_speaking_records_what_was_heard() -> None:
    sentences = ["Hello there.", "The quick brown fox jumps over the lazy dog.", "Never heard."]
    audio = [concat(silence(0.3), tone(1.0), silence(8.0), tone(1.0), silence(2.0))]
    async with Harness(
        audio, [reply(*(s + " " for s in sentences)), reply("Okay.")], ["tell me", "again"]
    ) as h:
        run = h.start()
        first = h.tts.samples_for(sentences[0])
        second = h.tts.samples_for(sentences[1])
        await until(lambda: h.io.played_samples >= first + second // 2)
        assert h.session.state == VoiceState.SPEAKING
        heard_before = h.io.played_samples
        assert h.session.interrupt()
        assert not h.session.interrupt()  # once only
        await until(lambda: h.session.state == VoiceState.LISTENING)
        async with asyncio.timeout(TIMEOUT):
            await run
    assert h.io.plays[0].stopped
    assert h.io.plays[0].samples == pytest.approx(heard_before, abs=240)
    (record,) = h.events(PlaybackInterrupted)
    assert record.spoken_text.startswith("Hello there. The quick brown")
    assert "lazy dog" not in record.spoken_text and "Never" not in record.spoken_text
    assert record.turn_id == h.events(UserMessage)[0].turn_id
    # The generation had already finished: no Interrupt was needed.
    assert not [e for e in h.memory.episodes if isinstance(e, Interrupt)]
    assert h.session.timings[0].outcome == "interrupted"
    assert h.session.timings[0].spoken_chars == len(record.spoken_text)
    # The next turn's history holds only what was spoken.
    history = h.llm.calls[1]
    assistant = [m for m in history if m.role == Role.ASSISTANT]
    assert [m.content for m in assistant] == [record.spoken_text]
    assert [u.text for u in h.events(UserMessage)] == ["tell me", "again"]


async def test_interrupt_while_thinking_before_any_text() -> None:
    pause = Pause()
    audio = [concat(silence(0.3), tone(1.0), silence(3.0), tone(1.0), silence(2.0))]
    async with Harness(
        audio, [[pause, *reply("Too late.")], reply("Fine.")], ["question", "again"]
    ) as h:
        bus_events = h.bus.subscribe(Interrupt, PlaybackInterrupted)
        run = h.start()
        await until(lambda: pause.reached.is_set())
        assert h.session.state == VoiceState.THINKING
        assert h.session.interrupt()
        async with asyncio.timeout(TIMEOUT):
            await run
    published = drain(bus_events)
    assert [type(e) for e in published] == [Interrupt]
    ended = h.events(TurnEnded)
    assert ended[0].reason == "interrupted"
    assert not h.events(PlaybackInterrupted)
    assert h.llm.exits[0] in ("cancelled", "closed")
    assert h.ui.ends[0] == ("interrupted", None)
    assert h.tts.started == ["Fine."]
    assert h.session.timings[0].outcome == "interrupted"
    assert h.session.timings[1].outcome == "completed"


async def test_interrupt_while_thinking_after_some_text_drops_the_reply() -> None:
    pause = Pause()
    audio = [concat(silence(0.3), tone(1.0), silence(3.0), tone(1.0), silence(2.0))]
    script = [LLMChunk(text="I think the"), pause, *reply(" answer is no.")]
    async with Harness(audio, [script, reply("Fine.")], ["question", "again"]) as h:
        run = h.start()
        await until(lambda: pause.reached.is_set())
        await until(lambda: h.ui.replies[0] == "I think the")
        assert h.session.interrupt()
        async with asyncio.timeout(TIMEOUT):
            await run
    (record,) = h.events(PlaybackInterrupted)
    assert record.spoken_text == ""
    interrupted = [e for e in h.events(AssistantMessage) if e.interrupted]
    assert interrupted[0].text == "I think the"  # what was generated...
    history = events_to_messages(
        h.memory.episodes[: h.memory.episodes.index(h.events(UserMessage)[1])]
    )
    assert [m.role for m in history] == [Role.USER]  # ...but nothing was heard


async def test_interrupt_while_speaking_and_still_generating() -> None:
    pause = Pause()
    audio = [concat(silence(0.3), tone(1.0), silence(4.0))]
    script = [LLMChunk(text="First sentence. Second"), pause, *reply(" sentence.")]
    async with Harness(audio, [script], ["question"]) as h:
        bus_events = h.bus.subscribe(Interrupt, PlaybackInterrupted)
        run = h.start()
        await until(lambda: h.io.played_samples >= h.tts.samples_for("First sentence."))
        for _ in range(20):  # the first sentence is over; playback waits for more
            await asyncio.sleep(0)
        assert h.io.played_samples == h.tts.samples_for("First sentence.")
        assert h.session.state == VoiceState.SPEAKING
        assert h.session.interrupt()
        async with asyncio.timeout(TIMEOUT):
            await run
    published = drain(bus_events)
    assert [type(e) for e in published] == [Interrupt, PlaybackInterrupted]
    assert published[1].spoken_text == "First sentence."
    history = events_to_messages(h.memory.episodes)
    assert [(m.role, m.content) for m in history] == [
        (Role.USER, "question"),
        (Role.ASSISTANT, "First sentence."),
    ]


async def test_spoken_text_counts_only_the_final_step() -> None:
    call = ToolCall(id="c1", name="echo", arguments={"text": "42"})
    step1 = [LLMChunk(text="Let me check. "), LLMChunk(tool_call=call)]
    step1.append(LLMChunk(finish_reason=FinishReason.TOOL_CALLS))
    step2 = reply("The answer is forty two. ", "It comes from the tool and is final.")
    audio = [concat(silence(0.3), tone(1.0), silence(6.0))]
    async with Harness(audio, [step1, step2], ["what is it"], tools=[EchoTool()]) as h:
        run = h.start()
        before = h.tts.samples_for("Let me check.") + h.tts.samples_for("The answer is forty two.")
        await until(lambda: h.io.played_samples >= before + 2400)
        h.session.interrupt()
        async with asyncio.timeout(TIMEOUT):
            await run
    (record,) = h.events(PlaybackInterrupted)
    assert record.spoken_text.startswith("The answer is forty two.")
    assert "Let me check" not in record.spoken_text
    history = events_to_messages(h.memory.episodes)
    assistant = [m for m in history if m.role == Role.ASSISTANT]
    assert assistant[0].content.strip() == "Let me check." and assistant[0].tool_calls
    assert assistant[1].content == record.spoken_text


async def test_interrupt_is_ignored_while_listening() -> None:
    async with Harness([silence(0.5)], [], []) as h:
        assert not h.session.interrupt()
        await h.run()
    assert not h.events(Interrupt)


async def test_late_events_of_an_interrupted_turn_are_handled() -> None:
    """The orchestrator ignores an Interrupt for a turn that has not started; the
    session sends it again once the turn shows up, and records that nothing was heard."""
    audio = [concat(silence(0.3), tone(1.0), silence(2.0))]
    async with Harness(audio, [], ["question"], orchestrate=False) as h:
        inbound = h.bus.subscribe(UserMessage, Interrupt, PlaybackInterrupted)
        run = h.start()
        async with asyncio.timeout(TIMEOUT):
            message = await anext(inbound)
        assert isinstance(message, UserMessage)
        assert h.session.interrupt()
        async with asyncio.timeout(TIMEOUT):
            first = await anext(inbound)
        assert isinstance(first, Interrupt) and first.turn_id is None
        # The turn starts late (as if the Interrupt had come too early).
        ids = {"session_id": "cli", "turn_id": "t1"}
        await h.bus.publish(TextDelta(**ids, text="Late text"))
        async with asyncio.timeout(TIMEOUT):
            second = await anext(inbound)
        assert isinstance(second, Interrupt) and second.turn_id == "t1"
        await h.bus.publish(TurnEnded(**ids, reason="interrupted"))
        async with asyncio.timeout(TIMEOUT):
            record = await anext(inbound)
        assert isinstance(record, PlaybackInterrupted)
        assert (record.turn_id, record.spoken_text) == ("t1", "")
        async with asyncio.timeout(TIMEOUT):
            await run


async def test_events_of_other_sessions_and_turns_are_ignored() -> None:
    audio = [concat(silence(0.3), tone(1.0), silence(2.0))]
    async with Harness(audio, [], ["question"], orchestrate=False) as h:
        inbound = h.bus.subscribe(UserMessage)
        run = h.start()
        async with asyncio.timeout(TIMEOUT):
            await anext(inbound)
        await h.bus.publish(TextDelta(session_id="other", turn_id="x", text="Not mine."))
        ids = {"session_id": "cli", "turn_id": "t1"}
        await h.bus.publish(TextDelta(**ids, text="Mine."))
        await h.bus.publish(TurnEnded(**ids, reason="completed"))
        async with asyncio.timeout(TIMEOUT):
            await run
    assert h.ui.replies[0] == "Mine."


# --- Ignoring and recovering ----------------------------------------------------------


async def test_an_empty_transcript_is_ignored() -> None:
    audio = [concat(silence(0.3), tone(1.0), silence(2.0))]
    async with Harness(audio, [], [""]) as h:
        await h.run()
    assert not h.events(UserMessage)
    assert h.llm.calls == []
    assert h.session.timings[0].outcome == "no_speech"
    assert h.ui.users == []


async def test_noise_too_short_never_reaches_stt() -> None:
    audio = [concat(silence(0.3), tone(0.2), silence(2.0))]
    async with Harness(audio, [], []) as h:
        await h.run()
    assert h.stt.calls == [] and h.session.timings == []


async def test_stt_failure_is_reported_and_listening_continues() -> None:
    audio = [concat(silence(0.3), tone(1.0), silence(2.0), tone(1.0), silence(2.0))]
    async with Harness(audio, [reply("Fine.")], [RuntimeError("model crashed"), "hello"]) as h:
        await h.run()
    assert any("could not transcribe" in n and "model crashed" in n for n in h.ui.notices)
    assert [u.text for u in h.events(UserMessage)] == ["hello"]
    assert [t.outcome for t in h.session.timings] == ["stt_error", "completed"]


async def test_llm_failure_is_reported_and_listening_continues() -> None:
    audio = [concat(silence(0.3), tone(1.0), silence(3.0), tone(1.0), silence(2.0))]
    failing = [LLMChunk(text="Half a sentence. Then"), Raise(RuntimeError("server gone"))]
    async with Harness(audio, [failing, reply("Back.")], ["one", "two"]) as h:
        await h.run()
    assert h.ui.ends[0][0] == "error" and "server gone" in (h.ui.ends[0][1] or "")
    # What had been generated before the failure is still spoken.
    assert h.tts.started[0] == "Half a sentence."
    assert [t.outcome for t in h.session.timings] == ["error", "completed"]


async def test_tts_failure_is_reported_and_the_text_still_shown() -> None:
    audio = [concat(silence(0.3), tone(1.0), silence(3.0))]
    tts = FakeTTS(fail_on="Broken")
    async with Harness(audio, [reply("Broken sentence. ", "Fine one.")], ["q"], tts=tts) as h:
        await h.run()
    assert any("could not speak" in n for n in h.ui.notices)
    assert h.ui.replies[0] == "Broken sentence. Fine one."
    assert h.io.played_samples == tts.samples_for("Fine one.")
    assert h.session.timings[0].outcome == "completed"


async def test_speech_at_the_end_of_input_is_still_answered() -> None:
    audio = [concat(silence(0.3), tone(1.0))]  # no trailing silence: flushed at the end
    async with Harness(audio, [reply("Got it.")], ["cut off"]) as h:
        await h.run()
    assert [u.text for u in h.events(UserMessage)] == ["cut off"]
    assert h.tts.finished == ["Got it."]


async def test_stop_while_speaking_ends_promptly_and_records_what_was_heard() -> None:
    audio = [concat(silence(0.3), tone(1.0), silence(20.0))]
    text = "One two three four five six seven eight nine ten eleven twelve."
    async with Harness(audio, [reply(text)], ["count"]) as h:
        run = h.start()
        await until(lambda: h.io.played_samples > 12000)
        h.session.stop()
        async with asyncio.timeout(TIMEOUT):
            await run
        await asyncio.sleep(0)
        h.bus.close()
        await h._orchestrator_task
    (record,) = h.events(PlaybackInterrupted)
    assert record.spoken_text.startswith("One two") and not record.spoken_text.endswith("twelve.")
    assert h.session.state == VoiceState.STOPPED


async def test_closing_the_bus_ends_the_session() -> None:
    pause = Pause()
    audio = [concat(silence(0.3), tone(1.0), silence(30.0))]
    async with Harness(audio, [[pause, *reply("x")]], ["q"]) as h:
        run = h.start()
        await until(lambda: pause.reached.is_set())
        h.bus.close()
        async with asyncio.timeout(TIMEOUT):
            await run
    assert h.ui.ends[-1] == ("error", "Lucy stopped")


async def test_run_only_once() -> None:
    async with Harness([silence(0.2)], [], []) as h:
        await h.run()
        with pytest.raises(RuntimeError):
            await h.session.run()


async def test_stt_is_prepared_when_speech_starts() -> None:
    audio = [concat(silence(0.3), tone(1.0), silence(3.0))]
    async with Harness(audio, [reply("Hi.")], ["hello"]) as h:
        await h.run()
    assert h.stt.prepared == 1


# --- Speech activity --------------------------------------------------------------------


async def test_speech_activity_follows_the_listener_and_is_not_stored() -> None:
    audio = [concat(silence(0.3), tone(1.0), silence(2.0), tone(0.2), silence(2.0))]
    async with Harness(audio, [reply("Hi.")], ["hello"]) as h:
        seen = h.bus.subscribe(SpeechActivity, UserMessage)
        await h.run()
    states = [e.state if isinstance(e, SpeechActivity) else "USER" for e in drain(seen)]
    # The short sound after the turn starts speech, then is discarded.
    assert states == [
        "idle",
        "speaking",
        "transcribing",
        "USER",
        "idle",
        "speaking",
        "idle",
        "ended",
    ]
    assert not [e for e in h.memory.episodes if e.type == "speech_activity"]


async def test_speech_activity_returns_to_idle_after_an_empty_transcript() -> None:
    audio = [concat(silence(0.3), tone(1.0), silence(2.0))]
    async with Harness(audio, [], [""]) as h:
        seen = h.bus.subscribe(SpeechActivity)
        await h.run()
    assert [e.state for e in drain(seen)] == ["idle", "speaking", "transcribing", "idle", "ended"]
