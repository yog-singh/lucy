"""Reminders spoken by the voice clients (ADR 0020): `VoiceSession.announce`, the wake
listener around it, and the delivery task in `lucy voice` and `lucy listen`."""

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from lucy.clients import cli
from lucy.clients.voice_cli import run_listen, run_voice
from lucy.core.config import AdapterConfig, LLMSettings, Settings, ToolsSettings, WakeSettings
from lucy.core.context import ContextBuilder
from lucy.core.events import (
    ConfirmationResolved,
    PlaybackInterrupted,
    TimerFired,
    TurnEnded,
    UserMessage,
)
from lucy.core.gateway import LLMGateway
from lucy.core.orchestrator import Orchestrator
from lucy.core.registry import Registry
from lucy.core.reminders import REMINDER_TIMER, ReminderKind, ReminderService, TimerReply
from lucy.core.types import AudioChunk, FinishReason, LLMChunk, ToolCall
from lucy.memory.sqlite_reminders import SQLiteReminderStore
from lucy.memory.sqlite_store import SQLiteMemoryStore
from lucy.tools.calendar_changes import DeleteCalendarEventTool
from lucy.tools.calendar_events import event_id
from lucy.voice.audio import FileAudioIO, Frame, write_wav
from lucy.voice.session import VoiceConfig, VoiceSession, VoiceState, deliver_reminders
from lucy.voice.vad import ScriptedVAD
from lucy.voice.wake import ListenerConfig, ListenerState, WakeListener, WakeStage, chime
from lucy.voice.wakeword import ScriptedWakeWord
from tests.fakes import FakeLLM, InMemoryMemoryStore, Pause, adapter_for
from tests.test_proactive import utc_announcement
from tests.test_tools_calendar import EVENTS, LONDON, NOW, FakeStore
from tests.test_voice_session import TIMEOUT, Harness, reply, until
from tests.test_voice_wake import loud, speech, wake_phrase
from tests.voice_fakes import FakeSTT, FakeTTS, concat, silence, tone

REMINDER = "Reminder: call the bank."  # 24 characters: 1.2 s of fake speech


class Reminders:
    """A "reminder" timer handler and a `fire` that both count what happens."""

    def __init__(self, text: str = REMINDER, *, due: int = 1) -> None:
        self.text = text
        self.due = due  # how many fires return an event
        self.fired_at: list[float] = []
        self.delivered = 0
        self.hold: asyncio.Event | None = None
        self.io: Any = None

    async def fire(self, session_id: str) -> TimerFired | None:
        if self.io is not None:
            self.fired_at.append(self.io.now())
        if self.due <= 0:
            return None
        self.due -= 1
        return TimerFired(session_id=session_id, name=REMINDER_TIMER, payload={})

    async def handle(self, event: TimerFired, turn_id: str) -> TimerReply:
        if self.hold is not None:
            await self.hold.wait()

        async def delivered() -> None:
            self.delivered += 1

        return TimerReply(text=self.text, delivered=delivered)


def harness(
    audio: Sequence[AudioChunk],
    scripts: Sequence[Any],
    transcripts: Sequence[str],
    reminders: Reminders,
    *,
    tools: Sequence[Any] = (),
    memory: InMemoryMemoryStore | None = None,
    handle: Any = None,
    slow: bool = False,
) -> Harness:
    h = Harness(audio, [], transcripts, orchestrate=False)
    if slow:
        h.io = SlowFileAudio(audio)
        h.session = VoiceSession(bus=h.bus, session_id="cli", io=h.io, vad=h.vad, stt=h.stt,
                                 tts=h.tts, ui=h.ui)  # fmt: skip
    h.llm = FakeLLM(scripts)
    if memory is not None:
        h.memory = memory
    h.orchestrator = Orchestrator(
        bus=h.bus,
        gateway=LLMGateway({"chat": h.llm}),
        memory=h.memory,
        context=ContextBuilder("You are Lucy."),
        tools=tools,
        timer_handlers={REMINDER_TIMER: handle or reminders.handle},
    )
    reminders.io = h.io
    return h


async def at(h: Harness, seconds: float) -> None:
    await until(lambda: h.io.now() >= seconds)


# --- VoiceSession.announce ------------------------------------------------------------


async def test_a_reminder_is_spoken_while_frames_are_dropped() -> None:
    # Announced at 1.0 s, the reminder plays for 1.2 s; speech meanwhile is not heard.
    audio = [concat(silence(1.3), tone(0.6), silence(3.0))]
    r = Reminders()
    async with harness(audio, [], [], r) as h:
        run = h.start()
        await at(h, 1.0)
        assert await h.session.announce(r.fire)
        assert h.session.state == VoiceState.LISTENING
        async with asyncio.timeout(TIMEOUT):
            await run
    assert h.tts.started == [REMINDER]  # through the normal speech path
    assert h.io.played_samples == h.tts.samples_for(REMINDER)
    assert h.stt.calls == [] and h.events(UserMessage) == []
    assert h.ui.replies[0] == REMINDER and h.ui.ends[0] == ("completed", None)
    assert VoiceState.THINKING in h.ui.states and VoiceState.SPEAKING in h.ui.states
    assert h.llm.calls == [] and r.delivered == 1
    assert h.session.timings == []  # proactive turns stay out of the latency records


async def test_announce_waits_out_an_utterance_and_its_reply() -> None:
    audio = [concat(silence(0.5), tone(1.0), silence(6.0))]
    r = Reminders()
    async with harness(audio, [reply("Sure thing.")], ["call me"], r) as h:
        run = h.start()
        await until(lambda: h.session.hearing_speech)
        assert await h.session.announce(r.fire)
        async with asyncio.timeout(TIMEOUT):
            await run
    (user_turn,) = h.session.timings
    assert r.fired_at[0] >= user_turn.playback_end  # claimed only once Lucy was idle
    assert h.tts.started == ["Sure thing.", REMINDER]
    assert h.ui.replies[:2] == ["Sure thing.", REMINDER]


async def test_a_user_turn_waiting_keeps_its_events_and_the_reminder_its_own() -> None:
    pause = Pause()
    audio = [concat(silence(0.5), tone(1.0), silence(6.0))]
    r = Reminders()
    async with harness(audio, [[pause, *reply("Your answer.")]], ["question"], r) as h:
        run = h.start()
        await until(lambda: pause.reached.is_set())
        announcing = asyncio.create_task(h.session.announce(r.fire))
        await asyncio.sleep(0.01)
        assert r.fired_at == []  # held behind the turn in progress
        pause.release.set()
        async with asyncio.timeout(TIMEOUT):
            assert await announcing
            await run
    user, timer = h.events(UserMessage)[0], [e for e in h.memory.episodes
                                              if isinstance(e, TimerFired)][0]  # fmt: skip
    assert user.turn_id != timer.turn_id
    assert h.ui.replies[:2] == ["Your answer.", REMINDER]
    assert h.session.timings[0].turn_id == user.turn_id


async def test_a_foreign_turn_queued_behind_the_reminder_is_not_given_its_events() -> None:
    # The reminder's turn is registered before it is published: a text turn on the same
    # session queued right behind it is neither spoken nor attached to the reminder.
    audio = [concat(silence(8.0))]
    r = Reminders()
    r.hold = asyncio.Event()
    async with harness(audio, [reply("Typed answer.")], [], r) as h:
        run = h.start()
        await at(h, 0.5)
        announcing = asyncio.create_task(h.session.announce(r.fire))
        await until(lambda: bool(r.fired_at))
        await asyncio.sleep(0)
        await h.bus.publish(UserMessage(session_id="cli", text="typed", source="text"))
        r.hold.set()
        async with asyncio.timeout(TIMEOUT):
            assert await announcing
            await until(lambda: len([e for e in h.memory.episodes
                                     if isinstance(e, TurnEnded)]) == 2)  # fmt: skip
            h.session.stop()
            await run
    assert h.tts.started == [REMINDER]
    assert h.ui.replies[0] == REMINDER


async def test_barge_in_on_a_reminder_records_what_was_heard() -> None:
    long = "Reminder: call the bank about the mortgage before they close."
    audio = [concat(silence(8.0))]
    r = Reminders(long)
    async with harness(audio, [], [], r) as h:
        run = h.start()
        await at(h, 0.5)
        announcing = asyncio.create_task(h.session.announce(r.fire))
        await until(lambda: h.io.played_samples >= h.tts.samples_for(long) // 3)
        assert h.session.state == VoiceState.SPEAKING
        assert h.session.interrupt()
        async with asyncio.timeout(TIMEOUT):
            assert await announcing
        assert h.session.state == VoiceState.LISTENING
        async with asyncio.timeout(TIMEOUT):
            await run
    (record,) = h.events(PlaybackInterrupted)
    timer = next(e for e in h.memory.episodes if isinstance(e, TimerFired))
    assert record.turn_id == timer.turn_id
    assert record.spoken_text and long.startswith(record.spoken_text.rstrip("."))
    assert r.delivered == 1  # marked once the words were logged, before playback


async def test_fire_returning_none_leaves_the_session_untouched() -> None:
    audio = [concat(silence(1.0), tone(1.0), silence(3.0))]
    r = Reminders(due=0)
    async with harness(audio, [reply("Heard you.")], ["hello"], r) as h:
        run = h.start()
        await at(h, 0.5)
        states = list(h.ui.states)
        assert not await h.session.announce(r.fire)
        assert h.ui.states == states and h.session.state == VoiceState.LISTENING
        async with asyncio.timeout(TIMEOUT):
            await run
    assert len(r.fired_at) == 1 and h.tts.started == ["Heard you."]
    assert [u.text for u in h.events(UserMessage)] == ["hello"]


async def test_a_pending_spoken_confirmation_is_not_disturbed(tmp_path: Path) -> None:
    clock = lambda: NOW  # noqa: E731
    dentist = EVENTS[0]
    store = FakeStore(EVENTS)
    tool = DeleteCalendarEventTool(store=store, clock=clock, local_zone=LONDON)
    call = ToolCall(id="c1", name=tool.name, arguments={"id": event_id(dentist)})
    script = [[LLMChunk(tool_call=call), LLMChunk(finish_reason=FinishReason.TOOL_CALLS)]]
    audio = [concat(silence(0.5), tone(1.0), silence(12.0), tone(0.5), silence(40.0))]
    memory = InMemoryMemoryStore()
    async with SQLiteReminderStore(tmp_path / "lucy.db", clock=clock) as reminders:
        await reminders.add(ReminderKind.REMINDER, "call the bank", NOW - timedelta(seconds=1))
        start = time.monotonic()

        def ticking() -> datetime:  # the service's backoff needs time to pass
            return NOW + timedelta(seconds=time.monotonic() - start)

        service = ReminderService(reminders, memory, utc_announcement, clock=ticking, poll=0.001)
        r = Reminders()
        h = harness(audio, [*script, reply("Done.")], ["Delete the dentist.", "Yes."], r,
                    tools=[tool], memory=memory, handle=service.handle, slow=True)  # fmt: skip
        async with h:
            run = h.start()
            await until(lambda: bool(h.tts.started))  # the question is being asked
            delivery = asyncio.create_task(deliver_reminders(h.session, service))
            try:
                await until(lambda: any("call the bank" in t for t in h.tts.started))
            finally:
                delivery.cancel()
                h.session.stop()
                await asyncio.gather(delivery, run, return_exceptions=True)
    assert [e.outcome for e in h.events(ConfirmationResolved)] == ["confirmed"]
    assert dentist not in store.stored
    assert h.tts.started[1:] == ["Done.", "Reminder: call the bank."]


# --- The wake listener ------------------------------------------------------------------


class Listening:
    """A `WakeListener` over a session built by `make_session`, with a reminder."""

    def __init__(self, audio: Sequence[AudioChunk], *, followup_s: float = 3.0,
                 chime_on: bool = True, transcripts: Sequence[str] = (),
                 replies: Sequence[str] = ()) -> None:  # fmt: skip
        self.r = Reminders()
        self.h = harness(audio, [reply(t) for t in replies], transcripts, self.r)
        self.detector = ScriptedWakeWord(score=loud)
        self.built = 0
        self.listener = WakeListener(
            io=self.h.io,
            stage=WakeStage(self.detector),
            make_session=self.make_session,
            ui=self.h.ui,
            config=ListenerConfig(followup_s=followup_s, chime=chime_on),
        )
        self.announced: asyncio.Task[bool] | None = None
        self.announce_on_build = False

    async def make_session(self, io: Any, ui: Any) -> VoiceSession:
        self.built += 1
        h = self.h
        session = VoiceSession(bus=h.bus, session_id="cli", io=io, vad=h.vad, stt=h.stt,
                               tts=h.tts, config=VoiceConfig(), ui=ui)  # fmt: skip
        h.session = session
        if self.announce_on_build:
            self.announce()
        return session

    def announce(self) -> None:
        assert self.listener.session is not None or self.h.session is not None
        self.announced = asyncio.create_task(
            self.h.session.announce(self.r.fire, cue=self.listener.cue)
        )

    def chimes(self) -> list[float]:
        length = chime((660.0, 990.0)).frames
        return [p.start for p in self.h.io.plays if p.samples == length]


async def test_a_reminder_while_asleep_cannot_wake_lucy_and_opens_the_follow_up() -> None:
    audio = [concat(
        silence(0.5), wake_phrase(), speech(0.8), silence(6.0),  # a turn, then asleep at ~5.6
        silence(2.0),  # the reminder is announced at 8.0 and spoken for 1.2 s
        silence(0.3), wake_phrase(0.6), silence(2.0),  # loud sound meanwhile: no wake
        speech(0.8), silence(6.0),  # answered without the wake phrase; then asleep again
    )]  # fmt: skip
    t = Listening(audio, transcripts=["hey jarvis hi", "thanks"], replies=["Hello.", "Okay."])
    async with t.h:
        run = asyncio.create_task(t.listener.run())
        await until(lambda: len(t.listener.sleeps) == 1)
        await at(t.h, 8.0)
        assert t.listener.state == ListenerState.WAITING
        t.announce()
        async with asyncio.timeout(TIMEOUT):
            assert await t.announced
            await run
    assert len(t.listener.wakes) == 1  # never woke on the loud sound during the reminder
    assert t.h.tts.started == ["Hello.", REMINDER, "Okay."]
    assert [u.text for u in t.h.events(UserMessage)] == ["hey jarvis hi", "thanks"]
    assert len(t.listener.sleeps) == 2  # the follow-up window closed with the timeout
    wake, sleep, cue, sleep2 = t.chimes()
    assert cue >= 8.0 and t.h.io.plays[3].start >= cue  # the chime, then the reminder
    assert sleep2 - cue > 3.0


async def test_no_chime_before_a_reminder_when_chimes_are_off() -> None:
    audio = [concat(silence(1.0), silence(8.0))]
    t = Listening(audio, chime_on=False)
    t.listener.rouse()  # builds the session at once, still waiting
    async with t.h:
        run = asyncio.create_task(t.listener.run())
        await until(lambda: t.h.session.state == VoiceState.LISTENING and t.built == 1)
        await at(t.h, 1.0)
        t.announce()
        async with asyncio.timeout(TIMEOUT):
            assert await t.announced
            await run
    assert t.chimes() == []
    assert t.h.tts.started == [REMINDER]


async def test_rouse_before_the_first_wake_speaks_the_reminder() -> None:
    audio = [concat(silence(1.0), silence(8.0))]
    t = Listening(audio)
    t.announce_on_build = True
    async with t.h:
        run = asyncio.create_task(t.listener.run())
        await at(t.h, 0.5)
        assert t.built == 0
        t.listener.rouse()
        async with asyncio.timeout(TIMEOUT):
            await until(lambda: t.announced is not None)
            assert await t.announced
            await run
    assert t.built == 1 and t.listener.wakes == []
    assert t.h.tts.started == [REMINDER]
    assert len(t.chimes()) == 2  # the cue before the reminder, then the window closes
    assert len(t.listener.sleeps) == 1


# --- The clients ------------------------------------------------------------------------


class SlowFileAudio(FileAudioIO):
    """`FileAudioIO` that gives real time between frames, so background tasks that touch
    the database keep pace with the simulated input."""

    async def frames(self) -> AsyncIterator[Frame]:  # type: ignore[override]
        async for frame in super().frames():
            await asyncio.sleep(0.002)
            yield frame


def client_settings(tmp_path: Path, *, reminders: bool, **wake: object) -> Settings:
    return Settings(
        llm=LLMSettings(roles={"chat": AdapterConfig(adapter="fake")}),
        memory=AdapterConfig(adapter="sqlite", options={"path": str(tmp_path / "lucy.db")}),
        stt=AdapterConfig(adapter="fake"),
        tts=AdapterConfig(adapter="fake"),
        tools=ToolsSettings(enabled=["set_reminder"] if reminders else []),
        wake=WakeSettings(**wake),  # type: ignore[arg-type]
    )


def client_registry(tts: FakeTTS) -> Registry:
    registry = Registry()
    registry.register("llm", "fake")(adapter_for(FakeLLM([])))
    registry.register("memory", "sqlite")(SQLiteMemoryStore)
    registry.register("stt", "fake")(adapter_for(FakeSTT([])))
    registry.register("tts", "fake")(adapter_for(tts))
    return registry


async def add_missed_reminder(tmp_path: Path) -> None:
    async with SQLiteReminderStore(tmp_path / "lucy.db") as store:
        due = datetime.now(UTC) - timedelta(minutes=5)
        await store.add(ReminderKind.REMINDER, "call the bank", due)


def wav(tmp_path: Path, seconds: float) -> Path:
    path = tmp_path / "in.wav"
    audio = silence(seconds)
    write_wav(path, audio.data, audio.sample_rate)
    return path


@pytest.mark.parametrize("enabled", [True, False])
async def test_lucy_voice_speaks_a_missed_reminder_at_start(tmp_path: Path, enabled: bool) -> None:
    await add_missed_reminder(tmp_path)
    tts = FakeTTS()
    path = wav(tmp_path, 3.0)
    args = cli._parser().parse_args(["voice", "--input-wav", str(path), "--fast"])
    tasks_before = asyncio.all_tasks()
    code = await run_voice(
        client_settings(tmp_path, reminders=enabled), args, out=RecordingOut(), err=RecordingOut(),
        registry=client_registry(tts), make_vad=lambda: ScriptedVAD(level=0.1),
        make_audio=lambda: SlowFileAudio([path]),
    )  # fmt: skip
    assert code == 0
    spoken = [t for t in tts.started if "call the bank" in t]
    assert len(spoken) == (1 if enabled else 0)
    if enabled:
        assert spoken[0].startswith("A reminder was due at")  # the late wording
    leftover = {t for t in asyncio.all_tasks() - tasks_before if "reminders" in t.get_name()}
    assert not leftover  # the delivery task was cancelled on exit


async def test_lucy_listen_rouses_for_a_reminder_due_before_the_first_wake(
    tmp_path: Path,
) -> None:
    await add_missed_reminder(tmp_path)
    tts = FakeTTS()
    path = wav(tmp_path, 4.0)
    args = cli._parser().parse_args(["listen", "--input-wav", str(path), "--fast"])
    code = await run_listen(
        client_settings(tmp_path, reminders=True, followup_seconds=1), args,
        out=RecordingOut(), err=RecordingOut(), registry=client_registry(tts),
        make_vad=lambda: ScriptedVAD(level=0.1), make_audio=lambda: SlowFileAudio([path]),
        make_detector=lambda: ScriptedWakeWord(score=loud),
    )  # fmt: skip
    assert code == 0
    assert [t for t in tts.started if "call the bank" in t]


class RecordingOut:
    def __init__(self) -> None:
        self.text = ""

    def write(self, text: str) -> int:
        self.text += text
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False
