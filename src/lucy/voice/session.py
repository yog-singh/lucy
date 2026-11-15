"""The half-duplex voice loop (stage 1 of docs/voice.md, ADR 0014 and 0016).

`VoiceSession` listens through an `AudioIO`, runs each frame through the VAD and
the `Endpointer`, transcribes a finished utterance with the STT port, publishes
it as `UserMessage(source="voice")` and speaks the turn's `SentenceReady` events
through a `SpeechPlayer` until the turn has ended and everything queued has
been played. Then it listens again.

Half-duplex: frames are read only while listening. Frames captured while
transcribing, thinking or speaking, and for `tail_ms` after playback ends, are
discarded unheard when listening resumes (by capture time), and the VAD and
endpointer are reset, so Lucy never hears herself.

Interrupting (`interrupt`, bound to a key press): playback stops at once, the
session returns to listening, and for the interrupted turn it publishes
`Interrupt` (if the turn is still generating) and `PlaybackInterrupted` with
the text actually heard (if the turn produced any text). Spoken text covers the
turn's final LLM step only, because rebuilding history replaces only that
step's message (`events_to_messages`); speech from an earlier step that ended
in tool calls is not counted. If the turn had not produced its first event yet,
the session keeps watching it and sends `Interrupt` again once it starts (the
orchestrator ignores an `Interrupt` for a turn that has not started) and
records `PlaybackInterrupted` with empty text if it produced any.

Turn matching: one subscription serves the whole session. Turns of a session
run in order, so the n-th `UserMessage` published belongs to the n-th new turn
id seen; this keeps late events of an abandoned turn away from the next one.

Speech activity: the session publishes `SpeechActivity` (transient, never
stored) as the listener changes: `speaking` when speech starts, `transcribing`
when an utterance is complete, `idle` once the transcript has been handed over
as a `UserMessage` (or turned out empty, failed, or the sound was discarded),
and `ended` when the session stops. The model residency policy uses it to keep
the model in memory while the user speaks.

Proactive turns (ADR 0020): `announce` speaks a turn Lucy starts herself, such
as a due reminder. It takes the session's floor (held by every utterance and
its turn), waits until the session is idle (listening, the user not speaking,
nothing being transcribed), only then calls `fire` to claim what is due, and
publishes the resulting `TimerFired` through the same path as a `UserMessage`:
the turn is registered as expected before publishing, so its events are never
given to a user turn, and its sentences are spoken through the same player,
with the same barge-in. Frames are dropped while announcing (half-duplex), and
the session listens again afterwards as after a reply. `deliver_reminders`
runs that in a loop for a client.

Voice enrolment (ADR 0021, amendment): when a turn's `enroll_voice` or
`replace_voice` call succeeds (seen as its `ToolStarted` and `ToolFinished`), the
session, still holding the floor, leads an enrolment once the turn has been
spoken (`_enrol`): it shows each sentence, says a short cue, and reads the
next utterance itself as a clip, never as a `UserMessage`, so no model turn,
no speaker tag and nothing for memory. "cancel" or "stop", Enter, a long
silence or too many bad readings end it with nothing enrolled. The outcome is
published as a `TimerFired` (`voice_enrolment`), so the orchestrator says and
logs it as a proactive turn. `enrolling` tells the wake listener to stay awake.

Timings: each turn's moments (speech end, endpoint, transcript, first text,
first sentence, first TTS audio, first audio played, playback end) are taken on
the `AudioIO` clock, logged as one line, and appended as one JSON line to
`timings_path`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from lucy.core.bus import EventBus, Subscription
from lucy.core.config import VoiceSettings
from lucy.core.events import (
    AssistantMessage,
    Event,
    Interrupt,
    PlaybackInterrupted,
    SentenceReady,
    SpeechActivity,
    TextDelta,
    TimerFired,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    UserMessage,
)
from lucy.core.ports import STT, TTS
from lucy.tools.voice_enrolment import ENROL_TOOLS, ENROLMENT_TIMER
from lucy.voice.audio import AudioIO, Frame
from lucy.voice.endpoint import Discarded, Endpointer, SpeechStarted, UtteranceComplete
from lucy.voice.enrol import (
    MIN_OVERLAP,
    MIN_SPEECH_S,
    ORDINALS,
    Enroller,
    EnrolmentRequest,
    is_cancel,
    overlap,
)
from lucy.voice.playback import SpeechPlayer
from lucy.voice.speakers import SpeakerTagger, await_tag
from lucy.voice.vad import VAD

logger = logging.getLogger(__name__)

TIMINGS_FILE_NAME: Final = "voice_timings.jsonl"


class VoiceState(StrEnum):
    """What the session is doing."""

    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    STOPPED = "stopped"
    # Only shown by `lucy listen` (`lucy.voice.wake`), never set by the session:
    WAITING = "waiting"  # for the wake phrase
    STARTING = "starting"  # loading the speech parts after the first wake


class VoiceUI(Protocol):
    """What the session tells its client (the terminal, a test)."""

    def state(self, state: VoiceState) -> None: ...

    def user(self, text: str) -> None:
        """The user's utterance, transcribed."""
        ...

    def reply_delta(self, text: str) -> None:
        """A piece of Lucy's reply text as it streams."""
        ...

    def reply_end(self, outcome: str, error: str | None) -> None:
        """The reply is over: "completed", "interrupted" or "error"."""
        ...

    def notice(self, text: str) -> None:
        """Something worth a line on screen (a tool in use, a failure)."""
        ...


class NullUI:
    """A `VoiceUI` that shows nothing."""

    def state(self, state: VoiceState) -> None:
        pass

    def user(self, text: str) -> None:
        pass

    def reply_delta(self, text: str) -> None:
        pass

    def reply_end(self, outcome: str, error: str | None) -> None:
        pass

    def notice(self, text: str) -> None:
        pass


@dataclass(frozen=True)
class VoiceConfig:
    """Listening parameters; see `lucy.core.config.VoiceSettings` for their meaning."""

    endpoint_silence_ms: int = 600
    min_speech_ms: int = 160
    preroll_ms: int = 300
    max_utterance_s: float = 30.0
    tail_ms: int = 300
    vad_threshold: float = 0.5
    min_utterance_ms: int = 250

    @classmethod
    def from_settings(cls, settings: VoiceSettings) -> VoiceConfig:
        return cls(
            endpoint_silence_ms=settings.endpoint_silence_ms,
            min_speech_ms=settings.min_speech_ms,
            preroll_ms=settings.preroll_ms,
            max_utterance_s=settings.max_utterance_s,
            tail_ms=settings.tail_ms,
            vad_threshold=settings.vad_threshold,
        )

    def endpointer(self) -> Endpointer:
        return Endpointer(
            threshold=self.vad_threshold,
            min_speech_ms=self.min_speech_ms,
            endpoint_silence_ms=self.endpoint_silence_ms,
            preroll_ms=self.preroll_ms,
            max_utterance_s=self.max_utterance_s,
            min_utterance_ms=self.min_utterance_ms,
        )


@dataclass
class TurnTimings:
    """When each step of one voice turn happened (seconds on the `AudioIO` clock).

    `outcome`: "completed", "interrupted", "error" (the turn failed), "no_speech"
    (empty transcript), "stt_error".
    """

    session_id: str
    at: str  # wall-clock time of the endpoint, ISO 8601 UTC
    speech_start: float
    speech_end: float
    endpoint: float
    utterance_s: float
    forced_endpoint: bool = False
    turn_id: str | None = None
    outcome: str = "completed"
    transcript: str = ""
    transcript_ready: float | None = None
    first_text_delta: float | None = None
    first_sentence: float | None = None
    first_tts_audio: float | None = None
    first_audio_played: float | None = None
    playback_end: float | None = None
    reply_chars: int = 0
    reply_audio_s: float = 0.0
    spoken_chars: int | None = None
    error: str | None = None
    speaker: str | None = None
    speaker_ready: float | None = None  # when the speaker tag was taken, if in time

    def stages(self) -> dict[str, float | None]:
        """Durations of each stage, and end of speech to first audio (`total`)."""

        def span(a: float | None, b: float | None) -> float | None:
            return None if a is None or b is None else round(b - a, 3)

        return {
            "endpointing": span(self.speech_end, self.endpoint),
            "stt": span(self.endpoint, self.transcript_ready),
            "llm_first_text": span(self.transcript_ready, self.first_text_delta),
            "llm_first_sentence": span(self.transcript_ready, self.first_sentence),
            "tts_first_audio": span(self.first_sentence, self.first_tts_audio),
            "output": span(self.first_tts_audio, self.first_audio_played),
            "total": span(self.speech_end, self.first_audio_played),
        }

    def to_json(self) -> dict[str, Any]:
        """The record for the timings file: stage durations, plus every moment relative
        to the end of speech."""
        data = asdict(self)
        moments = (
            "speech_start",
            "endpoint",
            "transcript_ready",
            "speaker_ready",
            "first_text_delta",
            "first_sentence",
            "first_tts_audio",
            "first_audio_played",
            "playback_end",
        )
        relative = {}
        for name in moments:
            value = data.pop(name)
            relative[name] = None if value is None else round(value - self.speech_end, 3)
        data.pop("speech_end")
        data["since_speech_end"] = relative
        data["stages"] = self.stages()
        data["utterance_s"] = round(self.utterance_s, 3)
        data["reply_audio_s"] = round(self.reply_audio_s, 3)
        return data

    def summary(self) -> str:
        """One line for the log."""
        stages = self.stages()

        def fmt(value: float | None) -> str:
            return "-" if value is None else f"{value:.2f}"

        return (
            f"voice turn {self.outcome}: end of speech to first audio {fmt(stages['total'])}s"
            f" (endpoint {fmt(stages['endpointing'])}, stt {fmt(stages['stt'])},"
            f" llm sentence {fmt(stages['llm_first_sentence'])},"
            f" tts {fmt(stages['tts_first_audio'])}, output {fmt(stages['output'])});"
            f" utterance {self.utterance_s:.1f}s, reply {self.reply_audio_s:.1f}s"
        )


class _Signal(StrEnum):
    INTERRUPTED = "interrupted"
    CLOSED = "closed"


@dataclass(eq=False)
class _Turn:
    """One published `UserMessage` and what has happened to its turn."""

    events: asyncio.Queue[Event | _Signal] = field(default_factory=asyncio.Queue)
    turn_id: str | None = None
    ended: TurnEnded | None = None
    text_seen: bool = False
    boundary: int = 0  # sentences before this belong to earlier LLM steps
    interrupted: bool = False  # the user stopped it; late events go to `_late`
    spoken: str = ""
    interrupt_sent: bool = False
    recorded: bool = False
    enrol_calls: dict[str, tuple[str, str]] = field(default_factory=dict)  # id -> (tool, name)
    enrolment: EnrolmentRequest | None = None


class VoiceSession:
    """One voice conversation on `session_id` (see the module docstring).

    `run` listens until the input ends (files) or `stop` is called, finishing
    the turn in progress first when the input ends. `interrupt` is safe to call
    at any time from the event loop.
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        session_id: str,
        io: AudioIO,
        vad: VAD,
        stt: STT,
        tts: TTS,
        config: VoiceConfig | None = None,
        ui: VoiceUI | None = None,
        timings_path: str | Path | None = None,
        lookahead: int = 1,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        speakers: SpeakerTagger | None = None,
        enroller: Enroller | None = None,
    ) -> None:
        self._bus = bus
        self._speakers = speakers
        self._enroller = enroller
        self._frames: Any = None  # the input's frame iterator, while `run` reads it
        self._enrolling = False
        self._enrol_cancel = False
        self._enrol_player: SpeechPlayer | None = None
        self._session_id = session_id
        self._io = io
        self._vad = vad
        self._stt = stt
        self._tts = tts
        self._config = config or VoiceConfig()
        self._ui: VoiceUI = ui if ui is not None else NullUI()
        self._timings_path = Path(timings_path) if timings_path is not None else None
        self._lookahead = lookahead
        self._wall_clock = wall_clock
        self._endpointer = self._config.endpointer()
        self._state = VoiceState.LISTENING
        self._gate_until = float("-inf")
        self._turn: _Turn | None = None
        self._player: SpeechPlayer | None = None
        self._unassigned: deque[_Turn] = deque()
        self._turns: dict[str, _Turn] = {}
        self._finished_ids: set[str] = set()
        self._subscription: Subscription | None = None
        self._main: asyncio.Task[None] | None = None
        self._stopping = False
        self._bus_closed = False
        self._background: set[asyncio.Task[None]] = set()
        self._activity: str | None = None
        self._floor = asyncio.Lock()  # held by each utterance and each announcement
        self._changed = asyncio.Event()  # set when the state or the speech activity changes
        self._announcing = False
        self._running = asyncio.Event()
        self.timings: list[TurnTimings] = []

    # --- Public surface ------------------------------------------------------------

    @property
    def state(self) -> VoiceState:
        return self._state

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def hearing_speech(self) -> bool:
        """Whether the user is speaking now (an utterance has started and not ended)."""
        return self._activity == "speaking"

    @property
    def enrolling(self) -> bool:
        """Whether a voice enrolment is in progress (the wake listener stays awake)."""
        return self._enrolling

    @property
    def idle(self) -> bool:
        """Listening, with the user not speaking and nothing being transcribed."""
        return (
            self._state == VoiceState.LISTENING
            and not self.hearing_speech
            and self._activity != "transcribing"
        )

    async def announce(
        self,
        fire: Callable[[str], Awaitable[TimerFired | None]],
        *,
        cue: Callable[[], Awaitable[object]] | None = None,
    ) -> bool:
        """Speak a proactive turn (see the module docstring); returns whether one was said.

        Waits for the floor and for the session to be idle, then calls
        `fire(session_id)`. None from `fire` (nothing due, or a confirmation
        question open) leaves the session as it was. `cue`, if given, is awaited
        after a successful `fire` and before the turn is published (the wake
        listener's chime)."""
        await self._running.wait()  # the turn events are routed only once `run` subscribed
        while True:
            if self._state == VoiceState.STOPPED or self._stopping:
                return False
            async with self._floor:
                if self.idle:
                    self._announcing = True  # from now on frames are dropped
                    try:
                        event = await fire(self._session_id)
                        if event is None:
                            return False
                        self._set_state(VoiceState.THINKING)
                        if cue is not None:
                            try:
                                await cue()
                            except Exception:
                                logger.exception("the announcement cue failed")
                        now = self._io.now()
                        timings = TurnTimings(
                            session_id=self._session_id,
                            at=self._wall_clock().isoformat(),
                            speech_start=now,
                            speech_end=now,
                            endpoint=now,
                            utterance_s=0.0,
                        )
                        try:
                            await self._converse(event, timings)
                        finally:
                            if not self._stopping:
                                self._listen_again(gate=True)
                        logger.info("proactive turn %s: %s", timings.turn_id, timings.outcome)
                        return True
                    finally:
                        self._announcing = False
            self._changed.clear()
            if not self.idle:
                await self._changed.wait()
            else:
                await asyncio.sleep(0)

    def forget_audio(self) -> None:
        """Reset the VAD and the endpointer, dropping any partial utterance (the wake
        listener calls it before handing over audio after a gap)."""
        self._vad.reset()
        self._endpointer.reset()

    async def run(self) -> None:
        """Listen and converse until the input ends or `stop` is called."""
        if self._subscription is not None:
            raise RuntimeError("VoiceSession.run can only be called once")
        self._subscription = self._bus.subscribe(
            TextDelta, SentenceReady, AssistantMessage, ToolStarted, ToolFinished, TurnEnded
        )
        pump = asyncio.create_task(self._pump(self._subscription), name="lucy-voice-events")
        self._main = asyncio.create_task(self._listen(), name="lucy-voice-listen")
        self._set_state(VoiceState.LISTENING)
        await self._speech_activity("idle")
        self._running.set()
        try:
            await self._main
        except asyncio.CancelledError:
            if not self._stopping or not self._main.cancelled():
                raise
        finally:
            await self._speech_activity("ended")
            if self._player is not None:
                self._player.stop()
            self._subscription.close()
            pump.cancel()
            await asyncio.gather(pump, *self._background, return_exceptions=True)
            self._set_state(VoiceState.STOPPED)

    def interrupt(self) -> bool:
        """Stop Lucy if she is thinking or speaking (see the module docstring).

        Returns whether there was anything to interrupt. During a voice enrolment
        it cancels the enrolment.
        """
        if self._enrolling and self._turn is None:
            self._enrol_cancel = True
            if self._enrol_player is not None:
                self._enrol_player.stop()
            return True
        turn, player = self._turn, self._player
        if turn is None or turn.interrupted:
            return False
        if self._state not in (VoiceState.THINKING, VoiceState.SPEAKING):
            return False
        turn.interrupted = True
        if player is not None:
            player.stop()
            turn.spoken = player.spoken_text(turn.boundary)
        turn.events.put_nowait(_Signal.INTERRUPTED)
        return True

    def stop(self) -> None:
        """End the session promptly: stop speech (recording what was heard) and return
        from `run`."""
        self._stopping = True
        self.interrupt()
        if self._player is not None:
            self._player.stop()
        if self._main is not None and not self._main.done():
            self._main.cancel()

    # --- Listening -------------------------------------------------------------------

    def _set_state(self, state: VoiceState) -> None:
        if state != self._state or state == VoiceState.LISTENING:
            self._state = state
            self._changed.set()
            try:
                self._ui.state(state)
            except Exception:
                logger.exception("the voice UI failed")

    def _listen_again(self, *, gate: bool) -> None:
        """Back to listening; with `gate`, ignore everything captured until `tail_ms` from now."""
        if gate:
            self._gate_until = self._io.now() + self._config.tail_ms / 1000
        self._vad.reset()
        self._endpointer.reset()
        self._set_state(VoiceState.LISTENING)

    async def _listen(self) -> None:
        frames = self._io.frames()
        self._frames = frames
        try:
            async for frame in frames:
                if self._bus_closed:
                    return
                await self._on_frame(frame)
            for event in self._endpointer.flush(self._io.now()):
                if isinstance(event, UtteranceComplete) and not self._bus_closed:
                    await self._utterance(event)
        finally:
            await frames.aclose()

    async def _on_frame(self, frame: Frame) -> None:
        if self._announcing or frame.time <= self._gate_until:
            return
        probability = self._vad(frame.audio)
        for event in self._endpointer.feed(frame.audio.data, probability, frame.time):
            if isinstance(event, UtteranceComplete):
                await self._utterance(event)
            elif isinstance(event, SpeechStarted):
                logger.debug("speech started at %.2f", event.time)
                await self._speech_activity("speaking")
                self._prepare_stt()
            elif isinstance(event, Discarded):
                logger.debug("ignored a sound: %s", event.reason)
                await self._speech_activity("idle")

    async def _speech_activity(
        self, state: Literal["speaking", "transcribing", "idle", "ended"]
    ) -> None:
        """Publish `SpeechActivity` when the listener's state changes (not while closed)."""
        if state == self._activity or self._bus.closed:
            return
        self._activity = state
        self._changed.set()
        try:
            await self._bus.publish(SpeechActivity(session_id=self._session_id, state=state))
        except RuntimeError:  # the bus has closed: Lucy is stopping
            pass

    def _prepare_stt(self) -> None:
        """Let the STT adapter get ready while the user is still speaking, if it can
        (an optional `prepare()`; see `ParakeetMLXSTT.prepare`)."""
        prepare = getattr(self._stt, "prepare", None)
        if prepare is None:
            return
        task = asyncio.create_task(prepare(), name="lucy-stt-prepare")
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    # --- One turn ----------------------------------------------------------------------

    async def _utterance(self, utterance: UtteranceComplete) -> None:
        async with self._floor:
            await self._utterance_turn(utterance)

    async def _utterance_turn(self, utterance: UtteranceComplete) -> None:
        timings = TurnTimings(
            session_id=self._session_id,
            at=self._wall_clock().isoformat(),
            speech_start=utterance.speech_start,
            speech_end=utterance.speech_end,
            endpoint=self._io.now(),
            utterance_s=utterance.duration,
            forced_endpoint=utterance.forced,
        )
        self._set_state(VoiceState.TRANSCRIBING)
        await self._speech_activity("transcribing")
        replied = recorded = False
        message_id = str(uuid.uuid4())
        tagging = (
            self._speakers.start(utterance.audio, message_id)
            if self._speakers is not None
            else None
        )  # concurrently with transcription; see `lucy.voice.speakers`
        try:
            try:
                transcript = await self._stt.transcribe(utterance.audio)
            except Exception as exc:
                logger.warning("speech recognition failed: %s", exc)
                self._notice(f"could not transcribe that ({type(exc).__name__}: {exc})")
                timings.outcome, timings.error = "stt_error", f"{type(exc).__name__}: {exc}"
                return
            timings.transcript_ready = self._io.now()
            text = transcript.text.strip()
            timings.transcript = text
            if not text:
                timings.outcome = "no_speech"
                return
            tag = None
            if tagging is not None and self._speakers is not None:
                tag = await await_tag(tagging, self._speakers.wait_s)
                timings.speaker_ready = self._io.now() if tag is not None else None
                timings.speaker = tag.speaker if tag is not None else None
            show = getattr(self._ui, "user_speaker", None)
            if tag is not None and tag.speaker is not None and show is not None:
                show(text, tag.speaker)
            else:
                self._ui.user(text)
            replied = True
            message = UserMessage(
                id=message_id,
                session_id=self._session_id,
                text=text,
                source="voice",
                speaker=tag.speaker if tag is not None else None,
                speaker_score=tag.score if tag is not None else None,
                audio_path=tag.audio_path if tag is not None else None,
                audio_s=round(utterance.duration, 3),
            )
            request = await self._converse(message, timings)
            if request is not None:
                self._record(timings)
                recorded = True
                await self._enrol(request)
        finally:
            await self._speech_activity("idle")
            if not recorded:
                self._record(timings)
            self._listen_again(gate=replied)

    async def _converse(
        self, inbound: UserMessage | TimerFired, timings: TurnTimings
    ) -> EnrolmentRequest | None:
        """Publish `inbound` and speak its turn. The turn is registered as expected
        before publishing: turns of a session run in publication order. Returns the
        voice enrolment the turn requested, if it completed with one."""
        turn = _Turn()
        io = self._io

        def first(name: str) -> Callable[[], None]:
            def mark() -> None:
                if getattr(timings, name) is None:
                    setattr(timings, name, io.now())
                if name == "first_audio_played" and not turn.interrupted:
                    self._set_state(VoiceState.SPEAKING)

            return mark

        player = SpeechPlayer(
            self._tts,
            io,
            lookahead=self._lookahead,
            on_first_audio=first("first_tts_audio"),
            on_first_played=first("first_audio_played"),
            on_error=lambda sentence, exc: self._notice(
                f"could not speak a sentence ({type(exc).__name__}: {exc})"
            ),
        )
        self._unassigned.append(turn)
        self._turn, self._player = turn, player
        self._set_state(VoiceState.THINKING)
        reply: list[str] = []
        try:
            await self._bus.publish(inbound)
            await self._speech_activity("idle")
            while not turn.interrupted:
                event = await turn.events.get()
                if event is _Signal.INTERRUPTED:
                    break
                if event is _Signal.CLOSED:
                    timings.outcome, timings.error = "error", "Lucy stopped"
                    self._ui.reply_end("error", "Lucy stopped")
                    break
                if isinstance(event, TextDelta):
                    if timings.first_text_delta is None:
                        timings.first_text_delta = io.now()
                    turn.text_seen = True
                    reply.append(event.text)
                    self._ui.reply_delta(event.text)
                elif isinstance(event, SentenceReady):
                    if timings.first_sentence is None:
                        timings.first_sentence = io.now()
                    player.add(event.text)
                elif isinstance(event, AssistantMessage):
                    if event.tool_calls:
                        turn.boundary = player.sentence_count
                elif isinstance(event, ToolStarted):
                    self._notice(f"[using {event.call.name}]")
                    name = event.call.arguments.get("name")
                    if event.call.name in ENROL_TOOLS and isinstance(name, str):
                        turn.enrol_calls[event.call.id] = (event.call.name, name)
                elif isinstance(event, ToolFinished):
                    call = turn.enrol_calls.get(event.result.call_id)
                    if call is not None and not event.result.is_error:
                        tool, name = call
                        turn.enrolment = EnrolmentRequest(
                            name=_enrolled_name(name), replace=tool == "replace_voice"
                        )
                elif isinstance(event, TurnEnded):
                    timings.outcome = event.reason
                    timings.error = event.error
                    self._ui.reply_end(event.reason, event.error)
                    break
            if not turn.interrupted:
                await player.wait()
        finally:
            timings.turn_id = turn.turn_id
            timings.reply_chars = len("".join(reply).strip())
            timings.reply_audio_s = player.played_seconds
            if timings.first_audio_played is not None:
                timings.playback_end = io.now()
            self._turn = self._player = None
            if turn.interrupted:
                timings.outcome = "interrupted"
                timings.spoken_chars = len(turn.spoken)
                if turn.ended is None:
                    self._ui.reply_end("interrupted", None)
                await self._report_interrupt(turn)
            await player.aclose()
        if turn.interrupted or timings.outcome != "completed":
            return None
        return turn.enrolment

    # --- Voice enrolment -------------------------------------------------------------

    async def _enrol(self, request: EnrolmentRequest) -> None:
        """Lead a voice enrolment (see the module docstring), then have the outcome said."""
        if self._enroller is None or self._frames is None:
            text = "Sorry, I can't record voices here: speaker identification is not running."
        else:
            self._enrolling, self._enrol_cancel = True, False
            try:
                text = await self._enrolment(self._enroller, request)
            except Exception:
                logger.exception("voice enrolment failed")
                text = "Sorry, the voice enrolment failed. Nothing was enrolled."
            finally:
                self._enrolling = False
                self._enrol_player = None
        logger.info("voice enrolment of %s: %s", request.name, text)
        now = self._io.now()
        timings = TurnTimings(
            session_id=self._session_id,
            at=self._wall_clock().isoformat(),
            speech_start=now,
            speech_end=now,
            endpoint=now,
            utterance_s=0.0,
        )
        self._listen_again(gate=True)
        event = TimerFired(
            session_id=self._session_id,
            name=ENROLMENT_TIMER,
            payload={"text": text, "speaker": request.name, "replace": request.replace},
        )
        await self._converse(event, timings)  # like `announce`: no latency record

    async def _enrolment(self, enroller: Enroller, request: EnrolmentRequest) -> str:
        """The enrolment itself; returns the outcome to say."""
        sentences = enroller.pick()
        count = len(sentences)
        stopped = "Nothing was enrolled."
        intro = (
            f"I'll show you {_number(count)} sentences. Read each one aloud after it "
            "appears. Say cancel to stop."
        )
        self._notice(intro)
        if not await self._cue(intro):
            return f"I stopped the voice enrolment. {stopped}"
        clips: list[Any] = []
        for index, sentence in enumerate(sentences):
            ordinal = ORDINALS[index] if index < len(ORDINALS) else f"Sentence {index + 1}"
            cue = f"{ordinal} sentence." if index < len(ORDINALS) else f"{ordinal}."
            failures = 0
            while True:
                if not await self._cue(cue):
                    return f"I stopped the voice enrolment. {stopped}"
                # Shown only once the session is listening (after the cue and the tail
                # drop), so a reading started the moment it appears is not clipped.
                heard = await self._hear(
                    enroller.silence_s,
                    on_listening=lambda i=index, s=sentence: self._show_sentence(i, count, s),
                )
                if heard == "cancelled":
                    return f"I stopped the voice enrolment. {stopped}"
                if heard == "silence":
                    return f"I didn't hear anything, so I stopped the voice enrolment. {stopped}"
                if heard == "ended":
                    return f"The voice enrolment ended early. {stopped}"
                audio, seconds, text = heard
                if is_cancel(text):
                    return f"Okay, I stopped the voice enrolment. {stopped}"
                if seconds < MIN_SPEECH_S:
                    problem = "That was too short."
                elif overlap(sentence, text) < MIN_OVERLAP:
                    problem = "That didn't sound like the sentence."
                else:
                    clips.append(audio)
                    break
                failures += 1
                logger.info("enrolment reading rejected (%s): %r", problem, text)
                if failures > enroller.retries:
                    return f"I couldn't get a clear reading, so I stopped. {stopped}"
                cue = f"{problem} Please read it again."
        self._notice("[saving the voice]")
        self._set_state(VoiceState.THINKING)
        return await enroller.enrol(request.name, clips)

    def _show_sentence(self, index: int, count: int, sentence: str) -> None:
        show = getattr(self._ui, "enrol_sentence", None)
        try:
            if show is not None:
                show(index + 1, count, sentence)
            else:
                self._ui.notice(f"Sentence {index + 1} of {count}, read aloud:  {sentence}")
        except Exception:
            logger.exception("the voice UI failed")

    async def _cue(self, text: str) -> bool:
        """Say a short code-authored cue (not logged); False if the user cancelled."""
        if self._enrol_cancel:
            return False
        player = SpeechPlayer(
            self._tts,
            self._io,
            lookahead=self._lookahead,
            on_error=lambda sentence, exc: self._notice(
                f"could not speak a sentence ({type(exc).__name__}: {exc})"
            ),
        )
        self._enrol_player = player
        self._set_state(VoiceState.SPEAKING)
        try:
            player.add(text)
            await player.wait()
        finally:
            self._enrol_player = None
            await player.aclose()
            self._listen_again(gate=True)
        return not self._enrol_cancel

    async def _hear(
        self, silence_s: float, *, on_listening: Callable[[], None] | None = None
    ) -> tuple[Any, float, str] | Literal["cancelled", "silence", "ended"]:
        """Read frames until one utterance is complete and transcribe it; it never
        becomes a `UserMessage`. `on_listening` is called once, at the first frame
        that is heard (past the post-playback drop), before that frame is used."""
        started: float | None = None
        while True:
            try:
                frame = await anext(self._frames)
            except StopAsyncIteration:
                for event in self._endpointer.flush(self._io.now()):
                    if isinstance(event, UtteranceComplete):
                        return await self._transcribe_reading(event)
                return "ended"
            if self._enrol_cancel:
                return "cancelled"
            if frame.time <= self._gate_until:
                continue
            if started is None:
                started = frame.time
                if on_listening is not None:
                    on_listening()
            probability = self._vad(frame.audio)
            for event in self._endpointer.feed(frame.audio.data, probability, frame.time):
                if isinstance(event, SpeechStarted):
                    await self._speech_activity("speaking")
                    self._prepare_stt()
                elif isinstance(event, Discarded):
                    await self._speech_activity("idle")
                    started = frame.time
                elif isinstance(event, UtteranceComplete):
                    return await self._transcribe_reading(event)
            if not self.hearing_speech and frame.time - started > silence_s:
                return "silence"

    async def _transcribe_reading(self, utterance: UtteranceComplete) -> tuple[Any, float, str]:
        self._set_state(VoiceState.TRANSCRIBING)
        await self._speech_activity("transcribing")
        try:
            text = (await self._stt.transcribe(utterance.audio)).text.strip()
        except Exception as exc:
            logger.warning("speech recognition failed during enrolment: %s", exc)
            text = ""
        await self._speech_activity("idle")
        try:
            self._ui.user(text)
        except Exception:
            logger.exception("the voice UI failed")
        return utterance.audio, utterance.duration, text

    async def _report_interrupt(self, turn: _Turn) -> None:
        """Publish what the interruption of `turn` calls for, as far as is known now; the
        rest follows from its late events (`_late`)."""
        try:
            if turn.ended is None and not turn.interrupt_sent:
                turn.interrupt_sent = turn.turn_id is not None
                await self._bus.publish(
                    Interrupt(session_id=self._session_id, turn_id=turn.turn_id)
                )
            if turn.turn_id is not None and turn.text_seen and not turn.recorded:
                turn.recorded = True
                await self._bus.publish(
                    PlaybackInterrupted(
                        session_id=self._session_id,
                        turn_id=turn.turn_id,
                        spoken_text=turn.spoken,
                    )
                )
        except RuntimeError:  # the bus has closed: Lucy is stopping
            logger.debug("could not report an interruption: the bus is closed")

    # --- Events --------------------------------------------------------------------------

    async def _pump(self, subscription: Subscription) -> None:
        """Route this session's turn events to their turns."""
        try:
            async for event in subscription:
                if event.session_id != self._session_id or event.turn_id is None:
                    continue
                if event.turn_id in self._finished_ids:
                    continue
                turn = self._turns.get(event.turn_id)
                if turn is None:
                    if not self._unassigned:
                        continue  # a turn this session did not start
                    turn = self._unassigned.popleft()
                    turn.turn_id = event.turn_id
                    self._turns[event.turn_id] = turn
                if isinstance(event, TurnEnded):
                    turn.ended = event
                    self._finished_ids.add(event.turn_id)
                    del self._turns[event.turn_id]
                if isinstance(event, TextDelta):
                    turn.text_seen = True
                if turn.interrupted:
                    await self._late(turn, event)
                else:
                    turn.events.put_nowait(event)
        finally:
            self._bus_closed = self._bus.closed
            for turn in [*self._turns.values(), *self._unassigned]:
                if not turn.interrupted:
                    turn.events.put_nowait(_Signal.CLOSED)

    async def _late(self, turn: _Turn, event: Event) -> None:
        """An event of a turn the user already interrupted."""
        try:
            if not isinstance(event, TurnEnded) and not turn.interrupt_sent:
                turn.interrupt_sent = True
                await self._bus.publish(
                    Interrupt(session_id=self._session_id, turn_id=turn.turn_id)
                )
            if isinstance(event, TurnEnded) and turn.text_seen and not turn.recorded:
                turn.recorded = True
                assert turn.turn_id is not None
                await self._bus.publish(
                    PlaybackInterrupted(
                        session_id=self._session_id,
                        turn_id=turn.turn_id,
                        spoken_text=turn.spoken,
                    )
                )
        except RuntimeError:
            logger.debug("could not report an interruption: the bus is closed")

    # --- Output ----------------------------------------------------------------------------

    def _notice(self, text: str) -> None:
        try:
            self._ui.notice(text)
        except Exception:
            logger.exception("the voice UI failed")

    def _record(self, timings: TurnTimings) -> None:
        self.timings.append(timings)
        if timings.outcome != "no_speech":
            logger.info("%s", timings.summary())
        if self._timings_path is None:
            return
        try:
            self._timings_path.parent.mkdir(parents=True, exist_ok=True)
            with self._timings_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(timings.to_json()) + "\n")
        except OSError as exc:
            logger.warning("cannot write voice timings to %s: %s", self._timings_path, exc)


def _enrolled_name(name: str) -> str:
    from lucy.tools.voice_enrolment import clean_name

    try:
        return clean_name(name)
    except ValueError:
        return name


def _number(count: int) -> str:
    names = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight")
    return names[count] if 0 <= count < len(names) else str(count)


class ReminderSource(Protocol):
    """What `deliver_reminders` needs (`lucy.core.reminders.ReminderService`)."""

    poll: float

    async def wait_due(self) -> None: ...

    async def fire(self, session_id: str) -> TimerFired | None: ...


async def deliver_reminders(
    session: VoiceSession,
    service: ReminderSource,
    *,
    cue: Callable[[], Awaitable[object]] | None = None,
) -> None:
    """Speak due reminders in `session` until cancelled, the bus closes or the session
    stops (ADR 0020). No grace before claiming: a voice client should win over chat."""
    while session.state != VoiceState.STOPPED:
        try:
            await service.wait_due()
            await session.announce(service.fire, cue=cue)
        except asyncio.CancelledError:
            raise
        except RuntimeError:  # the bus is closed: Lucy has stopped
            return
        except Exception:
            logger.exception("could not deliver a reminder")
            await asyncio.sleep(service.poll)
