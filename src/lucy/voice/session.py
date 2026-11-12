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

Timings: each turn's moments (speech end, endpoint, transcript, first text,
first sentence, first TTS audio, first audio played, playback end) are taken on
the `AudioIO` clock, logged as one line, and appended as one JSON line to
`timings_path`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import Callable
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
    ToolStarted,
    TurnEnded,
    UserMessage,
)
from lucy.core.ports import STT, TTS
from lucy.voice.audio import AudioIO, Frame
from lucy.voice.endpoint import Discarded, Endpointer, SpeechStarted, UtteranceComplete
from lucy.voice.playback import SpeechPlayer
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
    ) -> None:
        self._bus = bus
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
        self.timings: list[TurnTimings] = []

    # --- Public surface ------------------------------------------------------------

    @property
    def state(self) -> VoiceState:
        return self._state

    @property
    def session_id(self) -> str:
        return self._session_id

    async def run(self) -> None:
        """Listen and converse until the input ends or `stop` is called."""
        if self._subscription is not None:
            raise RuntimeError("VoiceSession.run can only be called once")
        self._subscription = self._bus.subscribe(
            TextDelta, SentenceReady, AssistantMessage, ToolStarted, TurnEnded
        )
        pump = asyncio.create_task(self._pump(self._subscription), name="lucy-voice-events")
        self._main = asyncio.create_task(self._listen(), name="lucy-voice-listen")
        self._set_state(VoiceState.LISTENING)
        await self._speech_activity("idle")
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

        Returns whether there was anything to interrupt.
        """
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
        if frame.time <= self._gate_until:
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
        replied = False
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
            self._ui.user(text)
            replied = True
            await self._converse(text, timings)
        finally:
            await self._speech_activity("idle")
            self._record(timings)
            self._listen_again(gate=replied)

    async def _converse(self, text: str, timings: TurnTimings) -> None:
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
            await self._bus.publish(
                UserMessage(session_id=self._session_id, text=text, source="voice")
            )
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
