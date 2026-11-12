"""The wake stage and the listener in front of the voice session (`lucy listen`).

Two parts, kept apart so the waiting part can later run on its own as a small
always-on process in front of a server:

- `WakeStage` is all that runs while waiting: frames go through a
  `WakeWordDetector` and into a rolling buffer (`buffer_s`, about 2 s); a score
  at or above `threshold` is a detection, except within `refractory_s` of the
  previous one.
- `WakeListener` reads the microphone, runs the stage while waiting, and on a
  detection wakes the voice session, building it on the first wake only
  (`make_session`, which loads the speech models and the app). The rolling
  buffer is handed to the session first, so a request said in the same breath as
  the wake phrase reaches the endpointer and the recogniser; `WakePhraseSTT`
  then removes the wake phrase from the transcript. After each turn (or after
  the wake phrase alone) the session keeps listening for `followup_s`; when
  that passes with no speech, the listener goes back to waiting.

States: waiting (only capture and the wake-word models run), awake (the
session's own listening, thinking and speaking). Half-duplex is the session's:
while it thinks or speaks it reads no frames, so neither the wake word nor
anything else is heard, and it discards what was captured meanwhile.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import logging
import re
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from lucy.core.types import OUTPUT_SAMPLE_RATE, AudioChunk, Transcript
from lucy.voice.audio import FRAME_SECONDS, AudioIO, Frame
from lucy.voice.session import VoiceState, VoiceUI
from lucy.voice.wakeword import WakeWordDetector

if TYPE_CHECKING:
    from lucy.core.ports import STT
    from lucy.voice.session import VoiceSession

logger = logging.getLogger(__name__)

REFRACTORY_S: Final = 2.0
BUFFER_S: Final = 2.0


# --- The waiting stage --------------------------------------------------------------


class WakeStage:
    """Detection on a stream of frames, with a rolling buffer of the latest audio."""

    def __init__(
        self,
        detector: WakeWordDetector,
        *,
        threshold: float = 0.5,
        refractory_s: float = REFRACTORY_S,
        buffer_s: float = BUFFER_S,
    ) -> None:
        self._detector = detector
        self._threshold = threshold
        self._refractory = refractory_s
        self._buffer: deque[Frame] = deque(maxlen=max(1, round(buffer_s / FRAME_SECONDS)))
        self._last_detection = float("-inf")
        self.last_score = 0.0

    def feed(self, frame: Frame) -> bool:
        """Whether the wake phrase was detected with this frame."""
        self._buffer.append(frame)
        score = self.last_score = self._detector(frame.audio)
        if score < self._threshold or frame.time - self._last_detection < self._refractory:
            return False
        self._last_detection = frame.time
        logger.info("wake phrase detected (score %.2f) at %.2f", score, frame.time)
        return True

    def take_buffer(self) -> list[Frame]:
        """The buffered frames, oldest first; the buffer is emptied."""
        frames = list(self._buffer)
        self._buffer.clear()
        return frames

    def reset(self) -> None:
        """Forget the buffer and the detector's state (the refractory period stays)."""
        self._buffer.clear()
        self._detector.reset()


# --- The wake phrase in transcripts ---------------------------------------------------

_WORD = re.compile(r"[\w']+")


def strip_wake_phrase(text: str, phrase: str, *, max_lead_words: int = 2) -> str:
    """`text` without the wake phrase at (or within `max_lead_words` words of) its start.

    Tolerant of punctuation, case and small transcription variants ("Hey, Jarvis.",
    "Hay Jarvis", "Jarvis"). Text without the phrase is returned unchanged.
    """
    target = _WORD.findall(phrase.lower())
    if not target:
        return text.strip()
    words = list(_WORD.finditer(text))
    joined_target = " ".join(target)
    best: tuple[float, int] | None = None  # (similarity, index of the last word matched)
    for start in range(min(max_lead_words + 1, len(words))):
        for size in {len(target) - 1, len(target), len(target) + 1}:
            if size < 1 or start + size > len(words) or (size < len(target) and start):
                continue  # the name alone counts only as the first word
            candidate = " ".join(w.group().lower() for w in words[start : start + size])
            if size < len(target):  # only the name (the phrase's last words) was heard
                ratio = difflib.SequenceMatcher(None, candidate, " ".join(target[-size:])).ratio()
                ratio = ratio if ratio >= 0.85 else 0.0
            elif size == len(target):  # word by word, so "tell jarvis" is not "hey jarvis"
                pairs = zip(candidate.split(), target, strict=True)
                if min(difflib.SequenceMatcher(None, a, b).ratio() for a, b in pairs) < 0.5:
                    continue
                ratio = difflib.SequenceMatcher(None, candidate, joined_target).ratio()
            else:  # a word split in two ("hey jar vis")
                ratio = difflib.SequenceMatcher(None, candidate, joined_target).ratio()
                ratio = ratio if ratio >= 0.85 else 0.0
            if ratio >= 0.75 and (best is None or ratio > best[0]):
                best = (ratio, start + size - 1)
    if best is None:
        return text.strip()
    rest = text[words[best[1]].end() :].lstrip(" \t\n,.;:!?-—…")
    return rest[:1].upper() + rest[1:] if rest else ""


class WakePhraseSTT:
    """An STT that removes the wake phrase from the start of each transcript."""

    def __init__(self, stt: STT, phrase: str) -> None:
        self._stt = stt
        self._phrase = phrase
        self.capabilities = stt.capabilities

    async def transcribe(self, audio: AudioChunk) -> Transcript:
        transcript = await self._stt.transcribe(audio)
        text = strip_wake_phrase(transcript.text, self._phrase)
        if text != transcript.text.strip():
            logger.info("removed the wake phrase: %r -> %r", transcript.text, text)
        return transcript.model_copy(update={"text": text})

    async def prepare(self) -> None:
        prepare = getattr(self._stt, "prepare", None)
        if prepare is not None:
            await prepare()


# --- Chimes ------------------------------------------------------------------------------


def chime(notes: tuple[float, ...], *, note_s: float = 0.09, amp: float = 0.12) -> AudioChunk:
    """Short sine notes with soft edges, in the core output format."""
    import numpy as np

    from lucy.voice.audio import float_to_pcm

    n = int(note_s * OUTPUT_SAMPLE_RATE)
    t = np.arange(n) / OUTPUT_SAMPLE_RATE
    envelope = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)  # 10 ms fades
    parts = [amp * envelope * np.sin(2 * np.pi * f * t) for f in notes]
    return AudioChunk(data=float_to_pcm(np.concatenate(parts)), sample_rate=OUTPUT_SAMPLE_RATE)


WAKE_NOTES: Final = (660.0, 990.0)  # rising: listening
SLEEP_NOTES: Final = (784.0, 523.0)  # falling: back to waiting


# --- The listener ------------------------------------------------------------------------


class ListenerState(StrEnum):
    WAITING = "waiting"
    AWAKE = "awake"
    STOPPED = "stopped"


@dataclass(frozen=True)
class ListenerConfig:
    phrase: str = "hey jarvis"
    followup_s: float = 8.0
    chime: bool = True


SessionFactory = Callable[[AudioIO, VoiceUI], Awaitable["VoiceSession"]]
"""Builds the voice session on the first wake: loads the speech parts, and returns a
session reading from the given `AudioIO` and reporting to the given UI."""


class _SessionIO:
    """The `AudioIO` the session sees: the real one, but frames come from the listener."""

    def __init__(self, io: AudioIO, frames: AsyncIterator[Frame]) -> None:
        self._io = io
        self._frames = frames

    def frames(self) -> AsyncIterator[Frame]:
        return self._frames

    async def play(
        self, chunks: AsyncIterable[AudioChunk], *, on_start: Callable[[], None] | None = None
    ) -> None:
        await self._io.play(chunks, on_start=on_start)

    def stop_playback(self) -> None:
        self._io.stop_playback()

    @property
    def played_samples(self) -> int:
        return self._io.played_samples

    def now(self) -> float:
        return self._io.now()

    def describe(self) -> str:
        return self._io.describe()

    async def close(self) -> None:
        pass  # the real `AudioIO` belongs to the caller


class _ObservingUI:
    """Passes the session's UI calls on, telling the listener about state changes."""

    def __init__(self, ui: VoiceUI, on_state: Callable[[VoiceState], None]) -> None:
        self._ui = ui
        self._on_state = on_state

    def state(self, state: VoiceState) -> None:
        self._on_state(state)
        self._ui.state(state)

    def user(self, text: str) -> None:
        self._ui.user(text)

    def reply_delta(self, text: str) -> None:
        self._ui.reply_delta(text)

    def reply_end(self, outcome: str, error: str | None) -> None:
        self._ui.reply_end(outcome, error)

    def notice(self, text: str) -> None:
        self._ui.notice(text)


class WakeListener:
    """Waits for the wake phrase, then runs a `VoiceSession` (see the module docstring).

    `run` returns when the input ends or `stop` is called. `interrupt` stops Lucy
    while she thinks or speaks.
    """

    def __init__(
        self,
        *,
        io: AudioIO,
        stage: WakeStage,
        make_session: SessionFactory,
        ui: VoiceUI,
        config: ListenerConfig | None = None,
    ) -> None:
        self._io = io
        self._stage = stage
        self._make_session = make_session
        self._ui = ui
        self._config = config or ListenerConfig()
        self._state = ListenerState.WAITING
        self._session: VoiceSession | None = None
        self._session_state = VoiceState.LISTENING
        self._deadline = float("inf")
        self._quiet_until = float("-inf")
        self._main: asyncio.Task[None] | None = None
        self._stopping = False
        self._chimes: set[asyncio.Task[None]] = set()
        self.wakes: list[float] = []  # capture time of each detection
        self.sleeps: list[float] = []  # when each return to waiting happened

    @property
    def state(self) -> ListenerState:
        return self._state

    @property
    def session(self) -> VoiceSession | None:
        return self._session

    async def run(self) -> None:
        self._main = asyncio.create_task(self._run(), name="lucy-listen")
        try:
            await self._main
        except asyncio.CancelledError:
            if not self._stopping or not self._main.cancelled():
                raise
        finally:
            for task in list(self._chimes):
                task.cancel()
            await asyncio.gather(*self._chimes, return_exceptions=True)
            self._state = ListenerState.STOPPED

    def interrupt(self) -> bool:
        return self._session.interrupt() if self._session is not None else False

    def stop(self) -> None:
        self._stopping = True
        if self._session is not None:
            self._session.stop()
        elif self._main is not None and not self._main.done():
            self._main.cancel()

    async def _run(self) -> None:
        source = self._io.frames()
        frames = self._frames(source)
        try:
            self._show(VoiceState.WAITING)
            try:
                first = await anext(frames)
            except StopAsyncIteration:
                return  # the input ended before any wake
            self._show(VoiceState.STARTING)
            ui = _ObservingUI(self._ui, self._session_state_changed)
            session = await self._make_session(_SessionIO(self._io, _prepend(first, frames)), ui)
            self._session = session
            if self._stopping:
                return
            await session.run()
        finally:
            await frames.aclose()
            with contextlib.suppress(Exception):
                await source.aclose()  # type: ignore[attr-defined]

    def _show(self, state: VoiceState) -> None:
        try:
            self._ui.state(state)
        except Exception:
            logger.exception("the voice UI failed")

    def _session_state_changed(self, state: VoiceState) -> None:
        self._session_state = state
        if state == VoiceState.LISTENING and self._state == ListenerState.AWAKE:
            self._deadline = self._io.now() + self._config.followup_s

    def _session_idle(self) -> bool:
        if self._session is None:
            return False  # still starting: the wake turn has not been heard yet
        return self._session_state == VoiceState.LISTENING and not self._session.hearing_speech

    async def _frames(self, source: AsyncIterator[Frame]) -> AsyncIterator[Frame]:
        """The frames for the session: nothing while waiting; on a wake, the rolling
        buffer, then live frames until the follow-up window passes."""
        async for frame in source:
            if self._state == ListenerState.WAITING:
                if frame.time <= self._quiet_until:
                    continue
                if not self._stage.feed(frame):
                    continue
                self._wake(frame.time)
                for buffered in self._stage.take_buffer():
                    yield buffered
                continue
            if self._session_idle() and frame.time >= self._deadline:
                self._sleep()
                continue
            yield frame

    def _wake(self, at: float) -> None:
        self.wakes.append(at)
        self._state = ListenerState.AWAKE
        self._deadline = at + self._config.followup_s
        if self._session is not None:
            self._session.forget_audio()
            self._show(VoiceState.LISTENING)
        self._chime(WAKE_NOTES)

    def _sleep(self) -> None:
        now = self._io.now()
        logger.info("no speech for %.0fs: waiting for the wake phrase", self._config.followup_s)
        self.sleeps.append(now)
        self._state = ListenerState.WAITING
        self._deadline = float("inf")
        self._stage.reset()
        self._show(VoiceState.WAITING)
        duration = self._chime(SLEEP_NOTES)
        self._quiet_until = now + duration + 0.3  # do not hear our own chime

    def _chime(self, notes: tuple[float, ...]) -> float:
        """Play a chime in the background (if enabled); returns its length in seconds."""
        if not self._config.chime:
            return 0.0
        sound = chime(notes)

        async def one() -> AsyncIterator[AudioChunk]:
            yield sound

        async def play() -> None:
            try:
                await self._io.play(one())
            except Exception as exc:
                logger.warning("could not play the chime: %s", exc)

        task = asyncio.create_task(play(), name="lucy-chime")
        self._chimes.add(task)
        task.add_done_callback(self._chimes.discard)
        return sound.frames / sound.sample_rate


async def _prepend(first: Frame, rest: AsyncIterator[Frame]) -> AsyncIterator[Frame]:
    yield first
    async for frame in rest:
        yield frame
