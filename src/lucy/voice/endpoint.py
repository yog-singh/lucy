"""Endpointing: deciding from per-frame speech probabilities when the user has spoken
and when they have finished.

`Endpointer` is a pure state machine with no audio processing of its own: feed
it each frame's audio, its speech probability and its capture time, and it
returns events.

- Speech starts after `min_speech_ms` of consecutive speech frames, so a click
  or a knock does not start a turn. The utterance then includes up to
  `preroll_ms` of audio from before that run, so the first syllable is kept.
- A frame is speech when its probability reaches `threshold`; once speech has
  started, it only counts as silence below `threshold - 0.15` (the hysteresis
  Silero recommends), so soft syllables do not end a turn.
- The utterance ends after `endpoint_silence_ms` of silence. Its audio runs to
  the last speech frame plus `postroll_ms`.
- An utterance reaching `max_utterance_s` is ended there (`forced`).
- An utterance whose speech (first to last speech frame) is shorter than
  `min_utterance_ms` is discarded as noise.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Final

from lucy.core.types import INPUT_SAMPLE_RATE, AudioChunk
from lucy.voice.audio import FRAME_SAMPLES

HYSTERESIS: Final = 0.15


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    """Speech began. `time` is when its first speech frame began."""

    time: float


@dataclass(frozen=True, slots=True)
class UtteranceComplete:
    """The user finished an utterance.

    `speech_start` / `speech_end`: start of the first and end of the last
    speech frame. `endpoint_time`: capture time of the frame that decided it.
    """

    audio: AudioChunk
    speech_start: float
    speech_end: float
    endpoint_time: float
    forced: bool = False

    @property
    def duration(self) -> float:
        """Length of the utterance's audio in seconds (including pre-roll)."""
        return self.audio.duration


@dataclass(frozen=True, slots=True)
class Discarded:
    """Detected speech too short to be an utterance."""

    speech_start: float
    speech_end: float
    reason: str


EndpointEvent = SpeechStarted | UtteranceComplete | Discarded


def _frames(ms: float, frame_ms: float) -> int:
    return max(0, math.ceil(ms / frame_ms - 1e-9))


class Endpointer:
    """Turns per-frame speech probabilities into utterances (see the module docstring)."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_speech_ms: int = 160,
        endpoint_silence_ms: int = 600,
        preroll_ms: int = 300,
        postroll_ms: int = 200,
        max_utterance_s: float = 30.0,
        min_utterance_ms: int = 250,
        frame_samples: int = FRAME_SAMPLES,
        sample_rate: int = INPUT_SAMPLE_RATE,
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        self.frame_seconds = frame_samples / sample_rate
        frame_ms = 1000 * self.frame_seconds
        self.threshold = threshold
        self.continue_threshold = max(0.01, threshold - HYSTERESIS)
        self.min_speech_frames = max(1, _frames(min_speech_ms, frame_ms))
        self.silence_frames = max(1, _frames(endpoint_silence_ms, frame_ms))
        self.preroll_frames = _frames(preroll_ms, frame_ms)
        self.postroll_frames = _frames(postroll_ms, frame_ms)
        self.max_frames = max(1, _frames(1000 * max_utterance_s, frame_ms))
        self.min_utterance_s = min_utterance_ms / 1000
        self._sample_rate = sample_rate
        self.reset()

    def reset(self) -> None:
        """Back to waiting for speech, forgetting any audio."""
        self._history: deque[tuple[bytes, float]] = deque(
            maxlen=self.preroll_frames + self.min_speech_frames
        )
        self._run = 0  # consecutive speech frames while waiting
        self._in_speech = False
        self._audio: list[bytes] = []
        self._speech_start = 0.0
        self._speech_end = 0.0
        self._last_speech_index = 0  # index in _audio of the last speech frame
        self._silence = 0
        self._speech_frames = 0  # frames since speech started (for the maximum)

    @property
    def in_speech(self) -> bool:
        """Whether an utterance is in progress."""
        return self._in_speech

    def feed(self, audio: bytes, probability: float, time: float) -> list[EndpointEvent]:
        """Process one frame (`time` = capture time of its end); return what it decided."""
        if not self._in_speech:
            return self._wait(audio, probability, time)
        return self._listen(audio, probability, time)

    def flush(self, time: float) -> list[EndpointEvent]:
        """End of input: complete (or discard) the utterance in progress, if any."""
        if not self._in_speech:
            self.reset()
            return []
        return [self._finish(time, forced=False)]

    def _wait(self, audio: bytes, probability: float, time: float) -> list[EndpointEvent]:
        self._history.append((audio, time))
        if probability < self.threshold:
            self._run = 0
            return []
        self._run += 1
        if self._run < self.min_speech_frames:
            return []
        # Speech: the run plus up to preroll_frames before it.
        frames = list(self._history)
        self._audio = [a for a, _ in frames]
        self._speech_start = time - self._run * self.frame_seconds
        self._speech_end = time
        self._last_speech_index = len(self._audio) - 1
        self._in_speech = True
        self._silence = 0
        self._speech_frames = self._run
        self._history.clear()
        self._run = 0
        events: list[EndpointEvent] = [SpeechStarted(self._speech_start)]
        if self._speech_frames >= self.max_frames:
            events.append(self._finish(time, forced=True))
        return events

    def _listen(self, audio: bytes, probability: float, time: float) -> list[EndpointEvent]:
        self._audio.append(audio)
        self._speech_frames += 1
        if probability >= self.continue_threshold:
            self._silence = 0
            self._speech_end = time
            self._last_speech_index = len(self._audio) - 1
        else:
            self._silence += 1
            if self._silence >= self.silence_frames:
                return [self._finish(time, forced=False)]
        if self._speech_frames >= self.max_frames:
            # Reached during trailing silence, it is an ordinary end.
            return [self._finish(time, forced=self._silence == 0)]
        return []

    def _finish(self, time: float, *, forced: bool) -> EndpointEvent:
        start, end = self._speech_start, self._speech_end
        keep = self._last_speech_index + 1 + (0 if forced else self.postroll_frames)
        audio = b"".join(self._audio[:keep])
        self.reset()
        if end - start < self.min_utterance_s - 1e-9:
            return Discarded(start, end, f"speech of {end - start:.2f}s is too short")
        chunk = AudioChunk(data=audio, sample_rate=self._sample_rate)
        return UtteranceComplete(chunk, start, end, time, forced)
