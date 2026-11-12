"""Test doubles and synthetic audio for the voice client. No models, devices or network."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence

import numpy as np

from lucy.core.types import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    AudioChunk,
    STTCaps,
    Transcript,
    TTSCaps,
)
from lucy.voice.audio import float_to_pcm


def tone(seconds: float, *, rate: int = INPUT_SAMPLE_RATE, amp: float = 0.5) -> AudioChunk:
    """A 440 Hz tone: "speech" for a level-based fake VAD."""
    t = np.arange(int(round(seconds * rate))) / rate
    return AudioChunk(data=float_to_pcm(amp * np.sin(2 * np.pi * 440 * t)), sample_rate=rate)


def silence(seconds: float, *, rate: int = INPUT_SAMPLE_RATE) -> AudioChunk:
    return AudioChunk(data=bytes(2 * int(round(seconds * rate))), sample_rate=rate)


def concat(*chunks: AudioChunk) -> AudioChunk:
    rates = {c.sample_rate for c in chunks}
    assert len(rates) == 1
    return AudioChunk(data=b"".join(c.data for c in chunks), sample_rate=rates.pop())


class FakeSTT:
    """Returns scripted transcripts in order (then ""), or raises a scripted exception."""

    capabilities = STTCaps(streaming=False)

    def __init__(self, texts: Sequence[str | Exception] = ()) -> None:
        self._texts = list(texts)
        self.calls: list[AudioChunk] = []
        self.prepared = 0

    async def prepare(self) -> None:
        self.prepared += 1

    async def transcribe(self, audio: AudioChunk) -> Transcript:
        self.calls.append(audio)
        await asyncio.sleep(0)
        item = self._texts.pop(0) if self._texts else ""
        if isinstance(item, Exception):
            raise item
        return Transcript(text=item)


class FakeTTS:
    """Synthesises `seconds_per_char` of tone per character, in `chunk_seconds` chunks.

    Texts containing `fail_on` raise after `fail_after` chunks. `started`,
    `finished` and `closed` record each synthesis by text; `gate`, if set, is
    awaited before each sentence's first chunk (to hold synthesis).
    """

    capabilities = TTSCaps(streaming=True)

    def __init__(
        self,
        *,
        seconds_per_char: float = 0.05,
        chunk_seconds: float = 0.1,
        fail_on: str | None = None,
        fail_after: int = 0,
        gate: Callable[[str], asyncio.Event | None] | None = None,
    ) -> None:
        self.seconds_per_char = seconds_per_char
        self.chunk_seconds = chunk_seconds
        self.fail_on = fail_on
        self.fail_after = fail_after
        self.gate = gate
        self.started: list[str] = []
        self.finished: list[str] = []
        self.closed: list[str] = []
        self.active = 0
        self.max_active = 0

    def samples_for(self, text: str) -> int:
        return int(round(len(text) * self.seconds_per_char * OUTPUT_SAMPLE_RATE))

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        self.started.append(text)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.gate is not None and (event := self.gate(text)) is not None:
                await event.wait()
            total = self.samples_for(text)
            step = int(self.chunk_seconds * OUTPUT_SAMPLE_RATE)
            sent = 0
            index = 0
            while sent < total:
                if self.fail_on is not None and self.fail_on in text and index >= self.fail_after:
                    raise RuntimeError(f"synthesis failed for {text!r}")
                n = min(step, total - sent)
                yield tone(n / OUTPUT_SAMPLE_RATE, rate=OUTPUT_SAMPLE_RATE, amp=0.3)
                sent += n
                index += 1
                await asyncio.sleep(0)
            if self.fail_on is not None and self.fail_on in text:
                raise RuntimeError(f"synthesis failed for {text!r}")
            self.finished.append(text)
        except GeneratorExit:
            self.closed.append(text)
            raise
        finally:
            self.active -= 1
