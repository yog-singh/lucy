"""Speaking a reply sentence by sentence, and knowing how much of it was heard (ADR 0016).

`SpeechPlayer` takes the sentences of one reply in order. Each is synthesised
with the TTS port as it arrives; the next sentence's synthesis starts while the
current one plays (`lookahead`), so sentences follow without a gap, but audio
is always played in sentence order. All of a reply's audio goes through one
`AudioIO.play` call, so the player knows where each sentence starts and ends
in the played stream and can compare that with `AudioIO.played_samples`.

What was spoken is approximate inside a sentence: whole sentences played count
in full, and of the sentence in progress the same proportion of its text as of
its audio is counted, cut back to a word boundary. Speech is not uniform over a
sentence (leading silence, pauses at commas), so the cut can be off by a few
words; stage 1 aligns at sentence granularity only.

Before synthesis, markdown leftovers and emoji are removed; a sentence with
nothing speakable left is skipped. A TTS failure for one sentence is logged and
that sentence skipped (it was shown on screen), not fatal.
"""

from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Final

from lucy.core.ports import TTS
from lucy.core.types import OUTPUT_SAMPLE_RATE, AudioChunk
from lucy.voice.audio import AudioIO

logger = logging.getLogger(__name__)

CHARS_PER_SECOND: Final = 14.0
"""Rough speaking rate, used only to estimate a sentence's length while it is
still being synthesised."""

_LINK: Final = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_URL: Final = re.compile(r"https?://\S+")
_CODE_FENCE: Final = re.compile(r"```[^\n]*")
_HEADING: Final = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_BULLET: Final = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.MULTILINE)
_QUOTE: Final = re.compile(r"^\s*>\s?", re.MULTILINE)
_EMPHASIS: Final = re.compile(r"(\*{1,3}|_{2,3}|~~|`+)")
_TABLE_RULE: Final = re.compile(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?")
_INVISIBLE: Final = frozenset({"‍", "︎", "️", "⃣"})


def speakable(text: str) -> str:
    """`text` without markdown leftovers and emoji, or "" if nothing speakable is left."""
    text = _CODE_FENCE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub(" ", text)
    text = _TABLE_RULE.sub(" ", text)
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _QUOTE.sub("", text)
    text = _EMPHASIS.sub("", text)
    text = text.replace("|", " ")
    kept = []
    for char in text:
        if char in _INVISIBLE:
            continue
        category = unicodedata.category(char)
        if category in ("So", "Cs", "Co") or (category == "Sk" and ord(char) > 0x2000):
            continue  # emoji, pictographs, skin-tone modifiers
        kept.append(char)
    text = " ".join("".join(kept).split())
    return text if any(c.isalnum() for c in text) else ""


def cut_at_words(text: str, fraction: float) -> str:
    """The first `fraction` of `text`, cut back to the last whole word."""
    if fraction >= 1:
        return text
    if fraction <= 0:
        return ""
    end = int(len(text) * fraction)
    if end >= len(text):
        return text
    head = text[:end]
    if not text[end].isspace() and not head[-1:].isspace():
        head = head.rsplit(None, 1)[0] if any(c.isspace() for c in head) else ""
    return head.rstrip(" ,;:-–—")


@dataclass
class _Sentence:
    text: str
    speech: str
    chunks: asyncio.Queue[AudioChunk | None] = field(default_factory=asyncio.Queue)
    task: asyncio.Task[None] | None = None
    start: int | None = None  # where its audio starts in this player's stream (samples)
    submitted: int = 0  # samples handed to playback so far
    synthesized: int = 0  # samples received from the TTS so far
    complete: bool = False  # the TTS has delivered all of it
    failed: bool = False

    @property
    def skipped(self) -> bool:
        return not self.speech

    @property
    def end(self) -> int | None:
        """Where its audio ends in the player's stream, once that is known."""
        if self.start is None or not self.complete:
            return None
        return self.start + self.synthesized


class SpeechPlayer:
    """Speaks one reply's sentences in order through `io` (see the module docstring).

    Use: `add` sentences as they become ready, `finish` when there are no more,
    `await wait()` until everything has been played. `stop` cuts speech off at
    once and returns the text actually spoken. `on_first_audio` is called when
    the TTS delivers the reply's first audio, `on_first_played` when its first
    sample is played, `on_error(sentence, exc)` when a sentence fails.
    """

    def __init__(
        self,
        tts: TTS,
        io: AudioIO,
        *,
        lookahead: int = 1,
        on_first_audio: Callable[[], None] | None = None,
        on_first_played: Callable[[], None] | None = None,
        on_error: Callable[[str, BaseException], None] | None = None,
    ) -> None:
        if lookahead < 0:
            raise ValueError("lookahead must be 0 or more")
        self._tts = tts
        self._io = io
        self._lookahead = lookahead
        self._on_first_audio = on_first_audio
        self._on_first_played = on_first_played
        self._on_error = on_error
        self._sentences: list[_Sentence] = []
        self._current = 0
        self._submitted = 0
        self._base: int | None = None
        self._changed = asyncio.Event()
        self._finished = False
        self._stopped: str | None = None
        self._play_task: asyncio.Task[None] | None = None
        self._got_audio = False
        self._played_at_stop: int | None = None

    # --- Feeding ------------------------------------------------------------------

    @property
    def sentence_count(self) -> int:
        """Sentences added so far."""
        return len(self._sentences)

    @property
    def started(self) -> bool:
        """Whether any sentence has been added."""
        return bool(self._sentences)

    @property
    def stopped(self) -> bool:
        return self._stopped is not None

    def add(self, text: str) -> None:
        """Queue a sentence to be spoken after the ones already added."""
        if self._finished or self._stopped is not None:
            return
        self._sentences.append(_Sentence(text=text, speech=speakable(text)))
        self._prefetch()
        if self._play_task is None:
            self._base = self._io.played_samples
            self._play_task = asyncio.create_task(
                self._io.play(self._source(), on_start=self._first_played), name="lucy-speech"
            )
        self._changed.set()

    def finish(self) -> None:
        """No more sentences will be added."""
        self._finished = True
        self._changed.set()

    async def wait(self) -> None:
        """Return once every sentence has been played (or speech was stopped)."""
        self.finish()
        if self._play_task is None:
            return
        try:
            await asyncio.shield(self._play_task)
        except asyncio.CancelledError:
            if self._play_task.cancelled() and self._stopped is not None:
                return  # stopped: the cancellation was ours
            raise
        except Exception:
            logger.exception("playback failed")

    # --- Stopping -------------------------------------------------------------------

    def stop(self) -> str:
        """Silence speech now, cancel what is pending, and return the text spoken (all
        sentences; see `spoken_text`). Idempotent."""
        if self._stopped is not None:
            return self._stopped
        self._played_at_stop = self._played()
        self._io.stop_playback()
        self._finished = True
        if self._play_task is not None:
            self._play_task.cancel()
        for sentence in self._sentences:
            if sentence.task is not None:
                sentence.task.cancel()
        self._changed.set()
        self._stopped = self.spoken_text()
        return self._stopped

    async def aclose(self) -> None:
        """Stop if needed and wait until every task of this player has ended."""
        if self._stopped is None and not self._done():
            self.stop()
        tasks = [s.task for s in self._sentences if s.task is not None]
        if self._play_task is not None:
            tasks.append(self._play_task)
        await asyncio.gather(*tasks, return_exceptions=True)

    def _done(self) -> bool:
        return self._play_task is None or self._play_task.done()

    # --- Progress -------------------------------------------------------------------

    def _played(self) -> int:
        if self._played_at_stop is not None:
            return self._played_at_stop
        if self._base is None:
            return 0
        return max(0, self._io.played_samples - self._base)

    @property
    def played_seconds(self) -> float:
        """Audio of this reply played so far."""
        return self._played() / OUTPUT_SAMPLE_RATE

    @property
    def sentences_played(self) -> int:
        """Sentences played in full (skipped and failed ones count as passed)."""
        played, count = self._played(), 0
        for sentence in self._sentences:
            if sentence.start is None:
                if sentence.skipped or sentence.failed:
                    count += 1
                    continue
                break
            if sentence.end is None or played < sentence.end:
                break
            count += 1
        return count

    def spoken_text(self, start: int = 0) -> str:
        """The text heard so far of the sentences from index `start` on: whole sentences
        played, plus the proportional part of the one in progress (approximate)."""
        played = self._played()
        parts: list[str] = []
        for sentence in self._sentences[start:]:
            if sentence.start is None:
                if sentence.skipped or sentence.failed:
                    continue
                break  # not reached yet
            if sentence.end is not None and played >= sentence.end:
                parts.append(sentence.text)
                continue
            if played <= sentence.start:
                break
            if sentence.end is not None:
                total = sentence.end - sentence.start
            else:
                estimate = len(sentence.speech) / CHARS_PER_SECOND * OUTPUT_SAMPLE_RATE
                total = max(sentence.submitted, int(estimate))
            parts.append(cut_at_words(sentence.text, (played - sentence.start) / max(1, total)))
            break
        return " ".join(part for part in parts if part)

    # --- Internals --------------------------------------------------------------------

    def _first_played(self) -> None:
        if self._on_first_played is not None:
            self._on_first_played()

    def _prefetch(self) -> None:
        """Start synthesis of the current sentence and up to `lookahead` after it."""
        last = min(len(self._sentences), self._current + self._lookahead + 1)
        for index in range(self._current, last):
            sentence = self._sentences[index]
            if sentence.task is None and not sentence.skipped and self._stopped is None:
                sentence.task = asyncio.create_task(self._synthesize(sentence))

    async def _synthesize(self, sentence: _Sentence) -> None:
        try:
            async with aclosing(self._tts.synthesize(sentence.speech)) as stream:
                async for chunk in stream:
                    if not self._got_audio:
                        self._got_audio = True
                        if self._on_first_audio is not None:
                            self._on_first_audio()
                    sentence.synthesized += chunk.frames
                    sentence.chunks.put_nowait(chunk)
            sentence.complete = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            sentence.failed = True
            logger.warning("could not speak %r: %s", sentence.text, exc)
            if self._on_error is not None:
                self._on_error(sentence.text, exc)
        finally:
            sentence.chunks.put_nowait(None)

    async def _source(self) -> AsyncIterator[AudioChunk]:
        """The reply's audio, sentence after sentence, recording where each one lies."""
        while True:
            while self._current >= len(self._sentences):
                if self._finished:
                    return
                self._changed.clear()
                await self._changed.wait()
            sentence = self._sentences[self._current]
            self._prefetch()
            if not sentence.skipped:
                while (chunk := await sentence.chunks.get()) is not None:
                    if sentence.start is None:
                        sentence.start = self._submitted
                    self._submitted += chunk.frames
                    sentence.submitted += chunk.frames
                    yield chunk
            self._current += 1
