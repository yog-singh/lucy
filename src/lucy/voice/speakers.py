"""Tagging each voice utterance with who spoke (stage 1 of speaker awareness, ADR 0021).

`SpeakerTagger.start(audio)` writes the utterance as a 16 kHz mono WAV into the
workspace's `utterances/` and identifies it on a worker thread, concurrently
with transcription; the session waits at most `wait_ms` for it once the
transcript is ready, and otherwise proceeds untagged. Kept clips are pruned by
age and total size; with `keep_utterances_days = 0` each clip is deleted right
after identification. The tag is a label only: nothing in the turn acts on it.
"""

from __future__ import annotations

import asyncio
import logging
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from lucy.core.ports import SpeakerIdentifier
from lucy.core.types import AudioChunk
from lucy.core.workspace import UTTERANCES, Workspace

logger = logging.getLogger(__name__)

PRUNE_EVERY: int = 50
"""Prune kept clips every this many utterances (and at startup)."""


@dataclass(frozen=True)
class SpeakerTag:
    speaker: str | None
    score: float | None
    audio_path: str | None  # relative to the workspace, None once deleted
    audio_s: float


def write_wav(path: Path, audio: AudioChunk) -> None:
    """Write 16-bit PCM mono `audio` as a WAV file (owner-only)."""
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(audio.sample_rate)
        out.writeframes(audio.data)
    path.chmod(0o600)


def prune_clips(
    directory: Path, *, keep_days: float, max_bytes: float, now: float | None = None
) -> int:
    """Delete clips older than `keep_days`, then the oldest until under `max_bytes`."""
    now = time.time() if now is None else now
    clips = sorted(
        (p for p in directory.glob("*.wav") if p.is_file()), key=lambda p: p.stat().st_mtime
    )
    removed = 0
    kept: list[tuple[Path, int]] = []
    for clip in clips:
        st = clip.stat()
        if now - st.st_mtime > keep_days * 86400:
            clip.unlink(missing_ok=True)
            removed += 1
        else:
            kept.append((clip, st.st_size))
    total = sum(size for _, size in kept)
    for clip, size in kept:  # oldest first
        if total <= max_bytes:
            break
        clip.unlink(missing_ok=True)
        total -= size
        removed += 1
    return removed


class SpeakerTagger:
    def __init__(
        self,
        identifier: SpeakerIdentifier,
        workspace: Workspace,
        *,
        wait_ms: int = 150,
        keep_days: float = 1.0,
        max_mb: float = 200.0,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._identifier = identifier
        self._workspace = workspace
        self.wait_s = wait_ms / 1000
        self._keep_days = keep_days
        self._max_bytes = max_mb * 1024 * 1024
        self._clock = clock
        self._count = 0
        self.last_identify_s: float | None = None
        self.prune()

    @property
    def identifier(self) -> SpeakerIdentifier:
        return self._identifier

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    def prune(self) -> None:
        try:
            prune_clips(
                self._workspace.dir(UTTERANCES),
                keep_days=self._keep_days,
                max_bytes=self._max_bytes,
            )
        except OSError as exc:
            logger.warning("could not prune utterance clips: %s", exc)

    def start(self, audio: AudioChunk, message_id: str) -> asyncio.Task[SpeakerTag]:
        """Write the clip (named by time and the `UserMessage` id it will belong to)
        and identify it in the background."""
        return asyncio.create_task(self._tag(audio, message_id), name="lucy-speaker-tag")

    async def _tag(self, audio: AudioChunk, message_id: str) -> SpeakerTag:
        audio_s = len(audio.data) / 2 / audio.sample_rate
        stamp = self._clock().strftime("%Y%m%dT%H%M%S.%fZ")
        path = self._workspace.dir(UTTERANCES) / f"{stamp}_{message_id}.wav"
        keep = self._keep_days > 0 and self._max_bytes > 0
        try:
            await asyncio.to_thread(write_wav, path, audio)
            started = time.perf_counter()
            matches = await asyncio.to_thread(self._identifier.identify, path)
            self.last_identify_s = time.perf_counter() - started
        except Exception as exc:
            logger.warning("speaker identification failed: %s", exc)
            path.unlink(missing_ok=True)
            return SpeakerTag(None, None, None, audio_s)
        finally:
            self._count += 1
            if self._count % PRUNE_EVERY == 0:
                await asyncio.to_thread(self.prune)
        if not keep:
            path.unlink(missing_ok=True)
        best = matches[0] if matches else None
        return SpeakerTag(
            speaker=best.name if best is not None and best.accepted else None,
            score=best.score if best is not None else None,
            audio_path=self._workspace.relative(path) if keep else None,
            audio_s=audio_s,
        )


async def await_tag(task: asyncio.Task[SpeakerTag], wait_s: float) -> SpeakerTag | None:
    """The tag if it is ready within `wait_s`; otherwise None (identification finishes
    in the background and is dropped)."""
    try:
        return await asyncio.wait_for(asyncio.shield(task), wait_s)
    except TimeoutError:
        return None
    except Exception:
        return None
