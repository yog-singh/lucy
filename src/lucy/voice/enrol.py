"""Voice enrolment led by code in the voice client (ADR 0021, amendment).

The `enroll_voice` / `replace_voice` tools only request a session; `VoiceSession`
runs it after the requesting turn has been spoken. This module holds what does not
need the session: the sentence pool, the loose transcript check, and `Enroller`,
which turns accepted clips into an enrolment (files under `speakers/NAME/`, the
`SpeakerIdentifier` port, a verification).

The sentences are shown on screen, never read out by TTS: the user reads in their
own voice and pace, and Lucy's voice is not imitated or captured.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from lucy.core.ports import SpeakerIdentifier
from lucy.core.types import AudioChunk
from lucy.core.workspace import SPEAKERS, Workspace

logger = logging.getLogger(__name__)

SENTENCES: Final = (
    "The quick brown fox jumps over the lazy dog near the quiet river bank.",
    "On Sunday morning we walked along the beach and watched the boats sail past.",
    "Please put the yellow cups on the top shelf, next to the big blue jug.",
    "A gentle breeze moved through the garden as the evening light began to fade.",
    "My neighbour bakes fresh bread every Thursday and shares it with the street.",
    "The train to the city leaves at half past seven, so we should not be late.",
    "Children laughed and played in the park while their parents chatted on benches.",
    "Thick fog rolled over the hills, and the village lights looked soft and warm.",
    "She opened the window, heard the birds singing, and decided to go for a jog.",
    "Every autumn the old oak tree drops its leaves across the narrow garden path.",
)
CANCEL_WORDS: Final = frozenset(
    {"cancel", "stop", "stop it", "cancel it", "cancel that", "never mind", "quit"}
)
ORDINALS: Final = ("First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth")
MIN_SPEECH_S: Final = 3.0
"""Shorter clips score near zero in the speaker model."""
MIN_OVERLAP: Final = 0.5
"""Fraction of the expected sentence's words that must be heard."""
_WORD: Final = re.compile(r"[a-z0-9']+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.casefold().replace("’", "'"))


def overlap(expected: str, heard: str) -> float:
    """The fraction of `expected`'s distinct words found in `heard` (loose, order-free)."""
    want = set(words(expected))
    if not want:
        return 1.0
    return len(want & set(words(heard))) / len(want)


def is_cancel(text: str) -> bool:
    """Whether the whole utterance is a cancel word ("cancel", "stop", ...)."""
    said = " ".join(w for w in words(text) if w not in ("lucy", "please"))
    return said in CANCEL_WORDS


def pick_sentences(count: int, rng: random.Random | None = None) -> list[str]:
    """`count` different sentences from the pool, in random order."""
    return (rng or random.Random()).sample(SENTENCES, min(count, len(SENTENCES)))


@dataclass(frozen=True)
class EnrolmentRequest:
    name: str
    replace: bool


class Enroller:
    """Enrols accepted clips through the `SpeakerIdentifier` port."""

    def __init__(
        self,
        identifier: SpeakerIdentifier,
        workspace: Workspace,
        *,
        sentences: int = 5,
        retries: int = 2,
        silence_s: float = 20.0,
        rng: random.Random | None = None,
    ) -> None:
        self.identifier = identifier
        self.workspace = workspace
        self.sentences = sentences
        self.retries = retries
        self.silence_s = silence_s
        self.rng = rng

    def pick(self) -> list[str]:
        return pick_sentences(self.sentences, self.rng)

    async def enrol(self, name: str, clips: Sequence[AudioChunk]) -> str:
        """Write `clips` to `speakers/NAME/` (replacing what was there), enrol them,
        verify with one clip, and return the sentence to say."""
        from lucy.voice.speakers import write_wav

        def store() -> list[Path]:
            target = self.workspace.dir(SPEAKERS, name)
            for old in target.iterdir():
                if old.is_file():
                    old.unlink()
                else:
                    shutil.rmtree(old, ignore_errors=True)
            paths = []
            for i, clip in enumerate(clips, start=1):
                path = target / f"enrol_{i}.wav"
                write_wav(path, clip)
                paths.append(path)
            return paths

        try:
            paths = await asyncio.to_thread(store)
            existing = await asyncio.to_thread(self.identifier.speakers)
            if name in existing:
                await asyncio.to_thread(self.identifier.remove, name)
            await asyncio.to_thread(self.identifier.enroll, name, paths)
        except Exception as exc:
            logger.warning("voice enrolment of %s failed: %s", name, exc)
            shutil.rmtree(self.workspace.resolve(Path(SPEAKERS) / name), ignore_errors=True)
            return f"Sorry, I could not enrol {name}'s voice: something went wrong saving it."
        try:
            matches = await asyncio.to_thread(self.identifier.identify, paths[-1])
        except Exception as exc:
            logger.warning("verifying the enrolment of %s failed: %s", name, exc)
            matches = []
        best = matches[0] if matches else None
        if best is not None and best.name == name and best.accepted:
            return f"Done. I'll recognise you as {name}."
        return (
            f"I enrolled {name}'s voice, but a test recording was not recognised, "
            "so recognition may be unreliable."
        )
