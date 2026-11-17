"""The glance at the start of each voice turn (ADR 0026, stage 3).

Mirrors the speaker tag (ADR 0021, `lucy.voice.speakers`): `Glancer.start()` is
called as a voice utterance ends and runs `VisionService.who` (one capture, face
matching, no scene model; the camera is released straight after) concurrently
with transcription. The session waits at most `wait_ms` for it once the
transcript is ready; a glance that is late is dropped (it finishes in the
background and is never attached to a later turn). A failing camera or
recogniser is logged once and the turn goes on without a glance.

The result is a `Seen` (names with scores, and a count of faces nobody matched):
no image or embedding is kept. A face match is a tag, never an authorisation:
nothing may gate an action, a tool, a confirmation or memory on it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from lucy.core.config import Settings
from lucy.core.events import Seen, SeenFace
from lucy.vision.faces import Person
from lucy.vision.service import VisionService

logger = logging.getLogger(__name__)


def seen_from(people: tuple[Person, ...]) -> Seen:
    named = tuple(
        SeenFace(name=p.name, score=round(p.score or 0.0, 3)) for p in people if p.name is not None
    )
    return Seen(people=named, unknown=sum(1 for p in people if p.name is None))


class Glancer:
    """Starts one glance per voice turn through `service` (see the module docstring)."""

    def __init__(
        self, service: VisionService, *, wait_ms: int = 400, warmup_frames: int | None = 2
    ) -> None:
        self._service = service
        self.wait_s = wait_ms / 1000
        self._warmup = warmup_frames
        self._failed = False  # log a failure once, not every turn
        self._pending: set[asyncio.Task[Seen | None]] = set()
        self.last_glance_s: float | None = None
        """Seconds the last finished glance took, capture included."""

    def start(self) -> asyncio.Task[Seen | None]:
        """Capture and match in the background; the task's result is None on failure."""
        task = asyncio.create_task(self._glance(), name="lucy-glance")
        self._pending.add(task)  # held until done, so a late glance is not collected early
        task.add_done_callback(self._pending.discard)
        return task

    async def _glance(self) -> Seen | None:
        started = time.perf_counter()
        try:
            people = await self._service.who(warmup_frames=self._warmup)
        except Exception as exc:
            if not self._failed:
                logger.warning("the glance failed (later failures are not logged): %s", exc)
                self._failed = True
            return None
        self.last_glance_s = round(time.perf_counter() - started, 3)
        return seen_from(people)


def build_glancer(
    settings: Settings, service: VisionService | None, notice: Callable[[str], None]
) -> Glancer | None:
    """The glancer for a voice session when `[vision] glance` is on and can work;
    otherwise None, with one `notice` saying why when glance is on."""
    vision = settings.vision
    if not vision.glance:
        return None
    reason: str | None = None
    if not vision.enabled:
        reason = "[vision] glance is on but [vision] is not enabled"
    elif not vision.faces.enabled:
        reason = "[vision] glance is on but [vision.faces] is not enabled"
    elif service is None:
        reason = "[vision] glance is on but the camera is not set up"
    elif not service.faces_ready:
        reason = f"[vision] glance is off: {service.faces_off or 'face matching is not ready'}"
    if reason is not None:
        notice(f"{reason}; no glance at the start of turns.")
        return None
    assert service is not None
    return Glancer(
        service, wait_ms=vision.glance_wait_ms, warmup_frames=vision.glance_warmup_frames
    )
