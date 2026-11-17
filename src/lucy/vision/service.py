"""`VisionService`: one look through the camera, described in words (ADR 0026).

Stage 1: `look(question)` takes one frame, asks the `vision` model role about it
with a short fixed instruction and returns the answer as text, which reaches the
chat model as a tool result. The frame lives only in memory for that one request
and is dropped afterwards; nothing is written to disk or to the event log
(`Message.images` is never serialised).

Stage 2: with a `FaceRecognizer` and a `FaceBook`, `see` also finds the faces in
the same frame, concurrently with the scene model, and matches them against the
book (`Sight.people`); `who` does only that, without the scene model; `enrol_face`
captures a few frames and stores the face's embeddings under a name. A face match
is a tag, never an authorisation: a photo of a person matches like the person.

Stage 3: `lucy.vision.glance.Glancer` calls `who` as each voice utterance ends.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Final

from lucy.core.gateway import LLMGateway, Priority
from lucy.core.ports import Camera, CameraError, DetectedFace, FaceRecognizer
from lucy.core.types import Image, Message, Role
from lucy.vision.faces import FaceBook, Person

logger = logging.getLogger(__name__)

INSTRUCTION: Final = (
    "You are looking through a camera on behalf of a voice assistant. Answer the "
    "question about this photo in plain spoken English, in two or three short "
    "sentences, with no lists or formatting. Say only what is visible in the photo; "
    "if you are unsure what something is, say so instead of guessing."
)
GENERAL_QUESTION: Final = "What do you see?"


class VisionError(RuntimeError):
    """A look failed; the message says why, in words fit for the user."""


@dataclass(frozen=True)
class Sight:
    """What one look found. `description` answers the question; `people` holds one
    `Person` per face in view (name or None, score, position), empty when face
    matching is off; `faces_note` says why names are missing, if they are.
    `seconds` is how long capture and answer took together."""

    description: str
    people: tuple[Person, ...] = field(default=())
    seconds: float = 0.0
    faces_note: str | None = None


@dataclass(frozen=True)
class FaceRules:
    """How faces are matched and enrolled (`[vision.faces]`)."""

    threshold: float = 0.40
    min_score: float = 0.6
    min_size: float = 0.08
    enrol_frames: int = 5
    enrol_interval: float = 0.5


class VisionService:
    """Looks through `camera` and asks the `role` model of `gateway` about the frame."""

    def __init__(
        self,
        *,
        gateway: LLMGateway,
        camera: Camera,
        role: str = "vision",
        timeout: float = 60.0,
        recognizer: FaceRecognizer | None = None,
        book: FaceBook | None = None,
        rules: FaceRules | None = None,
        faces_off: str | None = None,
    ) -> None:
        self._gateway = gateway
        self._camera = camera
        self._role = role
        self._timeout = timeout
        self._lock = asyncio.Lock()
        self._recognizer = recognizer if book is not None else None
        self._book = book
        self.rules = rules or FaceRules()
        self.faces_off = faces_off
        """Why face matching is off (for example the models are missing), if it is."""
        self.last_who_s: float | None = None
        """Seconds the last `who` spent finding and matching faces (capture excluded)."""

    @property
    def faces_ready(self) -> bool:
        return self._recognizer is not None and self.faces_off is None

    @property
    def book(self) -> FaceBook | None:
        return self._book

    @property
    def role(self) -> str:
        return self._role

    async def look(self, question: str = "") -> str:
        """The vision model's answer to `question` about one fresh frame."""
        return (await self.see(question)).description

    async def see(self, question: str = "") -> Sight:
        """One look: capture, ask, drop the frame. Raises `VisionError`."""
        question = " ".join(question.split()) or GENERAL_QUESTION
        started = time.monotonic()
        people: tuple[Person, ...] = ()
        note: str | None = self.faces_off if self._recognizer is not None else None
        async with self._lock:  # one camera, one look at a time
            try:
                async with asyncio.timeout(self._timeout):
                    image = await self._capture()
                    faces_task = None
                    if self.faces_ready:
                        faces_task = asyncio.create_task(self._people(image))
                    try:
                        text = await self._ask(image, question)
                    except BaseException:
                        if faces_task is not None:
                            faces_task.cancel()
                        raise
                    del image
                    if faces_task is not None:
                        try:
                            people = await faces_task
                        except Exception as exc:  # the description still stands
                            logger.warning("face matching failed during a look: %s", exc)
                            note = f"names unavailable: {exc}"
            except TimeoutError:
                raise VisionError(
                    f"Looking took longer than {self._timeout:g} seconds and was stopped."
                ) from None
        if not text:
            raise VisionError("The vision model gave no answer about the picture.")
        return Sight(
            description=text,
            people=people,
            seconds=round(time.monotonic() - started, 2),
            faces_note=note,
        )

    async def who(self, *, warmup_frames: int | None = None) -> tuple[Person, ...]:
        """The people in view now: one capture and face matching, no scene model.
        `warmup_frames` overrides the camera's own (the glance uses fewer). Raises
        `VisionError` (also when face matching is off)."""
        self._require_faces()
        async with self._lock:
            try:
                async with asyncio.timeout(self._timeout):
                    image = await self._capture(warmup_frames)
                    started = time.perf_counter()
                    people = await self._people(image)
                    self.last_who_s = round(time.perf_counter() - started, 3)
                    return people
            except TimeoutError:
                raise VisionError(
                    f"Looking took longer than {self._timeout:g} seconds and was stopped."
                ) from None

    async def enrol_face(self, name: str) -> int:
        """Capture `rules.enrol_frames` frames about `rules.enrol_interval` apart, each
        with exactly one clear face, and store their embeddings under `name`
        (replacing any it had). Returns how many were stored. Raises `VisionError`
        saying what went wrong; nothing is stored then. No frame is kept."""
        self._require_faces()
        assert self._book is not None
        rules = self.rules
        embeddings: list[tuple[float, ...]] = []
        async with self._lock:
            try:
                async with asyncio.timeout(self._timeout):
                    for index in range(rules.enrol_frames):
                        if index:
                            await asyncio.sleep(rules.enrol_interval)
                        image = await self._capture()
                        faces = await self._detect(image)
                        del image
                        embeddings.append(self._one_face(faces).embedding)
            except TimeoutError:
                raise VisionError(
                    f"Capturing the face took longer than {self._timeout:g} seconds."
                ) from None
        self._book.enroll(name, embeddings)
        return len(embeddings)

    async def aclose(self) -> None:
        """Stop the face recogniser (its worker), if it has one running."""
        close = getattr(self._recognizer, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:
                logger.warning("could not stop the face recogniser", exc_info=True)

    def _one_face(self, faces: list[DetectedFace]) -> DetectedFace:
        rules = self.rules
        clear = [f for f in faces if f.score >= rules.min_score]
        if len(clear) > 1:
            count = len(clear)
            words = {2: "two", 3: "three", 4: "four"}
            raise VisionError(
                f"I could see {words.get(count, str(count))} faces; only the person to be "
                "enrolled should be in front of the camera."
            )
        if not clear:
            raise VisionError(
                "I couldn't see a face clearly; the person should face the camera in good light."
            )
        face = clear[0]
        size = max(face.box[2] - face.box[0], face.box[3] - face.box[1])
        if size < rules.min_size:
            raise VisionError(
                "The face was too small in the picture; the person should come closer to "
                "the camera."
            )
        return face

    def _require_faces(self) -> None:
        if self._recognizer is None or self._book is None:
            raise VisionError("Face matching is off: [vision.faces] is not enabled.")
        if self.faces_off is not None:
            raise VisionError(self.faces_off)

    async def _detect(self, image: Image) -> list[DetectedFace]:
        assert self._recognizer is not None
        try:
            return await self._recognizer.faces(image)
        except Exception as exc:
            raise VisionError(f"Finding faces failed: {exc}") from exc

    async def _people(self, image: Image) -> tuple[Person, ...]:
        assert self._book is not None
        faces = [f for f in await self._detect(image) if f.score >= self.rules.min_score]
        matched = await asyncio.to_thread(self._book.match, faces, threshold=self.rules.threshold)
        return tuple(matched)

    async def _capture(self, warmup_frames: int | None = None) -> Image:
        try:
            if warmup_frames is not None:
                return await self._camera.capture(warmup_frames=warmup_frames)
            return await self._camera.capture()
        except CameraError as exc:
            raise VisionError(str(exc)) from exc

    async def _ask(self, image: Image, question: str) -> str:
        messages = [
            Message(role=Role.SYSTEM, content=INSTRUCTION),
            Message(role=Role.USER, content=question, images=(image,)),
        ]
        del image  # only the request holds the frame now
        parts: list[str] = []
        try:
            stream = self._gateway.stream(self._role, messages, priority=Priority.FOREGROUND)
            async with aclosing(stream):
                async for chunk in stream:
                    if chunk.text:
                        parts.append(chunk.text)
        except TimeoutError:  # the look's own timeout, handled in `see`
            raise
        except Exception as exc:
            raise VisionError(f"The vision model failed: {exc}") from exc
        finally:
            messages.clear()
        return "".join(parts).strip()
