"""Vision stage 3 (ADR 0026): the glance at the start of each voice turn. Everything is
faked: no camera is opened and no frame is captured."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from lucy.app import LucyApp
from lucy.core.config import FacesSettings, Settings, VisionSettings
from lucy.core.context import ContextBuilder, estimate_tokens, seen_line
from lucy.core.events import Seen, SeenFace, UserMessage, event_from_json
from lucy.core.gateway import LLMGateway
from lucy.core.ports import CameraError
from lucy.core.types import Image, MemoryContext, Message, Role
from lucy.core.workspace import Workspace
from lucy.vision.faces import FaceBook
from lucy.vision.glance import Glancer, build_glancer
from lucy.vision.service import VisionService
from tests.fakes import FakeLLM, adapter_for
from tests.test_app import make_registry, run_turn
from tests.test_faces import ASHA, JPEG, RAVI, STRANGER, FakeRecognizer, face, faces_settings
from tests.test_voice_session import Harness, reply
from tests.voice_fakes import concat, silence, tone

ONE_TURN = [concat(silence(0.5), tone(1.0), silence(3.0))]
TWO_TURNS = [concat(silence(0.5), tone(1.0), silence(4.0), tone(0.8), silence(3.0))]


class GlanceCamera:
    """Counts captures; can wait for a condition, or fail."""

    def __init__(self, *, gate: Any = None, error: Exception | None = None) -> None:
        self.captures = 0
        self.warmups: list[int | None] = []
        self.gate = gate
        self.error = error

    async def capture(self, *, warmup_frames: int | None = None) -> Image:
        self.captures += 1
        self.warmups.append(warmup_frames)
        if self.error is not None:
            raise self.error
        if self.gate is not None and self.captures == 1:
            while not self.gate():
                await asyncio.sleep(0.005)
        return JPEG


def glancer(
    tmp_path: Path, camera: GlanceCamera, *frames: Any, wait_ms: int = 2000
) -> tuple[Glancer, FakeRecognizer]:
    recognizer = FakeRecognizer(*frames)
    book = FaceBook(Workspace(tmp_path / "ws"))
    book.enroll("Asha", [ASHA])
    book.enroll("Ravi", [RAVI])
    service = VisionService(
        gateway=LLMGateway({"vision": FakeLLM([])}), camera=camera, recognizer=recognizer, book=book
    )
    return Glancer(service, wait_ms=wait_ms, warmup_frames=2), recognizer


def turn_block(call: list[Message]) -> str:
    """The per-turn system message right after the latest user message, or ""."""
    latest = max(i for i, m in enumerate(call) if m.role is Role.USER)
    after = call[latest + 1 : latest + 2]
    return after[0].content if after and after[0].role is Role.SYSTEM else ""


# --- In the voice session ---------------------------------------------------------------


async def test_glance_in_time_reaches_the_message_the_prompt_and_the_timings(
    tmp_path: Path,
) -> None:
    camera = GlanceCamera()
    g, _ = glancer(tmp_path, camera, [face(ASHA, left=0.1), face(STRANGER, left=0.7)])
    async with Harness(ONE_TURN, [reply("Hello Asha.")], ["hi lucy"], glancer=g) as h:
        await h.run()
    (user,) = h.events(UserMessage)
    assert user.seen is not None and user.seen.names == ["Asha"] and user.seen.unknown == 1
    assert user.seen.people[0].score == pytest.approx(1.0, abs=0.01)
    assert camera.captures == 1 and camera.warmups == [2]
    assert turn_block(h.llm.calls[0]).startswith(
        "In view of the camera as the user spoke: Asha and one person you don't recognise."
    )
    timing = h.session.timings[0]
    assert timing.seen == ["Asha"] and timing.seen_unknown == 1
    assert timing.to_json()["since_speech_end"]["seen_ready"] is not None


async def test_late_glance_is_dropped_and_never_reaches_a_later_turn(tmp_path: Path) -> None:
    holder: dict[str, Harness] = {}

    def first_turn_started() -> bool:  # the first capture finishes only after turn 1 began
        h = holder.get("h")
        return h is not None and len(h.events(UserMessage)) >= 1

    camera = GlanceCamera(gate=first_turn_started)
    g, recognizer = glancer(tmp_path, camera, [face(ASHA)], [face(RAVI)], wait_ms=50)
    scripts = [reply("First."), reply("Second.")]
    async with Harness(TWO_TURNS, scripts, ["one", "two"], glancer=g) as h:
        holder["h"] = h
        await h.run()
    first, second = h.events(UserMessage)
    assert first.seen is None and turn_block(h.llm.calls[0]) == ""
    assert recognizer.calls == 2  # the late glance did finish, and was dropped
    assert second.seen is not None and second.seen.names == ["Ravi"]
    assert h.session.timings[0].seen is None


async def test_failing_camera_never_breaks_a_turn_and_is_logged_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    camera = GlanceCamera(error=CameraError("The camera did not open."))
    g, _ = glancer(tmp_path, camera, [face(ASHA)])
    scripts = [reply("First."), reply("Second.")]
    with caplog.at_level(logging.WARNING):
        async with Harness(TWO_TURNS, scripts, ["one", "two"], glancer=g) as h:
            await h.run()
    assert [u.seen for u in h.events(UserMessage)] == [None, None]
    assert [t.outcome for t in h.session.timings] == ["completed", "completed"]
    assert camera.captures == 2
    assert len([r for r in caplog.records if "glance failed" in r.message]) == 1


async def test_failing_recogniser_never_breaks_a_turn(tmp_path: Path) -> None:
    class Broken(FakeRecognizer):
        async def faces(self, image: Image) -> list[Any]:
            raise RuntimeError("worker died")

    camera = GlanceCamera()
    g, _ = glancer(tmp_path, camera)
    g._service._recognizer = Broken()  # type: ignore[attr-defined]
    async with Harness(ONE_TURN, [reply("Hi.")], ["hi"], glancer=g) as h:
        await h.run()
    (user,) = h.events(UserMessage)
    assert user.seen is None and h.session.timings[0].outcome == "completed"


async def test_nobody_in_view_is_recorded_but_not_told(tmp_path: Path) -> None:
    g, _ = glancer(tmp_path, GlanceCamera(), [])
    async with Harness(ONE_TURN, [reply("Hi.")], ["hi"], glancer=g) as h:
        await h.run()
    (user,) = h.events(UserMessage)
    assert user.seen == Seen()
    assert turn_block(h.llm.calls[0]) == ""


async def test_no_glancer_means_no_glance(tmp_path: Path) -> None:
    async with Harness(ONE_TURN, [reply("Hi.")], ["hi"]) as h:
        await h.run()
    (user,) = h.events(UserMessage)
    assert user.seen is None and h.session.timings[0].seen is None


# --- Settings and wiring --------------------------------------------------------------------


def glance_settings(tmp_path: Path, *, glance: bool = True, **vision: Any) -> Settings:
    base = faces_settings(tmp_path)
    options = {**base.vision.model_dump(), "glance": glance, **vision}
    return base.model_copy(update={"vision": VisionSettings.model_validate(options)})


def test_glance_is_off_by_default() -> None:
    assert VisionSettings().glance is False
    assert VisionSettings().glance_wait_ms == 400
    assert VisionSettings().glance_warmup_frames == 2


def test_build_glancer_says_once_why_it_cannot_glance(tmp_path: Path) -> None:
    notices: list[str] = []
    assert build_glancer(glance_settings(tmp_path, glance=False), None, notices.append) is None
    assert notices == []
    off = glance_settings(tmp_path, enabled=False)
    assert build_glancer(off, None, notices.append) is None
    no_faces = glance_settings(tmp_path, faces=FacesSettings(enabled=False))
    assert build_glancer(no_faces, None, notices.append) is None
    g, _ = glancer(tmp_path, GlanceCamera())
    service = g._service  # type: ignore[attr-defined]
    service.faces_off = "face matching is off: run `lucy vision download`"
    assert build_glancer(glance_settings(tmp_path), service, notices.append) is None
    assert len(notices) == 3
    assert "[vision] is not enabled" in notices[0]
    assert "[vision.faces]" in notices[1]
    assert "lucy vision download" in notices[2]
    service.faces_off = None
    built = build_glancer(glance_settings(tmp_path, glance_wait_ms=250), service, notices.append)
    assert built is not None and built.wait_s == 0.25 and len(notices) == 3


async def test_chat_never_glances(tmp_path: Path) -> None:
    camera = GlanceCamera()
    registry = make_registry(FakeLLM([reply("Hello.")]))
    registry.register("camera", "fake")(adapter_for(camera))
    registry.register("faces", "fake")(adapter_for(FakeRecognizer([face(ASHA)])))
    settings = glance_settings(tmp_path, glance=True)
    settings = settings.model_copy(
        update={"tools": settings.tools.model_copy(update={"enabled": []})}
    )
    async with LucyApp(settings, registry=registry) as app:
        assert app.vision is not None  # built for the voice clients' glance
        await run_turn(app, "hello")
    assert camera.captures == 0


# --- The event --------------------------------------------------------------------------


def test_user_message_round_trips_with_and_without_seen() -> None:
    plain = UserMessage(id="1", session_id="s", text="hi", source="voice")
    seen = plain.model_copy(
        update={"seen": Seen(people=(SeenFace(name="Asha", score=0.71),), unknown=2)}
    )
    assert event_from_json(plain.model_dump_json()) == plain
    assert event_from_json(seen.model_dump_json()) == seen
    old = (
        '{"type":"user_message","schema_version":1,"id":"x","ts":"2026-01-01T00:00:00Z",'
        '"session_id":"cli","turn_id":null,"text":"hi","source":"voice"}'
    )
    event = event_from_json(old)
    assert isinstance(event, UserMessage) and event.seen is None


# --- What the model is told -----------------------------------------------------------------


def test_seen_line() -> None:
    def seen(*names: str, unknown: int = 0) -> Seen:
        return Seen(people=tuple(SeenFace(name=n, score=0.8) for n in names), unknown=unknown)

    assert seen_line(None) == ""
    assert seen_line(Seen()) == ""
    assert seen_line(seen("Yogesh")).startswith("In view of the camera as the user spoke: Yogesh.")
    assert seen_line(seen("Yogesh", unknown=1)).startswith(
        "In view of the camera as the user spoke: Yogesh and one person you don't recognise."
    )
    assert seen_line(seen("Asha", "Ravi", unknown=2)).startswith(
        "In view of the camera as the user spoke: Asha, Ravi and two people you don't recognise."
    )
    assert seen_line(seen(unknown=1)).startswith(
        "In view of the camera as the user spoke: one person you don't recognise."
    )
    assert "no proof of who is speaking" in seen_line(seen("Asha"))


NOW = datetime(2026, 11, 17, 9, 0, tzinfo=UTC)
EARLIER = [
    Message(role=Role.USER, content="hi"),
    Message(role=Role.ASSISTANT, content="hello"),
]


def test_the_line_goes_next_to_the_date_line() -> None:
    builder = ContextBuilder("You are Lucy.", clock=lambda: NOW)
    line = seen_line(Seen(people=(SeenFace(name="Asha", score=0.8),)))
    built = builder.build(
        memory=MemoryContext(statements=["Likes tea."]),
        history=[*EARLIER, Message(role=Role.USER, content="morning")],
        context_window=8192,
        turn_note=line,
    )
    assert built.messages[-1].content.startswith(
        "Today is Tuesday, 17 November 2026, 09:00 (UTC+00:00).\n" + line + "\n\n## Things"
    )
    system = ContextBuilder("You are Lucy.", clock=lambda: NOW, layout="system")
    first = system.build(
        memory=MemoryContext(), history=[*EARLIER], context_window=8192, turn_note=line
    )
    assert line in first.messages[0].content


def test_prefix_is_identical_with_and_without_a_glance_and_matches_the_pre_fill() -> None:
    builder = ContextBuilder("You are Lucy.", clock=lambda: NOW)
    memory = MemoryContext(profile="Lives in Pune.", instructions=["Be brief."])
    latest = Message(role=Role.USER, content="who's there?")
    line = seen_line(Seen(people=(SeenFace(name="Asha", score=0.8),), unknown=1))

    def build(user: Message, note: str = "") -> list[Message]:
        built = builder.build(
            memory=memory, history=[*EARLIER, user], context_window=8192, turn_note=note
        )
        return builder.prefix_messages(built)

    pre_fill = build(Message(role=Role.USER, content=""))  # the residency placeholder
    assert build(latest, line) == build(latest) == pre_fill
    assert estimate_tokens(line) < 60  # per turn only, after the cached prefix
