"""Vision stage 2 (ADR 0026): the face book, matching, `VisionService.who` and face
enrolment, names in `look`, the face tools, `lucy vision`, the insightface adapter's
worker plumbing and the settings. Everything is faked: no camera, no model."""

from __future__ import annotations

import argparse
import io
import json
import stat
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from lucy.adapters.faces_insightface import InsightFaceRecognizer, pack_present
from lucy.app import LucyApp
from lucy.clients import vision_cli
from lucy.core.config import (
    AdapterConfig,
    FacesSettings,
    LLMSettings,
    Settings,
    ToolsSettings,
    VisionSettings,
    WorkspaceSettings,
)
from lucy.core.events import ToolFinished
from lucy.core.gateway import LLMGateway
from lucy.core.ports import DetectedFace, FaceRecognizer, FaceRecognizerError
from lucy.core.types import FinishReason, Image, LLMChunk, Risk, ToolCall
from lucy.core.workspace import Workspace
from lucy.tools import build_tools
from lucy.tools.face_enrolment import EnrollFaceTool, ReplaceFaceTool
from lucy.tools.look import LookTool
from lucy.vision.faces import FaceBook, Person, describe_people, position_of
from lucy.vision.service import FaceRules, VisionError, VisionService
from tests.fakes import FakeLLM, adapter_for
from tests.test_app import make_registry, reply, run_turn

JPEG = Image(data=b"\xff\xd8\xff\xe0fake-jpeg", media_type="image/jpeg")
ROLE = AdapterConfig(adapter="fake", options={"model": "m"})


def vec(*head: float, dim: int = 8) -> tuple[float, ...]:
    v = np.zeros(dim, dtype=np.float32)
    v[: len(head)] = head
    v /= np.linalg.norm(v)
    return tuple(float(x) for x in v)


ASHA = vec(1, 0.1)
RAVI = vec(0, 1, 0.1)
STRANGER = vec(0, 0, 1)


def face(embedding: tuple[float, ...], *, left: float = 0.4, score: float = 0.9, size=0.2):
    return DetectedFace(box=(left, 0.3, left + size, 0.3 + size), score=score, embedding=embedding)


class FakeCamera:
    def __init__(self) -> None:
        self.captures = 0

    async def capture(self) -> Image:
        self.captures += 1
        return JPEG


class FakeRecognizer:
    """Returns the queued face lists in turn (the last one repeats)."""

    def __init__(self, *frames: list[DetectedFace]) -> None:
        self.frames = list(frames) or [[]]
        self.calls = 0
        self.closed = False

    async def faces(self, image: Image) -> list[DetectedFace]:
        assert image == JPEG
        result = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        return result

    async def aclose(self) -> None:
        self.closed = True


def make_service(
    tmp_path: Path,
    recognizer: Any,
    *,
    llm: FakeLLM | None = None,
    off: str | None = None,
    camera: FakeCamera | None = None,
) -> VisionService:
    return VisionService(
        gateway=LLMGateway({"vision": llm or FakeLLM([reply("A room.")])}),
        camera=camera or FakeCamera(),
        recognizer=recognizer,
        book=FaceBook(Workspace(tmp_path / "ws")),
        rules=FaceRules(enrol_frames=3, enrol_interval=0.0),
        faces_off=off,
    )


def faces_settings(tmp_path: Path, *, enabled: bool = True, tools=("look",)) -> Settings:
    return Settings(
        llm=LLMSettings(roles={"chat": ROLE, "vision": ROLE}),
        tools=ToolsSettings(enabled=list(tools)),
        vision=VisionSettings(
            enabled=True,
            camera=AdapterConfig(adapter="fake"),
            faces=FacesSettings(enabled=enabled, adapter="fake"),
        ),
        workspace=WorkspaceSettings(path=str(tmp_path / "ws")),
        memory=AdapterConfig(adapter="sqlite", options={"path": str(tmp_path / "db")}),
    )


# --- The face book and matching ------------------------------------------------------


def test_book_round_trips_and_is_owner_only(tmp_path: Path) -> None:
    book = FaceBook(Workspace(tmp_path / "ws"))
    assert book.people() == {}
    book.enroll("Asha", [ASHA, ASHA])
    book.enroll("Ravi", [RAVI])
    again = FaceBook(Workspace(tmp_path / "ws"))
    assert again.people() == {"Asha": 2, "Ravi": 1}
    assert stat.S_IMODE(again.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(again.path.parent.stat().st_mode) == 0o700
    assert again.find("asha") == "Asha"
    assert again.remove("ASHA") and not again.remove("Asha")
    assert FaceBook(Workspace(tmp_path / "ws")).people() == {"Ravi": 1}
    assert again.remove("Ravi") and FaceBook(Workspace(tmp_path / "ws")).people() == {}


def test_enrolling_again_replaces_and_keeps_no_photo(tmp_path: Path) -> None:
    book = FaceBook(Workspace(tmp_path / "ws"))
    book.enroll("Asha", [ASHA])
    book.enroll("asha", [ASHA, ASHA, ASHA])
    assert book.people() == {"asha": 3}
    assert [p.name for p in book.path.parent.iterdir()] == ["book.npz"]


def test_matching_best_per_person_threshold_and_one_face_per_name(tmp_path: Path) -> None:
    book = FaceBook(Workspace(tmp_path / "ws"))
    book.enroll("Asha", [vec(1, 0.5), ASHA])
    book.enroll("Ravi", [RAVI])
    scores = dict(book.scores(ASHA))
    assert scores["Asha"] == pytest.approx(1.0, abs=1e-5)
    people = book.match(
        [face(ASHA, left=0.0), face(STRANGER, left=0.7), face(vec(1, 0.15), left=0.4)],
        threshold=0.4,
    )
    assert [(p.name, p.position) for p in people] == [
        ("Asha", "left"),
        (None, "right"),
        (None, "centre"),  # Asha is already the better match on the left
    ]
    assert people[1].score is not None and people[1].score < 0.4


def test_match_with_empty_book_names_nobody(tmp_path: Path) -> None:
    people = FaceBook(Workspace(tmp_path / "ws")).match([face(ASHA)], threshold=0.4)
    assert people == [Person(name=None, score=None, position="centre")]


def test_positions_and_people_line() -> None:
    assert position_of(face(ASHA, left=0.0, size=0.1)) == "left"
    assert position_of(face(ASHA, left=0.8, size=0.1)) == "right"
    assert describe_people([]) == "People in view: nobody I can make out."
    line = describe_people([Person("Yogesh", 0.7, "left"), Person(None, 0.1, "right")])
    assert line == "People in view: Yogesh (left), one person I don't recognise."
    two = describe_people([Person(None, None, "left"), Person(None, None, "right")])
    assert two == "People in view: two people I don't recognise."
    alone = describe_people([Person(None, None, "centre")], enrolled=False)
    assert alone == (
        "People in view: one person. No faces are enrolled yet;"
        " the user can ask you to remember a face."
    )
    assert describe_people([], enrolled=False) == "People in view: nobody I can make out."


# --- VisionService ---------------------------------------------------------------------


async def test_see_fills_people_from_the_same_frame(tmp_path: Path) -> None:
    camera = FakeCamera()
    recognizer = FakeRecognizer([face(ASHA, left=0.0), face(STRANGER, left=0.7, score=0.3)])
    service = make_service(tmp_path, recognizer, camera=camera)
    assert service.book is not None
    service.book.enroll("Asha", [ASHA])
    sight = await service.see("who is here")
    assert sight.description == "A room." and camera.captures == 1 and recognizer.calls == 1
    assert [(p.name, p.position) for p in sight.people] == [("Asha", "left")]  # low score dropped


async def test_who_needs_no_scene_model(tmp_path: Path) -> None:
    llm = FakeLLM([])
    service = make_service(tmp_path, FakeRecognizer([face(RAVI)]), llm=llm)
    assert service.book is not None
    service.book.enroll("Ravi", [RAVI])
    (person,) = await service.who()
    assert person.name == "Ravi" and llm.calls == []
    assert service.last_who_s is not None


async def test_who_refuses_when_faces_are_off(tmp_path: Path) -> None:
    with pytest.raises(VisionError, match="lucy vision download"):
        await make_service(tmp_path, FakeRecognizer(), off="run `lucy vision download`").who()
    plain = VisionService(gateway=LLMGateway({"vision": FakeLLM([])}), camera=FakeCamera())
    with pytest.raises(VisionError, match="not enabled"):
        await plain.who()


async def test_failing_recogniser_keeps_the_description(tmp_path: Path) -> None:
    class Broken(FakeRecognizer):
        async def faces(self, image: Image) -> list[DetectedFace]:
            raise FaceRecognizerError("worker died")

    sight = await make_service(tmp_path, Broken()).see()
    assert sight.description == "A room." and sight.people == ()
    assert sight.faces_note is not None and "worker died" in sight.faces_note


async def test_enrol_face_takes_frames_and_stores_embeddings(tmp_path: Path) -> None:
    camera = FakeCamera()
    service = make_service(tmp_path, FakeRecognizer([face(ASHA)]), camera=camera)
    assert await service.enrol_face("Asha") == 3
    assert camera.captures == 3
    assert service.book is not None and service.book.people() == {"Asha": 3}


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        ([[face(ASHA), face(RAVI)]], "two faces"),
        ([[]], "couldn't see a face"),
        ([[face(ASHA, score=0.2)]], "couldn't see a face"),
        ([[face(ASHA, size=0.02)]], "too small"),
        ([[face(ASHA)], [face(ASHA)], [face(ASHA), face(RAVI), face(STRANGER)]], "three faces"),
    ],
)
async def test_enrol_face_refusals_store_nothing(
    tmp_path: Path, frames: list[list[DetectedFace]], message: str
) -> None:
    service = make_service(tmp_path, FakeRecognizer(*frames))
    with pytest.raises(VisionError, match=message):
        await service.enrol_face("Asha")
    assert service.book is not None and service.book.people() == {}


async def test_aclose_stops_the_recogniser(tmp_path: Path) -> None:
    recognizer = FakeRecognizer()
    await make_service(tmp_path, recognizer).aclose()
    assert recognizer.closed


# --- The look tool -------------------------------------------------------------------


async def test_look_starts_with_the_people_line(tmp_path: Path) -> None:
    settings = faces_settings(tmp_path)
    service = make_service(tmp_path, FakeRecognizer([face(ASHA, left=0.0), face(STRANGER)]))
    assert service.book is not None
    service.book.enroll("Yogesh", [ASHA])
    out = await LookTool(settings, vision_service=service).run({})
    first, second = out.content.splitlines()
    assert first == "People in view: Yogesh (left), one person I don't recognise."
    assert second == "Seen through the camera just now: A room."


async def test_look_with_an_empty_book_offers_enrolment(tmp_path: Path) -> None:
    service = make_service(tmp_path, FakeRecognizer([face(STRANGER)]))
    out = await LookTool(faces_settings(tmp_path), vision_service=service).run({})
    first, second = out.content.splitlines()
    assert first == (
        "People in view: one person. No faces are enrolled yet;"
        " the user can ask you to remember a face."
    )
    assert second == "Seen through the camera just now: A room."


async def test_look_with_an_empty_book_and_nobody_in_view(tmp_path: Path) -> None:
    service = make_service(tmp_path, FakeRecognizer([]))
    out = await LookTool(faces_settings(tmp_path), vision_service=service).run({})
    assert out.content.splitlines()[0] == "People in view: nobody I can make out."


async def test_look_with_an_enrolled_book_keeps_unknown_wording(tmp_path: Path) -> None:
    service = make_service(tmp_path, FakeRecognizer([face(STRANGER)]))
    assert service.book is not None
    service.book.enroll("Asha", [ASHA])
    out = await LookTool(faces_settings(tmp_path), vision_service=service).run({})
    assert out.content.splitlines()[0] == "People in view: one person I don't recognise."


async def test_look_without_face_models_has_no_names(tmp_path: Path) -> None:
    service = make_service(tmp_path, FakeRecognizer([face(ASHA)]), off="models missing")
    out = await LookTool(faces_settings(tmp_path), vision_service=service).run({})
    assert out.content == "Seen through the camera just now: A room."
    assert "enrolled" not in out.content


# --- The face tools --------------------------------------------------------------------


async def test_enroll_face_tool_enrols_and_refuses_known_names(tmp_path: Path) -> None:
    service = make_service(tmp_path, FakeRecognizer([face(ASHA)]))
    tool = EnrollFaceTool(faces_settings(tmp_path), vision_service=service)
    assert tool.risk is Risk.ACT
    out = await tool.run({"name": "asha"})
    assert not out.is_error and "Asha's face is enrolled" in out.content
    again = await tool.run({"name": "ASHA"})
    assert again.is_error and "replace_face" in again.content


@pytest.mark.parametrize("bad", ["my face", "me", "user", "Asha face", "../x", ""])
async def test_face_names_follow_the_speaker_rules(tmp_path: Path, bad: str) -> None:
    service = make_service(tmp_path, FakeRecognizer([face(ASHA)]))
    out = await EnrollFaceTool(faces_settings(tmp_path), vision_service=service).run({"name": bad})
    assert out.is_error
    assert service.book is not None and service.book.people() == {}


async def test_enroll_face_tool_reports_what_went_wrong(tmp_path: Path) -> None:
    service = make_service(tmp_path, FakeRecognizer([face(ASHA), face(RAVI)]))
    out = await EnrollFaceTool(faces_settings(tmp_path), vision_service=service).run(
        {"name": "Asha"}
    )
    assert out.is_error and "I could see two faces" in out.content
    assert "Nothing was stored" in out.content


async def test_face_tools_refuse_when_off_or_without_models(tmp_path: Path) -> None:
    off = await EnrollFaceTool(faces_settings(tmp_path, enabled=False)).run({"name": "Asha"})
    assert off.is_error and "[vision.faces]" in off.content
    service = make_service(tmp_path, FakeRecognizer(), off="run `lucy vision download`")
    missing = await EnrollFaceTool(faces_settings(tmp_path), vision_service=service).run(
        {"name": "Asha"}
    )
    assert missing.is_error and "lucy vision download" in missing.content


async def test_replace_face_previews_then_replaces(tmp_path: Path) -> None:
    service = make_service(tmp_path, FakeRecognizer([face(RAVI)]))
    tool = ReplaceFaceTool(faces_settings(tmp_path), vision_service=service)
    assert tool.risk is Risk.DANGEROUS
    assert (await tool.preview({"name": "Asha"})).is_error  # not enrolled
    assert service.book is not None
    service.book.enroll("Asha", [ASHA])
    preview = await tool.preview({"name": "asha"})
    assert not preview.is_error and preview.content == "Replace the enrolled face for Asha?"
    out = await tool.run({"name": "asha"})
    assert not out.is_error and "replaced" in out.content
    assert service.book.people() == {"Asha": 3}
    assert service.book.scores(RAVI)[0][1] == pytest.approx(1.0, abs=1e-5)


def test_face_tools_are_built() -> None:
    settings = Settings(
        llm=LLMSettings(roles={"chat": ROLE}),
        tools=ToolsSettings(enabled=["enroll_face", "replace_face"]),
    )
    assert {t.name for t in build_tools(settings)} == {"enroll_face", "replace_face"}


# --- In the app ------------------------------------------------------------------------


async def test_app_wires_faces_into_look(tmp_path: Path) -> None:
    call = ToolCall(id="c1", name="look", arguments={})
    llm = FakeLLM(
        [
            [LLMChunk(tool_call=call), LLMChunk(finish_reason=FinishReason.TOOL_CALLS)],
            reply("A person at a desk."),
            reply("I see you, Asha."),
        ]
    )
    recognizer = FakeRecognizer([face(ASHA)])
    registry = make_registry(llm)
    registry.register("camera", "fake")(adapter_for(FakeCamera()))
    registry.register("faces", "fake")(adapter_for(recognizer))
    settings = faces_settings(tmp_path)
    FaceBook(Workspace(settings.workspace.path)).enroll("Asha", [ASHA])
    async with LucyApp(settings, registry=registry) as app:
        events = await run_turn(app, "Who do you see?")
    (finished,) = [e for e in events if isinstance(e, ToolFinished)]
    assert finished.result.content.startswith("People in view: Asha (centre).")
    assert recognizer.closed


async def test_app_with_missing_models_keeps_look_working(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    call = ToolCall(id="c1", name="look", arguments={})
    llm = FakeLLM(
        [
            [LLMChunk(tool_call=call), LLMChunk(finish_reason=FinishReason.TOOL_CALLS)],
            reply("A desk."),
            reply("A desk."),
        ]
    )
    registry = make_registry(llm)
    registry.register("camera", "fake")(adapter_for(FakeCamera()))
    registry.register("faces", "fake")(adapter_for(InsightFaceRecognizer(models=tmp_path / "none")))
    async with LucyApp(faces_settings(tmp_path), registry=registry) as app:
        events = await run_turn(app, "What do you see?")
    (finished,) = [e for e in events if isinstance(e, ToolFinished)]
    assert finished.result.content == "Seen through the camera just now: A desk."
    notices = [r.message for r in caplog.records if "face matching is off" in r.message]
    assert len(notices) == 1 and "lucy vision download" in notices[0]


# --- lucy vision -----------------------------------------------------------------------


def run_cli(settings: Settings, *argv: str, **kw: Any) -> tuple[int, str, str]:
    action, *rest = argv
    ns = argparse.Namespace(vision_action=action, pack=None, name=rest[0] if rest else None)
    out, err = io.StringIO(), io.StringIO()
    code = vision_cli.run(settings, ns, out=out, err=err, **kw)
    return code, out.getvalue(), err.getvalue()


def test_cli_download_prints_the_licence(tmp_path: Path) -> None:
    seen: list[tuple[Path, str]] = []

    def fake_download(models: Path, pack: str) -> Path:
        seen.append((models, pack))
        return models / pack

    code, out, err = run_cli(faces_settings(tmp_path), "download", download=fake_download)
    assert code == 0 and seen == [(Path("data/models/faces"), "buffalo_l")]
    assert "non-commercial research" in err
    assert out.strip() == "data/models/faces/buffalo_l"


def test_cli_list_and_forget(tmp_path: Path) -> None:
    settings = faces_settings(tmp_path)
    assert run_cli(settings, "list")[1] == "No faces enrolled.\n"
    FaceBook(Workspace(settings.workspace.path)).enroll("Asha", [ASHA, ASHA])
    assert run_cli(settings, "list")[1] == "Asha\t2 picture(s)\n"
    code, _, err = run_cli(settings, "forget", "Ravi")
    assert code == 1 and "no face is enrolled for Ravi" in err
    assert run_cli(settings, "forget", "asha")[:2] == (0, "Forgot asha's face.\n")
    assert FaceBook(Workspace(settings.workspace.path)).people() == {}


# --- The adapter (no models, no network; a fake worker) --------------------------------


def test_adapter_options_and_missing_models(tmp_path: Path) -> None:
    rec = InsightFaceRecognizer(models=tmp_path)
    assert isinstance(rec, FaceRecognizer) and not rec.present
    assert rec.pack == "buffalo_l" and rec.provider == "coreml"
    for bad in ({"pack": "antelope"}, {"provider": "cuda"}, {"det_size": 100}):
        with pytest.raises(ValueError):
            InsightFaceRecognizer(models=tmp_path, **bad)
    (tmp_path / "buffalo_s").mkdir()
    (tmp_path / "buffalo_s" / "det_500m.onnx").write_bytes(b"x")
    assert not pack_present(tmp_path, "buffalo_s")
    (tmp_path / "buffalo_s" / "w600k_mbf.onnx").write_bytes(b"x")
    assert pack_present(tmp_path, "buffalo_s")


async def test_adapter_without_models_raises_naming_the_command(tmp_path: Path) -> None:
    with pytest.raises(FaceRecognizerError, match="lucy vision download"):
        await InsightFaceRecognizer(models=tmp_path).faces(JPEG)


FAKE_WORKER = textwrap.dedent(
    """
    import json, sys
    print(json.dumps({"ready": True, "load_s": 0.0, "providers": ["CPU"]}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        face = {"box": [0.1, 0.2, 0.3, 0.4], "score": 0.9, "embedding": [0.6, 0.8]}
        print(json.dumps({"id": req["id"], "faces": [face], "seconds": 0.0}), flush=True)
    """
)


async def test_adapter_speaks_to_its_worker(tmp_path: Path) -> None:
    script = tmp_path / "worker.py"
    script.write_text(FAKE_WORKER)
    rec = InsightFaceRecognizer(models=tmp_path, command=[sys.executable, "-I", str(script)])
    try:
        (found,) = await rec.faces(JPEG)
        assert found == DetectedFace(box=(0.1, 0.2, 0.3, 0.4), score=0.9, embedding=(0.6, 0.8))
        assert len(await rec.faces(JPEG)) == 1
    finally:
        await rec.aclose()
    assert json.dumps(rec.last_ready)  # the ready line was read


# --- Settings ----------------------------------------------------------------------------


def test_faces_settings_defaults_and_toml(tmp_path: Path) -> None:
    from lucy.core.config import load_settings

    faces = FacesSettings()
    assert not faces.enabled and faces.pack == "buffalo_l" and faces.threshold == 0.40
    with pytest.raises(ValidationError):
        FacesSettings(pack="antelope")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        FacesSettings(threshold=1.5)
    path = tmp_path / "lucy.toml"
    path.write_text(
        "[llm.roles]\n"
        'chat = { adapter = "ollama", model = "m" }\n'
        'vision = { adapter = "ollama", model = "v" }\n'
        "[tools]\n"
        'enabled = ["look", "enroll_face", "replace_face"]\n'
        "[vision]\nenabled = true\n"
        "[vision.faces]\n"
        'enabled = true\npack = "buffalo_s"\nthreshold = 0.35\n'
        'options = { provider = "cpu" }\n'
    )
    settings = load_settings(path)
    assert settings.vision.faces.pack == "buffalo_s"
    assert settings.vision.faces.options == {"provider": "cpu"}


def test_example_config_faces_lines_are_valid() -> None:
    example = (Path(__file__).resolve().parent.parent / "lucy.example.toml").read_text()
    lines = example.splitlines()
    start = lines.index("# [vision.faces]")
    block = []
    for line in lines[start + 1 :]:
        if not line.startswith("#"):
            break
        if "=" in line and line.startswith("# ") and not line.startswith("#  "):
            block.append(line[2:].split("#")[0].strip())
    import tomllib

    options = tomllib.loads("\n".join(block))
    faces = FacesSettings(**options)
    assert faces.enabled and faces.options == {"provider": "coreml"}
    InsightFaceRecognizer(pack=faces.pack, **faces.options)
