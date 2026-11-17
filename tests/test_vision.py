"""Vision stage 1 (ADR 0026): images in messages, the camera adapter (no camera is
opened), `VisionService`, the `look` tool, its tool group and its settings."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from lucy.adapters import camera_opencv
from lucy.adapters.camera_opencv import OpenCVCamera, encode_frame
from lucy.adapters.llm_openai_compat import OpenAICompatLLM, _message_to_wire
from lucy.app import LucyApp, build_selector
from lucy.core.config import AdapterConfig, LLMSettings, Settings, ToolsSettings, VisionSettings
from lucy.core.events import ToolFinished
from lucy.core.gateway import LLMGateway
from lucy.core.ports import Camera, CameraError
from lucy.core.types import FinishReason, Image, LLMChunk, Message, Role, ToolCall
from lucy.tools import GROUPS, build_tools
from lucy.tools.look import LookTool
from lucy.vision.service import GENERAL_QUESTION, INSTRUCTION, VisionError, VisionService
from tests.fakes import FakeLLM, Pause, adapter_for
from tests.test_app import make_registry, reply, run_turn

JPEG = Image(data=b"\xff\xd8\xff\xe0fake-jpeg", media_type="image/jpeg")
VISION_ROLE = AdapterConfig(adapter="fake", options={"model": "m"})


class FakeCamera:
    def __init__(self, error: Exception | None = None) -> None:
        self.captures = 0
        self._error = error

    async def capture(self) -> Image:
        self.captures += 1
        if self._error is not None:
            raise self._error
        return JPEG


def settings_with_vision(*, enabled: bool = True, tools: tuple[str, ...] = ("look",)) -> Settings:
    return Settings(
        llm=LLMSettings(roles={"chat": VISION_ROLE, "vision": VISION_ROLE}),
        tools=ToolsSettings(enabled=list(tools)),
        vision=VisionSettings(enabled=enabled, camera=AdapterConfig(adapter="fake")),
    )


# --- Messages on the wire --------------------------------------------------------


def test_message_without_images_goes_over_the_wire_as_before() -> None:
    message = Message(role=Role.USER, content="hello")
    assert _message_to_wire(message) == {"role": "user", "content": "hello"}
    assert message.images == ()


def test_message_with_images_becomes_content_parts() -> None:
    message = Message(role=Role.USER, content="what is this?", images=(JPEG,))
    wire = _message_to_wire(message)
    assert wire["role"] == "user"
    text, image = wire["content"]
    assert text == {"type": "text", "text": "what is this?"}
    assert image["type"] == "image_url"
    url = image["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,/9j/")


def test_request_body_keeps_plain_messages_unchanged() -> None:
    llm = OpenAICompatLLM(model="m", base_url="http://127.0.0.1:1/v1")
    plain = [Message(role=Role.SYSTEM, content="s"), Message(role=Role.USER, content="u")]
    seeing = [*plain[:1], Message(role=Role.USER, content="u", images=(JPEG,))]
    assert llm._request_body(plain, [])["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    body = llm._request_body(seeing, [])
    assert body["messages"][0] == {"role": "system", "content": "s"}
    assert isinstance(body["messages"][1]["content"], list)


def test_images_are_never_serialised() -> None:
    message = Message(role=Role.USER, content="look", images=(JPEG,))
    assert "images" not in message.model_dump()
    assert "fake-jpeg" not in message.model_dump_json()
    assert "fake-jpeg" not in repr(message)
    restored = Message.model_validate_json(message.model_dump_json())
    assert restored.images == () and restored.content == "look"


# --- The camera adapter (never opens a camera) -------------------------------------


def test_camera_options_are_checked() -> None:
    camera = OpenCVCamera()
    assert (camera.device, camera.max_side) == (0, 1024)
    assert isinstance(camera, Camera)
    for bad in ({"device": -1}, {"max_side": 10}, {"warmup_frames": 99}, {"jpeg_quality": 0}):
        with pytest.raises(ValueError):
            OpenCVCamera(**bad)


def test_encode_frame_downscales_and_encodes_jpeg() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.circle(frame, (640, 360), 200, (0, 0, 255), -1)
    image = encode_frame(cv2, frame, max_side=640, quality=80)
    assert image.media_type == "image/jpeg" and image.data[:3] == b"\xff\xd8\xff"
    decoded = cv2.imdecode(np.frombuffer(image.data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (360, 640)
    small = encode_frame(cv2, np.zeros((100, 50, 3), dtype=np.uint8), max_side=640, quality=80)
    assert cv2.imdecode(np.frombuffer(small.data, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (
        100,
        50,
    )


async def test_missing_opencv_says_how_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(CameraError, match="uv sync --extra vision"):
        await OpenCVCamera().capture()


class _ClosedCapture:
    released = False

    def __init__(self, device: int) -> None:
        self.device = device

    def isOpened(self) -> bool:  # noqa: N802 (OpenCV's name)
        return False

    def release(self) -> None:
        _ClosedCapture.released = True


async def test_camera_that_will_not_open_explains_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cv2 = type("cv2", (), {"VideoCapture": _ClosedCapture})
    monkeypatch.setattr(camera_opencv, "_import_cv2", lambda: fake_cv2)
    with pytest.raises(CameraError, match="Privacy & Security > Camera"):
        await OpenCVCamera().capture()
    assert _ClosedCapture.released


# --- VisionService -------------------------------------------------------------------


def service(llm: FakeLLM, camera: Any, **options: Any) -> VisionService:
    return VisionService(gateway=LLMGateway({"vision": llm}), camera=camera, **options)


async def test_look_sends_one_frame_with_the_fixed_instruction() -> None:
    llm = FakeLLM([reply("A red mug on a desk.")])
    camera = FakeCamera()
    assert await service(llm, camera).look("  what is   this? ") == "A red mug on a desk."
    assert camera.captures == 1
    (messages,) = llm.calls
    system, user = messages
    assert system.role is Role.SYSTEM and system.content == INSTRUCTION
    assert user.role is Role.USER and user.content == "what is this?"
    assert user.images == (JPEG,)
    assert llm.tools_seen == [[]]


async def test_see_defaults_to_a_general_question_and_no_people() -> None:
    llm = FakeLLM([reply("A room.")])
    sight = await service(llm, FakeCamera()).see()
    assert sight.description == "A room." and sight.people == ()
    assert llm.calls[0][1].content == GENERAL_QUESTION


async def test_camera_error_becomes_vision_error() -> None:
    camera = FakeCamera(CameraError("grant the terminal camera permission"))
    with pytest.raises(VisionError, match="camera permission"):
        await service(FakeLLM([]), camera).look("x")


async def test_empty_answer_and_model_failure_are_vision_errors() -> None:
    with pytest.raises(VisionError, match="no answer"):
        await service(FakeLLM([reply("  ")]), FakeCamera()).look("x")
    failing = FakeLLM([[]])
    failing.stream = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[method-assign]
    with pytest.raises(VisionError, match="boom"):
        await service(failing, FakeCamera()).look("x")


async def test_look_times_out() -> None:
    pause = Pause()
    llm = FakeLLM([[pause]])
    with pytest.raises(VisionError, match="longer than"):
        await service(llm, FakeCamera(), timeout=0.05).look("x")
    pause.release.set()
    await asyncio.sleep(0)


# --- The look tool -------------------------------------------------------------------


async def test_look_tool_refuses_when_vision_is_off() -> None:
    tool = LookTool(settings_with_vision(enabled=False))
    out = await tool.run({})
    assert out.is_error and "[vision]" in out.content


async def test_look_tool_refuses_without_the_role() -> None:
    settings = settings_with_vision(enabled=False)
    settings = settings.model_copy(update={"vision": VisionSettings(enabled=True, role="eyes")})
    out = await LookTool(settings).run({})
    assert out.is_error and "'eyes'" in out.content


async def test_look_tool_refuses_without_a_service() -> None:
    out = await LookTool(settings_with_vision()).run({})
    assert out.is_error and "not available" in out.content


async def test_look_tool_answers_and_reports_errors() -> None:
    llm = FakeLLM([reply("A blue pen."), reply("")])
    tool = LookTool(settings_with_vision(), vision_service=service(llm, FakeCamera()))
    out = await tool.run({"question": "what am I holding"})
    assert not out.is_error and "A blue pen." in out.content
    assert llm.calls[0][1].content == "what am I holding"
    failed = await tool.run({})
    assert failed.is_error and failed.content.startswith("Could not look:")
    assert llm.calls[1][1].content == "Describe what you see."


def test_look_is_built_and_is_read_tier() -> None:
    (tool,) = build_tools(settings_with_vision())
    assert isinstance(tool, LookTool) and tool.risk == "read"


# --- The tool group ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Can you see me?",
        "What am I holding?",
        "what is this",
        "take a look at this",
        "Use the camera",
        "dekho isko",
        "kya hai ye",
        "mujhe dikh nahi raha, tum dekh ke batao",
    ],
)
def test_vision_group_switches_on(text: str) -> None:
    selector = build_selector(
        settings_with_vision(tools=("time", "look")),
        build_tools(settings_with_vision(tools=("time", "look"))),
    )
    assert selector is not None
    assert "vision" in selector.matched(text)


@pytest.mark.parametrize(
    "text",
    ["Let me see.", "I see, thanks", "look up the news", "see you tomorrow", "Picture this"],
)
def test_vision_group_stays_off_for_everyday_words(text: str) -> None:
    tools = build_tools(settings_with_vision(tools=("time", "look")))
    selector = build_selector(settings_with_vision(tools=("time", "look")), tools)
    assert selector is not None
    assert "vision" not in selector.matched(text)


def test_vision_group_holds_the_look_tool() -> None:
    (group,) = [g for g in GROUPS if g.name == "vision"]
    assert group.tools == ("look", "enroll_face", "replace_face")


# --- Settings ------------------------------------------------------------------------


def test_vision_settings_default_off() -> None:
    settings = Settings(llm=LLMSettings(roles={"chat": VISION_ROLE}))
    assert not settings.vision.enabled and settings.vision.role == "vision"
    assert settings.vision.camera.adapter == "opencv"


def test_vision_enabled_needs_its_role() -> None:
    with pytest.raises(ValidationError, match="no 'vision' role"):
        Settings(
            llm=LLMSettings(roles={"chat": VISION_ROLE}),
            vision=VisionSettings(enabled=True),
        )


def test_vision_settings_from_toml(tmp_path: Path) -> None:
    from lucy.core.config import load_settings

    path = tmp_path / "lucy.toml"
    path.write_text(
        "[llm.roles]\n"
        'chat = { adapter = "ollama", model = "m" }\n'
        'vision = { adapter = "ollama", model = "lfm3-vl", extra = { max_tokens = 300 } }\n'
        "[tools]\n"
        'enabled = ["look"]\n'
        "[vision]\n"
        "enabled = true\n"
        'camera = { adapter = "opencv", device = 1, max_side = 768, warmup_frames = 3 }\n'
    )
    settings = load_settings(path)
    assert settings.vision.camera.options == {"device": 1, "max_side": 768, "warmup_frames": 3}
    with pytest.raises(ValidationError):
        VisionSettings(enabled=True, unknown=1)  # type: ignore[call-arg]


# --- In the app ----------------------------------------------------------------------


async def test_app_attaches_the_service_and_a_look_runs(tmp_path: Path) -> None:
    call = ToolCall(id="c1", name="look", arguments={"question": "what is the user holding"})
    llm = FakeLLM(
        [
            [LLMChunk(tool_call=call), LLMChunk(finish_reason=FinishReason.TOOL_CALLS)],
            reply("A blue pen."),  # the vision role (same fake adapter)
            reply("You're holding a blue pen."),
        ]
    )
    camera = FakeCamera()
    registry = make_registry(llm)
    registry.register("camera", "fake")(adapter_for(camera))
    settings = settings_with_vision().model_copy(
        update={"memory": AdapterConfig(adapter="sqlite", options={"path": str(tmp_path / "db")})}
    )
    async with LucyApp(settings, registry=registry) as app:
        events = await run_turn(app, "What am I holding?")
    (finished,) = [e for e in events if isinstance(e, ToolFinished)]
    assert not finished.result.is_error and "A blue pen." in finished.result.content
    assert camera.captures == 1
    assert llm.calls[1][1].images == (JPEG,)
    assert all(not m.images for m in llm.calls[2])


def test_example_config_vision_lines_are_valid(tmp_path: Path) -> None:
    from lucy.core.config import load_settings

    example = (Path(__file__).resolve().parent.parent / "lucy.example.toml").read_text()
    lines = example.splitlines()
    role = next(line[2:] for line in lines if line.startswith("# vision     = {"))
    start = lines.index("# [vision]")
    block = [line[2:] for line in lines[start:] if line.startswith("# ") and "=" in line][:4]
    path = tmp_path / "lucy.toml"
    path.write_text(
        '[llm.roles]\nchat = { adapter = "ollama", model = "m" }\n'
        + role
        + '\n[tools]\nenabled = ["look"]\n[vision]\n'
        + "\n".join(block)
        + "\n"
    )
    settings = load_settings(path)
    assert settings.vision.enabled and settings.llm.roles["vision"].options["model"] == "lfm3-vl"
    assert settings.vision.camera.options["max_side"] == 1024
    OpenCVCamera(**settings.vision.camera.options)
