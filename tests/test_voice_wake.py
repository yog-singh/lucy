"""The wake word: detector wrapper, model download, wake stage, phrase stripping and
`lucy listen` in file mode. Fakes only: no models, devices or network."""

import argparse
import io
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from lucy.clients import cli
from lucy.clients.voice_cli import download_wake_models, run_listen
from lucy.core.config import AdapterConfig, LLMSettings, Settings, WakeSettings
from lucy.core.events import UserMessage
from lucy.core.registry import Registry
from lucy.core.types import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, AudioChunk
from lucy.memory.sqlite_store import SQLiteMemoryStore
from lucy.voice.audio import FRAME_SAMPLES, FileAudioIO, Frame, pcm_to_float, write_wav
from lucy.voice.vad import ScriptedVAD
from lucy.voice.wake import WakeStage, chime, strip_wake_phrase
from lucy.voice.wakeword import (
    EMBEDDING_MODEL,
    HEY_JARVIS_MODEL,
    MEL_MODEL,
    MODEL_FILES,
    STEP_SAMPLES,
    ModelFile,
    OpenWakeWordDetector,
    ScriptedWakeWord,
    download_models,
    missing_models,
)
from tests.fakes import FakeLLM, adapter_for
from tests.test_voice_cli import reply
from tests.voice_fakes import FakeSTT, FakeTTS, concat, silence, tone

# --- The detector wrapper ------------------------------------------------------------


class FakeOrtSession:
    """Stands in for an `onnxruntime.InferenceSession` of one of the three models."""

    def __init__(self, kind: str, log: list) -> None:
        self.kind = kind
        self.log = log
        self.scores = iter(np.linspace(0.1, 0.9, 50))

    def get_inputs(self):
        return [SimpleNamespace(name="x.1", shape=[1, 16, 96])]

    def run(self, outputs, feeds):
        (name, value), = feeds.items()  # fmt: skip
        self.log.append((self.kind, name, value.shape, value.dtype))
        if self.kind == "mel":
            frames = int(np.ceil(value.shape[1] / 160 - 3))
            return [np.zeros((1, 1, frames, 32), dtype=np.float32)]
        if self.kind == "embed":
            return [np.zeros((value.shape[0], 1, 1, 96), dtype=np.float32)]
        return [np.array([[next(self.scores)]], dtype=np.float32)]


def fake_detector(tmp_path: Path) -> tuple[OpenWakeWordDetector, list]:
    log: list = []

    def factory(path: str) -> FakeOrtSession:
        name = Path(path).name
        kind = {"melspectrogram.onnx": "mel", "embedding_model.onnx": "embed"}.get(name, "phrase")
        return FakeOrtSession(kind, log)

    return OpenWakeWordDetector(tmp_path / "hey_jarvis_v0.1.onnx", session_factory=factory), log


def frame_of(samples: int = FRAME_SAMPLES) -> AudioChunk:
    return AudioChunk(data=bytes(2 * samples), sample_rate=INPUT_SAMPLE_RATE)


def test_detector_steps_every_1280_samples_with_openwakeword_shapes(tmp_path: Path) -> None:
    detector, log = fake_detector(tmp_path)
    detector.load()
    log.clear()  # the seed embeddings computed on load
    scores = [detector(frame_of()) for _ in range(5)]  # 2560 samples: two steps
    phrase_calls = [entry for entry in log if entry[0] == "phrase"]
    assert len(phrase_calls) == 2 == 5 * FRAME_SAMPLES // STEP_SAMPLES
    mel = [entry for entry in log if entry[0] == "mel"]
    assert [m[1:3] for m in mel] == [("input", (1, 1760))] * 2  # 480 samples of context
    embed = [entry for entry in log if entry[0] == "embed"]
    assert [e[1:3] for e in embed] == [("input_1", (1, 76, 32, 1))] * 2
    assert phrase_calls[0][1:] == ("x.1", (1, 16, 96), np.float32)
    # A score changes only when a step completes; the first five are zeroed.
    assert scores == [0.0] * 5


def test_detector_zeroes_five_predictions_then_scores_and_reset_rearms(tmp_path: Path) -> None:
    detector, _ = fake_detector(tmp_path)
    step = frame_of(STEP_SAMPLES)
    scores = [detector(step) for _ in range(7)]
    assert scores[:5] == [0.0] * 5 and scores[5] > 0 and scores[6] > scores[5]
    detector.reset()
    assert detector(frame_of(100)) == 0.0  # partial step: nothing new, score cleared
    assert [detector(step) for _ in range(5)] == [0.0] * 5


def test_detector_without_files_says_how_to_get_them(tmp_path: Path) -> None:
    detector = OpenWakeWordDetector(tmp_path / "hey_jarvis_v0.1.onnx")
    with pytest.raises(FileNotFoundError, match="lucy listen --download-models"):
        detector.load()
    assert len(missing_models(tmp_path / "hey_jarvis_v0.1.onnx")) == 3


# --- Downloading the models ------------------------------------------------------------


def files_for(contents: dict[str, bytes]) -> list[ModelFile]:
    import hashlib

    return [
        ModelFile(name, hashlib.sha256(data).hexdigest(), f"https://example.test/{name}")
        for name, data in contents.items()
    ]


async def test_download_saves_files_whose_hash_matches(tmp_path: Path) -> None:
    contents = {"a.onnx": b"model a", "b.onnx": b"model b"}
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=contents[request.url.path.lstrip("/")])

    lines: list[str] = []
    paths = await download_models(
        tmp_path / "m",
        files=files_for(contents),
        transport=httpx.MockTransport(handler),
        report=lines.append,
    )
    assert [p.read_bytes() for p in paths] == [b"model a", b"model b"]
    assert requests[0].headers["user-agent"] == "Lucy/0.1 (local personal assistant)"
    assert len(lines) == 2
    # Already there with the right hash: not fetched again.
    requests.clear()
    await download_models(
        tmp_path / "m", files=files_for(contents), transport=httpx.MockTransport(handler)
    )
    assert requests == []


async def test_download_refuses_a_hash_mismatch(tmp_path: Path) -> None:
    files = files_for({"a.onnx": b"expected"})
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b"tampered"))
    with pytest.raises(ValueError, match="does not match"):
        await download_models(tmp_path, files=files, transport=transport)
    assert list(tmp_path.iterdir()) == []


def test_the_pinned_files_are_openwakewords_release_assets() -> None:
    assert [f.name for f in MODEL_FILES] == [
        MEL_MODEL.name,
        EMBEDDING_MODEL.name,
        HEY_JARVIS_MODEL.name,
    ]
    for file in MODEL_FILES:
        assert file.url.startswith("https://github.com/dscripka/openWakeWord/releases/download/")
        assert len(file.sha256) == 64


async def test_listen_download_models_reports_a_failure(tmp_path: Path) -> None:
    out, err = io.StringIO(), io.StringIO()
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b"not the model"))
    model = tmp_path / "models" / "hey_jarvis_v0.1.onnx"
    code = await download_wake_models(model, out=out, err=err, transport=transport)
    assert code == 1
    assert "non-commercial" in err.getvalue() and "does not match" in err.getvalue()
    assert missing_models(model) != []


# --- The wake stage --------------------------------------------------------------------


def frames(n: int, start: float = 0.0) -> list[Frame]:
    step = FRAME_SAMPLES / INPUT_SAMPLE_RATE
    return [Frame(frame_of(), start + (i + 1) * step) for i in range(n)]


def test_stage_detects_at_the_threshold_once_within_the_refractory_period() -> None:
    detector = ScriptedWakeWord([0.2, 0.49, 0.5, 0.9, 0.9] + [0.0] * 60 + [0.8])
    stage = WakeStage(detector, threshold=0.5, refractory_s=2.0)
    hits = [i for i, f in enumerate(frames(66)) if stage.feed(f)]
    assert hits == [2, 65]  # frame 3 and 4 fall within 2 s of frame 2; frame 65 does not


def test_stage_keeps_a_rolling_buffer_of_about_two_seconds() -> None:
    stage = WakeStage(ScriptedWakeWord(), buffer_s=2.0)
    fed = frames(100)
    for f in fed:
        stage.feed(f)
    buffered = stage.take_buffer()
    assert buffered == fed[-len(buffered) :] and 1.9 < len(buffered) * 0.032 < 2.1
    assert stage.take_buffer() == []
    stage.reset()
    assert stage._detector.resets == 1  # type: ignore[attr-defined]


# --- The wake phrase in transcripts ------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hey Jarvis, what's the weather in Porto?", "What's the weather in Porto?"),
        ("hey jarvis what's the weather", "What's the weather"),
        ("Hey, Jarvis. What time is it?", "What time is it?"),
        ("Hay Jarvis, set a timer.", "Set a timer."),
        ("Hey Jarvis!", ""),
        ("Hey, Jarvis.", ""),
        ("Jarvis, what's up?", "What's up?"),
        ("Hey Jervis, tell me a joke.", "Tell me a joke."),
        ("Okay, hey Jarvis, call mum.", "Call mum."),
        ("What's the capital of Portugal?", "What's the capital of Portugal?"),
        ("Hey there, how are you?", "Hey there, how are you?"),
        ("Tell Jarvis I said hi.", "Tell Jarvis I said hi."),
        ("", ""),
    ],
)
def test_strip_wake_phrase(text: str, expected: str) -> None:
    assert strip_wake_phrase(text, "hey jarvis") == expected


def test_strip_another_phrase() -> None:
    assert strip_wake_phrase("Hey Lucy, good morning.", "Hey Lucy") == "Good morning."


def test_chimes_are_short_and_quiet() -> None:
    sound = chime((660.0, 990.0))
    assert sound.sample_rate == OUTPUT_SAMPLE_RATE
    assert 0.1 < sound.frames / OUTPUT_SAMPLE_RATE < 0.3
    assert np.abs(pcm_to_float(sound.data)).max() < 0.2


def test_wake_settings() -> None:
    wake = WakeSettings()
    assert (wake.phrase, wake.threshold, wake.followup_seconds, wake.chime) == (
        "hey jarvis",
        0.5,
        8.0,
        True,
    )
    assert wake.model == "data/models/wakeword/hey_jarvis_v0.1.onnx"
    with pytest.raises(ValidationError):
        WakeSettings(threshold=1.5)
    with pytest.raises(ValidationError):
        WakeSettings(phrase="")


# --- `lucy listen` in file mode ----------------------------------------------------------

WAKE_AMP = 0.9  # the fake detector hears the wake phrase as a loud tone
SPEECH_AMP = 0.5


def wake_phrase(seconds: float = 0.6) -> AudioChunk:
    return tone(seconds, amp=WAKE_AMP)


def speech(seconds: float = 1.0) -> AudioChunk:
    return tone(seconds, amp=SPEECH_AMP)


def loud(frame: AudioChunk) -> float:
    peak = float(np.max(np.abs(pcm_to_float(frame.data)))) if frame.data else 0.0
    return 1.0 if peak >= 0.8 else 0.0


class ListenSetup:
    def __init__(
        self,
        tmp_path: Path,
        replies: list[str],
        transcripts: list[str],
        *,
        tts: FakeTTS | None = None,
        **wake: object,
    ) -> None:
        self.tmp_path = tmp_path
        self.db = tmp_path / "data" / "lucy.db"
        self.llm = FakeLLM([reply(r) for r in replies])
        self.stt = FakeSTT(transcripts)
        self.tts = tts or FakeTTS()
        self.built: list[str] = []
        self.registry = Registry()
        self.registry.register("llm", "fake")(self._counted("llm", self.llm))
        self.registry.register("memory", "sqlite")(SQLiteMemoryStore)
        self.registry.register("stt", "fake")(self._counted("stt", self.stt))
        self.registry.register("tts", "fake")(self._counted("tts", self.tts))
        self.settings = Settings(
            llm=LLMSettings(roles={"chat": AdapterConfig(adapter="fake")}),
            memory=AdapterConfig(adapter="sqlite", options={"path": str(self.db)}),
            stt=AdapterConfig(adapter="fake"),
            tts=AdapterConfig(adapter="fake"),
            wake=WakeSettings(**wake),  # type: ignore[arg-type]
        )
        self.detector = ScriptedWakeWord(score=loud)
        self.audio: FileAudioIO | None = None
        self.out = io.StringIO()
        self.err = io.StringIO()

    def _counted(self, name: str, instance: object) -> type:
        built = self.built
        base = adapter_for(instance)

        class Counted(base):  # type: ignore[misc, valid-type]
            def __new__(cls, **options: object) -> object:
                built.append(name)
                return base.__new__(cls, **options)

        return Counted

    async def run(self, *chunks: AudioChunk, **options: object) -> int:
        path = self.tmp_path / "in.wav"
        audio = concat(*chunks)
        write_wav(path, audio.data, audio.sample_rate)
        args = cli._parser().parse_args(["listen", "--input-wav", str(path), "--fast"])

        def make_audio() -> FileAudioIO:
            self.audio = FileAudioIO([path], realtime=False)
            return self.audio

        return await run_listen(
            self.settings,
            args,
            out=self.out,
            err=self.err,
            registry=self.registry,
            make_vad=lambda: ScriptedVAD(level=0.1),
            make_audio=make_audio,
            make_detector=lambda: self.detector,
            **options,  # type: ignore[arg-type]
        )

    async def users(self) -> list[str]:
        if not self.db.exists():
            return []
        async with SQLiteMemoryStore(self.db) as store:
            episodes = await store.recent_episodes("cli", 1000)
        return [e.text for e in episodes if isinstance(e, UserMessage)]

    def chimes(self) -> list[float]:
        """Start times of the plays that were chimes (by their length), in order."""
        assert self.audio is not None
        length = chime((660.0, 990.0)).frames
        return [p.start for p in self.audio.plays if p.samples == length]


async def test_nothing_is_loaded_until_the_wake_phrase(tmp_path: Path) -> None:
    s = ListenSetup(tmp_path, ["Hi."], ["hello there"])
    assert await s.run(silence(1.0), speech(1.0), silence(2.0)) == 0
    assert s.built == [] and s.stt.calls == [] and not s.db.exists()
    assert s.detector.calls > 0
    assert "Say 'hey jarvis' to wake Lucy." in s.err.getvalue()


async def test_wake_then_a_request_in_one_breath(tmp_path: Path) -> None:
    s = ListenSetup(tmp_path, ["It is sunny."], ["Hey, Jarvis, what's the weather in Porto?"])
    assert await s.run(silence(0.5), wake_phrase(), speech(1.0), silence(2.0)) == 0
    assert s.built == ["llm", "stt", "tts"]
    assert await s.users() == ["What's the weather in Porto?"]
    assert s.out.getvalue() == "you> What's the weather in Porto?\nlucy> It is sunny.\n"
    # The recogniser heard the wake phrase and the request as one utterance.
    (heard,) = s.stt.calls
    assert heard.frames / INPUT_SAMPLE_RATE >= 1.6
    assert len(s.chimes()) == 1


async def test_wake_phrase_alone_then_the_request(tmp_path: Path) -> None:
    s = ListenSetup(tmp_path, ["Noon."], ["Hey Jarvis.", "what time is it"])
    code = await s.run(silence(0.5), wake_phrase(), silence(2.0), speech(1.0), silence(2.0))
    assert code == 0
    assert len(s.stt.calls) == 2
    assert await s.users() == ["what time is it"]


async def test_wake_phrase_alone_times_out_back_to_waiting(tmp_path: Path) -> None:
    s = ListenSetup(tmp_path, ["Never."], ["Hey Jarvis.", "unheard"], followup_seconds=3)
    await s.run(silence(0.5), wake_phrase(), silence(5.0), speech(1.0), silence(2.0))
    assert len(s.stt.calls) == 1  # the speech after the timeout was not listened to
    assert await s.users() == []
    wake, sleep = s.chimes()
    assert 3.0 < sleep - wake < 4.5
    assert s.detector.resets >= 1


async def test_follow_up_without_the_wake_phrase_then_the_window_expires(tmp_path: Path) -> None:
    s = ListenSetup(
        tmp_path, ["Lisbon.", "About ten million."], ["hey jarvis capital of portugal",
        "and its population", "not for lucy"], followup_seconds=4,
    )  # fmt: skip
    code = await s.run(
        silence(0.5), wake_phrase(), speech(1.0), silence(2.0),  # wake and a request
        speech(1.0), silence(6.0),  # a follow-up within the window, then it expires
        speech(1.0), silence(2.0),  # ignored: no wake phrase
    )  # fmt: skip
    assert code == 0
    assert await s.users() == ["Capital of portugal", "and its population"]
    assert len(s.stt.calls) == 2
    assert len(s.chimes()) == 2  # the wake and the return to waiting


async def test_a_second_wake_after_waiting_again(tmp_path: Path) -> None:
    s = ListenSetup(
        tmp_path, ["One.", "Two."], ["hey jarvis first", "hey jarvis second"], followup_seconds=2
    )
    await s.run(
        silence(0.5), wake_phrase(), speech(0.8), silence(5.0),
        wake_phrase(), speech(0.8), silence(2.0),
    )  # fmt: skip
    assert await s.users() == ["First", "Second"]
    assert s.built == ["llm", "stt", "tts"]  # built once, kept loaded
    assert len(s.chimes()) == 3  # wake, sleep, wake


async def test_the_wake_phrase_is_not_heard_while_lucy_speaks(tmp_path: Path) -> None:
    # The reply lasts 2 s (40 chars at 0.05 s each); a wake phrase said meanwhile is
    # neither a wake nor a turn.
    s = ListenSetup(tmp_path, ["x" * 39 + "."], ["hey jarvis talk to me"], followup_seconds=2)
    await s.run(
        silence(0.5), wake_phrase(), speech(0.8), silence(0.7),  # the reply starts here
        wake_phrase(0.6), silence(6.0),
    )  # fmt: skip
    assert len(s.stt.calls) == 1
    assert len(s.chimes()) == 2  # one wake, then back to waiting


async def test_chimes_can_be_turned_off(tmp_path: Path) -> None:
    s = ListenSetup(tmp_path, ["Hi."], ["hey jarvis hello"], chime=False, followup_seconds=1)
    await s.run(silence(0.5), wake_phrase(), speech(0.8), silence(4.0))
    assert s.chimes() == []
    assert await s.users() == ["Hello"]


async def test_missing_models_is_one_line(tmp_path: Path) -> None:
    s = ListenSetup(tmp_path, [], [], model=str(tmp_path / "nope" / "hey_jarvis_v0.1.onnx"))
    args = cli._parser().parse_args(["listen"])
    code = await run_listen(s.settings, args, out=s.out, err=s.err)
    assert code == 1
    lines = s.err.getvalue().splitlines()
    assert len(lines) == 1 and "lucy listen --download-models" in lines[0]


def test_listen_arguments() -> None:
    args = cli._parser().parse_args(["listen", "--input-wav", "a.wav", "--output-wav", "b.wav"])
    assert isinstance(args, argparse.Namespace)
    assert (args.command, args.session, args.download_models) == ("listen", "cli", False)
