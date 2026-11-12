"""`lucy voice` in file mode with fake models, and its terminal output."""

import argparse
import asyncio
import io
import json
from pathlib import Path

import pytest

from lucy.clients import cli, voice_cli
from lucy.clients.voice_cli import TerminalUI, run_voice
from lucy.core.config import AdapterConfig, LLMSettings, Settings
from lucy.core.events import PlaybackInterrupted, UserMessage
from lucy.core.registry import Registry
from lucy.core.types import FinishReason, LLMChunk
from lucy.memory.sqlite_store import SQLiteMemoryStore
from lucy.voice import session as session_module
from lucy.voice.audio import FileAudioIO, SoundDeviceIO, read_wav, write_wav
from lucy.voice.session import VoiceSession, VoiceState
from lucy.voice.vad import ScriptedVAD
from tests.fakes import FakeLLM, adapter_for
from tests.test_voice_audio import FakeSD
from tests.voice_fakes import FakeSTT, FakeTTS, concat, silence, tone


def reply(text: str) -> list[LLMChunk]:
    return [LLMChunk(text=text), LLMChunk(finish_reason=FinishReason.STOP)]


def parse(*argv: str) -> argparse.Namespace:
    return cli._parser().parse_args(["voice", *argv])


class Setup:
    def __init__(self, tmp_path: Path, replies: list[str], transcripts: list[str]) -> None:
        self.db = tmp_path / "data" / "lucy.db"
        self.llm = FakeLLM([reply(r) for r in replies])
        self.stt = FakeSTT(transcripts)
        self.tts = FakeTTS()
        self.registry = Registry()
        self.registry.register("llm", "fake")(adapter_for(self.llm))
        self.registry.register("memory", "sqlite")(SQLiteMemoryStore)
        self.registry.register("stt", "fake")(adapter_for(self.stt))
        self.registry.register("tts", "fake")(adapter_for(self.tts))
        self.settings = Settings(
            llm=LLMSettings(roles={"chat": AdapterConfig(adapter="fake")}),
            memory=AdapterConfig(adapter="sqlite", options={"path": str(self.db)}),
            stt=AdapterConfig(adapter="fake"),
            tts=AdapterConfig(adapter="fake"),
        )
        self.out = io.StringIO()
        self.err = io.StringIO()

    async def run(self, args: argparse.Namespace, **options) -> int:
        options.setdefault("make_vad", lambda: ScriptedVAD(level=0.1))
        return await run_voice(
            self.settings, args, out=self.out, err=self.err, registry=self.registry, **options
        )

    async def episodes(self) -> list:
        async with SQLiteMemoryStore(self.db) as store:
            return await store.recent_episodes("cli", 1000)


def wav(path: Path, *chunks) -> Path:
    audio = concat(*chunks)
    write_wav(path, audio.data, audio.sample_rate)
    return path


async def test_file_mode_runs_a_conversation(tmp_path: Path) -> None:
    s = Setup(tmp_path, ["Hello there.", "Bye now."], ["hi lucy", "goodbye"])
    first = wav(tmp_path / "a.wav", silence(0.3), tone(1.0), silence(1.0))
    second = wav(tmp_path / "b.wav", silence(0.3), tone(0.8), silence(1.0))
    out_wav = tmp_path / "out" / "lucy.wav"
    args = parse(
        "--input-wav", str(first), str(second), "--output-wav", str(out_wav), "--fast",
        "--wait-for-reply",
    )  # fmt: skip
    assert await s.run(args) == 0
    assert s.out.getvalue() == "you> hi lucy\nlucy> Hello there.\nyou> goodbye\nlucy> Bye now.\n"
    err = s.err.getvalue()
    assert "Session 'cli' (0 earlier events)." in err
    assert "Memory: off" in err
    assert "a.wav, b.wav (simulated time)" in err and "lucy.wav" in err
    # What was said went to the output file, and the turn timings beside the database.
    samples, rate = read_wav(out_wav)
    assert rate == 24000 and (abs(samples) > 0.01).sum() > 0
    lines = (s.db.parent / "voice_timings.jsonl").read_text().splitlines()
    assert [json.loads(line)["outcome"] for line in lines] == ["completed", "completed"]
    users = [e for e in await s.episodes() if isinstance(e, UserMessage)]
    assert [(u.text, u.source) for u in users] == [("hi lucy", "voice"), ("goodbye", "voice")]


async def test_file_mode_continues_the_chat_session(tmp_path: Path) -> None:
    s = Setup(tmp_path, ["One.", "Two."], ["first", "second"])
    clip = wav(tmp_path / "a.wav", silence(0.3), tone(1.0), silence(1.0))
    assert await s.run(parse("--input-wav", str(clip), "--fast")) == 0
    s.out, s.err = io.StringIO(), io.StringIO()
    assert await s.run(parse("--input-wav", str(clip), "--fast")) == 0
    assert "(3 earlier events)" in s.err.getvalue()
    assert [m.content for m in s.llm.calls[1][1:-1]] == ["first", "One.", "second"]


async def test_another_session_can_be_chosen(tmp_path: Path) -> None:
    s = Setup(tmp_path, ["One."], ["first"])
    clip = wav(tmp_path / "a.wav", silence(0.3), tone(1.0), silence(1.0))
    assert await s.run(parse("--input-wav", str(clip), "--fast", "--session", "kitchen")) == 0
    assert "Session 'kitchen'" in s.err.getvalue()
    async with SQLiteMemoryStore(s.db) as store:
        assert len(await store.recent_episodes("kitchen", 100)) == 3


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--output-wav", "x.wav"], "--output-wav needs --input-wav"),
        (["--fast"], "--fast needs --input-wav"),
        (["--input-wav", "/nonexistent/a.wav"], "no such input file"),
    ],
)
async def test_bad_arguments_exit_2(tmp_path: Path, argv: list[str], message: str) -> None:
    s = Setup(tmp_path, [], [])
    assert await s.run(parse(*argv)) == 2
    assert message in s.err.getvalue()


async def test_an_unreadable_input_file_exits_2(tmp_path: Path) -> None:
    s = Setup(tmp_path, [], [])
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"nope")
    assert await s.run(parse("--input-wav", str(bad))) == 2
    assert "cannot read WAV" in s.err.getvalue()


async def test_no_microphone_is_one_clear_line(tmp_path: Path) -> None:
    s = Setup(tmp_path, [], [])
    code = await s.run(parse(), make_audio=lambda: SoundDeviceIO(sd=FakeSD(no_input=True)))
    assert code == 1
    (line,) = [x for x in s.err.getvalue().splitlines() if x.startswith("lucy voice:")]
    assert "no microphone found" in line and "Privacy & Security > Microphone" in line


def test_voice_without_speech_sections_fails_clearly(tmp_path: Path, capsys) -> None:
    config = tmp_path / "lucy.toml"
    config.write_text(
        '[llm.roles]\nchat = { adapter = "ollama", model = "m" }\n\n'
        f'[memory]\nadapter = "sqlite"\npath = "{tmp_path / "lucy.db"}"\n'
    )
    clip = wav(tmp_path / "a.wav", silence(0.5))
    assert cli.main(["voice", "--config", str(config), "--input-wav", str(clip)]) == 1
    assert "voice needs [stt] and [tts] in the settings" in capsys.readouterr().err


def test_list_devices(capsys) -> None:
    assert voice_cli.list_devices(io_out := io.StringIO(), io.StringIO(), FakeSD()) == 0
    assert "Input devices (microphones):" in io_out.getvalue()
    assert "(default)" in io_out.getvalue()


def test_list_devices_reports_failure() -> None:
    class Broken(FakeSD):
        def query_devices(self, device=None, kind=None):
            raise OSError("PortAudio not initialized")

    err = io.StringIO()
    assert voice_cli.list_devices(io.StringIO(), err, Broken()) == 1
    assert "cannot list sound devices" in err.getvalue()


def test_the_voice_command_is_registered() -> None:
    args = parse("--list-devices")
    assert args.list_devices and args.session == "cli"
    help_text = cli._parser().format_help()
    assert "voice" in help_text


def test_timings_live_beside_the_database(tmp_path: Path) -> None:
    s = Setup(tmp_path, [], [])
    assert voice_cli.timings_path(s.settings) == s.db.parent / "voice_timings.jsonl"


# --- Terminal output ---------------------------------------------------------------


def make_ui(show_state: bool = False) -> tuple[TerminalUI, io.StringIO, io.StringIO]:
    out, err = io.StringIO(), io.StringIO()
    return TerminalUI(out, err, show_state=show_state), out, err


def test_ui_prints_the_conversation() -> None:
    ui, out, err = make_ui()
    ui.state(VoiceState.LISTENING)
    ui.user("what time is it")
    ui.reply_delta(" It is")
    ui.reply_delta(" noon.")
    ui.reply_end("completed", None)
    assert out.getvalue() == "you> what time is it\nlucy> It is noon.\n"
    assert err.getvalue() == ""


def test_ui_marks_interruptions_and_errors() -> None:
    ui, out, err = make_ui()
    ui.reply_delta("Once upon")
    ui.reply_end("interrupted", None)
    ui.reply_end("interrupted", None)
    ui.reply_delta("Half")
    ui.reply_end("error", "server gone")
    assert out.getvalue() == "lucy> Once upon [interrupted]\nlucy> [interrupted]\nlucy> Half\n"
    assert "error: server gone" in err.getvalue()


def test_ui_notice_breaks_an_open_reply_line() -> None:
    ui, out, err = make_ui()
    ui.reply_delta("Let me look")
    ui.notice("[using wikipedia]")
    ui.reply_delta("Found it.")
    ui.reply_end("completed", None)
    assert out.getvalue() == "lucy> Let me look\nlucy> Found it.\n"
    assert err.getvalue() == "[using wikipedia]\n"


def test_ui_state_indicator_stays_off_the_reply_line() -> None:
    ui, out, err = make_ui(show_state=True)
    ui.state(VoiceState.LISTENING)
    assert err.getvalue().endswith("(listening)")
    ui.user("hello")
    ui.state(VoiceState.THINKING)
    ui.reply_delta("Hi.")
    ui.state(VoiceState.SPEAKING)  # not drawn while the reply line is open
    assert "(speaking)" not in err.getvalue()
    ui.reply_end("completed", None)
    assert err.getvalue().endswith("(speaking)")
    assert out.getvalue() == "you> hello\nlucy> Hi.\n"


async def test_interrupt_from_a_scripted_run_is_recorded(tmp_path: Path, monkeypatch) -> None:
    """`run_voice` wires the session to the app: an interrupt mid-speech is persisted."""
    s = Setup(tmp_path, ["One two three four five six seven eight nine ten."], ["count"])
    clip = wav(tmp_path / "a.wav", silence(0.3), tone(1.0), silence(5.0))
    audio = FileAudioIO([clip])
    sessions: list[VoiceSession] = []

    class Recorded(VoiceSession):
        def __init__(self, **options) -> None:
            super().__init__(**options)
            sessions.append(self)

    monkeypatch.setattr(session_module, "VoiceSession", Recorded)

    async def interrupter() -> None:
        while audio.played_samples < 24000:  # one second of speech
            await asyncio.sleep(0)
        assert sessions[0].interrupt()

    task = asyncio.create_task(interrupter())
    assert await s.run(parse("--input-wav", str(clip)), make_audio=lambda: audio) == 0
    await asyncio.wait_for(task, 1)
    assert "lucy> One two three four five six seven eight nine ten.\n" in s.out.getvalue()
    (record,) = [e for e in await s.episodes() if isinstance(e, PlaybackInterrupted)]
    assert record.spoken_text.startswith("One two") and "ten." not in record.spoken_text
