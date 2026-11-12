import asyncio
import io
import logging
import os
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from lucy.app import LucyApp
from lucy.clients import cli
from lucy.clients.cli import ChatLoop, ReadLine, StdinReader, TerminalReader
from lucy.core.config import AdapterConfig, LLMSettings, Settings
from lucy.core.events import AssistantMessage, TurnEnded
from lucy.core.registry import Registry
from lucy.core.types import FinishReason, LLMChunk, ToolCall
from lucy.memory.sqlite_store import SQLiteMemoryStore
from tests.fakes import FakeLLM, Pause, Raise, Script, adapter_for

TIMEOUT = 5


def reply(text: str) -> list[LLMChunk]:
    return [LLMChunk(text=text), LLMChunk(finish_reason=FinishReason.STOP)]


class ScriptedInput:
    """An async line reader: returns `lines` in order, then end of input.

    With `block_at_end`, it instead waits forever once the lines run out (a user
    sitting at the prompt), setting `waiting`.
    """

    def __init__(self, lines: Sequence[str], *, block_at_end: bool = False) -> None:
        self._lines = list(lines)
        self._block = block_at_end
        self.waiting = asyncio.Event()

    @property
    def remaining(self) -> list[str]:
        return self._lines

    async def __call__(self) -> str | None:
        if self._lines:
            return self._lines.pop(0)
        if self._block:
            self.waiting.set()
            await asyncio.Event().wait()
        return None


class Harness:
    def __init__(self, db: Path, responses: Sequence[Script]) -> None:
        self.llm = FakeLLM(responses)
        registry = Registry()
        registry.register("llm", "fake")(adapter_for(self.llm))
        registry.register("memory", "sqlite")(SQLiteMemoryStore)
        settings = Settings(
            llm=LLMSettings(roles={"chat": AdapterConfig(adapter="fake")}),
            memory=AdapterConfig(adapter="sqlite", options={"path": str(db)}),
        )
        self.app = LucyApp(settings, registry=registry)
        self.out = io.StringIO()
        self.err = io.StringIO()

    def chat(
        self, read_line: ReadLine, *, echo_input: bool = False, reader_prompts: bool = False
    ) -> ChatLoop:
        return ChatLoop(
            self.app,
            "cli",
            read_line=read_line,
            out=self.out,
            err=self.err,
            echo_input=echo_input,
            reader_prompts=reader_prompts,
        )


@pytest.fixture
async def make_harness(tmp_path: Path) -> AsyncIterator:
    harnesses: list[Harness] = []

    async def make(responses: Sequence[Script]) -> Harness:
        harness = Harness(tmp_path / "lucy.db", responses)
        await harness.app.start()
        harnesses.append(harness)
        return harness

    yield make
    for harness in harnesses:
        await harness.app.stop()


@pytest.fixture
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)


async def run(chat: ChatLoop) -> int:
    async with asyncio.timeout(TIMEOUT):
        return await chat.run()


async def test_exchange_ignores_empty_lines_and_quits(make_harness) -> None:
    h = await make_harness([reply("Hi there.")])
    lines = ScriptedInput(["hello", "", "   ", "/quit", "never read"])
    assert await run(h.chat(lines)) == 0
    assert h.out.getvalue() == "you> lucy> Hi there.\nyou> you> you> "
    assert lines.remaining == ["never read"]
    assert "Session 'cli' (0 earlier events)" in h.err.getvalue()
    assert len(h.llm.calls) == 1


async def test_exit_command_and_echoed_input(make_harness) -> None:
    h = await make_harness([reply("Hello.")])
    assert await run(h.chat(ScriptedInput(["hi", "/exit"]), echo_input=True)) == 0
    assert h.out.getvalue() == "you> hi\nlucy> Hello.\nyou> /exit\n"


async def test_end_of_input_exits_after_the_turn(make_harness) -> None:
    h = await make_harness([reply("Bye for now.")])
    assert await run(h.chat(ScriptedInput(["last words"]))) == 0
    assert h.out.getvalue() == "you> lucy> Bye for now.\nyou> \n"
    episodes = await h.app.memory.recent_episodes("cli", 10)
    assert [e.type for e in episodes] == ["user_message", "assistant_message", "turn_ended"]


async def test_banner_counts_earlier_events(make_harness) -> None:
    first = await make_harness([reply("One.")])
    assert await run(first.chat(ScriptedInput(["one"]))) == 0
    await first.app.stop()
    second = await make_harness([])
    assert await run(second.chat(ScriptedInput([]))) == 0
    assert "Session 'cli' (3 earlier events)" in second.err.getvalue()


async def test_error_turn_is_one_line_and_chat_continues(
    make_harness, restore_logging: None
) -> None:
    log = io.StringIO()
    cli.configure_logging(verbose=False, stream=log)
    h = await make_harness(
        [[Raise(RuntimeError("cannot reach Ollama; is it running?"))], reply("Back again.")]
    )
    assert await run(h.chat(ScriptedInput(["first", "second"]))) == 0
    assert h.out.getvalue() == "you> lucy> \nyou> lucy> Back again.\nyou> \n"
    assert "error: RuntimeError: cannot reach Ollama; is it running?\n" in h.err.getvalue()
    # The orchestrator's own error log (with traceback) is not shown on top.
    assert log.getvalue() == ""


async def test_verbose_logging_keeps_the_traceback(make_harness, restore_logging: None) -> None:
    log = io.StringIO()
    cli.configure_logging(verbose=True, stream=log)
    h = await make_harness([[Raise(RuntimeError("boom"))]])
    assert await run(h.chat(ScriptedInput(["x"]))) == 0
    assert "turn" in log.getvalue() and "Traceback" in log.getvalue()


async def test_tool_activity_goes_to_stderr(make_harness) -> None:
    call = ToolCall(id="c1", name="weather", arguments={})
    h = await make_harness(
        [
            [LLMChunk(tool_call=call), LLMChunk(finish_reason=FinishReason.TOOL_CALLS)],
            reply("No weather tool yet."),
        ]
    )
    assert await run(h.chat(ScriptedInput(["weather?"]))) == 0
    assert "[using weather]\n" in h.err.getvalue()
    # The marker goes on its own line: the reply starts on the line after it.
    assert "lucy> \nNo weather tool yet.\n" in h.out.getvalue()


async def test_tool_marker_after_text_starts_a_new_line(make_harness) -> None:
    call = ToolCall(id="c1", name="wikipedia", arguments={"query": "Piranesi"})
    h = await make_harness(
        [
            [
                LLMChunk(text="Let me check."),
                LLMChunk(tool_call=call),
                LLMChunk(finish_reason=FinishReason.TOOL_CALLS),
            ],
            reply("Susanna Clarke."),
        ]
    )
    assert await run(h.chat(ScriptedInput(["who?"]))) == 0
    assert '[looking up "Piranesi" on wikipedia]\n' in h.err.getvalue()
    assert "lucy> Let me check.\nSusanna Clarke.\n" in h.out.getvalue()


@pytest.mark.parametrize(
    ("name", "arguments", "marker"),
    [
        (
            "wikipedia",
            {"query": "Mount Kilimanjaro"},
            '[looking up "Mount Kilimanjaro" on wikipedia]',
        ),
        (
            "web_search",
            {"query": "UN secretary-general"},
            '[searching the web for "UN secretary-general"]',
        ),
        ("fetch_page", {"url": "https://example.com/a/b?c=1"}, "[reading example.com]"),
        ("fetch_page", {"url": "not a url"}, "[reading a web page]"),
        ("fetch_page", {}, "[reading a web page]"),
        ("wikipedia", {}, "[using wikipedia]"),
        ("time", {"place": "Tokyo"}, "[checking the time in Tokyo]"),
        ("time", {"place": " New\nYork ", "at": "9am"}, "[checking the time in New York]"),
        ("time", {}, "[using time]"),
        ("weather", {"place": "Porto", "days": 2}, "[checking the weather in Porto]"),
        ("weather", {"days": 2}, "[using weather]"),
        ("lights", {"level": 3, "room": "hall"}, '[using lights: "hall"]'),
        ("lights", {}, "[using lights]"),
        (
            "web_search",
            {"query": "a\nb  " + "x" * 100},
            '[searching the web for "a b ' + "x" * 55 + '…"]',
        ),
    ],
)
def test_tool_marker(name: str, arguments: dict[str, object], marker: str) -> None:
    assert cli.tool_marker(ToolCall(id="c", name=name, arguments=arguments)) == marker


async def test_interrupt_while_streaming(make_harness) -> None:
    pause = Pause()
    h = await make_harness(
        [
            [LLMChunk(text="Once upon a time"), pause, LLMChunk(text=", the end.")],
            reply("Short one."),
        ]
    )
    chat = h.chat(ScriptedInput(["story", "again"]))
    async with asyncio.timeout(TIMEOUT):
        task = asyncio.create_task(chat.run())
        await pause.reached.wait()
        chat.interrupt()  # what the SIGINT handler calls
        assert await task == 0
    assert h.out.getvalue() == (
        "you> lucy> Once upon a time [interrupted]\nyou> lucy> Short one.\nyou> \n"
    )
    episodes = await h.app.memory.recent_episodes("cli", 20)
    interrupted = [e for e in episodes if isinstance(e, AssistantMessage) and e.interrupted]
    assert [e.text for e in interrupted] == ["Once upon a time"]
    reasons = [e.reason for e in episodes if isinstance(e, TurnEnded)]
    assert reasons == ["interrupted", "completed"]


async def test_interrupt_at_the_prompt_quits(make_harness) -> None:
    h = await make_harness([reply("Hello.")])
    lines = ScriptedInput(["hi"], block_at_end=True)
    chat = h.chat(lines)
    async with asyncio.timeout(TIMEOUT):
        task = asyncio.create_task(chat.run())
        await lines.waiting.wait()
        chat.interrupt()
        assert await task == 0
    assert h.out.getvalue() == "you> lucy> Hello.\nyou> \n"


async def test_stdin_reader_splits_lines_and_signals_end(tmp_path: Path) -> None:
    read_fd, write_fd = os.pipe()
    try:
        reader = StdinReader(read_fd)
        reader.start()
        os.write(write_fd, "one\r\ntwo\n\ncafé no newline".encode())
        os.close(write_fd)
        write_fd = -1
        async with asyncio.timeout(TIMEOUT):
            lines = [await reader.readline() for _ in range(6)]
    finally:
        if write_fd != -1:
            os.close(write_fd)
        os.close(read_fd)
    assert lines == ["one", "two", "", "café no newline", None, None]


# --- Line editing at a terminal (TerminalReader) ------------------------------------

LEFT, RIGHT, UP, HOME, END, BACKSPACE = "\x1b[D", "\x1b[C", "\x1b[A", "\x01", "\x05", "\x7f"


async def read_typed(reader: TerminalReader, pipe, keys: str) -> str | None:
    pipe.send_text(keys)
    async with asyncio.timeout(TIMEOUT):
        return await reader.readline()


async def test_terminal_reader_applies_cursor_moves_and_backspace(tmp_path: Path) -> None:
    with create_pipe_input() as pipe:
        reader = TerminalReader(input=pipe, output=DummyOutput())
        keys = f"I lkk{BACKSPACE}e tea{HOME}{RIGHT * 3}i{END} a lot{LEFT * 3}{BACKSPACE}\r"
        assert await read_typed(reader, pipe, keys) == "I like tea alot"


async def test_terminal_reader_ctrl_d_and_ctrl_c_end_input() -> None:
    with create_pipe_input() as pipe:
        reader = TerminalReader(input=pipe, output=DummyOutput())
        assert await read_typed(reader, pipe, "\x04") is None  # Ctrl-D on an empty line
        assert await read_typed(reader, pipe, "half typed\x03") is None  # Ctrl-C
        assert await read_typed(reader, pipe, "still works\r") == "still works"


async def test_terminal_reader_history_is_kept_across_runs(tmp_path: Path) -> None:
    history = tmp_path / "data" / "chat_history"  # the directory is created
    with create_pipe_input() as pipe:
        reader = TerminalReader(history_file=history, input=pipe, output=DummyOutput())
        assert await read_typed(reader, pipe, "first line\r") == "first line"
        assert await read_typed(reader, pipe, "second line\r") == "second line"
        assert await read_typed(reader, pipe, f"{UP}{UP}\r") == "first line"
    with create_pipe_input() as pipe:  # a new run reads the same file
        reader = TerminalReader(history_file=history, input=pipe, output=DummyOutput())
        assert await read_typed(reader, pipe, f"{UP} again\r") == "first line again"


async def test_terminal_reader_without_a_usable_history_file_still_reads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "a_file"
    blocker.write_text("")
    with create_pipe_input() as pipe:
        reader = TerminalReader(
            history_file=blocker / "chat_history", input=pipe, output=DummyOutput()
        )
        assert await read_typed(reader, pipe, "hello\r") == "hello"
        assert await read_typed(reader, pipe, f"{UP}!\r") == "hello!"  # in-memory history
    assert "cannot use the history file" in caplog.text


async def test_chat_with_a_prompting_reader_writes_no_prompt(make_harness) -> None:
    h = await make_harness([reply("Hi there.")])
    assert await run(h.chat(ScriptedInput(["hello"]), reader_prompts=True)) == 0
    # The reader draws "you> " and ends its own line, also on Ctrl-D / Ctrl-C.
    assert h.out.getvalue() == "lucy> Hi there.\n"


async def test_chat_over_a_terminal_reader_end_to_end(make_harness) -> None:
    h = await make_harness([reply("Noted."), reply("Again.")])
    with create_pipe_input() as pipe:
        reader = TerminalReader(input=pipe, output=DummyOutput())
        pipe.send_text("helo{LEFT}{LEFT}l\r".replace("{LEFT}", LEFT))
        chat = h.chat(reader.readline, reader_prompts=True)
        async with asyncio.timeout(TIMEOUT):
            task = asyncio.create_task(chat.run())
            while len(h.llm.calls) < 1:
                await asyncio.sleep(0.01)
            pipe.send_text(f"{UP}\r")
            while len(h.llm.calls) < 2:
                await asyncio.sleep(0.01)
            pipe.send_text("\x04")
            assert await task == 0
    texts = [
        e.text for e in await h.app.memory.recent_episodes("cli", 20) if e.type == "user_message"
    ]
    assert texts == ["hello", "hello"]
    assert h.out.getvalue() == "lucy> Noted.\nlucy> Again.\n"


def test_chat_history_lives_beside_the_database(tmp_path: Path) -> None:
    llm = LLMSettings(roles={"chat": AdapterConfig(adapter="fake")})
    db = tmp_path / "x" / "lucy.db"
    with_db = Settings(llm=llm, memory=AdapterConfig(adapter="sqlite", options={"path": str(db)}))
    assert cli.chat_history_path(with_db) == tmp_path / "x" / "chat_history"
    assert cli.chat_history_path(Settings(llm=llm)) == Path("data") / "chat_history"


def test_missing_default_config_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    restore_logging: None,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.main(["chat"]) == 2
    assert "copy lucy.example.toml to lucy.toml" in capsys.readouterr().err


def test_missing_explicit_config_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], restore_logging: None
) -> None:
    assert cli.main(["chat", "--config", str(tmp_path / "nope.toml")]) == 2
    assert "settings file not found" in capsys.readouterr().err


def test_invalid_config_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], restore_logging: None
) -> None:
    path = tmp_path / "lucy.toml"
    path.write_text("[llm.roles]\n")
    assert cli.main(["chat", "--config", str(path)]) == 2
    assert "chat" in capsys.readouterr().err
