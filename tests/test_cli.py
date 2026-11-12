import asyncio
import io
import logging
import os
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path

import pytest

from lucy.app import LucyApp
from lucy.clients import cli
from lucy.clients.cli import ChatLoop, StdinReader
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

    def chat(self, read_line: ScriptedInput, *, echo_input: bool = False) -> ChatLoop:
        return ChatLoop(
            self.app,
            "cli",
            read_line=read_line,
            out=self.out,
            err=self.err,
            echo_input=echo_input,
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
    assert "[using weather]" in h.err.getvalue()
    assert "lucy> No weather tool yet.\n" in h.out.getvalue()


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
