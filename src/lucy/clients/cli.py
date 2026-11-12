"""The `lucy` command: a streaming text chat in the terminal, memory tools and the eval.

`lucy chat` starts a `LucyApp` in this process and talks to it over the event
bus, like any other client would. `lucy memory ...` inspects and controls the
knowledge graph (`lucy.clients.memory_cli`) with an app that runs nothing in
the background. `lucy eval` runs the memory eval scenarios (`lucy.evals`)
against the configured models, each on a temporary database. The conversation
logic lives in `ChatLoop`, which takes an async line reader and output streams
so it can be tested without a terminal; `main` only parses arguments and wires
it to stdin, stdout, stderr and SIGINT.
"""

from __future__ import annotations

import argparse
import asyncio
import codecs
import json
import logging
import os
import signal
import sys
import threading
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Final, TextIO
from urllib.parse import urlsplit

from lucy.app import JobBacklog, LucyApp
from lucy.clients import memory_cli, voice_cli
from lucy.core.config import ConfigError, Settings, load_settings
from lucy.core.events import Interrupt, TextDelta, ToolStarted, TurnEnded, UserMessage
from lucy.core.types import ToolCall

DEFAULT_CONFIG: Final = Path("lucy.toml")
DEFAULT_SESSION: Final = "cli"
PROMPT: Final = "you> "
#: Input history of `lucy chat` at a terminal, kept next to the database.
HISTORY_FILE_NAME: Final = "chat_history"
QUIT_COMMANDS: Final = frozenset({"/quit", "/exit"})
_HISTORY_COUNT_LIMIT: Final = 1_000_000

ReadLine = Callable[[], Awaitable[str | None]]
"""Returns the next input line without its line ending, or None at end of input."""


class ChatLoop:
    """One interactive chat session against a running `LucyApp`.

    `interrupt` is what the SIGINT handler calls (tests call it directly):
    while a reply is streaming it publishes an `Interrupt` for the session and
    the loop keeps waiting for that turn's `TurnEnded`; at the prompt it ends
    the chat.

    With `reader_prompts`, `read_line` draws the prompt itself (`TerminalReader`)
    and the loop writes neither the prompt nor the line end after it.
    """

    def __init__(
        self,
        app: LucyApp,
        session_id: str,
        *,
        read_line: ReadLine,
        out: TextIO,
        err: TextIO,
        echo_input: bool = False,
        reader_prompts: bool = False,
    ) -> None:
        self._app = app
        self._session_id = session_id
        self._read_line = read_line
        self._out = out
        self._err = err
        self._echo_input = echo_input
        self._reader_prompts = reader_prompts
        self._in_turn = False
        self._quit = asyncio.Event()
        self._interrupts: set[asyncio.Task[None]] = set()

    def interrupt(self) -> None:
        """Handle Ctrl-C: interrupt the streaming reply, or quit at the prompt."""
        if not self._in_turn:
            self._quit.set()
            return
        task = asyncio.get_running_loop().create_task(self._publish_interrupt())
        self._interrupts.add(task)
        task.add_done_callback(self._interrupts.discard)

    async def _publish_interrupt(self) -> None:
        try:
            await self._app.bus.publish(Interrupt(session_id=self._session_id))
        except RuntimeError:  # the app has stopped: there is nothing left to interrupt
            pass

    async def run(self) -> int:
        """Chat until /quit, end of input or Ctrl-C at the prompt. Returns the exit code."""
        earlier = len(
            await self._app.memory.recent_episodes(self._session_id, _HISTORY_COUNT_LIMIT)
        )
        self._print_err(
            f"Session {self._session_id!r} ({earlier} earlier events). "
            "Type /quit or press Ctrl-D to exit; Ctrl-C interrupts a reply."
        )
        self._print_err(memory_banner(self._app))
        try:
            pending = await self._app.pending_jobs()
        except Exception:  # the job queue is unreadable: say nothing, chat anyway
            pending = 0
        if pending:
            self._print_err(f"{_jobs(pending)} pending; they run in the background while you chat.")
        while True:
            line = await self._next_line()
            if line is None:
                break
            text = line.strip()
            if not text:
                continue
            if text in QUIT_COMMANDS:
                break
            if not await self._turn(text):
                self._print_err("Lucy has stopped.")
                return 1
        return 0

    async def _next_line(self) -> str | None:
        """Prompt and wait for a line, or None on end of input or Ctrl-C at the prompt."""
        if not self._reader_prompts:
            self._write(PROMPT)
        reader = asyncio.ensure_future(self._read_line())
        quitter = asyncio.ensure_future(self._quit.wait())
        try:
            await asyncio.wait({reader, quitter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (reader, quitter):
                if not task.done():
                    task.cancel()
            await asyncio.gather(reader, quitter, return_exceptions=True)
        if self._quit.is_set() or reader.cancelled():
            self._write("\n")
            return None
        line = reader.result()
        if line is None and not self._reader_prompts:
            self._write("\n")
        elif self._echo_input:
            self._write(line + "\n")
        return line

    async def _turn(self, text: str) -> bool:
        """Send one message and stream the reply. False if the app stopped mid-turn."""
        bus = self._app.bus
        self._in_turn = True
        try:
            # Subscribe before publishing so no event of the turn can be missed.
            async with bus.subscribe(TextDelta, ToolStarted, TurnEnded) as events:
                await bus.publish(
                    UserMessage(session_id=self._session_id, text=text, source="text")
                )
                self._write("lucy> ")
                turn_id: str | None = None
                at_line_start = False
                started = False
                async for event in events:
                    if event.session_id != self._session_id:
                        continue
                    # This session's turns run one at a time, so the first turn seen is ours.
                    turn_id = turn_id or event.turn_id
                    if event.turn_id != turn_id:
                        continue
                    if isinstance(event, TextDelta):
                        delta = event.text if started else event.text.lstrip()
                        if delta:
                            started = True
                            self._write(delta)
                            at_line_start = delta.endswith("\n")
                    elif isinstance(event, ToolStarted):
                        # The marker gets a line of its own, before the reply text.
                        if not at_line_start:
                            self._write("\n")
                            at_line_start = True
                        self._print_err(tool_marker(event.call))
                    elif isinstance(event, TurnEnded):
                        if event.reason == "interrupted":
                            self._write(" [interrupted]\n")
                        elif not at_line_start:
                            self._write("\n")
                        if event.reason == "error":
                            self._print_err(f"error: {event.error}")
                        return True
            return False
        finally:
            self._in_turn = False

    def _write(self, text: str) -> None:
        self._out.write(text)
        self._out.flush()

    def _print_err(self, text: str) -> None:
        self._err.write(text + "\n")
        self._err.flush()


_MARKER_ARGUMENT_CHARS: Final = 60


def _one_line(value: object) -> str:
    """A model-chosen argument, on one line and shortened."""
    text = " ".join(str(value).split())
    if len(text) > _MARKER_ARGUMENT_CHARS:
        text = text[: _MARKER_ARGUMENT_CHARS - 1].rstrip() + "…"
    return text


def _quoted(value: object) -> str:
    """A model-chosen argument, on one line and shortened, in quotes."""
    return f'"{_one_line(value)}"'


def tool_marker(call: ToolCall) -> str:
    """The line shown while a tool runs, e.g. `[looking up "Mount Kilimanjaro" on wikipedia]`.

    `fetch_page` shows only the host of the address; `time` and `weather` show the
    place, e.g. `[checking the time in Tokyo]`. Other tools show their name and their first
    text argument, if any.
    """
    args = call.arguments
    query = args.get("query")
    if call.name == "wikipedia" and isinstance(query, str) and query.strip():
        return f"[looking up {_quoted(query)} on wikipedia]"
    if call.name == "web_search" and isinstance(query, str) and query.strip():
        return f"[searching the web for {_quoted(query)}]"
    place = args.get("place")
    if call.name == "time" and isinstance(place, str) and place.strip():
        return f"[checking the time in {_one_line(place)}]"
    if call.name == "weather" and isinstance(place, str) and place.strip():
        return f"[checking the weather in {_one_line(place)}]"
    if call.name == "fetch_page":
        url = args.get("url")
        try:
            host = urlsplit(url).hostname if isinstance(url, str) else None
        except ValueError:
            host = None
        return f"[reading {host}]" if host else "[reading a web page]"
    text = next((v for v in args.values() if isinstance(v, str) and v.strip()), None)
    return f"[using {call.name}: {_quoted(text)}]" if text else f"[using {call.name}]"


def _jobs(n: int) -> str:
    return f"{n} background job{'' if n == 1 else 's'}"


def memory_banner(app: LucyApp) -> str:
    """One line saying whether Lucy's memory is active."""
    if app.learning_active:
        return (
            "Memory: active. Lucy learns from the conversation in the background and"
            " remembers across sessions (see `lucy memory --help`)."
        )
    return f"Memory: off ({app.learning_off_reason}). Lucy sees only this session's conversation."


def backlog_message(backlog: JobBacklog | None) -> str | None:
    """The exit line about background work left over, or None if there is none."""
    if backlog is None or (not backlog.pending and backlog.interrupted is None):
        return None
    interrupted = f" (a {backlog.interrupted} job was interrupted)" if backlog.interrupted else ""
    return (
        f"{_jobs(backlog.pending)} left{interrupted}; they resume next time,"
        " or run `lucy memory work`."
    )


# --- stdin --------------------------------------------------------------------


class StdinReader:
    """Reads lines from a file descriptor in a daemon thread and hands them to asyncio.

    Why a thread, and why `os.read`: a blocking read cannot be cancelled, and
    `loop.add_reader` does not work when stdin is a regular file on macOS
    (kqueue). A thread works for terminals, pipes and files alike. It is a
    daemon so a read still blocked at exit never keeps the process alive, and
    it reads the raw descriptor with `os.read` rather than `sys.stdin`, so it
    never holds the `sys.stdin` buffer lock that interpreter shutdown would
    otherwise wait for ("could not acquire lock ... at interpreter shutdown").
    Lines are read ahead and queued; the chat only takes the next one when the
    previous turn is over.
    """

    def __init__(self, fd: int = 0, encoding: str = "utf-8") -> None:
        self._fd = fd
        self._encoding = encoding
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._eof = False

    def start(self) -> None:
        """Start the reader thread. Must be called from the event loop."""
        self._loop = asyncio.get_running_loop()
        threading.Thread(target=self._pump, name="lucy-stdin", daemon=True).start()

    async def readline(self) -> str | None:
        """The next line without its line ending, or None at end of input."""
        if self._eof:
            return None
        line = await self._queue.get()
        if line is None:
            self._eof = True
        return line

    def _pump(self) -> None:
        decoder = codecs.getincrementaldecoder(self._encoding)(errors="replace")
        buffer = ""
        try:
            while True:
                chunk = os.read(self._fd, 65536)
                if not chunk:
                    break
                buffer += decoder.decode(chunk)
                *lines, buffer = buffer.split("\n")
                for line in lines:
                    self._put(line.removesuffix("\r"))
            buffer += decoder.decode(b"", final=True)
            if buffer:
                self._put(buffer.removesuffix("\r"))
        except OSError:
            pass
        finally:
            self._put(None)

    def _put(self, line: str | None) -> None:
        assert self._loop is not None
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, line)
        except RuntimeError:  # the loop has closed; nobody is listening any more
            pass


class TerminalReader:
    """Reads lines at an interactive terminal with line editing and input history.

    Why prompt_toolkit rather than the standard library's `readline` (with
    `input()` in a thread): a blocking `input()` cannot be cancelled, so a
    Ctrl-C or a quit while it waits would leave the thread blocked with the
    terminal still in readline's raw mode, and `input()` reads through
    `sys.stdin`, whose buffer lock interpreter shutdown then waits for (see
    `StdinReader`). macOS Pythons also ship libedit instead of GNU readline,
    which behaves differently. prompt_toolkit's `prompt_async` runs on the event
    loop, is cancellable, and restores the terminal when it returns or is
    cancelled. It is used only here, for an interactive terminal; piped input
    keeps `StdinReader`.

    How the existing behaviour is kept: while the prompt is shown the terminal
    is in raw mode, so Ctrl-C arrives as a key, not as SIGINT; prompt_toolkit
    raises `KeyboardInterrupt` for it and `EOFError` for Ctrl-D on an empty
    line, and both mean "end of input" (the chat exits). Between prompts the
    terminal is back in normal mode, so Ctrl-C during a reply is SIGINT and
    reaches the chat's own handler. `handle_sigint=False` stops prompt_toolkit
    from installing (and on return removing) its own SIGINT handler, which
    would remove the chat's; `set_exception_handler=False` leaves the loop's
    exception handler alone.
    """

    def __init__(
        self,
        prompt: str = PROMPT,
        history_file: Path | None = None,
        *,
        input: Any = None,
        output: Any = None,
    ) -> None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory, History, InMemoryHistory

        history: History = InMemoryHistory()
        if history_file is not None:
            try:
                history_file.parent.mkdir(parents=True, exist_ok=True)
                history_file.touch()
                history = FileHistory(history_file)
            except OSError as exc:  # unwritable: keep history for this run only
                logging.getLogger(__name__).warning(
                    "cannot use the history file %s: %s", history_file, exc
                )
        self._prompt = prompt
        self._session: Any = PromptSession(history=history, input=input, output=output)

    async def readline(self) -> str | None:
        """The next line, edited at the prompt, or None on Ctrl-D or Ctrl-C."""
        try:
            line: str = await self._session.prompt_async(
                self._prompt, handle_sigint=False, set_exception_handler=False
            )
        except (EOFError, KeyboardInterrupt):
            return None
        return line


def chat_history_path(settings: Settings) -> Path:
    """Where `lucy chat` keeps input history: beside the SQLite database, else in data/."""
    path = settings.memory.options.get("path") if settings.memory is not None else None
    if isinstance(path, str) and path and path != ":memory:":
        return Path(path).parent / HISTORY_FILE_NAME
    return Path("data") / HISTORY_FILE_NAME


# --- Logging ------------------------------------------------------------------


class _QuietFormatter(logging.Formatter):
    """Formats records without tracebacks (non-verbose mode)."""

    def format(self, record: logging.LogRecord) -> str:
        return f"{record.levelname.lower()}: {record.getMessage()}"


def _not_a_turn_failure(record: logging.LogRecord) -> bool:
    """Drop the orchestrator's "turn failed" record: the chat prints that error itself."""
    return not (record.name == "lucy.core.orchestrator" and record.msg == "turn %s failed")


def configure_logging(verbose: bool, stream: TextIO | None = None) -> None:
    """Warnings and above, one line each, by default; everything from Lucy with `verbose`."""
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    if verbose:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        )
        root.setLevel(logging.DEBUG)
        for noisy in ("httpcore", "httpx", "aiosqlite", "asyncio"):
            logging.getLogger(noisy).setLevel(logging.INFO)
    else:
        handler.setFormatter(_QuietFormatter())
        handler.addFilter(_not_a_turn_failure)
        root.setLevel(logging.WARNING)


# --- Entry point --------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, help=f"settings file (default: ./{DEFAULT_CONFIG})")
    common.add_argument("--verbose", action="store_true", help="log info and debug messages")

    parser = argparse.ArgumentParser(prog="lucy", description="Lucy, a local personal assistant.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    chat = commands.add_parser("chat", parents=[common], help="chat with Lucy in the terminal")
    chat.add_argument(
        "--session",
        default=DEFAULT_SESSION,
        help=f"conversation to continue or start (default: {DEFAULT_SESSION!r})",
    )
    voice_cli.add_parser(commands, common)

    memory = commands.add_parser(
        "memory",
        help="inspect and control what Lucy has learned",
        description="Inspect and control Lucy's memory. Ids are shown as #12.",
    )
    actions = memory.add_subparsers(dest="action", required=True, metavar="ACTION")
    actions.add_parser(
        "status", parents=[common], help="counts of episodes, statements, entities and jobs"
    )
    profile = actions.add_parser("profile", parents=[common], help="the current profile")
    profile.add_argument("--history", action="store_true", help="list every version")
    search = actions.add_parser(
        "search", parents=[common], help="what retrieval would give a turn for QUERY"
    )
    search.add_argument("query", metavar="QUERY")
    search.add_argument("--limit", type=int, default=20, help="statements shown (default: 20)")
    about = actions.add_parser(
        "about", parents=[common], help="an entity's summary, aliases and statements"
    )
    about.add_argument("name", metavar="NAME")
    why = actions.add_parser(
        "why", parents=[common], help="a statement in full, with the episodes it came from"
    )
    why.add_argument("id", type=int, metavar="ID", help="statement id")
    correct = actions.add_parser(
        "correct", parents=[common], help="replace a statement with a corrected one"
    )
    correct.add_argument("id", type=int, metavar="ID", help="statement id")
    correct.add_argument("text", metavar="NEW_TEXT", help="the corrected statement, a sentence")
    forget = actions.add_parser(
        "forget",
        parents=[common],
        help="delete statements and the conversation turns they came from",
    )
    forget.add_argument("ids", type=int, nargs="*", metavar="ID", help="statement ids")
    forget.add_argument("--query", metavar="TEXT", help="forget the statements matching TEXT")
    forget.add_argument(
        "--limit",
        type=int,
        default=memory_cli.FORGET_QUERY_LIMIT,
        help=f"statements --query collects at most (default: {memory_cli.FORGET_QUERY_LIMIT})",
    )
    forget.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    work = actions.add_parser(
        "work", parents=[common], help="run background jobs now, until none is due"
    )
    work.add_argument(
        "--maintenance",
        action="store_true",
        help="also run all maintenance (profile, summaries, ...) now, after the queue drains",
    )
    rebuild = actions.add_parser(
        "rebuild", parents=[common], help="clear the graph and re-extract the whole episode log"
    )
    rebuild.add_argument("--yes", action="store_true", help="do not ask for confirmation")

    evaluate = commands.add_parser(
        "eval",
        parents=[common],
        help="run the memory eval scenarios against the configured models",
        description=(
            "Run scripted memory scenarios against the real configured models, each on a"
            " temporary database with simulated time, and report scores. Exits non-zero"
            " if any scenario fails."
        ),
    )
    evaluate.add_argument(
        "--scenario", action="append", default=[], metavar="NAME", help="run only NAME (repeatable)"
    )
    evaluate.add_argument("--category", metavar="NAME", help="run only scenarios of this category")
    evaluate.add_argument("--list", action="store_true", help="list the scenarios and exit")
    evaluate.add_argument(
        "--compare",
        nargs="+",
        type=Path,
        metavar="REPORT",
        help="OLD.json: run, then print what changed since OLD; OLD.json NEW.json: only compare",
    )
    evaluate.add_argument(
        "--out",
        type=Path,
        default=Path("data/evals"),
        help="directory for the JSON report (default: data/evals)",
    )
    return parser


def _load_settings(config: Path | None) -> Settings:
    if config is None and not DEFAULT_CONFIG.exists():
        raise ConfigError(
            f"no {DEFAULT_CONFIG} in the current directory; "
            "copy lucy.example.toml to lucy.toml and adjust it, or pass --config PATH"
        )
    return load_settings(config if config is not None else DEFAULT_CONFIG)


async def _chat(settings: Settings, session_id: str) -> int:
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    read_line: ReadLine
    if interactive:  # line editing and history (see `TerminalReader`)
        read_line = TerminalReader(PROMPT, chat_history_path(settings)).readline
    else:
        reader = StdinReader(sys.stdin.fileno())
        reader.start()
        read_line = reader.readline
    loop = asyncio.get_running_loop()
    async with LucyApp(settings) as app:
        chat = ChatLoop(
            app,
            session_id,
            read_line=read_line,
            out=sys.stdout,
            err=sys.stderr,
            echo_input=not sys.stdin.isatty(),
            reader_prompts=interactive,
        )
        # Ctrl-C is handled by the event loop, not as KeyboardInterrupt: the handler
        # runs on the loop between callbacks, so it can publish an Interrupt or wake
        # the prompt safely, and the blocking stdin read in the reader thread never
        # needs to be interrupted.
        try:
            loop.add_signal_handler(signal.SIGINT, chat.interrupt)
            handler_installed = True
        except (NotImplementedError, RuntimeError):  # e.g. Windows: Ctrl-C ends the chat
            handler_installed = False
        try:
            code = await chat.run()
        finally:
            if handler_installed:
                loop.remove_signal_handler(signal.SIGINT)
    if (message := backlog_message(app.backlog)) is not None:
        print(message, file=sys.stderr)
    return code


def _ask(prompt: str) -> bool:
    """Ask on the terminal; anything but y/yes (or end of input) is no."""
    try:
        answer = input(prompt)
    except EOFError:
        print(file=sys.stdout)
        return False
    return answer.strip().lower() in {"y", "yes"}


async def run_memory_command(app: LucyApp, args: argparse.Namespace, out: TextIO) -> int:
    """Run one `lucy memory` action against a started app (built with background=False)."""
    confirm: memory_cli.Confirm = (lambda _: True) if getattr(args, "yes", False) else _ask
    try:
        match args.action:
            case "status":
                return await memory_cli.status(app, out)
            case "profile":
                return await memory_cli.profile(app, out, history=args.history)
            case "search":
                return await memory_cli.search(app, out, args.query, limit=args.limit)
            case "about":
                return await memory_cli.about(app, out, args.name)
            case "why":
                return await memory_cli.why(app, out, args.id)
            case "correct":
                return await memory_cli.correct(app, out, args.id, args.text)
            case "forget":
                return await memory_cli.forget(
                    app, out, args.ids, query=args.query, limit=args.limit, confirm=confirm
                )
            case "work":
                return await memory_cli.work(app, out, maintenance=args.maintenance)
            case "rebuild":
                return await memory_cli.rebuild(app, out, confirm=confirm)
    except memory_cli.NotConfigured as exc:
        print(f"lucy memory: {exc}", file=out)
        return 2
    raise AssertionError(f"unknown memory action {args.action!r}")


async def _memory(settings: Settings, args: argparse.Namespace) -> int:
    async with LucyApp(settings, background=False) as app:
        return await run_memory_command(app, args, sys.stdout)


def _eval_without_running(args: argparse.Namespace) -> int | None:
    """`lucy eval --list` and `--compare OLD NEW`, which need no settings; else None."""
    from lucy.evals import command, report
    from lucy.evals.scenario import ScenarioError, load_scenarios

    if args.list:
        try:
            scenarios = command.select(load_scenarios(), args.scenario, args.category)
        except (ScenarioError, ValueError) as exc:
            print(f"lucy eval: {exc}", file=sys.stderr)
            return 2
        command.list_scenarios(scenarios, sys.stdout)
        return 0
    if args.compare is not None and len(args.compare) > 2:
        print("lucy eval: --compare takes OLD.json, or OLD.json NEW.json", file=sys.stderr)
        return 2
    if args.compare is not None and len(args.compare) == 2:
        old, new = (json.loads(p.read_text(encoding="utf-8")) for p in args.compare)
        print(report.compare_reports(old, new))
        return 0
    return None


async def _eval(settings: Settings, args: argparse.Namespace) -> int:
    from lucy.evals.command import run_eval

    config = str(args.config if args.config is not None else DEFAULT_CONFIG)
    try:
        return await run_eval(
            settings,
            sys.stdout,
            names=args.scenario,
            category=args.category,
            config=config,
            out_dir=args.out,
            compare=args.compare[0] if args.compare else None,
        )
    except ValueError as exc:
        print(f"lucy eval: {exc}", file=sys.stderr)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the `lucy` command. Returns the process exit code."""
    args = _parser().parse_args(argv)
    configure_logging(args.verbose)
    if args.command == "eval":
        try:
            done = _eval_without_running(args)
        except (OSError, ValueError) as exc:
            print(f"lucy eval: {exc}", file=sys.stderr)
            return 2
        if done is not None:
            return done
    if args.command == "voice" and args.list_devices:
        return voice_cli.list_devices(sys.stdout, sys.stderr)
    try:
        settings = _load_settings(args.config)
    except ConfigError as exc:
        print(f"lucy: {exc}", file=sys.stderr)
        return 2
    try:
        if args.command == "memory":
            return asyncio.run(_memory(settings, args))
        if args.command == "eval":
            return asyncio.run(_eval(settings, args))
        if args.command == "voice":
            return asyncio.run(voice_cli.run_voice(settings, args, out=sys.stdout, err=sys.stderr))
        return asyncio.run(_chat(settings, args.session))
    except KeyboardInterrupt:  # Ctrl-C while starting or stopping
        print(file=sys.stderr)
        return 130
    except Exception as exc:
        if args.verbose:
            logging.getLogger(__name__).exception("lucy failed")
        print(f"lucy: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
