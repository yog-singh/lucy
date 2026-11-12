"""`lucy voice`: talk with Lucy through the microphone and speaker (stage 1, half-duplex).

It starts a `LucyApp` with speech (`speech=True`), opens the sound devices and
runs a `VoiceSession` on the same session as `lucy chat` (`"cli"` unless
`--session` is given), so voice and text continue one conversation. The
transcript and the reply are printed as in chat; a state indicator
(listening, thinking, speaking) is drawn on stderr when it is a terminal.

Keys: Enter stops Lucy while she is thinking or speaking; Ctrl-C, Ctrl-D or
typing q then Enter quits, stopping audio at once.

File mode (`--input-wav FILE ... --output-wav FILE`) runs the same session
over audio files through `FileAudioIO` instead of the devices, in real time
(or simulated time with `--fast`), and exits once the input is exhausted and
the last reply has been spoken. `--wait-for-reply` makes each file one user
turn that starts only after Lucy has answered the previous one.

This module's top level imports nothing heavy: `lucy chat` imports it to build
the argument parser, and must not load numpy, onnxruntime or the audio stack.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TextIO

if TYPE_CHECKING:
    from lucy.app import LucyApp
    from lucy.core.config import Settings
    from lucy.core.registry import Registry
    from lucy.voice.audio import AudioIO
    from lucy.voice.session import VoiceState

logger = logging.getLogger(__name__)

DEFAULT_SESSION: Final = "cli"
QUIT_KEYS: Final = frozenset({"q", "quit", "/quit", "/exit"})


def add_parser(commands: Any, common: argparse.ArgumentParser) -> None:
    """Register the `voice` subcommand on the `lucy` parser's subcommands."""
    voice = commands.add_parser(
        "voice",
        parents=[common],
        help="talk with Lucy using the microphone and speaker",
        description=(
            "Talk with Lucy. She listens, answers aloud, and ignores the microphone while she"
            " speaks. Press Enter to stop her; Ctrl-C or Ctrl-D quits."
        ),
    )
    voice.add_argument(
        "--session",
        default=DEFAULT_SESSION,
        help=f"conversation to continue or start (default: {DEFAULT_SESSION!r}, as in chat)",
    )
    voice.add_argument(
        "--list-devices", action="store_true", help="list microphones and speakers, then exit"
    )
    voice.add_argument(
        "--input-wav",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="read speech from WAV files instead of the microphone (test and demo mode)",
    )
    voice.add_argument(
        "--output-wav",
        type=Path,
        metavar="FILE",
        help="with --input-wav: write what Lucy says to this WAV file instead of the speaker",
    )
    voice.add_argument(
        "--fast",
        action="store_true",
        help="with --input-wav: simulated time instead of real time (latencies are not real)",
    )
    voice.add_argument(
        "--wait-for-reply",
        action="store_true",
        help="with --input-wav: each file is one turn, started after Lucy has answered",
    )


def add_listen_parser(commands: Any, common: argparse.ArgumentParser) -> None:
    """Register the `listen` subcommand: `lucy voice` behind a wake phrase."""
    listen = commands.add_parser(
        "listen",
        parents=[common],
        help="wait for the wake phrase, then talk with Lucy",
        description=(
            "Wait in the background for the wake phrase ([wake] phrase, 'hey jarvis' by"
            " default), then talk as in `lucy voice`. After a reply you can speak again"
            " without the wake phrase for [wake] followup_seconds. Enter stops Lucy while"
            " she is thinking or speaking; Ctrl-C or Ctrl-D quits."
        ),
    )
    listen.add_argument(
        "--session",
        default=DEFAULT_SESSION,
        help=f"conversation to continue or start (default: {DEFAULT_SESSION!r}, as in chat)",
    )
    listen.add_argument(
        "--download-models",
        action="store_true",
        help="download openWakeWord's models (CC BY-NC-SA 4.0, non-commercial), then exit",
    )
    listen.add_argument(
        "--input-wav",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="read sound from WAV files instead of the microphone (test and demo mode)",
    )
    listen.add_argument(
        "--output-wav",
        type=Path,
        metavar="FILE",
        help="with --input-wav: write what Lucy plays to this WAV file instead of the speaker",
    )
    listen.add_argument(
        "--fast",
        action="store_true",
        help="with --input-wav: simulated time instead of real time (latencies are not real)",
    )
    listen.add_argument(
        "--wait-for-reply",
        action="store_true",
        help="with --input-wav: each file starts after Lucy has answered the previous one",
    )


def list_devices(out: TextIO, err: TextIO, sd: Any = None) -> int:
    """`lucy voice --list-devices`."""
    from lucy.voice.audio import AudioDeviceError
    from lucy.voice.audio import list_devices as describe

    try:
        print(describe(sd), file=out)
    except AudioDeviceError as exc:
        print(f"lucy voice: {exc}", file=err)
        return 1
    return 0


def timings_path(settings: Settings) -> Path:
    """Where turn timings are appended: beside the SQLite database, else in data/."""
    from lucy.clients.cli import chat_history_path
    from lucy.voice.session import TIMINGS_FILE_NAME

    return chat_history_path(settings).parent / TIMINGS_FILE_NAME


class TerminalUI:
    """Prints the conversation to `out`, and a state indicator to `err` if `show_state`."""

    def __init__(self, out: TextIO, err: TextIO, *, show_state: bool) -> None:
        self._out = out
        self._err = err
        self._show_state = show_state
        self._line_open = False
        self._indicator = False
        self._state: VoiceState | None = None

    def _clear(self) -> None:
        if self._indicator:
            self._err.write("\r\x1b[K")
            self._err.flush()
            self._indicator = False

    def _draw(self) -> None:
        if self._show_state and self._state is not None and not self._line_open:
            self._err.write(f"\r\x1b[K({self._state.value})")
            self._err.flush()
            self._indicator = True

    def _write(self, text: str) -> None:
        self._out.write(text)
        self._out.flush()

    def state(self, state: VoiceState) -> None:
        self._state = state
        self._draw()

    def user(self, text: str) -> None:
        self._clear()
        self._write(f"you> {text}\n")
        self._draw()

    def reply_delta(self, text: str) -> None:
        if not self._line_open:
            text = text.lstrip()
            if not text:
                return
            self._clear()
            self._write("lucy> ")
            self._line_open = True
        self._write(text)

    def reply_end(self, outcome: str, error: str | None) -> None:
        self._clear()
        if outcome == "interrupted":
            self._write(" [interrupted]\n" if self._line_open else "lucy> [interrupted]\n")
        elif self._line_open:
            self._write("\n")
        self._line_open = False
        if error:
            self._err.write(f"error: {error}\n")
            self._err.flush()
        self._draw()

    def notice(self, text: str) -> None:
        self._clear()
        if self._line_open:
            self._write("\n")
            self._line_open = False
        self._err.write(text + "\n")
        self._err.flush()
        self._draw()


@contextlib.contextmanager
def _no_echo(fd: int) -> Iterator[None]:
    """Stop the terminal echoing keys (Enter would print blank lines); restore on exit."""
    try:
        import termios
    except ImportError:  # not a POSIX terminal
        yield
        return
    try:
        saved = termios.tcgetattr(fd)
    except termios.error:
        yield
        return
    quiet = list(saved)
    quiet[3] &= ~termios.ECHO
    try:
        termios.tcsetattr(fd, termios.TCSANOW, quiet)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, saved)


def _check_args(args: argparse.Namespace) -> str | None:
    if args.input_wav is None:
        for flag, value in (
            ("--output-wav", args.output_wav),
            ("--fast", args.fast),
            ("--wait-for-reply", args.wait_for_reply),
        ):
            if value:
                return f"{flag} needs --input-wav"
        return None
    missing = [str(p) for p in args.input_wav if not p.is_file()]
    if missing:
        return f"no such input file: {', '.join(missing)}"
    return None


async def run_voice(
    settings: Settings,
    args: argparse.Namespace,
    *,
    out: TextIO,
    err: TextIO,
    registry: Registry | None = None,
    make_vad: Callable[[], Any] | None = None,
    make_audio: Callable[[], AudioIO] | None = None,
    interactive: bool | None = None,
) -> int:
    """Run `lucy voice` with parsed `args`; returns the exit code.

    `registry`, `make_vad` and `make_audio` are for tests (fakes instead of the
    real models and devices). `interactive` (default: stdin and stderr are
    terminals, and not in file mode) enables the keyboard and state indicator.
    """
    from lucy.app import LucyApp
    from lucy.clients.cli import backlog_message
    from lucy.voice.audio import AudioDeviceError, FileAudioIO, SoundDeviceIO
    from lucy.voice.vad import SileroVAD

    problem = _check_args(args)
    if problem is not None:
        print(f"lucy voice: {problem}", file=err)
        return 2
    file_mode = args.input_wav is not None
    if interactive is None:
        interactive = not file_mode and sys.stdin.isatty() and err.isatty()

    def default_audio() -> AudioIO:
        if file_mode:
            return FileAudioIO(
                args.input_wav,
                realtime=not args.fast,
                output_path=args.output_wav,
                wait_for_reply=args.wait_for_reply,
            )
        voice = settings.voice
        return SoundDeviceIO(input_device=voice.input_device, output_device=voice.output_device)

    try:
        io = (make_audio or default_audio)()
    except ValueError as exc:  # unreadable input file
        print(f"lucy voice: {exc}", file=err)
        return 2
    vad = (make_vad or SileroVAD)()
    code = 0
    async with LucyApp(settings, registry=registry, speech=True) as app:
        try:
            if isinstance(io, SoundDeviceIO):
                await io.start()
            load = getattr(vad, "load", None)
            if load is not None:
                load()
            code = await _converse(app, io, vad, args, out=out, err=err, interactive=interactive)
        except AudioDeviceError as exc:
            print(f"lucy voice: {exc}", file=err)
            code = 1
        finally:
            await io.close()
    if (message := backlog_message(app.backlog)) is not None:
        print(message, file=err)
    return code


async def _converse(
    app: LucyApp,
    io: AudioIO,
    vad: Any,
    args: argparse.Namespace,
    *,
    out: TextIO,
    err: TextIO,
    interactive: bool,
) -> int:
    from lucy.clients.cli import memory_banner
    from lucy.voice.session import VoiceConfig, VoiceSession

    ui = TerminalUI(out, err, show_state=interactive)
    session = VoiceSession(
        bus=app.bus,
        session_id=args.session,
        io=io,
        vad=vad,
        stt=app.stt,
        tts=app.tts,
        config=VoiceConfig.from_settings(app.settings.voice),
        ui=ui,
        timings_path=timings_path(app.settings),
    )
    earlier = len(await app.memory.recent_episodes(args.session, 1_000_000))
    print(f"Session {args.session!r} ({earlier} earlier events).", file=err)
    print(memory_banner(app), file=err)
    print(f"Audio: {io.describe()}", file=err)
    if interactive:
        print(
            "Speak after '(listening)'. Enter stops Lucy while she thinks or speaks;"
            " Ctrl-C or Ctrl-D quits.",
            file=err,
        )
    else:
        print("Ctrl-C quits.", file=err)
    err.flush()

    await _run_with_keys(session.run(), session.interrupt, session.stop, err, interactive)
    return 0


async def _run_with_keys(
    run: Any,
    interrupt: Callable[[], object],
    stop: Callable[[], None],
    err: TextIO,
    interactive: bool,
) -> None:
    """Await `run` with Ctrl-C bound to `stop` and, if `interactive`, Enter to
    `interrupt` and q or Ctrl-D to `stop`."""
    from lucy.clients.cli import StdinReader

    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, stop)
        handler_installed = True
    except (NotImplementedError, RuntimeError):
        handler_installed = False
    keys: asyncio.Task[None] | None = None
    echo = _no_echo(sys.stdin.fileno()) if interactive else contextlib.nullcontext()
    try:
        with echo:
            if interactive:
                reader = StdinReader(sys.stdin.fileno())
                reader.start()

                async def read_keys() -> None:
                    while (line := await reader.readline()) is not None:
                        if line.strip().lower() in QUIT_KEYS:
                            break
                        interrupt()
                    stop()

                keys = asyncio.create_task(read_keys(), name="lucy-voice-keys")
            await run
    finally:
        if keys is not None:
            keys.cancel()
            await asyncio.gather(keys, return_exceptions=True)
        if handler_installed:
            loop.remove_signal_handler(signal.SIGINT)
        if interactive:
            err.write("\r\x1b[K")
            err.flush()


# --- lucy listen ------------------------------------------------------------------------

MISSING_MODELS_HINT: Final = (
    "wake-word models not found in {directory}; run `lucy listen --download-models`"
    " (openWakeWord's pretrained models, CC BY-NC-SA 4.0: non-commercial use only)"
)


async def download_wake_models(
    model: str | Path, *, out: TextIO, err: TextIO, transport: Any = None
) -> int:
    """`lucy listen --download-models`: fetch the models into the phrase model's directory."""
    from lucy.voice.wakeword import download_models

    directory = Path(model).parent
    print(
        "Downloading openWakeWord's models (licensed CC BY-NC-SA 4.0: non-commercial use"
        f" only) into {directory}",
        file=err,
    )
    try:
        await download_models(directory, transport=transport, report=lambda s: print(s, file=out))
    except Exception as exc:
        print(f"lucy listen: download failed: {exc}", file=err)
        return 1
    return 0


async def run_listen(
    settings: Settings,
    args: argparse.Namespace,
    *,
    out: TextIO,
    err: TextIO,
    registry: Registry | None = None,
    make_vad: Callable[[], Any] | None = None,
    make_audio: Callable[[], AudioIO] | None = None,
    make_detector: Callable[[], Any] | None = None,
    interactive: bool | None = None,
) -> int:
    """Run `lucy listen` with parsed `args`; returns the exit code.

    While waiting only the audio input and the wake-word models run; the app (with
    the speech recogniser and synthesiser) and the VAD start on the first wake and
    stay up. `registry`, `make_vad`, `make_audio` and `make_detector` are for tests.
    """
    from lucy.app import LucyApp
    from lucy.clients.cli import backlog_message, memory_banner
    from lucy.voice.audio import AudioDeviceError, FileAudioIO, SoundDeviceIO
    from lucy.voice.session import VoiceConfig, VoiceSession
    from lucy.voice.vad import SileroVAD
    from lucy.voice.wake import ListenerConfig, WakeListener, WakePhraseSTT, WakeStage
    from lucy.voice.wakeword import OpenWakeWordDetector, missing_models

    wake = settings.wake
    if args.download_models:
        return await download_wake_models(wake.model, out=out, err=err)
    problem = _check_args(args)
    if problem is not None:
        print(f"lucy listen: {problem}", file=err)
        return 2
    if make_detector is None:
        if missing_models(Path(wake.model)):
            print(f"lucy listen: {MISSING_MODELS_HINT.format(directory=Path(wake.model).parent)}",
                  file=err)  # fmt: skip
            return 1
        detector = OpenWakeWordDetector(wake.model)
        detector.load()
    else:
        detector = make_detector()
    file_mode = args.input_wav is not None
    if interactive is None:
        interactive = not file_mode and sys.stdin.isatty() and err.isatty()

    def default_audio() -> AudioIO:
        if file_mode:
            return FileAudioIO(
                args.input_wav,
                realtime=not args.fast,
                output_path=args.output_wav,
                wait_for_reply=args.wait_for_reply,
            )
        voice = settings.voice
        return SoundDeviceIO(input_device=voice.input_device, output_device=voice.output_device)

    try:
        io = (make_audio or default_audio)()
    except ValueError as exc:
        print(f"lucy listen: {exc}", file=err)
        return 2
    ui = TerminalUI(out, err, show_state=interactive)
    app: LucyApp | None = None

    async def make_session(session_io: AudioIO, session_ui: Any) -> VoiceSession:
        nonlocal app
        app = LucyApp(settings, registry=registry, speech=True)
        await app.start()
        vad = (make_vad or SileroVAD)()
        load = getattr(vad, "load", None)
        if load is not None:
            load()
        earlier = len(await app.memory.recent_episodes(args.session, 1_000_000))
        ui.notice(f"Session {args.session!r} ({earlier} earlier events). {memory_banner(app)}")
        return VoiceSession(
            bus=app.bus,
            session_id=args.session,
            io=session_io,
            vad=vad,
            stt=WakePhraseSTT(app.stt, wake.phrase),
            tts=app.tts,
            config=VoiceConfig.from_settings(settings.voice),
            ui=session_ui,
            timings_path=timings_path(settings),
        )

    listener = WakeListener(
        io=io,
        stage=WakeStage(detector, threshold=wake.threshold),
        make_session=make_session,
        ui=ui,
        config=ListenerConfig(
            phrase=wake.phrase, followup_s=wake.followup_seconds, chime=wake.chime
        ),
    )
    print(f"Audio: {io.describe()}", file=err)
    print(
        f"Say {wake.phrase!r} to wake Lucy."
        + (" Enter stops her while she thinks or speaks;" if interactive else "")
        + " Ctrl-C quits.",
        file=err,
    )
    err.flush()
    code = 0
    try:
        if isinstance(io, SoundDeviceIO):
            await io.start()
        await _run_with_keys(listener.run(), listener.interrupt, listener.stop, err, interactive)
    except AudioDeviceError as exc:
        print(f"lucy listen: {exc}", file=err)
        code = 1
    finally:
        await io.close()
        if app is not None:
            await app.stop()
    if app is not None and (message := backlog_message(app.backlog)) is not None:
        print(message, file=err)
    return code
