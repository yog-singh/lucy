"""`lucy wakeword`: record, train and test a personal wake phrase (ADR 0028).

`record` saves takes of the phrase (and `--negative`: everyday audio without it)
from the same microphone as `lucy listen`, `train` runs `lucy.voice.wakeword_train`
and prints the threshold sweep, and `test` shows live scores for tuning the
threshold. The top level imports nothing heavy (`lucy chat` builds this parser).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TextIO

if TYPE_CHECKING:
    from lucy.core.config import Settings
    from lucy.voice.audio import AudioIO

DEFAULT_PHRASE: Final = "hey lucy"
DATA_DIR: Final = Path("data/wakeword")
MODELS_DIR: Final = Path("data/models/wakeword")
READY_S: Final = 1.0  # pause before each take's cue
TIP_EVERY: Final = 10
TIPS: Final = (
    "vary it: step back a metre or two",
    "vary it: say it quietly",
    "vary it: say it quickly",
    "vary it: say it slowly",
    "vary it: say it louder, as if from another room",
    "vary it: turn your head away from the microphone",
    "vary it: say it tired, or with a smile",
)


def add_parser(commands: Any, common: argparse.ArgumentParser) -> None:
    wakeword = commands.add_parser(
        "wakeword",
        parents=[common],
        help="train a personal wake phrase from your own recordings",
        description=(
            "Record takes of the phrase and some everyday audio without it, train a"
            " small model on them, then test it. Use it in `lucy listen` with the"
            " [wake] lines `train` prints."
        ),
    )
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--phrase", default=DEFAULT_PHRASE, help=f"the phrase (default: {DEFAULT_PHRASE!r})"
    )
    shared.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"where recordings are kept, one directory per phrase (default: {DATA_DIR})",
    )
    shared.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help=f"the openWakeWord models, and where the trained model goes (default: {MODELS_DIR})",
    )
    actions = wakeword.add_subparsers(dest="wakeword_action", required=True, metavar="ACTION")
    record = actions.add_parser(
        "record", parents=[common, shared], help="record takes of the phrase (or --negative audio)"
    )
    record.add_argument("--count", type=int, default=80, help="takes to record (default: 80)")
    record.add_argument(
        "--negative",
        action="store_true",
        help="record everyday audio without the phrase instead (one file a minute)",
    )
    record.add_argument(
        "--minutes", type=float, default=10.0, help="with --negative: how long (default: 10)"
    )
    train = actions.add_parser("train", parents=[common, shared], help="train and evaluate")
    train.add_argument(
        "--augmentations",
        type=int,
        default=30,
        help="augmented copies of each take (default: 30)",
    )
    test = actions.add_parser(
        "test", parents=[common, shared], help="show live scores from the microphone"
    )
    test.add_argument("--threshold", type=float, help="default: the one stored by `train`")


def run(
    settings: Settings,
    args: argparse.Namespace,
    *,
    out: TextIO,
    err: TextIO,
    make_audio: Callable[[], AudioIO] | None = None,
    make_detector: Callable[[Path], Any] | None = None,
) -> int:
    """Run one `lucy wakeword` action; `make_audio` and `make_detector` are for tests."""
    from lucy.voice.wakeword_train import slug

    try:
        slug(args.phrase)
    except ValueError as exc:
        print(f"lucy wakeword: {exc}", file=err)
        return 2
    action = args.wakeword_action
    if action == "train":
        return _train(args, out=out, err=err)
    if action == "record" and args.negative:
        if args.minutes <= 0:
            print("lucy wakeword: --minutes must be positive", file=err)
            return 2
        main: Callable[[AudioIO], Any] = lambda io: _record_negative(io, args, out=out, err=err)  # noqa: E731
    elif action == "record":
        if args.count <= 0:
            print("lucy wakeword: --count must be positive", file=err)
            return 2
        main = lambda io: _record_takes(io, args, out=out, err=err)  # noqa: E731
    elif action == "test":
        from lucy.voice.wakeword import OpenWakeWordDetector, missing_models
        from lucy.voice.wakeword_train import model_path

        path = model_path(args.models_dir, args.phrase)
        if make_detector is None:
            missing = missing_models(path)
            if missing:
                print(
                    f"lucy wakeword: missing {', '.join(map(str, missing))}; run"
                    " `lucy wakeword train` (and `lucy listen --download-models` first)",
                    file=err,
                )
                return 1
        detector = (make_detector or OpenWakeWordDetector)(path)
        main = lambda io: _test(io, detector, path, args, out=out, err=err)  # noqa: E731
    else:
        print(f"lucy wakeword: unknown action {action}", file=err)
        return 2
    return asyncio.run(_with_microphone(settings, main, err=err, make_audio=make_audio))


# --- Train ------------------------------------------------------------------------------


def _train(args: argparse.Namespace, *, out: TextIO, err: TextIO) -> int:
    from lucy.voice.wakeword_train import TrainingError, train_phrase

    try:
        result = train_phrase(
            args.phrase,
            args.data_dir,
            args.models_dir,
            augmentations=args.augmentations,
            report=lambda line: print(line, file=err, flush=True),
        )
    except TrainingError as exc:
        print(f"lucy wakeword: {exc}", file=err)
        return 1
    for line in result.skipped:
        print(f"skipped {line}", file=err)
    held_takes = result.takes[1]
    hours = result.rows[0].hours if result.rows else 0.0
    print(f"\nHeld out: {held_takes} take(s), {hours * 60:.1f} min of negative audio.", file=out)
    print("threshold  detected  false wakes  per hour", file=out)
    for row in result.rows:
        mark = "  <- recommended" if row.threshold == result.threshold else ""
        print(
            f"{row.threshold:9.2f}  {row.detected:>3}/{row.takes:<4}  {row.false_wakes:>11}"
            f"  {row.per_hour:8.1f}{mark}",
            file=out,
        )
    seconds = result.seconds
    print(
        f"\nFeatures {seconds['features']:.0f} s, training {seconds['training']:.0f} s"
        f" ({result.log.epochs} epochs), evaluation {seconds['evaluation']:.0f} s.",
        file=out,
    )
    print(f"Saved {result.path}. For `lucy listen`, put this in lucy.toml:\n", file=out)
    print("[wake]", file=out)
    print(f'phrase    = "{args.phrase}"', file=out)
    print(f'model     = "{result.path.as_posix()}"', file=out)
    print(f"threshold = {result.threshold:.2f}", file=out)
    print(
        "\nThen try `lucy wakeword test` to see the scores live and adjust the threshold.",
        file=out,
    )
    return 0


# --- The microphone ---------------------------------------------------------------------


async def _with_microphone(
    settings: Settings,
    main: Callable[[AudioIO], Any],
    *,
    err: TextIO,
    make_audio: Callable[[], AudioIO] | None,
) -> int:
    from lucy.voice.audio import AudioDeviceError, SoundDeviceIO

    def default_audio() -> AudioIO:
        voice = settings.voice
        return SoundDeviceIO(input_device=voice.input_device, output_device=voice.output_device)

    io = (make_audio or default_audio)()
    if isinstance(io, SoundDeviceIO):
        io.on_notice = lambda line: print(line, file=err)
    try:
        if isinstance(io, SoundDeviceIO):
            await io.start()
            print(f"Microphone: {io.describe()}", file=err)
        return await main(io)
    except AudioDeviceError as exc:
        print(f"lucy wakeword: {exc}", file=err)
        return 1
    finally:
        await io.close()


async def _samples(frames: AsyncIterator[Any], seconds: float, *, since: float) -> tuple[Any, bool]:
    """`seconds` of float samples from frames captured after `since`; False if input
    ended first."""
    import numpy as np

    from lucy.voice.audio import pcm_to_float

    wanted = int(seconds * 16000)
    parts, have = [], 0
    async for frame in frames:
        if frame.time < since:
            continue  # captured before the cue
        part = pcm_to_float(frame.audio.data)
        parts.append(part)
        have += len(part)
        if have >= wanted:
            return np.concatenate(parts)[:wanted], True
    return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), False


def _next_number(directory: Path) -> int:
    numbers = [int(p.stem) for p in directory.glob("*.wav") if p.stem.isdigit()]
    return max(numbers, default=0) + 1


def _save(path: Path, samples: Any) -> None:
    from lucy.voice.audio import float_to_pcm, write_wav

    write_wav(path, float_to_pcm(samples), 16000)


async def _record_takes(io: AudioIO, args: argparse.Namespace, *, out: TextIO, err: TextIO) -> int:
    from lucy.voice.wakeword_train import TAKE_SECONDS, check_take, phrase_dir

    directory = phrase_dir(args.data_dir, args.phrase) / "positive"
    directory.mkdir(parents=True, exist_ok=True)
    number = _next_number(directory)
    frames = aiter(io.frames())
    saved = 0
    print(
        f"Recording {args.count} takes into {directory} (adding to the"
        f' {number - 1} there). For each: wait for the cue, then say "{args.phrase}"'
        " once, naturally. Ctrl-C stops; takes saved so far are kept.",
        file=err,
    )
    try:
        while saved < args.count:
            if saved and saved % TIP_EVERY == 0:
                print(f"  ({TIPS[(saved // TIP_EVERY - 1) % len(TIPS)]})", file=err)
            print(f"take {saved + 1}/{args.count}: get ready...", end="", file=err, flush=True)
            await _samples(frames, READY_S, since=io.now())
            print(f"\r>>> say: {args.phrase} <<<            ", end="", file=err, flush=True)
            samples, more = await _samples(frames, TAKE_SECONDS, since=io.now())
            if not more:
                print("\nthe microphone stopped", file=err)
                break
            problem = check_take(samples)
            if problem is not None:
                print(f"\r  again: {problem}                ", file=err)
                continue
            path = directory / f"{number:03d}.wav"
            _save(path, samples)
            number += 1
            saved += 1
            print(f"\r  saved {path.name}                    ", file=err)
    finally:
        print(f"\n{saved} take(s) saved in {directory}.", file=out)
    return 0


async def _record_negative(
    io: AudioIO, args: argparse.Namespace, *, out: TextIO, err: TextIO
) -> int:
    from lucy.voice.wakeword_train import SEGMENT_S, phrase_dir

    directory = phrase_dir(args.data_dir, args.phrase) / "negative"
    directory.mkdir(parents=True, exist_ok=True)
    number = _next_number(directory)
    frames = aiter(io.frames())
    print(
        f"Recording {args.minutes:g} min of everyday audio into {directory}, one file a"
        " minute. Talk normally, read something aloud, leave the TV or music on, but never"
        f' say "{args.phrase}". Ctrl-C stops; what was recorded is kept.',
        file=err,
    )
    total = args.minutes * 60
    done = 0.0
    files = 0
    try:  # a minute interrupted by Ctrl-C is not saved
        while done < total:
            length = min(SEGMENT_S, total - done)
            samples, more = await _samples(frames, length, since=float("-inf"))
            if len(samples) >= 16000 * 3:
                _save(directory / f"{number:03d}.wav", samples)
                number, files = number + 1, files + 1
            done += len(samples) / 16000
            print(f"  {done / 60:.1f} of {args.minutes:g} min", file=err)
            if not more:
                break
    finally:
        print(f"{files} file(s) saved in {directory}.", file=out)
    return 0


# --- Test -------------------------------------------------------------------------------


async def _test(
    io: AudioIO, detector: Any, path: Path, args: argparse.Namespace, *, out: TextIO, err: TextIO
) -> int:
    from lucy.voice.wakeword import NumpyHead

    if args.threshold is not None:
        threshold = args.threshold
    elif path.suffix == ".npz" and path.is_file():
        threshold = NumpyHead.load(path).threshold
    else:
        threshold = 0.5
    if hasattr(detector, "load"):
        detector.load()
    print(
        f'Say "{args.phrase}" now and then. Threshold {threshold:.2f}. Enter or Ctrl-C stops.',
        file=err,
    )
    stop = asyncio.Event()
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    reading = False

    def on_enter() -> None:
        entered.set()
        stop.set()

    with contextlib.suppress(Exception):  # not a terminal, or no stdin
        loop.add_reader(sys.stdin.fileno(), on_enter)
        reading = True

    async def listen() -> None:
        above, peak, since = False, 0.0, None
        async for frame in io.frames():
            score = detector(frame.audio)
            peak = max(peak, score)
            if score >= threshold and not above:
                print(f"{frame.time:9.1f} s  score {score:.2f}  detected", file=out, flush=True)
            above = score >= threshold
            since = frame.time if since is None else since
            if frame.time - since >= 2.0:
                print(f"{frame.time:9.1f} s  peak {peak:.2f}", file=err, flush=True)
                peak, since = 0.0, frame.time
        stop.set()

    task = asyncio.create_task(listen())
    try:
        await stop.wait()
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        if reading:
            loop.remove_reader(sys.stdin.fileno())
        if entered.is_set():
            with contextlib.suppress(Exception):
                sys.stdin.readline()
    return 0
