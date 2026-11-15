"""`lucy speakers`: enrol, list, remove and test speakers (ADR 0021).

Also `build_tagger`, which the voice clients use to tag utterances when
`[speakers] enabled` is set and the model weights are present.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, TextIO

from lucy.core.config import Settings
from lucy.core.ports import SpeakerIdentifier
from lucy.core.workspace import SPEAKERS, Workspace, WorkspaceError, check_name

MISSING_WEIGHTS_HINT: Final = (
    "speaker identification is off: the model is not at {path};"
    " run `lucy speakers download` to fetch it"
)


def add_parser(commands: Any, common: argparse.ArgumentParser) -> None:
    speakers = commands.add_parser(
        "speakers", parents=[common], help="enrol voices so voice turns record who spoke"
    )
    actions = speakers.add_subparsers(dest="speakers_action", required=True, metavar="ACTION")
    actions.add_parser("download", parents=[common], help="download the speaker model (~27 MB)")
    enroll = actions.add_parser(
        "enroll", parents=[common], help="enrol NAME from audio files (copied to the workspace)"
    )
    enroll.add_argument("name")
    enroll.add_argument("files", nargs="+", type=Path)
    actions.add_parser("list", parents=[common], help="enrolled speakers")
    remove = actions.add_parser("remove", parents=[common], help="forget an enrolled speaker")
    remove.add_argument("name")
    identify = actions.add_parser(
        "identify", parents=[common], help="rank enrolled speakers against an audio file"
    )
    identify.add_argument("file", type=Path)


def make_identifier(settings: Settings) -> Any:
    from lucy.adapters.speaker_voicefingerprint import BOOK_NAME, VoiceFingerprintIdentifier

    workspace = Workspace(settings.workspace.path)
    return VoiceFingerprintIdentifier(
        weights=settings.speakers.weights,
        book_path=workspace.dir(SPEAKERS) / BOOK_NAME,
        threshold=settings.speakers.threshold,
    )


def build_enroller(settings: Settings, tagger: Any) -> Any:
    """The voice session's `Enroller`, sharing `tagger`'s identifier; None when there
    is no tagger or neither enrolment tool is enabled."""
    from lucy.tools.voice_enrolment import ENROL_TOOLS
    from lucy.voice.enrol import Enroller

    if tagger is None or not ENROL_TOOLS & set(settings.tools.enabled):
        return None
    return Enroller(
        tagger.identifier, tagger.workspace, sentences=settings.speakers.enrol_sentences
    )


def build_tagger(
    settings: Settings,
    notice: Callable[[str], None],
    identifier: SpeakerIdentifier | None = None,
) -> Any:
    """A `SpeakerTagger`, or None when disabled or the weights are missing (one notice)."""
    from lucy.voice.speakers import SpeakerTagger

    config = settings.speakers
    if not config.enabled:
        return None
    if identifier is None:
        if not Path(config.weights).is_file():
            notice(MISSING_WEIGHTS_HINT.format(path=config.weights))
            return None
        identifier = make_identifier(settings)
    return SpeakerTagger(
        identifier,
        Workspace(settings.workspace.path),
        wait_ms=config.wait_ms,
        keep_days=config.keep_utterances_days,
        max_mb=config.max_utterances_mb,
    )


def run(
    settings: Settings,
    args: argparse.Namespace,
    *,
    out: TextIO,
    err: TextIO,
    identifier: SpeakerIdentifier | None = None,
    download: Callable[[Path], Path] | None = None,
) -> int:
    """Run one `lucy speakers` action; `identifier` and `download` are for tests."""
    action = args.speakers_action
    weights = Path(settings.speakers.weights)
    if action == "download":
        from lucy.adapters.speaker_voicefingerprint import download_weights

        print(f"Downloading the WeSpeaker ResNet34 model into {weights.parent}", file=err)
        try:
            path = (download or download_weights)(weights.parent)
        except Exception as exc:
            print(f"lucy speakers: download failed: {exc}", file=err)
            return 1
        print(path, file=out)
        return 0
    if identifier is None:
        if not weights.is_file():
            print(f"lucy speakers: {MISSING_WEIGHTS_HINT.format(path=weights)}", file=err)
            return 1
        identifier = make_identifier(settings)
    workspace = Workspace(settings.workspace.path)
    try:
        if action == "enroll":
            name = check_name(args.name)
            missing = [f for f in args.files if not f.is_file()]
            if missing:
                print(f"lucy speakers: no such file: {missing[0]}", file=err)
                return 2
            target = workspace.dir(SPEAKERS, name)
            copies = []
            for source in args.files:
                copy = target / source.name
                if source.resolve() != copy.resolve():
                    shutil.copyfile(source, copy)
                    copy.chmod(0o600)
                copies.append(copy)
            identifier.enroll(name, copies)
            print(f"Enrolled {name} from {len(copies)} clip(s).", file=out)
            return 0
        if action == "list":
            speakers = identifier.speakers()
            if not speakers:
                print("No speakers enrolled.", file=out)
            for name, clips in speakers.items():
                print(f"{name}\t{clips} clip(s)", file=out)
            return 0
        if action == "remove":
            name = check_name(args.name)
            if name not in identifier.speakers():
                print(f"lucy speakers: {name} is not enrolled", file=err)
                return 1
            identifier.remove(name)
            shutil.rmtree(workspace.resolve(Path(SPEAKERS) / name), ignore_errors=True)
            print(f"Removed {name}.", file=out)
            return 0
        if action == "identify":
            matches = identifier.identify(args.file)
            if not matches:
                print("No speakers enrolled.", file=out)
            for match in matches:
                verdict = "match" if match.accepted else "-"
                print(f"{match.name}\t{match.score:.3f}\t{verdict}", file=out)
            return 0
    except WorkspaceError as exc:
        print(f"lucy speakers: {exc}", file=err)
        return 2
    print(f"lucy speakers: unknown action {action}", file=err)
    return 2
