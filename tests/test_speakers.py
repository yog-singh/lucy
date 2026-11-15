"""Speaker tagging, the workspace and `lucy speakers` (ADR 0021), with a fake identifier."""

from __future__ import annotations

import argparse
import io
import os
import stat
import threading
import time
import wave
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lucy.clients import speakers_cli
from lucy.clients.cli import _parser
from lucy.core.config import Settings, SpeakersSettings, WorkspaceSettings
from lucy.core.context import ContextBuilder
from lucy.core.events import UserMessage, event_from_json
from lucy.core.history import events_to_messages
from lucy.core.ports import SpeakerIdentifier, SpeakerMatch
from lucy.core.types import AudioChunk, MemoryContext
from lucy.core.workspace import UTTERANCES, Workspace, WorkspaceError
from lucy.memory.extraction import render_episode
from lucy.memory.models import EpisodeRecord
from lucy.voice.speakers import SpeakerTagger, prune_clips, write_wav
from tests.test_voice_session import Harness, reply
from tests.voice_fakes import concat, silence, tone


class FakeIdentifier:
    def __init__(self, matches: Sequence[SpeakerMatch] = (), delay: float = 0.0) -> None:
        self.matches = list(matches)
        self.delay = delay
        self.seen: list[Path] = []
        self.enrolled: dict[str, list[Path]] = {}
        self.done = threading.Event()

    def identify(self, path: Path) -> list[SpeakerMatch]:
        assert path.is_file()
        with wave.open(str(path)) as clip:
            assert clip.getnchannels() == 1 and clip.getframerate() == 16000
        self.seen.append(path)
        time.sleep(self.delay)
        self.done.set()
        return list(self.matches)

    def enroll(self, name: str, paths: Sequence[Path]) -> None:
        self.enrolled.setdefault(name, []).extend(paths)

    def speakers(self) -> dict[str, int]:
        return {n: len(p) for n, p in self.enrolled.items()}

    def remove(self, name: str) -> None:
        del self.enrolled[name]


YOGESH = [SpeakerMatch("yogesh", 0.71, True), SpeakerMatch("guest", 0.2, False)]


def test_fake_is_a_speaker_identifier() -> None:
    assert isinstance(FakeIdentifier(), SpeakerIdentifier)


# --- Workspace ----------------------------------------------------------------------


def test_workspace_is_created_owner_only_and_refuses_escapes(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    sub = ws.dir("speakers", "alice")
    assert sub == (tmp_path / "ws" / "speakers" / "alice").resolve()
    for path in (ws.root, sub.parent, sub):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
    for bad in ("..", "../x", "a/b", ".hidden", ""):
        with pytest.raises(WorkspaceError):
            ws.dir(bad)
    with pytest.raises(WorkspaceError):
        ws.resolve("../outside")
    with pytest.raises(WorkspaceError):
        ws.relative(tmp_path / "elsewhere.wav")
    assert ws.relative(sub / "x.wav") == "speakers/alice/x.wav"


# --- Clip retention -------------------------------------------------------------------


def _clip(directory: Path, name: str, size: int, age_days: float, now: float) -> Path:
    path = directory / name
    path.write_bytes(b"x" * size)
    os.utime(path, (now - age_days * 86400,) * 2)
    return path


def test_prune_by_age_then_size(tmp_path: Path) -> None:
    now = time.time()
    old = _clip(tmp_path, "old.wav", 10, 8, now)
    a = _clip(tmp_path, "a.wav", 100, 3, now)
    b = _clip(tmp_path, "b.wav", 100, 2, now)
    c = _clip(tmp_path, "c.wav", 100, 1, now)
    removed = prune_clips(tmp_path, keep_days=7, max_bytes=250, now=now)
    assert removed == 2
    assert not old.exists() and not a.exists() and b.exists() and c.exists()


def _chunk(seconds: float) -> AudioChunk:
    return AudioChunk(data=b"\x01\x00" * int(16000 * seconds), sample_rate=16000)


async def test_tagger_keeps_clip_and_names_it_by_time_and_message(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    clock = lambda: datetime(2026, 11, 15, 9, 30, tzinfo=UTC)  # noqa: E731
    tagger = SpeakerTagger(FakeIdentifier(YOGESH), ws, clock=clock)
    tag = await tagger.start(_chunk(1.0), "msg-1")
    assert tag.speaker == "yogesh" and tag.score == pytest.approx(0.71)
    assert tag.audio_s == pytest.approx(1.0)
    assert tag.audio_path == "utterances/20261115T093000.000000Z_msg-1.wav"
    path = ws.resolve(tag.audio_path)
    assert path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o600


async def test_keep_zero_deletes_clip_after_identification(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    tagger = SpeakerTagger(FakeIdentifier(YOGESH), ws, keep_days=0)
    tag = await tagger.start(_chunk(0.5), "m")
    assert tag.speaker == "yogesh" and tag.audio_path is None
    assert list(ws.dir(UTTERANCES).iterdir()) == []


async def test_unknown_voice_has_score_but_no_name(tmp_path: Path) -> None:
    tagger = SpeakerTagger(
        FakeIdentifier([SpeakerMatch("yogesh", 0.1, False)]), Workspace(tmp_path)
    )
    tag = await tagger.start(_chunk(0.5), "m")
    assert tag.speaker is None and tag.score == pytest.approx(0.1)


# --- In the voice session ---------------------------------------------------------------


AUDIO = [concat(silence(0.5), tone(1.0), silence(3.0))]


async def test_tag_reaches_user_message_and_log(tmp_path: Path) -> None:
    tagger = SpeakerTagger(FakeIdentifier(YOGESH), Workspace(tmp_path), wait_ms=2000)
    async with Harness(AUDIO, [reply("Hello.")], ["hi lucy"], speakers=tagger) as h:
        await h.run()
    (user,) = h.events(UserMessage)
    assert user.speaker == "yogesh" and user.speaker_score == pytest.approx(0.71)
    assert user.audio_path is not None and user.audio_path.endswith(f"_{user.id}.wav")
    assert (tmp_path / user.audio_path).is_file()
    assert user.audio_s == pytest.approx(1.5, abs=0.07)
    assert h.session.timings[0].speaker == "yogesh"
    assert event_from_json(user.model_dump_json()) == user


async def test_slow_identification_does_not_hold_the_turn(tmp_path: Path) -> None:
    slow = FakeIdentifier(YOGESH, delay=0.5)
    tagger = SpeakerTagger(slow, Workspace(tmp_path), wait_ms=20)
    async with Harness(AUDIO, [reply("Hello.")], ["hi lucy"], speakers=tagger) as h:
        await h.run()
    (user,) = h.events(UserMessage)
    assert user.speaker is None and user.speaker_score is None and user.audio_path is None
    assert user.text == "hi lucy" and h.llm.calls
    slow.done.wait(2)


async def test_no_tagger_means_no_tag(tmp_path: Path) -> None:
    async with Harness(AUDIO, [reply("Hello.")], ["hi lucy"]) as h:
        await h.run()
    (user,) = h.events(UserMessage)
    assert user.speaker is None and user.audio_path is None
    assert user.audio_s == pytest.approx(1.5, abs=0.07)


# --- No behaviour change ------------------------------------------------------------------


def test_old_user_message_json_still_loads() -> None:
    old = (
        '{"type":"user_message","schema_version":1,"id":"x","ts":"2026-01-01T00:00:00Z",'
        '"session_id":"cli","turn_id":null,"text":"hi","source":"voice"}'
    )
    event = event_from_json(old)
    assert isinstance(event, UserMessage) and event.speaker is None and event.audio_s is None


def test_model_request_is_identical_with_and_without_a_tag() -> None:
    plain = UserMessage(id="1", session_id="s", text="what's on today?", source="voice")
    tagged = plain.model_copy(
        update={"speaker": "yogesh", "speaker_score": 0.8, "audio_path": "u/x.wav", "audio_s": 2.0}
    )
    assert events_to_messages([plain]) == events_to_messages([tagged])
    now = datetime(2026, 11, 15, 9, 0, tzinfo=UTC)
    builder = ContextBuilder("You are Lucy.", clock=lambda: now)

    def request(event: UserMessage) -> object:
        return builder.build(
            memory=MemoryContext(), history=events_to_messages([event]), context_window=8192
        )

    assert request(plain) == request(tagged)
    assert render_episode(EpisodeRecord(id=1, event=plain)) == render_episode(
        EpisodeRecord(id=1, event=tagged)
    )


# --- Disabled / missing weights --------------------------------------------------------


def _settings(tmp_path: Path, **speakers: object) -> Settings:
    return Settings.model_validate(
        {
            "llm": {"roles": {"chat": {"adapter": "x"}}},
            "workspace": WorkspaceSettings(path=str(tmp_path / "ws")).model_dump(),
            "speakers": SpeakersSettings(
                weights=str(tmp_path / "missing.onnx"), **speakers
            ).model_dump(),
        }
    )


def test_disabled_builds_no_tagger(tmp_path: Path) -> None:
    notices: list[str] = []
    assert speakers_cli.build_tagger(_settings(tmp_path), notices.append) is None
    assert notices == [] and not (tmp_path / "ws").exists()


def test_missing_weights_turns_tagging_off_with_one_notice(tmp_path: Path) -> None:
    notices: list[str] = []
    settings = _settings(tmp_path, enabled=True)
    assert speakers_cli.build_tagger(settings, notices.append) is None
    assert len(notices) == 1 and "lucy speakers download" in notices[0]
    assert not (tmp_path / "ws").exists()


# --- CLI ------------------------------------------------------------------------------


def _run(settings: Settings, argv: list[str], fake: FakeIdentifier) -> tuple[int, str, str]:
    args: argparse.Namespace = _parser().parse_args(["speakers", *argv])
    out, err = io.StringIO(), io.StringIO()
    code = speakers_cli.run(settings, args, out=out, err=err, identifier=fake)
    return code, out.getvalue(), err.getvalue()


def test_cli_enroll_list_identify_remove(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    fake = FakeIdentifier(YOGESH)
    clip = tmp_path / "a.wav"
    write_wav(clip, _chunk(0.2))
    code, out, _ = _run(settings, ["enroll", "yogesh", str(clip)], fake)
    assert code == 0 and "Enrolled yogesh" in out
    copied = tmp_path / "ws" / "speakers" / "yogesh" / "a.wav"
    assert copied.is_file() and fake.enrolled["yogesh"] == [copied.resolve()]
    assert _run(settings, ["list"], fake)[1] == "yogesh\t1 clip(s)\n"
    code, out, _ = _run(settings, ["identify", str(clip)], fake)
    assert out.splitlines() == ["yogesh\t0.710\tmatch", "guest\t0.200\t-"]
    assert _run(settings, ["remove", "yogesh"], fake)[0] == 0
    assert not copied.parent.exists() and fake.enrolled == {}
    assert _run(settings, ["remove", "yogesh"], fake)[0] == 1
    assert _run(settings, ["enroll", "../evil", str(clip)], fake)[0] == 2


def test_cli_without_weights_says_how_to_download(tmp_path: Path) -> None:
    args = _parser().parse_args(["speakers", "list"])
    err = io.StringIO()
    assert speakers_cli.run(_settings(tmp_path), args, out=io.StringIO(), err=err) == 1
    assert "lucy speakers download" in err.getvalue()


def test_cli_download_uses_the_weights_directory(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    args = _parser().parse_args(["speakers", "download"])
    seen: list[Path] = []
    out = io.StringIO()
    code = speakers_cli.run(
        settings, args, out=out, err=io.StringIO(), download=lambda d: seen.append(d) or d / "w"
    )
    assert code == 0 and seen == [tmp_path]
