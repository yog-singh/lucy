"""The voice-fingerprint adapter: short speech is tiled to the encoder's window (no model)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lucy.adapters.speaker_voicefingerprint import VoiceFingerprintIdentifier, tile_to

SR = 16000
WINDOW = 3 * SR


def test_tile_to_repeats_short_audio_to_length() -> None:
    wav = np.arange(5, dtype=np.float32)
    out = tile_to(wav, 12)
    assert out.tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]


def test_tile_to_leaves_long_and_empty_audio_alone() -> None:
    long = np.ones(20, dtype=np.float32)
    assert tile_to(long, 12) is long
    empty = np.zeros(0, dtype=np.float32)
    assert tile_to(empty, 12) is empty


class _Chunk:
    def window_samples(self, sample_rate: int) -> int:
        return 3 * sample_rate


class FakeEncoder:
    sample_rate = SR
    chunk_config = _Chunk()

    def __init__(self) -> None:
        self.seen: list[int] = []

    def embed_utterance(self, wav: np.ndarray) -> np.ndarray:
        self.seen.append(len(wav))
        # A "voice" is its constant level: the embedding points along it.
        level = float(np.mean(wav))
        return np.array([level, 1.0 - level])


class FakeBook:
    def __init__(self) -> None:
        self.speakers: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.speakers)

    def add(self, name: str, embeddings: np.ndarray) -> None:
        self.speakers[name] = embeddings

    def centroids(self) -> tuple[list[str], np.ndarray]:
        names = sorted(self.speakers)
        return names, np.stack([self.speakers[n].mean(axis=0) for n in names])


class FakeRecognizer:
    def __init__(self, clips: dict[str, np.ndarray]) -> None:
        self.clips = clips
        self.encoder = FakeEncoder()
        self.book = FakeBook()
        self.saved: list[Path] = []

    def preprocess(self, source: str) -> np.ndarray:
        return self.clips[Path(source).name]

    def score_embeddings(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    def save(self, path: Path) -> Path:
        self.saved.append(Path(path))
        return Path(path)


def _identifier(tmp_path: Path, clips: dict[str, np.ndarray]) -> tuple[Any, FakeRecognizer]:
    ident = VoiceFingerprintIdentifier(
        weights=tmp_path / "w.onnx", book_path=tmp_path / "book.npz", threshold=0.9
    )
    fake = FakeRecognizer(clips)
    ident._recognizer = fake  # skip loading the real model
    return ident, fake


def test_short_speech_is_tiled_before_embedding(tmp_path: Path) -> None:
    clips = {
        "enrol.wav": np.full(5 * SR, 0.8, dtype=np.float32),
        "yes.wav": np.full(SR // 2, 0.8, dtype=np.float32),
        "other.wav": np.full(SR // 2, 0.1, dtype=np.float32),
    }
    ident, fake = _identifier(tmp_path, clips)
    ident.enroll("yogesh", [Path("enrol.wav")])
    assert fake.encoder.seen == [5 * SR]  # long enough: unchanged
    assert fake.saved == [tmp_path / "book.npz"]

    [match] = ident.identify(Path("yes.wav"))
    assert fake.encoder.seen[-1] == WINDOW
    assert match.name == "yogesh"
    assert match.accepted
    assert match.score == pytest.approx(1.0)

    [match] = ident.identify(Path("other.wav"))
    assert not match.accepted


def test_identify_ranks_speakers_by_score(tmp_path: Path) -> None:
    clips = {
        "a.wav": np.full(4 * SR, 0.9, dtype=np.float32),
        "b.wav": np.full(4 * SR, 0.2, dtype=np.float32),
        "q.wav": np.full(SR, 0.25, dtype=np.float32),
    }
    ident, _ = _identifier(tmp_path, clips)
    ident.enroll("a", [Path("a.wav")])
    ident.enroll("b", [Path("b.wav")])
    matches = ident.identify(Path("q.wav"))
    assert [m.name for m in matches] == ["b", "a"]
    assert matches[0].score > matches[1].score


def test_identify_with_empty_book_returns_nothing(tmp_path: Path) -> None:
    ident, _ = _identifier(tmp_path, {})
    assert ident.identify(Path("q.wav")) == []


def test_enroll_needs_audio(tmp_path: Path) -> None:
    ident, _ = _identifier(tmp_path, {})
    with pytest.raises(ValueError):
        ident.enroll("yogesh", [])
