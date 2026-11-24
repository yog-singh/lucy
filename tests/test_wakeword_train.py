"""The personal wake phrase (ADR 0028): the numpy head, training and `lucy wakeword`.
Fakes only: no models, no microphone."""

import argparse
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lucy.clients import cli, wakeword_cli
from lucy.core.types import INPUT_SAMPLE_RATE, AudioChunk
from lucy.voice.audio import FRAME_SAMPLES, Frame, float_to_pcm, read_wav, write_wav
from lucy.voice.wakeword import (
    CLIP_FIRST_STEP,
    STEP_SAMPLES,
    NumpyHead,
    OpenWakeWordDetector,
    ScriptedWakeWord,
    clip_embeddings,
    missing_models,
)
from lucy.voice.wakeword_train import (
    POSITIVE_AFTER_S,
    WINDOW,
    SweepRow,
    TrainingError,
    Windows,
    check_take,
    gather,
    recommend,
    slug,
    speech_bounds,
    split_by_recording,
    train_head,
    train_phrase,
    wakes,
    window_end,
    window_labels,
)

RATE = INPUT_SAMPLE_RATE


class FakeSession:
    """The mel or embedding model: the mel 'frames' carry each piece's loudness."""

    def __init__(self, kind: str, log: list) -> None:
        self.kind, self.log = kind, log

    def get_inputs(self):
        return [SimpleNamespace(name="x", shape=[1, 16, 96])]

    def run(self, outputs, feeds):
        ((name, value),) = feeds.items()
        self.log.append((self.kind, value.shape))
        if self.kind == "mel":
            frames = int(np.ceil(value.shape[1] / 160 - 3))
            level = float(np.abs(value).mean()) / 1000
            return [np.full((1, 1, frames, 32), (level - 2) * 10, dtype=np.float32)]
        # embedding: the mean of the newest 8 mel frames, in every dimension
        newest = value[:, -8:, :, 0].mean(axis=(1, 2))
        return [np.repeat(newest[:, None], 96, axis=1).reshape(-1, 1, 1, 96).astype(np.float32)]


def sessions(log: list | None = None):
    log = [] if log is None else log

    def factory(path: str) -> FakeSession:
        return FakeSession("mel" if "melspectrogram" in path else "embed", log)

    return factory


def constant_head(score: float, features: int = 16) -> NumpyHead:
    logit = np.log(score / (1 - score))
    return NumpyHead(
        w1=np.zeros((features * 96, 32), np.float32),
        b1=np.zeros(32, np.float32),
        w2=np.zeros((32, 1), np.float32),
        b2=np.array([logit], np.float32),
        mean=np.zeros(96, np.float32),
        std=np.ones(96, np.float32),
        phrase="hey lucy",
        threshold=0.6,
    )


# --- The head in the detector ---------------------------------------------------------


def test_head_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    head = NumpyHead(
        w1=rng.normal(size=(1536, 32)).astype(np.float32),
        b1=rng.normal(size=32).astype(np.float32),
        w2=rng.normal(size=(32, 1)).astype(np.float32),
        b2=np.zeros(1, np.float32),
        mean=rng.normal(size=96).astype(np.float32),
        std=np.ones(96, np.float32),
        phrase="hey lucy",
        threshold=0.65,
    )
    path = tmp_path / "hey_lucy.npz"
    head.save(path)
    loaded = NumpyHead.load(path)
    assert (loaded.phrase, loaded.threshold, loaded.features) == ("hey lucy", 0.65, 16)
    windows = rng.normal(size=(3, 16, 96)).astype(np.float32)
    assert np.allclose(loaded.scores(windows), head.scores(windows))
    assert np.all((loaded.scores(windows) >= 0) & (loaded.scores(windows) <= 1))


def test_head_rejects_another_format(tmp_path: Path) -> None:
    path = tmp_path / "x.npz"
    constant_head(0.5).save(path)
    data = dict(np.load(path))
    data["version"] = np.array(99)
    np.savez(path, **data)
    with pytest.raises(ValueError, match="format 99"):
        NumpyHead.load(path)


def test_detector_runs_an_npz_head(tmp_path: Path) -> None:
    path = tmp_path / "hey_lucy.npz"
    constant_head(0.8).save(path)
    detector = OpenWakeWordDetector(path, session_factory=sessions())
    frame = AudioChunk(data=bytes(2 * STEP_SAMPLES), sample_rate=RATE)
    scores = [detector(frame) for _ in range(8)]
    assert scores[:5] == [0.0] * 5  # warm-up
    assert scores[5:] == pytest.approx([0.8] * 3)


def test_missing_models_for_an_npz_head(tmp_path: Path) -> None:
    path = tmp_path / "hey_lucy.npz"
    constant_head(0.5).save(path)
    assert missing_models(path) == [
        tmp_path / "melspectrogram.onnx",
        tmp_path / "embedding_model.onnx",
    ]


def test_clip_embeddings_match_the_streaming_steps() -> None:
    log: list = []
    factory = sessions(log)
    mel, embed = factory("melspectrogram.onnx"), factory("embedding_model.onnx")
    samples = np.zeros(STEP_SAMPLES * 30, np.float32)
    out = clip_embeddings(mel, embed, samples, batch=8)
    assert out.shape == (30 - CLIP_FIRST_STEP + 1, 96)
    mel_calls = [shape for kind, shape in log if kind == "mel"]
    assert mel_calls == [(1, 1760)] * 30  # one piece per step, as when streaming
    assert [shape[0] for kind, shape in log if kind == "embed"] == [8, 8, 5]
    assert len(clip_embeddings(mel, embed, samples[: STEP_SAMPLES * 9])) == 0


# --- Takes and labels -----------------------------------------------------------------


def take(lead: float = 0.5, speech: float = 0.6, length: float = 2.0, level: float = 0.3):
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1e-3, int(length * RATE)).astype(np.float32)
    t = np.arange(int(speech * RATE)) / RATE
    start = int(lead * RATE)
    x[start : start + len(t)] += level * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    return x


def test_speech_bounds_and_take_checks() -> None:
    start, end = speech_bounds(take())
    assert abs(start - 0.5 * RATE) <= 320 and abs(end - 1.1 * RATE) <= 320
    assert check_take(take()) is None
    assert check_take(np.random.default_rng(0).normal(0, 1e-3, 2 * RATE)).startswith("silent")
    assert check_take(take(level=1.2)).startswith("clipped")
    assert check_take(take(lead=1.6, speech=0.4)).startswith("cut off")


def test_window_labels_around_the_end_of_speech() -> None:
    middle, end = 3 * RATE, 3 * RATE + 8000
    labels = window_labels(80, middle, end)
    assert (labels[: WINDOW - 1] == -1).all()
    for j, label in enumerate(labels[WINDOW - 1 :], start=WINDOW - 1):
        t = window_end(j)
        if end <= t <= end + POSITIVE_AFTER_S * RATE:
            assert label == 1
        elif t < middle or t > end + 1.5 * RATE:
            assert label == 0
        else:
            assert label == -1
    assert 4 <= (labels == 1).sum() <= 5


def test_split_by_recording() -> None:
    train, test = split_by_recording(50)
    assert len(test) == 10 and sorted(train + test) == list(range(50))
    assert split_by_recording(50) == (train, test)  # repeatable
    assert len(split_by_recording(2)[1]) == 1
    assert slug("Hey, Lucy!") == "hey_lucy"


# --- Training and evaluation ----------------------------------------------------------


def test_training_separates_a_separable_set() -> None:
    rng = np.random.default_rng(0)

    def make(count: int) -> tuple:
        windows = Windows()
        for k in range(count):
            emb = rng.normal(size=(40, 96)).astype(np.float32)
            positive = k % 4 == 0
            if positive:
                emb[20:36, :8] += 2.0
            labels = np.full(40, -1, np.int8)
            labels[35] = 1 if positive else 0
            labels[18] = 0
            windows.add(emb, labels)
        return windows.arrays()

    train, held = make(200), make(40)
    head, log = train_head(train, held, max_epochs=40)
    emb, ends, labels = held
    scores = head.scores(gather(emb, ends))
    assert scores[labels == 1].min() > scores[labels == 0].max()
    assert log.best_epoch >= 1


def test_wakes_follow_the_refractory_period() -> None:
    scores = np.zeros(100)
    scores[[10, 11, 20, 60]] = 0.9  # steps of 80 ms: 10 and 20 are 0.8 s apart
    assert wakes(scores, 0.5) == 2
    assert wakes(scores, 0.95) == 0


def row(threshold: float, detected: int, false_wakes: int) -> SweepRow:
    return SweepRow(threshold, detected, 10, false_wakes, 0.5)


def test_recommend_the_lowest_clean_threshold() -> None:
    rows = [row(0.3, 10, 3), row(0.4, 10, 0), row(0.5, 9, 0), row(0.6, 7, 0)]
    assert recommend(rows) == 0.4
    rows = [row(0.3, 10, 3), row(0.4, 8, 0), row(0.5, 7, 0)]
    assert recommend(rows) == 0.4  # nothing clean detects 90%: fewest wakes, most detected
    rows = [row(0.3, 10, 3), row(0.4, 10, 2), row(0.5, 9, 2)]
    assert recommend(rows) == 0.4
    assert row(0.3, 1, 3).per_hour == 6.0


def write_takes(directory: Path, count: int) -> None:
    for k in range(count):
        write_wav(directory / f"{k + 1:03d}.wav", float_to_pcm(take(lead=0.3 + 0.02 * k)), RATE)


def fake_frontend(samples: np.ndarray) -> np.ndarray:
    steps = len(samples) // STEP_SAMPLES
    levels = [
        np.abs(samples[k * STEP_SAMPLES : (k + 1) * STEP_SAMPLES]).mean() for k in range(steps)
    ]
    return np.repeat(np.array(levels[CLIP_FIRST_STEP - 1 :], np.float32)[:, None], 96, axis=1)


def test_train_phrase_end_to_end_with_fakes(tmp_path: Path) -> None:
    data = tmp_path / "wakeword" / "hey_lucy"
    write_takes(data / "positive", 20)
    noise = np.random.default_rng(2).normal(0, 0.05, 20 * RATE).astype(np.float32)
    write_wav(data / "negative" / "001.wav", float_to_pcm(noise), RATE)
    models = tmp_path / "models"
    seen = []

    def make_detector(path: Path) -> ScriptedWakeWord:
        seen.append(NumpyHead.load(path).phrase)
        return ScriptedWakeWord(default=0.0)

    result = train_phrase(
        "hey lucy",
        tmp_path / "wakeword",
        models,
        augmentations=2,
        frontend=fake_frontend,
        make_detector=make_detector,
    )
    assert seen == ["hey lucy"]
    assert result.path == models / "hey_lucy.npz" and result.path.is_file()
    assert list(models.iterdir()) == [result.path]  # the trial copy is gone
    assert result.takes == (16, 4)
    assert NumpyHead.load(result.path).threshold == result.threshold == 0.3
    assert [r.detected for r in result.rows] == [0] * 13


# --- lucy wakeword --------------------------------------------------------------------


def parse(*argv: str) -> argparse.Namespace:
    return cli._parser().parse_args(["wakeword", *argv])


def test_parse_wakeword_commands() -> None:
    args = parse("record")
    assert (args.wakeword_action, args.count, args.negative, args.phrase) == (
        "record", 80, False, "hey lucy",
    )  # fmt: skip
    args = parse("record", "--negative", "--minutes", "2", "--phrase", "Hey Mira")
    assert (args.negative, args.minutes, args.phrase) == (True, 2.0, "Hey Mira")
    args = parse("train", "--data-dir", "x", "--models-dir", "y")
    assert (args.data_dir, args.models_dir, args.augmentations) == (Path("x"), Path("y"), 30)
    assert parse("test", "--threshold", "0.7").threshold == 0.7


def run_cli(*argv: str, **kwargs) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    settings = SimpleNamespace(voice=SimpleNamespace(input_device=None, output_device=None))
    code = wakeword_cli.run(settings, parse(*argv), out=out, err=err, **kwargs)
    return code, out.getvalue(), err.getvalue()


def test_train_needs_enough_takes_and_negatives(tmp_path: Path) -> None:
    dirs = ("--data-dir", str(tmp_path / "w"), "--models-dir", str(tmp_path / "m"))
    code, _, err = run_cli("train", *dirs)
    assert code == 1 and "0 take(s)" in err and "at least 20" in err
    write_takes(tmp_path / "w" / "hey_lucy" / "positive", 20)
    code, _, err = run_cli("train", *dirs)
    assert code == 1 and "no negative audio" in err
    write_wav(tmp_path / "w" / "hey_lucy" / "negative" / "a.wav", bytes(2 * RATE * 5), RATE)
    code, _, err = run_cli("train", *dirs)
    assert code == 1 and "lucy listen --download-models" in err
    with pytest.raises(TrainingError):
        train_phrase("hey lucy", tmp_path / "w", tmp_path / "m")


class FakeMic:
    """Frames from a fixed recording; `now` is the time of the newest frame read."""

    def __init__(self, samples: np.ndarray) -> None:
        self._samples = samples
        self._now = 0.0
        self.closed = False

    async def frames(self):
        for k in range(len(self._samples) // FRAME_SAMPLES):
            chunk = self._samples[k * FRAME_SAMPLES : (k + 1) * FRAME_SAMPLES]
            self._now = (k + 1) * FRAME_SAMPLES / RATE
            yield Frame(AudioChunk(data=float_to_pcm(chunk), sample_rate=RATE), self._now)

    def now(self) -> float:
        return self._now

    async def close(self) -> None:
        self.closed = True


def test_record_takes_appends_and_rejects(tmp_path: Path) -> None:
    positive = tmp_path / "hey_lucy" / "positive"
    write_takes(positive, 1)
    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 1e-3, 3 * RATE).astype(np.float32)  # rejected: silent
    good = np.concatenate([rng.normal(0, 1e-3, RATE), take()]).astype(np.float32)
    mic = FakeMic(np.concatenate([quiet, good, good, quiet]))
    code, out, err = run_cli(
        "record", "--count", "2", "--data-dir", str(tmp_path), make_audio=lambda: mic
    )
    assert code == 0 and mic.closed
    assert "again: silent" in err and "2 take(s) saved" in out
    assert sorted(p.name for p in positive.iterdir()) == ["001.wav", "002.wav", "003.wav"]
    samples, rate = read_wav(positive / "003.wav")
    assert rate == RATE and len(samples) == 2 * RATE


def test_record_negative_in_files(tmp_path: Path) -> None:
    mic = FakeMic(np.random.default_rng(0).normal(0, 0.05, 10 * RATE).astype(np.float32))
    code, out, _ = run_cli(
        "record", "--negative", "--minutes", "0.1", "--data-dir", str(tmp_path),
        make_audio=lambda: mic,
    )  # fmt: skip
    assert code == 0 and "1 file(s) saved" in out
    samples, _ = read_wav(tmp_path / "hey_lucy" / "negative" / "001.wav")
    assert len(samples) == 6 * RATE


def test_test_without_a_model(tmp_path: Path) -> None:
    code, _, err = run_cli("test", "--models-dir", str(tmp_path))
    assert code == 1 and "lucy wakeword train" in err
