"""Training a personal wake phrase from the user's own recordings (ADR 0028).

The phrase model is a `NumpyHead` (`lucy.voice.wakeword`) on openWakeWord's frozen
speech embeddings: the mel and embedding models stay as they are, and only a small
numpy network is trained, on

- positives: each recorded take (about 2 s, one phrase) placed on a background
  (low noise, or a stretch of the negative recordings at 5 to 20 dB SNR) with
  random gain, speed and lead-in. Windows ending from the end of speech to 320 ms
  after it are positive; windows ending before the middle of the phrase, or more
  than 1.5 s after its end, are negative (the greeting alone, trailing audio);
- negatives: every window of the negative recordings, at three random gains.

Takes and negative recordings (cut into one-minute segments) are split into
training and held-out sets by recording, never by augmented copy. Training is
Adam on weighted binary cross-entropy with early stopping on the held-out loss.
The held-out recordings are then run through the real streaming detector to count
detected takes and false wakes per hour at a range of thresholds.

No microphone and no printing here: `lucy.clients.wakeword_cli` does that.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from lucy.core.types import INPUT_SAMPLE_RATE, AudioChunk
from lucy.voice.audio import resample, to_input_format
from lucy.voice.wake import REFRACTORY_S
from lucy.voice.wakeword import (
    CLIP_FIRST_STEP,
    EMBEDDING_MODEL,
    MEL_MODEL,
    STEP_SAMPLES,
    NumpyHead,
    OpenWakeWordDetector,
    clip_embeddings,
    open_session,
)

RATE: Final = INPUT_SAMPLE_RATE
WINDOW: Final = 16  # embeddings per prediction
EMBEDDING_DIM: Final = 96
DATA_DIR: Final = Path("data/wakeword")
TAKE_SECONDS: Final = 2.0
MIN_TAKES: Final = 20
POSITIVE_AFTER_S: Final = 0.32  # positive windows end up to this long after the speech
TRAILING_NEGATIVE_S: Final = 1.5  # negative windows end more than this after the speech
SEGMENT_S: Final = 60.0  # negative recordings are split into segments this long
THRESHOLDS: Final = tuple(round(0.3 + 0.05 * i, 2) for i in range(13))  # 0.3 .. 0.9
MIN_DETECTION: Final = 0.9


class TrainingError(ValueError):
    """Training cannot start (too few takes, no negatives, missing models)."""


def slug(phrase: str) -> str:
    """The phrase as a file name: "Hey Lucy" -> "hey_lucy"."""
    name = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")
    if not name:
        raise ValueError(f"no letters or digits in the phrase {phrase!r}")
    return name


def phrase_dir(root: Path, phrase: str) -> Path:
    """Where the takes (`positive/`) and negative recordings (`negative/`) live."""
    return Path(root) / slug(phrase)


def model_path(models_dir: Path, phrase: str) -> Path:
    return Path(models_dir) / f"{slug(phrase)}.npz"


# --- Audio checks ---------------------------------------------------------------------


def speech_bounds(samples: np.ndarray, rate: int = RATE) -> tuple[int, int] | None:
    """(start, end) sample of the speech in a take by frame energy, or None if there is
    no clear speech. Frames of 20 ms count as speech from 25 dB under the loudest
    frame (and at least 10 dB over the quietest tenth)."""
    hop = rate // 50
    count = len(samples) // hop
    if count < 3:
        return None
    frames = np.asarray(samples[: count * hop], dtype=np.float64).reshape(count, hop)
    level = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-9)
    peak, floor = float(level.max()), float(np.percentile(level, 10))
    if peak < -50 or peak - floor < 15:
        return None
    loud = np.flatnonzero(level >= max(peak - 25, floor + 10))
    return int(loud[0]) * hop, (int(loud[-1]) + 1) * hop


def check_take(samples: np.ndarray, rate: int = RATE) -> str | None:
    """Why a take is unusable ("silent", "clipped", "cut off"), or None if it is fine."""
    if len(samples) and float(np.max(np.abs(samples))) >= 0.99:
        return "clipped: too loud, move back a little"
    bounds = speech_bounds(samples, rate)
    if bounds is None:
        return "silent: nothing was heard"
    if bounds[1] > len(samples) - int(0.06 * rate):
        return "cut off: the phrase ran to the end, start sooner after the cue"
    return None


# --- Loading recordings -----------------------------------------------------------------


@dataclass
class Recording:
    name: str
    samples: np.ndarray  # float32 in [-1, 1] at 16 kHz


def _wavs(directory: Path) -> list[Path]:
    return sorted(p for p in Path(directory).glob("*.wav") if p.is_file())


def load_takes(directory: Path) -> tuple[list[Recording], list[str]]:
    """The usable takes in `directory`, and a line for each one skipped."""
    takes, skipped = [], []
    for path in _wavs(directory):
        try:
            samples = to_input_format(path)
        except ValueError as exc:
            skipped.append(f"{path.name}: {exc}")
            continue
        if speech_bounds(samples) is None:
            skipped.append(f"{path.name}: no speech found")
            continue
        takes.append(Recording(path.name, samples))
    return takes, skipped


def load_negatives(directory: Path, segment_s: float = SEGMENT_S) -> list[Recording]:
    """Every WAV in `directory`, at 16 kHz, cut into segments of up to `segment_s`."""
    segment = int(segment_s * RATE)
    out = []
    for path in _wavs(directory):
        samples = to_input_format(path)
        for k, start in enumerate(range(0, len(samples), segment)):
            piece = samples[start : start + segment]
            if len(piece) >= RATE * 3:
                out.append(Recording(f"{path.name}#{k}", piece))
    return out


def split_by_recording(
    count: int, held_out: float = 0.2, *, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Indices of recordings to train on and to hold out (at least one of each when
    there are two or more)."""
    order = np.random.default_rng(seed).permutation(count).tolist()
    test = round(count * held_out)
    if count >= 2:
        test = min(max(test, 1), count - 1)
    return sorted(order[test:]), sorted(order[:test])


# --- Features ---------------------------------------------------------------------------


class Frontend:
    """openWakeWord's mel and embedding models, for whole clips."""

    def __init__(self, mel: Any, embed: Any) -> None:
        self.mel = mel
        self.embed = embed

    @classmethod
    def open(cls, models_dir: Path, *, threads: int | None = None) -> Frontend:
        models_dir = Path(models_dir)
        missing = [
            models_dir / f.name for f in (MEL_MODEL, EMBEDDING_MODEL)
            if not (models_dir / f.name).is_file()
        ]  # fmt: skip
        if missing:
            raise TrainingError(
                f"wake-word models missing ({', '.join(map(str, missing))});"
                " run `lucy listen --download-models`"
            )
        threads = threads or min(8, os.cpu_count() or 1)
        return cls(
            open_session(models_dir / MEL_MODEL.name, threads=1),
            open_session(models_dir / EMBEDDING_MODEL.name, threads=threads),
        )

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        """Embeddings (n, 96) of float samples in [-1, 1]; see `clip_embeddings`."""
        pcm = np.clip(np.round(np.asarray(samples) * 32768.0), -32768, 32767)
        return clip_embeddings(self.mel, self.embed, pcm.astype(np.float32))


def window_end(index: int) -> int:
    """The sample (in clip coordinates) at which the window ending with embedding
    `index` of `Frontend(clip)` ends."""
    return STEP_SAMPLES * (index + CLIP_FIRST_STEP)


def window_labels(count: int, middle: int, end: int) -> np.ndarray:
    """Labels for the windows ending at each of `count` embeddings of a positive clip:
    1 from the end of speech to `POSITIVE_AFTER_S` after it; 0 before the middle of
    the phrase and more than `TRAILING_NEGATIVE_S` after its end; -1 (unused)
    otherwise and for the first `WINDOW - 1` (incomplete) windows."""
    labels = np.full(count, -1, dtype=np.int8)
    for j in range(WINDOW - 1, count):
        t = window_end(j)
        if end <= t <= end + POSITIVE_AFTER_S * RATE:
            labels[j] = 1
        elif t < middle or t > end + TRAILING_NEGATIVE_S * RATE:
            labels[j] = 0
    return labels


@dataclass
class Windows:
    """Feature windows as indices into one array of embeddings."""

    embeddings: list[np.ndarray] = field(default_factory=list)
    ends: list[np.ndarray] = field(default_factory=list)
    labels: list[np.ndarray] = field(default_factory=list)
    _size: int = 0

    def add(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        keep = np.flatnonzero(labels >= 0)
        self.embeddings.append(embeddings.astype(np.float32))
        self.ends.append(self._size + keep)
        self.labels.append(labels[keep].astype(np.float32))
        self._size += len(embeddings)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(embeddings (n, 96), window end indices (m,), labels (m,))."""
        if not self.embeddings:
            empty = np.zeros(0)
            return np.zeros((0, EMBEDDING_DIM), np.float32), empty.astype(int), empty
        return (
            np.concatenate(self.embeddings),
            np.concatenate(self.ends),
            np.concatenate(self.labels),
        )


def gather(embeddings: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Windows (m, 16, 96) ending at `ends`."""
    return embeddings[ends[:, None] + np.arange(1 - WINDOW, 1)[None, :]]


def _gain(rng: np.random.Generator, db: float = 6.0) -> float:
    return float(10 ** (rng.uniform(-db, db) / 20))


def _stretch(source: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if len(source) < length:
        source = np.tile(source, length // max(1, len(source)) + 1)
    start = int(rng.integers(0, len(source) - length + 1))
    return source[start : start + length]


def augment_take(
    take: np.ndarray, backgrounds: Sequence[np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, int, int]:
    """One positive example: (clip, middle of the phrase, end of speech), in samples."""
    bounds = speech_bounds(take)
    assert bounds is not None
    factor = rng.uniform(0.92, 1.08)  # speed
    if abs(factor - 1) > 1e-3:
        take = resample(take, RATE, round(RATE / factor))
        bounds = (round(bounds[0] / factor), round(bounds[1] / factor))
    take = take * _gain(rng)
    start, end = bounds
    # Enough audio before the end of speech for a full window, plus a random lead-in;
    # enough after it for the trailing negatives.
    offset = max(0, int((WINDOW + CLIP_FIRST_STEP) * STEP_SAMPLES * 1.05) - end)
    offset += int(rng.uniform(0, 0.5) * RATE)
    length = offset + max(len(take), end + int((TRAILING_NEGATIVE_S + 0.6) * RATE))
    if backgrounds and rng.random() < 0.5:
        background = _stretch(backgrounds[int(rng.integers(len(backgrounds)))], length, rng)
        speech = take[start:end]
        speech_rms = float(np.sqrt(np.mean(speech**2))) if len(speech) else 0.0
        noise_rms = float(np.sqrt(np.mean(background**2)))
        if noise_rms > 1e-6:
            snr = rng.uniform(5, 20)
            background = background * (speech_rms / noise_rms / 10 ** (snr / 20))
    else:
        background = rng.normal(0, 10 ** rng.uniform(-4, -3), length)
    clip = np.asarray(background, dtype=np.float32).copy()
    clip[offset : offset + len(take)] += take
    return np.clip(clip, -1, 1), offset + (start + end) // 2, offset + end


def positive_windows(
    frontend: Callable[[np.ndarray], np.ndarray],
    takes: Sequence[Recording],
    backgrounds: Sequence[np.ndarray],
    augmentations: int,
    rng: np.random.Generator,
    into: Windows,
) -> int:
    """Add `augmentations` examples of each take to `into`; returns the clip count."""
    clips = 0
    for take in takes:
        for _ in range(augmentations):
            clip, middle, end = augment_take(take.samples, backgrounds, rng)
            embeddings = frontend(clip)
            into.add(embeddings, window_labels(len(embeddings), middle, end))
            clips += 1
    return clips


def negative_windows(
    frontend: Callable[[np.ndarray], np.ndarray],
    recordings: Sequence[Recording],
    passes: int,
    stride: int,
    rng: np.random.Generator,
    into: Windows,
) -> None:
    """Add every `stride`-th window of each recording, at `passes` random gains.

    Each recording is preceded by embeddings of the noise the detector starts from
    after a reset (`seed_noise`), so the windows just after a reset are trained on.
    """
    seed = frontend(seed_noise())[1 - WINDOW :]
    for recording in recordings:
        for _ in range(passes):
            embeddings = frontend(np.clip(recording.samples * _gain(rng), -1, 1))
            embeddings = np.concatenate([seed, embeddings])
            labels = np.full(len(embeddings), -1, dtype=np.int8)
            labels[WINDOW - 1 :: stride] = 0
            into.add(embeddings, labels)


def seed_noise() -> np.ndarray:
    """The 4 s of noise `OpenWakeWordDetector` fills its history with on a reset."""
    noise = np.random.default_rng(0).integers(-1000, 1000, RATE * 4).astype(np.float32)
    return noise / 32768.0


# --- Training ---------------------------------------------------------------------------


@dataclass
class TrainLog:
    epochs: int = 0
    best_epoch: int = 0
    train_loss: list[float] = field(default_factory=list)
    held_out_loss: list[float] = field(default_factory=list)


def _sample_weights(labels: np.ndarray, negative_weight: float) -> np.ndarray:
    """Each class weighs the same in total, then negatives `negative_weight` times
    more; scaled to a mean of 1."""
    positives = max(1, int(labels.sum()))
    negatives = max(1, len(labels) - positives)
    weights = np.where(labels > 0.5, 1.0 / positives, negative_weight / negatives)
    return (weights / weights.mean()).astype(np.float32)


def _forward(params: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.maximum(x @ params["w1"] + params["b1"], 0.0)
    return hidden, (hidden @ params["w2"] + params["b2"]).reshape(-1)


def _loss(logits: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    # Binary cross-entropy from logits, numerically stable.
    per = np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
    return float(np.mean(weights * per))


def train_head(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    held_out: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    phrase: str = "",
    hidden: int = 32,
    negative_weight: float = 10.0,
    learning_rate: float = 1e-3,
    l2: float = 1e-4,
    batch: int = 256,
    max_epochs: int = 60,
    patience: int = 6,
    seed: int = 0,
) -> tuple[NumpyHead, TrainLog]:
    """Train the head on `(embeddings, ends, labels)` sets; keeps the parameters with
    the lowest held-out loss (the training set's own loss if `held_out` is empty)."""
    rng = np.random.default_rng(seed)
    emb, ends, labels = train
    mean = emb.mean(axis=0).astype(np.float32)
    std = (emb.std(axis=0) + 1e-3).astype(np.float32)
    features = WINDOW * EMBEDDING_DIM

    def inputs(embeddings: np.ndarray, at: np.ndarray) -> np.ndarray:
        return ((gather(embeddings, at) - mean) / std).reshape(len(at), features)

    params = {
        "w1": (rng.normal(0, np.sqrt(2 / features), (features, hidden))).astype(np.float32),
        "b1": np.zeros(hidden, np.float32),
        "w2": (rng.normal(0, np.sqrt(1 / hidden), (hidden, 1))).astype(np.float32),
        "b2": np.zeros(1, np.float32),
    }
    moments = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in params.items()}
    weights = _sample_weights(labels, negative_weight)
    v_emb, v_ends, v_labels = held_out if len(held_out[1]) else train
    v_weights = _sample_weights(v_labels, negative_weight)

    def evaluate(embeddings: np.ndarray, at: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        logits = np.concatenate(
            [_forward(params, inputs(embeddings, at[i : i + 4096]))[1]
             for i in range(0, len(at), 4096)] or [np.zeros(0)]
        )  # fmt: skip
        return _loss(logits, y, w)

    log = TrainLog()
    best, best_loss, step = None, np.inf, 0
    beta1, beta2 = 0.9, 0.999
    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(ends))
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            x, y, w = inputs(emb, ends[idx]), labels[idx], weights[idx]
            hid, logits = _forward(params, x)
            g = (w * (1 / (1 + np.exp(-logits)) - y) / len(idx)).astype(np.float32)[:, None]
            dh = (g @ params["w2"].T) * (hid > 0)
            grads = {
                "w2": hid.T @ g + l2 * params["w2"],
                "b2": g.sum(axis=0),
                "w1": x.T @ dh + l2 * params["w1"],
                "b1": dh.sum(axis=0),
            }
            step += 1
            for k, grad in grads.items():
                m, v = moments[k]
                m *= beta1
                m += (1 - beta1) * grad
                v *= beta2
                v += (1 - beta2) * grad**2
                m_hat = m / (1 - beta1**step)
                v_hat = v / (1 - beta2**step)
                params[k] -= (learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)).astype(np.float32)
        log.epochs = epoch
        log.train_loss.append(evaluate(emb, ends, labels, weights))
        held = evaluate(v_emb, v_ends, v_labels, v_weights)
        log.held_out_loss.append(held)
        if held < best_loss - 1e-5:
            best_loss, best, log.best_epoch = held, {k: v.copy() for k, v in params.items()}, epoch
        elif epoch - log.best_epoch >= patience:
            break
    assert best is not None
    head = NumpyHead(**best, mean=mean, std=std, phrase=phrase, features=WINDOW)
    return head, log


# --- Evaluation -------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepRow:
    threshold: float
    detected: int
    takes: int
    false_wakes: int
    hours: float

    @property
    def per_hour(self) -> float:
        return self.false_wakes / self.hours if self.hours > 0 else 0.0


def _stream_scores(detector: Any, samples: np.ndarray) -> np.ndarray:
    """The detector's score after each 80 ms step of `samples`, from a reset."""
    detector.reset()
    pcm = np.clip(np.round(samples * 32768.0), -32768, 32767).astype("<i2")
    steps = len(pcm) // STEP_SAMPLES
    return np.array(
        [
            detector(AudioChunk(data=pcm[k * STEP_SAMPLES : (k + 1) * STEP_SAMPLES].tobytes(),
                                sample_rate=RATE))
            for k in range(steps)
        ]
    )  # fmt: skip


def wakes(scores: np.ndarray, threshold: float, refractory_s: float = REFRACTORY_S) -> int:
    """Detections in a score stream (one per 80 ms step), as `WakeStage` counts them."""
    count, last = 0, -np.inf
    for k in np.flatnonzero(scores >= threshold):
        t = (k + 1) * STEP_SAMPLES / RATE
        if t - last >= refractory_s:
            count, last = count + 1, t
    return count


def evaluate(
    detector: Any,
    takes: Sequence[Recording],
    negatives: Sequence[Recording],
    thresholds: Sequence[float] = THRESHOLDS,
    *,
    seed: int = 0,
) -> list[SweepRow]:
    """Run the streaming `detector` over each take (1.5 s of low noise before, 1 s
    after) and over each negative recording; count takes detected and false wakes."""
    rng = np.random.default_rng(seed)
    peaks = []
    for take in takes:
        before = rng.normal(0, 3e-4, int(1.5 * RATE)).astype(np.float32)
        after = rng.normal(0, 3e-4, RATE).astype(np.float32)
        scores = _stream_scores(detector, np.concatenate([before, take.samples, after]))
        peaks.append(float(scores.max()) if len(scores) else 0.0)
    streams = [_stream_scores(detector, rec.samples) for rec in negatives]
    hours = sum(len(rec.samples) for rec in negatives) / RATE / 3600
    return [
        SweepRow(
            threshold=t,
            detected=sum(p >= t for p in peaks),
            takes=len(takes),
            false_wakes=sum(wakes(s, t) for s in streams),
            hours=hours,
        )
        for t in thresholds
    ]


def recommend(rows: Sequence[SweepRow], min_detection: float = MIN_DETECTION) -> float:
    """The lowest threshold with no false wakes that detects at least `min_detection`
    of the takes; else the one with the fewest false wakes, then the most takes
    detected, then the lowest."""
    good = [r for r in rows if r.false_wakes == 0 and r.detected >= min_detection * r.takes]
    if good:
        return min(r.threshold for r in good)
    return min(rows, key=lambda r: (r.false_wakes, -r.detected, r.threshold)).threshold


# --- The whole run ----------------------------------------------------------------------


@dataclass
class TrainResult:
    path: Path
    threshold: float
    rows: list[SweepRow]
    log: TrainLog
    takes: tuple[int, int]  # train, held out
    negative_minutes: tuple[float, float]
    windows: tuple[int, int]  # positive, negative training windows
    skipped: list[str]
    seconds: dict[str, float]


def check_data(directory: Path, min_takes: int = MIN_TAKES) -> None:
    """Raise `TrainingError` if there are too few takes or no negative audio."""
    takes = _wavs(directory / "positive")
    if len(takes) < min_takes:
        raise TrainingError(
            f"{len(takes)} take(s) in {directory / 'positive'}; at least {min_takes} are needed"
            " (record them with `lucy wakeword record`)"
        )
    if not _wavs(directory / "negative"):
        raise TrainingError(
            f"no negative audio in {directory / 'negative'}"
            " (record it with `lucy wakeword record --negative`)"
        )


def train_phrase(
    phrase: str,
    data_root: Path,
    models_dir: Path,
    *,
    augmentations: int = 30,
    negative_passes: int = 3,
    negative_stride: int = 1,
    seed: int = 0,
    frontend: Callable[[np.ndarray], np.ndarray] | None = None,
    make_detector: Callable[[Path], Any] | None = None,
    report: Callable[[str], None] = lambda line: None,
) -> TrainResult:
    """Train, evaluate and save `models_dir/<slug>.npz` from `data_root/<slug>/`.

    `frontend` and `make_detector` are for tests (the real ones need the mel and
    embedding models in `models_dir`).
    """
    directory = phrase_dir(data_root, phrase)
    check_data(directory)
    if frontend is None:
        frontend = Frontend.open(models_dir)
    seconds: dict[str, float] = {}
    started = time.perf_counter()
    takes, skipped = load_takes(directory / "positive")
    if len(takes) < MIN_TAKES:
        raise TrainingError(f"only {len(takes)} usable take(s); at least {MIN_TAKES} are needed")
    negatives = load_negatives(directory / "negative")
    if len(negatives) < 1:
        raise TrainingError(f"the negative audio in {directory / 'negative'} is under 3 s")
    take_train, take_test = split_by_recording(len(takes), seed=seed)
    neg_train, neg_test = split_by_recording(len(negatives), seed=seed + 1)
    if not neg_train:  # a single segment: train on it, and hold out nothing
        neg_train, neg_test = neg_test, []
    train_takes = [takes[i] for i in take_train]
    test_takes = [takes[i] for i in take_test]
    train_neg = [negatives[i] for i in neg_train]
    test_neg = [negatives[i] for i in neg_test]
    backgrounds = [rec.samples for rec in train_neg]
    rng = np.random.default_rng(seed)
    report(
        f"takes: {len(train_takes)} to train on, {len(test_takes)} held out;"
        f" negative audio: {_minutes(train_neg):.1f} min to train on,"
        f" {_minutes(test_neg):.1f} min held out"
    )

    report(f"extracting features ({len(train_takes)} x {augmentations} positive clips)...")
    train_set, held_set = Windows(), Windows()
    positive_windows(frontend, train_takes, backgrounds, augmentations, rng, train_set)
    negative_windows(frontend, train_neg, negative_passes, negative_stride, rng, train_set)
    positive_windows(frontend, test_takes, backgrounds, max(1, augmentations // 6), rng, held_set)
    negative_windows(frontend, test_neg, 1, negative_stride, rng, held_set)
    train_arrays, held_arrays = train_set.arrays(), held_set.arrays()
    seconds["features"] = time.perf_counter() - started
    positives = int(train_arrays[2].sum())
    report(
        f"windows: {positives} positive, {len(train_arrays[2]) - positives} negative"
        f" ({seconds['features']:.0f} s)"
    )

    started = time.perf_counter()
    head, log = train_head(train_arrays, held_arrays, phrase=phrase, seed=seed)
    seconds["training"] = time.perf_counter() - started
    report(
        f"trained {log.epochs} epochs, best {log.best_epoch} (held-out loss"
        f" {log.held_out_loss[log.best_epoch - 1]:.4f}, {seconds['training']:.0f} s)"
    )

    started = time.perf_counter()
    path = model_path(models_dir, phrase)
    trial = path.with_name(path.stem + ".tmp.npz")
    head.save(trial)
    try:
        detector = (make_detector or OpenWakeWordDetector)(trial)
        report("evaluating on the held-out recordings with the streaming detector...")
        rows = evaluate(detector, test_takes, test_neg, seed=seed)
    finally:
        trial.unlink(missing_ok=True)
    seconds["evaluation"] = time.perf_counter() - started
    head.threshold = recommend(rows)
    head.save(path)
    return TrainResult(
        path=path,
        threshold=head.threshold,
        rows=rows,
        log=log,
        takes=(len(train_takes), len(test_takes)),
        negative_minutes=(_minutes(train_neg), _minutes(test_neg)),
        windows=(positives, len(train_arrays[2]) - positives),
        skipped=skipped,
        seconds=seconds,
    )


def _minutes(recordings: Sequence[Recording]) -> float:
    return sum(len(r.samples) for r in recordings) / RATE / 60
