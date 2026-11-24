"""Speaker identification with `voice-fingerprint` (WeSpeaker ResNet34 on ONNX Runtime).

The library is imported, and its ONNX session built, on first use, so importing
this module costs nothing. The weights are always given as a file path, which
makes the library skip its Hugging Face download (`models.resolve_weights`
returns the path or raises); `lucy speakers download` is the only network call.
Every method blocks (file reads and inference): call them off the event loop.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from lucy.core.ports import SpeakerMatch

WEIGHTS_REPO: Final = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
WEIGHTS_FILE: Final = "voxceleb_resnet34_LM.onnx"
BOOK_NAME: Final = "book.npz"


class SpeakerWeightsMissing(RuntimeError):
    pass


class VoiceFingerprintIdentifier:
    """`SpeakerIdentifier` over a speaker book stored at `book_path` (.npz)."""

    def __init__(self, *, weights: str | Path, book_path: str | Path, threshold: float) -> None:
        self._weights = Path(weights)
        self._book_path = Path(book_path)
        self._threshold = threshold
        self._recognizer: Any = None
        self._lock = threading.Lock()

    @property
    def weights_present(self) -> bool:
        return self._weights.is_file()

    def load(self) -> None:
        """Build the ONNX session and read the speaker book (idempotent)."""
        with self._lock:
            self._load()

    def _load(self) -> Any:
        if self._recognizer is not None:
            return self._recognizer
        if not self._weights.is_file():
            raise SpeakerWeightsMissing(
                f"speaker model not found at {self._weights}; run `lucy speakers download`"
            )
        from voicefingerprint import Config, EncoderConfig, VoiceRecognizer

        config = Config(encoder=EncoderConfig(weights=str(self._weights)))
        recognizer = VoiceRecognizer(config=config, threshold=self._threshold)
        if self._book_path.is_file():
            recognizer.load(self._book_path)
        self._recognizer = recognizer
        return recognizer

    def _embed(self, recognizer: Any, path: Path) -> Any:
        """The library's levelling and VAD, then speech tiled to fill the encoder's window.

        Left to itself the library zero-pads speech shorter than its window (3 s), and
        the padded frames dominate the filterbank's mean normalisation, so a short clip
        of an enrolled voice scores near 0 (checks/speaker_short, ADR 0021).
        """
        speech = recognizer.preprocess(str(path))
        encoder = recognizer.encoder
        window = encoder.chunk_config.window_samples(encoder.sample_rate)
        return encoder.embed_utterance(tile_to(speech, window))

    def identify(self, path: Path) -> list[SpeakerMatch]:
        with self._lock:
            recognizer = self._load()
            if len(recognizer.book) == 0:
                return []
            embedding = self._embed(recognizer, path)
            names, centroids = recognizer.book.centroids()
            scores = [float(recognizer.score_embeddings(c, embedding)) for c in centroids]
        ranked = sorted(zip(names, scores, strict=True), key=lambda m: -m[1])
        return [SpeakerMatch(n, s, s >= self._threshold) for n, s in ranked]

    def enroll(self, name: str, paths: Sequence[Path]) -> None:
        import numpy as np

        if not paths:
            raise ValueError(f"no audio supplied for {name!r}")
        with self._lock:
            recognizer = self._load()
            embeddings = np.stack([self._embed(recognizer, p) for p in paths])
            recognizer.book.add(name, embeddings)
            recognizer.save(self._book_path)

    def speakers(self) -> dict[str, int]:
        with self._lock:
            book = self._load().book
            return {s["name"]: s["utterances"] for s in book.summary()}

    def remove(self, name: str) -> None:
        with self._lock:
            recognizer = self._load()
            recognizer.book.remove(name)
            recognizer.save(self._book_path)


def tile_to(wav: Any, length: int) -> Any:
    """Repeat `wav` end to end until it is `length` samples; longer or empty input is unchanged."""
    import numpy as np

    n = len(wav)
    if n == 0 or n >= length:
        return wav
    return np.tile(wav, -(-length // n))[:length]


def download_weights(directory: Path) -> Path:
    """Fetch the ONNX weights into `directory` (the one network call; explicit)."""
    from huggingface_hub import hf_hub_download

    directory.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(WEIGHTS_REPO, WEIGHTS_FILE, local_dir=directory))
