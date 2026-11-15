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

    def identify(self, path: Path) -> list[SpeakerMatch]:
        with self._lock:
            recognizer = self._load()
            if len(recognizer.book) == 0:
                return []
            matches = recognizer.identify(str(path), top_k=len(recognizer.book))
        return [SpeakerMatch(m.name, float(m.score), bool(m.accepted)) for m in matches]

    def enroll(self, name: str, paths: Sequence[Path]) -> None:
        with self._lock:
            recognizer = self._load()
            recognizer.enroll(name, [str(p) for p in paths])
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


def download_weights(directory: Path) -> Path:
    """Fetch the ONNX weights into `directory` (the one network call; explicit)."""
    from huggingface_hub import hf_hub_download

    directory.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(WEIGHTS_REPO, WEIGHTS_FILE, local_dir=directory))
