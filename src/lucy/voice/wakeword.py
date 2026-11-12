"""Wake-word detection: how likely it is that the wake phrase has just been said.

`OpenWakeWordDetector` runs openWakeWord's three ONNX models directly with
`onnxruntime` (one thread), without the `openwakeword` package: a
mel-spectrogram model, a shared speech-embedding model and a per-phrase
classifier. The streaming logic (80 ms steps of 1280 samples, 480 samples of
context for the mel model, the `x/10 + 2` mel scaling, 76-frame embedding
windows, 16 embeddings per prediction, the first five predictions zeroed) follows
openWakeWord's own ONNX path (`openwakeword/utils.py` `AudioFeatures` and
`openwakeword/model.py` `Model.predict`, https://github.com/dscripka/openWakeWord,
Apache-2.0); this is a small rewrite of only what Lucy needs.

The model files are not part of Lucy: the pretrained phrase models are licensed
CC BY-NC-SA 4.0 (non-commercial use only). `download_models` fetches them from
openWakeWord's GitHub release into `data/models/wakeword/` and checks each
against the SHA-256 pinned here (`lucy listen --download-models`).

`ScriptedWakeWord` is the test double, as `ScriptedVAD` is for the VAD.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from lucy.core.types import AudioChunk

logger = logging.getLogger(__name__)

MODELS_DIR: Final = Path("data/models/wakeword")
"""Where `lucy listen --download-models` puts the model files (relative to the working
directory, like the database)."""
RELEASE_URL: Final = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
USER_AGENT: Final = "Lucy/0.1 (local personal assistant)"


@dataclass(frozen=True)
class ModelFile:
    name: str
    sha256: str
    url: str


MEL_MODEL: Final = ModelFile(
    "melspectrogram.onnx",
    "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    RELEASE_URL + "melspectrogram.onnx",
)
EMBEDDING_MODEL: Final = ModelFile(
    "embedding_model.onnx",
    "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
    RELEASE_URL + "embedding_model.onnx",
)
HEY_JARVIS_MODEL: Final = ModelFile(
    "hey_jarvis_v0.1.onnx",
    "94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb",
    RELEASE_URL + "hey_jarvis_v0.1.onnx",
)
MODEL_FILES: Final = (MEL_MODEL, EMBEDDING_MODEL, HEY_JARVIS_MODEL)

STEP_SAMPLES: Final = 1280  # 80 ms at 16 kHz: one prediction
_MEL_CONTEXT: Final = 160 * 3  # extra samples the mel model needs before each step
_MEL_BINS: Final = 32
_EMBEDDING_WINDOW: Final = 76  # mel frames per embedding
_EMBEDDING_DIM: Final = 96
_WARMUP_PREDICTIONS: Final = 5  # zeroed after a reset, as openWakeWord does
_SEED_SAMPLES: Final = 16000 * 4


@runtime_checkable
class WakeWordDetector(Protocol):
    """A wake-phrase score per capture frame in the core input format (16 kHz mono)."""

    def __call__(self, frame: AudioChunk) -> float:
        """The latest score, from 0 to 1, that the phrase has just been said."""
        ...

    def reset(self) -> None:
        """Forget the audio heard so far."""
        ...


class MissingModelsError(FileNotFoundError):
    """The wake-word model files are not there. The message says how to get them."""


def missing_models(model: Path) -> list[Path]:
    """The files `OpenWakeWordDetector(model)` needs that do not exist: the phrase
    model, and the mel and embedding models beside it."""
    needed = [model, model.parent / MEL_MODEL.name, model.parent / EMBEDDING_MODEL.name]
    return [path for path in needed if not path.is_file()]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def download_models(
    directory: Path = MODELS_DIR,
    *,
    files: Iterable[ModelFile] = MODEL_FILES,
    transport: Any = None,
    report: Callable[[str], None] = lambda line: None,
) -> list[Path]:
    """Download `files` into `directory`, checking each against its pinned SHA-256.

    A file already there with the right hash is kept. A download with the wrong
    hash is not written and raises `ValueError`. Returns the paths. Only the URL and
    a generic User-Agent are sent. `transport` is an httpx transport (tests).
    """
    import httpx

    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(60.0),
        follow_redirects=True,
        transport=transport,
        trust_env=False,
    ) as client:
        for file in files:
            path = directory / file.name
            paths.append(path)
            if path.is_file() and _sha256(path.read_bytes()) == file.sha256:
                report(f"{path} is already there")
                continue
            response = await client.get(file.url)
            response.raise_for_status()
            data = response.content
            digest = _sha256(data)
            if digest != file.sha256:
                raise ValueError(
                    f"{file.name}: SHA-256 {digest} does not match the pinned {file.sha256};"
                    " not saved"
                )
            partial = path.with_suffix(path.suffix + ".part")
            partial.write_bytes(data)
            partial.replace(path)
            report(f"{path} ({len(data):,} bytes, sha256 {digest})")
    return paths


class OpenWakeWordDetector:
    """openWakeWord's ONNX pipeline (see the module docstring), loaded on `load` or
    the first frame.

    `model` is the phrase classifier; `melspectrogram.onnx` and
    `embedding_model.onnx` are expected beside it. Frames of any length are
    accepted; a new score is computed for every 1280 samples, and between steps
    the last score is returned.
    """

    def __init__(self, model: str | Path, *, session_factory: Callable[..., Any] | None = None):
        self._model = Path(model)
        self._session_factory = session_factory
        self._mel: Any = None
        self._embed: Any = None
        self._classify: Any = None
        self._classify_input = ""
        self._features_needed = 16
        self._seed_features: Any = None
        # Streaming state (see `reset`).
        self._pending: Any = None
        self._context: Any = None
        self._mels: Any = None
        self._features: Any = None
        self._predictions = 0
        self._score = 0.0

    def _session(self, path: Path) -> Any:
        if self._session_factory is not None:
            return self._session_factory(str(path))
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def load(self) -> None:
        """Load the three models now (raises `MissingModelsError` if a file is missing)."""
        if self._classify is not None:
            return
        import numpy as np

        missing = missing_models(self._model)
        if missing and self._session_factory is None:
            raise MissingModelsError(
                f"wake-word model files missing ({', '.join(str(p) for p in missing)});"
                " run `lucy listen --download-models`"
            )
        directory = self._model.parent
        self._mel = self._session(directory / MEL_MODEL.name)
        self._embed = self._session(directory / EMBEDDING_MODEL.name)
        self._classify = self._session(self._model)
        first = self._classify.get_inputs()[0]
        self._classify_input = first.name
        if isinstance(first.shape[1], int):
            self._features_needed = first.shape[1]
        # openWakeWord starts the embedding history with embeddings of 4 s of low
        # random noise; a fixed seed keeps runs repeatable.
        noise = np.random.default_rng(0).integers(-1000, 1000, _SEED_SAMPLES).astype(np.float32)
        mels = self._melspectrogram(noise)
        windows = [
            mels[i : i + _EMBEDDING_WINDOW] for i in range(0, len(mels) - _EMBEDDING_WINDOW + 1, 8)
        ]
        self._seed_features = np.stack([self._embedding(w) for w in windows])
        self.reset()

    def _melspectrogram(self, samples: Any) -> Any:
        """Mel frames (n, 32) of int16-valued float32 `samples`, scaled as openWakeWord does."""
        import numpy as np

        output = self._mel.run(None, {"input": samples[None, :].astype(np.float32)})[0]
        return np.squeeze(output).reshape(-1, _MEL_BINS) / 10.0 + 2.0

    def _embedding(self, window: Any) -> Any:
        """The 96-dim embedding of 76 mel frames."""
        import numpy as np

        batch = window.astype(np.float32)[None, :, :, None]
        return self._embed.run(None, {"input_1": batch})[0].reshape(_EMBEDDING_DIM)

    def reset(self) -> None:
        if self._classify is None:
            return
        import numpy as np

        self._pending = np.empty(0, dtype=np.float32)
        self._context = np.zeros(_MEL_CONTEXT, dtype=np.float32)
        self._mels = np.ones((_EMBEDDING_WINDOW, _MEL_BINS), dtype=np.float32)
        self._features = self._seed_features[-self._features_needed :].copy()
        self._predictions = 0
        self._score = 0.0

    def __call__(self, frame: AudioChunk) -> float:
        import numpy as np

        if self._classify is None:
            self.load()
        samples = np.frombuffer(frame.data, dtype="<i2").astype(np.float32)
        self._pending = np.concatenate([self._pending, samples])
        while len(self._pending) >= STEP_SAMPLES:
            step, self._pending = self._pending[:STEP_SAMPLES], self._pending[STEP_SAMPLES:]
            self._score = self._step(step)
        return self._score

    def _step(self, step: Any) -> float:
        import numpy as np

        audio = np.concatenate([self._context, step])
        self._context = audio[-_MEL_CONTEXT:]
        self._mels = np.vstack([self._mels, self._melspectrogram(audio)])[-_EMBEDDING_WINDOW:]
        embedding = self._embedding(self._mels)
        self._features = np.vstack([self._features, embedding[None, :]])[-self._features_needed :]
        features = self._features[None, :, :].astype(np.float32)
        score = float(self._classify.run(None, {self._classify_input: features})[0].reshape(-1)[0])
        self._predictions += 1
        return 0.0 if self._predictions <= _WARMUP_PREDICTIONS else score


class ScriptedWakeWord:
    """A detector for tests: each call returns the next scripted score (then `default`).

    `score`, if given, computes a score from the frame instead. `calls` counts
    frames seen and `resets` counts `reset` calls.
    """

    def __init__(
        self,
        script: Iterable[float] = (),
        *,
        default: float = 0.0,
        score: Callable[[AudioChunk], float] | None = None,
    ) -> None:
        self._script = list(script)
        self._default = default
        self._score = score
        self.calls = 0
        self.resets = 0

    def __call__(self, frame: AudioChunk) -> float:
        self.calls += 1
        if self._script:
            return self._script.pop(0)
        if self._score is not None:
            return self._score(frame)
        return self._default

    def reset(self) -> None:
        self.resets += 1
