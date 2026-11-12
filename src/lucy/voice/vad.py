"""Voice-activity detection: the probability that a 32 ms frame contains speech.

`SileroVAD` runs the Silero VAD v6 ONNX model (MIT licence, vendored in
`models/`, see `models/NOTICE.md`) directly with `onnxruntime`, which avoids the
`silero-vad` package's PyTorch dependency. It is stateful (it carries an RNN
state and 64 samples of context between frames), so frames must be fed in
order and `reset` called between unrelated stretches of audio. Measured cost:
about 64 µs per frame on one thread.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from importlib import resources
from typing import Any, Final, Protocol, runtime_checkable

from lucy.core.types import INPUT_SAMPLE_RATE, AudioChunk
from lucy.voice.audio import FRAME_SAMPLES, pcm_to_float

logger = logging.getLogger(__name__)

SILERO_MODEL: Final = "silero_vad.onnx"
_CONTEXT: Final = 64  # samples of the previous frame the model sees at 16 kHz


@runtime_checkable
class VAD(Protocol):
    """Speech probability per frame of `FRAME_SAMPLES` samples in the core input format."""

    def __call__(self, frame: AudioChunk) -> float:
        """The probability, from 0 to 1, that `frame` contains speech."""
        ...

    def reset(self) -> None:
        """Forget the state carried from earlier frames."""
        ...


class SileroVAD:
    """Silero VAD v6 on `onnxruntime`, loaded on first use (or `load`).

    `model_path` defaults to the vendored model file.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path
        self._session: Any = None
        self._state: Any = None
        self._context: Any = None
        self._rate: Any = None

    def load(self) -> None:
        """Load the model now (about 0.07 s) rather than on the first frame."""
        if self._session is not None:
            return
        import numpy as np
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        if self._model_path is not None:
            self._session = ort.InferenceSession(
                self._model_path, options, providers=["CPUExecutionProvider"]
            )
        else:
            model = resources.files("lucy.voice").joinpath("models", SILERO_MODEL).read_bytes()
            self._session = ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])
        self._rate = np.array(INPUT_SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        import numpy as np

        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT, dtype=np.float32)

    def __call__(self, frame: AudioChunk) -> float:
        import numpy as np

        if self._session is None:
            self.load()
        if frame.sample_rate != INPUT_SAMPLE_RATE or frame.channels != 1:
            raise ValueError(f"Silero VAD takes {INPUT_SAMPLE_RATE} Hz mono frames")
        samples = pcm_to_float(frame.data)
        if len(samples) != FRAME_SAMPLES:
            raise ValueError(f"Silero VAD takes frames of {FRAME_SAMPLES} samples")
        model_input = np.concatenate([self._context, samples])[None, :]
        output, self._state = self._session.run(
            None, {"input": model_input, "state": self._state, "sr": self._rate}
        )
        self._context = samples[-_CONTEXT:]
        return float(output[0, 0])


class ScriptedVAD:
    """A VAD for tests: probabilities from a script, else from the frame's level.

    With `script`, each call returns the next value (then `default` once the
    script runs out). Otherwise, with `level`, a frame whose peak amplitude
    (0 to 1) reaches `level` scores 1.0 and any other 0.0. `calls` counts
    frames seen and `resets` counts `reset` calls.
    """

    def __init__(
        self,
        script: Iterable[float] | None = None,
        *,
        level: float | None = None,
        default: float = 0.0,
        score: Callable[[AudioChunk], float] | None = None,
    ) -> None:
        self._script = list(script) if script is not None else None
        self._level = level
        self._default = default
        self._score = score
        self.calls = 0
        self.resets = 0

    def __call__(self, frame: AudioChunk) -> float:
        self.calls += 1
        if self._script is not None:
            return self._script.pop(0) if self._script else self._default
        if self._score is not None:
            return self._score(frame)
        if self._level is not None:
            import numpy as np

            peak = float(np.max(np.abs(pcm_to_float(frame.data)))) if frame.data else 0.0
            return 1.0 if peak >= self._level else 0.0
        return self._default

    def reset(self) -> None:
        self.resets += 1
