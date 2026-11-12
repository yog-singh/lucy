"""STT adapter running NVIDIA Parakeet TDT on Apple silicon through `parakeet-mlx` (ADR 0015).

The model runs in this process. Importing this module imports neither MLX nor
`parakeet_mlx`: they are imported, and the weights loaded, on `warm_up` or the
first transcription, so `lucy chat` never pays for them.

- Every MLX call (loading and transcribing) runs on one dedicated worker
  thread, off the event loop, and transcriptions are serialised: one at a time.
- `cache_limit_mb` caps MLX's buffer cache. Without it, the survey measured the
  process footprint nearly doubling after a few utterances.
- Input must be the core input format (16 kHz mono 16-bit); anything else is
  rejected with `ValueError`, since resampling belongs to the audio client.
- Empty or near-silent audio returns an empty transcript without running the
  model. Parakeet itself was silent on silence and noise in the survey; the
  check only saves the call.
- Weights come from the Hugging Face cache (`~/.cache/huggingface/hub`) without
  any network request when they are there, and are downloaded on first use
  otherwise (2.5 GB for the default model). `model` may also be a local
  directory with `config.json` and `model.safetensors`.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Final

from lucy.core import registry
from lucy.core.types import (
    INPUT_CHANNELS,
    INPUT_SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
    AudioChunk,
    STTCaps,
    Transcript,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL: Final = "mlx-community/parakeet-tdt-0.6b-v2"
MIN_SAMPLES: Final = INPUT_SAMPLE_RATE // 10
"""Audio shorter than 0.1 s is not transcribed."""
SILENCE_DBFS: Final = -60.0
"""Audio whose RMS level is below this is treated as silence and not transcribed."""


PREPARE_AFTER: Final = 3.0
"""`prepare` does nothing if the model was used less than this many seconds ago."""


class STTError(RuntimeError):
    """The speech recogniser could not be loaded or failed on an utterance."""


def _default_backend() -> Any:
    """The real `parakeet_mlx` backend: (load, transcribe, set_cache_limit, clear)."""
    return _ParakeetBackend()


class _ParakeetBackend:
    """Thin wrapper over `parakeet_mlx` and `mlx`, imported on first use."""

    def __init__(self) -> None:
        self._mx: Any = None
        self._get_logmel: Any = None

    def _modules(self) -> None:
        if self._mx is None:
            import mlx.core as mx
            from parakeet_mlx.audio import get_logmel

            self._mx, self._get_logmel = mx, get_logmel

    def set_cache_limit(self, limit_bytes: int) -> None:
        self._modules()
        self._mx.set_cache_limit(limit_bytes)

    def load(self, model: str) -> Any:
        self._modules()
        from parakeet_mlx import from_pretrained

        # From the local cache when the weights are there, so loading makes no network
        # request (from_pretrained would ask the Hub for the latest revision each time).
        source = _cached_snapshot(model) or model
        # parakeet-mlx 0.7.0 reads config.json with `json.load(open(...))`, leaving the
        # file to the garbage collector, which warns (ResourceWarning) as it closes it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return from_pretrained(source)

    def transcribe(self, model: Any, samples: Any) -> str:
        self._modules()
        mel = self._get_logmel(self._mx.array(samples), model.preprocessor_config)
        results = model.generate(mel)
        return results[0].text if results else ""

    def clear(self) -> None:
        if self._mx is not None:
            self._mx.clear_cache()


def _cached_snapshot(model: str) -> str | None:
    """The local directory holding `model`: itself if it is a directory, else its
    Hugging Face cache snapshot if fully downloaded, else None."""
    if Path(model).is_dir():
        return model
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model, local_files_only=True)
    except Exception:
        return None


@registry.register("stt", "parakeet_mlx")
class ParakeetMLXSTT:
    """Parakeet TDT speech recognition, one pass over each finished utterance.

    Options: `model` (Hugging Face id or local directory), `cache_limit_mb`
    (MLX buffer cache cap; 0 disables the cap). `backend` is for tests: an
    object with `set_cache_limit(bytes)`, `load(model)`,
    `transcribe(model, float32 samples) -> str` and `clear()`.
    """

    capabilities = STTCaps(streaming=False)

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        cache_limit_mb: int = 256,
        backend: Any = None,
    ) -> None:
        if cache_limit_mb < 0:
            raise ValueError("cache_limit_mb must be 0 (no cap) or positive")
        self.model = model
        self.cache_limit_mb = cache_limit_mb
        self._backend = backend if backend is not None else _default_backend()
        self._model: Any = None
        self._executor: ThreadPoolExecutor | None = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._last_used = -math.inf

    @property
    def loaded(self) -> bool:
        """Whether the model weights are in memory."""
        return self._model is not None

    def _pool(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lucy-stt")
        return self._executor

    async def _call(self, fn: Any, *args: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self._pool(), fn, *args)

    def _load_blocking(self) -> Any:
        if self._model is None:
            started = time.perf_counter()
            if self.cache_limit_mb:
                self._backend.set_cache_limit(self.cache_limit_mb * 2**20)
            try:
                self._model = self._backend.load(self.model)
            except Exception as exc:
                raise STTError(
                    f"cannot load the speech recogniser {self.model!r}: {type(exc).__name__}: "
                    f"{exc} (the weights are downloaded from Hugging Face on first use)"
                ) from exc
            logger.info(
                "loaded speech recogniser %s in %.1fs", self.model, time.perf_counter() - started
            )
        return self._model

    async def warm_up(self) -> None:
        """Load the model and run it once, so the first utterance is not slowed by loading
        and kernel compilation (about 3 s and 1.3 s measured)."""
        self._check_open()
        async with self._lock:
            model = await self._call(self._load_blocking)
            import numpy as np

            # Quiet noise rather than zeros, so the full decoder path runs once.
            noise = np.random.default_rng(0).normal(0, 0.01, INPUT_SAMPLE_RATE).astype("float32")
            await self._call(self._backend.transcribe, model, noise)
            self._last_used = time.monotonic()

    async def prepare(self) -> None:
        """Bring a loaded model back into fast memory before an utterance arrives.

        The voice session calls this when the user starts speaking. Under memory
        pressure (the chat model in Ollama on the same GPU) the first
        transcription after a pause took 6 to 8 s instead of 0.1 s, almost all of
        it spent once on paging; a tiny transcription started while the user is
        still talking pays that cost early (measured: 5.7 s for it, then 0.1 to
        0.5 s for the utterance). Skipped if the model is not loaded or was used
        in the last `PREPARE_AFTER` seconds. Never raises.
        """
        if self._closed or self._model is None:
            return
        if time.monotonic() - self._last_used < PREPARE_AFTER:
            return
        self._last_used = time.monotonic()
        try:
            import numpy as np

            noise = np.random.default_rng(0).normal(0, 0.01, 4000).astype("float32")
            async with self._lock:
                if self._model is not None:
                    started = time.perf_counter()
                    await self._call(self._backend.transcribe, self._model, noise)
                    logger.debug(
                        "prepared the speech recogniser in %.2fs", time.perf_counter() - started
                    )
        except Exception as exc:
            logger.debug("preparing the speech recogniser failed: %s", exc)

    async def transcribe(self, audio: AudioChunk) -> Transcript:
        """Transcribe one utterance in the core input format."""
        self._check_open()
        if audio.sample_rate != INPUT_SAMPLE_RATE or audio.channels != INPUT_CHANNELS:
            raise ValueError(
                f"Parakeet needs {INPUT_SAMPLE_RATE} Hz mono audio (the core input format), "
                f"got {audio.sample_rate} Hz with {audio.channels} channel(s)"
            )
        if len(audio.data) % SAMPLE_WIDTH_BYTES:
            raise ValueError("audio data is not a whole number of 16-bit samples")
        import numpy as np

        samples = np.frombuffer(audio.data, dtype="<i2").astype("float32") / 32768.0
        if len(samples) < MIN_SAMPLES or _dbfs(samples) < SILENCE_DBFS:
            return Transcript(text="")
        waited = time.perf_counter()
        async with self._lock:
            model = await self._call(self._load_blocking)
            started = time.perf_counter()
            try:
                text = await self._call(self._backend.transcribe, model, samples)
            except Exception as exc:
                raise STTError(f"speech recognition failed: {type(exc).__name__}: {exc}") from exc
        self._last_used = time.monotonic()
        logger.debug(
            "transcribed %.1fs of audio in %.2fs (%.2fs waiting)",
            audio.duration,
            time.perf_counter() - started,
            started - waited,
        )
        return Transcript(text=" ".join(str(text).split()))

    async def aclose(self) -> None:
        """Free the model and stop the worker thread. Idempotent."""
        self._closed = True
        async with self._lock:
            if self._executor is not None:
                if self._model is not None:
                    self._model = None
                    await self._call(self._backend.clear)
                executor, self._executor = self._executor, None
                executor.shutdown(wait=False)

    def _check_open(self) -> None:
        if self._closed:
            raise STTError("the speech recogniser has been closed")


def _dbfs(samples: Any) -> float:
    """RMS level of float samples in dBFS (-inf for digital silence)."""
    import numpy as np

    rms = float(np.sqrt(np.mean(np.square(samples, dtype="float64"))))
    return 20 * math.log10(rms) if rms > 0 else -math.inf
