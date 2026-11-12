"""The Parakeet STT adapter against a fake backend (no MLX, no model)."""

import asyncio
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import pytest

from lucy.adapters.stt_parakeet_mlx import DEFAULT_MODEL, ParakeetMLXSTT, STTError
from lucy.core import registry
from lucy.core.types import AudioChunk
from tests.voice_fakes import silence, tone


class FakeBackend:
    def __init__(self, text: str = "  hello   world ", delay: float = 0.0) -> None:
        self.text = text
        self.delay = delay
        self.loads: list[str] = []
        self.calls: list[np.ndarray] = []
        self.cache_limit: int | None = None
        self.cleared = 0
        self.active = 0
        self.max_active = 0
        self.threads: set[int] = set()
        self.fail_load: Exception | None = None
        self.fail_transcribe: Exception | None = None

    def set_cache_limit(self, limit: int) -> None:
        self.cache_limit = limit

    def load(self, model: str) -> Any:
        self.threads.add(threading.get_ident())
        if self.fail_load is not None:
            raise self.fail_load
        self.loads.append(model)
        return object()

    def transcribe(self, model: Any, samples: np.ndarray) -> str:
        self.threads.add(threading.get_ident())
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            if self.fail_transcribe is not None:
                raise self.fail_transcribe
            self.calls.append(samples)
            return self.text
        finally:
            self.active -= 1

    def clear(self) -> None:
        self.cleared += 1


async def test_loading_is_lazy_and_warm_up_loads_and_runs_once() -> None:
    backend = FakeBackend()
    stt = ParakeetMLXSTT(backend=backend)
    assert not stt.loaded and backend.loads == []
    await stt.warm_up()
    assert stt.loaded
    assert backend.loads == [DEFAULT_MODEL]
    assert backend.cache_limit == 256 * 2**20
    assert len(backend.calls) == 1  # one dummy pass to compile kernels
    await stt.transcribe(tone(1.0))
    assert backend.loads == [DEFAULT_MODEL]
    await stt.aclose()


async def test_transcribes_with_normalised_whitespace() -> None:
    backend = FakeBackend()
    stt = ParakeetMLXSTT(backend=backend, model="some/model", cache_limit_mb=0)
    transcript = await stt.transcribe(tone(1.0))
    assert transcript.text == "hello world"
    assert backend.loads == ["some/model"]
    assert backend.cache_limit is None  # 0: no cap
    (samples,) = backend.calls
    assert samples.dtype == np.float32 and len(samples) == 16000
    assert np.max(np.abs(samples)) == pytest.approx(0.5, abs=0.01)
    assert stt.capabilities.streaming is False
    await stt.aclose()


async def test_transcriptions_are_serialised_on_one_worker_thread() -> None:
    backend = FakeBackend(delay=0.05)
    stt = ParakeetMLXSTT(backend=backend)
    results = await asyncio.gather(*(stt.transcribe(tone(0.5)) for _ in range(4)))
    assert [r.text for r in results] == ["hello world"] * 4
    assert backend.max_active == 1
    assert len(backend.threads) == 1 and threading.get_ident() not in backend.threads
    await stt.aclose()


async def test_the_event_loop_is_not_blocked() -> None:
    backend = FakeBackend(delay=0.2)
    stt = ParakeetMLXSTT(backend=backend)
    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    ticker = asyncio.create_task(tick())
    await stt.transcribe(tone(0.5))
    ticker.cancel()
    assert ticks >= 10
    await stt.aclose()


@pytest.mark.parametrize(
    "audio",
    [
        silence(1.0),
        AudioChunk(data=b"", sample_rate=16000),
        tone(0.05),  # too short
        tone(1.0, amp=0.0005),  # about -69 dBFS
    ],
    ids=["silence", "empty", "too-short", "near-silent"],
)
async def test_silence_returns_an_empty_transcript_without_the_model(audio: AudioChunk) -> None:
    backend = FakeBackend()
    stt = ParakeetMLXSTT(backend=backend)
    assert (await stt.transcribe(audio)).text == ""
    assert backend.calls == [] and backend.loads == []


@pytest.mark.parametrize(
    ("audio", "match"),
    [
        (tone(1.0, rate=24000), "16000 Hz mono"),
        (AudioChunk(data=bytes(3200), sample_rate=16000, channels=2), "2 channel"),
        (AudioChunk(data=bytes(3201), sample_rate=16000), "whole number"),
    ],
)
async def test_other_formats_are_rejected(audio: AudioChunk, match: str) -> None:
    stt = ParakeetMLXSTT(backend=FakeBackend())
    with pytest.raises(ValueError, match=match):
        await stt.transcribe(audio)


async def test_model_failures_are_stt_errors() -> None:
    backend = FakeBackend()
    backend.fail_load = OSError("no such repo")
    stt = ParakeetMLXSTT(backend=backend)
    with pytest.raises(STTError, match="cannot load the speech recogniser.*no such repo"):
        await stt.transcribe(tone(1.0))
    backend.fail_load = None
    backend.fail_transcribe = RuntimeError("metal error")
    with pytest.raises(STTError, match="speech recognition failed.*metal error"):
        await stt.transcribe(tone(1.0))


async def test_aclose_frees_the_model() -> None:
    backend = FakeBackend()
    stt = ParakeetMLXSTT(backend=backend)
    await stt.warm_up()
    await stt.aclose()
    assert not stt.loaded and backend.cleared == 1
    await stt.aclose()  # idempotent
    with pytest.raises(STTError, match="closed"):
        await stt.transcribe(tone(1.0))


def test_negative_cache_limit_is_rejected() -> None:
    with pytest.raises(ValueError):
        ParakeetMLXSTT(cache_limit_mb=-1, backend=FakeBackend())


def test_registered() -> None:
    assert registry.get("stt", "parakeet_mlx") is ParakeetMLXSTT


def test_importing_lucy_for_chat_loads_no_speech_stack() -> None:
    """`lucy chat` imports the CLI and the app's adapter modules; none may pull in MLX,
    onnxruntime, sounddevice or the voice package."""
    code = (
        "import sys, lucy.clients.cli, lucy.app\n"
        "from lucy.core import registry\n"
        "registry.load_adapters(lucy.app.BUILTIN_ADAPTER_MODULES)\n"
        "lucy.clients.cli._parser()\n"
        "heavy = ['mlx', 'parakeet_mlx', 'onnxruntime', 'sounddevice', 'soxr', 'lucy.voice']\n"
        "print([m for m in heavy if m in sys.modules])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )
    assert result.stdout.strip() == "[]"


def test_a_local_directory_or_cached_snapshot_is_used_without_the_network(tmp_path) -> None:
    from lucy.adapters.stt_parakeet_mlx import _cached_snapshot

    assert _cached_snapshot(str(tmp_path)) == str(tmp_path)
    # Not in the cache: None (the real loader then downloads it).
    assert _cached_snapshot("lucy-tests/no-such-model-anywhere") is None


async def test_prepare_runs_the_model_only_when_loaded_and_idle(monkeypatch) -> None:
    import lucy.adapters.stt_parakeet_mlx as module

    backend = FakeBackend()
    stt = ParakeetMLXSTT(backend=backend)
    await stt.prepare()  # not loaded: nothing to bring back
    assert backend.calls == [] and backend.loads == []
    await stt.warm_up()
    await stt.prepare()  # just used: skipped
    assert len(backend.calls) == 1
    monkeypatch.setattr(module, "PREPARE_AFTER", 0.0)
    await stt.prepare()
    assert len(backend.calls) == 2 and len(backend.calls[-1]) == 4000
    backend.fail_transcribe = RuntimeError("gpu busy")
    await stt.prepare()  # never raises
    await stt.aclose()
    await stt.prepare()  # closed: nothing
