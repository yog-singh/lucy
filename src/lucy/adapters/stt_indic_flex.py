"""STT adapter running bodhan-ai/indic-transcribe-flex in a worker process (ADR 0023).

English and Hindi (and the two mixed) with all output in Latin script. The model needs
PyTorch and Transformers, which stay out of Lucy's environment: it runs in
`indic_flex_worker.py` (shipped next to this module), a PEP 723 script with pinned
packages started with `uv run --script`, which loads the model once and answers JSON
lines on stdin/stdout (see the worker's docstring). Per utterance:

1. input checks as the Parakeet adapter: 16 kHz mono 16-bit; under 0.1 s or quieter
   than -60 dBFS returns an empty transcript without asking the worker;
2. the worker transcribes (language identification restricted to `languages`, mode
   "mixed", float16 on MPS, greedy, generation capped at the guard's token bound);
3. any Devanagari left is written in Latin (`lucy.voice.translit.to_latin`);
4. the junk guard (`stt_indic_flex_guard`) drops invented text, returned as empty text
   (the voice session treats that as no speech) and logged with the reason.

The worker starts lazily (or on `warm_up`), is kept running, and is spoken to by one
thread, one request at a time. A request that takes longer than `timeout` kills the
worker and fails; a worker that dies is restarted once and the request retried. The
worker runs in its own process group, stopped on `aclose` and at interpreter exit. It
gets a minimal environment; when the model is in the Hugging Face cache the offline
variables are set and no token is passed. The worker's stderr goes to Lucy's debug log.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import contextlib
import json
import logging
import math
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
import weakref
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Final

from lucy.adapters.stt_indic_flex_guard import Guard, collapse_repeats, junk_reason
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

DEFAULT_MODEL: Final = "bodhan-ai/indic-transcribe-flex"
DEFAULT_REVISION: Final = "ecba9bac9013c0340fe814b336897ee51ed80be9"
"""The commit whose model code was reviewed (checks/README.md, stt_accuracy)."""
MODEL_PAGE: Final = "https://huggingface.co/bodhan-ai/indic-transcribe-flex"
WORKER_SCRIPT: Final = Path(__file__).with_name("indic_flex_worker.py")
LANGUAGES: Final = ("en", "hi")
"""Languages measured with this adapter (Latin output covers Devanagari only)."""
MODES: Final = ("mixed", "native", "romanized")
DEVICES: Final = ("mps", "cpu")
MIN_SAMPLES: Final = INPUT_SAMPLE_RATE // 10
"""Audio shorter than 0.1 s is not transcribed."""
SILENCE_DBFS: Final = -60.0
"""Audio whose RMS level is below this is not sent to the worker."""
PREPARE_AFTER: Final = 3.0
"""`prepare` does nothing if the worker was used less than this many seconds ago."""
STOP_GRACE: Final = 3.0
"""Seconds the worker gets to exit after its stdin closes, before it is killed."""
ENV_KEEP: Final = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "XDG_CACHE_HOME",
)
"""Variables passed to the worker (plus any `UV_*`); everything else is left out."""


class STTError(RuntimeError):
    """The speech recogniser could not be started or failed on an utterance."""


class _WorkerGone(Exception):
    """The worker exited (or its pipe broke) during a request."""


def find_uv() -> str | None:
    """The `uv` executable: `$UV` (set by `uv run`), then PATH, then the usual places."""
    candidates = [os.environ.get("UV"), shutil.which("uv")]
    home = Path.home()
    candidates += [
        str(home / ".local" / "bin" / "uv"),
        str(home / ".cargo" / "bin" / "uv"),
        "/opt/homebrew/bin/uv",
        "/usr/local/bin/uv",
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _model_cached(model: str, revision: str) -> bool:
    """Whether `model` at `revision` is fully in the Hugging Face cache (no network)."""
    try:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(model, revision=revision, local_files_only=True))
    except Exception:
        return False
    return (path / "model.safetensors").exists() and (path / "indic_transcribe.py").exists()


# --- The worker process -------------------------------------------------------------

_LIVE: weakref.WeakSet[_Worker] = weakref.WeakSet()


@atexit.register
def _stop_all() -> None:
    for worker in list(_LIVE):
        worker.kill()


class _Worker:
    """One running worker process: its pipes, a stdout reader and a stderr pump."""

    def __init__(self, command: Sequence[str], env: dict[str, str]) -> None:
        self.proc = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,  # its own process group, so uv's child goes too
        )
        self.lines: queue.Queue[bytes] = queue.Queue()
        _LIVE.add(self)
        threading.Thread(target=self._read, name="lucy-stt-worker-out", daemon=True).start()
        threading.Thread(target=self._pump, name="lucy-stt-worker-err", daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        with contextlib.suppress(OSError, ValueError):
            for line in self.proc.stdout:
                self.lines.put(line)
        self.lines.put(b"")  # end of file

    def _pump(self) -> None:
        assert self.proc.stderr is not None
        with contextlib.suppress(OSError, ValueError):
            for raw in self.proc.stderr:
                text = raw.decode("utf-8", "replace").strip()[:500]
                if text:
                    logger.debug("[stt worker] %s", text)

    def send(self, obj: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(json.dumps(obj).encode() + b"\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise _WorkerGone(f"cannot write to the worker: {type(exc).__name__}") from None

    def receive(self, timeout: float) -> dict[str, Any] | None:
        """The next JSON line, or None on timeout; raises `_WorkerGone` at end of file."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                line = self.lines.get(timeout=min(remaining, 0.2))
            except queue.Empty:
                # A grandchild may hold the pipe open after the worker died: poll too.
                if self.proc.poll() is not None and self.lines.empty():
                    raise _WorkerGone(f"the worker exited (code {self.proc.returncode})") from None
                continue
            if not line:
                self.lines.put(b"")
                raise _WorkerGone(f"the worker exited (code {self.proc.poll()})")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                logger.debug("[stt worker stdout] %s", line[:200].decode("utf-8", "replace"))

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self, grace: float = STOP_GRACE) -> None:
        """Close stdin, wait up to `grace` seconds, then stop the whole process group."""
        with contextlib.suppress(OSError, ValueError):
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.proc.wait(timeout=grace)
        self.kill()

    def kill(self) -> None:
        """SIGTERM then SIGKILL the worker's process group (never Lucy's)."""
        pgid = self.proc.pid  # a session leader: its group id is its pid
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if not _group_alive(pgid):
                break
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, sig)
            deadline = time.monotonic() + 1.0
            while _group_alive(pgid) and time.monotonic() < deadline:
                time.sleep(0.01)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.proc.wait(timeout=1.0)
        _LIVE.discard(self)


def _group_alive(pgid: int) -> bool:
    if pgid == os.getpgrp():
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --- The adapter ---------------------------------------------------------------------


@registry.register("stt", "indic_flex")
class IndicFlexSTT:
    """indic-transcribe-flex speech recognition in a worker, one pass per utterance.

    Options (all from `[stt]`): `model`, `revision` (a full commit hash), `languages`
    (subset of en, hi; one forces it, two let the model choose between them), `mode`
    (mixed, native, romanized), `latin` (write Devanagari in Latin), `device` (mps,
    cpu), `timeout` (seconds per utterance), `start_timeout` (seconds for the worker to
    load, including building its environment the first time), and the guard's
    `min_dbfs`, `max_tokens_per_s`, `token_slack`, `min_mean_logprob`, `max_repeats`.
    `command` is for tests: the worker command line instead of `uv run --script ...`.
    """

    capabilities = STTCaps(streaming=False)

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        revision: str = DEFAULT_REVISION,
        languages: Sequence[str] = LANGUAGES,
        mode: str = "mixed",
        latin: bool = True,
        device: str = "mps",
        timeout: float = 20.0,
        start_timeout: float = 900.0,
        min_dbfs: float = -55.0,
        max_tokens_per_s: float = 25.0,
        token_slack: float = 10.0,
        min_mean_logprob: float = -1.0,
        max_repeats: int = 4,
        command: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a Hugging Face model id")
        if not isinstance(revision, str) or len(revision) != 40 or not _is_hex(revision):
            raise ValueError("revision must be a full 40-character commit hash")
        if isinstance(languages, str) or not languages:
            raise ValueError(f"languages must be a non-empty list drawn from {list(LANGUAGES)}")
        langs = list(dict.fromkeys(languages))
        unknown = [lang for lang in langs if lang not in LANGUAGES]
        if unknown:
            raise ValueError(f"languages {unknown} not supported; use some of {list(LANGUAGES)}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        if not isinstance(latin, bool):
            raise ValueError("latin must be true or false")
        if device not in DEVICES:
            raise ValueError(f"device must be one of {list(DEVICES)}")
        for name, value in (("timeout", timeout), ("start_timeout", start_timeout)):
            if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                raise ValueError(f"{name} must be a positive number of seconds")
        self.guard = Guard(
            min_dbfs=float(min_dbfs),
            max_tokens_per_s=float(max_tokens_per_s),
            token_slack=float(token_slack),
            min_mean_logprob=float(min_mean_logprob),
            max_repeats=int(max_repeats),
        )
        self.model, self.revision, self.languages = model, revision, langs
        self.mode, self.latin, self.device = mode, latin, device
        self.timeout, self.start_timeout = float(timeout), float(start_timeout)
        self._command = list(command) if command is not None else None
        self._worker: _Worker | None = None
        self._next_id = 0
        self._executor: ThreadPoolExecutor | None = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._last_used = -math.inf

    @property
    def loaded(self) -> bool:
        """Whether the worker is running with the model loaded."""
        return self._worker is not None and self._worker.alive()

    # --- Worker lifecycle (on the adapter's one thread) ---------------------------

    def _worker_args(self) -> list[str]:
        lang = self.languages[0] if len(self.languages) == 1 else "auto"
        args = [
            f"--model={self.model}",
            f"--revision={self.revision}",
            f"--lang={lang}",
            f"--mode={self.mode}",
            f"--device={self.device}",
            f"--dtype={'float16' if self.device == 'mps' else 'float32'}",
            f"--tokens-base={self.guard.token_slack}",
            f"--tokens-per-s={self.guard.max_tokens_per_s}",
        ]
        if lang == "auto":
            args.append(f"--allowed={','.join(self.languages)}")
        return args

    def _environment(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k in ENV_KEEP or k.startswith("UV_")}
        env.update(PYTHONUNBUFFERED="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="0")
        if self._command is None and _model_cached(self.model, self.revision):
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        else:  # a download may be needed: the token, if any, goes to the worker only
            for key in ("HF_TOKEN", "HF_TOKEN_PATH"):
                if key in os.environ:
                    env[key] = os.environ[key]
        return env

    def _start_blocking(self) -> _Worker:
        if self._worker is not None and self._worker.alive():
            return self._worker
        if self._worker is not None:
            self._worker.kill()
            self._worker = None
        if self._command is not None:
            command = self._command
        else:
            uv = find_uv()
            if uv is None:
                raise STTError(
                    "the indic_flex speech recogniser needs `uv` to run its worker, and it was "
                    "not found (looked at $UV, PATH, ~/.local/bin, ~/.cargo/bin, /opt/homebrew/"
                    "bin); install it from https://docs.astral.sh/uv/ or use adapter = "
                    '"parakeet_mlx"'
                )
            command = [uv, "run", "--quiet", "--script", str(WORKER_SCRIPT)]
        started = time.perf_counter()
        try:
            worker = _Worker([*command, *self._worker_args()], self._environment())
        except OSError as exc:
            raise STTError(f"cannot start the speech worker: {exc}") from exc
        try:
            reply = worker.receive(self.start_timeout)
        except _WorkerGone as exc:
            worker.kill()
            raise STTError(
                f"the speech worker exited while loading {self.model} ({exc}); see the debug "
                f"log for its output. If the model is missing, accept its licence at "
                f"{MODEL_PAGE} and log in to Hugging Face (`hf auth login`)"
            ) from None
        if reply is None or "fatal" in reply or not reply.get("ready"):
            worker.kill()
            if reply is None:
                why = f"it was not ready after {self.start_timeout:.0f}s"
            else:
                why = str(reply.get("fatal") or reply)
            raise STTError(f"cannot load the speech recogniser {self.model}: {why}")
        self._worker = worker
        logger.info(
            "speech worker ready: %s on %s %s in %.1fs (%.1fs loading the model)",
            self.model,
            reply.get("device"),
            reply.get("dtype"),
            time.perf_counter() - started,
            reply.get("load_s", 0.0),
        )
        return worker

    def _ask_once(self, pcm: bytes) -> dict[str, Any]:
        worker = self._start_blocking()
        self._next_id += 1
        rid = self._next_id
        worker.send({"id": rid, "pcm": base64.b64encode(pcm).decode("ascii")})
        while True:
            reply = worker.receive(self.timeout)
            if reply is None:
                self._discard_worker()
                raise STTError(
                    f"speech recognition took longer than {self.timeout:g}s; the worker was "
                    "stopped and will be restarted on the next utterance"
                )
            if reply.get("id") != rid:
                continue  # a late reply to an earlier request
            if "error" in reply:
                raise STTError(f"speech recognition failed: {reply['error']}")
            return reply

    def _ask_blocking(self, pcm: bytes) -> dict[str, Any]:
        """One request; if the worker dies, restart it once and retry."""
        try:
            return self._ask_once(pcm)
        except _WorkerGone as exc:
            logger.warning("the speech worker died (%s); restarting it", exc)
            self._discard_worker()
        try:
            return self._ask_once(pcm)
        except _WorkerGone as exc:
            self._discard_worker()
            raise STTError(
                f"the speech worker died twice in a row ({exc}); see the debug log"
            ) from None

    def _discard_worker(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.kill()  # whatever is left of its process group

    def _stop_blocking(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.stop()

    def _pool(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lucy-stt")
        return self._executor

    async def _call(self, fn: Any, *args: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self._pool(), fn, *args)

    # --- The port --------------------------------------------------------------------

    async def warm_up(self) -> None:
        """Start the worker (load the model) and transcribe once, so the first utterance
        does not pay for loading."""
        self._check_open()
        async with self._lock:
            await self._call(self._start_blocking)
            await self._call(self._ask_blocking, _noise(INPUT_SAMPLE_RATE))
            self._last_used = time.monotonic()

    async def prepare(self) -> None:
        """Bring the model back into fast memory before an utterance arrives (as the
        Parakeet adapter does: the voice session calls this when the user starts
        speaking). A tiny transcription in the worker; skipped if the worker is not
        running or was used in the last `PREPARE_AFTER` seconds. Never raises."""
        if self._closed or not self.loaded:
            return
        if time.monotonic() - self._last_used < PREPARE_AFTER:
            return
        self._last_used = time.monotonic()
        try:
            async with self._lock:
                if self.loaded:
                    started = time.perf_counter()
                    await self._call(self._ask_blocking, _noise(4000))
                    logger.debug(
                        "prepared the speech recogniser in %.2fs", time.perf_counter() - started
                    )
        except Exception as exc:
            logger.debug("preparing the speech recogniser failed: %s", exc)

    async def transcribe(self, audio: AudioChunk) -> Transcript:
        """Transcribe one utterance in the core input format; Latin script out."""
        self._check_open()
        if audio.sample_rate != INPUT_SAMPLE_RATE or audio.channels != INPUT_CHANNELS:
            raise ValueError(
                f"indic_flex needs {INPUT_SAMPLE_RATE} Hz mono audio (the core input format), "
                f"got {audio.sample_rate} Hz with {audio.channels} channel(s)"
            )
        if len(audio.data) % SAMPLE_WIDTH_BYTES:
            raise ValueError("audio data is not a whole number of 16-bit samples")
        level = _pcm_dbfs(audio.data)
        if len(audio.data) // SAMPLE_WIDTH_BYTES < MIN_SAMPLES or level < SILENCE_DBFS:
            return Transcript(text="")
        waited = time.perf_counter()
        async with self._lock:
            started = time.perf_counter()
            reply = await self._call(self._ask_blocking, audio.data)
        self._last_used = time.monotonic()
        raw = " ".join(str(reply.get("text", "")).split())
        lang = reply.get("lang") if reply.get("lang") in self.languages else None
        if self.latin:  # imported here: `lucy chat` must not load the voice package
            from lucy.voice.translit import to_latin

            text = to_latin(raw)
        else:
            text = raw
        reason = junk_reason(text, audio.duration, reply, level, self.guard)
        if reason:
            logger.info(
                "dropped a transcript (%s, %.1fs of audio): %r", reason, audio.duration, raw
            )
            text = ""
        else:
            text = collapse_repeats(text, times=self.guard.max_repeats)
        logger.debug(
            "transcribed %.1fs of audio in %.2fs (%.2fs waiting; worker %.2fs, %s tokens, %s)",
            audio.duration,
            time.perf_counter() - started,
            started - waited,
            reply.get("seconds", 0.0),
            reply.get("tokens"),
            reply.get("lang"),
        )
        return Transcript(text=text, language=lang)

    async def aclose(self) -> None:
        """Stop the worker (and its process group) and the thread. Idempotent."""
        self._closed = True
        async with self._lock:
            if self._executor is not None:
                await self._call(self._stop_blocking)
                executor, self._executor = self._executor, None
                executor.shutdown(wait=False)
            elif self._worker is not None:
                self._stop_blocking()

    def _check_open(self) -> None:
        if self._closed:
            raise STTError("the speech recogniser has been closed")


def _is_hex(text: str) -> bool:
    try:
        int(text, 16)
    except ValueError:
        return False
    return True


def _noise(samples: int) -> bytes:
    """Quiet noise (about -40 dBFS) as PCM, so the whole decoder path runs."""
    import numpy as np

    noise = np.random.default_rng(0).normal(0, 0.01, samples)
    return (noise * 32767).clip(-32768, 32767).astype("<i2").tobytes()


def _pcm_dbfs(data: bytes) -> float:
    """RMS level of 16-bit PCM in dBFS (-inf for digital silence or no samples)."""
    import numpy as np

    samples = np.frombuffer(data, dtype="<i2").astype("float64") / 32768.0
    if not len(samples):
        return -math.inf
    rms = float(np.sqrt(np.mean(np.square(samples))))
    return 20 * math.log10(rms) if rms > 0 else -math.inf
