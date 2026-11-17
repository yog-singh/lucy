"""TTS adapter running NVIDIA Magpie TTS on the CPU in a worker process (ADR 0024).

English and Hindi in one voice, through NeMo-Speech.cpp (installed by hand under
`data/tools/nemo-speech/`, see docs/voice.md). `magpie_worker.py` (shipped next to this
module, standard library only) is started with Lucy's own Python; it loads the runtime's
C library with ctypes, loads the model once and streams the PCM the library hands to its
callback (about 186 ms of audio per chunk) as JSON lines. The runtime's HTTP server
(`nemo-speech serve`) is not used: its `/v1/audio/speech` route returns whole clips.
Process handling is shared with the `indic_flex` recogniser (`worker_process`).

Per sentence the adapter picks the language: `hi-IN` when the sentence contains
Devanagari, otherwise `en-US`; the `language` option forces one. With `hi-IN` Magpie
reads Devanagari and digits (its Sparrowhawk normaliser turns 7:30, 1250, 40% into
words) but spells Latin-script words letter by letter, so `prepare` respells Latin
words inside a Devanagari sentence in Devanagari (`lucy.core.devanagari`); a sentence
that is mostly Latin words goes as `en-US` instead. A Latin-script sentence that reads
as romanised Hindi (`lucy.core.hinglish.latin_language`: the chat model sometimes writes
Hindi in Latin letters despite the persona) is converted to Devanagari
(`romanised_to_devanagari`) and sent as `hi-IN`; one too short to tell ("Ok.") follows
the language of the previous sentence. `roman_hindi = false` turns the conversion off.

Streaming and stopping: each request has an id; the worker answers with several audio
messages and a final "done". A reader thread routes the worker's messages by id to the
request's queue, so audio is yielded as it arrives and audio of an abandoned request
can never reach another. Requests are served by the worker in order, so a second request
(the player synthesises one sentence ahead) waits behind the first. Closing a generator
early sends a cancel, which the worker's callback honours at the next chunk.

Timeouts: `first_audio_timeout` from the worker starting the request to its first
audio, `timeout` for the whole request; either kills the worker (it is restarted on the
next request). A worker that dies is restarted once and the request retried if none of
its audio was yielded yet; otherwise the request fails with `TTSError`.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import queue
import sys
import threading
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, Final

from lucy.adapters.tts_openai_compat import TTSError, _Resampler
from lucy.adapters.worker_process import WorkerGone, WorkerProcess, worker_environment
from lucy.core import registry
from lucy.core.devanagari import respell_latin_words, romanised_to_devanagari
from lucy.core.hinglish import latin_language
from lucy.core.types import OUTPUT_SAMPLE_RATE, SAMPLE_WIDTH_BYTES, AudioChunk, TTSCaps

logger = logging.getLogger(__name__)

DEFAULT_ROOT: Final = "data/tools/nemo-speech"
MODEL_FILE: Final = "models/magpie_tts_multilingual_357m.v2607.q8_0.gguf"
CODEC_FILE: Final = "models/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf"
TOKENIZER_GLOB: Final = "models/nvidia/magpie_tts_multilingual_357m/*/tokenizer"
LIB_DIR: Final = "cpu/lib"
TN_DIR: Final = "tn_configs"
WORKER_SCRIPT: Final = Path(__file__).with_name("magpie_worker.py")
LANGUAGES: Final = ("en-US", "hi-IN")
WARM_UP_TEXTS: Final = ("Ready.", "ठीक है, 7 बजे।")
"""One per language: each language's tokenizer and normaliser load on first use."""
WHERE: Final = (
    "install NeMo-Speech.cpp 0.3.0 and the Magpie files by hand as docs/voice.md describes "
    "(github.com/NVIDIA-NeMo/NeMo-Speech.cpp releases; huggingface.co/nvidia/"
    "magpie_tts_multilingual_357m and nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps)"
)


def has_devanagari(text: str) -> bool:
    return any("ऀ" <= c <= "ॿ" for c in text)


def prepare(
    text: str,
    language: str | None = None,
    *,
    previous: str | None = None,
    roman_hindi: bool = True,
) -> tuple[str, str]:
    """The text to send and its language code: `hi-IN` with Devanagari (Latin words
    respelled in Devanagari, unless they are most of the words: then `en-US` as is);
    for Latin text, with `roman_hindi`, `hi-IN` converted to Devanagari when it reads as
    romanised Hindi, or when it is too short to tell and `previous` (the code of the
    previous sentence) was `hi-IN`; `en-US` otherwise. `language` forces the code
    (Latin words are still respelled for `hi-IN`)."""
    if language == "en-US":
        return text, "en-US"
    if language is None and not has_devanagari(text):
        verdict = latin_language(text) if roman_hindi else "en"
        if verdict == "hi" or (verdict is None and previous == "hi-IN"):
            return romanised_to_devanagari(text), "hi-IN"
        return text, "en-US"
    respelled, latin, total = respell_latin_words(text)
    if language is None and total and latin * 2 > total:
        return text, "en-US"
    return respelled, "hi-IN"


def resolve_paths(
    root: str, model: str | None, codec: str | None, tokenizer: str | None, tn: str | None
) -> dict[str, Path]:
    """The runtime's files from the options (defaults under `root`); `TTSError` naming
    each one that is missing."""
    base = Path(root).expanduser()
    if tokenizer is None:
        found = sorted(base.glob(TOKENIZER_GLOB))
        tok = found[0] if found else base / TOKENIZER_GLOB
    else:
        tok = Path(tokenizer).expanduser()
    paths = {
        "runtime library folder": base / LIB_DIR,
        "model": Path(model).expanduser() if model else base / MODEL_FILE,
        "codec": Path(codec).expanduser() if codec else base / CODEC_FILE,
        "tokenizer folder": tok,
        "text-normalisation grammars": Path(tn).expanduser() if tn else base / TN_DIR,
    }
    missing = [f"{what} ({path})" for what, path in paths.items() if not path.exists()]
    lib = paths["runtime library folder"]
    if lib.exists() and not any(lib.glob("libnemo_speech_tts.*")):
        missing.append(f"libnemo_speech_tts in {lib}")
    if missing:
        raise TTSError(f"the magpie voice is missing {'; '.join(missing)}: {WHERE}")
    return paths


class _Request:
    """One synthesis request: its id and the queue its worker messages go to."""

    def __init__(self, rid: int, generation: int) -> None:
        self.id = rid
        self.generation = generation  # which worker it was sent to
        self.messages: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()


@registry.register("tts", "magpie")
class MagpieTTS:
    """Magpie speech in a worker, streamed sentence by sentence (module docstring).

    Options (all from `[tts]`): `root` (the NeMo-Speech.cpp folder, default
    data/tools/nemo-speech), `model`, `codec`, `tokenizer`, `tn` (paths overriding the
    defaults under `root`), `voice` (default Sofia; checked against the model's voices
    when the worker starts), `language` (`en-US` or `hi-IN` to force one), `threads`
    (CPU threads for synthesis, 4), `seed` (fixed sampling seed so a sentence sounds the
    same each time; null for a random one), `timeout` (seconds per sentence),
    `first_audio_timeout`, `start_timeout`, `roman_hindi` (convert romanised Hindi
    sentences to Devanagari for `hi-IN`, default true). `command` is for tests: the worker command
    line instead of Lucy's Python running `magpie_worker.py` (paths are then not checked).
    """

    capabilities = TTSCaps(streaming=True, languages=frozenset({"en", "hi"}))

    def __init__(
        self,
        *,
        root: str = DEFAULT_ROOT,
        model: str | None = None,
        codec: str | None = None,
        tokenizer: str | None = None,
        tn: str | None = None,
        voice: str = "Sofia",
        language: str | None = None,
        threads: int = 4,
        seed: int | None = 0,
        timeout: float = 30.0,
        first_audio_timeout: float = 5.0,
        start_timeout: float = 60.0,
        roman_hindi: bool = True,
        command: Sequence[str] | None = None,
    ) -> None:
        for name, value in (("root", root), ("model", model), ("codec", codec),
                            ("tokenizer", tokenizer), ("tn", tn)):  # fmt: skip
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a path")
        if not isinstance(voice, str) or not voice.strip():
            raise ValueError("voice must be a voice name, e.g. Sofia")
        if language is not None and language not in LANGUAGES:
            raise ValueError(f"language must be one of {list(LANGUAGES)} (or left out)")
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 64:
            raise ValueError("threads must be an integer from 1 to 64")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError("seed must be a non-negative integer (or left out for none)")
        for name, value in (
            ("timeout", timeout),
            ("first_audio_timeout", first_audio_timeout),
            ("start_timeout", start_timeout),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                raise ValueError(f"{name} must be a positive number of seconds")
        if not isinstance(roman_hindi, bool):
            raise ValueError("roman_hindi must be true or false")
        self.root, self.voice, self.language = root, voice, language
        self.roman_hindi = roman_hindi
        self._previous_lang: str | None = None  # the last sentence's code
        self._paths = (model, codec, tokenizer, tn)
        self.threads, self.seed = threads, seed
        self.timeout, self.first_audio_timeout = float(timeout), float(first_audio_timeout)
        self.start_timeout = float(start_timeout)
        self.sample_rate = OUTPUT_SAMPLE_RATE  # the worker's, once it is ready
        self._command = list(command) if command is not None else None
        self._worker: WorkerProcess | None = None
        self._generation = 0
        self._pending: dict[int, _Request] = {}
        self._next_id = 0
        self._start_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._closed = False

    @property
    def loaded(self) -> bool:
        """Whether the worker is running with the model loaded."""
        return self._worker is not None and self._worker.alive()

    # --- The worker ----------------------------------------------------------------

    def _on_line(self, generation: int, line: bytes) -> None:
        """On the worker's reader thread: route one message to its request."""
        current = generation == self._generation
        if not line:
            self._deliver(generation, None)
            if current:
                self._ready.put(None)
            return
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("[tts worker stdout] %s", line[:200].decode("utf-8", "replace"))
            return
        if "id" not in msg:
            if current:
                self._ready.put(msg)
            return
        self._deliver(generation, msg)

    def _deliver(self, generation: int, msg: dict[str, Any] | None) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        with contextlib.suppress(RuntimeError):  # the loop closed meanwhile
            loop.call_soon_threadsafe(self._route, generation, msg)

    def _route(self, generation: int, msg: dict[str, Any] | None) -> None:
        """On the event loop: hand `msg` (None: the worker is gone) to its request."""
        if msg is None:
            for request in list(self._pending.values()):
                if request.generation == generation:
                    request.messages.put_nowait(None)
            return
        request = self._pending.get(msg["id"])
        if request is not None and request.generation == generation:
            request.messages.put_nowait(msg)  # anything else is for an abandoned request

    def _start_blocking(self, generation: int) -> WorkerProcess:
        if self._command is not None:
            command, args = self._command, [f"--threads={self.threads}", f"--voice={self.voice}"]
        else:
            p = resolve_paths(self.root, *self._paths)
            command = [sys.executable, "-I", str(WORKER_SCRIPT)]
            args = [
                f"--lib-dir={p['runtime library folder']}",
                f"--model={p['model']}",
                f"--codec={p['codec']}",
                f"--tokenizer={p['tokenizer folder']}",
                f"--tn={p['text-normalisation grammars']}",
                f"--threads={self.threads}",
                f"--voice={self.voice}",
            ]
        if self.seed is not None:
            args.append(f"--seed={self.seed}")
        self._ready = queue.Queue()
        try:
            worker = WorkerProcess(
                [*command, *args],
                worker_environment(offline=True),
                lambda line: self._on_line(generation, line),
                name="tts worker",
                log=logger,
            )
        except OSError as exc:
            raise TTSError(f"cannot start the speech worker: {exc}") from exc
        reply: dict[str, Any] | None = {"fatal": f"not ready after {self.start_timeout:.0f}s"}
        deadline = time.monotonic() + self.start_timeout
        while time.monotonic() < deadline:
            try:
                reply = self._ready.get(timeout=0.2)
                break
            except queue.Empty:
                if not worker.alive() and self._ready.empty():
                    reply = None
                    break
        if reply is None or "fatal" in reply or not reply.get("ready"):
            worker.kill()
            if reply is None:
                why = "the worker exited while loading it; see the debug log for its output"
            else:
                why = str(reply.get("fatal") or reply)
            raise TTSError(f"cannot load the magpie voice: {why}; {WHERE}")
        voices = [str(v) for v in reply.get("voices") or []]
        if voices and self.voice.lower() not in {v.lower() for v in voices}:
            worker.kill()
            raise TTSError(
                f"unknown voice {self.voice!r} for magpie; choose one of "
                f"{', '.join(voices)} (`voice` in [tts])"
            )
        rate = reply.get("sample_rate")
        self.sample_rate = int(rate) if isinstance(rate, int) and rate > 0 else OUTPUT_SAMPLE_RATE
        logger.info(
            "speech worker ready: magpie at %d Hz in %.1fs", self.sample_rate,
            float(reply.get("load_s", 0.0)),
        )  # fmt: skip
        return worker

    async def _ensure_worker(self) -> tuple[WorkerProcess, int]:
        """The running worker and its generation, started if needed."""
        self._loop = asyncio.get_running_loop()
        async with self._start_lock:
            if self._closed:
                raise TTSError("the magpie voice has been closed")
            if self._worker is not None and self._worker.alive():
                return self._worker, self._generation
            self._discard_worker()
            self._generation += 1
            worker = await asyncio.to_thread(self._start_blocking, self._generation)
            self._worker = worker
            return worker, self._generation

    def _discard_worker(self, generation: int | None = None) -> None:
        """Kill the worker (only if it is still `generation`, when given)."""
        if generation is not None and generation != self._generation:
            return
        worker, self._worker = self._worker, None
        if worker is not None:
            threading.Thread(target=worker.kill, daemon=True).start()

    # --- The port ------------------------------------------------------------------

    async def warm_up(self) -> None:
        """Start the worker (load the model, under a second) and synthesise a short
        phrase in each language, so the first reply does not pay for loading a
        tokenizer or normaliser (about 0.5 s each). Raises `TTSError`."""
        for text in WARM_UP_TEXTS:
            if self.language is None or prepare(text)[1] == self.language:
                async for _ in self.synthesize(text):
                    pass

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """Stream `text` as speech in the core output format (24 kHz mono 16-bit).

        Blank text yields nothing. Raises `TTSError`."""
        if self._closed:
            raise TTSError("the magpie voice has been closed")
        if not text.strip():
            return
        spoken, lang = prepare(
            text, self.language, previous=self._previous_lang, roman_hindi=self.roman_hindi
        )
        self._previous_lang = lang
        yielded = False
        for attempt in (1, 2):
            try:
                async for chunk in self._attempt(spoken, lang):
                    yielded = True
                    yield chunk
                return
            except WorkerGone as exc:
                if yielded or attempt == 2:
                    raise TTSError(
                        f"the speech worker died during a sentence ({exc}); see the debug log"
                    ) from None
                logger.warning("the speech worker died (%s); restarting it", exc)

    async def _attempt(self, text: str, lang: str) -> AsyncIterator[AudioChunk]:
        worker, generation = await self._ensure_worker()
        self._next_id += 1
        request = _Request(self._next_id, generation)
        ahead = sum(1 for r in self._pending.values() if r.generation == generation)
        self._pending[request.id] = request
        done = False
        resampler = _Resampler(self.sample_rate) if self.sample_rate != OUTPUT_SAMPLE_RATE else None
        try:
            worker.send(
                {"id": request.id, "text": text, "voice": self.voice, "lang": lang,
                 "seed": self.seed}
            )  # fmt: skip
            # Before it starts, a request may wait for those ahead of it (each bounded).
            deadline = time.monotonic() + self.first_audio_timeout + ahead * self.timeout
            started_at: float | None = None
            heard = False
            while True:
                try:
                    msg = await self._next_message(request, worker, deadline)
                except TimeoutError:
                    self._discard_worker(generation)
                    what = "the first audio" if not heard else "the whole sentence"
                    raise TTSError(
                        f"the speech worker took too long for {what} of {text[:40]!r}; it was "
                        "stopped and will be restarted on the next sentence"
                    ) from None
                if msg is None:
                    self._discard_worker(generation)
                    raise WorkerGone(f"the worker exited (code {worker.proc.poll()})")
                if "error" in msg:
                    done = True
                    raise TTSError(f"speech synthesis failed: {msg['error']}")
                if msg.get("start"):
                    started_at = time.monotonic()
                    deadline = started_at + self.first_audio_timeout
                elif "audio" in msg:
                    if not heard:
                        heard = True
                        deadline = (started_at or time.monotonic()) + self.timeout
                    pcm = base64.b64decode(msg["audio"])
                    pcm = pcm[: len(pcm) - len(pcm) % SAMPLE_WIDTH_BYTES]
                    if resampler is not None:
                        pcm = resampler.feed(pcm)
                    if pcm:
                        yield AudioChunk(data=pcm, sample_rate=OUTPUT_SAMPLE_RATE)
                elif msg.get("done"):
                    done = True
                    if resampler is not None and (tail := resampler.flush()):
                        yield AudioChunk(data=tail, sample_rate=OUTPUT_SAMPLE_RATE)
                    logger.debug(
                        "synthesised %r (%s): first audio %.2fs, total %.2fs",
                        text[:40], lang, msg.get("first_s") or 0.0, msg.get("total_s") or 0.0,
                    )  # fmt: skip
                    return
        finally:
            self._pending.pop(request.id, None)
            if not done and worker is self._worker:
                with contextlib.suppress(WorkerGone):
                    worker.send({"cancel": request.id})  # abandon it; its audio is dropped

    @staticmethod
    async def _next_message(
        request: _Request, worker: WorkerProcess, deadline: float
    ) -> dict[str, Any] | None:
        """The request's next message (None: the worker is gone); `TimeoutError` at
        `deadline`. Polls the process too."""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            try:
                return await asyncio.wait_for(request.messages.get(), min(remaining, 0.2))
            except TimeoutError:
                if not worker.alive() and request.messages.empty():
                    return None

    async def aclose(self) -> None:
        """Stop the worker (and its process group). Idempotent."""
        self._closed = True
        async with self._start_lock:
            worker, self._worker = self._worker, None
            if worker is not None:
                await asyncio.to_thread(worker.stop)
