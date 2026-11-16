"""Named recogniser configurations for the accuracy check.

Every recogniser has the same shape: `load()` (load weights, timed separately by
run.py), `transcribe(pcm) -> str` with `pcm` 16 kHz mono 16-bit little-endian PCM
(Lucy's core input format), and `close()`. They are synchronous: run.py times each
call. Recognisers backed by a Lucy STT adapter go through the registry and the
adapter's own `transcribe(AudioChunk)`, so they measure exactly what Lucy runs.

Adding a configuration: write a factory `def _name() -> Recogniser` and add it to
CONFIGS.

`indic_flex*` are not Lucy adapters: they run the gated Hugging Face model
bodhan-ai/indic-transcribe-flex (a 1.2B fine-tune of NVIDIA Canary 1B v2 for Indian
languages and Indian-accented English) through `WorkerRecogniser`, which drives
flex_worker.py, a PEP 723 script with pinned torch and transformers started by
`uv run --script`, so none of that enters Lucy's environment. The worker loads the
model code and weights of one reviewed commit (flex_worker.REVISION) from the
Hugging Face cache; it needs HF_TOKEN (or a saved login) with the licence accepted
the first time, to download. `indic_flex` forces English with mixed mode (Latin
script, inverse text normalisation to digits); `indic_flex_auto` lets the model
identify the language per clip; `indic_flex_fp16` is `indic_flex` with float16
weights and `indic_flex_cpu` is `indic_flex` on the CPU (float32), both for checking.

`whisper_turbo*` and `qwen3_asr*` are not Lucy adapters either: they run in
mlx_asr_worker.py (pinned mlx-whisper 0.4.3 or mlx-qwen3-asr 0.4.4, the same worker
protocol). `whisper_turbo` is Whisper large-v3-turbo (mlx-community MLX weights),
English forced, temperature 0.0 with no fallback, otherwise library defaults;
`whisper_turbo_prompt` adds `initial_prompt` = `prompt_sentence(read_words())`.
`qwen3_asr` is Qwen/Qwen3-ASR-0.6B, language English, greedy; `qwen3_asr_context`
passes the words as the model's context (system prompt), comma separated.
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from lucy.core import registry
from lucy.core.types import INPUT_CHANNELS, INPUT_SAMPLE_RATE, AudioChunk

WORDS_FILE = Path(__file__).resolve().parent / "words.txt"
BOOST_WEIGHT = 2.0
"""Log-probability bonus per boosted word or phrase (parakeet-mlx `boost.words`)."""
BEAM_SIZE = 5


class Recogniser(Protocol):
    name: str

    def load(self) -> None: ...

    def transcribe(self, pcm: bytes) -> str: ...

    def close(self) -> None: ...


class AdapterRecogniser:
    """A Lucy STT adapter, created through the registry, driven synchronously."""

    def __init__(self, name: str, module: str, adapter: str, **options: Any) -> None:
        self.name = name
        registry.load_adapters([module])
        self._stt = registry.create("stt", adapter, **options)
        self._loop = asyncio.new_event_loop()

    def load(self) -> None:
        # Load only (no warm-up transcription), so the load time is the weights alone.
        load = getattr(self._stt, "_load_blocking", None) or getattr(
            self._stt, "_start_blocking", None
        )
        if load is not None:
            self._loop.run_until_complete(self._stt._call(load))

    def transcribe(self, pcm: bytes) -> str:
        chunk = AudioChunk(data=pcm, sample_rate=INPUT_SAMPLE_RATE, channels=INPUT_CHANNELS)
        return self._loop.run_until_complete(self._stt.transcribe(chunk)).text

    def close(self) -> None:
        aclose = getattr(self._stt, "aclose", None)
        if aclose is not None:
            self._loop.run_until_complete(aclose())
        self._loop.close()


def read_words(path: Path = WORDS_FILE) -> list[str]:
    """Boost words: one word or phrase per line; blank lines and # comments ignored."""
    lines = (line.split("#", 1)[0].strip() for line in path.read_text().splitlines())
    return [line for line in lines if line]


def _beam_backend(words: list[str] | None) -> Any:
    """A parakeet backend that decodes with beam search and, given words, boosts them.

    Uses the API of the installed parakeet-mlx (0.7.0): `parakeet_mlx.Beam(booster=...)`
    in `DecodingConfig`, with `boost.lift(boost.words({...}), model.vocabulary)`.
    Boosting is case-sensitive on the characters the model emits."""
    from lucy.adapters.stt_parakeet_mlx import _ParakeetBackend

    class BeamBackend(_ParakeetBackend):
        def __init__(self) -> None:
            super().__init__()
            self._config: Any = None

        def _decoding(self, model: Any) -> Any:
            if self._config is None:
                from parakeet_mlx import Beam, DecodingConfig, boost

                beam = Beam(beam_size=BEAM_SIZE)
                if words:
                    machine = boost.words({w: BOOST_WEIGHT for w in words})
                    beam = Beam(beam_size=BEAM_SIZE, booster=boost.lift(machine, model.vocabulary))
                self._config = DecodingConfig(decoding=beam)
            return self._config

        def transcribe(self, model: Any, samples: Any) -> str:
            self._modules()
            mel = self._get_logmel(self._mx.array(samples), model.preprocessor_config)
            results = model.generate(mel, decoding_config=self._decoding(model))
            return results[0].text if results else ""

    return BeamBackend()


_PARAKEET = "lucy.adapters.stt_parakeet_mlx"


def _parakeet() -> Recogniser:
    """Today's adapter exactly: library defaults (greedy decoding), no biasing."""
    return AdapterRecogniser("parakeet", _PARAKEET, "parakeet_mlx")


def _parakeet_beam() -> Recogniser:
    """Control for parakeet_boost: the same beam search with no words boosted."""
    return AdapterRecogniser("parakeet_beam", _PARAKEET, "parakeet_mlx", backend=_beam_backend([]))


def _parakeet_boost() -> Recogniser:
    """Beam search with the words in words.txt boosted."""
    return AdapterRecogniser(
        "parakeet_boost", _PARAKEET, "parakeet_mlx", backend=_beam_backend(read_words())
    )


FLEX_WORKER = Path(__file__).resolve().parent / "flex_worker.py"
MLX_WORKER = Path(__file__).resolve().parent / "mlx_asr_worker.py"


class WorkerRecogniser:
    """A model run by a worker script (flex_worker.py, mlx_asr_worker.py) in its own uv env.

    The worker's packages stay out of Lucy's environment: it is a PEP 723 script
    started with `uv run --script`, kept running across clips, and spoken to in JSON
    lines (see flex_worker.py's docstring). `load()` returns once the model is loaded;
    `worker_load_s` is the worker's own load time (without the environment start)."""

    def __init__(self, name: str, script: Path = FLEX_WORKER, **options: str) -> None:
        self.name = name
        self._script = script
        self._args = [f"--{k}={v}" for k, v in options.items()]
        self._proc: subprocess.Popen[str] | None = None
        self.worker_load_s: float | None = None
        self.device: str | None = None
        self.langs: list[str] = []
        self.extras: list[dict[str, Any]] = []  # per clip: the reply less text and lang
        self.ready: dict[str, Any] = {}

    def _ask(self, request: dict[str, Any]) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        return self._reply()

    def _reply(self) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdout
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.name}: worker exited (code {self._proc.poll()})")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(f"{self.name}: {reply['error']}")
        return reply

    def load(self) -> None:
        cmd = ["uv", "run", "--quiet", "--script", str(self._script), *self._args]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        ready = self._reply()
        self.ready = ready
        self.worker_load_s = ready["load_s"]
        self.device = f"{ready['device']} {ready['dtype']}"

    def transcribe(self, pcm: bytes) -> str:
        reply = self._ask({"pcm": base64.b64encode(pcm).decode("ascii")})
        self.langs.append(reply["lang"])
        self.extras.append({k: v for k, v in reply.items() if k not in ("text", "lang")})
        return reply["text"]

    def stats(self) -> dict[str, Any]:
        """The worker's own figures (peak resident size, GPU memory, in MB)."""
        return {
            **self._ask({"stats": True}),
            "worker_load_s": self.worker_load_s,
            "ready": self.ready,
            "device": self.device,
            "langs": dict(Counter(self.langs)),
        }

    def close(self) -> None:
        if self._proc is not None:
            if self._proc.stdin:
                self._proc.stdin.close()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


def _indic_flex() -> Recogniser:
    """indic-transcribe-flex, English forced, mixed mode (Latin script, digits)."""
    return WorkerRecogniser("indic_flex", lang="en", mode="mixed")


def _indic_flex_auto() -> Recogniser:
    """indic-transcribe-flex, language identified per clip by the model, mixed mode."""
    return WorkerRecogniser("indic_flex_auto", lang="auto", mode="mixed")


def _indic_flex_fp16() -> Recogniser:
    """indic_flex with float16 weights on MPS."""
    return WorkerRecogniser("indic_flex_fp16", lang="en", mode="mixed", dtype="float16")


def _indic_flex_cpu() -> Recogniser:
    """indic_flex on the CPU, to check the MPS output against."""
    return WorkerRecogniser("indic_flex_cpu", lang="en", mode="mixed", device="cpu")


def prompt_sentence(words: list[str]) -> str:
    """The boost words as one natural sentence, for Whisper's `initial_prompt`."""
    return f"Names and words that come up: {', '.join(words[:-1])} and {words[-1]}."


def _whisper_turbo() -> Recogniser:
    """Whisper large-v3-turbo, English, temperature 0 (greedy), no prompt."""
    return WorkerRecogniser("whisper_turbo", MLX_WORKER, model="whisper")


def _whisper_turbo_prompt() -> Recogniser:
    """whisper_turbo with the words in words.txt as `initial_prompt`."""
    prompt = prompt_sentence(read_words())
    return WorkerRecogniser("whisper_turbo_prompt", MLX_WORKER, model="whisper", prompt=prompt)


def _qwen3_asr() -> Recogniser:
    """Qwen3-ASR 0.6B, English, greedy, no context."""
    return WorkerRecogniser("qwen3_asr", MLX_WORKER, model="qwen3")


def _qwen3_asr_context() -> Recogniser:
    """qwen3_asr with the words in words.txt as the model's context."""
    context = ", ".join(read_words())
    return WorkerRecogniser("qwen3_asr_context", MLX_WORKER, model="qwen3", prompt=context)


# --- Multilingual configurations (English, Hindi, mixed) -------------------------
# All flex ones use float16 on MPS. "native" mode writes each language in its own
# script (scored); "mixed" is Latin script with digits, what a chat model would get
# in English mode.


def _flex(name: str, **options: str) -> Callable[[], Recogniser]:
    return lambda: WorkerRecogniser(name, dtype="float16", **options)


def _mlx(name: str, model: str, lang: str) -> Callable[[], Recogniser]:
    return lambda: WorkerRecogniser(name, MLX_WORKER, model=model, lang=lang)


MULTILINGUAL: dict[str, Callable[[], Recogniser]] = {
    "whisper_turbo_auto": _mlx("whisper_turbo_auto", "whisper", "auto"),
    "whisper_turbo_hi": _mlx("whisper_turbo_hi", "whisper", "hi"),
    "qwen3_asr_auto": _mlx("qwen3_asr_auto", "qwen3", "auto"),
    "qwen3_asr_hi": _mlx("qwen3_asr_hi", "qwen3", "hi"),
    "indic_flex_hi": _flex("indic_flex_hi", lang="hi", mode="native"),
    "indic_flex_hi_mixed": _flex("indic_flex_hi_mixed", lang="hi", mode="mixed"),
    "indic_flex_auto_native": _flex("indic_flex_auto_native", lang="auto", mode="native"),
    "indic_flex_enhi": _flex("indic_flex_enhi", lang="auto", mode="native", allowed="en,hi"),
    "indic_flex_enhi_mixed": _flex(
        "indic_flex_enhi_mixed", lang="auto", mode="mixed", allowed="en,hi"
    ),
}
"""`*_auto`: the model identifies the language per clip (Whisper among its 99, Qwen
among its 30, flex among its recommended set); `*_hi`: Hindi forced. `indic_flex_enhi*`:
flex identification restricted to English and Hindi (`allowed_langs`), a router."""

# --- Latin-only configurations (flex alone, English and Hindi, Latin script out) -----
# Lucy's choice: one recogniser for English, Hindi and mixed speech whose output is
# always Latin script. `flex_*_roman` use the model's romanized mode (everything
# transliterated by the model); `*_latin` take mixed mode and write any Devanagari
# left in Latin with translit.to_latin (pure Python, deterministic).


class LatinRecogniser(WorkerRecogniser):
    """A flex worker whose output has any Devanagari written in Latin (translit.py)."""

    def transcribe(self, pcm: bytes) -> str:
        from lucy.voice.translit import to_latin

        raw = super().transcribe(pcm)
        self.extras[-1]["raw"] = raw
        return to_latin(raw)


def _latin(name: str, **options: str) -> Callable[[], Recogniser]:
    return lambda: LatinRecogniser(name, dtype="float16", **options)


LATIN: dict[str, Callable[[], Recogniser]] = {
    "flex_en_mixed": _flex("flex_en_mixed", lang="en", mode="mixed"),
    "flex_en_roman": _flex("flex_en_roman", lang="en", mode="romanized"),
    "flex_enhi_roman": _flex("flex_enhi_roman", lang="auto", mode="romanized", allowed="en,hi"),
    "flex_hi_roman": _flex("flex_hi_roman", lang="hi", mode="romanized"),
    "flex_enhi_mixed_latin": _latin(
        "flex_enhi_mixed_latin", lang="auto", mode="mixed", allowed="en,hi"
    ),
    "flex_hi_mixed_latin": _latin("flex_hi_mixed_latin", lang="hi", mode="mixed"),
}
"""a. `flex_en_mixed` English forced, mixed (identification never runs); b.
`flex_en_roman`; c. `flex_enhi_roman` identification among en/hi, romanized; d.
`flex_hi_roman` Hindi forced, romanized; e. `flex_enhi_mixed_latin` en/hi
identification, mixed, then translit; `flex_hi_mixed_latin` Hindi forced, mixed, then
translit. All float16 on MPS, greedy (the model's generation_config: no sampling, one
beam)."""


def _flex_adapter() -> Recogniser:
    """Lucy's `indic_flex` STT adapter with its defaults, through the registry: the
    shipped form of `flex_enhi_mixed_latin` (worker, translit and guard)."""
    return AdapterRecogniser("flex_adapter", "lucy.adapters.stt_indic_flex", "indic_flex")


CONFIGS: dict[str, Callable[[], Recogniser]] = {
    "flex_adapter": _flex_adapter,
    "parakeet": _parakeet,
    "parakeet_beam": _parakeet_beam,
    "parakeet_boost": _parakeet_boost,
    "indic_flex": _indic_flex,
    "indic_flex_auto": _indic_flex_auto,
    "indic_flex_fp16": _indic_flex_fp16,
    "indic_flex_cpu": _indic_flex_cpu,
    "whisper_turbo": _whisper_turbo,
    "whisper_turbo_prompt": _whisper_turbo_prompt,
    "qwen3_asr": _qwen3_asr,
    "qwen3_asr_context": _qwen3_asr_context,
    **MULTILINGUAL,
    **LATIN,
}
