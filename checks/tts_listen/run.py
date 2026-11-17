"""Synthesise every sentence with every text-to-speech candidate, for a listening test.

usage (from the repo root):
  uv run python checks/tts_listen/run.py [--engines phonon,maya,magpie_adapter,dhvaani,mithi]
      [--maya-voices Ahana,Vaidehi] [--sentences id,id] [--label NAME] [--out DIR]
      [--no-loaded] [--no-check] [--chat-model gemma4:12b-mlx]

Writes OUT/LABEL/<column>/<nn>.wav (a column is one model with one voice or setting),
results.json (every measured row and each engine's facts) and index.html (the page
to listen on, opened from disk). Default OUT data/checks/tts_listen/. See
checks/README.md.

Each engine runs in its own worker (maya_worker.py, dhvaani_worker.py,
mithi_worker.py: PEP 723 scripts under `uv run --script`) except today's voice,
Ollama `phonon`, which goes through Lucy's own `ollama` TTS adapter. Per engine: the
worker is started (load time), one warm-up sentence is synthesised and discarded,
then every sentence in a language the engine speaks is timed ("idle"). Unless
--no-loaded, the chat model is then loaded in Ollama by a one-token request with a
keep-alive and LOADED_IDS are timed again ("loaded"); the chat model is unloaded
again (a keep_alive 0 request) before the next engine. Unless --no-check, every
idle clip except Bengali is transcribed by Lucy's own recognisers through the
registry (Parakeet for English, `indic_flex` for Hindi, mixed and romanised Hindi)
and gross failures are flagged (report.flags).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "stt_accuracy"))
from common import out_dir  # noqa: E402
from report import (  # noqa: E402
    compare_loaded,
    flags,
    render_page,
    rtf,
    simulate_playback,
    summarise,
)

OLLAMA = "http://localhost:11434"
LOADED_IDS = ["en_time", "en_name", "hi_date", "mix_email", "bn_weather"]
LOADED_IDS += ["mix_time", "roman_amount", "en_long", "mix_name"]  # the light1 set
# Synthesised back to back while the chat model generates (--generating).
GEN_IDS = ["en_time", "mix_time", "hi_amount", "roman_temp", "en_name", "mix_name"]
GEN_IDS += ["roman_calendar", "en_long"]
GEN_PROMPT = "Write a long, detailed essay on the history of the Indian railways."
NEMO = HERE.parent.parent / "data" / "tools" / "nemo-speech"
MAGPIE_GGUF = "magpie_tts_multilingual_357m.v2607.q8_0.gguf"
CODEC_GGUF = "nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf"
MAGPIE_TOKENIZER = (
    "nvidia/magpie_tts_multilingual_357m/19806879b16d3f2ccf28fb112b1bcd16a3c7923e/tokenizer"
)
WARM_UP = {"en": "Okay.", "hi": "ठीक है।", "bn": "ঠিক আছে।"}

ENGINES: dict[str, dict[str, Any]] = {
    "phonon": {
        "title": "phonon (today)",
        "licence": "Apache-2.0",
        "langs": {"en"},
        "variants": [{"column": "phonon_maeve", "title": "phonon · Maeve", "opts": {}}],
    },
    "maya": {
        "title": "maya2-native 8-bit",
        "script": "maya_worker.py",
        "licence": "Apache-2.0 (codec CC-BY-4.0)",
        "langs": {"en", "hi", "bn"},
    },
    "magpie_adapter": {
        "title": "Magpie through Lucy's adapter (CPU, streaming)",
        "licence": "NVIDIA Open Model License (runtime Apache-2.0)",
        "langs": {"en", "hi"},
        "variants": [{"column": "magpie_adapter", "title": "Magpie adapter · Sofia", "opts": {}}],
    },
    "dhvaani": {
        "title": "DhVaani 0.5",
        "script": "dhvaani_worker.py",
        "licence": "CC-BY-NC-4.0 (personal use only)",
        "langs": {"en", "hi", "bn"},
        "variants": [
            {"column": "dhvaani_s16", "title": "DhVaani · 16 steps", "opts": {"num_step": 16}},
            {"column": "dhvaani_s8", "title": "DhVaani · 8 steps", "opts": {"num_step": 8}},
        ],
    },
    "magpie": {
        "title": "Magpie TTS 357M v2607 Q8_0 (NeMo-Speech.cpp, CPU)",
        "licence": "NVIDIA Open Model License; runtime Apache-2.0",
        "langs": {"en", "hi"},
        "device": "cpu",
        "variants": [
            {"column": "magpie_sofia", "title": "Magpie CPU · Sofia", "opts": {"voice": "Sofia"}},
            {"column": "magpie_aria", "title": "Magpie CPU · Aria", "opts": {"voice": "Aria"}},
        ],
    },
    "magpie_metal": {
        "title": "Magpie TTS 357M v2607 Q8_0 (NeMo-Speech.cpp, Metal)",
        "licence": "NVIDIA Open Model License; runtime Apache-2.0",
        "langs": {"en", "hi"},
        "device": "metal",
        "variants": [
            {
                "column": "magpie_metal_sofia",
                "title": "Magpie Metal · Sofia",
                "opts": {"voice": "Sofia"},
            },
        ],
    },
    "supertonic": {
        "title": "Supertonic 3 (ONNX, CPU)",
        "script": "supertonic_worker.py",
        "licence": "OpenRAIL (model), MIT (package)",
        "langs": {"en", "hi"},
        "variants": [
            {"column": "supertonic_f1", "title": "Supertonic · F1", "opts": {"voice": "F1"}},
            {"column": "supertonic_f2", "title": "Supertonic · F2", "opts": {"voice": "F2"}},
        ],
    },
    "mithi": {
        "title": "mithi v1 flash",
        "script": "mithi_worker.py",
        "licence": "Apache-2.0",
        "langs": {"en", "hi", "bn"},
        "variants": [{"column": "mithi_charvi", "title": "mithi · Charvi", "opts": {}}],
    },
}


class Worker:
    """A worker script in its own uv environment (see worker_common.py)."""

    def __init__(self, script: str, *args: str) -> None:
        env = {**os.environ, "HF_HUB_DISABLE_XET": "1", "PYTHONWARNINGS": "ignore"}
        self.log = open(os.devnull, "w")  # noqa: SIM115
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            ["uv", "run", "--quiet", "--script", str(HERE / script), *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            env=env,
        )
        self.ready = self._reply()
        self.ready["start_s"] = time.perf_counter() - t0

    def _reply(self) -> dict[str, Any]:
        assert self.proc.stdout
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"worker exited (code {self.proc.poll()})")
        return json.loads(line)

    def ask(self, request: dict[str, Any]) -> dict[str, Any]:
        assert self.proc.stdin
        self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        return self._reply()

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


class Phonon:
    """A Lucy TTS adapter through the registry, timed like a worker: Ollama `phonon`
    (`ollama`), or the `magpie` adapter (`magpie_adapter` engine)."""

    def __init__(self, adapter: str = "ollama", **options: Any) -> None:
        from lucy.core import registry

        registry.load_adapters(["lucy.adapters.tts_openai_compat", "lucy.adapters.tts_magpie"])
        self.adapter = adapter
        options = options or {"model": "phonon", "voice": "Maeve"}
        self.tts = registry.create("tts", adapter, **options)
        self.loop = asyncio.new_event_loop()
        t0 = time.perf_counter()
        self.loop.run_until_complete(self.tts.warm_up())
        self.ready = {
            "load_s": time.perf_counter() - t0,
            "device": "Ollama" if adapter == "ollama" else "worker (uv)",
            "versions": {"ollama": _ollama_version()} if adapter == "ollama" else {},
            "revision": "phonon:latest" if adapter == "ollama" else self.tts.revision[:12],
        }

    async def _synth(self, text: str, out: str) -> dict[str, Any]:
        import numpy as np
        from worker_common import write_wav

        t0 = time.perf_counter()
        first, data = None, bytearray()
        async for chunk in self.tts.synthesize(text):
            if first is None:
                first = time.perf_counter() - t0
            data += chunk.data
        total = time.perf_counter() - t0
        write_wav(out, np.frombuffer(bytes(data), "<i2").astype(np.float32) / 32768, 24000)
        return {"first_s": first or total, "total_s": total, "duration_s": len(data) / 2 / 24000}

    def ask(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("stats"):
            return {"ollama_ps": _ollama_ps("phonon")} if self.adapter == "ollama" else {}
        return self.loop.run_until_complete(self._synth(request["text"], request["out"]))

    def close(self) -> None:
        self.loop.run_until_complete(self.tts.aclose())
        self.loop.close()


class NemoServe:
    """Magpie TTS under `nemo-speech serve` (NeMo-Speech.cpp, installed in data/tools/
    nemo-speech/{cpu,metal}), as a child process, over its OpenAI-compatible
    /v1/audio/speech route (raw PCM16 at the codec's 22.05 kHz). That route buffers
    the whole clip, so the first audio is the whole clip. Sparrowhawk text
    normalisation is on (the release's tn_configs, which have `hi` and `en`)."""

    def __init__(self, device: str = "cpu", port: int = 8765, threads: int = 4) -> None:
        root = NEMO / device
        models = NEMO / "models"
        self.url = f"http://127.0.0.1:{port}"
        self.log = open(NEMO / f"serve-{device}.log", "w")  # noqa: SIM115
        cmd = [str(root / "bin" / "nemo-speech"), "serve", "--host", "127.0.0.1"]
        cmd += ["--port", str(port), "--device", device, "--tts.threads", str(threads)]
        cmd += ["--tts.magpie-model", str(models / MAGPIE_GGUF)]
        cmd += ["--tts.codec-model", str(models / CODEC_GGUF)]
        cmd += ["--tts.tokenizer-model-dir", str(models / MAGPIE_TOKENIZER)]
        cmd += ["--tts.tn-model-dir", str(NEMO / "tn_configs")]
        env = {**os.environ, "NEMO_SPEECH_MODEL_DIR": str(models)}
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"nemo-speech serve exited ({self.proc.returncode})")
            try:
                if httpx.get(f"{self.url}/ready", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.perf_counter() - t0 > 120:
                raise RuntimeError("nemo-speech serve not ready after 120 s")
            time.sleep(0.2)
        info = httpx.get(f"{self.url}/v1/models", timeout=10).json()["data"][0]
        version = subprocess.run(
            [str(root / "bin" / "nemo-speech"), "--version"], capture_output=True, text=True
        ).stdout.strip()
        self.ready = {
            "load_s": time.perf_counter() - t0,
            "device": f"{device}, {threads} threads",
            "voices": info.get("voices"),
            "versions": {"nemo-speech": version},
            "revision": "magpie 446bff55 (q8_0), codec fc00890b",
            "download_bytes": sum((models / f).stat().st_size for f in (MAGPIE_GGUF, CODEC_GGUF)),
        }

    def ask(self, request: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        from worker_common import write_wav

        if request.get("stats"):
            return proc_stats(self.proc.pid)
        lang = {"en": "en-US", "hi": "hi-IN"}[request["lang"]]
        body = {"input": request["text"], "language": lang, "response_format": "pcm"}
        body["voice"] = request["opts"].get("voice", "Sofia")
        t0 = time.perf_counter()
        r = httpx.post(f"{self.url}/v1/audio/speech", json=body, timeout=300)
        total = time.perf_counter() - t0
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        audio = np.frombuffer(r.content, "<i2").astype(np.float32) / 32768
        write_wav(request["out"], audio, 22050)
        return {
            "first_s": total,
            "total_s": total,
            "duration_s": audio.size / 22050,
            "sample_rate": 22050,
            "peak": float(np.abs(audio).max()) if audio.size else 0.0,
        }

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def proc_stats(pid: int) -> dict[str, Any]:
    """Resident size and thread count of a process and its descendants (`ps`)."""
    pids, todo = [], [pid]
    while todo:
        p = todo.pop()
        pids.append(p)
        kids = subprocess.run(["pgrep", "-P", str(p)], capture_output=True, text=True).stdout
        todo += [int(k) for k in kids.split()]
    rss = threads = 0
    for p in pids:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(p)], capture_output=True, text=True)
        rss += int(out.stdout.strip() or 0)
        lines = subprocess.run(["ps", "-M", "-p", str(p)], capture_output=True, text=True)
        threads += max(0, len(lines.stdout.splitlines()) - 1)
    return {"rss_mb": rss / 1024, "threads": threads}


class Generation:
    """A long streamed generation by the chat model in a thread; token arrival times."""

    def __init__(self, model: str) -> None:
        import threading

        self.stamps: list[float] = []
        self.stop = threading.Event()
        self.model = model
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        body = {"model": self.model, "prompt": GEN_PROMPT, "stream": True, "keep_alive": "10m"}
        body["options"] = {"num_predict": 4000}
        with httpx.stream("POST", f"{OLLAMA}/api/generate", json=body, timeout=600) as r:
            for line in r.iter_lines():
                if line and json.loads(line).get("response"):
                    self.stamps.append(time.perf_counter())
                if self.stop.is_set():
                    break

    def rate(self, t0: float, t1: float) -> float | None:
        n = sum(1 for t in self.stamps if t0 <= t < t1)
        return n / (t1 - t0) if t1 > t0 else None

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=30)


def generating(engine: Any, col: dict[str, Any], sentences: list, run: Path, args: Any):
    """GEN_IDS synthesised back to back while the chat model writes a long answer: rows
    (phase "generating"), the playback simulation, and the chat model's tokens/s in
    the windows before, during and after."""
    gen = Generation(args.chat_model)
    while not gen.stamps:
        time.sleep(0.1)
    time.sleep(1.0)
    b0 = time.perf_counter()
    time.sleep(5.0)
    start = time.perf_counter()
    rows = synth_rows(engine, [col], sentences, run, "generating")
    end = time.perf_counter()
    for r in rows:
        if "error" not in r:
            r["ready_s"] = r.pop("_ready") - start
    time.sleep(5.0)
    a1 = time.perf_counter()
    gen.close()
    return rows, {
        **simulate_playback(rows),
        "chat_tps_before": gen.rate(b0, start),
        "chat_tps_during": gen.rate(start, end),
        "chat_tps_after": gen.rate(end, a1),
        "window_s": end - start,
    }


def _ollama_version() -> str:
    try:
        return httpx.get(f"{OLLAMA}/api/version", timeout=5).json().get("version", "?")
    except httpx.HTTPError:
        return "?"


def _ollama_ps(prefix: str = "") -> list[dict[str, Any]]:
    models = httpx.get(f"{OLLAMA}/api/ps", timeout=10).json().get("models", [])
    return [
        {
            "name": m["name"],
            "size_mb": m.get("size", 0) / 2**20,
            "vram_mb": m.get("size_vram", 0) / 2**20,
        }
        for m in models
        if m["name"].startswith(prefix)
    ]


def chat_model(model: str, keep_alive: str | int) -> float:
    """Load (or, with keep_alive 0, unload) the chat model with a one-token request."""
    t0 = time.perf_counter()
    body = {"model": model, "prompt": "Hi", "stream": False, "keep_alive": keep_alive}
    body["options"] = {"num_predict": 1}
    if keep_alive == 0:
        body = {"model": model, "keep_alive": 0}
    httpx.post(f"{OLLAMA}/api/generate", json=body, timeout=300).raise_for_status()
    return time.perf_counter() - t0


def synth_rows(
    engine: Any,
    columns: list[dict[str, Any]],
    sentences: list[dict[str, Any]],
    run: Path,
    phase: str,
) -> list[dict[str, Any]]:
    rows = []
    for col in columns:
        for n, s in enumerate(sentences, 1):
            stem = f"{n:02d}" if phase == "idle" else f"loaded_{n:02d}"
            wav = Path(col["column"]) / f"{stem}.wav"
            req = {"text": s["text"], "lang": s["lang"], "out": str(run / wav), "opts": col["opts"]}
            row: dict[str, Any] = {"column": col["column"], "id": s["id"], "phase": phase}
            try:
                reply = engine.ask(req)
            except Exception as e:  # noqa: BLE001
                reply = {"error": f"{type(e).__name__}: {e}"}
            if "error" in reply:
                row["error"] = reply["error"]
            else:
                row["_ready"] = time.perf_counter()
                row.update(
                    first_s=reply["first_s"],
                    total_s=reply["total_s"],
                    duration_s=reply["duration_s"],
                    rtf=rtf(reply["total_s"], reply["duration_s"]),
                    peak=reply.get("peak"),
                    wav=str(wav),
                )
            if phase != "generating":
                row.pop("_ready", None)
            rows.append(row)
            shown = row.get("error") or f"first {row['first_s']:.2f}s total {row['total_s']:.2f}s"
            print(f"  {phase:6} {col['column']:16} {s['id']:15} {shown}", flush=True)
    return rows


def run_engine(
    name: str, spec: dict[str, Any], sentences: list[dict[str, Any]], run: Path, args: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mine = [s for s in sentences if s["lang"] in spec["langs"]]
    columns = spec["variants"]
    print(f"{name}: starting", flush=True)
    facts: dict[str, Any] = {"engine": name, "title": spec["title"], "licence": spec["licence"]}
    engine: Any = None
    try:
        if name == "phonon":
            engine = Phonon()
        elif name == "magpie_adapter":
            engine = Phonon("magpie", voice="Sofia")
        elif name.startswith("magpie"):
            engine = NemoServe(spec["device"])
        else:
            extra = ["--ref-dir", str(run / "reference")] if name == "dhvaani" else []
            device = ["--device", args.device] if name in ("dhvaani", "mithi") else []
            try:
                engine = Worker(spec["script"], *device, *extra)
            except RuntimeError:
                if not device:
                    raise
                engine = Worker(spec["script"], "--device", "cpu", *extra)
        facts["ready"] = engine.ready
        lang = next(iter(sorted(spec["langs"] - {"bn"}, reverse=True)))
        t0 = time.perf_counter()
        warm = engine.ask(
            {
                "text": WARM_UP[lang],
                "lang": lang,
                "out": str(run / "warmup" / f"{name}.wav"),
                "opts": columns[0]["opts"],
            }
        )
        facts["warm_up_s"] = time.perf_counter() - t0
        facts["warm_up"] = warm
        rows = synth_rows(engine, columns, mine, run, "idle")
        facts["stats_idle"] = engine.ask({"stats": True})
        if isinstance(engine, Worker):
            facts["stats_idle"].update(proc_stats(engine.proc.pid))
        if not args.no_loaded:
            loaded = [s for s in mine if s["id"] in LOADED_IDS]
            facts["chat_load_s"] = chat_model(args.chat_model, "10m")
            facts["ollama_ps_loaded"] = _ollama_ps()
            rows += synth_rows(engine, columns[:1], loaded, run, "loaded")
            facts["stats_loaded"] = engine.ask({"stats": True})
            if args.generating:
                gen_sentences = [s for s in mine if s["id"] in GEN_IDS]
                gen_rows, facts["generating"] = generating(
                    engine, columns[0], gen_sentences, run, args
                )
                rows += gen_rows
                print(f"  generating: {facts['generating']}", flush=True)
            chat_model(args.chat_model, 0)
        return facts, rows
    except Exception as e:  # noqa: BLE001
        facts["error"] = f"{type(e).__name__}: {e}"
        print(f"{name}: failed: {facts['error']}", flush=True)
        return facts, []
    finally:
        if engine is not None:
            engine.close()


def score(row: dict[str, Any], sentence: dict[str, Any]) -> None:
    """Word (and, for Hindi, character) error of row["asr"] against the sentence, and
    the gross-failure flags. Hindi is compared romanised and folded on both sides."""
    from stt_scoring import edits, normalise, roman_score, text_lattice
    from translit import fold

    text, heard = sentence["text"], row["asr"]
    if sentence["lang"] == "en":
        e, n = edits(text, heard)
        ref, hyp = normalise(text), normalise(heard)
        row["wer"], row["cer"] = e / max(1, n), None
    else:
        sc = roman_score(text_lattice(text), heard)
        ref, hyp = fold(text).split(), fold(heard).split()
        row["wer"] = sc.word_edits / max(1, sc.ref_words)
        row["cer"] = sc.char_edits / max(1, sc.ref_chars)
    row["flags"] = flags(ref, hyp, row["wer"], row["cer"])


def machine_check(
    rows: list[dict[str, Any]], sentences: dict[str, dict[str, Any]], run: Path
) -> None:
    """Transcribe idle clips with Lucy's recognisers and flag gross failures in place."""
    import numpy as np
    from recognisers import _flex_adapter, _parakeet

    from lucy.voice.audio import float_to_pcm, read_wav, resample

    todo = [r for r in rows if r.get("phase") == "idle" and "wav" in r]
    for lang, factory in (("en", _parakeet), ("hi", _flex_adapter)):
        mine = [r for r in todo if sentences[r["id"]]["lang"] == lang]
        if not mine:
            continue
        rec = factory()
        rec.load()
        try:
            for r in mine:
                audio, rate = read_wav(run / r["wav"])
                pcm = float_to_pcm(resample(np.asarray(audio), rate, 16000))
                heard = rec.transcribe(pcm)
                r["asr"] = heard
                score(r, sentences[r["id"]])
                print(
                    f"  check {r['column']:16} {r['id']:15} wer {r['wer']:.2f} {r['flags']} | {heard}"
                )
        finally:
            rec.close()


def column_facts(facts: dict[str, Any], col: dict[str, Any]) -> dict[str, str]:
    ready = facts.get("ready", {})
    stats = facts.get("stats_idle", {})
    mem = [f"RSS {stats['max_rss_mb']:.0f} MB"] if "max_rss_mb" in stats else []
    if "rss_mb" in stats:
        mem.append(f"process tree RSS {stats['rss_mb']:.0f} MB, {stats['threads']} threads")
    if "mlx_peak_mb" in stats:
        mem.append(f"MLX peak {stats['mlx_peak_mb']:.0f} MB")
    if "mps_driver_mb" in stats:
        mem.append(f"MPS {stats['mps_driver_mb']:.0f} MB")
    for m in stats.get("ollama_ps", []):
        mem.append(f"Ollama {m['size_mb']:.0f} MB")
    size = ready.get("download_bytes")
    return {
        "Licence": facts["licence"],
        "Device": str(ready.get("device", "?")),
        "Download": f"{size / 1e9:.2f} GB" if size else "(Ollama)",
        "Load": f"{ready['load_s']:.1f} s" if "load_s" in ready else "–",
        "Memory": ", ".join(mem) or "–",
        "Status": facts.get("error", "ran"),
        **_condition_facts(facts),
    }


def _condition_facts(facts: dict[str, Any]) -> dict[str, str]:
    out = {}
    if facts.get("loaded_summary"):
        m = facts["loaded_summary"]
        out["Chat model loaded, idle"] = (
            f"median total {m['total_s_idle']:.2f} s quiet -> {m['total_s_loaded']:.2f} s "
            f"loaded (n={m['n']})"
        )
    g = facts.get("generating")
    if g:
        tps = [_fmt_tps(g.get(k)) for k in ("chat_tps_before", "chat_tps_during", "chat_tps_after")]
        out["Chat model generating"] = (
            f"{g['underruns']} underruns, {g['underrun_s']:.2f} s waiting over "
            f"{len(GEN_IDS)} sentences; median RTF {_fmt_tps(g.get('rtf_median'))}; chat model "
            f"tokens/s before / during / after: {' / '.join(tps)}"
        )
    return out


def _fmt_tps(v: float | None) -> str:
    return "–" if v is None else f"{v:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="phonon,maya,dhvaani,mithi")
    ap.add_argument("--maya-voices", default="Vaidehi,Ahana")
    ap.add_argument("--sentences", default="", help="comma-separated ids (default all)")
    ap.add_argument("--device", default="mps", help="for the PyTorch engines; cpu on failure")
    ap.add_argument("--chat-model", default="gemma4:12b-mlx")
    ap.add_argument("--no-loaded", action="store_true")
    ap.add_argument(
        "--generating", action="store_true", help="also time GEN_IDS during a generation"
    )
    ap.add_argument("--set", default="", help="only sentences in this set (sentences.json `sets`)")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--page-only", action="store_true", help="rescore and rewrite the page")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sentences = json.loads((HERE / "sentences.json").read_text())
    if args.set:
        sentences = [s for s in sentences if args.set in s.get("sets", [])]
    if args.sentences:
        keep = args.sentences.split(",")
        sentences = [s for s in sentences if s["id"] in keep]
    run = out_dir("tts_listen", args.out) / args.label
    run.mkdir(parents=True, exist_ok=True)
    ENGINES["maya"]["variants"] = [
        {"column": f"maya2_{v.lower()}", "title": f"maya2 · {v}", "opts": {"voice": v}}
        for v in args.maya_voices.split(",")
    ]

    if args.page_only:
        saved = json.loads((run / "results.json").read_text())
        by_id = {s["id"]: s for s in sentences}
        for r in saved["rows"]:
            if r.get("asr") is not None:
                score(r, by_id[r["id"]])
        for name in saved["facts"]:
            if name == "maya":
                voices = [c["column"] for c in saved["rows"] if c["column"].startswith("maya2_")]
                ENGINES["maya"]["variants"] = [
                    {"column": c, "title": f"maya2 · {c[6:].title()}", "opts": {}}
                    for c in dict.fromkeys(voices)
                ]
        _write(run, sentences, saved["facts"], saved["rows"])
        print(f"page: {run / 'index.html'}")
        return 0
    all_facts: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for name in args.engines.split(","):
        facts, mine = run_engine(name, ENGINES[name], sentences, run, args)
        all_facts[name] = facts
        rows += mine
        _write(run, sentences, all_facts, rows)

    if not args.no_check:
        try:
            machine_check(rows, {s["id"]: s for s in sentences}, run)
        except Exception as e:  # noqa: BLE001
            all_facts["check_error"] = f"{type(e).__name__}: {e}"
            print(f"machine check failed: {all_facts['check_error']}")
    _write(run, sentences, all_facts, rows)
    print(json.dumps({"idle": summarise(rows), "loaded": compare_loaded(rows)}, indent=1))
    print(f"page: {run / 'index.html'}")
    return 0


def _write(run: Path, sentences: list[dict[str, Any]], facts: dict[str, Any], rows: list) -> None:
    (run / "results.json").write_text(
        json.dumps({"facts": facts, "rows": rows}, ensure_ascii=False, indent=1)
    )
    loaded, gen = compare_loaded(rows), summarise(rows, "generating")
    for name, f in facts.items():
        if name in ENGINES and isinstance(f, dict):
            col0 = ENGINES[name]["variants"][0]["column"]
            if col0 in loaded:
                f["loaded_summary"] = loaded[col0]
            if "generating" in f and col0 in gen:
                f["generating"]["rtf_median"] = gen[col0]["rtf_median"]
    columns = []
    for name, f in facts.items():
        if name not in ENGINES:
            continue
        for col in ENGINES[name]["variants"]:
            columns.append({**col, "facts": column_facts(f, col)})
    notes = [
        "Each cell: time to first audio (first streamed chunk for phonon and maya2, the "
        "whole clip otherwise), total synthesis time, audio length, real-time factor.",
        "maya2 samples with temperature 0.9, so a second run differs. DhVaani clones the "
        "demo Space's Hindi female reference for every language.",
        "Magpie goes through `nemo-speech serve` /v1/audio/speech, which returns the whole "
        "clip (no streaming), 22.05 kHz, Sparrowhawk text normalisation on; Supertonic 3 "
        "returns the whole clip at 44.1 kHz with no number normaliser. 'Chat model "
        "generating': the first voice of the engine synthesises 8 sentences back to back "
        "during a long Ollama generation; an underrun is a wait between sentences when "
        "they are played back to back as soon as each is ready.",
        "'heard' is what Lucy's own recogniser made of the clip (Parakeet for English, "
        "indic_flex in Latin script for Hindi and mixed); Bengali is not checked.",
    ]
    (run / "index.html").write_text(
        render_page("TTS listening test", sentences, columns, rows, notes), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
