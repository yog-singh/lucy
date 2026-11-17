"""Shared serving loop for the tts_listen workers (imported by the worker scripts).

Pure standard library plus numpy, so every worker environment can import it (a
worker is run as a script, so its own folder is on sys.path). Protocol, one JSON
line each way:

  ready (sent once): {"ready": true, "load_s", "device", ...extra}
  {"text", "lang", "out", "opts": {...}} -> {"first_s", "total_s", "duration_s",
                                             "sample_rate", "chunks", "peak"}
  {"stats": true}                       -> {"max_rss_mb", ...extra}

`first_s` is the time from the request to the first audio the model hands back:
the first streamed chunk for a streaming model, the whole clip otherwise. The
audio is written to "out" as 16-bit mono WAV. Anything a library prints goes to
stderr; errors are reported as {"error": "..."} and the worker keeps serving.
"""

from __future__ import annotations

import contextlib
import json
import resource
import sys
import time
import wave
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

OUT = sys.stdout

Synth = Callable[[str, str, dict[str, Any]], Iterator[np.ndarray]]


def send(obj: dict[str, Any]) -> None:
    OUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    OUT.flush()


def max_rss_mb() -> float:
    # macOS reports ru_maxrss in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def write_wav(path: str | Path, audio: np.ndarray, rate: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def dir_bytes(*paths: str | Path) -> int:
    """Total size of the files under `paths` (symlinks followed: HF cache snapshots)."""
    total = 0
    for p in paths:
        p = Path(p)
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
        total += sum(f.stat().st_size for f in files)
    return total


def serve(
    synth: Synth,
    rate: int,
    ready: dict[str, Any],
    stats: Callable[[], dict[str, Any]] = dict,
    sync: Callable[[], None] = lambda: None,
) -> int:
    """Answer requests on stdin until it closes. `synth` yields float32 chunks."""
    send({"ready": True, **ready, "load_max_rss_mb": max_rss_mb()})
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if req.get("stats"):
                send({"max_rss_mb": max_rss_mb(), **stats()})
                continue
            chunks: list[np.ndarray] = []
            t0 = time.perf_counter()
            first = None
            with contextlib.redirect_stdout(sys.stderr):
                for chunk in synth(req["text"], req["lang"], req.get("opts") or {}):
                    chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
                    if first is None and chunk.size:
                        sync()
                        first = time.perf_counter() - t0
                    chunks.append(chunk)
                sync()
            total = time.perf_counter() - t0
            audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
            write_wav(req["out"], audio, rate)
            send(
                {
                    "first_s": first if first is not None else total,
                    "total_s": total,
                    "duration_s": audio.size / rate,
                    "sample_rate": rate,
                    "chunks": len(chunks),
                    "peak": float(np.abs(audio).max()) if audio.size else 0.0,
                }
            )
        except Exception as e:  # report and keep serving
            send({"error": f"{type(e).__name__}: {e}"})
    return 0
