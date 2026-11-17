"""Worker for Lucy's `magpie` TTS adapter: NVIDIA Magpie TTS through NeMo-Speech.cpp's C API.

Started by `lucy.adapters.tts_magpie` as `python -I magpie_worker.py ...` (standard
library only: it loads `libnemo_speech_tts` with ctypes), so a crash in the native
runtime cannot take Lucy down (ADR 0024). Not imported by Lucy. The HTTP route of
`nemo-speech serve` buffers the whole clip; the C API's `nemo_speech_tts_synthesize_text`
hands PCM to a callback as it is decoded (a few codec frames at a time), which is what
this worker streams. JSON lines:

  stdout, once:  {"ready": true, "load_s", "sample_rate", "voices": [...], "version"}
                 or {"fatal": "..."} and exit 2 if the runtime or a model cannot be loaded
  stdin:         {"id": N, "text", "voice", "lang", "seed"}
                 {"cancel": N}   abandon request N (queued or in progress)
  stdout:        {"id": N, "start": true}             when N begins
                 {"id": N, "audio": "<base64 s16le mono at sample_rate>"}   (several)
                 {"id": N, "done": true, "cancelled", "first_s", "total_s", "samples"}
                 or {"id": N, "error": "..."}

Requests are served one at a time in arrival order; a reader thread takes stdin, so a
cancel arrives during synthesis and the callback returns false at the next chunk.
Anything the runtime prints goes to stderr: the protocol has a private copy of stdout.
"""

import argparse
import base64
import ctypes as C
import json
import os
import queue
import sys
import threading
import time


class ModelConfig(C.Structure):
    _fields_ = [
        ("size", C.c_size_t),
        ("magpie_model", C.c_char_p),
        ("codec_model", C.c_char_p),
        ("tokenizer_model_dir", C.c_char_p),
        ("text_normalizer_model_dir", C.c_char_p),
    ]


class RuntimeConfig(C.Structure):
    _fields_ = [
        ("size", C.c_size_t),
        ("speaker", C.c_int32),
        ("threads", C.c_int32),
        ("codec_threads", C.c_int32),
        ("seed", C.c_int32),
        ("steps", C.c_int32),
        ("top_k", C.c_int32),
        ("chunk_frames", C.c_int32),
        ("codec_queue_depth", C.c_int32),
        ("codec_history_frames", C.c_int32),
        ("codec_future_frames", C.c_int32),
        ("window_ms", C.c_int32),
        ("temperature", C.c_float),
        ("override_temperature", C.c_bool),
        ("cfg_scale", C.c_float),
        ("override_cfg_scale", C.c_bool),
        ("use_cfg", C.c_bool),
        ("use_local_transformer", C.c_bool),
        ("use_kv_cache", C.c_bool),
        ("use_stateful_codec", C.c_bool),
        ("codec_cpu", C.c_bool),
        ("flush_partial_chunk", C.c_bool),
        ("verbose", C.c_bool),
        ("lt_backend", C.c_int),
        ("sampling_backend", C.c_int),
        ("uma_mode", C.c_int),
        ("longform_mode", C.c_int),
        ("lt_fp32", C.c_bool),
    ]


class SynthesizerConfig(C.Structure):
    _fields_ = [
        ("size", C.c_size_t),
        ("model", C.POINTER(ModelConfig)),
        ("runtime", C.POINTER(RuntimeConfig)),
        ("default_language_code", C.c_char_p),
        ("default_voice_name", C.c_char_p),
    ]


class SynthesisOptions(C.Structure):
    _fields_ = [
        ("size", C.c_size_t),
        ("request_id", C.c_char_p),
        ("language_code", C.c_char_p),
        ("speaker", C.c_int32),
        ("seed", C.c_int32),
        ("steps", C.c_int32),
        ("top_k", C.c_int32),
        ("temperature", C.c_float),
        ("override_temperature", C.c_bool),
        ("cfg_scale", C.c_float),
        ("override_cfg_scale", C.c_bool),
        ("voice_name", C.c_char_p),
        ("output_sample_rate", C.c_int32),
    ]


PCM_CALLBACK = C.CFUNCTYPE(C.c_bool, C.POINTER(C.c_uint8), C.c_size_t, C.c_void_p)


def _protocol_out():
    """A private copy of stdout for the protocol; fd 1 then points at stderr."""
    out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return out


def _load(lib_dir):
    lib = C.CDLL(os.path.join(lib_dir, "libnemo_speech_tts.dylib" if sys.platform == "darwin"
                              else "libnemo_speech_tts.so"))  # fmt: skip
    lib.nemo_speech_tts_runtime_config_default.restype = RuntimeConfig
    lib.nemo_speech_tts_synthesis_options_default.restype = SynthesisOptions
    lib.nemo_speech_tts_create.argtypes = [C.POINTER(SynthesizerConfig), C.POINTER(C.c_void_p)]
    lib.nemo_speech_tts_create.restype = C.c_int
    lib.nemo_speech_tts_sample_rate.argtypes = [C.c_void_p]
    lib.nemo_speech_tts_sample_rate.restype = C.c_int32
    lib.nemo_speech_tts_speaker_count.argtypes = [C.c_void_p]
    lib.nemo_speech_tts_speaker_count.restype = C.c_int32
    lib.nemo_speech_tts_speaker_name.argtypes = [C.c_void_p, C.c_size_t]
    lib.nemo_speech_tts_speaker_name.restype = C.c_char_p
    lib.nemo_speech_tts_synthesize_text.argtypes = [
        C.c_void_p, C.POINTER(SynthesisOptions), C.c_char_p, PCM_CALLBACK, C.c_void_p,
        C.c_void_p,
    ]  # fmt: skip
    lib.nemo_speech_tts_synthesize_text.restype = C.c_int
    lib.nemo_speech_tts_last_error.restype = C.c_char_p
    lib.nemo_speech_tts_version.restype = C.c_char_p
    return lib


def _error(lib):
    return (lib.nemo_speech_tts_last_error() or b"unknown error").decode("utf-8", "replace")


def main() -> int:
    out = _protocol_out()
    lock = threading.Lock()

    def send(obj):
        with lock:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out.flush()

    ap = argparse.ArgumentParser()
    ap.add_argument("--lib-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--codec", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tn", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--voice", default="Sofia")
    ap.add_argument("--seed", type=int, default=-1)
    args = ap.parse_args()

    started = time.perf_counter()
    try:
        lib = _load(args.lib_dir)
        model = ModelConfig(C.sizeof(ModelConfig), args.model.encode(), args.codec.encode(),
                            args.tokenizer.encode(), args.tn.encode())  # fmt: skip
        runtime = lib.nemo_speech_tts_runtime_config_default()
        runtime.threads = args.threads
        runtime.seed = args.seed
        cfg = SynthesizerConfig(C.sizeof(SynthesizerConfig), C.pointer(model), C.pointer(runtime),
                                b"en-US", args.voice.encode())  # fmt: skip
        handle = C.c_void_p()
        if lib.nemo_speech_tts_create(C.byref(cfg), C.byref(handle)) != 0:
            raise RuntimeError(_error(lib))
        rate = lib.nemo_speech_tts_sample_rate(handle)
        voices = [
            lib.nemo_speech_tts_speaker_name(handle, i).decode("utf-8", "replace")
            for i in range(lib.nemo_speech_tts_speaker_count(handle))
        ]
    except Exception as exc:
        send({"fatal": f"{type(exc).__name__}: {exc}"})
        return 2

    send(
        {
            "ready": True,
            "load_s": round(time.perf_counter() - started, 3),
            "sample_rate": rate,
            "voices": voices,
            "version": (lib.nemo_speech_tts_version() or b"").decode(),
        }
    )

    requests: queue.Queue = queue.Queue()
    cancelled: set = set()

    def read():
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "cancel" in msg:
                cancelled.add(msg["cancel"])
            else:
                requests.put(msg)
        requests.put(None)

    threading.Thread(target=read, daemon=True).start()

    while (req := requests.get()) is not None:
        rid = req.get("id")
        t0 = time.perf_counter()
        state = {"first": None, "samples": 0, "stopped": rid in cancelled}

        def on_pcm(pcm, n, _user, rid=rid, t0=t0, state=state):
            if rid in cancelled:
                state["stopped"] = True
                return False
            if n:
                data = C.string_at(pcm, n)
                if state["first"] is None:
                    state["first"] = round(time.perf_counter() - t0, 4)
                state["samples"] += n // 2
                send({"id": rid, "audio": base64.b64encode(data).decode("ascii")})
            return True

        try:
            if not state["stopped"]:
                send({"id": rid, "start": True})
                opts = lib.nemo_speech_tts_synthesis_options_default()
                opts.language_code = (req.get("lang") or "en-US").encode()
                opts.voice_name = (req.get("voice") or args.voice).encode()
                if req.get("seed") is not None:
                    opts.seed = int(req["seed"])
                callback = PCM_CALLBACK(on_pcm)
                status = lib.nemo_speech_tts_synthesize_text(
                    handle, C.byref(opts), str(req["text"]).encode("utf-8"), callback, None, None
                )
                if status not in (0, 4):  # OK, CANCELLED
                    raise RuntimeError(_error(lib))
            send(
                {
                    "id": rid,
                    "done": True,
                    "cancelled": state["stopped"],
                    "first_s": state["first"],
                    "total_s": round(time.perf_counter() - t0, 4),
                    "samples": state["samples"],
                }
            )
        except Exception as exc:  # report and keep serving
            send({"id": rid, "error": f"{type(exc).__name__}: {exc}"})
        cancelled.discard(rid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
