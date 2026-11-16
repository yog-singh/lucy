# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "mlx==0.32.4",
#   "mlx-whisper==0.4.3",
#   "mlx-qwen3-asr==0.4.4",
#   "numpy==2.5.4",
#   "numba==0.68.1",
#   "torch==2.11.0",
#   "huggingface-hub==1.5.0",
# ]
# [tool.uv]
# exclude-newer = "2026-11-16T00:00:00Z"
# ///
"""Worker for the `whisper_turbo*` and `qwen3_asr*` recognisers, in its own uv env.

Run by recognisers.WorkerRecogniser as `uv run --script mlx_asr_worker.py --model M`,
so mlx-whisper (which pulls in torch) and mlx-qwen3-asr never enter Lucy's environment.
It loads one model once, from the Hugging Face cache at a pinned revision (downloading
it the first time), prints one JSON line `{"ready": true, "load_s", "device", "dtype"}`,
then answers one JSON line per request line on stdin:

  {"pcm": "<base64 16 kHz mono s16le>"} -> {"text", "seconds", "lang"}
  {"stats": true}                       -> {"max_rss_mb", "mlx_peak_mb", "mlx_active_mb"}

--model whisper: mlx-community/whisper-large-v3-turbo (MLX conversion of OpenAI's
  whisper-large-v3-turbo) through mlx-whisper, language --lang, temperature 0.0 only
  (greedy, no temperature fallback), otherwise the library defaults; --prompt TEXT is
  passed as `initial_prompt`.
--model qwen3: Qwen/Qwen3-ASR-0.6B (the official weights, converted at load) through
  mlx-qwen3-asr, language --lang (as "English", "Hindi"), greedy (the runtime decodes at temperature 0.0);
  --prompt TEXT is passed as `context` (the system prompt the model is trained to use).
--lang CODE forces a language (default en); --lang auto lets the model identify it per
clip, and the reply's "lang" is the language it chose. Both run float16 on the GPU. Errors are reported as {"error": "..."}.
"""

import argparse
import base64
import contextlib
import json
import resource
import sys
import time
import warnings

MODELS = {
    # repo, reviewed revision, files fetched
    "whisper": (
        "mlx-community/whisper-large-v3-turbo",
        "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb",
        ["*.json", "*.safetensors"],
    ),
    "qwen3": (
        "Qwen/Qwen3-ASR-0.6B",
        "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
        ["*.json", "*.safetensors", "*.txt"],
    ),
}


QWEN_LANGUAGES = {"en": "English", "hi": "Hindi"}
"""Codes accepted by --lang for Qwen. Its config.json `support_languages` lists 30
languages: Hindi, but not Bengali, Telugu or Kannada."""

OUT = sys.stdout  # replies only; anything the libraries print goes to stderr


def _send(obj: dict) -> None:
    OUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    OUT.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--prompt", default="")
    ap.add_argument("--lang", default="en", help='a language code (en, hi, ...) or "auto"')
    args = ap.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        return serve(args)


def serve(args: argparse.Namespace) -> int:
    import mlx.core as mx
    import numpy as np
    from huggingface_hub import snapshot_download

    started = time.perf_counter()
    repo, revision, patterns = MODELS[args.model]
    try:
        path = snapshot_download(
            repo, revision=revision, allow_patterns=patterns, local_files_only=True
        )
    except Exception:
        path = snapshot_download(repo, revision=revision, allow_patterns=patterns)

    if args.model == "whisper":
        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder

        ModelHolder.get_model(path, mx.float16)
        options = {"initial_prompt": args.prompt} if args.prompt else {}
        language = None if args.lang == "auto" else args.lang

        def run(samples: np.ndarray) -> tuple[str, str]:
            out = mlx_whisper.transcribe(
                samples, path_or_hf_repo=path, language=language, temperature=0.0, **options
            )
            return out["text"].strip(), out.get("language") or "?"
    else:
        from mlx_qwen3_asr import load_model, transcribe

        model, _config = load_model(path, dtype=mx.float16)
        language = None if args.lang == "auto" else QWEN_LANGUAGES[args.lang]

        def run(samples: np.ndarray) -> tuple[str, str]:
            out = transcribe(samples, model=model, language=language, context=args.prompt)
            return out.text.strip(), out.language or "?"

    mx.eval(mx.zeros(1))
    _send(
        {
            "ready": True,
            "load_s": time.perf_counter() - started,
            "device": str(mx.default_device()),
            "dtype": "float16",
        }
    )

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if req.get("stats"):
                _send(
                    {
                        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                        "mlx_peak_mb": mx.get_peak_memory() / 2**20,
                        "mlx_active_mb": mx.get_active_memory() / 2**20,
                    }
                )
                continue
            pcm = np.frombuffer(base64.b64decode(req["pcm"]), dtype="<i2")
            samples = pcm.astype(np.float32) / 32768.0
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                text, lang = run(samples)
            _send({"text": text, "seconds": time.perf_counter() - t0, "lang": lang})
        except Exception as e:  # report and keep serving
            _send({"error": f"{type(e).__name__}: {e}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
