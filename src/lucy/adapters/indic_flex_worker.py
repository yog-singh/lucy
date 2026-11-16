# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.11.0",
#   "torchaudio==2.11.0",
#   "transformers==5.5.3",
#   "sentencepiece==0.2.1",
#   "safetensors==0.7.0",
#   "numpy==2.3.5",
#   "huggingface-hub==1.5.0",
# ]
# [tool.uv]
# exclude-newer = "2026-11-17T00:00:00Z"
# ///
"""Worker for Lucy's `indic_flex` STT adapter: bodhan-ai/indic-transcribe-flex in its own env.

Started by `lucy.adapters.stt_indic_flex` as `uv run --script indic_flex_worker.py`, so
PyTorch and Transformers never enter Lucy's environment (ADR 0023). Not imported by
Lucy. It loads the model once (the model's own code and weights from the Hugging Face
cache at one reviewed commit; the `nemo/` folder is never fetched) and then serves
requests, one at a time, as JSON lines:

  stdout, once:   {"ready": true, "load_s", "device", "dtype", "revision"}
                  or {"fatal": "..."} and exit 2 if the model cannot be loaded
  stdin:          {"id": N, "pcm": "<base64 16 kHz mono s16le>"}
  stdout:         {"id": N, "text", "lang", "tokens", "mean_logprob", "max_new_tokens",
                   "seconds"}  or  {"id": N, "error": "..."}

Generation is capped at min(512, tokens_base + tokens_per_s x seconds of audio), so a
decoding loop cannot run long. Everything the libraries print goes to stderr: the
protocol is written to a private copy of the original stdout.
"""

import argparse
import base64
import json
import os
import sys
import time
import warnings

SKIP = ["nemo/*", "*.png"]
MAX_TOKENS = 512


def _protocol_out():
    """A private copy of stdout for the protocol; fd 1 then points at stderr."""
    out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return out


def _load_path(repo: str, revision: str) -> str:
    """The snapshot directory, from the cache or downloaded; raises with a clear message."""
    from huggingface_hub import snapshot_download

    page = f"https://huggingface.co/{repo}"
    try:
        return snapshot_download(
            repo, revision=revision, ignore_patterns=SKIP, local_files_only=True
        )
    except Exception:
        pass
    try:
        return snapshot_download(repo, revision=revision, ignore_patterns=SKIP)
    except Exception as exc:
        name = type(exc).__name__
        if "Gated" in name or "401" in str(exc) or "403" in str(exc):
            why = "access is refused (the model is gated)"
        else:
            why = f"it is not in the Hugging Face cache and could not be downloaded ({name})"
        raise RuntimeError(
            f"cannot load {repo} at revision {revision[:8]}: {why}. Accept the licence "
            f"(Indic Open Model License) at {page} with your Hugging Face account, then "
            "log in with `hf auth login` or set HF_TOKEN, and start Lucy once with network "
            "access to download the weights (4.9 GB)."
        ) from None


def main() -> int:
    out = _protocol_out()

    def send(obj: dict) -> None:
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        out.flush()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--lang", default="auto", help='a language code, or "auto" (identification)')
    ap.add_argument("--allowed", default="", help="with --lang auto: comma-separated codes")
    ap.add_argument("--mode", default="mixed")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16", choices=["float32", "float16"])
    ap.add_argument("--tokens-base", type=float, default=10.0)
    ap.add_argument("--tokens-per-s", type=float, default=25.0)
    args = ap.parse_args()

    started = time.perf_counter()
    try:
        path = _load_path(args.model, args.revision)
        import numpy as np
        import torch

        sys.path.insert(0, path)  # the model code ships inside the download
        from indic_transcribe import IndicTranscribe

        asr = IndicTranscribe.from_pretrained(
            path, device=args.device, dtype=getattr(torch, args.dtype)
        )
    except Exception as exc:
        send(
            {"fatal": str(exc) if isinstance(exc, RuntimeError) else f"{type(exc).__name__}: {exc}"}
        )
        return 2

    generate = asr.model.generate
    last: dict = {}

    def scored_generate(*a, **kw):  # the same decoding, plus the score of each step
        result = generate(*a, **kw, output_scores=True, return_dict_in_generate=True)
        steps = asr.model.compute_transition_scores(
            result.sequences, result.scores, normalize_logits=True
        )[0]
        lp = steps.float().cpu().tolist()
        last.update(tokens=len(lp), logprobs=lp)
        return result.sequences

    asr.model.generate = scored_generate
    lang = None if args.lang == "auto" else args.lang
    extra = {"allowed_langs": args.allowed.split(",")} if args.allowed and lang is None else {}
    send(
        {
            "ready": True,
            "load_s": round(time.perf_counter() - started, 3),
            "device": args.device,
            "dtype": args.dtype,
            "revision": args.revision,
        }
    )

    for line in sys.stdin:
        if not line.strip():
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            pcm = np.frombuffer(base64.b64decode(req["pcm"]), dtype="<i2")
            wav = torch.from_numpy(pcm.astype(np.float32) / 32768.0)
            cap = max(
                1, min(MAX_TOKENS, int(args.tokens_base + args.tokens_per_s * len(pcm) / 16000))
            )
            last.clear()
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = asr.transcribe(
                    wav,
                    lang=lang,
                    mode=args.mode,
                    return_lid=lang is None,
                    max_new_tokens=cap,
                    **extra,
                )
            if args.device == "mps":
                torch.mps.synchronize()
            text, used = (result[0], result[1]["lang"]) if lang is None else (result, lang)
            lp = last.get("logprobs") or [0.0]
            send(
                {
                    "id": rid,
                    "text": text,
                    "lang": used,
                    "tokens": last.get("tokens", 0),
                    "mean_logprob": sum(lp) / len(lp),
                    "max_new_tokens": cap,
                    "seconds": round(time.perf_counter() - t0, 4),
                }
            )
        except Exception as exc:  # report and keep serving
            send({"id": rid, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
