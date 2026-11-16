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
# ///
"""Worker for the `indic_flex` recognisers: bodhan-ai/indic-transcribe-flex in its own env.

Run by recognisers.WorkerRecogniser as `uv run --script flex_worker.py [options]`, so
torch and transformers never enter Lucy's environment. It loads the model once
(Transformers path, the reviewed revision only; the `nemo/` folder is never fetched),
prints one JSON line `{"ready": true, "load_s", "device", "dtype"}`, then answers one
JSON line per request line on stdin:

  {"pcm": "<base64 16 kHz mono s16le>"} -> {"text", "seconds", "lang", "tokens",
                                            "mean_logprob", "min_logprob",
                                            "nospeech_logprob", "max_new_tokens"}
  {"stats": true}                       -> {"max_rss_mb", "mps_allocated_mb",
                                            "mps_driver_mb"}

Options: --lang CODE or "auto" (identification picks it), --mode native|mixed|romanised,
--device mps|cpu, --dtype float32|float16, --allowed en,hi (with --lang auto:
identification chooses only among these, the model's `allowed_langs`),
--tokens-base B --tokens-per-s R (cap generated tokens at min(512, B + R x seconds);
default 512 for every clip, the wrapper's own cap). Scores: the model's generate() is
wrapped so it also returns per-step scores; `tokens` is the number generated,
`mean_logprob`/`min_logprob` the log-probabilities of the chosen tokens, and
`nospeech_logprob` that of the vocabulary's <|nospeech|> token at the first step.
The ready line also gives the resident peak and MPS memory right after loading.
Errors are reported as {"error": "..."}.
"""

import argparse
import base64
import json
import resource
import sys
import time
import warnings

REPO_ID = "bodhan-ai/indic-transcribe-flex"
REVISION = "ecba9bac9013c0340fe814b336897ee51ed80be9"  # the commit whose code was reviewed
SKIP = ["nemo/*", "*.png"]


def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en")
    ap.add_argument("--mode", default="mixed")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--allowed", default="", help="with --lang auto: comma-separated codes")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--tokens-base", type=float, default=0.0)
    ap.add_argument("--tokens-per-s", type=float, default=0.0)
    args = ap.parse_args()

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download

    started = time.perf_counter()
    try:
        path = snapshot_download(
            REPO_ID, revision=REVISION, ignore_patterns=SKIP, local_files_only=True
        )
    except Exception:
        path = snapshot_download(REPO_ID, revision=REVISION, ignore_patterns=SKIP)
    sys.path.insert(0, path)  # the model code ships inside the download
    from indic_transcribe import IndicTranscribe

    dtype = getattr(torch, args.dtype)
    asr = IndicTranscribe.from_pretrained(path, device=args.device, dtype=dtype)
    nospeech_id = asr.tokenizer.nospeech_id
    generate = asr.model.generate
    last: dict = {}

    def scored_generate(*a, **kw):  # same decoding, plus the scores of each step
        out = generate(*a, **kw, output_scores=True, return_dict_in_generate=True)
        steps = asr.model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True
        )[0]
        first = torch.log_softmax(out.scores[0][0].float(), dim=-1)[nospeech_id]
        lp = steps.float().cpu().tolist()
        last.update(tokens=len(lp), logprobs=lp, nospeech=float(first))
        return out.sequences

    asr.model.generate = scored_generate
    lang = None if args.lang == "auto" else args.lang
    extra = {"allowed_langs": args.allowed.split(",")} if args.allowed and lang is None else {}
    _send(
        {
            "ready": True,
            "load_s": time.perf_counter() - started,
            "device": args.device,
            "dtype": args.dtype,
            "load_max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            **(
                {
                    "load_mps_allocated_mb": torch.mps.current_allocated_memory() / 2**20,
                    "load_mps_driver_mb": torch.mps.driver_allocated_memory() / 2**20,
                }
                if args.device == "mps"
                else {}
            ),
        }
    )

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if req.get("stats"):
                out = {"max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
                if args.device == "mps":
                    out["mps_allocated_mb"] = torch.mps.current_allocated_memory() / 2**20
                    out["mps_driver_mb"] = torch.mps.driver_allocated_memory() / 2**20
                _send(out)
                continue
            pcm = np.frombuffer(base64.b64decode(req["pcm"]), dtype="<i2")
            wav = torch.from_numpy(pcm.astype(np.float32) / 32768.0)
            cap = 512
            if args.tokens_per_s or args.tokens_base:
                cap = min(512, int(args.tokens_base + args.tokens_per_s * len(pcm) / 16000))
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = asr.transcribe(
                    wav,
                    lang=lang,
                    mode=args.mode,
                    return_lid=lang is None,
                    max_new_tokens=max(1, cap),
                    **extra,
                )
            if args.device == "mps":
                torch.mps.synchronize()
            text, used = (out[0], out[1]["lang"]) if lang is None else (out, lang)
            lp = last.get("logprobs") or [0.0]
            _send(
                {
                    "text": text,
                    "seconds": time.perf_counter() - t0,
                    "lang": used,
                    "tokens": last.get("tokens", 0),
                    "mean_logprob": sum(lp) / len(lp),
                    "min_logprob": min(lp),
                    "nospeech_logprob": last.get("nospeech"),
                    "max_new_tokens": cap,
                }
            )
        except Exception as e:  # report and keep serving
            _send({"error": f"{type(e).__name__}: {e}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
