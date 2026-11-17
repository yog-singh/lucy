# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "supertonic==1.3.1",
#   "onnxruntime==1.23.2",
#   "numpy==2.3.5",
#   "soundfile==0.13.1",
#   "huggingface-hub==1.5.0",
# ]
# [tool.uv]
# exclude-newer = "2026-11-16T00:00:00Z"
# ///
"""Worker for Supertonic 3 (ONNX, 99M) through the `supertonic` package, in its own uv env.

Run by run.py as `uv run --script supertonic_worker.py [--model-dir DIR] [--threads N]
[--provider cpu|coreml]`; see worker_common.py for the protocol. supertonic 1.3.1
(the reviewed release; pure Python over onnxruntime) pins Supertone/supertonic-3 at
REVISION. The ONNX files, voice styles and licence are fetched once at that revision
into --model-dir (default data/tools/supertonic3, git-ignored) and loaded with
auto_download off, so the package makes no network call of its own.

Each request: the voice style opts.voice (F1 ... F5 are the female presets), lang =
request lang ("hi" for Hindi in either script), the package defaults (8 flow steps,
speed 1.05). No streaming: the whole clip is the first audio. Text is passed as
written: the package has no number normaliser.
"""

import argparse
import contextlib
import os
import sys
import time
from importlib.metadata import version

REPO = "Supertone/supertonic-3"
REVISION = "724fb5abbf5502583fb520898d45929e62f02c0b"  # pinned by supertonic 1.3.1
PATTERNS = ["onnx/*", "voice_styles/*", "LICENSE"]
DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "tools", "supertonic3")


def main() -> int:
    from worker_common import dir_bytes, serve

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_DIR)
    ap.add_argument("--threads", type=int, default=None, help="onnxruntime intra-op threads")
    ap.add_argument("--provider", default="cpu", choices=["cpu", "coreml"])
    args = ap.parse_args()
    model_dir = os.path.abspath(args.model_dir)

    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        from huggingface_hub import snapshot_download

        if not os.path.exists(os.path.join(model_dir, "onnx", "vocoder.onnx")):
            snapshot_download(REPO, revision=REVISION, allow_patterns=PATTERNS, local_dir=model_dir)
        started = time.perf_counter()
        import supertonic.loader as loader

        if args.provider == "coreml":
            loader.DEFAULT_ONNX_PROVIDERS = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            loader.DEFAULT_ONNX_PROVIDERS = ["CPUExecutionProvider"]
        from supertonic import TTS

        tts = TTS(
            model="supertonic-3",
            model_dir=model_dir,
            auto_download=False,
            intra_op_num_threads=args.threads,
        )
        styles = {}
        load_s = time.perf_counter() - started

    def synth(text, lang, opts):
        name = opts.get("voice", "F1")
        if name not in styles:
            styles[name] = tts.get_voice_style(voice_name=name)
        np.random.seed(int(opts.get("seed", 0)))
        wav, _ = tts.synthesize(text, voice_style=styles[name], lang=lang)
        yield np.asarray(wav, dtype=np.float32).reshape(-1)

    return serve(
        synth,
        tts.sample_rate,
        {
            "load_s": load_s,
            "device": f"onnxruntime {args.provider}",
            "threads": args.threads or "auto",
            "versions": {p: version(p) for p in ("supertonic", "onnxruntime")},
            "revision": REVISION,
            "download_bytes": dir_bytes(
                os.path.join(model_dir, "onnx"), os.path.join(model_dir, "voice_styles")
            ),
            "cache": model_dir,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
