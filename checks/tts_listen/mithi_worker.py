# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.11.0",
#   "transformers==5.5.3",
#   "safetensors==0.7.0",
#   "sentencepiece==0.2.1",
#   "numpy==2.3.5",
#   "soundfile==0.13.1",
#   "misaki[en]==0.9.4",
#   "espeakng-loader==0.2.4",
#   "phonemizer-fork==3.3.2",
#   "num2words==0.5.14",
#   "spacy==3.8.7",
#   "click==8.2.1",
#   "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#   "huggingface-hub==1.5.0",
# ]
# [tool.uv]
# exclude-newer = "2026-11-16T00:00:00Z"
# ///
"""Worker for Oomni/mithi-v1-flash (Apache-2.0, 99M, PyTorch) in its own uv env.

Run by run.py as `uv run --script mithi_worker.py [--device mps|cpu]`; see
worker_common.py for the protocol. It imports the model's own `mithi_tts` package
from the reviewed revision's snapshot (samples and assets are not fetched) and calls
`Mithi.synthesize(text, lang, voice)` (voice opts.voice, default charvi; the
model's own text normaliser; seed from its config). Not streaming: the first audio
is the whole clip.

misaki's English G2P (used for any Latin-script text, in every language) loads the
spaCy model en_core_web_sm and, if it is missing, runs `pip install` at runtime; it
is pinned above (spaCy's official 3.8.0 wheel) so that never happens. mithi copies
espeak-ng's data to ~/.cache/mithi only if its path is over 140 characters.
"""

import argparse
import contextlib
import os
import sys
import time
from importlib.metadata import version

REPO = "Oomni/mithi-v1-flash"
REVISION = "94bc455ae91679460cad7a5d3d30b7baf9c1cf13"
SKIP = ["samples/*", "assets/*"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    from worker_common import dir_bytes, serve

    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from huggingface_hub import snapshot_download

        started = time.perf_counter()
        try:
            path = snapshot_download(
                REPO, revision=REVISION, ignore_patterns=SKIP, local_files_only=True
            )
        except Exception:
            path = snapshot_download(REPO, revision=REVISION, ignore_patterns=SKIP)
        sys.path.insert(0, path)  # the model code ships inside the download
        from mithi_tts import SR, Mithi

        tts = Mithi(path, device=args.device)
        load_s = time.perf_counter() - started

    def sync():
        if args.device == "mps":
            torch.mps.synchronize()

    def synth(text, lang, opts):
        yield tts.synthesize(text, lang=lang, voice=opts.get("voice", "charvi"))

    def stats():
        if args.device != "mps":
            return {}
        return {
            "mps_allocated_mb": torch.mps.current_allocated_memory() / 2**20,
            "mps_driver_mb": torch.mps.driver_allocated_memory() / 2**20,
        }

    return serve(
        synth,
        SR,
        {
            "load_s": load_s,
            "device": args.device,
            "dtype": "float32",
            "versions": {p: version(p) for p in ("torch", "transformers", "misaki", "spacy")},
            "revision": REVISION,
            "download_bytes": dir_bytes(path),
            "cache": os.path.dirname(os.path.dirname(path)),
        },
        stats,
        sync,
    )


if __name__ == "__main__":
    raise SystemExit(main())
