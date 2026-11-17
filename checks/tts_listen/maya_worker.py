# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "mlx==0.32.4",
#   "mlx-audio==0.6.5",
#   "numpy==2.3.5",
#   "huggingface-hub==1.5.0",
# ]
# [tool.uv]
# exclude-newer = "2026-11-16T00:00:00Z"
# ///
"""Worker for maya2-native (8-bit MLX conversion) through mlx-audio, in its own uv env.

Run by run.py as `uv run --script maya_worker.py`; see worker_common.py for the
protocol. Weights: aruntemme/maya2-native-mlx-8bit at a reviewed revision (an 8-bit
conversion of maya-research/maya2-native, Apache-2.0; weights and config only, no
code). mlx-audio 0.6.5 is the first release reviewed here with the `maya2` model
(it arrived in 0.6.3). The Mimi codec that turns the model's codes into audio is
one file from kyutai/moshiko-pytorch-bf16 (CC-BY-4.0, 385 MB); it is fetched at a
pinned revision here and handed to the model, so mlx-audio makes no network call
of its own. The speaker encoder (only for cloning) is never loaded.

Each request: the named voice (opts.voice), `lang_code` = request lang, streaming
with the library's default 0.32 s chunks, its built-in text normaliser on, the
library's default sampling (temperature 0.9) with the MLX seed fixed per sentence.
"""

import contextlib
import os
import sys
import time
from importlib.metadata import version

REPO = "aruntemme/maya2-native-mlx-8bit"
REVISION = "ee18411e5f4f69dcd526b66d2b18dbb8e2f21c8e"
CODEC_REPO = "kyutai/moshiko-pytorch-bf16"
CODEC_FILE = "tokenizer-e351c8d8-checkpoint125.safetensors"
CODEC_REVISION = "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289"
PATTERNS = ["*.json", "*.safetensors", "LICENSE"]


def main() -> int:
    from worker_common import dir_bytes, serve

    with contextlib.redirect_stdout(sys.stderr):
        import mlx.core as mx
        import numpy as np
        from huggingface_hub import hf_hub_download, snapshot_download

        started = time.perf_counter()
        try:
            path = snapshot_download(
                REPO, revision=REVISION, allow_patterns=PATTERNS, local_files_only=True
            )
        except Exception:
            path = snapshot_download(REPO, revision=REVISION, allow_patterns=PATTERNS)
        try:
            codec = hf_hub_download(
                CODEC_REPO, CODEC_FILE, revision=CODEC_REVISION, local_files_only=True
            )
        except Exception:
            codec = hf_hub_download(CODEC_REPO, CODEC_FILE, revision=CODEC_REVISION)

        from mlx_audio.codec.models.mimi import Mimi
        from mlx_audio.codec.models.mimi.mimi import mimi_202407
        from mlx_audio.tts.utils import load

        model = load(path)
        mimi = Mimi(mimi_202407(32))
        mimi.load_pytorch_weights(codec, strict=True)
        mimi.eval()
        model._codec = mimi  # what Model._load_codec would build, from the pinned file
        mx.eval(model.parameters())
        load_s = time.perf_counter() - started

    def synth(text, lang, opts):
        mx.random.seed(int(opts.get("seed", 0)))
        for result in model.generate(
            text, voice=opts.get("voice", "Ahana"), lang_code=lang, stream=True
        ):
            yield np.asarray(result.audio)

    def stats():
        return {
            "mlx_peak_mb": mx.get_peak_memory() / 2**20,
            "mlx_active_mb": mx.get_active_memory() / 2**20,
        }

    return serve(
        synth,
        model.sample_rate,
        {
            "load_s": load_s,
            "device": str(mx.default_device()),
            "dtype": "8-bit backbone",
            "versions": {p: version(p) for p in ("mlx-audio", "mlx", "transformers")},
            "revision": REVISION,
            "download_bytes": dir_bytes(path, codec),
            "cache": os.path.dirname(os.path.dirname(path)),
        },
        stats,
    )


if __name__ == "__main__":
    raise SystemExit(main())
