# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.11.0",
#   "torchaudio==2.11.0",
#   "numpy==2.3.5",
#   "safetensors==0.7.0",
#   "soundfile==0.13.1",
#   "vocos==0.1.0",
#   "pydub==0.25.1",
#   "packaging",
#   "huggingface-hub==1.5.0",
# ]
# [tool.uv]
# exclude-newer = "2026-11-16T00:00:00Z"
# ///
"""Worker for ARTPARK-IISc/DhVaani-0.5 (CC-BY-NC-4.0, gated) in its own uv env.

Run by run.py as `uv run --script dhvaani_worker.py [--device mps|cpu]`; see
worker_common.py for the protocol. It loads the model the way the repo's own
dhvaani.py does (the bundled `_backend/zipvoice` code of the reviewed revision, its
weights, the Vocos vocoder) but without dhvaani.py itself, which points HF_HOME at a
folder inside the model snapshot. The vocoder charactr/vocos-mel-24khz (MIT, 54 MB)
is fetched at a pinned revision and loaded with `weights_only=True`.

Voice: cloned from the Hindi female reference clip and its transcript shipped by the
model's demo Space (ARTPARK-IISc/DhVaani-Demo, samples/hindi_female.b64 and
transcripts.txt, pinned revision), the same reference for every language. Settings:
opts.num_step (16, the default "quality", or 8, "faster"), guidance 1.0, speed 1.0,
seed 666 (the repo defaults). Not streaming: the first audio is the whole clip.

Shims (no behaviour change): `torch.utils.tensorboard` is stubbed (imported by the
training helpers, unused at inference), and torchaudio's file load/save (which need
torchcodec in torchaudio 2.9+) are replaced by soundfile.
"""

import argparse
import base64
import contextlib
import os
import sys
import time
import types
from importlib.metadata import version

REPO = "ARTPARK-IISc/DhVaani-0.5"
REVISION = "582a342b3eb310f973a8e0af06798b453234b6ab"
VOCOS = "charactr/vocos-mel-24khz"
VOCOS_REVISION = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
SPACE = "ARTPARK-IISc/DhVaani-Demo"
SPACE_REVISION = "5702e95a1d282b1ae7ba3e932752d6bfbea638f9"
REF_STEM = "hindi_female"


def _fetch(repo, revision, **kw):
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, revision=revision, local_files_only=True, **kw)
    except Exception:
        return snapshot_download(repo, revision=revision, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--ref-dir", required=True, help="where the decoded reference wav goes")
    args = ap.parse_args()
    from worker_common import dir_bytes, serve

    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import soundfile as sf
        import torch

        started = time.perf_counter()
        path = _fetch(REPO, REVISION, ignore_patterns=["samples/*"])
        vocos_path = _fetch(VOCOS, VOCOS_REVISION, allow_patterns=["config.yaml", "*.bin"])
        space = _fetch(
            SPACE,
            SPACE_REVISION,
            repo_type="space",
            allow_patterns=[f"samples/{REF_STEM}.b64", "samples/transcripts.txt"],
        )
        transcripts = {}
        for line in open(os.path.join(space, "samples", "transcripts.txt"), encoding="utf-8"):
            if ":" in line:
                stem, text = line.split(":", 1)
                transcripts[stem.strip()] = text.strip()
        ref_text = transcripts[REF_STEM]
        ref_wav = os.path.join(args.ref_dir, f"{REF_STEM}.wav")
        os.makedirs(args.ref_dir, exist_ok=True)
        with open(os.path.join(space, "samples", f"{REF_STEM}.b64")) as f:
            data = base64.b64decode(f.read().strip())
        with open(ref_wav, "wb") as f:
            f.write(data)

        sys.modules.setdefault(
            "torch.utils.tensorboard", types.SimpleNamespace(SummaryWriter=object)
        )
        sys.path.insert(0, os.path.join(path, "_backend"))
        import json

        import safetensors.torch as st
        import torchaudio
        import zipvoice.bin.infer_zipvoice as iz
        from zipvoice.models.zipvoice import ZipVoice
        from zipvoice.tokenizer.tokenizer import SimpleTokenizer
        from zipvoice.utils.feature import VocosFbank

        def load_prompt_wav(p, sampling_rate):
            audio, rate = sf.read(p, dtype="float32", always_2d=True)
            wav = torch.from_numpy(audio.T.copy())
            if rate != sampling_rate:
                wav = torchaudio.functional.resample(wav, rate, sampling_rate)
            return wav

        captured = {}
        iz.load_prompt_wav = load_prompt_wav
        iz.torchaudio = types.SimpleNamespace(
            save=lambda p, wav, sample_rate: captured.update(wav=wav.numpy())
        )

        device = torch.device(args.device)
        cfg = json.loads(open(os.path.join(path, "model.json")).read())
        rate = cfg["feature"]["sampling_rate"]
        tokenizer = SimpleTokenizer(token_file=os.path.join(path, "tokens.txt"))
        model = ZipVoice(**cfg["model"], vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id)
        sd = st.load_file(os.path.join(path, "model.safetensors"))
        sd = {k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")}
        model.load_state_dict(sd, strict=True)
        model = model.to(device).eval()
        feature = VocosFbank()
        vocoder = iz.get_vocoder(vocos_path).to(device).eval()
        load_s = time.perf_counter() - started

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()

    def synth(text, lang, opts):
        torch.manual_seed(666)
        captured.clear()
        with torch.inference_mode():
            iz.generate_sentence(
                save_path="unused.wav",
                prompt_text=ref_text,
                prompt_wav=ref_wav,
                text=text,
                model=model,
                vocoder=vocoder,
                tokenizer=tokenizer,
                feature_extractor=feature,
                device=device,
                num_step=int(opts.get("num_step", 16)),
                guidance_scale=1.0,
                speed=1.0,
                sampling_rate=rate,
            )
        yield np.asarray(captured["wav"]).reshape(-1)

    def stats():
        if device.type != "mps":
            return {}
        return {
            "mps_allocated_mb": torch.mps.current_allocated_memory() / 2**20,
            "mps_driver_mb": torch.mps.driver_allocated_memory() / 2**20,
        }

    return serve(
        synth,
        rate,
        {
            "load_s": load_s,
            "device": args.device,
            "dtype": "float32",
            "versions": {p: version(p) for p in ("torch", "torchaudio", "vocos")},
            "revision": REVISION,
            "download_bytes": dir_bytes(path, vocos_path),
            "cache": os.path.dirname(os.path.dirname(path)),
            "ref_text": ref_text,
        },
        stats,
        sync,
    )


if __name__ == "__main__":
    raise SystemExit(main())
