# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow==22.0.0", "numpy==2.3.5", "huggingface-hub==1.5.0"]
# [tool.uv]
# exclude-newer = "2026-11-16T00:00:00Z"
# ///
"""Public test clips for the multilingual comparison, written as eval directories run.py reads.

usage (from the repo root):
  HF_HUB_DISABLE_XET=1 uv run --script checks/stt_accuracy/public_data.py [--out DIR]

Writes, under DIR (default data/stt_eval/public/, ignored by git):

- monsoon_hi/: a fixed sample of VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN
  (CC-BY-4.0, conversational Hindi with code-switching, 753 clips, one 143 MB parquet,
  downloaded at a pinned revision into DIR/monsoon_raw/). The sample is seeded (SEED)
  and stratified by length: 40 clips under 2 s, 55 of 2 to 5 s, 55 over 5 s, numbered
  in a seeded random order (clips/NNN_<id>.wav), so `--clips 'clips/0[0-2]*'` is a
  random 30-clip subset. truth.jsonl holds, per clip, `lattice` (the dataset's accepted
  written forms per span), `truth` (the first form of each span, for display) and
  `lang` "hi".
- noise/: 2 s of digital silence and of white noise at -60, -40 and -26 dBFS RMS
  (clips/*.wav, no truth), for `run.py --compare --eval-dir DIR/noise`.
"""

import argparse
import json
import random
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MONSOON = "VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN"
MONSOON_REVISION = "a4b7375e9adea16c16100295976cdb0aabd9fb6d"
PARQUET = "data/test-00000-of-00001.parquet"
SEED = 20261116
PER_BUCKET = {"<2s": 40, "2-5s": 55, ">5s": 55}


def bucket(seconds: float) -> str:
    return "<2s" if seconds < 2 else ("2-5s" if seconds <= 5 else ">5s")


def monsoon(out: Path) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    raw = out / "monsoon_raw"
    path = hf_hub_download(
        MONSOON, PARQUET, repo_type="dataset", revision=MONSOON_REVISION, local_dir=raw
    )
    rows = pq.read_table(path, columns=["id", "lattice", "audio_length_s"]).to_pylist()
    rng = random.Random(SEED)
    chosen = []
    for b, n in PER_BUCKET.items():
        pool = sorted((r for r in rows if bucket(r["audio_length_s"]) == b), key=lambda r: r["id"])
        chosen += rng.sample(pool, n)
    rng.shuffle(chosen)
    wanted = {r["id"]: i for i, r in enumerate(chosen)}
    audio = pq.read_table(path, columns=["id", "audio"]).to_pylist()
    d = out / "monsoon_hi"
    (d / "clips").mkdir(parents=True, exist_ok=True)
    entries = {}
    for a in audio:
        if a["id"] not in wanted:
            continue
        data = a["audio"]["bytes"]
        if data[:4] != b"RIFF":
            raise SystemExit(f"{a['id']}: not a WAV file")
        name = f"clips/{wanted[a['id']]:03d}_{a['id']}.wav"
        (d / name).write_bytes(data)
        r = chosen[wanted[a["id"]]]
        entries[name] = {
            "clip": name,
            "truth": " ".join(span[0] for span in r["lattice"] if span),
            "source": "monsoon",
            "lattice": r["lattice"],
            "lang": "hi",
        }
    with (d / "truth.jsonl").open("w") as f:
        for name in sorted(entries):
            f.write(json.dumps(entries[name], ensure_ascii=False) + "\n")
    print(f"{len(entries)} clips in {d}")


def noise(out: Path) -> None:
    import numpy as np

    d = out / "noise" / "clips"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for label, dbfs in [("silence", None), ("noise-60", -60), ("noise-40", -40), ("noise-26", -26)]:
        x = np.zeros(32000) if dbfs is None else rng.standard_normal(32000) * 10 ** (dbfs / 20)
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
        with wave.open(str(d / f"{label}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm)
    print(f"silence and noise in {d}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data" / "stt_eval" / "public"))
    out = Path(ap.parse_args().out)
    monsoon(out)
    noise(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
