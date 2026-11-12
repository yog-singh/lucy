"""Analyse a run: transcripts vs references, replies vs what the output audio says (Parakeet on
the output WAV, cut into replies at silences), and the timing table.

usage (from the repo root): uv run python checks/voice_e2e/analyse.py [--label LABEL] [--out DIR]
Reads OUT/LABEL/run.json, reply.wav and voice_timings.jsonl from run_cli.py and writes
OUT/LABEL/analysis.json (default OUT is data/checks/voice_e2e/). See checks/README.md.
"""

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from common import out_dir
from wer import wer

from lucy.adapters.stt_parakeet_mlx import ParakeetMLXSTT
from lucy.core.types import AudioChunk
from lucy.voice.audio import float_to_pcm, read_wav, resample
from lucy.voice.session import TIMINGS_FILE_NAME

E = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--label", default="run")
ap.add_argument("--out", help="output directory (default: data/checks/voice_e2e/)")
args = ap.parse_args()
run_dir = out_dir("voice_e2e", args.out) / args.label
out_wav = run_dir / "reply.wav"
timings_file = run_dir / TIMINGS_FILE_NAME
run = json.load(open(run_dir / "run.json"))
refs = json.load(open(E / "inputs" / "turns.json"))[: run["turns"]]
users = [l[2][5:] for l in run["lines"] if l[1] == "out" and l[2].startswith("you> ")]
replies = [l[2][6:] for l in run["lines"] if l[1] == "out" and l[2].startswith("lucy> ")]
timings = [json.loads(x) for x in timings_file.read_text().splitlines()]


def segments(samples, rate, gap=1.2, level=0.01):
    loud = np.abs(samples) > level
    idx = np.flatnonzero(loud)
    if not len(idx):
        return []
    out, start, last = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - last > gap * rate:
            out.append((start, last))
            start = i
        last = i
    out.append((start, last))
    return [(s, e) for s, e in out if (e - s) / rate > 0.3]


async def main():
    samples, rate = read_wav(out_wav)
    segs = segments(samples, rate)
    stt = ParakeetMLXSTT()
    await stt.warm_up()
    heard = []
    for s, e in segs:
        piece = resample(samples[max(0, s - 2400) : e + 2400], rate, 16000)
        heard.append(
            (
                round(s / rate, 2),
                round((e - s) / rate, 2),
                (
                    await stt.transcribe(AudioChunk(data=float_to_pcm(piece), sample_rate=16000))
                ).text,
            )
        )
    await stt.aclose()
    rows = []
    for i, ref in enumerate(refs):
        t = timings[i] if i < len(timings) else {}
        row = {
            "turn": ref["id"],
            "voice": f"{ref['engine']}/{ref['voice']}",
            "said": ref["text"],
            "transcript": users[i] if i < len(users) else None,
            "stt_wer": round(wer(ref["text"], users[i]), 3) if i < len(users) else None,
            "reply": replies[i] if i < len(replies) else None,
            "reply_heard": heard[i][2] if i < len(heard) else None,
            "reply_round_trip_wer": round(wer(replies[i], heard[i][2]), 3)
            if i < min(len(replies), len(heard))
            else None,
            "outcome": t.get("outcome"),
            "stages": t.get("stages"),
            "utterance_s": t.get("utterance_s"),
            "reply_audio_s": t.get("reply_audio_s"),
        }
        rows.append(row)
    totals = [
        r["stages"]["total"] for r in rows if r["stages"] and r["stages"]["total"] is not None
    ]

    def dist(key):
        v = [r["stages"][key] for r in rows if r["stages"] and r["stages"].get(key) is not None]
        return (
            {
                "n": len(v),
                "min": min(v),
                "median": round(statistics.median(v), 3),
                "mean": round(statistics.mean(v), 3),
                "max": max(v),
            }
            if v
            else None
        )

    summary = {
        k: dist(k)
        for k in (
            "endpointing",
            "stt",
            "llm_first_text",
            "llm_first_sentence",
            "tts_first_audio",
            "output",
            "total",
        )
    }
    json.dump(
        {"rows": rows, "summary": summary, "segments": heard},
        open(run_dir / "analysis.json", "w"),
        indent=1,
    )
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    print(json.dumps(summary, indent=1))
    print(
        "segments:",
        len(heard),
        "replies:",
        len(replies),
        "users:",
        len(users),
        "timings:",
        len(timings),
    )
    print("written to", run_dir / "analysis.json")


asyncio.run(main())
