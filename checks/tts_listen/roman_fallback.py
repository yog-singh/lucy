"""Does converting romanised Hindi replies to Devanagari make Magpie say them better?

usage (from the repo root): uv run python checks/tts_listen/roman_fallback.py
    [--rows ROWS.json ...] [--limit N] [--out DIR]
Takes every Latin-script sentence that `lucy.core.hinglish.latin_language` judges Hindi
from the Hindi and mixed replies in reply_language rows (default: all of
data/checks/reply_language/*/rows.json), synthesises each through the `magpie` adapter
twice, with the romanised-Hindi fallback on (converted to Devanagari, `hi-IN`) and off
(as written, `en-US`), transcribes both with the `indic_flex` recogniser (Latin output)
and scores word error against the romanised original after the spelling fold of
checks/stt_accuracy (`translit.fold`). Writes rows.json, the WAVs and a listening page
(index.html; a third column shows the model's Devanagari reply to the same question
from rows that have one, when available) to OUT (default
data/checks/tts_listen/roman_fallback/). The recogniser is Lucy's, so a high score means
Lucy's own ears understand the speech, not that it sounds natural: listen to the page.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import re
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "stt_accuracy"))
from report import render_page  # noqa: E402
from translit import fold  # noqa: E402

from lucy.core import registry  # noqa: E402
from lucy.core.devanagari import romanised_to_devanagari  # noqa: E402
from lucy.core.hinglish import latin_language  # noqa: E402
from lucy.core.sentences import SentenceSplitter  # noqa: E402
from lucy.core.types import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, AudioChunk  # noqa: E402
from lucy.voice.audio import resample  # noqa: E402

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
DEFAULT_ROWS = "data/checks/reply_language/*/rows.json"


def collect(paths: list[str]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Unique romanised Hindi sentences (with their question) and, per question, one
    Devanagari reply from the rows."""
    sentences: list[dict[str, Any]] = []
    seen: set[str] = set()
    deva: dict[str, str] = {}
    for path in paths:
        for row in json.loads(Path(path).read_text()):
            reply = row.get("reply") or ""
            if row.get("lang") == "en" or not reply:
                continue
            if DEVANAGARI.search(reply):
                deva.setdefault(row["q"], reply)
                continue
            splitter = SentenceSplitter()
            for text in [*splitter.feed(reply), *splitter.flush()]:
                text = text.strip()
                if text and text not in seen and latin_language(text) == "hi":
                    seen.add(text)
                    sentences.append({"q": row["q"], "text": text})
    return sentences, deva


def word_errors(truth: str, heard: str) -> tuple[int, int]:
    ref, hyp = fold(truth).split(), fold(heard).split()
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1], len(ref)


async def synth(tts: Any, text: str) -> bytes:
    return b"".join([c.data async for c in tts.synthesize(text)])


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(OUTPUT_SAMPLE_RATE)
        w.writeframes(pcm)


async def run(args: argparse.Namespace) -> int:
    paths = args.rows or sorted(glob.glob(DEFAULT_ROWS))
    sentences, deva = collect(paths)
    if args.limit:
        sentences = sentences[: args.limit]
    out = Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    print(f"{len(sentences)} sentences from {len(paths)} rows files", flush=True)
    registry.load_adapters(["lucy.adapters.tts_magpie", "lucy.adapters.stt_indic_flex"])
    on = registry.get("tts", "magpie")(roman_hindi=True)
    off = registry.get("tts", "magpie")(roman_hindi=False)
    stt = registry.get("stt", "indic_flex")()
    rows: list[dict[str, Any]] = []
    try:
        for n, s in enumerate(sentences):
            sid = f"s{n:02d}"
            row: dict[str, Any] = {"id": sid, **s, "converted": romanised_to_devanagari(s["text"])}
            for column, tts in (("on", on), ("off", off)):
                tts._previous_lang = None  # each sentence on its own
                pcm = await synth(tts, s["text"])
                write_wav(out / "wav" / f"{sid}-{column}.wav", pcm)
                samples = np.frombuffer(pcm, dtype=np.int16)
                low = resample(samples / 32768.0, OUTPUT_SAMPLE_RATE, INPUT_SAMPLE_RATE)
                low = (np.clip(low, -1, 1) * 32767).astype(np.int16)
                heard = (await stt.transcribe(AudioChunk(data=low.tobytes(),
                         sample_rate=INPUT_SAMPLE_RATE, channels=1))).text  # fmt: skip
                errors, words = word_errors(s["text"], heard)
                row[column] = {"heard": heard, "errors": errors, "words": words,
                               "duration_s": len(samples) / OUTPUT_SAMPLE_RATE}  # fmt: skip
            print(f"{sid} on {row['on']['errors']}/{row['on']['words']} "
                  f"off {row['off']['errors']}/{row['off']['words']}  {s['text'][:60]}",
                  flush=True)  # fmt: skip
            rows.append(row)
    finally:
        for engine in (on, off, stt):
            await engine.aclose()
    totals = {
        c: (sum(r[c]["errors"] for r in rows), sum(r[c]["words"] for r in rows))
        for c in ("on", "off")
    }
    exact = {c: sum(r[c]["errors"] == 0 for r in rows) for c in ("on", "off")}
    summary = {c: {"wer": e / max(1, w), "errors": e, "words": w, "exact": exact[c]}
               for c, (e, w) in totals.items()}  # fmt: skip
    (out / "rows.json").write_text(json.dumps({"summary": summary, "rows": rows},
                                              ensure_ascii=False, indent=1))  # fmt: skip
    print("SUMMARY", json.dumps(summary), flush=True)
    write_page(out, rows, deva, summary)
    return 0


def write_page(out: Path, rows: list[dict[str, Any]], deva: dict[str, str], summary: dict) -> None:
    sentences, measured = [], []
    for r in rows:
        model = deva.get(r["q"])
        text = r["text"] + (f"   [model's Devanagari reply to the same question: {model}]"
                            if model else "")  # fmt: skip
        sentences.append({"id": r["id"], "kind": "romanised", "lang": "hi", "text": text})
        for column, shown in (("on", r["converted"]), ("off", r["text"])):
            d = r[column]["duration_s"]
            measured.append({
                "column": column, "id": r["id"], "phase": "idle",
                "wav": f"wav/{r['id']}-{column}.wav", "first_s": 0.0, "total_s": 0.0,
                "duration_s": d, "rtf": None,
                "asr": f"{r[column]['heard']}  ({r[column]['errors']}/{r[column]['words']} "
                       f"word errors; sent: {shown})",
            })  # fmt: skip
    columns = [
        {"column": "on", "title": "Converted to Devanagari (hi-IN)",
         "facts": {"Word error (fold)": f"{summary['on']['wer']:.0%}",
                   "Heard exactly": f"{summary['on']['exact']}/{len(rows)}"}},
        {"column": "off", "title": "As written (en-US)",
         "facts": {"Word error (fold)": f"{summary['off']['wer']:.0%}",
                   "Heard exactly": f"{summary['off']['exact']}/{len(rows)}"}},
    ]  # fmt: skip
    notes = [
        "Romanised Hindi sentences from the reply_language check rows, spoken by Magpie "
        "(voice Sofia, seed 0) with the fallback on and off, transcribed by indic_flex.",
        "Word error is against the romanised original after the stt_accuracy spelling fold. "
        "Timings are not measured on this page (shown as 0).",
    ]
    page = render_page("Romanised Hindi fallback", sentences, columns, measured, notes)
    (out / "index.html").write_text(page)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", help=f"reply_language rows files (default {DEFAULT_ROWS})")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/checks/tts_listen/roman_fallback")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
