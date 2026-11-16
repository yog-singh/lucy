"""Score recogniser configurations on the labelled clips.

usage (from the repo root):
  uv run python checks/stt_accuracy/run.py [CONFIG ...] [--label NAME] [--clips GLOB]
                                           [--compare] [--eval-dir DIR] [--out DIR]

CONFIG names come from recognisers.CONFIGS (default: parakeet). Each configuration is
run over every clip in DIR/truth.jsonl that has a truth (skipped clips are left out),
in its own run directory OUT/LABEL/<config>/ with results.json (summary and every
clip) and errors.txt (each clip whose normalised output differs from the truth).
A table comparing the configurations is printed. Default DIR data/stt_eval/,
default OUT data/checks/stt_accuracy/. See checks/README.md.

--compare runs every clip instead (live clips and enrolment clips, labelled or not;
skipped ones too), scores only those with a truth, and also writes OUT/LABEL/
compare.txt: per clip its length, the truth if any, what Lucy heard live and each
configuration's output, marked where the configurations disagree.

Timing: the model is loaded first (`load`; MLX loads weights lazily, so most of the
real loading cost lands in `warm`), then the first clip is transcribed once as a
warm-up (`warm`, not in the per-clip times), then every clip is transcribed once
and timed. Peak memory is MLX's peak allocation during the configuration (Metal
buffers, which hold the weights) and the process's peak resident size so far.
Indic references (a truth with a `lattice`, as written by public_data.py) are scored
with stt_scoring.indic_score: WER, exact and per-length WER against the native-script
lattice, plus CER and the WER against every accepted form ("cer", "wer_lattice").
A worker recogniser's chosen language is stored per clip ("lang") and counted by
length ("langs_by_length").
A recogniser with a `stats()` method (one that runs in a worker process) adds its own
figures, such as the worker's peak resident size, under "worker" in the summary.
"""

import argparse
import fnmatch
import gc
import json
import resource
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import out_dir
from recognisers import CONFIGS
from stt_scoring import (
    corpus_wer,
    edits,
    exact,
    indic_score,
    length_bucket,
    list_clips,
    load_truth,
    native_lattice,
    normal_text,
    percentile,
    roman_score,
    side_by_side,
)
from translit import non_latin_share

from lucy.voice.audio import float_to_pcm, to_input_format

REPO = Path(__file__).resolve().parents[2]
BUCKETS = ("<2s", "2-5s", ">5s")


def _mlx_peak(reset: bool = False) -> float | None:
    try:
        import mlx.core as mx
    except ImportError:
        return None
    if reset:
        mx.reset_peak_memory()
        return None
    return mx.get_peak_memory() / 2**20


def run_config(name: str, items: list[dict]) -> dict:
    rec = CONFIGS[name]()
    gc.collect()
    _mlx_peak(reset=True)
    started = time.perf_counter()
    rec.load()
    load_s = time.perf_counter() - started
    rows = []
    try:
        t0 = time.perf_counter()
        rec.transcribe(items[0]["pcm"])  # warm-up, not in the per-clip times
        warmup_s = time.perf_counter() - t0
        for it in items:
            t0 = time.perf_counter()
            output = rec.transcribe(it["pcm"])
            took = time.perf_counter() - t0
            row = {
                "clip": it["clip"],
                "audio_s": round(it["audio_s"], 3),
                "truth": it["truth"],
                "output": output,
                "seconds": round(took, 4),
            }
            langs = getattr(rec, "langs", None)
            if langs:
                row["lang"] = langs[-1]
            extras = getattr(rec, "extras", None)
            if extras:
                row["extra"] = extras[-1]
            row["non_latin"] = round(non_latin_share(output), 4)
            if it.get("lattice") is not None:
                rom = roman_score(it["lattice"], output)
                plain = roman_score(it["lattice"], output, fold=False)
                row.update(
                    roman_edits=rom.word_edits,
                    roman_words=rom.ref_words,
                    roman_char_edits=rom.char_edits,
                    roman_chars=rom.ref_chars,
                    roman_plain_edits=plain.word_edits,
                    roman_plain_words=plain.ref_words,
                )
                full = indic_score(it["lattice"], output)
                nat = indic_score(native_lattice(it["lattice"]), output)
                row.update(
                    edits=nat.word_edits,
                    words=nat.ref_words,
                    exact=nat.exact,
                    char_edits=nat.char_edits,
                    chars=nat.ref_chars,
                    lattice_edits=full.word_edits,
                    lattice_words=full.ref_words,
                )
            elif it["truth"] is not None:
                e, n = edits(it["truth"], output)
                row.update(edits=e, words=n, exact=exact(it["truth"], output))
            rows.append(row)
        stats = getattr(rec, "stats", None)
        worker = stats() if stats is not None else None
    finally:
        peak = _mlx_peak()
        rec.close()
    times = [r["seconds"] for r in rows]
    short = [r["seconds"] for r in rows if r["audio_s"] < 2]
    rows_all, rows = rows, [r for r in rows if r["truth"] is not None]
    pairs = [(r["truth"], r["output"]) for r in rows]
    indic = [r for r in rows if "chars" in r]
    summary = {
        "config": name,
        "clips": len(rows),
        "clips_run": len(rows_all),
        "wer": corpus_wer(pairs),
        "wer_by_length": {
            b: (
                corpus_wer(
                    [(r["truth"], r["output"]) for r in rows if length_bucket(r["audio_s"]) == b]
                )
                if any(length_bucket(r["audio_s"]) == b for r in rows)
                else None
            )
            for b in BUCKETS
        },
        "clips_by_length": {
            b: sum(length_bucket(r["audio_s"]) == b for r in rows) for b in BUCKETS
        },
        "exact_rate": sum(r["exact"] for r in rows) / max(1, len(rows)),
        "mean_s": statistics.fmean(times) if times else None,
        "p95_s": percentile(times, 95),
        "mean_s_under_2s": statistics.fmean(short) if short else None,
        "p95_s_under_2s": percentile(short, 95) if short else None,
        "load_s": load_s,
        "warmup_s": warmup_s,
        "mlx_peak_mb": peak,
        "process_max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 2**20,  # bytes on macOS
        "worker": worker,
    }
    if indic:  # Indic references: WER, CER and exact against the native-script lattice

        def ratio(e: str, n: str, rs: list[dict]) -> float:
            return sum(r[e] for r in rs) / max(1, sum(r[n] for r in rs))

        summary["wer"] = ratio("edits", "words", indic)
        summary["wer_by_length"] = {
            b: ratio("edits", "words", [r for r in indic if length_bucket(r["audio_s"]) == b])
            for b in BUCKETS
        }
        summary["cer"] = ratio("char_edits", "chars", indic)
        summary["wer_lattice"] = ratio("lattice_edits", "lattice_words", indic)
        summary["wer_roman"] = ratio("roman_edits", "roman_words", indic)
        summary["cer_roman"] = ratio("roman_char_edits", "roman_chars", indic)
        summary["wer_roman_unfolded"] = ratio("roman_plain_edits", "roman_plain_words", indic)
    letters = [r["non_latin"] for r in rows_all]
    summary["non_latin_mean"] = statistics.fmean(letters) if letters else 0.0
    summary["clips_with_non_latin"] = sum(x > 0 for x in letters)
    if any("lang" in r for r in rows_all):
        summary["langs_by_length"] = {
            b: dict(Counter(r.get("lang") for r in rows_all if length_bucket(r["audio_s"]) == b))
            for b in BUCKETS
        }
    return {"summary": summary, "clips": rows_all}


def _pct(x: float | None) -> str:
    return "  -  " if x is None else f"{100 * x:5.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", default=["parakeet"], help=f"from: {', '.join(CONFIGS)}")
    ap.add_argument("--label", default="run")
    ap.add_argument("--clips", help="only clips whose path matches this glob, e.g. 'enrol_*'")
    ap.add_argument("--compare", action="store_true", help="every clip, plus compare.txt")
    ap.add_argument("--eval-dir", default=str(REPO / "data" / "stt_eval"))
    ap.add_argument("--out", help="output directory (default: data/checks/stt_accuracy/)")
    args = ap.parse_args()
    unknown = [c for c in args.configs if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configuration(s) {unknown}; known: {list(CONFIGS)}")
    eval_dir = Path(args.eval_dir)
    truth = load_truth(eval_dir / "truth.jsonl")
    if args.compare:
        heard = {c.clip: c.heard for c in list_clips(eval_dir)}
        chosen = [
            (c, truth[c].truth if c in truth and not truth[c].skipped else None) for c in heard
        ]
    else:
        heard = {}
        chosen = [
            (t.clip, t.truth) for t in truth.values() if t.truth is not None and not t.skipped
        ]
    if args.clips:
        chosen = [(c, t) for c, t in chosen if fnmatch.fnmatch(c, args.clips)]
    if not chosen:
        raise SystemExit("no clips: run label.py (and auto_enrol.py) first, or use --compare")
    items = []
    for clip, text in chosen:
        samples = to_input_format(eval_dir / clip)
        items.append(
            {
                "clip": clip,
                "truth": text,
                "lattice": truth[clip].lattice if clip in truth else None,
                "pcm": float_to_pcm(samples),
                "audio_s": len(samples) / 16000,
            }
        )
    n_labelled = sum(it["truth"] is not None for it in items)
    skipped = sum(t.skipped for t in truth.values())
    print(f"{len(items)} clips, {n_labelled} labelled ({skipped} skipped in truth.jsonl)")
    base = out_dir("stt_accuracy", args.out) / args.label
    summaries = []
    outputs: dict[str, dict[str, str]] = {}
    for name in args.configs:
        print(f"running {name} ...", flush=True)
        result = run_config(name, items)
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        with (d / "errors.txt").open("w") as f:
            for r in result["clips"]:
                if r["truth"] is not None and not r["exact"]:
                    f.write(
                        f"{r['clip']}  ({r['audio_s']:.1f}s, {r['edits']}/{r['words']} edits)\n"
                    )
                    f.write(f"  truth:  {r['truth']}\n  output: {r['output']}\n")
                    f.write(
                        f"  normalised: {normal_text(r['truth'])!r} vs {normal_text(r['output'])!r}\n\n"
                    )
        summaries.append(result["summary"])
        outputs[name] = {r["clip"]: r["output"] for r in result["clips"]}
    if args.compare:
        rows = [
            {
                "clip": it["clip"],
                "audio_s": it["audio_s"],
                "truth": it["truth"],
                "heard": heard.get(it["clip"]),
            }
            for it in items
        ]
        (base / "compare.txt").write_text(side_by_side(rows, outputs))
        print(f"side by side in {base}/compare.txt")
    s0 = summaries[0]
    counts = " ".join(f"{b}:{s0['clips_by_length'][b]}" for b in BUCKETS)
    print(f"\nWER and exact in %, times in seconds per clip; clips by length {counts}")
    print(
        f"{'config':16} {'WER':>6} {'<2s':>6} {'2-5s':>6} {'>5s':>6} {'exact':>6} {'mean':>7} {'p95':>7} {'load':>6} {'warm':>6} {'mlxMB':>6} {'rssMB':>6}"
    )
    for s in summaries:
        w = s["wer_by_length"]
        mlx = f"{s['mlx_peak_mb']:6.0f}" if s["mlx_peak_mb"] is not None else "     -"
        print(
            f"{s['config']:16} {_pct(s['wer']):>6} {_pct(w['<2s']):>6} {_pct(w['2-5s']):>6} {_pct(w['>5s']):>6} "
            f"{_pct(s['exact_rate']):>6} {s['mean_s']:7.3f} {s['p95_s']:7.3f} {s['load_s']:6.1f} {s['warmup_s']:6.1f} {mlx} {s['process_max_rss_mb']:6.0f}"
        )
        if s["mean_s_under_2s"] is not None:
            print(
                f"{'':16} under 2 s: mean {s['mean_s_under_2s']:.3f}, p95 {s['p95_s_under_2s']:.3f}"
            )
        if "cer" in s:
            print(
                f"{'':16} native-script lattice: CER {_pct(s['cer'])}; "
                f"WER with every accepted form (Latin too) {_pct(s['wer_lattice'])}"
            )
            print(
                f"{'':16} romanised, folded: WER {_pct(s['wer_roman'])}, CER "
                f"{_pct(s['cer_roman'])}; unfolded WER {_pct(s['wer_roman_unfolded'])}"
            )
        print(
            f"{'':16} non-Latin letters: {s['clips_with_non_latin']} clips, "
            f"mean share {_pct(s['non_latin_mean'])}"
        )
        if s["worker"]:
            print(f"{'':16} worker: {json.dumps(s['worker'])}")
    print(f"\nresults in {base}/<config>/results.json, differing clips in errors.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
