"""Tables for the multilingual comparison, from run.py's results (no model is run).

usage (from the repo root, after the runs listed in checks/README.md, stt_accuracy):
  uv run python checks/stt_accuracy/multilingual_report.py [--out DIR]

Reads OUT/{multi_hi, multi_hi_parakeet30, multi_en, multi_noise, cand1}/<config>/
results.json (OUT default data/checks/stt_accuracy/), rescores the Hindi clips with the
current stt_scoring.indic_score, and writes OUT/multi_report.txt:

A. per configuration and language: WER, CER, exact, time, memory;
B. language identification by clip length, for every configuration that chose it;
C. code-switched Hindi clips (inferred: a span of the reference offers a Latin form of
   four letters or more that is an English dictionary word, /usr/share/dict/words),
   with each configuration's output;
D. designs: one model with automatic language; Parakeet for English plus flex routed by
   flex's English/Hindi identification;
E. silence and noise outputs.
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stt_scoring import (
    corpus_wer,
    exact,
    indic_score,
    length_bucket,
    native_lattice,
    percentile,
)

REPO = Path(__file__).resolve().parents[2]
BUCKETS = ("<2s", "2-5s", ">5s")
LANG_CODES = {"english": "en", "hindi": "hi", "urdu": "ur"}
SCRIPTS = [
    ("Devanagari", r"[ऀ-ॿ]"),
    ("Latin", r"[A-Za-z]"),
    ("Arabic", r"[؀-ۿ]"),
    ("other Indic", r"[ঀ-෿]"),
]


def code(lang: str | None) -> str:
    if lang is None:
        return "-"
    return LANG_CODES.get(lang.lower(), lang.lower())


def load(out: Path, label: str) -> dict[str, dict]:
    d = out / label
    if not d.exists():
        return {}
    return {p.parent.name: json.loads(p.read_text()) for p in sorted(d.glob("*/results.json"))}


def memory(summary: dict) -> str:
    w = summary.get("worker") or {}
    if "mps_allocated_mb" in w:
        return f"{w['max_rss_mb']:.0f} RSS / {w['mps_allocated_mb']:.0f} MPS"
    if "mlx_peak_mb" in w:
        return f"{w['max_rss_mb']:.0f} RSS / {w['mlx_peak_mb']:.0f} MLX"
    if summary.get("mlx_peak_mb"):
        return f"{summary['process_max_rss_mb']:.0f} RSS / {summary['mlx_peak_mb']:.0f} MLX"
    return "-"


def scripts(text: str) -> str:
    found = [name for name, rx in SCRIPTS if re.search(rx, text)]
    return "+".join(found) or "empty"


def hindi_table(results: dict[str, dict], lattices: dict[str, list]) -> list[str]:
    lines = [
        "| Config | Clips | WER native | CER native | WER any form | Exact | <2 s | 2-5 s | >5 s"
        " | Mean / p95 s | Memory MB | Output script |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, res in results.items():
        rows = res["clips"]
        tot = {"we": 0, "wn": 0, "ce": 0, "cn": 0, "le": 0, "ln": 0, "ex": 0}
        by_b: dict[str, list[int]] = {b: [0, 0] for b in BUCKETS}
        for r in rows:
            lat = lattices[r["clip"]]
            nat = indic_score(native_lattice(lat), r["output"])
            full = indic_score(lat, r["output"])
            tot["we"] += nat.word_edits
            tot["wn"] += nat.ref_words
            tot["ce"] += nat.char_edits
            tot["cn"] += nat.ref_chars
            tot["le"] += full.word_edits
            tot["ln"] += full.ref_words
            tot["ex"] += nat.exact
            b = by_b[length_bucket(r["audio_s"])]
            b[0] += nat.word_edits
            b[1] += nat.ref_words
        times = [r["seconds"] for r in rows]
        sc = Counter(scripts(r["output"]) for r in rows).most_common(3)
        pct = lambda e, n: f"{100 * e / max(1, n):.1f}"  # noqa: E731
        lines.append(
            f"| `{name}` | {len(rows)} | {pct(tot['we'], tot['wn'])} | {pct(tot['ce'], tot['cn'])}"
            f" | {pct(tot['le'], tot['ln'])} | {pct(tot['ex'], len(rows))}"
            + "".join(f" | {pct(*by_b[b])}" for b in BUCKETS)
            + f" | {statistics.fmean(times):.3f} / {percentile(times, 95):.3f}"
            f" | {memory(res['summary'])} | {', '.join(f'{k} {v}' for k, v in sc)} |"
        )
    return lines


def english_table(results: dict[str, dict]) -> list[str]:
    lines = [
        "| Config | Clips | WER | Exact | <2 s | 2-5 s | >5 s | Mean / p95 s | Memory MB |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, res in results.items():
        rows = [r for r in res["clips"] if r["truth"] is not None]
        pairs = [(r["truth"], r["output"]) for r in rows]
        times = [r["seconds"] for r in res["clips"]]
        by_b = [
            corpus_wer(
                [(r["truth"], r["output"]) for r in rows if length_bucket(r["audio_s"]) == b]
            )
            for b in BUCKETS
        ]
        ex = sum(exact(t, o) for t, o in pairs) / max(1, len(pairs))
        lines.append(
            f"| `{name}` | {len(rows)} | {100 * corpus_wer(pairs):.1f} | {100 * ex:.1f}"
            + "".join(f" | {100 * w:.1f}" for w in by_b)
            + f" | {statistics.fmean(times):.3f} / {percentile(times, 95):.3f}"
            f" | {memory(res['summary'])} |"
        )
    return lines


def lid_table(sets: dict[str, tuple[str, dict[str, dict], set[str] | None]]) -> list[str]:
    lines = [
        "| Config | True language (clips) | Chosen: overall | <2 s | 2-5 s | >5 s | Correct |",
        "|---|---|---|---|---|---|---|",
    ]
    for true, (lang, results, only) in sets.items():
        for name, res in results.items():
            rows = [r for r in res["clips"] if "lang" in r and (only is None or r["clip"] in only)]
            if not rows or name in ("indic_flex_hi", "indic_flex_hi_mixed", "indic_flex_fp16"):
                continue
            if name.endswith("_hi") or name in ("whisper_turbo", "qwen3_asr", "parakeet"):
                continue
            cells = []
            for sel in (None, *BUCKETS):
                rs = [r for r in rows if sel is None or length_bucket(r["audio_s"]) == sel]
                c = Counter(code(r["lang"]) for r in rs).most_common(4)
                ok = sum(code(r["lang"]) == lang for r in rs)
                cells.append(
                    f"{ok}/{len(rs)} "
                    + ("(" + ", ".join(f"{k} {v}" for k, v in c) + ")" if rs else "")
                )
            correct = sum(code(r["lang"]) == lang for r in rows) / len(rows)
            lines.append(
                f"| `{name}` | {true} ({len(rows)}) | "
                + " | ".join(cells)
                + f" | {100 * correct:.0f} % |"
            )
    return lines


def code_switched(lattices: dict[str, list]) -> list[str]:
    words_file = Path("/usr/share/dict/words")
    english = (
        {w.strip().lower() for w in words_file.read_text().splitlines()}
        if words_file.exists()
        else set()
    )
    out = []
    for clip, lat in lattices.items():
        for span in lat:
            native = [f for f in span if not re.search("[A-Za-z]", f)]
            latin = [f for f in span if re.fullmatch("[A-Za-z]{4,}", f)]
            if native and any(f.lower() in english for f in latin):
                out.append(clip)
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data" / "checks" / "stt_accuracy"))
    out = Path(ap.parse_args().out)
    truth_file = REPO / "data" / "stt_eval" / "public" / "monsoon_hi" / "truth.jsonl"
    entries = [json.loads(line) for line in truth_file.read_text().splitlines() if line.strip()]
    lattices = {e["clip"]: e["lattice"] for e in entries}
    hi = load(out, "multi_hi")
    hi_p = load(out, "multi_hi_parakeet30")
    en = load(out, "multi_en")
    cand = load(out, "cand1")
    noise = load(out, "multi_noise")
    rep: list[str] = []

    rep += ["A. Hindi (Monsoon sample, 150 clips; parakeet on the first 30)", ""]
    rep += hindi_table({**hi_p, **hi}, lattices)
    rep += ["", "A. English (the user's 73 labelled clips)", ""]
    rep += english_table(
        {
            **{
                k: v
                for k, v in cand.items()
                if k in ("parakeet", "whisper_turbo", "qwen3_asr", "qwen3_asr_context")
            },
            **en,
        }
    )

    en_labelled = {r["clip"] for res in en.values() for r in res["clips"] if r["truth"] is not None}
    rep += ["", "B. Language chosen (correct/clips, then the most chosen codes)", ""]
    rep += lid_table({"en": ("en", en, en_labelled), "hi": ("hi", hi, None)})
    rep += ["", "B. The user's skipped clips (Hindi speech among them), language and output", ""]
    for name, res in en.items():
        for r in res["clips"]:
            if r["truth"] is None:
                rep.append(
                    f"{r['clip']} ({r['audio_s']:.1f} s) `{name}` [{r.get('lang')}]: {r['output']}"
                )

    cs = code_switched(lattices)
    rep += ["", f"C. Code-switched clips (inferred): {len(cs)} of {len(lattices)}; first 10", ""]
    shown = [
        "whisper_turbo_auto", "whisper_turbo_hi", "qwen3_asr_auto", "qwen3_asr_hi",
        "indic_flex_hi", "indic_flex_hi_mixed", "indic_flex_enhi", "indic_flex_fp16",
    ]  # fmt: skip
    by_clip = {n: {r["clip"]: r for r in hi[n]["clips"]} for n in shown if n in hi}
    for clip in cs[:10]:
        any_row = next(iter(by_clip.values()))[clip]
        rep.append(f"{clip} ({any_row['audio_s']:.1f} s)")
        rep.append(f"  truth: {any_row['truth']}")
        for n, rows in by_clip.items():
            r = rows[clip]
            rep.append(f"  {n} [{r.get('lang', '')}]: {r['output']}")
        rep.append("")
    # Code-switched subset scores
    rep += ["C. Native-script WER on the code-switched subset only", ""]
    for n in hi:
        rows = [r for r in hi[n]["clips"] if r["clip"] in cs]
        e = sum(
            indic_score(native_lattice(lattices[r["clip"]]), r["output"]).word_edits for r in rows
        )
        w = sum(
            indic_score(native_lattice(lattices[r["clip"]]), r["output"]).ref_words for r in rows
        )
        le = sum(indic_score(lattices[r["clip"]], r["output"]).word_edits for r in rows)
        lw = sum(indic_score(lattices[r["clip"]], r["output"]).ref_words for r in rows)
        rep.append(
            f"  {n}: native {100 * e / max(1, w):.1f} %, any form {100 * le / max(1, lw):.1f} % ({len(rows)} clips)"
        )

    # D. Parakeet for English, flex for Hindi, routed by flex's en/hi identification
    if "indic_flex_enhi_mixed" in en and "parakeet" in cand and "indic_flex_enhi" in hi:
        router_en = {r["clip"]: r for r in en["indic_flex_enhi_mixed"]["clips"]}
        pk = {r["clip"]: r for r in cand["parakeet"]["clips"]}
        pairs_routed, pairs_flex, wrong = [], [], 0
        for clip in en_labelled:
            r = router_en[clip]
            pairs_flex.append((r["truth"], r["output"]))
            if r.get("lang") == "en":
                pairs_routed.append((r["truth"], pk[clip]["output"]))
            else:
                wrong += 1
                pairs_routed.append((r["truth"], r["output"]))
        rh = hi["indic_flex_enhi"]["clips"]
        to_en = sum(r.get("lang") != "hi" for r in rh)
        rep += [
            "",
            "D. Router: flex identification restricted to en/hi (`indic_flex_enhi*`)",
            f"  English clips: routed to Hindi {wrong} of {len(en_labelled)}; English WER with Parakeet"
            f" on the English-routed clips and flex mixed output on the rest {100 * corpus_wer(pairs_routed):.1f} %;"
            f" flex mixed alone {100 * corpus_wer(pairs_flex):.1f} %",
            f"  Hindi clips: routed to English {to_en} of {len(rh)}",
        ]

    rep += ["", "E. Silence and noise (2 s)", ""]
    for name, res in noise.items():
        for r in res["clips"]:
            rep.append(f"  {name} {r['clip']} [{r.get('lang', '')}]: {r['output'][:160]!r}")

    text = "\n".join(rep) + "\n"
    (out / "multi_report.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
