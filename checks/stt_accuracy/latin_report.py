"""Report on the Latin-only flex configurations (`recognisers.LATIN`) and the junk guard.

Reads the runs `latin_en` (the user's clips, --compare), `latin_hi` (Monsoon Hindi)
and `latin_noise` (silence and noise) under data/checks/stt_accuracy/ and writes
data/checks/stt_accuracy/latin_report.txt: the results table, how numbers come out,
examples side by side, and the guard (`guard.py`) applied to every clip.

    uv run python checks/stt_accuracy/latin_report.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from guard import GUARD, clip_dbfs, collapse_repeats, junk_reason
from stt_scoring import load_truth, percentile, roman_score

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "checks" / "stt_accuracy"
EVAL = REPO / "data" / "stt_eval"
CONFIGS = [
    "flex_en_mixed",
    "flex_en_roman",
    "flex_enhi_roman",
    "flex_hi_roman",
    "flex_enhi_mixed_latin",
    "flex_hi_mixed_latin",
]
NUMBER_WORDS = set(
    "one two three four five six seven eight nine ten eleven twelve thirteen fifteen twenty "
    "thirty forty fifty hundred thousand ek do teen char chaar paanch panch chhe chhah saat "
    "aath nau das bees pachaas sau hazaar hazar lakh".split()
)
# The user's own Hindi clips (skipped in truth.jsonl as not English) and hard English ones.
USER_HINDI = ["6.570206Z_84fbc3ca", "4.063735Z_4eda910e"]
HARD_EN = [
    "Yogesh",
    "Dario Amodei",
    "Accel investors",
    "look up Accel",
    "unread emails",
    "What time is it?",
    "Hey Jarvis, what time",
]


def load(label: str, config: str) -> dict | None:
    p = OUT / label / config / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def numbers(clips: list[dict]) -> str:
    digits = sum(bool(re.search(r"\d", c["output"])) for c in clips)
    words = sum(bool(NUMBER_WORDS & set(re.findall(r"[a-z]+", c["output"].lower()))) for c in clips)
    return f"{digits} with digits, {words} with number words"


def main() -> int:
    lines: list[str] = []
    w = lines.append
    runs = {c: (load("latin_en", c), load("latin_hi", c), load("latin_noise", c)) for c in CONFIGS}
    # re-score Hindi with the current fold, so the table never mixes fold versions
    lattices = {
        k: t.lattice for k, t in load_truth(EVAL / "public" / "monsoon_hi" / "truth.jsonl").items()
    }
    for _, hi, _ in runs.values():
        if hi is None:
            continue
        for r in hi["clips"]:
            f = roman_score(lattices[r["clip"]], r["output"])
            u = roman_score(lattices[r["clip"]], r["output"], fold=False)
            g = roman_score(lattices[r["clip"]], collapse_repeats(r["output"]))
            r.update(
                roman_guard_edits=g.word_edits,
                roman_edits=f.word_edits,
                roman_words=f.ref_words,
                roman_char_edits=f.char_edits,
                roman_chars=f.ref_chars,
                roman_plain_edits=u.word_edits,
                roman_plain_words=u.ref_words,
            )
        cs = hi["clips"]
        hi["summary"].update(
            wer_roman=sum(r["roman_edits"] for r in cs) / sum(r["roman_words"] for r in cs),
            cer_roman=sum(r["roman_char_edits"] for r in cs) / sum(r["roman_chars"] for r in cs),
            wer_roman_unfolded=sum(r["roman_plain_edits"] for r in cs)
            / sum(r["roman_plain_words"] for r in cs),
            wer_roman_guard=sum(r["roman_guard_edits"] for r in cs)
            / sum(r["roman_words"] for r in cs),
        )
    w("Latin-only flex configurations (float16, MPS, greedy). WER/CER in %, times s/clip.")
    w("")
    w(
        f"{'config':24} {'EN WER':>6} {'<2s':>5} {'exact':>5} {'EN s mean/p95':>14} "
        f"{'HI WERf':>7} {'+rep':>5} {'CERf':>5} {'WERu':>5} {'exact':>5} {'HI s mean/p95':>14} "
        f"{'nonLatin EN/HI':>14}"
    )
    for c, (en, hi, _) in runs.items():
        if en is None or hi is None:
            w(f"{c:24} (missing)")
            continue
        se, sh = en["summary"], hi["summary"]
        hi_exact = sum(r["roman_edits"] == 0 for r in hi["clips"]) / len(hi["clips"])
        w(
            f"{c:24} {pct(se['wer']):>6} {pct(se['wer_by_length']['<2s']):>5} "
            f"{pct(se['exact_rate']):>5} {se['mean_s']:6.3f}/{se['p95_s']:.3f} "
            f"{pct(sh['wer_roman']):>7} {pct(sh['wer_roman_guard']):>5} "
            f"{pct(sh['cer_roman']):>5} "
            f"{pct(sh['wer_roman_unfolded']):>5} {pct(hi_exact):>5} "
            f"{sh['mean_s']:6.3f}/{sh['p95_s']:.3f} "
            f"{se['clips_with_non_latin']:>6}/{sh['clips_with_non_latin']}"
        )
    w("")
    w("WERf/CERf: romanised and folded (translit.fold); WERu: romanised, not folded;")
    w("+rep: WERf after guard.collapse_repeats (a decoding loop kept once).")
    w("nonLatin: clips whose output has any non-Latin letter (EN run includes skipped clips).")
    w("")
    w("Language chosen (identification configurations), and numbers:")
    for c, (en, hi, _) in runs.items():
        if en is None or hi is None:
            continue
        le = Counter(r.get("lang") for r in en["clips"] if r["truth"] is not None)
        lh = Counter(r.get("lang") for r in hi["clips"])
        w(f"  {c}: EN {dict(le)}; HI {dict(lh)}")
        w(f"    numbers: EN {numbers(en['clips'])}; HI {numbers(hi['clips'])}")
    w("")

    # examples
    en_rows = {c: {r["clip"]: r for r in (runs[c][0] or {"clips": []})["clips"]} for c in CONFIGS}
    hi_rows = {c: {r["clip"]: r for r in (runs[c][1] or {"clips": []})["clips"]} for c in CONFIGS}
    base_en = en_rows[CONFIGS[0]]
    base_hi = hi_rows[CONFIGS[0]]

    def block(title: str, rows: dict, clip: str, ref: str | None) -> None:
        r0 = rows[CONFIGS[0]][clip]
        w(f"--- {title} {clip[-44:]} ({r0['audio_s']:.1f} s)")
        w(f"  {'reference':22} {ref}")
        for c in CONFIGS:
            r = rows[c].get(clip)
            if r is not None:
                lang = f" [{r['lang']}]" if "allowed" in c or "enhi" in c else ""
                w(f"  {c:22} {r['output']}{lang}")

    w("EXAMPLES: Hindi and mixed")
    for key in USER_HINDI:
        clip = next((k for k in base_en if key in k), None)
        if clip:
            block("user Hindi", en_rows, clip, "(not labelled)")
    for clip in sorted(base_hi)[:10]:
        block("Monsoon", hi_rows, clip, base_hi[clip]["truth"])
    w("")
    w("EXAMPLES: English")
    seen = []
    for key in HARD_EN:
        clip = next(
            (k for k, r in base_en.items() if r["truth"] and key in r["truth"] and k not in seen),
            None,
        )
        if clip:
            seen.append(clip)
            block("English", en_rows, clip, base_en[clip]["truth"])
    for clip in sorted(base_en):
        if len(seen) >= 8:
            break
        if base_en[clip]["truth"] and clip not in seen and base_en[clip]["audio_s"] < 2:
            seen.append(clip)
            block("English", en_rows, clip, base_en[clip]["truth"])
    w("")

    # guard
    w(f"GUARD {GUARD}")
    for c, (en, hi, noise) in runs.items():
        if en is None or hi is None or noise is None:
            continue
        drops = []
        for label, res, eval_dir in (
            ("en", en, EVAL),
            ("hi", hi, EVAL / "public" / "monsoon_hi"),
            ("noise", noise, EVAL / "public" / "noise"),
        ):
            for r in res["clips"]:
                why = junk_reason(
                    r["output"], r["audio_s"], r.get("extra", {}), clip_dbfs(eval_dir / r["clip"])
                )
                kind = "real" if r["truth"] is not None else "unlabelled"
                if label == "noise":
                    kind = "noise"
                drops.append((label, kind, r, why))
        real = [d for d in drops if d[1] == "real"]
        false = [d for d in real if d[3]]
        w(
            f"  {c}: real clips dropped {len(false)} of {len(real)}; noise dropped "
            f"{sum(bool(d[3]) for d in drops if d[1] == 'noise')} of "
            f"{sum(d[1] == 'noise' for d in drops)}; unlabelled dropped "
            f"{sum(bool(d[3]) for d in drops if d[1] == 'unlabelled')} of "
            f"{sum(d[1] == 'unlabelled' for d in drops)}"
        )
        for label, kind, r, why in drops:
            if why or kind != "real":
                ex = r.get("extra", {})
                w(
                    f"    {label:5} {kind:10} {r['audio_s']:5.1f}s tok {ex.get('tokens')} "
                    f"lp {ex.get('mean_logprob', 0):.2f} ns {ex.get('nospeech_logprob') or 0:.1f} "
                    f"-> {'DROP ' + why if why else 'keep'}: {r['output'][:70]!r}"
                )
        lp = [r.get("extra", {}).get("mean_logprob", 0) for _, k, r, _ in real]
        tps = [r.get("extra", {}).get("tokens", 0) / max(r["audio_s"], 0.1) for _, k, r, _ in real]
        w(
            f"    real clips: mean logprob min {min(lp):.2f}, p5 {percentile(lp, 5):.2f}; "
            f"tokens/s max {max(tps):.1f}, p99 {percentile(tps, 99):.1f}"
        )
    text = "\n".join(lines) + "\n"
    (OUT / "latin_report.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
