"""Speaker match on short clips: the library's current path against fixes done before it.

Scores every clip against the enrolled book (read only) several ways, buckets the
results by speech length (after the library's own energy VAD) and reports how many of
the owner's clips pass and how many impostor clips (other speakers) wrongly pass.

    uv run python checks/speaker_short/run.py
    uv run python checks/speaker_short/run.py --limit-impostors 30   # quicker

Positives: data/workspace/utterances/*.wav and data/stt_eval/clips/*.wav (the owner),
plus cuts of the owner's longer clips to short lengths. Impostors: the public Hindi
clips in data/stt_eval/public/monsoon_hi/clips/, full length and cut to the same short
lengths, and the noise clips. Enrolment clips (data/stt_eval/enrol_*.wav) are not
scored: they are in the book. Nothing is written except under data/checks/speaker_short/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import REPO, out_dir  # noqa: E402

SR = 16000
WINDOW = 3 * SR
BUCKETS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 1e9)]
BUCKET_NAMES = ["<1 s", "1-2 s", "2-3 s", "3-5 s", ">5 s"]
CUTS_S = [0.7, 1.2, 1.7, 2.5, 4.0]


def tile(wav: np.ndarray, length: int = WINDOW) -> np.ndarray:
    if len(wav) >= length or len(wav) == 0:
        return wav
    return np.tile(wav, -(-length // len(wav)))[:length]


class Methods:
    def __init__(self, weights: Path, book: Path) -> None:
        from voicefingerprint import Config, EncoderConfig, VoiceRecognizer
        from voicefingerprint.audio import normalize_volume
        from voicefingerprint.vad import EnergyVAD, apply_vad

        self.rec = VoiceRecognizer(config=Config(encoder=EncoderConfig(weights=str(weights))))
        self.rec.load(book)  # reads only
        self.enc = self.rec.encoder
        self.names, self.centroids = self.rec.book.centroids()
        self._norm = lambda w: normalize_volume(w, -26.0)
        self._vad = lambda w: apply_vad(w, EnergyVAD(), SR)

    def speech(self, raw: np.ndarray) -> np.ndarray:
        """What the library embeds today: levelled, then energy-VAD trimmed."""
        return self._vad(self._norm(raw))

    def _score(self, emb: np.ndarray) -> float:
        from voicefingerprint.scoring import cosine_matrix

        return float(cosine_matrix(emb[None, :], self.centroids)[0].max())

    def nopad(self, wav: np.ndarray) -> np.ndarray:
        if len(wav) >= WINDOW:
            return self.enc.embed_utterance(wav)
        from voicefingerprint.scoring import l2_normalize

        return l2_normalize(self.enc._run(self.enc.features(wav)[None])[0])

    def all(self, raw: np.ndarray) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        sp = self.speech(raw)
        variants = {
            "current": lambda: self.enc.embed_utterance(sp),
            "tile_raw": lambda: self.enc.embed_utterance(self.speech(tile(raw))),
            "trim_tile": lambda: self.enc.embed_utterance(tile(sp)),
            "novad_tile": lambda: self.enc.embed_utterance(
                tile(self._norm(raw)) if len(sp) < WINDOW else sp
            ),
            "trim_nopad": lambda: self.nopad(sp),
        }
        for name, fn in variants.items():
            t = time.perf_counter()
            score = self._score(fn())
            out[name] = (score, time.perf_counter() - t)
        return out


def load(path: Path) -> np.ndarray:
    from voicefingerprint.audio import load_audio

    return load_audio(path, SR)


def middle_cut(raw: np.ndarray, seconds: float) -> np.ndarray | None:
    n = int(seconds * SR)
    if len(raw) < n + SR // 2:
        return None
    start = (len(raw) - n) // 2
    return raw[start : start + n]


def bucket(seconds: float) -> int:
    return next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= seconds < hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights", default=str(REPO / "data/models/speaker/voxceleb_resnet34_LM.onnx")
    )
    ap.add_argument("--book", default=str(REPO / "data/workspace/speakers/book.npz"))
    ap.add_argument("--out")
    ap.add_argument("--label", default="run")
    ap.add_argument("--limit-impostors", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--report", metavar="ROWS", help="re-print the tables from a rows file")
    args = ap.parse_args()
    if args.report:
        report(json.loads(Path(args.report).read_text()), args.threshold)
        return
    out = out_dir("speaker_short", args.out)

    m = Methods(Path(args.weights), Path(args.book))
    owner = sorted((REPO / "data/workspace/utterances").glob("*.wav")) + sorted(
        (REPO / "data/stt_eval/clips").glob("*.wav")
    )
    impostor = sorted((REPO / "data/stt_eval/public/monsoon_hi/clips").glob("*.wav"))
    if args.limit_impostors:
        impostor = impostor[: args.limit_impostors]
    noise = sorted((REPO / "data/stt_eval/public/noise/clips").glob("*.wav"))
    truth = (REPO / "data/stt_eval/truth.jsonl").read_text().splitlines()
    skipped = {Path(d["clip"]).name for d in map(json.loads, truth) if d.get("skipped")}

    rows: list[dict] = []

    def add(kind: str, source: str, raw: np.ndarray) -> None:
        try:
            sp = m.speech(raw)
            scores = m.all(raw)
        except Exception as exc:  # e.g. silence: nothing left after VAD
            rows.append({"kind": kind, "source": source, "error": str(exc)})
            return
        rows.append(
            {
                "kind": kind,
                "source": source,
                "audio_s": len(raw) / SR,
                "speech_s": len(sp) / SR,
                **{f"{k}": v[0] for k, v in scores.items()},
                **{f"{k}_ms": v[1] * 1000 for k, v in scores.items()},
            }
        )

    for p in owner:
        raw = load(p)
        if p.name in skipped:  # labelled as not a clean owner clip (TV in the background)
            add("skipped", p.name, raw)
            continue
        add("owner", p.name, raw)
        if len(raw) / SR >= 3.0:
            for c in CUTS_S:
                cut = middle_cut(raw, c)
                if cut is not None:
                    add("owner_cut", f"{p.name}@{c}", cut)
    for p in impostor:
        raw = load(p)
        add("impostor", p.name, raw)
        for c in CUTS_S:
            cut = middle_cut(raw, c)
            if cut is not None:
                add("impostor_cut", f"{p.name}@{c}", cut)
    for p in noise:
        add("noise", p.name, load(p))

    (out / f"rows_{args.label}.json").write_text(json.dumps(rows, indent=1))
    report(rows, args.threshold)


METHODS = ["current", "tile_raw", "trim_tile", "novad_tile", "trim_nopad"]


def report(rows: list[dict], thr: float) -> None:
    ok = [r for r in rows if "error" in r]
    for r in ok:
        print(f"error: {r['kind']} {r['source']}: {r['error']}")
    rows = [r for r in rows if "error" not in r]
    groups = {
        "owner (real clips)": ["owner"],
        "owner cuts": ["owner_cut"],
        "impostor (full + cuts)": ["impostor", "impostor_cut"],
    }
    for title, kinds in groups.items():
        sel = [r for r in rows if r["kind"] in kinds]
        print(f"\n## {title}: pass at {thr:.2f} (median score) by speech length")
        print("| method | " + " | ".join(BUCKET_NAMES) + " | all |")
        print("|---|" + "---|" * (len(BUCKETS) + 1))
        for meth in METHODS:
            cells = []
            for b in [*range(len(BUCKETS)), None]:
                s = [r[meth] for r in sel if b is None or bucket(r["speech_s"]) == b]
                if not s:
                    cells.append("-")
                    continue
                n = sum(x >= thr for x in s)
                cells.append(f"{n}/{len(s)} ({np.median(s):.2f})")
            print(f"| {meth} | " + " | ".join(cells) + " |")
    noise = [r for r in rows if r["kind"] in ("noise", "skipped")]
    for r in noise:
        print(
            f"{r['kind']} {r['source'][:24]} ({r['speech_s']:.1f} s): "
            + ", ".join(f"{k}={r[k]:.2f}" for k in METHODS)
        )

    print("\n## short clips (speech < 3 s): owner (real + cuts) vs impostor (full + cuts)")
    pos_all = [r for r in rows if r["kind"] in ("owner", "owner_cut")]
    neg_all = [r for r in rows if r["kind"] in ("impostor", "impostor_cut")]
    for label, cond in [("< 3 s", lambda r: r["speech_s"] < 3), ("all", lambda r: True)]:
        pos = [r for r in pos_all if cond(r)]
        neg = [r for r in neg_all if cond(r)]
        print(f"\n### {label}: {len(pos)} owner, {len(neg)} impostor")
        print(
            f"| method | FRR@{thr:.2f} | FAR@{thr:.2f} | max impostor | EER | EER thr | FRR@thr(FAR=0) | thr(FAR=0) |"
        )
        print("|---|---|---|---|---|---|---|---|")
        for meth in METHODS:
            p = np.array([r[meth] for r in pos])
            n = np.array([r[meth] for r in neg])
            frr = (p < thr).mean()
            far = (n >= thr).mean()
            ts = np.arange(0.0, 0.9, 0.005)
            gaps = [abs((p < t).mean() - (n >= t).mean()) for t in ts]
            i = int(np.argmin(gaps))
            eer = ((p < ts[i]).mean() + (n >= ts[i]).mean()) / 2
            t0 = n.max() + 1e-3
            print(
                f"| {meth} | {frr:.1%} | {far:.1%} | {n.max():.2f} | {eer:.1%} | {ts[i]:.3f} "
                f"| {(p < t0).mean():.1%} | {t0:.2f} |"
            )
    print("\n## time per clip (median ms)")
    for meth in METHODS:
        short = [r[f"{meth}_ms"] for r in rows if r["speech_s"] < 3]
        longc = [r[f"{meth}_ms"] for r in rows if r["speech_s"] >= 3]
        print(f"{meth}: short {np.median(short):.0f} ms, long {np.median(longc):.0f} ms")


if __name__ == "__main__":
    main()
