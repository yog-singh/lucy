"""Label the enrolment clips (enrol_*.wav) automatically.

usage (from the repo root): uv run python checks/stt_accuracy/auto_enrol.py [--eval-dir DIR]

Each clip is the user reading one of the ten pool sentences in
src/lucy/voice/enrol.py, but which one is not stored. Each clip is transcribed with
today's recogniser (`parakeet`) and matched to the closest pool sentence by word error
rate; the pool sentence (not the transcript) becomes the truth. A clip is left
unlabelled when the match is ambiguous (see `match_enrolment` in stt_scoring.py).
Clips already in truth.jsonl are not touched. The match score printed is the WER of
the transcript against the chosen sentence, with the margin to the second best.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recognisers import CONFIGS
from stt_scoring import Truth, append_truth, load_truth, match_enrolment

from lucy.voice.audio import float_to_pcm, to_input_format
from lucy.voice.enrol import SENTENCES

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(REPO / "data" / "stt_eval"))
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    truth_file = eval_dir / "truth.jsonl"
    done = load_truth(truth_file)
    clips = [p for p in sorted(eval_dir.glob("enrol_*.wav")) if p.name not in done]
    if not clips:
        print("no unlabelled enrolment clips")
        return 0
    rec = CONFIGS["parakeet"]()
    rec.load()
    try:
        heard = {p.name: rec.transcribe(float_to_pcm(to_input_format(p))) for p in clips}
    finally:
        rec.close()
    for m in match_enrolment(heard, SENTENCES):
        status = f"-> {m.sentence!r}" if m.sentence else f"LEFT UNLABELLED ({m.why})"
        print(f"{m.clip}: WER {m.score:.2f}, margin {m.margin:.2f} {status}\n  heard: {m.heard}")
        if m.sentence:
            append_truth(truth_file, Truth(m.clip, m.sentence, source="enrol_auto", heard=m.heard))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
