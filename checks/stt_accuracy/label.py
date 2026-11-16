"""Write the correct text of each recorded clip, for the accuracy check.

usage (from the repo root): uv run python checks/stt_accuracy/label.py [--eval-dir DIR]

For each clip not yet in DIR/truth.jsonl (default DIR: data/stt_eval/): plays it
(afplay), shows what Lucy heard live, and asks:

  Enter        the heard text is correct
  any text     the correct text (replaces the heard text)
  r            replay
  s            skip: not the user speaking, TV, unusable (then an optional reason)
  q            quit (everything answered so far is saved)

Each answer is appended to truth.jsonl at once, so quitting and running again carries
on where it stopped. To redo a clip, delete its line from truth.jsonl. See
checks/README.md.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stt_scoring import Truth, append_truth, list_clips, load_truth

REPO = Path(__file__).resolve().parents[2]


def play(path: Path) -> None:
    try:
        subprocess.run(["afplay", str(path)], check=False)
    except KeyboardInterrupt:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(REPO / "data" / "stt_eval"))
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    truth_file = eval_dir / "truth.jsonl"
    done = load_truth(truth_file)
    todo = [c for c in list_clips(eval_dir) if c.clip not in done]
    print(
        f"{len(done)} labelled, {len(todo)} to go. Enter=correct, text=fix, r=replay, s=skip, q=quit"
    )
    for n, clip in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] {clip.clip}")
        heard = clip.heard
        print(f"  heard: {heard if heard is not None else '(no live transcript)'}")
        play(eval_dir / clip.clip)
        while True:
            try:
                answer = input("  > ").strip()
            except EOFError:
                return 0
            if answer == "q":
                return 0
            if answer == "r":
                play(eval_dir / clip.clip)
                continue
            if answer == "s":
                reason = input("  reason (optional): ").strip()
                entry = Truth(clip.clip, None, skipped=True, reason=reason, heard=heard)
            elif answer == "":
                if not heard:
                    print("  nothing was heard: type the text, or s to skip")
                    continue
                entry = Truth(clip.clip, heard, heard=heard)
            else:
                entry = Truth(clip.clip, answer, heard=heard)
            append_truth(truth_file, entry)
            break
    print("\nall clips labelled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
