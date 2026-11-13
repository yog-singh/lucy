"""How Lucy's replies end: offers of help, follow-up menus, other questions or none.

usage (from the repo root): uv run python checks/endings/endings.py [--config FILE]
    [--label LABEL] [--out DIR] [--turns N]
One session of turns that provoked closing offers in real use (greetings, small talk,
a joke, a space conversation, short facts, thanks, an ambiguous request), on a fresh
database in OUT/LABEL/ (default data/checks/endings/run/). See checks/README.md.
"""

import argparse
import asyncio
import re
import sys
from collections import Counter
from pathlib import Path

from lucy.app import LucyApp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DEFAULT_CONFIG, fresh_db, isolated_settings, out_dir
from tool_choice.choice import turn

TURNS = [  # mostly real user messages from the log; "ok" marks where a question is fine
    ("Hey Lucy, how are you?", "ok"),
    ("I'm doing good, thanks for asking.", ""),
    ("Nothing that I need help with right now.", ""),
    ("Why don't you throw in some jokes?", ""),
    ("Hmm, nice one.", ""),
    ("Let's talk about something else. Should we talk about the cosmos?", "ok"),
    ("Yeah, space exploration, black holes and exoplanets do interest me.", "ok"),
    ("How do astronomers find exoplanets?", ""),
    ("What would happen if you fell into a black hole?", ""),
    ("How big is the biggest black hole we know of?", ""),
    ("Thank you.", ""),
    ("What is the capital of Japan?", ""),
    ("How many legs does a spider have?", ""),
    ("Cool, thank you.", ""),
    ("Can you tell me more about him?", "ok"),  # nobody to refer to: should ask
    ("How are you doing?", "ok"),
    ("I'm doing good.", ""),
    ("Thanks, that's all for now.", ""),
]
OFFER = re.compile(
    r"(anything else|anything) (?:\w+ )?(?:i can|can i|i could) (?:help|do|assist)|how can i (?:help|assist)"
    r"|what can i (?:help|do for)|(?:let me know|feel free)\b.*\b(?:help|need|anything)|here (?:if|whenever) you need|feel free to reach out"
    r"|anything (?:else )?(?:on your mind|you need)|what(?:'s| is) on your mind|(?:help|assist) you with",
    re.I,
)
MENU = re.compile(
    r"(?:would|do) you (?:like|want|prefer)|(?:anything|something)(?: else| specific| in particular)? you (?:would|'?d) like|shall (?:we|i)|should we|anything else you'?d like"
    r"|(?:are you )?interested in (?:hearing|learning|knowing)|are you (?:more )?(?:interested|curious)|want (?:to hear|me to)|or (?:would you )?prefer"
    r"|what else would you like|is there (?:a|any)\w* (?:particular|specific)",
    re.I,
)


def last_sentence(text: str) -> str:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    return parts[-1] if parts else ""


def classify(text: str) -> str:
    last = last_sentence(text)
    if OFFER.search(last):
        return "offer"
    if last.rstrip("\"' )").endswith("?"):
        return "menu" if MENU.search(last) else "question"
    return "none"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", help="output directory (default: data/checks/endings/)")
    ap.add_argument("--turns", type=int, help="run only the first N turns (smoke run)")
    args = ap.parse_args()
    tag = args.label
    db = fresh_db(out_dir("endings", args.out) / tag)
    turns = TURNS[: args.turns] if args.turns else TURNS
    counts = Counter()
    async with LucyApp(isolated_settings(args.config, db)) as app:
        for text, ok in turns:
            out, tools, reason, dt, first = await turn(app, tag, text)
            kind = classify(out)
            counts[kind] += 1
            flag = {"offer": "O", "menu": "M", "question": "?", "none": "."}[kind]
            print(
                f"[{flag}{'*' if ok else ' '}] {dt:5.1f}s tools={tools} q={text!r}\n      a={out!r}",
                flush=True,
            )
    print(
        f"SUMMARY {tag}: {len(turns)} replies; offer {counts['offer']}, menu {counts['menu']}, "
        f"other question {counts['question']}, no question {counts['none']}"
    )


if __name__ == "__main__":
    asyncio.run(main())
