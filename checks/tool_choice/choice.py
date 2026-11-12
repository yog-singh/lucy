"""25 turns in one session against the real app, live tools; scores tool choice and persona.

usage (from the repo root): uv run python checks/tool_choice/choice.py [--config FILE]
    [--label LABEL] [--out DIR] [--turns N]
The app runs on a fresh database in OUT/LABEL/ (default data/checks/tool_choice/run/).
See checks/README.md.
"""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

from lucy.app import LucyApp
from lucy.core.events import TextDelta, ToolStarted, TurnEnded, UserMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DEFAULT_CONFIG, fresh_db, isolated_settings, out_dir

TURNS = [  # (message, expectation: T lookup, N none, A either, keywords that must appear)
    ("Hi Lucy, how is your day going?", "N", []),
    ("My sister Ana lives in Lyon.", "N", []),
    ("I work as a nurse at a children's hospital.", "N", []),
    ("What is 17 times 23?", "N", ["391"]),
    ("What is the capital of France?", "N", ["paris"]),
    ("Do you think cats or dogs make better pets?", "N", []),
    ("How tall is Mount Kilimanjaro?", "T", []),
    ("Who was the first person to reach its summit?", "T", []),
    ("Who is Kristrún Frostadóttir?", "T", []),
    ("What is the Voynich manuscript?", "A", []),
    ("Who won the most recent Formula 1 Grand Prix?", "T", []),
    ("What is the current price of Bitcoin in US dollars?", "T", []),
    ("Where does my sister live?", "N", ["lyon"]),
    ("What do I do for work?", "N", ["nurse"]),
    ("What is the population of Iceland?", "T", []),
    ("At what temperature does water boil at sea level in Fahrenheit?", "N", ["212"]),
    ("What was the main topic of the latest Apple product announcement?", "T", []),
    ("What's the weather in Lisbon today?", "T", []),
    ("Tell me a joke.", "N", []),
    ("What is 144 divided by 12?", "N", ["12"]),
    ("Who is the current CEO of Apple?", "T", ["ternus"]),
    ("What's today's date?", "N", ["12"]),
    ("Can you remind me what my sister's name is?", "N", ["ana"]),
    ("What time is it in Tokyo right now?", "A", []),
    ("Thanks, that's all for now.", "N", []),
]
CLAIM = re.compile(
    r"\b(i(?:'ve| have| will|'ll)? (?:noted|saved|stored|made a note|added|remember(?:ed)? that)|noted that|i'll keep that in mind)",
    re.I,
)


async def turn(app, sid, text):
    out = ""
    tools = []
    t0 = time.monotonic()
    first = None
    async with app.bus.subscribe(TextDelta, ToolStarted, TurnEnded) as events:
        await app.bus.publish(UserMessage(session_id=sid, text=text, source="text"))
        async for e in events:
            if e.session_id != sid:
                continue
            if isinstance(e, TextDelta):
                out += e.text
                first = first or time.monotonic() - t0
            elif isinstance(e, ToolStarted):
                tools.append(f"{e.call.name}:{json.dumps(e.call.arguments)[:60]}")
            elif isinstance(e, TurnEnded):
                return out, tools, e.reason, time.monotonic() - t0, first


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", help="output directory (default: data/checks/tool_choice/)")
    ap.add_argument("--turns", type=int, help="run only the first N turns (smoke run)")
    args = ap.parse_args()
    tag = args.label
    db = fresh_db(out_dir("tool_choice", args.out) / tag)
    turns = TURNS[: args.turns] if args.turns else TURNS
    ok_choice = ok_kw = claims = 0
    ft_t = []
    ft_n = []
    async with LucyApp(isolated_settings(args.config, db)) as app:
        for text, exp, kws in turns:
            out, tools, reason, dt, first = await turn(app, tag, text)
            (ft_t if tools else ft_n).append(first or dt)
            choice = exp == "A" or (exp == "T") == bool(tools)
            kw = all(k in out.lower() for k in kws)
            claim = bool(CLAIM.search(out))
            ok_choice += choice
            ok_kw += kw
            claims += claim
            flag = ("." if choice else "C") + ("." if kw else "K") + ("M" if claim else ".")
            print(
                f"[{flag}] {exp} {dt:5.1f}s first={first or -1:4.1f}s tools={tools} q={text!r}\n      a={out!r}",
                flush=True,
            )
    print(
        f"SUMMARY {tag}: tool choice {ok_choice}/{len(turns)}, expected content {ok_kw}/{len(turns)}, memory-action claims {claims}"
    )
    md = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")
    print(
        f"FIRSTWORD {tag}: lookup turns median {md(ft_t):.2f}s (n={len(ft_t)}), ordinary turns median {md(ft_n):.2f}s (n={len(ft_n)})"
    )


if __name__ == "__main__":
    asyncio.run(main())
