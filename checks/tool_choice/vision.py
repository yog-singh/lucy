"""Tool choice for the vision tools (ADR 0026) with dynamic tool loading (ADR 0025).

usage (from the repo root): uv run python checks/tool_choice/vision.py --config lucy.toml
    --out DIR --label LABEL [--reps N] [--cases a,b] [--show-specs]
One user message per case (some after a scripted earlier exchange), each in a fresh
session, through the real app, prompt assembly, gateway and chat model, with the
config's enabled tools and MCP servers replaced by the stubs of `dynamic.py`, and
`look`, `enroll_face` and `replace_face` enabled. `[vision]` itself is switched off in
the copied settings, so no camera or face model is created, let alone opened; the stubs
carry the real names, descriptions and parameters. Only the model's first step is
scored: when it calls a tool, the turn is interrupted. No user name is known (fresh
database, learning off). `--show-specs` prints the specifications sent with the vision
group on, and exits. `--no-stale-filter` keeps stale exchanges in the prompt
(`lucy.core.toolchanges`), to measure what the filter changes. A `*_recorded` case
writes a `ToolsConfigured` record of the tools without `enroll_face` and `replace_face`
before its earlier exchanges. See checks/README.md.
"""

import argparse
import asyncio
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_CONFIG, fresh_db, isolated_settings, out_dir  # noqa: E402
from dynamic import Stub, StubProvider  # noqa: E402

VISION = {"look", "enroll_face", "replace_face"}
FACE_TOOLS = {"enroll_face", "replace_face"}
ASK = "ask"  # no tool; the reply asks for a name
NO_VISION = "no_vision"  # anything but a vision tool
PLACEHOLDER = re.compile(r"(?i)^\s*(me|user|my face|myself|the user|you|i)?\s*$")
ASKS_NAME = re.compile(r"(?i)\bname\b|naam|नाम|who should|whose|कौन")
REFUSE_FACE = "I cannot remember your face. I do not store any information about what you look like or any other personal details from our conversation."
REFUSE_SEE = "I can't see you. I don't have a camera or any way to see what is around you."
NO_RECOGNISE = "I can see you, but I cannot recognize who you are. My vision tool only describes what it sees in the moment and does not have the ability to identify specific individuals."
PROFILE = "Sam lives in Pune. Priya is Sam's sister. Sam prefers Lucy to speak in English."
SAM = ("enroll_face", "(?i)sam")

# case -> (category, earlier exchange [(user, assistant)], message, expectation)
# A "known" case has PROFILE as what Lucy knows about the user; the others know nothing.
# expectation: a tool name (its `name` argument must match `args`, if given), ASK or NO_VISION
CASES: dict[str, tuple[str, list[tuple[str, str]], str, Any]] = {
    "enrol_can_you": ("enrol", [], "Can you remember my face?", ASK),
    "enrol_remember": ("enrol", [], "Remember my face.", ASK),
    "enrol_learn": ("enrol", [], "Learn my face", ASK),
    "enrol_named": ("enrol", [], "My name is Asha, remember my face", ("enroll_face", "(?i)asha")),
    "enrol_hindi": ("enrol", [], "mera chehra yaad rakho", ASK),
    "enrol_after_refusal": ("history", [("Can you remember my face?", REFUSE_FACE)], "Remember my face.", ASK),
    "enrol_named_after_refusal": ("history", [("Remember my face.", REFUSE_FACE)], "I'm Ravi. Please remember my face.", ("enroll_face", "(?i)ravi")),
    # the user's name is known (a profile, as in a real session): call enroll_face with it
    "known_can_you": ("known", [], "Can you remember my face?", SAM),
    "known_remember": ("known", [], "Remember my face.", SAM),
    "known_learn": ("known", [], "Learn my face", SAM),
    "known_hindi": ("known", [], "mera chehra yaad rakho", SAM),
    "known_after_refusal": ("known", [("Can you look at me and recognize me?", NO_RECOGNISE), ("Can you remember my face?", REFUSE_FACE)], "Remember my face.", SAM),
    "known_after_memory_talk": ("known", [("Is this stored in your memory?", "No, I do not store this in my memory. I only have access to the information we have discussed in our current conversation."), ("Can you look at me and recognize me?", NO_RECOGNISE)], "No, remember my face.", SAM),
    "known_after_no_recognise": ("known", [("Can you look at me and recognize me?", NO_RECOGNISE)], "No, remember my face.", SAM),
    # As in a real session: the refusals were given while the face tools were not
    # configured, and the log records that (`ToolsConfigured`, ADR 0025 amendment
    # 2026-11-17), so the stale-exchange filter leaves them out of the prompt.
    "enrol_after_refusals_recorded": ("recorded", [("Can you remember my face?", REFUSE_FACE), ("Remember my face.", REFUSE_FACE), ("Please learn my face", REFUSE_FACE)], "Remember my face.", ASK),
    "known_after_refusals_recorded": ("known_recorded", [("Remember my face.", REFUSE_FACE), ("Can you remember my face?", REFUSE_FACE), ("Remember my face please.", REFUSE_FACE)], "Remember my face.", SAM),
    "look_holding": ("look", [], "what am I holding?", ("look", None)),
    "look_see_me": ("look", [], "can you see me?", ("look", None)),
    "look_who": ("look", [], "who is here?", ("look", None)),
    "look_hindi": ("look", [], "ye kya hai?", ("look", None)),
    "look_after_refusal": ("history", [("can you see me?", REFUSE_SEE)], "what am I holding?", ("look", None)),
    "ctl_let_me_see": ("control", [], "let me see", NO_VISION),
    "ctl_news": ("control", [], "look up the news", NO_VISION),
    "ctl_weather": ("control", [], "what's the weather", NO_VISION),
    "ctl_look_at_time": ("control", [], "Look at the time, I'm late! What time is it in London?", NO_VISION),
}  # fmt: skip


def score(expect: Any, calls: list[dict[str, Any]], reply: str) -> str:
    """pass, or why not: refused, asked_name (the name was known), wrong_tool, bad_name
    (a placeholder or the wrong name), made_up_name (no name was known)."""
    names = [c["name"] for c in calls if c["name"] != "more_tools"]
    if expect == NO_VISION:
        return "wrong_tool" if VISION & set(names) else "pass"
    if expect == ASK:
        if names:
            enrol = [c for c in calls if c["name"] == "enroll_face"]
            if enrol and all(PLACEHOLDER.match(str(c["args"].get("name", ""))) for c in enrol):
                return "bad_name"
            return "wrong_tool" if not enrol else "made_up_name"
        return "pass" if ASKS_NAME.search(reply) and "?" in reply else "refused"
    tool, args = expect
    hits = [c for c in calls if c["name"] == tool]
    if not hits:
        if names:
            return "wrong_tool"
        return "asked_name" if tool == "enroll_face" and ASKS_NAME.search(reply) else "refused"
    if args and not any(re.search(args, json.dumps(c["args"])) for c in hits):
        return "bad_name"
    return "pass"


def settings_for(args):
    db = fresh_db(out_dir("tool_choice_vision", args.out) / args.label)
    s = isolated_settings(args.config, db)
    enabled = [
        *s.tools.enabled,
        *(n for n in ("look", "enroll_face", "replace_face") if n not in s.tools.enabled),
    ]
    update: dict[str, Any] = {
        "tools": s.tools.model_copy(update={"dynamic": True, "enabled": enabled}),
        "learning": s.learning.model_copy(update={"enabled": False}),
        "vision": s.vision.model_copy(update={"enabled": False, "glance": False}),
    }
    if s.speakers is not None:
        update["speakers"] = s.speakers.model_copy(update={"enabled": False})
    return s.model_copy(update=update), db


async def first_step(app, sid: str, text: str, limit: float = 180) -> dict[str, Any]:
    from lucy.core.events import AssistantMessage, Interrupt, TurnEnded, UserMessage

    st: dict[str, Any] = {"reply": "", "calls": [], "error": None}

    async def run():
        async with app.bus.subscribe(AssistantMessage, TurnEnded) as evs:
            await app.bus.publish(UserMessage(session_id=sid, text=text, source="text"))
            first = True
            async for e in evs:
                if e.session_id != sid:
                    continue
                if isinstance(e, AssistantMessage) and first:
                    first = False
                    st["reply"] = e.text
                    st["calls"] = [{"name": c.name, "args": c.arguments} for c in e.tool_calls]
                    if e.tool_calls:  # score the first step only: stop the turn
                        await app.bus.publish(Interrupt(session_id=sid))
                elif isinstance(e, TurnEnded):
                    return

    try:
        await asyncio.wait_for(run(), limit)
    except TimeoutError:
        st["error"] = "timeout"
    return st


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", help="output directory (default: data/checks/tool_choice_vision/)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cases", help="comma-separated case names (default: all)")
    ap.add_argument("--show-specs", action="store_true")
    ap.add_argument("--no-stale-filter", action="store_true")
    args = ap.parse_args()
    if args.no_stale_filter:
        from lucy.core.toolchanges import ToolChanges

        ToolChanges.leave_out_stale = lambda self, episodes: (list(episodes), 0)  # type: ignore[method-assign]

    from lucy.app import LucyApp, build_selector
    from lucy.core.events import AssistantMessage, ToolsConfigured, UserMessage
    from lucy.tools import build_tools

    settings, db = settings_for(args)
    if args.show_specs:
        sel = build_selector(settings, build_tools(settings))
        assert sel is not None
        for spec in sel.specs({"vision"}):
            if spec.name in VISION:
                print(json.dumps(spec.model_dump(), indent=1, ensure_ascii=False))
        return
    names = args.cases.split(",") if args.cases else list(CASES)
    rows: list[dict[str, Any]] = []

    def tool_builder(s):
        return [Stub(t) for t in build_tools(s)]

    def provider_builder(s):
        return [StubProvider(s)]

    async with LucyApp(
        settings, background=False, tool_builder=tool_builder, provider_builder=provider_builder
    ) as app:
        store = app.memory
        retrieve = store.retrieve
        profile = {"text": ""}

        async def with_profile(query: str, budget: int):
            found = await retrieve(query, budget)
            return found.model_copy(update={"profile": profile["text"]})

        store.retrieve = with_profile  # type: ignore[method-assign]
        for rep in range(args.reps):
            for name in names:
                category, earlier, text, expect = CASES[name]
                sid = f"{args.label}-{name}-{rep}"
                profile["text"] = PROFILE if category.startswith("known") else ""
                if category.endswith("recorded"):
                    before = tuple(t.name for t in app._tools if t.name not in FACE_TOOLS)
                    await app.memory.append_episode(ToolsConfigured(session_id=sid, names=before))
                for user, assistant in earlier:
                    await app.memory.append_episode(
                        UserMessage(session_id=sid, text=user, source="text")
                    )
                    await app.memory.append_episode(
                        AssistantMessage(session_id=sid, text=assistant)
                    )
                st = await first_step(app, sid, text)
                outcome = score(expect, st["calls"], st["reply"]) if not st["error"] else "error"
                rows.append(
                    {
                        "case": name,
                        "category": category,
                        "rep": rep,
                        "q": text,
                        "outcome": outcome,
                        **st,
                    }
                )
                calls = " ".join(
                    f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)[:60]})"
                    for c in st["calls"]
                )
                print(
                    f"[{outcome:12}] {name}#{rep} q={text!r}\n      calls: {calls or '-'}\n      a={st['reply'].strip()[:160]!r}",
                    flush=True,
                )
    path = db.parent / "rows.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    by = defaultdict(list)
    for r in rows:
        by[r["case"]].append(r["outcome"])
    print(f"SUMMARY {args.label}")
    for name, outs in by.items():
        c = Counter(outs)
        print(f"  {name:26} {c['pass']}/{len(outs)}  {dict(c)}")
    total = sum(r["outcome"] == "pass" for r in rows)
    print(f"  ALL {total}/{len(rows)}\nrows: {path}")


if __name__ == "__main__":
    asyncio.run(main())
