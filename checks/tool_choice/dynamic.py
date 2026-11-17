"""Tool choice with dynamic tool loading (ADR 0025) against the feature-off baseline.

usage (from the repo root): uv run python checks/tool_choice/dynamic.py --config lucy.toml
    --out DIR --label LABEL (--dynamic | --static) [--reps N] [--cases a,b]
       uv run python checks/tool_choice/dynamic.py --summarise ROWS.json [ROWS.json ...]
Short scripted conversations, each in a fresh session, through the real app, prompt
assembly, gateway and chat model, with the config's enabled tools and MCP servers,
every one a stub: the real name, description, parameters, risk and grouping, but a fixed
result (nothing is looked up, added, set or read; a dangerous stub has a fixed preview,
so confirmations are asked as usual). MCP tools come from the lock file; no server is
started. Learning and speaker recognition are off, the database is fresh in OUT/LABEL/.
`--dynamic`/`--static` set `[tools] dynamic`; `--keep-turns` and `--core` override.
Each turn is scored against the tools it should call (`score_turn`). See checks/README.md.
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DEFAULT_CONFIG, fresh_db, isolated_settings, out_dir  # noqa: E402

META = "more_tools"
CANT = "cant"  # a request Lucy has no tool for: no action tool, at most one more_tools
ANY = "any"  # anything goes (the turn after a confirmation question)
CAL, ADD, EDIT, DEL = (
    "calendar",
    "add_calendar_event",
    "edit_calendar_event",
    "delete_calendar_event",
)
REM, SETR, TIMER, CANCEL = "reminders", "set_reminder", "set_timer", "cancel_reminder"
SEARCH, READ, THREAD = "search_email", "read_email", "read_email_thread"
FILLER = ["Tell me a joke.", "Ha, nice.", "How are you today?", "What is 9 times 7?", "Cool."]


def T(*tools: str, args: str | None = None) -> dict[str, Any]:
    """Expect one of `tools` called, its arguments (JSON) matching regex `args`."""
    return {"tools": list(tools), "args": args}


# case -> (category, [(user text, expectation)]); expectation: T(...), None, CANT or ANY.
CASES: dict[str, tuple[str, list[tuple[str, Any]]]] = {
    # core tools
    "core_weather": ("core", [("What's the weather in Lisbon today?", T("weather"))]),
    "core_rain": ("core", [("Will it rain in Delhi tomorrow?", T("weather"))]),
    "core_time": ("core", [("What time is it in Tokyo right now?", T("time"))]),
    "core_web": ("core", [("Who won the most recent Formula 1 Grand Prix?", T("web_search", "fetch_page", "wikipedia"))]),
    "core_wiki": ("core", [("Tell me about the history of the Eiffel Tower.", T("wikipedia", "web_search"))]),
    "core_hi_weather": ("core", [("Kal Mumbai ka mausam kaisa rahega?", T("weather"))]),
    # a group whose keywords match (English)
    "kw_cal_list": ("keyword", [("What's on my calendar tomorrow?", T(CAL))]),
    "kw_cal_add": ("keyword", [("Add a dentist appointment on Friday at 4 pm.", T(ADD, args="(?i)dentist"))]),
    "kw_reminder": ("keyword", [("Remind me at 6 pm to call mum.", T(SETR, args="(?i)mum|mom|mother"))]),
    "kw_timer": ("keyword", [("Set a timer for ten minutes.", T(TIMER, SETR, args="10|600|ten"))]),
    "kw_cancel_rem": ("keyword", [("Cancel my reminder to call the plumber.", T(CANCEL))]),
    "kw_email_search": ("keyword", [("Search my email for the electricity bill.", T(SEARCH, args="(?i)electric"))]),
    "kw_email_read": ("keyword", [("Read me the latest email from Amazon.", T(READ, THREAD, SEARCH, args="(?i)amazon|m1"))]),
    "kw_voice": ("keyword", [("Please enroll my wife Priya's voice.", T("enroll_voice", args="(?i)priya"))]),
    # the same needs with no keyword: only more_tools reaches them
    "meta_cal_got_on": ("no-keyword", [("What have I got on tomorrow?", T(CAL))]),
    "meta_cal_clear": ("no-keyword", [("Is my Thursday morning clear?", T(CAL))]),
    "meta_cal_pencil": ("no-keyword", [("Pencil in lunch with Ana on Saturday at one.", T(ADD, args="(?i)ana|lunch"))]),
    "meta_ping": ("no-keyword", [("Ping me in ten minutes to check the oven.", T(SETR, TIMER, args="10|600|ten|oven"))]),
    "meta_wake": ("no-keyword", [("Wake me at six tomorrow.", T(SETR, TIMER, args="6|06|six"))]),
    "meta_nudge": ("no-keyword", [("What have you promised to nudge me about?", T(REM))]),
    "meta_bank": ("no-keyword", [("Anything from the bank this week?", T(SEARCH, args="(?i)bank"))]),
    "meta_amazon": ("no-keyword", [("Did Amazon write back to me?", T(SEARCH, READ, args="(?i)amazon"))]),
    "meta_recognise": ("no-keyword", [("I want you to recognise my brother Ravi when he talks.", T("enroll_voice", args="(?i)ravi"))]),
    # romanised Hindi
    "hi_schedule": ("hindi", [("Kal mera schedule kya hai?", T(CAL))]),
    "hi_free": ("hindi", [("Kal shaam main free hoon kya?", T(CAL))]),
    "hi_add": ("hindi", [("Parso shaam paanch baje doctor ke paas jaana hai, calendar mein daal do.", T(ADD, args="(?i)doctor"))]),
    "hi_remind": ("hindi", [("Das minute baad yaad dila dena ki chai bana rahi hoon.", T(SETR, TIMER, args="10|600|chai|tea"))]),
    "hi_wake": ("hindi", [("Subah chhe baje utha dena.", T(SETR, TIMER, args="6|06|six"))]),
    "hi_mail": ("hindi", [("Koi nayi mail aayi kya?", T(SEARCH))]),
    "hi_bank": ("hindi", [("Bank se koi message aaya kya is hafte?", T(SEARCH, args="(?i)bank"))]),
    # follow-ups after the group was switched on
    "fu_calendar": ("follow-up", [("What's on my calendar tomorrow?", T(CAL)), ("And on Friday?", T(CAL))]),
    "fu_reminder": ("follow-up", [("Remind me at 8 pm to water the plants.", T(SETR)), ("Actually make that 9 pm.", T(SETR, CANCEL))]),
    "fu_email": ("follow-up", [("Search my email for the flight booking.", T(SEARCH)), ("Open the first one.", T(READ, THREAD))]),
    # after the group expired (keep_turns 6: five filler turns in between)
    "expired_cal": ("expired", [("What's on my calendar tomorrow?", T(CAL)), *[(f, None) for f in FILLER], ("And what about Friday?", T(CAL))]),
    # changes that go through the confirmation question
    "confirm_delete": ("confirm", [("What's on my calendar tomorrow?", T(CAL)), ("Delete the workout.", T(DEL)), ("Yes, delete it.", ANY)]),
    "confirm_edit": ("confirm", [("What's on my calendar tomorrow?", T(CAL)), ("Move the dentist to 4 pm.", T(EDIT)), ("Yes.", ANY)]),
    # no tool needed
    "none_hello": ("no-tool", [("Hi Lucy, how are you?", None)]),
    "none_joke": ("no-tool", [("Tell me a joke.", None)]),
    "none_math": ("no-tool", [("What is 17 times 23?", None)]),
    "none_capital": ("no-tool", [("What's the capital of Japan?", None)]),
    "none_sky": ("no-tool", [("Why is the sky blue?", None)]),
    "none_hi_tired": ("no-tool", [("Aaj ka din bahut thaka dene wala tha.", None)]),
    "none_thanks": ("no-tool", [("Thanks, that's all for now.", None)]),
    # things Lucy cannot do
    "cant_pizza": ("cannot", [("Order me a pizza from Domino's.", CANT)]),
    "cant_lights": ("cannot", [("Turn off the living room lights.", CANT)]),
    "cant_send": ("cannot", [("Send an email to Ana saying I'll be late.", CANT)]),
    "cant_music": ("cannot", [("Play some jazz music.", CANT)]),
}  # fmt: skip

WEATHER = (
    "Weather from Open-Meteo for {place}, observed at 11:00 local time today.\n"
    "Now: 24°C, feels like 25°C, haze, wind 8 km/h, humidity 52%.\n"
    "Today: high 28°C, low 14°C, mainly clear, chance of rain 0%.\n"
    "Tomorrow: high 27°C, low 13°C, partly cloudy, chance of rain 40%."
)
EVENTS = (
    "Events on Wednesday 18 November:\n"
    "- 09:30-10:00 Team stand-up (id E1)\n- 15:00-16:00 Dentist (id E2)\n"
    "- 18:00-19:00 Workout (id E3)"
)
MAILS = (
    "3 messages:\n- id m1, from Amazon <ship@amazon.in>, 16 Nov, 'Your order has shipped'\n"
    "- id m2, from HDFC Bank <alerts@hdfcbank.net>, 15 Nov, 'Statement for October'\n"
    "- id m3, from IndiGo <noreply@goindigo.in>, 14 Nov, 'Booking confirmed: DEL-BOM 22 Nov'"
)


def result(name: str, args: dict[str, Any]) -> str:
    q = " ".join(str(v) for v in args.values())
    return {
        "wikipedia": f"Wikipedia, '{q}': (summary of the article; details omitted in this test).",
        "web_search": f"Results for '{q}':\n1. example.com: an article about {q}.",
        "fetch_page": "The page says: (text omitted in this test).",
        "weather": WEATHER.format(place=args.get("location") or args.get("place") or q),
        CAL: "No events in that range." if re.search(r"(?i)thu|fri|sat|19|20|21", q) else EVENTS,
        ADD: f"Added: {q}.",
        EDIT: "Changed.",
        DEL: "Deleted.",
        REM: "Pending:\n- id R1, 'call the plumber', today 18:00\n- id R2, 'water the plants', today 20:00",
        SETR: f"Reminder set: {q}.",
        TIMER: f"Timer started: {q}.",
        CANCEL: "Cancelled.",
        "enroll_voice": "Enrolment started; the user will be asked to read five sentences.",
        "replace_voice": "Re-enrolment started.",
        SEARCH: MAILS,
        READ: "From Amazon, 16 Nov: Your order of a kettle has shipped and arrives Thursday.",
        THREAD: "One message, from Amazon, 16 Nov: Your order of a kettle has shipped.",
    }.get(name, "Done.")


class Stub:
    """A tool with the real one's name, description, parameters, risk and grouping
    attributes, and a fixed result (and preview, for a dangerous one)."""

    def __init__(self, real: Any) -> None:
        self.name, self.description, self.parameters = real.name, real.description, real.parameters
        self.risk = real.risk
        for attr in ("sends_out", "private_source", "server", "destination"):
            if hasattr(real, attr):
                setattr(self, attr, getattr(real, attr))

    async def preview(self, args: dict[str, Any]):
        from lucy.core.types import ToolOutput

        target = {DEL: "Delete 'Workout'", EDIT: "Move 'Dentist'"}.get(
            self.name, f"Run {self.name}"
        )
        return ToolOutput(content=f"{target} on Wednesday 18 November?")

    async def run(self, args: dict[str, Any]):
        from lucy.core.types import ToolOutput

        return ToolOutput(content=result(self.name, args))


class StubProvider:
    """The configured MCP servers' tools from the lock file, as stubs; nothing starts."""

    def __init__(self, settings: Any) -> None:
        from lucy.adapters.mcp.provider import McpToolProvider

        class NoSecrets:
            async def get(self, service: str) -> str:
                raise RuntimeError("no secrets in a check")

        real = McpToolProvider(settings.mcp, secrets=NoSecrets(), root=settings.mcp.path)
        self._tools = [Stub(t) for t in real.tools()]
        self._guidance = real.guidance()

    async def start(self) -> None:
        pass

    def tools(self) -> list[Any]:
        return list(self._tools)

    def guidance(self) -> list[str]:
        return list(self._guidance)

    async def aclose(self) -> None:
        pass


class RetryCounter(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "produced no output" in record.getMessage():
            self.count += 1


async def turn(app, sid: str, text: str, limit: float) -> dict[str, Any]:
    from lucy.core.events import (
        AssistantMessage,
        ConfirmationRequested,
        TextDelta,
        ToolStarted,
        TurnEnded,
        UserMessage,
    )

    st: dict[str, Any] = {"reply": "", "confirm": "", "calls": [], "steps": 0, "first": None}
    t0 = time.monotonic()

    async def run():
        async with app.bus.subscribe(
            TextDelta, ToolStarted, ConfirmationRequested, AssistantMessage, TurnEnded
        ) as evs:
            await app.bus.publish(UserMessage(session_id=sid, text=text, source="text"))
            async for e in evs:
                if e.session_id != sid:
                    continue
                if isinstance(e, TextDelta):
                    st["reply"] += e.text
                    st["first"] = st["first"] or round(time.monotonic() - t0, 2)
                elif isinstance(e, AssistantMessage):
                    st["steps"] += 1
                elif isinstance(e, ToolStarted):
                    if not any(c["id"] == e.call.id for c in st["calls"]):
                        st["calls"].append(
                            {"id": e.call.id, "name": e.call.name, "args": e.call.arguments}
                        )
                elif isinstance(e, ConfirmationRequested):
                    st["confirm"] = e.question
                    st["calls"].append(
                        {"id": e.call.id, "name": e.call.name, "args": e.call.arguments}
                    )
                elif isinstance(e, TurnEnded):
                    return e.error if e.reason == "error" else None

    try:
        st["error"] = await asyncio.wait_for(run(), limit)
    except TimeoutError:
        st["error"] = f"no end within {limit:.0f} s"
    st["secs"] = round(time.monotonic() - t0, 2)
    return st


def score_turn(row: dict[str, Any], dangerous: set[str]) -> dict[str, Any]:
    """Outcome of one turn: correct, wrong_args, wrong, missing, needless or ok (no tool
    needed, none called), plus the more_tools verdicts and the extra model steps."""
    exp = row["expect"]
    calls = [c for c in row["calls"] if c["name"] != META]
    metas = [str(c["args"].get("group", "")) for c in row["calls"] if c["name"] == META]
    names = [c["name"] for c in calls]
    want_group = row.get("want_group")
    s: dict[str, Any] = {"meta": len(metas)}
    if exp == ANY:
        s["outcome"] = "ok"
    elif exp is None:
        s["outcome"] = "needless" if calls else "ok"
    elif exp == CANT:
        act = [
            n
            for n in names
            if n not in ("wikipedia", "web_search", "fetch_page", "time", "weather")
        ]
        s["outcome"] = "wrong" if act else "ok"
    else:
        hits = [c for c in calls if c["name"] in exp["tools"]]
        if not hits:
            s["outcome"] = "wrong" if calls else "missing"
        elif exp["args"] and not any(
            re.search(exp["args"], json.dumps(c["args"], ensure_ascii=False)) for c in hits
        ):
            s["outcome"] = "wrong_args"
        else:
            s["outcome"] = "correct"
    # more_tools: needed (the group was off and it named it), needless, wrong group, repeated
    on = set(row.get("on_before") or [])
    s["meta_needed"] = s["meta_needless"] = s["meta_wrong"] = 0
    for g in metas:
        if want_group and g == want_group and g not in on:
            s["meta_needed"] += 1
        elif want_group and g != want_group and g in row.get("groups", []):
            s["meta_wrong"] += 1
        else:
            s["meta_needless"] += 1
    s["meta_repeated"] = sum(n - 1 for n in Counter(metas).values() if n > 1)
    s["meta_needed"] = min(s["meta_needed"], 1)
    if exp in (None, ANY, CANT):
        minimum = 1
    elif any(t in dangerous for t in exp["tools"]):
        minimum = 1  # the turn ends at the confirmation question
    else:
        minimum = 2
    s["extra_steps"] = max(0, row["steps"] - minimum)
    s["empty"] = not row["reply"].strip() and not row["confirm"]
    return s


def summarise(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    dangerous = {DEL, EDIT, "replace_voice"}
    for r in rows:
        r.update(score_turn(r, dangerous))
    by = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r)
    print(f"SUMMARY {label}: {len(rows)} turns")
    head = "category     turns correct wr_args wrong missing needless | meta need nless wrong rep | xsteps empty retry err | first(med) total(med)"
    print(head)
    total: dict[str, Any] = {}
    for cat in [*sorted(by), "ALL"]:
        rs = rows if cat == "ALL" else by[cat]
        o = Counter(r["outcome"] for r in rs)
        firsts = [r["first"] for r in rs if r["first"] is not None]
        line = {
            "turns": len(rs),
            "correct": o["correct"] + o["ok"],
            "wrong_args": o["wrong_args"],
            "wrong": o["wrong"],
            "missing": o["missing"],
            "needless": o["needless"],
            "meta": sum(r["meta"] for r in rs),
            "meta_needed": sum(r["meta_needed"] for r in rs),
            "meta_needless": sum(r["meta_needless"] for r in rs),
            "meta_wrong": sum(r["meta_wrong"] for r in rs),
            "meta_repeated": sum(r["meta_repeated"] for r in rs),
            "extra_steps": sum(r["extra_steps"] for r in rs),
            "empty": sum(r["empty"] for r in rs),
            "errors": sum(r.get("error") is not None for r in rs),
            "retries": sum(r["retries"] for r in rs),
            "first": round(median(firsts), 2) if firsts else float("nan"),
            "secs": round(median(r["secs"] for r in rs), 2),
        }
        total[cat] = line
        print(
            f"{cat:12} {line['turns']:5} {line['correct']:7} {line['wrong_args']:7} {line['wrong']:5} "
            f"{line['missing']:7} {line['needless']:8} | {line['meta']:4} {line['meta_needed']:4} "
            f"{line['meta_needless']:5} {line['meta_wrong']:5} {line['meta_repeated']:3} | "
            f"{line['extra_steps']:6} {line['empty']:5} {line['retries']:5} {line['errors']:3} | "
            f"{line['first']:10} {line['secs']:10}"
        )
    tok = [r["prompt_chars"] for r in rows if r.get("prompt_chars")]
    if tok:
        print(f"first-step prompt (chars of system + tools + messages): median {median(tok):.0f}")
    return total


def settings_for(args):
    db = fresh_db(out_dir("tool_choice_dynamic", args.out) / args.label)
    s = isolated_settings(args.config, db)
    tools = {"dynamic": args.dynamic}
    if args.keep_turns:
        tools["keep_turns"] = args.keep_turns
    if args.core is not None:
        tools["core"] = [n for n in args.core.split(",") if n]
    update = {"tools": s.tools.model_copy(update=tools)}
    update["learning"] = s.learning.model_copy(update={"enabled": False})
    if s.speakers is not None:
        update["speakers"] = s.speakers.model_copy(update={"enabled": False})
    return s.model_copy(update=update), db


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", help="output directory (default: data/checks/tool_choice_dynamic/)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dynamic", dest="dynamic", action="store_true", default=True)
    mode.add_argument("--static", dest="dynamic", action="store_false")
    ap.add_argument("--keep-turns", type=int)
    ap.add_argument("--core", help="comma-separated core tools (overrides [tools] core)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--cases", help="comma-separated case names (default: all)")
    ap.add_argument("--summarise", nargs="+", metavar="ROWS")
    args = ap.parse_args()
    if args.summarise:
        for path in args.summarise:
            summarise(path, json.loads(Path(path).read_text()))
        return

    import sqlite3

    from lucy.app import LucyApp, build_selector
    from lucy.core.events import UserMessage
    from lucy.tools import build_tools

    settings, db = settings_for(args)
    names = args.cases.split(",") if args.cases else list(CASES)
    counter = RetryCounter()
    logging.getLogger("lucy.core.gateway").addHandler(counter)
    rows: list[dict[str, Any]] = []
    holder: dict[str, Any] = {}

    def tool_builder(s):
        return [Stub(t) for t in build_tools(s)]

    def provider_builder(s):
        holder["provider"] = StubProvider(s)
        return [holder["provider"]]

    async with LucyApp(
        settings, background=False, tool_builder=tool_builder, provider_builder=provider_builder
    ) as app:
        all_tools = app._tools  # noqa: SLF001 (a check reads the configured tools)
        sel = build_selector(
            settings.model_copy(
                update={"tools": settings.tools.model_copy(update={"dynamic": True})}
            ),
            all_tools,
            holder["provider"].guidance(),
        )
        groups = list(sel.groups) if sel else []
        from lucy.core.ports import tool_spec

        spec_chars = {
            t.name: len(json.dumps(tool_spec(t).model_dump()))
            for t in [*all_tools, *([sel.meta] if sel else [])]
        }
        print(f"dynamic={settings.tools.dynamic} core={sel.core if sel else None} groups={groups}")
        for rep in range(args.reps):
            for name in names:
                category, script = CASES[name]
                sid = f"{args.label}-{name}-{rep}"
                history: list[Any] = []
                for i, (text, expect) in enumerate(script):
                    want = expect["tools"][0] if isinstance(expect, dict) else None
                    want_group = sel.group_of(want) if (sel and want) else None
                    on_before: list[str] = []
                    if sel and settings.tools.dynamic:
                        on_before = sorted(
                            sel.groups_on([*history, UserMessage(session_id=sid, text=text)])
                        )
                    before = counter.count
                    st = await turn(app, sid, text, 240)
                    con = sqlite3.connect(db)
                    eps = con.execute(
                        "select payload from episodes where session_id=? order by id", (sid,)
                    ).fetchall()
                    trace = con.execute(
                        "select payload from turn_traces order by started_at desc limit 1"
                    ).fetchone()
                    con.close()
                    from lucy.core.events import event_from_json

                    history = [event_from_json(p[0]) for p in eps]
                    prompt_chars = offered = None
                    fallbacks = 0
                    if trace:
                        steps = json.loads(trace[0]).get("steps") or [{}]
                        offered = list(steps[0].get("tools") or [])
                        prompt_chars = len(json.dumps(steps[0].get("messages", ""))) + sum(
                            spec_chars.get(n, 0) for n in offered
                        )
                        fallbacks = sum(bool(x.get("fallback")) for x in steps)
                    row = {
                        "case": name,
                        "category": category,
                        "rep": rep,
                        "turn": i,
                        "q": text,
                        "expect": expect,
                        "want_group": want_group,
                        "groups": groups,
                        "on_before": on_before,
                        "kw_matched": sorted(sel.matched(text)) if sel else [],
                        "offered": offered,
                        "prompt_chars": prompt_chars,
                        "retries": counter.count - before + fallbacks,
                        **st,
                    }
                    row.update(score_turn(row, {DEL, EDIT, "replace_voice"}))
                    rows.append(row)
                    calls = " ".join(
                        f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)[:70]})"
                        for c in st["calls"]
                    )
                    print(
                        f"[{row['outcome']:10}] {name}#{i} steps={st['steps']} meta={row['meta']} "
                        f"on={on_before} kw={row['kw_matched']} {st['secs']}s q={text!r}\n"
                        f"      calls: {calls or '-'}\n      a={st['reply'].strip()[:160]!r}"
                        + (f"\n      confirm={st['confirm']!r}" if st["confirm"] else "")
                        + (f"\n      error={st['error']!r}" if st["error"] else ""),
                        flush=True,
                    )
    path = db.parent / "rows.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    summarise(args.label, rows)
    print(f"rows: {path}")


if __name__ == "__main__":
    asyncio.run(main())
