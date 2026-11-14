"""Does Lucy call a calendar edit or delete straight away, leaving the confirming to the tool?

usage (from the repo root): uv run python checks/confirm_once/confirm_once.py [--config FILE]
    [--label LABEL] [--out DIR] [--reps N] [--cases a,b] [--silent-after SECONDS]
Scripted cases through the real app and chat model, each in a fresh session, with the
calendar tools on a FAKE in-memory calendar (never the real one) and a fresh database
in OUT/LABEL/ (default data/checks/confirm_once/run/). See checks/README.md.
A turn that ends in an error (`SilentStreamError`: no output even on the fallback role)
or runs past a hard limit is scored "silent". To make that cheap, the check (only) gives
every LLM role a first-output timeout of SECONDS (default 15).
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

from lucy.app import LucyApp
from lucy.core.events import ConfirmationRequested, TextDelta, ToolStarted, TurnEnded, UserMessage
from lucy.tools.calendar_changes import DeleteCalendarEventTool, EditCalendarEventTool
from lucy.tools.calendar_dates import system_zone
from lucy.tools.calendar_events import AddCalendarEventTool, CalendarTool
from lucy.tools.local_time import TimeTool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DEFAULT_CONFIG, REPO, fresh_db, isolated_settings, out_dir

sys.path.insert(0, str(REPO))
from tests.test_tools_calendar import FakeStore, ev  # in-memory calendar store

ZONE = system_zone()
CONFIRM_TOOLS = {"edit_calendar_event", "delete_calendar_event"}
# Final turn expectations: "call" = the tool's confirmation question with no question of
# the model's own before it; "ask" = no tool confirmation, the model asks which event;
# "none" = no edit or delete at all.
CASES = {
    "add_then_delete": (
        ["Add a workout to my calendar tomorrow at 5pm."],
        "Delete the event you just created.",
        "call",
    ),
    "list_then_delete": (["What's on my calendar tomorrow?"], "Delete the workout one.", "call"),
    "list_then_move": (
        ["What's on my calendar this week?"],
        "Move my dentist appointment to Friday at 10.",
        "call",
    ),
    "two_match": (["What's on my calendar tomorrow?"], "Delete the meeting.", "ask"),
    "no_listing": ([], "Delete my workout tomorrow.", "call"),
    "two_match_unlisted": ([], "Delete my meeting tomorrow.", "ask"),
    "no_request": (["What's on my calendar tomorrow?"], "Is the workout at 3?", "none"),
}


def events() -> list:
    today = datetime.now(UTC).astimezone(ZONE).date()
    tomorrow = today + timedelta(days=1)
    dentist = today + timedelta(days=3)
    t = lambda d, h: datetime.combine(d, time(h), ZONE)  # noqa: E731
    return [
        ev("Team meeting", t(tomorrow, 10), t(tomorrow, 11), calendar="Work"),
        ev("Budget meeting", t(tomorrow, 13), t(tomorrow, 14), calendar="Work"),
        ev("Workout", t(tomorrow, 15), t(tomorrow, 16), calendar="Work"),
        ev("Dentist", t(dentist, 11), t(dentist, 12), location="1 High Street"),
    ]


def quick_silence(settings, seconds: float):
    """`settings` with every LLM role ending a silent stream after `seconds` (check only)."""
    roles = {
        name: role.model_copy(update={"first_output_timeout": seconds})
        for name, role in settings.llm.roles.items()
    }
    llm = settings.llm.model_copy(update={"roles": roles})
    return settings.model_copy(update={"llm": llm})


class Holder:
    """A calendar store whose fake contents are replaced for each case."""

    def __init__(self) -> None:
        self.store = FakeStore()

    def __getattr__(self, name: str):
        return getattr(self.store, name)


async def turn(app, sid, text, limit):
    """(model text before any confirmation, text after, tools, confirmation, error)."""
    state = {"before": "", "after": "", "tools": [], "confirm": None}
    try:
        error = await asyncio.wait_for(_turn(app, sid, text, state), limit)
    except TimeoutError:
        error = f"no end within {limit:.0f} s"
    return state["before"], state["after"], state["tools"], state["confirm"], error


async def _turn(app, sid, text, state):
    before = after = ""
    tools: list[str] = []
    confirm = None
    async with app.bus.subscribe(TextDelta, ToolStarted, ConfirmationRequested, TurnEnded) as evs:
        await app.bus.publish(UserMessage(session_id=sid, text=text, source="text"))
        async for e in evs:
            if e.session_id != sid:
                continue
            if isinstance(e, TextDelta):
                if confirm is None:
                    before += e.text
                else:
                    after += e.text
            elif isinstance(e, ToolStarted):
                tools.append(f"{e.call.name}:{json.dumps(e.call.arguments)[:70]}")
            elif isinstance(e, ConfirmationRequested):
                confirm = e
            elif isinstance(e, TurnEnded):
                return e.error if e.reason == "error" else None
            state.update(before=before, after=after, tools=tools, confirm=confirm)


def score(expect, before, confirm, earlier_confirm, error) -> tuple[bool, str]:
    asked = "?" in before
    if error is not None:
        return False, "silent"
    if earlier_confirm:
        return False, "early"  # edit or delete before the user asked for it
    if expect == "call":
        if confirm is None:
            return False, "own-q" if asked else "nocall"
        return (False, "double") if asked else (True, "ok")
    if expect == "ask":
        if confirm is not None:
            return False, "guessed"
        return (True, "ok") if asked else (False, "noask")
    return (confirm is None, "ok" if confirm is None else "acted")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", help="output directory (default: data/checks/confirm_once/)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cases", help="comma-separated case names (default: all)")
    ap.add_argument("--silent-after", type=float, default=15.0)
    args = ap.parse_args()
    tag = args.label
    names = args.cases.split(",") if args.cases else list(CASES)
    db = fresh_db(out_dir("confirm_once", args.out) / tag)
    holder = Holder()

    def build(settings):
        s = settings.tools.calendar
        kw = {"store": holder, "local_zone": ZONE}
        hint = {"changes_enabled": True}  # as build_tools does with edit/delete enabled
        return [
            TimeTool(),
            CalendarTool(s, **kw, **hint),
            AddCalendarEventTool(s, **kw, **hint),
            EditCalendarEventTool(s, **kw),
            DeleteCalendarEventTool(s, **kw),
        ]

    limit = 4 * args.silent_after + 30  # a backstop; the first-output timeouts end it sooner
    totals: dict[str, dict[str, int]] = {}
    rows = []
    async with LucyApp(
        quick_silence(isolated_settings(args.config, db), args.silent_after),
        background=False,
        tool_builder=build,
    ) as app:
        for rep in range(args.reps):
            for name in names:
                setup, final, expect = CASES[name]
                holder.store = FakeStore(events())
                sid = f"{tag}-{name}-{rep}"
                early = False
                log = []
                for text in setup:
                    b, a, tools, c, err = await turn(app, sid, text, limit)
                    early |= c is not None
                    log.append({"q": text, "a": b + a, "tools": tools, "error": err})
                b, a, tools, c, err = await turn(app, sid, final, limit)
                ok, why = score(expect, b, c, early, err)
                counts = totals.setdefault(name, {})
                counts[why] = counts.get(why, 0) + 1
                log.append({"q": final, "before": b, "after": a, "tools": tools, "error": err})
                rows.append({"case": name, "rep": rep, "ok": ok, "why": why, "turns": log})
                print(f"[{'.' if ok else 'X'}] {name:18} {why:7} tools={tools}", flush=True)
                print(f"      model={b.strip()!r}\n      code={a.strip()!r}", flush=True)
                if err is not None:
                    print(f"      error={err!r}", flush=True)
    (db.parent / "rows.json").write_text(json.dumps(rows, indent=2))
    print(f"SUMMARY {tag} ({args.reps} reps):")
    passed = 0
    for name in names:
        counts = totals.get(name, {})
        passed += counts.get("ok", 0)
        rest = " ".join(f"{k}={v}" for k, v in sorted(counts.items()) if k != "ok")
        print(f"  {name:18} {counts.get('ok', 0)}/{args.reps}  {rest}")
    print(f"  total            {passed}/{args.reps * len(names)}")


if __name__ == "__main__":
    asyncio.run(main())
