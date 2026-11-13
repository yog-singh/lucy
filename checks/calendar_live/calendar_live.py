"""A live smoke test of the calendar tools on this Mac's real calendar.

usage (from the repo root): uv run python checks/calendar_live/calendar_live.py --write
    [--calendar NAME]

It WRITES TO THE REAL CALENDAR, so without --write it only prints what it would do.
With --write it creates one event, "Lucy test event - safe to delete", on Wednesday
12 June 2030 at 10:00 (in NAME, default: the Calendar app's default calendar), lists
it back, edits it (new title, moved to Thursday 13 June 2030 at 11:00, a place),
deletes it, and checks each step against the store. It touches only the event it
created: every edit and delete goes through that event's own id, and the cleanup
deletes only an event with the test's identifier and title. On any failure it
deletes the test event if it still exists, and says so. Every step is printed.

The tools are called directly (preview, then run), not through the model or the
confirmation flow; those are covered by the tests with a fake store.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, timedelta
from typing import Any

from lucy.core.config import CalendarSettings
from lucy.tools.calendar_changes import DeleteCalendarEventTool, EditCalendarEventTool
from lucy.tools.calendar_dates import midnight, system_zone
from lucy.tools.calendar_events import AddCalendarEventTool, CalendarTool, event_id
from lucy.tools.calendar_store import EventKitStore, StoredEvent, run_blocking

TITLE = "Lucy test event - safe to delete"
EDITED = "Lucy test event (edited) - safe to delete"
DAY = date(2030, 6, 12)


class CheckFailed(Exception):
    pass


def check(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        raise CheckFailed(what)


async def mine(store: EventKitStore, uid: str, zone: Any) -> list[StoredEvent]:
    """Events with the test event's identifier, around the test days."""
    found = await run_blocking(
        store.events, midnight(DAY - timedelta(1), zone), midnight(DAY + timedelta(3), zone), None
    )
    return [e for e in found if e.uid == uid]


async def step(tool: Any, args: dict[str, Any]) -> str:
    """Preview then run, as the orchestrator would after a yes."""
    if hasattr(tool, "preview"):
        question = await tool.preview(args)
        print(f"  question: {question.content}")
        check(not question.is_error, f"{tool.name} preview is not an error")
        again = await tool.preview(args)
        check(again.content == question.content, "the preview is the same when asked again")
    result = await tool.run(args)
    print(f"  result: {result.content}")
    check(not result.is_error, f"{tool.name} succeeded")
    return result.content


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="really write to the calendar")
    parser.add_argument("--calendar", help="calendar to use (default: the system default)")
    args = parser.parse_args()
    if not args.write:
        print(__doc__)
        print("Nothing was done. Pass --write to run it on the real calendar.")
        return 0

    zone = system_zone()
    store = EventKitStore(zone)
    settings = CalendarSettings()
    kw: dict[str, Any] = {"store": store, "local_zone": zone}
    reader = CalendarTool(settings, **kw)
    adder = AddCalendarEventTool(settings, **kw)
    editor = EditCalendarEventTool(settings, **kw)
    deleter = DeleteCalendarEventTool(settings, **kw)
    created: StoredEvent | None = None
    try:
        print(f"1. Create {TITLE!r} on {DAY:%A %d %B %Y} at 10:00")
        new = {"title": TITLE, "start": f"{DAY.isoformat()}T10:00", "duration_minutes": 30}
        if args.calendar:
            new["calendar"] = args.calendar
        await step(adder, new)
        listed = await run_blocking(
            store.events, midnight(DAY, zone), midnight(DAY + timedelta(1), zone), None
        )
        ours = [e for e in listed if e.title == TITLE]
        check(len(ours) == 1, "exactly one test event exists on that day")
        created = ours[0]
        print(f"  created: uid {created.uid}, id {event_id(created)}, calendar {created.calendar}")

        print("2. Read it back with the calendar tool")
        text = await step(reader, {"start": DAY.isoformat()})
        check(event_id(created) in text and TITLE in text, "the listing shows it with its id")

        print("3. Edit it: new title, move to the next day at 11:00 (length kept), a place")
        edit = {
            "id": event_id(created),
            "title": EDITED,
            "start": f"{(DAY + timedelta(1)).isoformat()}T11:00",
            "location": "Nowhere",
        }
        text = await step(editor, edit)
        after = await mine(store, created.uid, zone)
        print(f"  identifier after the edit: {[e.uid for e in after] or 'not found'}")
        check(len(after) == 1, "the event keeps its identifier after the edit")
        edited = after[0]
        created = edited
        check(edited.title == EDITED, "the title changed")
        check(
            edited.start.date() == DAY + timedelta(1) and edited.start.hour == 11,
            "it starts on the next day at 11:00",
        )
        check(edited.end - edited.start == timedelta(minutes=30), "it still lasts 30 minutes")
        check(edited.location == "Nowhere", "the place is set")
        check(event_id(edited) in text, "the result gives the new id")

        print("4. Delete it")
        await step(deleter, {"id": event_id(edited)})
        check(await mine(store, edited.uid, zone) == [], "it is gone")
        created = None
        print("All steps passed; the calendar is as it was.")
        return 0
    except Exception as exc:  # CheckFailed, or the store refusing
        print(f"Stopped: {exc}")
        return 1
    finally:
        if created is not None:
            await cleanup(store, created, zone)


async def cleanup(store: EventKitStore, created: StoredEvent, zone: Any) -> None:
    left = [e for e in await mine(store, created.uid, zone) if e.title in (TITLE, EDITED)]
    for event in left:
        print(f"Cleanup: deleting the test event {event.title!r} at {event.start:%Y-%m-%d %H:%M}")
        await run_blocking(store.remove, event)
    remaining = await mine(store, created.uid, zone)
    print("Cleanup: done." if not remaining else f"Cleanup: still there: {remaining}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
