"""The calendar tools: `calendar` lists the user's events (read tier) and
`add_calendar_event` adds one (act tier), in the macOS calendar store on this
machine. Nothing leaves the machine.

Dates come in as the user's words ("friday afternoon", "tomorrow 3pm") or ISO
dates and are resolved in code (`calendar_dates`); every result states the
resolved dates with weekdays, so the model reads back what actually happened.

Event ids: each listed event carries a short id such as `k3f9qz-20261117`: six
characters of a hash of EventKit's `eventIdentifier`, then the local date the
occurrence starts on. All occurrences of a repeating event share the
`eventIdentifier`, so the date picks the occurrence. `find_event` turns an id back
into the event by listing that one day, so ids survive a restart of Lucy; an
event moved to another day since it was listed is not found (list again).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import re
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, time, timedelta, tzinfo
from typing import Any, Final

from lucy.core.config import CalendarSettings
from lucy.core.types import Risk, ToolOutput
from lucy.tools.calendar_dates import (
    DateError,
    Span,
    day_text,
    local,
    midnight,
    moment_text,
    resolve,
    span_text,
    system_zone,
    when_text,
)
from lucy.tools.calendar_store import (
    Access,
    CalendarStore,
    CalendarStoreError,
    EventKitStore,
    NewEvent,
    StoredEvent,
    run_blocking,
)
from lucy.tools.common import cap, error, required_text, shorten, squash

UPCOMING_DAYS: Final = 14
"""How far ahead "now" looks when no end is given ("what's my next meeting")."""
MAX_RANGE_DAYS: Final = 62
MAX_EVENT_DAYS: Final = 14
DEFAULT_MINUTES: Final = 60
FREE_FROM: Final = time(8)
FREE_UNTIL: Final = time(20)
"""Free time is listed between these hours when a whole single day is asked for."""
NOTES_CHARS: Final = 160
MAX_TITLE: Final = 200
MAX_TEXT: Final = 2000

_ID: Final = re.compile(r"^([a-z2-7]{6})-(\d{8})$")


def event_id(event: StoredEvent) -> str:
    """The short id of one event occurrence (see the module docstring)."""
    digest = hashlib.sha1(event.uid.encode()).digest()
    return f"{base64.b32encode(digest).decode().lower()[:6]}-{event.start:%Y%m%d}"


def find_event(
    store: CalendarStore, ident: str, zone: tzinfo, calendars: Sequence[str] | None = None
) -> StoredEvent | None:
    """The event occurrence `ident` names, or None (blocking; for the edit and delete
    tools to come). It lists only the day in the id."""
    match = _ID.match(ident.strip().lower())
    if not match:
        return None
    try:
        day = datetime.strptime(match.group(2), "%Y%m%d").date()
    except ValueError:
        return None
    found = store.events(midnight(day, zone), midnight(day + timedelta(1), zone), calendars)
    return next((e for e in found if event_id(e) == match.group(0)), None)


def access_problem(access: Access) -> str:
    """Why the calendar cannot be used, and how the user can fix it."""
    settings = (
        "The user can allow it in System Settings > Privacy & Security > Calendars, by giving "
        "Full Access to the app Lucy runs in (such as Terminal), then restarting Lucy."
    )
    return {
        Access.DENIED: f"Lucy does not have access to the calendar. {settings}",
        Access.WRITE_ONLY: (
            "Lucy may only add events, not read the calendar. "
            + settings.replace("allow it", "change that")
        ),
        Access.NOT_DETERMINED: (
            "macOS is asking whether Lucy may use the calendar. The user should answer the "
            "dialog (choose Allow Full Access), then ask again. If no dialog appeared: "
            + settings[0].lower()
            + settings[1:]
        ),
        Access.RESTRICTED: (
            "Calendar access is restricted on this Mac (for example by device management or "
            "Screen Time), so Lucy cannot use the calendar."
        ),
        Access.UNAVAILABLE: (
            "The calendar is not available: it needs macOS and the pyobjc-framework-EventKit "
            "package (installed with Lucy on a Mac)."
        ),
    }.get(access, "Lucy cannot use the calendar.")


class _CalendarBase:
    """What both calendar tools share: the store, the clock, access and threading."""

    def __init__(
        self,
        settings: CalendarSettings | None = None,
        *,
        store: CalendarStore | None = None,
        clock: Callable[[], datetime] | None = None,
        local_zone: tzinfo | None = None,
    ) -> None:
        self.settings = settings or CalendarSettings()
        self._zone = local_zone or system_zone()
        self._store = store or shared_store(self._zone)
        self._clock = clock or (lambda: datetime.now(UTC))

    def _now(self) -> datetime:
        return self._clock().astimezone(self._zone)

    async def _call(self, fn: Callable[..., Any], *args: Any, timeout: float | None = None) -> Any:
        async with asyncio.timeout(timeout or self.settings.timeout):
            return await run_blocking(fn, *args)

    async def _check_access(self) -> str | None:
        """None when the calendar can be read and written, else the problem."""
        access = await self._call(self._store.access)
        if access is Access.NOT_DETERMINED:
            wait = self.settings.permission_timeout
            try:
                access = await self._call(self._store.request_access, wait, timeout=wait + 5)
            except TimeoutError:
                access = Access.NOT_DETERMINED
        return None if access is Access.FULL else access_problem(access)

    async def _guarded(self, work: Callable[[], Awaitable[ToolOutput]]) -> ToolOutput:
        try:
            problem = await self._check_access()
            if problem:
                return error(problem)
            return await work()
        except DateError as exc:
            return error(str(exc))
        except CalendarStoreError as exc:
            return error(str(exc))
        except TimeoutError:
            return error("The calendar did not answer in time. Try again in a moment.")


def shared_store(zone: tzinfo) -> CalendarStore:
    """The process's one EventKit store (both tools use it)."""
    global _SHARED
    if _SHARED is None:
        _SHARED = EventKitStore(zone)
    return _SHARED


_SHARED: EventKitStore | None = None


class CalendarTool(_CalendarBase):
    """The user's calendar events in a range of days (risk tier read; no network)."""

    name = "calendar"
    description = (
        "List the user's calendar events, to answer what is on, whether they are free, or "
        "when their next event is. Pass 'start' (and optionally 'end') in the user's own words "
        "or as ISO dates; do not work dates out yourself. Examples: 'today', 'tomorrow', "
        "'friday', 'friday afternoon', 'next week', '20 November', '2026-11-20'. For the "
        "next event, pass start 'now'. Each event has an id."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "start": {
                "type": "string",
                "description": "Day or time to list from, as the user said it, e.g. 'today', "
                "'friday afternoon', 'next week', '2026-11-20', or 'now' for upcoming events. "
                "Default 'today'.",
            },
            "end": {
                "type": "string",
                "description": "Optional last day (included) or end time, same forms. Leave "
                "it out for just the day or part of day in 'start'.",
            },
            "query": {
                "type": "string",
                "description": "Optional words to look for in event titles, places and notes.",
            },
        },
        "required": [],
    }
    risk = Risk.READ

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        return await self._guarded(lambda: self._list(args))

    async def _list(self, args: dict[str, Any]) -> ToolOutput:
        now = self._now()
        start_text = required_text(args, "start") or "today"
        start = resolve(start_text, now)
        end_text = required_text(args, "end")
        upcoming = start.is_point and start.start == now and end_text is None
        if end_text is not None:
            end = resolve(end_text, now, base=start.start.date())
            last = end.start if end.is_point else end.end
        elif upcoming:
            last = midnight(now.date() + timedelta(UPCOMING_DAYS + 1), self._zone)
        elif start.is_point:
            last = midnight(start.start.date() + timedelta(1), self._zone)
        else:
            last = start.end
        first = start.start
        if last <= first:
            raise DateError(
                f"The end ({moment_text(last)}) is not after the start ({moment_text(first)})."
            )
        if last - first > timedelta(days=MAX_RANGE_DAYS):
            raise DateError(f"That range is too long; ask for at most {MAX_RANGE_DAYS} days.")
        wanted = list(self.settings.calendars) or None
        found: list[StoredEvent] = await self._call(self._store.events, first, last, wanted)
        query = required_text(args, "query")
        if query:
            words = query.casefold().split()
            found = [
                e
                for e in found
                if all(w in f"{e.title} {e.location} {e.notes}".casefold() for w in words)
            ]
        if upcoming:
            heading = (
                f"Calendar from now ({moment_text(now)}) to {day_text(last.date() - timedelta(1))}"
            )
        else:
            span = Span(first, last, "days" if _whole_days(first, last) else "part")
            heading = f"Calendar for {span_text(span)}"
        matching = f" matching {query!r}" if query else ""
        lines = [f"{heading} (the user's local time):"]
        if not found:
            lines = [f"{heading} (the user's local time): no events{matching}."]
        else:
            lines[0] = (
                f"{heading} (the user's local time), {len(found)} event"
                + ("s" if len(found) != 1 else "")
                + f"{matching}:"
            )
            lines.extend(f"- {_describe(e)}" for e in found)
        free = None if (query or upcoming) else _free_time(found, first, last, self._zone)
        if free:
            lines.append(free)
        return ToolOutput(content=cap("\n".join(lines), self.settings.max_chars))


def _whole_days(first: datetime, last: datetime) -> bool:
    return first.time() == time(0) and last.time() == time(0)


def _describe(event: StoredEvent) -> str:
    text = f"{when_text(event.start, event.end, event.all_day)}: {event.title or '(no title)'}"
    extras = [f"calendar {event.calendar}"] if event.calendar else []
    if event.location:
        extras.append(f"at {squash(event.location)}")
    if event.recurring:
        extras.append("repeating")
    if event.notes.strip():
        extras.append(f"notes: {shorten(squash(event.notes), NOTES_CHARS)}")
    extras.append(f"id {event_id(event)}")
    return f"{text} ({'; '.join(extras)})"


def _free_time(
    events: list[StoredEvent], first: datetime, last: datetime, zone: tzinfo
) -> str | None:
    """Free stretches within one day: the asked-for hours, or FREE_FROM to FREE_UNTIL
    for a whole day. All-day events do not count as busy. None for longer ranges."""
    if (
        last - first > timedelta(hours=25)
        or first.date() != (last - timedelta(microseconds=1)).date()
    ):
        return None
    if _whole_days(first, last):
        first = local(first.date(), FREE_FROM, zone, strict=False)
        last = local(first.date(), FREE_UNTIL, zone, strict=False)
        window = f"between {FREE_FROM:%H:%M} and {FREE_UNTIL:%H:%M}"
    else:
        window = (
            f"between {first:%H:%M} and {'midnight' if last.time() == time(0) else f'{last:%H:%M}'}"
        )
    busy = sorted((max(e.start, first), min(e.end, last)) for e in events if not e.all_day)
    gaps: list[tuple[datetime, datetime]] = []
    cursor = first
    for begin, finish in busy:
        if begin > cursor:
            gaps.append((cursor, begin))
        cursor = max(cursor, finish)
    if cursor < last:
        gaps.append((cursor, last))
    gaps = [(a, b) for a, b in gaps if b - a >= timedelta(minutes=5)]
    note = " (all-day events not counted as busy)" if any(e.all_day for e in events) else ""
    if not gaps:
        return f"No free time {window}{note}."
    if gaps == [(first, last)]:
        return f"Free the whole time {window}{note}."
    end_label = lambda m: "midnight" if m.time() == time(0) and m > first else f"{m:%H:%M}"  # noqa: E731
    listed = ", ".join(f"{a:%H:%M} to {end_label(b)}" for a, b in gaps)
    return f"Free time {window}: {listed}{note}."


class AddCalendarEventTool(_CalendarBase):
    """Add one event to the user's calendar (risk tier act; no network)."""

    name = "add_calendar_event"
    description = (
        "Add an event to the user's calendar. Only when the user asks for it. Pass 'start' "
        "in the user's own words or as an ISO date and time, e.g. 'friday 3pm', 'tomorrow "
        "at 9:30', '2026-11-20T15:00'; do not work dates out yourself. If the user gave no "
        "time, ask for one (or set all_day for a whole-day event). Give 'end' or "
        "'duration_minutes' if the user said how long; otherwise it lasts one hour."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "What the event is, short."},
            "start": {
                "type": "string",
                "description": "When it starts, as the user said it: a day with a time such "
                "as 'friday 3pm', or an ISO date-time. For an all-day event, just the day.",
            },
            "end": {
                "type": "string",
                "description": "Optional end: a time such as '16:00' (same day), a day and "
                "time, or for all-day events the last day.",
            },
            "duration_minutes": {
                "type": "integer",
                "description": "Optional length in minutes, instead of 'end'.",
            },
            "all_day": {"type": "boolean", "description": "True for a whole-day event."},
            "location": {"type": "string", "description": "Optional place."},
            "notes": {"type": "string", "description": "Optional notes."},
            "calendar": {
                "type": "string",
                "description": "Optional calendar name; leave out for the default calendar.",
            },
        },
        "required": ["title", "start"],
    }
    risk = Risk.ACT

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        return await self._guarded(lambda: self._add(args))

    async def _add(self, args: dict[str, Any]) -> ToolOutput:
        title = required_text(args, "title")
        start_text = required_text(args, "start")
        if title is None or start_text is None:
            return error("add_calendar_event needs a 'title' and a 'start' (a day and time).")
        if len(title) > MAX_TITLE:
            return error(f"The title is too long (at most {MAX_TITLE} characters).")
        now = self._now()
        all_day = args.get("all_day") is True or str(args.get("all_day")).lower() == "true"
        start = resolve(start_text, now)
        end_text = required_text(args, "end")
        minutes = _minutes(args.get("duration_minutes"))
        if minutes == -1:
            return error("'duration_minutes' must be a whole number of minutes, at least 1.")
        if end_text is not None and minutes is not None:
            return error("Pass either 'end' or 'duration_minutes', not both.")
        defaulted = False
        if all_day:
            if start.kind != "days" or start.days != 1:
                return error(
                    f"For an all-day event, 'start' must be one day without a time; "
                    f"{start_text!r} is not. Use for example 'friday' or '2026-11-20'."
                )
            first, last = start.start, start.end
            if end_text is not None:
                end = resolve(end_text, now, base=first.date())
                if end.kind != "days":
                    return error(
                        "For an all-day event, 'end' must be the last day, without a time."
                    )
                last = end.end
            elif minutes is not None:
                return error("An all-day event takes an 'end' day, not 'duration_minutes'.")
            if first.date() < now.date():
                return error(
                    f"{day_text(first.date())} is in the past; check the date with the user."
                )
        else:
            if not start.is_point:
                what = "a part of the day" if start.kind == "part" else "a day without a time"
                return error(
                    f"{start_text!r} is {what} ({span_text(start)}), not a start time. Ask the "
                    "user what time it starts, or set all_day for a whole-day event."
                )
            first = start.start
            if end_text is not None:
                end = resolve(end_text, now, base=first.date())
                if not end.is_point:
                    return error(f"'end' must be a time, such as '16:00'; {end_text!r} is not.")
                last = end.start
            else:
                defaulted = minutes is None
                elapsed = timedelta(minutes=minutes or DEFAULT_MINUTES)
                last = (first.astimezone(UTC) + elapsed).astimezone(self._zone)
            if first < now - timedelta(minutes=1):
                return error(
                    f"{moment_text(first)} is in the past; check the date and time with the user."
                )
        if last <= first:
            return error(
                f"The end ({moment_text(last)}) is not after the start ({moment_text(first)})."
            )
        if last - first > timedelta(days=MAX_EVENT_DAYS):
            return error(
                f"That event would last more than {MAX_EVENT_DAYS} days; check with the user."
            )
        calendar = await self._target_calendar(required_text(args, "calendar"))
        if isinstance(calendar, ToolOutput):
            return calendar
        new = NewEvent(
            title=title,
            start=first,
            end=last,
            all_day=all_day,
            calendar=calendar,
            location=(required_text(args, "location") or "")[:MAX_TEXT],
            notes=(required_text(args, "notes") or "")[:MAX_TEXT],
        )
        overlapping = [
            e
            for e in await self._call(self._store.events, first, last, None)
            if not e.all_day and not all_day
        ]
        saved: StoredEvent = await self._call(self._store.add, new)
        lines = [
            f"Added to the calendar {saved.calendar}: {saved.title}, "
            f"{when_text(saved.start, saved.end, saved.all_day)} (the user's local time)."
        ]
        if saved.location:
            lines.append(f"Place: {saved.location}.")
        if defaulted:
            lines.append("No end or length was given, so it lasts one hour.")
        if overlapping:
            lines.append(
                "It overlaps with: "
                + "; ".join(
                    f"{e.title or '(no title)'}, {when_text(e.start, e.end, False)}"
                    for e in overlapping
                )
                + "."
            )
        lines.append(f"Event id: {event_id(saved)}.")
        return ToolOutput(content=cap(" ".join(lines), self.settings.max_chars))

    async def _target_calendar(self, asked: str | None) -> str | ToolOutput:
        """The calendar to add to: the one named, else the configured default, else the
        store's; an error naming the writable calendars if that is not one."""
        writable = [c.name for c in await self._call(self._store.calendars) if c.writable]
        name = asked or self.settings.default_calendar
        if name is None:
            name = await self._call(self._store.default_calendar)
        match = next((c for c in writable if name and c.casefold() == name.casefold()), None)
        if match is None:
            options = ", ".join(writable) or "none"
            what = (
                f"There is no calendar called {name!r} that Lucy can add to"
                if name
                else ("There is no default calendar")
            )
            return error(f"{what}. Calendars that can be added to: {options}.")
        return match


def _minutes(value: object) -> int | None:
    """Minutes asked for; None if not given, -1 if not usable."""
    if value is None or value == "":
        return None
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 1:
        return -1
    return int(value)
