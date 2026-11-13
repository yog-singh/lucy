"""The calendar tools that change events: `edit_calendar_event` and
`delete_calendar_event` (risk tier dangerous: the orchestrator asks the user first,
ADR 0006).

Both take an event id from a `calendar` result and act on that one occurrence only;
for a repeating event the rest of the series is never touched. `preview` resolves
the id and any new dates now and states the real target in absolute terms (no
"tomorrow"), as one spoken question; `run` does the same resolution again and then
the change. The orchestrator previews again just before running and runs nothing if
the text differs, so a relative date that resolves differently at confirm time (a
request made just before midnight, confirmed just after) is not acted on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from lucy.core.types import Risk, ToolOutput
from lucy.tools.calendar_dates import (
    day_text,
    local,
    midnight,
    moment_text,
    resolve,
    span_text,
    when_text,
)
from lucy.tools.calendar_events import (
    MAX_EVENT_DAYS,
    MAX_TEXT,
    MAX_TITLE,
    _CalendarBase,
    _minutes,
    event_id,
    find_event,
)
from lucy.tools.calendar_store import EventChanges, StoredEvent
from lucy.tools.common import cap, error, required_text, shorten, squash

SPOKEN_CHARS = 80
"""Longest place or notes text read out in a question."""

_ID_PARAM: dict[str, Any] = {
    "type": "string",
    "description": "The event's id, exactly as a calendar tool result in this conversation "
    "gave it, e.g. 'k3f9qz-20261117'. Never make one up.",
}


class _Refused(Exception):
    """The call cannot be asked about; the message is for the model."""


def _quote(text: str) -> str:
    return f"'{shorten(squash(text), SPOKEN_CHARS)}'"


def _title(event: StoredEvent) -> str:
    return _quote(event.title or "(no title)")


class _ChangeBase(_CalendarBase):
    risk = Risk.DANGEROUS

    async def preview(self, args: dict[str, Any]) -> ToolOutput:
        return await self._guarded(lambda: self._safely(self._question, args))

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        return await self._guarded(lambda: self._safely(self._apply, args))

    async def _safely(self, step: Any, args: dict[str, Any]) -> ToolOutput:
        try:
            return await step(args)
        except _Refused as exc:
            return error(str(exc))

    async def _question(self, args: dict[str, Any]) -> ToolOutput:
        raise NotImplementedError

    async def _apply(self, args: dict[str, Any]) -> ToolOutput:
        raise NotImplementedError

    async def _target(self, args: dict[str, Any]) -> StoredEvent:
        """The occurrence the id names, if it may be changed; else `_Refused`."""
        ident = required_text(args, "id")
        if ident is None:
            raise _Refused(
                f"{self.name} needs the event's 'id' from a calendar tool result. List the "
                "calendar first."
            )
        wanted = list(self.settings.calendars) or None
        event: StoredEvent | None = await self._call(
            find_event, self._store, ident, self._zone, wanted
        )
        if event is None:
            raise _Refused(
                f"There is no event with id {ident!r} now: ids come only from calendar tool "
                "results, and the event may have been moved or deleted since. List the "
                "calendar again with the calendar tool and use the id it gives."
            )
        if event.invited_by:
            raise _Refused(
                f"{_title(event)} is an invitation from {event.invited_by}; Lucy cannot change "
                "or delete it. The user can respond to it in the Calendar app."
            )
        if not event.calendar_writable:
            raise _Refused(
                f"{_title(event)} is in the calendar {event.calendar}, which cannot be changed."
            )
        return event


def _occurrence_note(event: StoredEvent) -> str:
    return ", this occurrence only" if event.recurring else ""


class DeleteCalendarEventTool(_ChangeBase):
    """Delete one event occurrence (risk tier dangerous; asks the user first)."""

    name = "delete_calendar_event"
    description = (
        "Delete one event from the user's calendar, by the id a calendar tool result gave. "
        "Only when the user asks for it. The user is asked to confirm before anything is "
        "deleted. For a repeating event, only the one occurrence is deleted."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"id": _ID_PARAM},
        "required": ["id"],
    }

    async def _question(self, args: dict[str, Any]) -> ToolOutput:
        event = await self._target(args)
        return ToolOutput(
            content=f"Delete {_title(event)} on {when_text(event.start, event.end, event.all_day)}"
            f", from the calendar {event.calendar}{_occurrence_note(event)}?"
        )

    async def _apply(self, args: dict[str, Any]) -> ToolOutput:
        event = await self._target(args)
        await self._call(self._store.remove, event)
        text = (
            f"Deleted from the calendar {event.calendar}: {event.title or '(no title)'}, "
            f"{when_text(event.start, event.end, event.all_day)} (the user's local time)."
        )
        if event.recurring:
            text += " Only this occurrence was deleted; the rest of the repeating event remains."
        return ToolOutput(content=cap(text, self.settings.max_chars))


@dataclass(frozen=True)
class _Plan:
    event: StoredEvent
    changes: EventChanges


class EditCalendarEventTool(_ChangeBase):
    """Change one event occurrence (risk tier dangerous; asks the user first)."""

    name = "edit_calendar_event"
    description = (
        "Change one event in the user's calendar, by the id a calendar tool result gave: its "
        "title, time, place or notes. Only when the user asks for it. Pass only what changes. "
        "Pass a new 'start' in the user's own words, e.g. 'friday 3pm', 'tomorrow at 9:30', "
        "or just a day such as 'friday' to keep the time; do not work dates out yourself. "
        "Moving the start keeps the event's length unless 'end' or 'duration_minutes' is "
        "given. The user is asked to confirm before anything changes. For a repeating event, "
        "only the one occurrence changes."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "id": _ID_PARAM,
            "title": {"type": "string", "description": "Optional new title."},
            "start": {
                "type": "string",
                "description": "Optional new start, as the user said it: a day and time such "
                "as 'friday 3pm', a day alone to keep the time, or an ISO date-time. For an "
                "all-day event, a day.",
            },
            "end": {
                "type": "string",
                "description": "Optional new end: a time such as '16:00', a day and time, or "
                "for all-day events the last day.",
            },
            "duration_minutes": {
                "type": "integer",
                "description": "Optional new length in minutes, instead of 'end'.",
            },
            "location": {
                "type": "string",
                "description": "Optional new place; an empty string removes it.",
            },
            "notes": {
                "type": "string",
                "description": "Optional new notes (replacing the old); an empty string "
                "removes them.",
            },
        },
        "required": ["id"],
    }

    async def _plan(self, args: dict[str, Any]) -> _Plan:
        event = await self._target(args)
        title = required_text(args, "title")
        if title is not None and len(title) > MAX_TITLE:
            raise _Refused(f"The title is too long (at most {MAX_TITLE} characters).")
        start, end = self._times(event, args)
        location = _cleared_or_text(args, "location")
        notes = _cleared_or_text(args, "notes")
        changes = EventChanges(
            title=title if title is not None and title != event.title else None,
            start=start if start != event.start else None,
            end=end if end != event.end else None,
            location=location if location is not None and location != event.location else None,
            notes=notes if notes is not None and notes != event.notes else None,
        )
        if changes == EventChanges():
            raise _Refused(
                f"Nothing would change in {_title(event)}: pass the new title, start, end, "
                "length, place or notes the user asked for."
            )
        return _Plan(event, changes)

    def _times(self, event: StoredEvent, args: dict[str, Any]) -> tuple[datetime, datetime]:
        """The event's start and end after the change (unchanged when none is asked)."""
        start_text = required_text(args, "start")
        end_text = required_text(args, "end")
        minutes = _minutes(args.get("duration_minutes"))
        if minutes == -1:
            raise _Refused("'duration_minutes' must be a whole number of minutes, at least 1.")
        if end_text is not None and minutes is not None:
            raise _Refused("Pass either 'end' or 'duration_minutes', not both.")
        if start_text is None and end_text is None and minutes is None:
            return event.start, event.end
        now = self._now()
        if event.all_day:
            first, last = event.start, event.end
            if minutes is not None:
                raise _Refused("An all-day event takes an 'end' day, not 'duration_minutes'.")
            if start_text is not None:
                start = resolve(start_text, now)
                if start.kind != "days" or start.days != 1:
                    raise _Refused(
                        f"{_title(event)} is an all-day event: 'start' must be one day without a "
                        f"time; {start_text!r} is not. Changing it to a timed event is not "
                        "supported; the user can do that in the Calendar app."
                    )
                first = start.start
                moved = first.date() - event.start.date()
                last = midnight(event.end.date() + moved, self._zone)
            if end_text is not None:
                end = resolve(end_text, now, base=first.date())
                if end.kind != "days":
                    raise _Refused(
                        "For an all-day event, 'end' must be the last day, without a time."
                    )
                last = end.end
            if start_text is not None and first.date() < now.date():
                raise _Refused(
                    f"{day_text(first.date())} is in the past; check the date with the user."
                )
        else:
            first, last = event.start, event.end
            if start_text is not None:
                start = resolve(start_text, now)
                if start.kind == "days" and start.days == 1:
                    first = local(start.start.date(), event.start.time(), self._zone)
                elif start.is_point:
                    first = start.start
                else:
                    what = "a part of the day" if start.kind == "part" else "several days"
                    raise _Refused(
                        f"{start_text!r} is {what} ({span_text(start)}), not a start time. Ask "
                        "the user what time it should start."
                    )
                # Keep the length (in real time) unless a new end or length is given.
                length = event.end.astimezone(UTC) - event.start.astimezone(UTC)
                last = (first.astimezone(UTC) + length).astimezone(self._zone)
                if first < now - timedelta(minutes=1):
                    raise _Refused(
                        f"{moment_text(first)} is in the past; check the date and time with the "
                        "user."
                    )
            if end_text is not None:
                end = resolve(end_text, now, base=first.date())
                if not end.is_point:
                    raise _Refused(f"'end' must be a time, such as '16:00'; {end_text!r} is not.")
                last = end.start
            elif minutes is not None:
                last = (first.astimezone(UTC) + timedelta(minutes=minutes)).astimezone(self._zone)
        if last <= first:
            raise _Refused(
                f"The end ({moment_text(last)}) would not be after the start "
                f"({moment_text(first)})."
            )
        if last - first > timedelta(days=MAX_EVENT_DAYS):
            raise _Refused(
                f"That event would last more than {MAX_EVENT_DAYS} days; check with the user."
            )
        return first, last

    async def _question(self, args: dict[str, Any]) -> ToolOutput:
        plan = await self._plan(args)
        event, changes = plan.event, plan.changes
        old_when = when_text(event.start, event.end, event.all_day)
        clauses: list[str] = []
        if changes.start is not None or changes.end is not None:
            new_when = when_text(
                changes.start or event.start, changes.end or event.end, event.all_day
            )
            head = f"Move {_title(event)} from {old_when}, to {new_when}"
            if changes.title is not None:
                clauses.append(f"rename it to {_quote(changes.title)}")
        elif changes.title is not None:
            head = f"Rename {_title(event)} on {old_when}, to {_quote(changes.title)}"
        else:
            head = f"Change {_title(event)} on {old_when}:"
        clauses.extend(_detail_clauses(event, changes))
        if not clauses:
            text = head
        elif head.endswith(":"):
            text = f"{head} {_join(clauses)}"
        else:
            text = f"{head}, and {_join(clauses)}"
        return ToolOutput(content=f"{text}{_occurrence_note(event)}?")

    async def _apply(self, args: dict[str, Any]) -> ToolOutput:
        plan = await self._plan(args)
        before = plan.event
        saved: StoredEvent = await self._call(self._store.update, before, plan.changes)
        lines = [
            f"Changed in the calendar {saved.calendar}: {saved.title or '(no title)'}, "
            f"{when_text(saved.start, saved.end, saved.all_day)} (the user's local time)."
        ]
        was: list[str] = []
        if saved.title != before.title:
            was.append(f"title was {before.title or '(no title)'!r}")
        if (saved.start, saved.end) != (before.start, before.end):
            was.append(f"time was {when_text(before.start, before.end, before.all_day)}")
        if saved.location != before.location:
            was.append(f"place now {saved.location!r}" if saved.location else "place removed")
        if saved.notes != before.notes:
            was.append("notes replaced" if saved.notes else "notes removed")
        if was:
            lines.append("Changes: " + "; ".join(was) + ".")
        if before.recurring:
            lines.append(
                "Only this occurrence changed; the rest of the repeating event is as it was."
            )
        old_id, new_id = event_id(before), event_id(saved)
        if new_id != old_id:
            lines.append(f"Its event id is now {new_id} (it was {old_id}).")
        else:
            lines.append(f"Event id: {new_id}.")
        return ToolOutput(content=cap(" ".join(lines), self.settings.max_chars))


def _cleared_or_text(args: dict[str, Any], name: str) -> str | None:
    """The new text for `name`: None if not given, "" to clear it."""
    value = args.get(name)
    if not isinstance(value, str):
        return None
    return value.strip()[:MAX_TEXT]


def _detail_clauses(event: StoredEvent, changes: EventChanges) -> list[str]:
    clauses: list[str] = []
    if changes.location is not None:
        if not changes.location:
            clauses.append(f"remove the place {_quote(event.location)}")
        elif event.location:
            clauses.append(
                f"change the place from {_quote(event.location)} to {_quote(changes.location)}"
            )
        else:
            clauses.append(f"set the place to {_quote(changes.location)}")
    if changes.notes is not None:
        if not changes.notes:
            clauses.append("remove the notes")
        elif event.notes:
            clauses.append(f"replace the notes with {_quote(changes.notes)}")
        else:
            clauses.append(f"add the notes {_quote(changes.notes)}")
    return clauses


def _join(clauses: list[str]) -> str:
    if len(clauses) <= 2:
        return " and ".join(clauses)
    return ", ".join(clauses[:-1]) + ", and " + clauses[-1]
