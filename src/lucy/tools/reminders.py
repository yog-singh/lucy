"""The reminder tools: `reminders` lists them (read tier), `set_reminder`,
`set_timer` and `cancel_reminder` change them (act tier). They are stored in the
`reminders` table of the Lucy SQLite database, so they survive a restart and are
shared by every Lucy process on that file.

When is the user's words ("tomorrow 9am", "in 20 minutes"), resolved in code
(`calendar_dates.resolve_moment`); every result states the resolved day, date and
time, so the model reads back what was actually set. Ids are "r" and the row id
("r12"); row ids are never reused.

Speaking a reminder when it is due is not done here: `announcement` builds the
words, and `lucy.core.reminders.ReminderService` delivers them (ADR 0020).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any, Final

from lucy.core.config import DEFAULT_MEMORY, RemindersSettings, Settings
from lucy.core.reminders import Reminder, ReminderKind, ReminderStatus, ReminderStore
from lucy.core.types import Risk, ToolOutput
from lucy.memory.sqlite_reminders import SQLiteReminderStore
from lucy.tools.calendar_dates import (
    DateError,
    moment_text,
    parse_offset,
    resolve_moment,
    system_zone,
)
from lucy.tools.common import cap, error, required_text

MAX_TEXT: Final = 300
MAX_TIMER: Final = timedelta(hours=24)
LATE: Final = timedelta(minutes=2)
"""A reminder spoken more than this after it was due says when it was due."""
MAX_ANNOUNCED: Final = 5
_ORDINALS: Final = ("First", "Second", "Third", "Fourth", "Fifth")
_ID: Final = re.compile(r"^r?\s*(\d+)$")


def reminder_database(settings: Settings) -> Path:
    """The database file the reminder tools use: the SQLite memory store's.

    Raises `ValueError` when memory is not the SQLite store on a file."""
    memory = settings.memory or DEFAULT_MEMORY
    path = memory.options.get("path")
    if memory.adapter != "sqlite" or not isinstance(path, str) or not path or path == ":memory:":
        raise ValueError(
            "the reminder tools (reminders, set_reminder, set_timer, cancel_reminder) keep "
            "reminders in the SQLite memory database, so they need [memory] adapter = "
            '"sqlite" with a file path'
        )
    return Path(path)


def duration_text(length: timedelta, *, adjective: bool = False) -> str:
    """e.g. "10 minutes", "1 hour and 30 minutes", "90 seconds"; with `adjective`,
    singular units as in "a 10 minute timer"."""
    seconds = round(length.total_seconds())
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours == 0 and minutes < 2 and secs and seconds < 120:
        minutes, secs = 0, seconds  # "90 seconds" reads better than "1 minute and 30 seconds"
    parts = []
    for value, unit in ((hours, "hour"), (minutes, "minute"), (secs, "second")):
        if value:
            parts.append(f"{value} {unit}" + ("" if adjective or value == 1 else "s"))
    if not parts:
        return "0 seconds"
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _clock_text(when: datetime, now: datetime) -> str:
    """ "17:00" today, else "Sunday 15 November 2026, 17:00"."""
    return f"{when:%H:%M}" if when.date() == now.date() else moment_text(when)


def _until(when: datetime, now: datetime) -> str:
    if when <= now:
        return "due now"
    left = when - now
    rounded = timedelta(minutes=round(left.total_seconds() / 60))
    if rounded < timedelta(minutes=1):
        return "in less than a minute"
    return f"in {duration_text(rounded)}"


def _parse_id(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        match = _ID.match(value.strip().lower())
        if match:
            return int(match.group(1))
    return None


class _ReminderBase:
    """What the reminder tools share: the store, the clock and the user's zone."""

    risk = Risk.ACT

    def __init__(
        self,
        settings: RemindersSettings | None = None,
        *,
        store: ReminderStore,
        clock: Callable[[], datetime] | None = None,
        local_zone: tzinfo | None = None,
        owns_store: bool = True,
    ) -> None:
        self.settings = settings or RemindersSettings()
        self._store = store
        self._owns_store = owns_store
        self._zone = local_zone or system_zone()
        self._clock = clock or (lambda: datetime.now(UTC))

    def _now(self) -> datetime:
        return self._clock().astimezone(self._zone)

    def _local(self, when: datetime) -> datetime:
        return when.astimezone(self._zone)

    def _summary(self, reminder: Reminder, now: datetime) -> str:
        """e.g. "Saturday 14 November 2026, 17:00: call the bank", or for a timer
        "10 minutes, ending 14:32"."""
        due = self._local(reminder.due_at)
        if reminder.kind is ReminderKind.TIMER:
            label = f" for {reminder.text}" if reminder.text else ""
            return f"{duration_text(reminder.length)}{label}, ending {_clock_text(due, now)}"
        return f"{moment_text(due)}: {reminder.text}"

    async def aclose(self) -> None:
        """Close the store, unless it is shared (`owns_store=False`; its owner closes it)."""
        if not self._owns_store:
            return
        close = getattr(self._store, "close", None)
        if close is not None:
            await close()


class RemindersTool(_ReminderBase):
    """The user's pending reminders and timers (risk tier read)."""

    name = "reminders"
    description = (
        "List the user's pending reminders and timers, with their ids and when each is due. "
        "Use it for what reminders or timers they have, how long a timer has left, or to "
        "find the id of one to cancel."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional words to look for in what the reminders are about.",
            }
        },
        "required": [],
    }
    risk = Risk.READ

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        now = self._now()
        found = await self._store.upcoming()
        query = required_text(args, "query")
        if query:
            words = query.casefold().split()
            found = [r for r in found if all(w in r.text.casefold() for w in words)]
        matching = f" matching {query!r}" if query else ""
        if not found:
            return ToolOutput(content=f"No pending reminders or timers{matching}.")
        lines: list[str] = []
        reminders = [r for r in found if r.kind is ReminderKind.REMINDER]
        timers = [r for r in found if r.kind is ReminderKind.TIMER]
        if reminders:
            plural = "s" if len(reminders) != 1 else ""
            lines.append(f"{len(reminders)} reminder{plural}{matching}:")
            for r in reminders:
                due = self._local(r.due_at)
                late = "; overdue, not yet spoken" if due <= now else ""
                lines.append(f"- {moment_text(due)}: {r.text} (id {r.label}{late})")
        if timers:
            plural = "s" if len(timers) != 1 else ""
            lines.append(f"{len(timers)} timer{plural}{matching}:")
            for r in timers:
                due = self._local(r.due_at)
                label = f"{r.text}: " if r.text else ""
                lines.append(
                    f"- {label}{duration_text(r.length)}, ends {_clock_text(due, now)} "
                    f"({_until(due, now)}; id {r.label})"
                )
        lines.append("Times are the user's local time.")
        return ToolOutput(content=cap("\n".join(lines), self.settings.max_chars))


class SetReminderTool(_ReminderBase):
    """Set a one-off reminder (risk tier act)."""

    name = "set_reminder"
    description = (
        "Set a reminder that Lucy speaks when it is due. Only when the user asks to be "
        "reminded. Pass 'when' in the user's own words, e.g. 'tomorrow 9am', 'friday at "
        "17:30', 'in 20 minutes'; do not work times out yourself. If the user gave no time, "
        "ask for one."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "What to remind the user about, short, e.g. 'call the bank'.",
            },
            "when": {
                "type": "string",
                "description": "When, as the user said it, e.g. 'tomorrow 9am', 'in 20 "
                "minutes', 'at 5'.",
            },
        },
        "required": ["text", "when"],
    }

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        text, when = required_text(args, "text"), required_text(args, "when")
        if text is None or when is None:
            return error(
                "set_reminder needs 'text' (what to remind about) and 'when' (the user's "
                "words for the time). If the user gave no time, ask for one."
            )
        if len(text) > MAX_TEXT:
            return error(f"The text is too long (at most {MAX_TEXT} characters).")
        now = self._now()
        try:
            moment = resolve_moment(when, now)
        except DateError as exc:
            return error(str(exc))
        saved = await self._store.add(ReminderKind.REMINDER, text, moment.at)
        due = self._local(saved.due_at)
        soon = f" ({_until(due, now)})" if moment.offset is not None else ""
        result = f"Reminder {saved.label} set for {moment_text(due)}{soon}: {saved.text}."
        if moment.note:
            result += f" {moment.note}"
        return ToolOutput(content=result)


class SetTimerTool(_ReminderBase):
    """Start a countdown timer (risk tier act)."""

    name = "set_timer"
    description = (
        "Start a timer that Lucy announces when it ends. Pass 'duration' as the user said "
        "it, e.g. '10 minutes', 'an hour and a half', 'ninety seconds'. For a time of day "
        "use set_reminder instead."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "duration": {
                "type": "string",
                "description": "How long, as the user said it, e.g. '10 minutes'.",
            },
            "label": {
                "type": "string",
                "description": "Optional short name, e.g. 'pasta'.",
            },
        },
        "required": ["duration"],
    }

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        duration = required_text(args, "duration")
        if duration is None:
            return error("set_timer needs a 'duration', such as '10 minutes'.")
        label = required_text(args, "label") or ""
        if len(label) > MAX_TEXT:
            return error(f"The label is too long (at most {MAX_TEXT} characters).")
        try:
            length = parse_offset(duration)
        except DateError as exc:
            return error(str(exc))
        if length is None:
            return error(
                f"Could not read {duration!r} as a length of time, such as '10 minutes' or "
                "'an hour and a half'. For a time of day, use set_reminder."
            )
        if length > MAX_TIMER:
            return error("A timer can run for at most 24 hours; for longer, set a reminder.")
        now = self._now()
        ends = (now.astimezone(UTC) + length).astimezone(self._zone)
        saved = await self._store.add(ReminderKind.TIMER, label, ends)
        named = f" ({label})" if label else ""
        end = self._local(saved.due_at)
        at = f"at {end:%H:%M}" if end.date() == now.date() else f"on {moment_text(end)}"
        return ToolOutput(
            content=f"Timer {saved.label}{named} set for {duration_text(length)}; it ends {at}."
        )


class CancelReminderTool(_ReminderBase):
    """Cancel a pending reminder or timer (risk tier act)."""

    name = "cancel_reminder"
    description = (
        "Cancel a pending reminder or timer by its id (such as 'r12'), taken from a "
        "reminders, set_reminder or set_timer result in this conversation."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "The reminder or timer id, e.g. 'r12'."}
        },
        "required": ["id"],
    }

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        raw = args.get("id")
        ident = _parse_id(raw)
        if ident is None:
            return error(
                f"{raw!r} is not a reminder id; ids look like 'r12'. Call the reminders tool "
                "for the ids."
            )
        now = self._now()
        cancelled = await self._store.cancel(ident)
        if cancelled is not None:
            return ToolOutput(
                content=f"Cancelled {cancelled.kind.value} r{ident} "
                f"({self._summary(cancelled, now)})."
            )
        existing = await self._store.get(ident)
        if existing is not None and existing.status is ReminderStatus.DELIVERED:
            return error(
                f"{existing.kind.value.capitalize()} r{ident} has already gone off, so there is "
                "nothing to cancel."
            )
        if existing is not None and existing.status is ReminderStatus.CANCELLED:
            return error(f"{existing.kind.value.capitalize()} r{ident} was already cancelled.")
        return error(f"No pending reminder r{ident}. Call the reminders tool for the ids.")


# --- What is said when reminders are due ----------------------------------------


def _due_phrase(due: datetime, now: datetime) -> str:
    """ "at 17:00 today", or "on Friday 13 November at 17:00"."""
    if due.date() == now.date():
        return f"at {due:%H:%M} today"
    return f"on {due:%A} {due.day} {due:%B} at {due:%H:%M}"


def _timer_name(reminder: Reminder) -> str:
    label = f" for {reminder.text}" if reminder.text else ""
    return f"{duration_text(reminder.length, adjective=True)} timer{label}"


def _one(reminder: Reminder, now: datetime) -> str:
    due = reminder.due_at.astimezone(now.tzinfo)
    late = now - due > LATE
    if reminder.kind is ReminderKind.TIMER:
        if late:
            return f"Your {_timer_name(reminder)} ended {_due_phrase(due, now)}."
        return f"Your {_timer_name(reminder)} is done."
    if late:
        return f"A reminder was due {_due_phrase(due, now)}: {reminder.text}."
    return f"Reminder: {reminder.text}."


def _item(reminder: Reminder, now: datetime) -> str:
    """One entry of a list of several, after its ordinal."""
    due = reminder.due_at.astimezone(now.tzinfo)
    when = f", due {_due_phrase(due, now)}" if now - due > LATE else ""
    if reminder.kind is ReminderKind.TIMER:
        return f"{when}: your {_timer_name(reminder)} is done."
    return f"{when}: {reminder.text}."


def announcement(reminders: Sequence[Reminder], now: datetime) -> str:
    """What Lucy says when `reminders` are due, at `now` (aware, in the user's zone).

    Plain spoken sentences: one reminder as "Reminder: call the bank." (or when it
    was due, if late); several numbered, oldest first, at most five, then how many
    more there are."""
    if not reminders:
        return ""
    ordered = sorted(reminders, key=lambda r: (r.due_at, r.id))
    if len(ordered) == 1:
        return _one(ordered[0], now)
    count = len(ordered)
    if all(r.kind is ReminderKind.TIMER for r in ordered):
        opening = f"You have {count} timers done."
    else:
        opening = f"You have {count} reminders."
    parts = [opening]
    for ordinal, reminder in zip(_ORDINALS, ordered[:MAX_ANNOUNCED], strict=False):
        parts.append(f"{ordinal}{_item(reminder, now)}")
    if count > MAX_ANNOUNCED:
        parts.append(f"And {count - MAX_ANNOUNCED} more; ask me for your reminders.")
    return " ".join(parts)


def build_store(settings: Settings, clock: Callable[[], datetime]) -> SQLiteReminderStore:
    """A reminder store on the memory database (raises `ValueError` if there is none)."""
    return SQLiteReminderStore(reminder_database(settings), clock=clock)


REMINDER_TOOLS: Final = frozenset({"reminders", "set_reminder", "set_timer", "cancel_reminder"})


def local_announcement(reminders: Sequence[Reminder], now: datetime) -> str:
    """`announcement` with `now` in the system's zone (for `ReminderService`)."""
    return announcement(reminders, now.astimezone(system_zone()))
