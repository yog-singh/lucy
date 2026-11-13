"""The calendar store behind the calendar tools: a small protocol, and the macOS
implementation on EventKit (the store the Calendar app uses).

The tools only see `CalendarStore`, so their logic is tested with a fake and never
needs calendar permission. `EventKitStore` imports EventKit (pyobjc) lazily, on
the first call, so Lucy starts without it when the tools are off or on another
system. Its methods block; the tools call them on one worker thread
(`run_blocking`), never on the event loop.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo
from enum import StrEnum
from typing import Any, Protocol

from lucy.tools.calendar_dates import midnight


class Access(StrEnum):
    """What macOS allows this process to do with the calendar."""

    FULL = "full"
    WRITE_ONLY = "write_only"
    DENIED = "denied"
    RESTRICTED = "restricted"
    NOT_DETERMINED = "not_determined"
    UNAVAILABLE = "unavailable"
    """No EventKit here (not macOS, or pyobjc-framework-EventKit not installed)."""


@dataclass(frozen=True)
class StoredEvent:
    """One event (one occurrence, for a repeating event) as the store holds it."""

    uid: str
    """The store's identifier of the event; shared by all occurrences of a repeating one."""
    title: str
    start: datetime
    """Timezone-aware. For an all-day event, local midnight of the first day."""
    end: datetime
    """Exclusive. For an all-day event, local midnight after the last day."""
    all_day: bool
    calendar: str
    location: str = ""
    notes: str = ""
    recurring: bool = False
    calendar_writable: bool = True
    """False when the event's calendar does not allow changes (subscribed, holidays)."""
    invited_by: str | None = None
    """Who organised it, when that is someone else (an invitation); None otherwise."""


@dataclass(frozen=True)
class EventChanges:
    """What to change in one occurrence; None leaves a field as it is. `end` is
    exclusive, as in `StoredEvent`; an empty location or notes clears it."""

    title: str | None = None
    start: datetime | None = None
    end: datetime | None = None
    location: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class NewEvent:
    title: str
    start: datetime
    end: datetime
    """Exclusive, as in `StoredEvent`."""
    all_day: bool
    calendar: str
    location: str = ""
    notes: str = ""


@dataclass(frozen=True)
class CalendarInfo:
    name: str
    writable: bool


class CalendarStoreError(Exception):
    """The store refused or failed; the message is for the model."""


class CalendarStore(Protocol):
    """A calendar store. Every method blocks."""

    def access(self) -> Access: ...

    def request_access(self, timeout: float) -> Access:
        """Ask macOS for full access (it shows a dialog once); the resulting access."""
        ...

    def calendars(self) -> list[CalendarInfo]: ...

    def default_calendar(self) -> str | None:
        """The calendar new events go to by default, if the store has one."""
        ...

    def events(
        self, start: datetime, end: datetime, calendars: Sequence[str] | None
    ) -> list[StoredEvent]:
        """Events overlapping [start, end), in the named calendars (None: all)."""
        ...

    def add(self, event: NewEvent) -> StoredEvent: ...

    def update(self, event: StoredEvent, changes: EventChanges) -> StoredEvent:
        """Change the one occurrence `event` is (never the rest of a repeating
        series); the event as saved. Raises `CalendarStoreError` if it is gone."""
        ...

    def remove(self, event: StoredEvent) -> None:
        """Delete the one occurrence `event` is (never the rest of a repeating series).
        Raises `CalendarStoreError` if it is gone."""
        ...


_EXECUTOR: ThreadPoolExecutor | None = None


async def run_blocking[T](fn: Callable[..., T], *args: Any) -> T:
    """Run `fn` on the calendar worker thread (one thread, so EventKit objects stay
    on the thread that made them)."""
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lucy-calendar")
    return await asyncio.get_running_loop().run_in_executor(_EXECUTOR, fn, *args)


class EventKitStore:
    """The macOS calendar store, through EventKit. Use one instance per process."""

    def __init__(self, zone: tzinfo) -> None:
        self._zone = zone
        self._ek: Any = None
        self._store: Any = None

    def _eventkit(self) -> Any:
        if self._ek is None:
            import EventKit  # type: ignore[import-not-found]

            self._ek = EventKit
        return self._ek

    def _event_store(self) -> Any:
        if self._store is None:
            self._store = self._eventkit().EKEventStore.alloc().init()
        return self._store

    def access(self) -> Access:
        try:
            ek = self._eventkit()
        except ImportError:
            return Access.UNAVAILABLE
        status = ek.EKEventStore.authorizationStatusForEntityType_(ek.EKEntityTypeEvent)
        return {
            ek.EKAuthorizationStatusFullAccess: Access.FULL,
            ek.EKAuthorizationStatusWriteOnly: Access.WRITE_ONLY,
            ek.EKAuthorizationStatusDenied: Access.DENIED,
            ek.EKAuthorizationStatusRestricted: Access.RESTRICTED,
            ek.EKAuthorizationStatusNotDetermined: Access.NOT_DETERMINED,
        }.get(status, Access.DENIED)

    def request_access(self, timeout: float) -> Access:
        try:
            ek = self._eventkit()
        except ImportError:
            return Access.UNAVAILABLE
        done = threading.Event()

        def answered(granted: bool, error: Any) -> None:
            done.set()

        store = self._event_store()
        if hasattr(store, "requestFullAccessToEventsWithCompletion_"):  # macOS 14+
            store.requestFullAccessToEventsWithCompletion_(answered)
        else:
            store.requestAccessToEntityType_completion_(ek.EKEntityTypeEvent, answered)
        done.wait(timeout)
        access = self.access()
        if access is Access.FULL:
            self._store = None  # a store made before access was granted sees no calendars
        return access

    def _calendar_objects(self) -> list[Any]:
        ek = self._eventkit()
        return list(self._event_store().calendarsForEntityType_(ek.EKEntityTypeEvent) or [])

    def calendars(self) -> list[CalendarInfo]:
        return [
            CalendarInfo(str(c.title()), bool(c.allowsContentModifications()))
            for c in self._calendar_objects()
        ]

    def default_calendar(self) -> str | None:
        calendar = self._event_store().defaultCalendarForNewEvents()
        return str(calendar.title()) if calendar is not None else None

    def events(
        self, start: datetime, end: datetime, calendars: Sequence[str] | None
    ) -> list[StoredEvent]:
        store = self._event_store()
        chosen = None
        if calendars is not None:
            wanted = {name.casefold() for name in calendars}
            chosen = [c for c in self._calendar_objects() if str(c.title()).casefold() in wanted]
            if not chosen:
                return []
        predicate = store.predicateForEventsWithStartDate_endDate_calendars_(
            _nsdate(start), _nsdate(end), chosen
        )
        found = [self._convert(e) for e in (store.eventsMatchingPredicate_(predicate) or [])]
        return sorted(found, key=lambda e: (e.start, e.title))

    def add(self, event: NewEvent) -> StoredEvent:
        ek = self._eventkit()
        store = self._event_store()
        target = next(
            (c for c in self._calendar_objects() if str(c.title()) == event.calendar), None
        )
        if target is None:
            raise CalendarStoreError(f"There is no calendar called {event.calendar!r}.")
        new = ek.EKEvent.eventWithEventStore_(store)
        new.setTitle_(event.title)
        new.setAllDay_(event.all_day)
        new.setStartDate_(_nsdate(event.start))
        # EventKit's all-day events end on the last day, not at the next midnight.
        end = event.end - timedelta(seconds=1) if event.all_day else event.end
        new.setEndDate_(_nsdate(end))
        if event.location:
            new.setLocation_(event.location)
        if event.notes:
            new.setNotes_(event.notes)
        new.setCalendar_(target)
        ok, error = store.saveEvent_span_commit_error_(new, ek.EKSpanThisEvent, True, None)
        if not ok:
            reason = error.localizedDescription() if error is not None else "unknown reason"
            raise CalendarStoreError(f"The calendar did not save the event: {reason}.")
        return self._convert(new)

    def _occurrence(self, event: StoredEvent) -> Any:
        """The EventKit object of the occurrence `event` was read from: same
        identifier, same start."""
        store = self._event_store()
        predicate = store.predicateForEventsWithStartDate_endDate_calendars_(
            _nsdate(event.start - timedelta(days=1)), _nsdate(event.start + timedelta(days=1)), None
        )
        for found in store.eventsMatchingPredicate_(predicate) or []:
            converted = self._convert(found)
            if converted.uid == event.uid and converted.start == event.start:
                return found
        raise CalendarStoreError(
            "That event is no longer in the calendar as it was; list the calendar again."
        )

    def _save(self, item: Any, what: str) -> None:
        ek = self._eventkit()
        ok, error = self._event_store().saveEvent_span_commit_error_(
            item, ek.EKSpanThisEvent, True, None
        )
        if not ok:
            reason = error.localizedDescription() if error is not None else "unknown reason"
            raise CalendarStoreError(f"The calendar did not {what}: {reason}.")

    def update(self, event: StoredEvent, changes: EventChanges) -> StoredEvent:
        item = self._occurrence(event)
        if changes.title is not None:
            item.setTitle_(changes.title)
        if changes.start is not None:
            item.setStartDate_(_nsdate(changes.start))
        if changes.end is not None:
            # EventKit's all-day events end on the last day, not at the next midnight.
            end = changes.end - timedelta(seconds=1) if event.all_day else changes.end
            item.setEndDate_(_nsdate(end))
        if changes.location is not None:
            item.setLocation_(changes.location or None)
        if changes.notes is not None:
            item.setNotes_(changes.notes or None)
        # EKSpanThisEvent: only this occurrence; a repeating event's others stay.
        self._save(item, "save the change")
        return self._convert(item)

    def remove(self, event: StoredEvent) -> None:
        ek = self._eventkit()
        item = self._occurrence(event)
        ok, error = self._event_store().removeEvent_span_commit_error_(
            item, ek.EKSpanThisEvent, True, None
        )
        if not ok:
            reason = error.localizedDescription() if error is not None else "unknown reason"
            raise CalendarStoreError(f"The calendar did not delete the event: {reason}.")

    def _convert(self, event: Any) -> StoredEvent:
        start = _from_nsdate(event.startDate()).astimezone(self._zone)
        end = _from_nsdate(event.endDate()).astimezone(self._zone)
        all_day = bool(event.isAllDay())
        if all_day:
            last = (end - timedelta(seconds=1)).date() if end > start else start.date()
            start = midnight(start.date(), self._zone)
            end = midnight(last + timedelta(days=1), self._zone)
        calendar = event.calendar()
        organizer = event.organizer()
        invited_by = None
        if organizer is not None and not organizer.isCurrentUser():
            invited_by = str(organizer.name() or organizer.URL() or "someone else")
        return StoredEvent(
            uid=str(event.eventIdentifier() or event.calendarItemIdentifier() or ""),
            title=str(event.title() or ""),
            start=start,
            end=end,
            all_day=all_day,
            calendar=str(calendar.title()) if calendar is not None else "",
            location=str(event.location() or ""),
            notes=str(event.notes() or ""),
            recurring=bool(event.hasRecurrenceRules()),
            calendar_writable=bool(calendar.allowsContentModifications())
            if calendar is not None
            else False,
            invited_by=invited_by,
        )


def _nsdate(moment: datetime) -> Any:
    from Foundation import NSDate  # type: ignore[import-not-found]

    return NSDate.dateWithTimeIntervalSince1970_(moment.timestamp())


def _from_nsdate(value: Any) -> datetime:
    return datetime.fromtimestamp(float(value.timeIntervalSince1970()), UTC)
