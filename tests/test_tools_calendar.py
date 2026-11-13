"""The calendar tools against a fake store: date resolution in code (relative
forms, ambiguity, midnight, week boundaries, daylight saving), listing, free time,
filtering, all-day events, creation and its validation, permission problems and
result capping. Nothing here touches the real calendar."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from lucy.core.config import CalendarSettings, LLMSettings, Settings, ToolsSettings
from lucy.core.types import Risk
from lucy.tools import build_tools
from lucy.tools.calendar_dates import DateError, resolve, when_text
from lucy.tools.calendar_events import AddCalendarEventTool, CalendarTool, event_id, find_event
from lucy.tools.calendar_store import (
    Access,
    CalendarInfo,
    CalendarStoreError,
    EventChanges,
    NewEvent,
    StoredEvent,
)

LONDON = ZoneInfo("Europe/London")
# Saturday 14 November 2026, 10:30 in London (GMT, UTC+0).
NOW = datetime(2026, 11, 14, 10, 30, tzinfo=LONDON)


def elapsed(start: datetime, end: datetime) -> timedelta:
    """Real time between two moments (subtracting in one zone gives wall-clock time)."""
    return end.astimezone(UTC) - start.astimezone(UTC)


def at(day: int, hour: int, minute: int = 0, month: int = 11) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=LONDON)


class FakeStore:
    def __init__(self, events: Sequence[StoredEvent] = (), access: Access = Access.FULL) -> None:
        self.stored = list(events)
        self._access = access
        self.grant_on_request: Access | None = None
        self.requests = 0
        self.added: list[NewEvent] = []
        self.updates: list[tuple[StoredEvent, EventChanges]] = []
        self.removed: list[StoredEvent] = []
        self.queries: list[tuple[datetime, datetime, Sequence[str] | None]] = []

    def access(self) -> Access:
        return self._access

    def request_access(self, timeout: float) -> Access:
        self.requests += 1
        if self.grant_on_request is not None:
            self._access = self.grant_on_request
        return self._access

    def calendars(self) -> list[CalendarInfo]:
        return [
            CalendarInfo("Home", True),
            CalendarInfo("Work", True),
            CalendarInfo("Holidays", False),
        ]

    def default_calendar(self) -> str | None:
        return "Home"

    def events(
        self, start: datetime, end: datetime, calendars: Sequence[str] | None
    ) -> list[StoredEvent]:
        self.queries.append((start, end, calendars))
        return sorted(
            (
                e
                for e in self.stored
                if e.start < end
                and (e.end > start or e.end == e.start == start)
                and (calendars is None or e.calendar in calendars)
            ),
            key=lambda e: e.start,
        )

    def add(self, event: NewEvent) -> StoredEvent:
        self.added.append(event)
        saved = StoredEvent(
            uid=f"NEW-{len(self.added)}",
            title=event.title,
            start=event.start,
            end=event.end,
            all_day=event.all_day,
            calendar=event.calendar,
            location=event.location,
            notes=event.notes,
        )
        self.stored.append(saved)
        return saved

    def _index(self, event: StoredEvent) -> int:
        """The one occurrence `event` is: same uid and start (like EventKit's this-event
        span, other occurrences of a repeating event are separate entries here)."""
        for i, e in enumerate(self.stored):
            if e.uid == event.uid and e.start == event.start:
                return i
        raise CalendarStoreError("That event is no longer in the calendar as it was.")

    def update(self, event: StoredEvent, changes: EventChanges) -> StoredEvent:
        i = self._index(event)
        self.updates.append((event, changes))
        old = self.stored[i]
        saved = replace(
            old,
            title=old.title if changes.title is None else changes.title,
            start=changes.start or old.start,
            end=changes.end or old.end,
            location=old.location if changes.location is None else changes.location,
            notes=old.notes if changes.notes is None else changes.notes,
        )
        self.stored[i] = saved
        return saved

    def remove(self, event: StoredEvent) -> None:
        del self.stored[self._index(event)]
        self.removed.append(event)


def ev(title: str, start: datetime, end: datetime, **kw: object) -> StoredEvent:
    fields = {"uid": f"UID-{title}", "calendar": "Home", "all_day": False} | kw
    return StoredEvent(title=title, start=start, end=end, **fields)  # type: ignore[arg-type]


EVENTS = [
    ev("Dentist", at(17, 15), at(17, 16), location="1 High Street"),
    ev("Standup", at(20, 9), at(20, 9, 30), calendar="Work", recurring=True, uid="STANDUP"),
    ev("Lunch with Sam", at(20, 13), at(20, 14), notes="Bring the book"),
    ev("Review", at(20, 15), at(20, 16), calendar="Work"),
    ev("Conference", at(23, 0), at(25, 0), all_day=True),
]


def tools(store: FakeStore, **settings: object) -> tuple[CalendarTool, AddCalendarEventTool]:
    s = CalendarSettings(**settings)  # type: ignore[arg-type]
    kw = {"store": store, "clock": lambda: NOW, "local_zone": LONDON}
    return CalendarTool(s, **kw), AddCalendarEventTool(s, **kw)  # type: ignore[arg-type]


async def read(args: dict[str, object], store: FakeStore | None = None, **settings: object) -> str:
    result = await tools(store or FakeStore(EVENTS), **settings)[0].run(args)
    assert not result.is_error, result.content
    return result.content


async def add(args: dict[str, object], store: FakeStore | None = None) -> str:
    result = await tools(store or FakeStore(EVENTS))[1].run(args)
    assert not result.is_error, result.content
    return result.content


async def add_error(args: dict[str, object], store: FakeStore | None = None) -> str:
    result = await tools(store or FakeStore(EVENTS))[1].run(args)
    assert result.is_error, result.content
    return result.content


# --- Date resolution ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("today", at(14, 0), at(15, 0)),
        ("Tomorrow", at(15, 0), at(16, 0)),
        ("the day after tomorrow", at(16, 0), at(17, 0)),
        ("yesterday", at(13, 0), at(14, 0)),
        ("saturday", at(14, 0), at(15, 0)),  # today is Saturday: today
        ("friday", at(20, 0), at(21, 0)),
        ("this Friday", at(20, 0), at(21, 0)),
        ("fri", at(20, 0), at(21, 0)),
        ("this week", at(14, 0), at(16, 0)),  # today to Sunday (weeks start on Monday)
        ("next week", at(16, 0), at(23, 0)),
        ("this weekend", at(14, 0), at(16, 0)),
        ("20 November", at(20, 0), at(21, 0)),
        ("November 20th", at(20, 0), at(21, 0)),
        ("2026-11-20", at(20, 0), at(21, 0)),
        ("friday afternoon", at(20, 12), at(20, 18)),
        ("tonight", at(14, 18), at(15, 0)),
        ("tomorrow morning", at(15, 6), at(15, 12)),
    ],
)
def test_day_forms_resolve_to_local_ranges(text: str, start: datetime, end: datetime) -> None:
    span = resolve(text, NOW)
    assert (span.start, span.end) == (start, end)


@pytest.mark.parametrize(
    ("text", "moment"),
    [
        ("friday 3pm", at(20, 15)),
        ("Friday at 15:30", at(20, 15, 30)),
        ("3pm friday", at(20, 15)),
        ("tomorrow at noon", at(15, 12)),
        ("2026-11-20T15:00", at(20, 15)),
        ("2026-11-20 15:00", at(20, 15)),
        ("2026-11-20 3pm", at(20, 15)),
        ("2026-11-20T15:00:00Z", at(20, 15)),  # with an offset: converted to local time
        ("2026-11-20T15:00:00+01:00", at(20, 14)),
        ("4pm", at(14, 16)),  # a bare time is today
    ],
)
def test_day_and_time_forms_resolve_to_a_moment(text: str, moment: datetime) -> None:
    span = resolve(text, NOW)
    assert span.is_point and span.start == moment


def test_month_day_without_a_year_is_the_nearest_such_date() -> None:
    assert resolve("2 January", NOW).start.date() == date(2027, 1, 2)
    assert resolve("10 November", NOW).start.date() == date(2026, 11, 10)
    assert resolve("10 November 2027", NOW).start.date() == date(2027, 11, 10)


def test_week_boundaries_from_sunday_and_monday() -> None:
    sunday = datetime(2026, 11, 15, 23, 59, tzinfo=LONDON)
    assert resolve("this week", sunday).end == at(16, 0)
    assert resolve("next week", sunday).start == at(16, 0)
    assert resolve("sunday", sunday).start == at(15, 0)  # today
    monday = datetime(2026, 11, 16, 0, 0, tzinfo=LONDON)  # just after midnight
    assert resolve("today", monday).start == at(16, 0)
    assert resolve("next week", monday).start == at(23, 0)
    assert resolve("monday", monday).start == at(16, 0)


@pytest.mark.parametrize(
    ("text", "hint"),
    [
        ("next friday", "Friday 20 November 2026 or Friday 27 November 2026"),
        ("next weekend", "Saturday 21 November 2026"),
        ("11/12", "ambiguous"),
        ("sometime soon", "Accepted:"),
        ("31 February", "not a date"),
        ("next week 3pm", "several days"),
        ("friday 25:00", "Accepted:"),
    ],
)
def test_ambiguous_or_unreadable_input_is_refused(text: str, hint: str) -> None:
    with pytest.raises(DateError, match=hint):
        resolve(text, NOW)


def test_daylight_saving_days_and_skipped_hours() -> None:
    before = datetime(2026, 3, 27, 12, tzinfo=LONDON)  # Friday; clocks go forward Sunday 29
    sunday = resolve("sunday", before)
    assert elapsed(sunday.start, sunday.end) == timedelta(hours=23)
    with pytest.raises(DateError, match="does not exist"):
        resolve("sunday 1:30", before)
    autumn = resolve("2026-10-25", NOW)  # clocks go back
    assert elapsed(autumn.start, autumn.end) == timedelta(hours=25)


def test_when_text_states_weekdays_and_handles_midnight_and_all_day() -> None:
    assert when_text(at(17, 15), at(17, 16), False) == "Tuesday 17 November 2026, 15:00 to 16:00"
    assert when_text(at(17, 23), at(18, 0), False) == "Tuesday 17 November 2026, 23:00 to midnight"
    assert when_text(at(17, 23), at(18, 1), False) == (
        "Tuesday 17 November 2026, 23:00 to Wednesday 18 November 2026, 01:00"
    )
    assert when_text(at(23, 0), at(24, 0), True) == "Monday 23 November 2026, all day"
    assert when_text(at(23, 0), at(25, 0), True) == (
        "Monday 23 November 2026 to Tuesday 24 November 2026, all day"
    )


# --- Listing -----------------------------------------------------------------


async def test_listing_a_day_states_dates_ids_and_free_time() -> None:
    text = await read({"start": "friday"})
    assert text.splitlines() == [
        "Calendar for Friday 20 November 2026 (the user's local time), 3 events:",
        "- Friday 20 November 2026, 09:00 to 09:30: Standup (calendar Work; repeating; "
        f"id {event_id(EVENTS[1])})",
        "- Friday 20 November 2026, 13:00 to 14:00: Lunch with Sam (calendar Home; notes: "
        f"Bring the book; id {event_id(EVENTS[2])})",
        "- Friday 20 November 2026, 15:00 to 16:00: Review (calendar Work; "
        f"id {event_id(EVENTS[3])})",
        "Free time between 08:00 and 20:00: 08:00 to 09:00, 09:30 to 13:00, 14:00 to 15:00, "
        "16:00 to 20:00.",
    ]


async def test_free_on_friday_afternoon() -> None:
    text = await read({"start": "friday afternoon"})
    assert text.startswith("Calendar for Friday 20 November 2026, 12:00 to 18:00")
    assert "Standup" not in text
    assert text.endswith(
        "Free time between 12:00 and 18:00: 12:00 to 13:00, 14:00 to 15:00, 16:00 to 18:00."
    )


async def test_an_empty_afternoon_is_free_the_whole_time() -> None:
    text = await read({"start": "tomorrow afternoon"})
    assert "no events" in text
    assert "Free the whole time between 12:00 and 18:00." in text


async def test_next_event_from_now_looks_two_weeks_ahead() -> None:
    store = FakeStore(EVENTS)
    text = await read({"start": "now"}, store)
    first_line, next_line = text.splitlines()[:2]
    assert first_line.startswith(
        "Calendar from now (Saturday 14 November 2026, 10:30) to Saturday 28 November 2026"
    )
    assert "Dentist" in next_line and "Tuesday 17 November 2026, 15:00" in next_line
    assert store.queries[0][1] == at(29, 0)
    assert "Free" not in text


async def test_range_with_end_day_included_and_all_day_events() -> None:
    text = await read({"start": "monday", "end": "2026-11-24"})
    assert text.startswith("Calendar for Monday 16 November 2026 to Tuesday 24 November 2026")
    assert "Monday 23 November 2026 to Tuesday 24 November 2026, all day: Conference" in text


async def test_all_day_events_are_not_busy_time() -> None:
    text = await read({"start": "2026-11-23"})
    assert "all day: Conference" in text
    assert "Free the whole time between 08:00 and 20:00 (all-day events not counted as busy)." in (
        text
    )


async def test_query_filters_titles_places_and_notes() -> None:
    text = await read({"start": "today", "end": "2026-11-30", "query": "book"})
    assert "Lunch with Sam" in text and "Dentist" not in text and "matching 'book'" in text
    text = await read({"start": "today", "end": "2026-11-30", "query": "high street"})
    assert "Dentist" in text and "Lunch" not in text
    text = await read({"start": "today", "end": "2026-11-30", "query": "piano"})
    assert "no events matching 'piano'" in text


async def test_configured_calendars_limit_the_listing() -> None:
    store = FakeStore(EVENTS)
    text = await read({"start": "friday"}, store, calendars=["Work"])
    assert store.queries[0][2] == ["Work"]
    assert "Lunch" not in text and "Review" in text


async def test_bad_ranges_are_errors_that_say_what_was_resolved() -> None:
    reading = tools(FakeStore(EVENTS))[0]
    result = await reading.run({"start": "friday", "end": "monday"})
    # "monday" is the coming Monday (16 November), before Friday.
    assert result.is_error and "is not after the start (Friday 20 November 2026" in result.content
    result = await reading.run({"start": "today", "end": "2027-06-01"})
    assert result.is_error and "too long" in result.content
    result = await reading.run({"start": "next friday"})
    assert result.is_error and "ambiguous" in result.content


async def test_results_are_capped_to_max_chars() -> None:
    many = [ev(f"Meeting {i} " + "x" * 40, at(20, 8), at(20, 9), uid=f"u{i}") for i in range(60)]
    text = await read({"start": "friday"}, FakeStore(many), max_chars=500)
    assert len(text) <= 500 and text.endswith(
        "[Truncated: the rest was cut to keep the result short.]"
    )


# --- Ids ---------------------------------------------------------------------


def test_ids_are_short_stable_and_name_the_occurrence() -> None:
    first = event_id(EVENTS[1])
    assert len(first) == 15 and first.endswith("-20261120")
    assert event_id(EVENTS[1]) == first
    next_week = replace(EVENTS[1], start=at(27, 9), end=at(27, 9, 30))
    assert event_id(next_week).split("-")[0] == first.split("-")[0]
    assert event_id(next_week) != first


def test_find_event_lists_only_the_day_in_the_id() -> None:
    store = FakeStore(EVENTS)
    found = find_event(store, event_id(EVENTS[2]).upper(), LONDON)
    assert found == EVENTS[2]
    assert store.queries == [(at(20, 0), at(21, 0), None)]
    assert find_event(store, "zzzzzz-20261120", LONDON) is None
    assert find_event(store, "not an id", LONDON) is None


# --- Creating ----------------------------------------------------------------


async def test_add_states_exactly_what_was_created() -> None:
    store = FakeStore(EVENTS)
    text = await add(
        {"title": "Haircut", "start": "friday 11am", "duration_minutes": 45, "location": "Barber"},
        store,
    )
    assert store.added == [
        NewEvent("Haircut", at(20, 11), at(20, 11, 45), False, "Home", "Barber", "")
    ]
    assert text == (
        "Added to the calendar Home: Haircut, Friday 20 November 2026, 11:00 to 11:45 (the "
        "user's local time). Place: Barber. Event id: "
        f"{event_id(store.stored[-1])}."
    )


async def test_add_defaults_to_one_hour_and_reports_overlaps() -> None:
    text = await add({"title": "Call", "start": "friday 13:30"})
    assert "Friday 20 November 2026, 13:30 to 14:30" in text
    assert "it lasts one hour" in text
    assert "It overlaps with: Lunch with Sam, Friday 20 November 2026, 13:00 to 14:00." in text


async def test_add_with_end_time_and_named_calendar() -> None:
    store = FakeStore()
    text = await add(
        {"title": "Planning", "start": "monday 9:00", "end": "10:30", "calendar": "work"}, store
    )
    assert store.added[0].calendar == "Work"
    assert "calendar Work: Planning, Monday 16 November 2026, 09:00 to 10:30" in text


async def test_add_all_day_events() -> None:
    store = FakeStore()
    text = await add(
        {"title": "Holiday", "start": "2026-12-24", "end": "2026-12-26", "all_day": True}, store
    )
    assert store.added[0].start == at(24, 0, month=12) and store.added[0].end == at(27, 0, month=12)
    assert "Thursday 24 December 2026 to Saturday 26 December 2026, all day" in text
    text = await add({"title": "Birthday", "start": "tomorrow", "all_day": True}, store)
    assert "Sunday 15 November 2026, all day" in text


async def test_add_over_a_daylight_saving_change_keeps_the_real_length() -> None:
    store = FakeStore()
    await add({"title": "Night shift", "start": "2027-03-28T00:30", "duration_minutes": 120}, store)
    added = store.added[0]
    assert elapsed(added.start, added.end) == timedelta(hours=2)
    assert added.end.hour == 3 and added.end.minute == 30  # 01:00 to 02:00 does not exist


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"title": "Lunch", "start": "friday"}, "not a start time. Ask the user"),
        ({"title": "Lunch", "start": "friday afternoon"}, "a part of the day"),
        ({"title": "Lunch", "start": "next friday 1pm"}, "ambiguous"),
        ({"title": "Lunch", "start": "today 9am"}, "in the past"),
        ({"title": "Lunch", "start": "friday 1pm", "end": "12:00"}, "not after the start"),
        (
            {"title": "Lunch", "start": "friday 1pm", "end": "14:00", "duration_minutes": 30},
            "not both",
        ),
        ({"title": "Lunch", "start": "friday 1pm", "duration_minutes": 0}, "at least 1"),
        ({"title": "Lunch", "start": "friday 1pm", "calendar": "Holidays"}, "Home, Work"),
        ({"title": "Lunch", "start": "friday 1pm", "calendar": "Gym"}, "no calendar called 'Gym'"),
        ({"title": "Trip", "start": "friday 1pm", "all_day": True}, "one day without a time"),
        (
            {"title": "Trip", "start": "friday", "duration_minutes": 60, "all_day": True},
            "not 'duration_minutes'",
        ),
        ({"title": "Lunch"}, "needs a 'title' and a 'start'"),
        ({"title": "Long", "start": "friday 1pm", "end": "2026-12-20 9am"}, "more than 14 days"),
    ],
)
async def test_add_validation_errors(args: dict[str, object], message: str) -> None:
    store = FakeStore(EVENTS)
    assert message in await add_error(args, store)
    assert store.added == []


async def test_default_calendar_from_settings() -> None:
    store = FakeStore()
    adding = tools(store, default_calendar="Work")[1]
    result = await adding.run({"title": "Sync", "start": "monday 10am"})
    assert not result.is_error and store.added[0].calendar == "Work"


# --- Permission --------------------------------------------------------------


@pytest.mark.parametrize("access", [Access.DENIED, Access.WRITE_ONLY])
async def test_no_access_gives_how_to_grant_it(access: Access) -> None:
    store = FakeStore(EVENTS, access=access)
    for tool in tools(store):
        result = await tool.run({"title": "x", "start": "friday 1pm"})
        assert result.is_error
        assert "System Settings > Privacy & Security > Calendars" in result.content
    assert store.requests == 0 and store.added == []


async def test_undetermined_access_is_requested_once_per_call() -> None:
    store = FakeStore(EVENTS, access=Access.NOT_DETERMINED)
    store.grant_on_request = Access.FULL
    assert "Standup" in await read({"start": "friday"}, store)
    assert store.requests == 1
    unanswered = FakeStore(EVENTS, access=Access.NOT_DETERMINED)
    result = await tools(unanswered)[0].run({})
    assert result.is_error and "answer the dialog" in result.content
    assert unanswered.requests == 1


async def test_unavailable_and_restricted() -> None:
    for access, text in [(Access.UNAVAILABLE, "pyobjc"), (Access.RESTRICTED, "restricted")]:
        result = await tools(FakeStore(access=access))[0].run({})
        assert result.is_error and text in result.content


# --- Registration ------------------------------------------------------------


def test_tiers_names_and_registration_without_eventkit() -> None:
    reading, adding = tools(FakeStore())
    assert (reading.name, reading.risk) == ("calendar", Risk.READ)
    assert (adding.name, adding.risk) == ("add_calendar_event", Risk.ACT)
    assert adding.parameters["required"] == ["title", "start"]
    settings = Settings(
        llm=LLMSettings(roles={"chat": {"adapter": "fake"}}),
        tools=ToolsSettings(enabled=["calendar", "add_calendar_event"]),
    )
    built = build_tools(settings, clock=lambda: NOW)
    assert [t.name for t in built] == ["calendar", "add_calendar_event"]


def test_utc_clock_is_shown_in_local_time() -> None:
    reading = CalendarTool(
        store=FakeStore(),
        clock=lambda: datetime(2026, 11, 14, 23, 30, tzinfo=UTC),
        local_zone=ZoneInfo("Asia/Tokyo"),
    )
    assert resolve("today", reading._now()).start.date() == date(2026, 11, 15)
