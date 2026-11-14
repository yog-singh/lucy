"""Reminders and timers: moments resolved in code (offsets, number words, parts of
the day, bare hours, refusals, daylight saving), the four tools against a
temporary database, building them from the settings, and what is said when
reminders are due."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from lucy.core.config import AdapterConfig, LLMSettings, Settings, ToolsSettings
from lucy.core.reminders import Reminder, ReminderKind, ReminderStatus
from lucy.core.types import Risk
from lucy.memory.sqlite_reminders import SQLiteReminderStore
from lucy.tools import build_tools
from lucy.tools.calendar_dates import DateError, resolve, resolve_moment
from lucy.tools.reminders import (
    CancelReminderTool,
    RemindersTool,
    SetReminderTool,
    SetTimerTool,
    announcement,
    duration_text,
)

LONDON = ZoneInfo("Europe/London")
# Saturday 14 November 2026, 10:30 in London (GMT, UTC+0).
NOW = datetime(2026, 11, 14, 10, 30, tzinfo=LONDON)


def at(day: int, hour: int, minute: int = 0, month: int = 11) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=LONDON)


def moment(text: str, now: datetime = NOW) -> datetime:
    return resolve_moment(text, now).at


# --- Moments -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "minutes"),
    [
        ("in 20 minutes", 20),
        ("in 2 hours", 120),
        ("in an hour", 60),
        ("half an hour", 30),
        ("in half an hour", 30),
        ("an hour and a half", 90),
        ("one and a half hours", 90),
        ("10 minutes", 10),
        ("10 mins", 10),
        ("in twenty five minutes", 25),
        ("in twenty-five minutes", 25),
        ("sixty minutes", 60),
        ("forty five minutes", 45),
        ("a quarter of an hour", 15),
        ("1 hour and 15 minutes", 75),
        ("1h30", 90),
        ("in 20 minutes time", 20),
        ("20 minutes from now", 20),
    ],
)
def test_offsets_are_added_to_now(text: str, minutes: int) -> None:
    resolved = resolve_moment(text, NOW)
    assert resolved.offset == timedelta(minutes=minutes)
    assert resolved.at == NOW + timedelta(minutes=minutes)


def test_offsets_in_seconds() -> None:
    assert resolve_moment("ninety seconds", NOW).offset == timedelta(seconds=90)


@pytest.mark.parametrize("text", ["in 3 days", "in a week", "in two weeks"])
def test_offsets_in_days_or_weeks_ask_for_a_day_and_time(text: str) -> None:
    with pytest.raises(DateError, match="day and a time"):
        resolve_moment(text, NOW)


def test_offsets_are_real_time_across_a_daylight_saving_change() -> None:
    # 25 October 2026: London clocks go back from 02:00 BST to 01:00 GMT.
    before = datetime(2026, 10, 25, 0, 30, tzinfo=LONDON)
    later = resolve_moment("in 2 hours", before).at
    assert later.astimezone(UTC) - before.astimezone(UTC) == timedelta(hours=2)
    assert (later.hour, later.minute) == (1, 30) and later.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("tomorrow morning at 9", at(15, 9)),
        ("at 9 tomorrow morning", at(15, 9)),
        ("9 in the evening", at(14, 21)),
        ("tomorrow evening at 7", at(15, 19)),
        ("friday afternoon at 3", at(20, 15)),
        ("tonight at 8", at(14, 20)),
        ("tonight at 12", at(15, 0)),
        ("tomorrow 9am", at(15, 9)),
        ("friday 3pm", at(20, 15)),
        ("17:30", at(14, 17, 30)),
        ("2026-11-20T15:00", at(20, 15)),
    ],
)
def test_days_with_times_and_parts_of_the_day(text: str, expected: datetime) -> None:
    resolved = resolve_moment(text, NOW)
    assert resolved.at == expected and resolved.note is None and resolved.offset is None


@pytest.mark.parametrize(
    ("now", "text", "expected"),
    [
        (NOW, "at 5", at(14, 17)),  # 05:00 has passed
        (NOW, "5", at(14, 17)),
        (NOW, "5 o'clock", at(14, 17)),
        (NOW, "at 11", at(14, 11)),  # 11:00 is still to come
        (NOW, "at 12", at(14, 12)),
        (at(14, 8), "at 9", at(14, 9)),  # morning: the sooner reading is the morning
        (at(14, 13), "at 1", at(15, 1)),  # 13:00 just passed: tomorrow 01:00
        (at(14, 23, 30), "at 12", at(15, 0)),  # midnight, not noon tomorrow
        (at(14, 23, 30), "at 11", at(15, 11)),
    ],
)
def test_a_bare_hour_is_the_soonest_future_reading(
    now: datetime, text: str, expected: datetime
) -> None:
    resolved = resolve_moment(text, now)
    assert resolved.at == expected
    assert resolved.note is not None and f"read as {expected:%H:%M}" in resolved.note


@pytest.mark.parametrize(
    ("text", "readings"),
    [
        ("tomorrow at 5", ("05:00", "17:00", "Sunday")),
        ("friday at 5", ("05:00", "17:00", "Friday")),
        ("friday 9", ("09:00", "21:00", "Friday")),
        ("tomorrow at 12", ("noon", "midnight", "Sunday")),
    ],
)
def test_a_bare_hour_on_a_named_day_asks_morning_or_evening(
    text: str, readings: tuple[str, ...]
) -> None:
    with pytest.raises(DateError) as caught:
        resolve_moment(text, NOW)
    message = str(caught.value)
    assert "no am or pm" in message and "morning or the evening" in message
    assert all(reading in message for reading in readings)


def test_the_calendar_still_reads_a_bare_hour_as_morning() -> None:
    assert resolve("5", NOW).start == at(14, 5)


@pytest.mark.parametrize("text", ["9am", "09:00", "yesterday 3pm", "now"])
def test_past_times_are_refused(text: str) -> None:
    with pytest.raises(DateError, match="past|later than now"):
        resolve_moment(text, NOW)


@pytest.mark.parametrize(
    ("text", "what"),
    [
        ("tomorrow", "a day without a time"),
        ("friday", "a day without a time"),
        ("next week", "a day without a time"),
        ("tomorrow morning", "a part of the day"),
        ("this evening", "a part of the day"),
        ("tonight", "a part of the day"),
    ],
)
def test_a_day_or_part_of_day_without_a_time_asks_what_time(text: str, what: str) -> None:
    with pytest.raises(DateError, match=f"{what}.*Ask the user what time"):
        resolve_moment(text, NOW)


def test_unreadable_words_list_what_is_accepted() -> None:
    with pytest.raises(DateError, match="Accepted: a length of time"):
        resolve_moment("whenever", NOW)


def test_a_time_skipped_by_daylight_saving_is_refused() -> None:
    # 28 March 2027: London clocks jump from 01:00 to 02:00.
    with pytest.raises(DateError, match="does not exist"):
        resolve_moment("28 March 2027 1:30am", datetime(2027, 3, 20, 12, tzinfo=LONDON))


# --- Tools ---------------------------------------------------------------------


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SQLiteReminderStore]:
    async with SQLiteReminderStore(tmp_path / "lucy.db", clock=lambda: NOW) as opened:
        yield opened


def tools(store: SQLiteReminderStore, now: datetime = NOW) -> dict[str, object]:
    options = {"store": store, "clock": lambda: now, "local_zone": LONDON}
    return {
        "list": RemindersTool(**options),  # type: ignore[arg-type]
        "set": SetReminderTool(**options),  # type: ignore[arg-type]
        "timer": SetTimerTool(**options),  # type: ignore[arg-type]
        "cancel": CancelReminderTool(**options),  # type: ignore[arg-type]
    }


async def run(tool: object, **args: object) -> tuple[str, bool]:
    output = await tool.run(args)  # type: ignore[attr-defined]
    return output.content, output.is_error


async def test_set_reminder_states_the_resolved_day_date_and_time(
    store: SQLiteReminderStore,
) -> None:
    t = tools(store)
    text, failed = await run(t["set"], text="call the bank", when="today 5pm")
    assert not failed
    assert text == "Reminder r1 set for Saturday 14 November 2026, 17:00: call the bank."
    text, _ = await run(t["set"], text="tea", when="in 20 minutes")
    assert text == "Reminder r2 set for Saturday 14 November 2026, 10:50 (in 20 minutes): tea."
    text, _ = await run(t["set"], text="bins", when="at 5")
    assert text.startswith("Reminder r3 set for Saturday 14 November 2026, 17:00: bins.")
    assert "No am or pm was given" in text


async def test_set_reminder_refusals(store: SQLiteReminderStore) -> None:
    t = tools(store)
    text, failed = await run(t["set"], text="call the bank", when="tomorrow")
    assert failed and "Ask the user what time" in text
    text, failed = await run(t["set"], text="call the bank", when="9am")
    assert failed and "in the past" in text
    text, failed = await run(t["set"], text="call the bank")
    assert failed and "needs 'text'" in text
    assert await store.upcoming() == []


async def test_set_timer(store: SQLiteReminderStore) -> None:
    t = tools(store)
    text, failed = await run(t["timer"], duration="10 minutes")
    assert not failed and text == "Timer r1 set for 10 minutes; it ends at 10:40."
    text, _ = await run(t["timer"], duration="an hour and a half", label="bread")
    assert text == "Timer r2 (bread) set for 1 hour and 30 minutes; it ends at 12:00."
    text, failed = await run(t["timer"], duration="5pm")
    assert failed and "use set_reminder" in text
    text, failed = await run(t["timer"], duration="25 hours")
    assert failed and "at most 24 hours" in text
    late = tools(store, at(14, 23, 55))
    text, _ = await run(late["timer"], duration="ten minutes")
    assert text == "Timer r3 set for 10 minutes; it ends on Sunday 15 November 2026, 00:05."


async def test_listing_and_query(store: SQLiteReminderStore) -> None:
    t = tools(store)
    text, _ = await run(t["list"])
    assert text == "No pending reminders or timers."
    await run(t["set"], text="call the bank", when="today 17:00")
    await run(t["set"], text="passport", when="tomorrow 9am")
    await run(t["timer"], duration="7 minutes")
    text, failed = await run(t["list"])
    assert not failed
    assert text == (
        "2 reminders:\n"
        "- Saturday 14 November 2026, 17:00: call the bank (id r1)\n"
        "- Sunday 15 November 2026, 09:00: passport (id r2)\n"
        "1 timer:\n"
        "- 7 minutes, ends 10:37 (in 7 minutes; id r3)\n"
        "Times are the user's local time."
    )
    text, _ = await run(t["list"], query="Bank")
    assert text == (
        "1 reminder matching 'Bank':\n"
        "- Saturday 14 November 2026, 17:00: call the bank (id r1)\n"
        "Times are the user's local time."
    )
    text, _ = await run(t["list"], query="dentist")
    assert text == "No pending reminders or timers matching 'dentist'."
    later = tools(store, at(14, 10, 35))
    text, _ = await run(later["list"])
    assert "- 7 minutes, ends 10:37 (in 2 minutes; id r3)" in text


async def test_cancel(store: SQLiteReminderStore) -> None:
    t = tools(store)
    await run(t["set"], text="call the bank", when="today 17:00")
    await run(t["timer"], duration="10 minutes")
    text, failed = await run(t["cancel"], id="r1")
    assert not failed
    assert text == "Cancelled reminder r1 (Saturday 14 November 2026, 17:00: call the bank)."
    text, _ = await run(t["cancel"], id="2")
    assert text == "Cancelled timer r2 (10 minutes, ending 10:40)."
    text, failed = await run(t["cancel"], id="r1")
    assert failed and text == "Reminder r1 was already cancelled."
    text, failed = await run(t["cancel"], id="r99")
    assert failed and text == "No pending reminder r99. Call the reminders tool for the ids."
    text, failed = await run(t["cancel"], id="k3f9qz-20261117")
    assert failed and "is not a reminder id" in text


async def test_a_delivered_reminder_cannot_be_cancelled(store: SQLiteReminderStore) -> None:
    t = tools(store)
    await run(t["set"], text="tea", when="in 1 minute")
    await store.claim_due(NOW + timedelta(minutes=1), "x")
    await store.mark_delivered([1], "x", "s", "t")
    text, failed = await run(t["cancel"], id="r1")
    assert failed and text == "Reminder r1 has already gone off, so there is nothing to cancel."


def test_risk_tiers() -> None:
    assert RemindersTool.risk is Risk.READ
    assert SetReminderTool.risk is Risk.ACT
    assert SetTimerTool.risk is Risk.ACT
    assert CancelReminderTool.risk is Risk.ACT


REMINDER_TOOLS = ["reminders", "set_reminder", "set_timer", "cancel_reminder"]


async def test_built_from_settings_on_the_memory_database(tmp_path: Path) -> None:
    db = tmp_path / "lucy.db"
    settings = Settings(
        llm=LLMSettings(roles={"chat": {"adapter": "fake"}}),
        memory=AdapterConfig(adapter="sqlite", options={"path": str(db)}),
        tools=ToolsSettings(enabled=REMINDER_TOOLS),
    )
    built = build_tools(settings, clock=lambda: NOW)
    assert [t.name for t in built] == REMINDER_TOOLS
    output = await built[1].run({"text": "bins", "when": "in 2 hours"})
    assert not output.is_error and db.exists()
    async with SQLiteReminderStore(db) as check:
        assert [r.text for r in await check.upcoming()] == ["bins"]
    for tool in built:
        await tool.aclose()  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "memory",
    [
        AdapterConfig(adapter="sqlite", options={"path": ":memory:"}),
        AdapterConfig(adapter="other", options={"path": "x.db"}),
    ],
)
def test_building_without_a_sqlite_file_fails_clearly(memory: AdapterConfig) -> None:
    settings = Settings(
        llm=LLMSettings(roles={"chat": {"adapter": "fake"}}),
        memory=memory,
        tools=ToolsSettings(enabled=["set_reminder"]),
    )
    with pytest.raises(ValueError, match="SQLite memory database"):
        build_tools(settings)


# --- Announcements ---------------------------------------------------------------


def reminder(
    ident: int,
    text: str,
    due: datetime,
    kind: ReminderKind = ReminderKind.REMINDER,
    created: datetime | None = None,
) -> Reminder:
    created = created or due - timedelta(hours=1)
    return Reminder(
        id=ident,
        kind=kind,
        text=text,
        due_at=due.astimezone(UTC),
        status=ReminderStatus.FIRING,
        created_at=created.astimezone(UTC),
        updated_at=due.astimezone(UTC),
    )


def test_announcing_one_reminder() -> None:
    due = at(14, 17)
    assert announcement([reminder(1, "call the bank", due)], due) == "Reminder: call the bank."
    assert (
        announcement([reminder(1, "call the bank", due)], due + timedelta(minutes=2))
        == "Reminder: call the bank."
    )
    assert (
        announcement([reminder(1, "call the bank", due)], due + timedelta(minutes=3))
        == "A reminder was due at 17:00 today: call the bank."
    )
    assert (
        announcement([reminder(1, "call the bank", at(13, 17))], at(14, 9))
        == "A reminder was due on Friday 13 November at 17:00: call the bank."
    )


def test_announcing_a_timer() -> None:
    due = at(14, 10, 40)
    timer = reminder(1, "", due, ReminderKind.TIMER, created=due - timedelta(minutes=10))
    assert announcement([timer], due) == "Your 10 minute timer is done."
    pasta = reminder(2, "pasta", due, ReminderKind.TIMER, created=due - timedelta(minutes=90))
    assert announcement([pasta], due) == "Your 1 hour and 30 minute timer for pasta is done."


def test_announcing_several_oldest_first() -> None:
    now = at(14, 17, 30)
    due = [
        reminder(2, "passport", at(14, 17, 30)),
        reminder(1, "call the bank", at(14, 17)),
        reminder(3, "", now, ReminderKind.TIMER, created=now - timedelta(minutes=10)),
    ]
    assert announcement(due, now) == (
        "You have 3 reminders. First, due at 17:00 today: call the bank. Second: passport. "
        "Third: your 10 minute timer is done."
    )


def test_announcing_more_than_five() -> None:
    now = at(14, 18)
    due = [reminder(i, f"thing {i}", at(14, 17, i)) for i in range(1, 9)]
    said = announcement(due, now)
    assert said.startswith("You have 8 reminders. First, due at 17:01 today: thing 1.")
    assert "Fifth, due at 17:05 today: thing 5." in said and "thing 6" not in said
    assert said.endswith("And 3 more; ask me for your reminders.")


def test_duration_text() -> None:
    assert duration_text(timedelta(minutes=1)) == "1 minute"
    assert duration_text(timedelta(seconds=90)) == "90 seconds"
    assert duration_text(timedelta(hours=2, minutes=5)) == "2 hours and 5 minutes"
    assert duration_text(timedelta(hours=1, minutes=1, seconds=5)) == (
        "1 hour, 1 minute and 5 seconds"
    )
