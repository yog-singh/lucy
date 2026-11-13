"""The edit and delete calendar tools against the fake store: preview questions,
refusals, keeping the length on a move, one occurrence of a repeating event, run
results and ids, the midnight case, and the confirmation flow end to end through
the orchestrator. Nothing here touches the real calendar."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from lucy.core.events import ConfirmationRequested, ConfirmationResolved
from lucy.core.types import Risk, ToolCall
from lucy.tools.calendar_changes import DeleteCalendarEventTool, EditCalendarEventTool
from lucy.tools.calendar_events import event_id
from lucy.tools.calendar_store import EventChanges
from tests.test_orchestrator import Harness, calls, text
from tests.test_tools_calendar import EVENTS, LONDON, NOW, FakeStore, at, ev

DENTIST = EVENTS[0]
STANDUP = EVENTS[1]
STANDUP_NEXT = ev(
    "Standup", at(27, 9), at(27, 9, 30), calendar="Work", recurring=True, uid="STANDUP"
)
HOLIDAY = ev("Bank holiday", at(16, 0), at(17, 0), all_day=True, calendar="Holidays",
             calendar_writable=False)  # fmt: skip
INVITE = ev("Board meeting", at(18, 11), at(18, 12), invited_by="Alex Smith")
CONFERENCE = EVENTS[4]


def store() -> FakeStore:
    return FakeStore([*EVENTS, STANDUP_NEXT, HOLIDAY, INVITE])


def make(s: FakeStore, clock: Any = None) -> tuple[EditCalendarEventTool, DeleteCalendarEventTool]:
    kw: dict[str, Any] = {"store": s, "clock": clock or (lambda: NOW), "local_zone": LONDON}
    return EditCalendarEventTool(**kw), DeleteCalendarEventTool(**kw)


async def preview(tool: Any, **args: Any) -> str:
    result = await tool.preview(args)
    assert not result.is_error, result.content
    return result.content


async def refused(tool: Any, **args: Any) -> str:
    result = await tool.preview(args)
    assert result.is_error, result.content
    run = await tool.run(args)
    assert run.is_error and run.content == result.content
    return result.content


def test_both_tools_are_dangerous_and_confirmable() -> None:
    for tool in make(store()):
        assert tool.risk == Risk.DANGEROUS and callable(tool.preview)
        assert tool.parameters["required"] == ["id"]


# --- Delete --------------------------------------------------------------------------


async def test_delete_preview_names_the_real_event() -> None:
    s = store()
    _, delete = make(s)
    assert await preview(delete, id=event_id(DENTIST)) == (
        "Delete 'Dentist' on Tuesday 17 November 2026, 15:00 to 16:00, from the calendar Home?"
    )
    assert await preview(delete, id=event_id(CONFERENCE)) == (
        "Delete 'Conference' on Monday 23 November 2026 to Tuesday 24 November 2026, all day, "
        "from the calendar Home?"
    )
    assert s.removed == [] and s.updates == []  # a preview changes nothing


async def test_delete_recurring_says_and_does_one_occurrence() -> None:
    s = store()
    _, delete = make(s)
    assert await preview(delete, id=event_id(STANDUP)) == (
        "Delete 'Standup' on Friday 20 November 2026, 09:00 to 09:30, from the calendar Work, "
        "this occurrence only?"
    )
    result = await delete.run({"id": event_id(STANDUP)})
    assert result.content == (
        "Deleted from the calendar Work: Standup, Friday 20 November 2026, 09:00 to 09:30 "
        "(the user's local time). Only this occurrence was deleted; the rest of the repeating "
        "event remains."
    )
    assert s.removed == [STANDUP] and STANDUP_NEXT in s.stored


async def test_delete_run_refetches_and_reports() -> None:
    s = store()
    _, delete = make(s)
    result = await delete.run({"id": event_id(DENTIST)})
    assert result.content == (
        "Deleted from the calendar Home: Dentist, Tuesday 17 November 2026, 15:00 to 16:00 "
        "(the user's local time)."
    )
    assert DENTIST not in s.stored
    again = await delete.run({"id": event_id(DENTIST)})  # gone now: stale id
    assert again.is_error and "List the calendar again" in again.content


@pytest.mark.parametrize(
    ("ident", "expected"),
    [
        (None, "needs the event's 'id'"),
        ("dentist", "There is no event with id 'dentist'"),
        ("aaaaaa-20261117", "List the calendar again"),
        (event_id(HOLIDAY), "'Bank holiday' is in the calendar Holidays, which cannot be changed"),
        (event_id(INVITE), "'Board meeting' is an invitation from Alex Smith"),
    ],
)
async def test_both_tools_refuse_bad_targets(ident: str | None, expected: str) -> None:
    s = store()
    for tool in make(s):
        args = {"title": "New"} if tool.name == "edit_calendar_event" else {}
        if ident is not None:
            args["id"] = ident
        assert expected in await refused(tool, **args)
    assert s.removed == [] and s.updates == []


async def test_configured_calendars_limit_what_can_be_found() -> None:
    from lucy.core.config import CalendarSettings

    s = store()
    delete = DeleteCalendarEventTool(
        CalendarSettings(calendars=["Work"]), store=s, clock=lambda: NOW, local_zone=LONDON
    )
    assert "List the calendar again" in await refused(delete, id=event_id(DENTIST))


# --- Edit ----------------------------------------------------------------------------


async def test_move_keeps_the_length_and_states_old_and_new() -> None:
    edit, _ = make(store())
    assert await preview(edit, id=event_id(DENTIST), start="18 November 10:00") == (
        "Move 'Dentist' from Tuesday 17 November 2026, 15:00 to 16:00, "
        "to Wednesday 18 November 2026, 10:00 to 11:00?"
    )
    # A day alone keeps the time of day.
    assert await preview(edit, id=event_id(DENTIST), start="2026-11-19") == (
        "Move 'Dentist' from Tuesday 17 November 2026, 15:00 to 16:00, "
        "to Thursday 19 November 2026, 15:00 to 16:00?"
    )
    assert await preview(edit, id=event_id(DENTIST), duration_minutes=30) == (
        "Move 'Dentist' from Tuesday 17 November 2026, 15:00 to 16:00, "
        "to Tuesday 17 November 2026, 15:00 to 15:30?"
    )
    assert await preview(edit, id=event_id(DENTIST), end="17:15") == (
        "Move 'Dentist' from Tuesday 17 November 2026, 15:00 to 16:00, "
        "to Tuesday 17 November 2026, 15:00 to 17:15?"
    )


async def test_preview_never_uses_relative_words() -> None:
    edit, _ = make(store())
    question = await preview(edit, id=event_id(DENTIST), start="tomorrow 9:30")
    assert question == (
        "Move 'Dentist' from Tuesday 17 November 2026, 15:00 to 16:00, "
        "to Sunday 15 November 2026, 09:30 to 10:30?"
    )
    assert "tomorrow" not in question


async def test_rename_and_other_details_in_one_question() -> None:
    edit, _ = make(store())
    assert await preview(edit, id=event_id(DENTIST), title="Doctor", location="2 Low Road") == (
        "Rename 'Dentist' on Tuesday 17 November 2026, 15:00 to 16:00, to 'Doctor', "
        "and change the place from '1 High Street' to '2 Low Road'?"
    )
    assert await preview(edit, id=event_id(DENTIST), start="18 November 10:00", title="Doctor") == (
        "Move 'Dentist' from Tuesday 17 November 2026, 15:00 to 16:00, "
        "to Wednesday 18 November 2026, 10:00 to 11:00, and rename it to 'Doctor'?"
    )
    assert await preview(edit, id=event_id(DENTIST), location="", notes="Floss") == (
        "Change 'Dentist' on Tuesday 17 November 2026, 15:00 to 16:00: "
        "remove the place '1 High Street' and add the notes 'Floss'?"
    )


async def test_edit_recurring_says_and_changes_one_occurrence() -> None:
    s = store()
    edit, _ = make(s)
    args = {"id": event_id(STANDUP), "start": "2026-11-20T10:00"}
    assert await preview(edit, **args) == (
        "Move 'Standup' from Friday 20 November 2026, 09:00 to 09:30, "
        "to Friday 20 November 2026, 10:00 to 10:30, this occurrence only?"
    )
    result = await edit.run(args)
    assert "Only this occurrence changed" in result.content
    assert STANDUP_NEXT in s.stored  # the next occurrence is untouched
    [(target, _)] = s.updates
    assert target == STANDUP


async def test_edit_run_reports_changes_and_the_new_id() -> None:
    s = store()
    edit, _ = make(s)
    old = event_id(DENTIST)
    result = await edit.run({"id": old, "start": "18 November 10:00", "title": "Doctor"})
    [(target, changes)] = s.updates
    assert target == DENTIST
    assert changes == EventChanges(title="Doctor", start=at(18, 10), end=at(18, 11))
    moved = next(e for e in s.stored if e.title == "Doctor")
    new = event_id(moved)
    assert new != old and new.endswith("-20261118")
    assert result.content == (
        "Changed in the calendar Home: Doctor, Wednesday 18 November 2026, 10:00 to 11:00 "
        "(the user's local time). Changes: title was 'Dentist'; time was Tuesday 17 November "
        f"2026, 15:00 to 16:00. Its event id is now {new} (it was {old})."
    )
    # Same day: the id stays.
    again = await edit.run({"id": new, "notes": "Floss"})
    assert again.content.endswith(f"Changes: notes replaced. Event id: {new}.")


async def test_all_day_event_moves_by_days() -> None:
    edit, _ = make(store())
    assert await preview(edit, id=event_id(CONFERENCE), start="2026-11-30") == (
        "Move 'Conference' from Monday 23 November 2026 to Tuesday 24 November 2026, all day, "
        "to Monday 30 November 2026 to Tuesday 1 December 2026, all day?"
    )
    assert "Changing it to a timed event is not supported" in await refused(
        edit, id=event_id(CONFERENCE), start="2026-11-30T10:00"
    )


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({}, "Nothing would change in 'Dentist'"),
        ({"title": "Dentist", "location": "1 High Street"}, "Nothing would change"),
        ({"start": "2026-11-17T15:00"}, "Nothing would change"),
        ({"start": "next friday 10:00"}, "ambiguous"),
        ({"start": "11/12 10:00"}, "ambiguous"),
        ({"start": "friday afternoon"}, "not a start time"),
        ({"start": "2026-11-13T10:00"}, "is in the past"),
        ({"end": "14:00"}, "would not be after the start"),
        ({"end": "16:30", "duration_minutes": 30}, "not both"),
        ({"duration_minutes": 0}, "at least 1"),
        ({"duration_minutes": 60 * 24 * 15}, "more than 14 days"),
        ({"title": "x" * 201}, "too long"),
        ({"start": "gibberish"}, "Could not read"),
    ],
)
async def test_edit_refusals(args: dict[str, Any], expected: str) -> None:
    s = store()
    edit, _ = make(s)
    assert expected in await refused(edit, id=event_id(DENTIST), **args)
    assert s.updates == []


async def test_midnight_resolves_again_and_changes_the_question() -> None:
    """'tomorrow' asked at 23:59 and confirmed at 00:01 means another day: the preview
    differs, which is what stops the orchestrator."""
    times = [datetime(2026, 11, 14, 23, 59, 30, tzinfo=LONDON)]
    edit, _ = make(store(), clock=lambda: times[-1])
    args = {"id": event_id(DENTIST), "start": "tomorrow 10:00"}
    asked = await preview(edit, **args)
    assert "Sunday 15 November 2026, 10:00 to 11:00" in asked
    times.append(datetime(2026, 11, 15, 0, 0, 30, tzinfo=LONDON))
    again = await preview(edit, **args)
    assert "Monday 16 November 2026, 10:00 to 11:00" in again and again != asked


# --- Through the orchestrator --------------------------------------------------------


def _call(name: str, **args: Any) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=args)


async def _ask(h: Harness) -> None:
    await h.say("delete the dentist")
    await h.until_turns_end()


def _outcomes(h: Harness) -> list[str]:
    return [e.outcome for e in h.memory.episodes if isinstance(e, ConfirmationResolved)]


async def test_delete_end_to_end_yes_runs() -> None:
    s = store()
    _, delete = make(s)
    expected = await preview(delete, id=event_id(DENTIST))
    script = [calls(_call(delete.name, id=event_id(DENTIST))), text("Deleted.")]
    async with Harness(script, tools=[delete]) as h:
        await _ask(h)
        [request] = [e for e in h.memory.episodes if isinstance(e, ConfirmationRequested)]
        assert request.question == expected
        assert DENTIST in s.stored
        await h.say("Yes.")
        await h.until_turns_end()
    assert _outcomes(h) == ["confirmed"]
    assert DENTIST not in s.stored


async def test_delete_end_to_end_no_keeps_it() -> None:
    s = store()
    _, delete = make(s)
    script = [calls(_call(delete.name, id=event_id(DENTIST))), text("Kept.")]
    async with Harness(script, tools=[delete]) as h:
        await _ask(h)
        await h.say("no")
        await h.until_turns_end()
    assert _outcomes(h) == ["declined"] and DENTIST in s.stored


async def test_delete_end_to_end_event_changed_since_asking_runs_nothing() -> None:
    s = store()
    _, delete = make(s)
    script = [calls(_call(delete.name, id=event_id(DENTIST))), text("It changed.")]
    async with Harness(script, tools=[delete]) as h:
        await _ask(h)
        renamed = DENTIST.__class__(**{**DENTIST.__dict__, "title": "Dentist (moved room)"})
        s.stored[s.stored.index(DENTIST)] = renamed
        await h.say("yes")
        await h.until_turns_end()
    assert _outcomes(h) == ["changed"] and renamed in s.stored and s.removed == []


async def test_edit_end_to_end_across_midnight_runs_nothing() -> None:
    s = store()
    times = [datetime(2026, 11, 14, 23, 59, 30, tzinfo=LONDON)]
    edit, _ = make(s, clock=lambda: times[-1])
    script = [
        calls(_call(edit.name, id=event_id(DENTIST), start="tomorrow 10:00")),
        text("The day changed."),
    ]
    async with Harness(script, tools=[edit]) as h:
        await _ask(h)
        times.append(datetime(2026, 11, 15, 0, 0, 30, tzinfo=LONDON))
        await h.say("yes")
        await h.until_turns_end()
    assert _outcomes(h) == ["changed"] and s.updates == [] and DENTIST in s.stored
