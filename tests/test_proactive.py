"""Reminders and proactive turns (ADR 0020): `ReminderService` claims due reminders,
the orchestrator says them in a turn with no model call, marks them delivered after
speaking, and never lets a reminder become the answer to a confirmation question."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from lucy.core.confirm import open_request
from lucy.core.events import (
    AssistantMessage,
    ConfirmationRequested,
    ConfirmationResolved,
    TextDelta,
    TimerFired,
    TurnEnded,
    UserMessage,
)
from lucy.core.history import events_to_messages
from lucy.core.orchestrator import NOT_MADE
from lucy.core.reminders import (
    REMINDER_TIMER,
    Reminder,
    ReminderKind,
    ReminderService,
    ReminderStatus,
    TimerReply,
)
from lucy.core.types import Role, ToolCall
from lucy.memory.sqlite_reminders import SQLiteReminderStore
from lucy.tools.reminders import announcement
from tests.fakes import InMemoryMemoryStore, Pause
from tests.test_confirmation import ARGS, ask, delete, outcomes, tool
from tests.test_orchestrator import Harness, calls, text

NOW = datetime(2026, 11, 14, 16, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


def utc_announcement(reminders: Sequence[Reminder], now: datetime) -> str:
    return announcement(reminders, now.astimezone(UTC))


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
async def store(tmp_path: Path, clock: Clock) -> AsyncIterator[SQLiteReminderStore]:
    async with SQLiteReminderStore(tmp_path / "lucy.db", clock=clock) as opened:
        yield opened


def service_for(
    store: SQLiteReminderStore, memory: InMemoryMemoryStore, clock: Clock, **options: Any
) -> ReminderService:
    options.setdefault("poll", 0.01)
    return ReminderService(store, memory, utc_announcement, clock=clock, **options)


def rig(
    store: SQLiteReminderStore, clock: Clock, scripts: Sequence[Any] = (), **options: Any
) -> tuple[Harness, ReminderService]:
    memory = InMemoryMemoryStore()
    service = service_for(store, memory, clock)
    handlers = options.pop("timer_handlers", {REMINDER_TIMER: service.handle})
    harness = Harness(scripts, memory=memory, timer_handlers=handlers, **options)
    return harness, service


async def fire(h: Harness, service: ReminderService, session: str = "s") -> TimerFired:
    event = await service.fire(session)
    assert event is not None
    await h.bus.publish(event)
    return event


def types(h: Harness) -> list[str]:
    return [e.type for e in h.memory.episodes]


def said(events: Sequence[Any]) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


# --- The proactive turn ------------------------------------------------------------


async def test_a_reminder_turn_makes_no_model_call(store, clock) -> None:
    saved = await store.add(ReminderKind.REMINDER, "call the bank", NOW)
    h, service = rig(store, clock)
    async with h:
        event = await fire(h, service)
        events = await h.until_turns_end()
    assert h.llm.calls == []
    assert types(h) == ["timer_fired", "assistant_message", "turn_ended"]
    assert said(events) == "Reminder: call the bank."
    logged = h.memory.episodes[0]
    assert isinstance(logged, TimerFired) and logged.id == event.id
    assert logged.payload == {"ids": [saved.id], "owner": service.owner}
    done = await store.get(saved.id)
    assert done is not None and done.status is ReminderStatus.DELIVERED
    assert done.delivered_turn == logged.turn_id and done.delivered_session == "s"


async def test_a_reminder_queues_behind_a_running_turn(store, clock) -> None:
    await store.add(ReminderKind.REMINDER, "call the bank", NOW)
    pause = Pause()
    h, service = rig(store, clock, [[pause, *text("Hello.")]])
    async with h:
        await h.say("hi")
        await asyncio.wait_for(pause.reached.wait(), 1)
        await fire(h, service)
        await asyncio.sleep(0.05)
        assert "timer_fired" not in types(h)
        pause.release.set()
        await h.until_turns_end(2)
    assert types(h) == [
        "user_message",
        "assistant_message",
        "turn_ended",
        "timer_fired",
        "assistant_message",
        "turn_ended",
    ]


async def test_a_reminder_cancelled_before_its_turn_says_nothing(store, clock) -> None:
    saved = await store.add(ReminderKind.REMINDER, "call the bank", NOW)
    h, service = rig(store, clock)
    async with h:
        event = await service.fire("s")
        assert event is not None
        await store.cancel(saved.id)
        await h.bus.publish(event)
        events = await h.until_turns_end()
    assert said(events) == ""
    assert types(h) == ["timer_fired", "turn_ended"]


async def test_a_failure_before_the_mark_redelivers_late_after_the_lease(store, clock) -> None:
    saved = await store.add(ReminderKind.REMINDER, "call the bank", NOW)
    memory = InMemoryMemoryStore()
    service = service_for(store, memory, clock)

    async def crashing(event: TimerFired, turn_id: str) -> TimerReply | None:
        reply = await service.handle(event, turn_id)
        assert reply is not None

        async def fail() -> None:
            raise OSError("disk gone")

        return TimerReply(text=reply.text, delivered=fail)

    h = Harness([], memory=memory, timer_handlers={REMINDER_TIMER: crashing})
    async with h:
        await fire(h, service)
        first = await h.until_turns_end()
    assert said(first) == "Reminder: call the bank."
    still = await store.get(saved.id)
    assert still is not None and still.status is ReminderStatus.FIRING

    # A fresh process (another owner) takes it over once the lease has lapsed.
    clock.now = NOW + timedelta(minutes=3)
    later = service_for(store, memory, clock)
    h2 = Harness([], memory=memory, timer_handlers={REMINDER_TIMER: later.handle})
    async with h2:
        async with asyncio.timeout(1):
            await later.wait_due()
        await fire(h2, later)
        again = await h2.until_turns_end()
    assert said(again) == "A reminder was due at 16:00 today: call the bank."
    done = await store.get(saved.id)
    assert done is not None and done.status is ReminderStatus.DELIVERED


async def test_several_due_reminders_are_one_message(store, clock) -> None:
    await store.add(ReminderKind.REMINDER, "call the bank", NOW)
    await store.add(ReminderKind.REMINDER, "water the plants", NOW)
    h, service = rig(store, clock)
    async with h:
        await fire(h, service)
        events = await h.until_turns_end()
    assert said(events) == ("You have 2 reminders. First: call the bank. Second: water the plants.")
    assert types(h).count("assistant_message") == 1
    assert await store.upcoming() == []


async def test_an_unknown_timer_name_ends_the_turn_silently(store, clock) -> None:
    h, _ = rig(store, clock)
    async with h:
        await h.bus.publish(TimerFired(session_id="s", name="brief"))
        events = await h.until_turns_end()
    assert said(events) == "" and types(h) == ["timer_fired", "turn_ended"]
    assert h.llm.calls == []


async def test_fire_with_nothing_due_returns_none(store, clock) -> None:
    await store.add(ReminderKind.REMINDER, "later", NOW + timedelta(hours=1))
    _, service = rig(store, clock)
    assert await service.fire("s") is None


# --- Waiting -------------------------------------------------------------------------


async def test_wait_due_wakes_for_a_reminder_added_on_another_connection(
    tmp_path: Path, store, clock
) -> None:
    service = service_for(store, InMemoryMemoryStore(), clock)
    waiter = asyncio.create_task(service.wait_due())
    await asyncio.sleep(0.05)
    assert not waiter.done()
    async with SQLiteReminderStore(tmp_path / "lucy.db", clock=clock) as other:
        await other.add(ReminderKind.TIMER, "", NOW)
        async with asyncio.timeout(1):
            await waiter


# --- Confirmation ------------------------------------------------------------------


async def test_fire_holds_while_a_confirmation_is_open_until_it_resolves(store, clock) -> None:
    await store.add(ReminderKind.REMINDER, "call the bank", NOW)
    danger = tool()
    h, service = rig(store, clock, [calls(delete()), text("Fine, kept.")], tools=[danger])
    async with h:
        await ask(h)
        [request] = [e for e in h.memory.episodes if isinstance(e, ConfirmationRequested)]
        clock.now = request.ts
        assert open_request(h.memory.episodes, clock.now) == request
        assert await service.fire("s") is None
        await h.say("no")
        await h.until_turns_end()
        assert open_request(h.memory.episodes, clock.now) is None
        await fire(h, service)
        events = await h.until_turns_end()
    assert outcomes(h) == ["declined"] and danger.runs == []
    assert "call the bank" in said(events)


async def test_fire_holds_until_a_confirmation_expires(store, clock) -> None:
    await store.add(ReminderKind.REMINDER, "call the bank", NOW)
    h, service = rig(store, clock, [calls(delete())], tools=[tool()])
    async with h:
        await ask(h)
        [request] = [e for e in h.memory.episodes if isinstance(e, ConfirmationRequested)]
        clock.now = request.expires_at - timedelta(seconds=1)
        assert await service.fire("s") is None
        clock.now = request.expires_at + timedelta(seconds=1)
        await fire(h, service)
        events = await h.until_turns_end()
    # The unanswered request is ended by the reminder turn, and the user told so.
    assert outcomes(h) == ["superseded"]
    assert said(events).endswith(NOT_MADE) and "call the bank" in said(events)


async def test_a_yes_after_a_reminder_runs_nothing(store, clock) -> None:
    await store.add(ReminderKind.REMINDER, "call the bank", NOW)
    danger = tool()
    h, service = rig(store, clock, [calls(delete()), text("Yes to what?")], tools=[danger])
    async with h:
        await ask(h)
        # The race: the reminder was claimed just before the question was asked.
        claimed = await service.store.claim_due(NOW, service.owner)
        ids = [r.id for r in claimed]
        payload = {"ids": ids, "owner": service.owner}
        await h.bus.publish(TimerFired(session_id="s", name=REMINDER_TIMER, payload=payload))
        events = await h.until_turns_end()
        await h.say("yes")
        await h.until_turns_end()
    assert danger.runs == []
    assert outcomes(h) == ["superseded"]
    assert said(events) == f"Reminder: call the bank. {NOT_MADE}"
    resolved = next(e for e in h.memory.episodes if isinstance(e, ConfirmationResolved))
    timer = next(e for e in h.memory.episodes if isinstance(e, TimerFired))
    assert resolved.turn_id == timer.turn_id
    assert len(h.llm.calls) == 2  # the "yes" went to the model as an ordinary message


async def test_a_confirmation_asked_after_a_reminder_confirms_normally(store, clock) -> None:
    await store.add(ReminderKind.REMINDER, "call the bank", NOW)
    danger = tool()
    h, service = rig(store, clock, [calls(delete()), text("Deleted.")], tools=[danger])
    async with h:
        await fire(h, service)
        await h.until_turns_end()
        await ask(h)
        await h.say("yes")
        await h.until_turns_end()
    assert danger.runs == [ARGS] and outcomes(h) == ["confirmed"]


# --- History -------------------------------------------------------------------------


def test_a_proactive_reply_merges_into_the_previous_reply() -> None:
    user = UserMessage(session_id="s", text="hello", turn_id="t1")
    hi = AssistantMessage(session_id="s", turn_id="t1", text="Hi.")
    timer = TimerFired(session_id="s", turn_id="t2", name=REMINDER_TIMER)
    reminder = AssistantMessage(session_id="s", turn_id="t2", text="Reminder: call the bank.")
    ended = TurnEnded(session_id="s", turn_id="t1", reason="completed")
    messages = events_to_messages([user, hi, ended, timer, reminder])
    assert [(m.role, m.content) for m in messages] == [
        (Role.USER, "hello"),
        (Role.ASSISTANT, "Hi.\n\nReminder: call the bank."),
    ]


def test_a_reply_after_tool_calls_is_not_merged() -> None:
    call = ToolCall(id="c1", name="echo", arguments={})
    from lucy.core.events import ToolFinished
    from lucy.core.types import ToolResult

    events = [
        UserMessage(session_id="s", text="hello"),
        AssistantMessage(session_id="s", tool_calls=[call]),
        ToolFinished(session_id="s", result=ToolResult(call_id="c1", content="ok")),
        AssistantMessage(session_id="s", text="Done."),
    ]
    roles = [m.role for m in events_to_messages(events)]
    assert roles == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]


async def test_a_user_turn_after_a_reminder_sees_alternating_roles(store, clock) -> None:
    await store.add(ReminderKind.REMINDER, "call the bank", NOW)
    h, service = rig(store, clock, [text("Hi."), text("You're welcome.")])
    async with h:
        await h.say("hello")
        await h.until_turns_end()
        await fire(h, service)
        await h.until_turns_end()
        await h.say("thanks")
        await h.until_turns_end()
    conversation = [(m.role, m.content) for m in h.llm.calls[1] if m.role != Role.SYSTEM]
    assert conversation[-3:] == [
        (Role.USER, "hello"),
        (Role.ASSISTANT, "Hi.\n\nReminder: call the bank."),
        (Role.USER, "thanks"),
    ]
