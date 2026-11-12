import json
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from lucy.core.events import (
    INBOUND_EVENTS,
    OUTBOUND_EVENTS,
    AssistantMessage,
    ClientConnected,
    Event,
    Interrupt,
    SentenceReady,
    TextDelta,
    TimerFired,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    UserMessage,
    event_from_json,
    event_to_json,
)
from lucy.core.types import ToolCall, ToolResult

SAMPLES: list[Event] = [
    UserMessage(session_id="s", text="hello", source="voice"),
    Interrupt(session_id="s", turn_id="t1"),
    TimerFired(session_id="s", name="reminder", payload={"what": "tea", "n": [1, 2]}),
    ClientConnected(session_id="s", client_id="cli"),
    TextDelta(session_id="s", turn_id="t1", text="Hel"),
    SentenceReady(session_id="s", turn_id="t1", text="Hello there."),
    ToolStarted(
        session_id="s", turn_id="t1", call=ToolCall(id="c1", name="echo", arguments={"text": "x"})
    ),
    ToolFinished(
        session_id="s", turn_id="t1", result=ToolResult(call_id="c1", content="x", is_error=False)
    ),
    AssistantMessage(
        session_id="s",
        turn_id="t1",
        text="Let me ch",
        tool_calls=[ToolCall(id="c2", name="echo", arguments={"text": "y"})],
        interrupted=True,
    ),
    TurnEnded(session_id="s", turn_id="t1", reason="error", error="boom"),
]


def test_samples_cover_every_event_type() -> None:
    assert {type(e) for e in SAMPLES} == set(INBOUND_EVENTS + OUTBOUND_EVENTS)


@pytest.mark.parametrize("event", SAMPLES, ids=lambda e: type(e).__name__)
def test_round_trip_through_json(event: Event) -> None:
    parsed = event_from_json(event_to_json(event))
    assert type(parsed) is type(event)
    assert parsed == event


def test_type_discriminators_are_unique() -> None:
    types = [cls.model_fields["type"].default for cls in INBOUND_EVENTS + OUTBOUND_EVENTS]
    assert len(set(types)) == len(types)


def test_defaults_id_and_utc_timestamp() -> None:
    a = Interrupt(session_id="s")
    b = Interrupt(session_id="s")
    assert a.id != b.id
    assert a.ts.tzinfo is not None and a.ts.utcoffset() == timedelta(0)
    assert a.turn_id is None


def test_timestamp_normalised_to_utc() -> None:
    local = datetime(2026, 1, 1, 12, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    event = Interrupt(session_id="s", ts=local)
    assert event.ts == local
    assert event.ts.tzinfo == UTC


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ValidationError):
        Interrupt(session_id="s", ts=datetime(2026, 1, 1))


def test_events_are_frozen() -> None:
    event = TextDelta(session_id="s", text="a")
    with pytest.raises(ValidationError):
        event.text = "b"  # type: ignore[misc]


def test_unknown_type_rejected() -> None:
    with pytest.raises(ValidationError):
        event_from_json('{"type": "nope", "session_id": "s"}')


def test_schema_version_serialised() -> None:
    data = json.loads(event_to_json(Interrupt(session_id="s")))
    assert data["schema_version"] == 1


def test_schema_version_defaults_when_absent() -> None:
    event = event_from_json('{"type": "interrupt", "session_id": "s"}')
    assert event.schema_version == 1


def test_assistant_message_defaults() -> None:
    event = AssistantMessage(session_id="s")
    assert event.text == "" and event.tool_calls == [] and not event.interrupted


def test_user_message_sources() -> None:
    for source in ("text", "voice", "correction"):
        message = UserMessage(session_id="corrections", text="x", source=source)
        assert event_from_json(event_to_json(message)) == message
    with pytest.raises(ValidationError):
        UserMessage(session_id="s", text="x", source="telepathy")  # type: ignore[arg-type]
    # Old JSON without a source still parses as typed text.
    old = '{"type": "user_message", "session_id": "s", "text": "hi", "ts": "2026-01-01T00:00:00Z"}'
    parsed = event_from_json(old)
    assert isinstance(parsed, UserMessage) and parsed.source == "text"
