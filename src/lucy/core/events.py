"""Typed, immutable events: every input to and output from the core (ADR 0003).

The persisted event stream is the episode log, so these classes are a long-lived
schema. Each concrete event has a unique `type` literal used as the JSON
discriminator.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from lucy.core.types import ToolCall, ToolResult


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class Event(BaseModel):
    """Base for all events. Not used directly; subclasses fix `type`.

    `schema_version` is the version of this event type's schema. It is bumped
    when the type's fields change incompatibly, so persisted events can be
    migrated. JSON without it parses as version 1.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str
    schema_version: int = 1
    id: str = Field(default_factory=_new_id)
    ts: AwareDatetime = Field(default_factory=_now)
    session_id: str
    turn_id: str | None = None

    @field_validator("ts")
    @classmethod
    def _to_utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)


# --- Inbound ------------------------------------------------------------------


class UserMessage(Event):
    """The user said or typed something.

    `source` is how it reached Lucy: typed, spoken, or a correction the user made
    with a memory tool (`lucy memory correct`), which is logged as a message so
    that the episode log stays the source of truth for what the user said.
    """

    type: Literal["user_message"] = "user_message"
    text: str
    source: Literal["text", "voice", "correction"] = "text"


class Interrupt(Event):
    """Cancel the running turn (barge-in)."""

    type: Literal["interrupt"] = "interrupt"


class PlaybackInterrupted(Event):
    """Spoken playback of a turn's reply was cut short (ADR 0016).

    Published by a voice client when the user stops Lucy while she is speaking
    (or thinking). `spoken_text` is the reply text actually heard: whole
    sentences played plus an approximate part of the sentence in progress, for
    the turn's final LLM step. The orchestrator only persists it; rebuilding the
    conversation replaces that turn's reply with `spoken_text`
    (`lucy.core.history.events_to_messages`).
    """

    type: Literal["playback_interrupted"] = "playback_interrupted"
    spoken_text: str


class TimerFired(Event):
    """A scheduled timer elapsed; lets Lucy start a turn on her own."""

    type: Literal["timer_fired"] = "timer_fired"
    name: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ClientConnected(Event):
    """A client attached to the session."""

    type: Literal["client_connected"] = "client_connected"
    client_id: str


# --- Outbound -----------------------------------------------------------------


class TextDelta(Event):
    """A piece of assistant text as it streams from the LLM."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class SentenceReady(Event):
    """A complete sentence of assistant text, ready for TTS."""

    type: Literal["sentence_ready"] = "sentence_ready"
    text: str


class ToolStarted(Event):
    """A tool call is about to run."""

    type: Literal["tool_started"] = "tool_started"
    call: ToolCall


class ToolFinished(Event):
    """A tool call finished, successfully or not."""

    type: Literal["tool_finished"] = "tool_finished"
    result: ToolResult


class AssistantMessage(Event):
    """The assistant's complete output for one LLM step.

    Emitted once per LLM step, after its streaming finishes, so the conversation
    can be rebuilt from the log as UserMessage / AssistantMessage / ToolFinished
    without stitching `TextDelta` fragments. `text` is the text actually
    delivered; on interruption, only the part that was delivered.
    """

    type: Literal["assistant_message"] = "assistant_message"
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    interrupted: bool = False


class TurnEnded(Event):
    """A turn is over. `error` is set when `reason` is "error"."""

    type: Literal["turn_ended"] = "turn_ended"
    reason: Literal["completed", "interrupted", "error"]
    error: str | None = None


INBOUND_EVENTS: tuple[type[Event], ...] = (
    UserMessage,
    Interrupt,
    PlaybackInterrupted,
    TimerFired,
    ClientConnected,
)
OUTBOUND_EVENTS: tuple[type[Event], ...] = (
    TextDelta,
    SentenceReady,
    ToolStarted,
    ToolFinished,
    AssistantMessage,
    TurnEnded,
)

AnyEvent = Annotated[
    UserMessage
    | Interrupt
    | PlaybackInterrupted
    | TimerFired
    | ClientConnected
    | TextDelta
    | SentenceReady
    | ToolStarted
    | ToolFinished
    | AssistantMessage
    | TurnEnded,
    Field(discriminator="type"),
]
"""Any concrete event, discriminated by its `type` field."""

_ANY_EVENT: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)


def event_to_json(event: Event) -> str:
    """Serialise an event to a JSON string."""
    return event.model_dump_json()


def event_from_json(data: str | bytes) -> AnyEvent:
    """Parse a JSON string back into the concrete event class named by its `type`.

    Raises `pydantic.ValidationError` for unknown types or invalid fields.
    """
    return _ANY_EVENT.validate_json(data)
