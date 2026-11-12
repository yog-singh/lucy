"""Rebuilding the conversation from the episode log (ADR 0003, 0004)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from lucy.core.events import (
    AnyEvent,
    AssistantMessage,
    PlaybackInterrupted,
    ToolFinished,
    UserMessage,
)
from lucy.core.types import Message, Role, ToolResult

SHORTENED_MARKER: Final = "[earlier lookup shortened]"
"""Ends a tool result from an earlier turn that `events_to_messages` shortened."""

EARLIER_RESULT_CHARS: Final = 300
"""Characters kept after the first line of an earlier turn's tool result."""


def tool_message(result: ToolResult) -> Message:
    """The tool message that feeds `result` back to the LLM."""
    return Message(role=Role.TOOL, content=result.content, tool_call_id=result.call_id)


def _cut(text: str, limit: int) -> str:
    """At most `limit` characters of `text`, not ending in the middle of a word."""
    kept = text[:limit]
    if text[limit : limit + 1].strip():
        kept = kept.rsplit(None, 1)[0] if len(kept.split(None, 1)) > 1 else kept
    return kept.rstrip()


def shorten_tool_result(content: str, limit: int = EARLIER_RESULT_CHARS) -> str:
    """`content` cut to its first line plus about `limit` characters, then `SHORTENED_MARKER`.

    The first line is itself capped at `limit` characters. Content that is
    already that short is returned unchanged.
    """
    first, _, rest = content.partition("\n")
    if len(first) > limit:
        head = _cut(first, limit)
    elif len(rest) <= limit:
        return content
    else:
        head = f"{first}\n{_cut(rest, limit)}"
    return f"{head} {SHORTENED_MARKER}"


def events_to_messages(
    events: Sequence[AnyEvent], *, earlier_result_chars: int | None = EARLIER_RESULT_CHARS
) -> list[Message]:
    """Rebuild a valid chat transcript from persisted events, oldest first.

    `UserMessage`, `AssistantMessage` and `ToolFinished` become user, assistant
    and tool messages; every other event is ignored. Interrupted turns can leave
    loose ends, which are removed so the result is always valid:

    - an assistant tool call with no `ToolFinished` later in the list is dropped,
      and an assistant message left with no text and no tool calls is dropped;
    - a `ToolFinished` that answers no earlier kept tool call is dropped (this
      happens when the window of events starts between a call and its result).

    An interrupted `AssistantMessage` keeps the text that was delivered.

    Spoken replies (ADR 0016): when a turn has a `PlaybackInterrupted` event, the
    text of that turn's final `AssistantMessage` is replaced by the event's
    `spoken_text`, since that is what the user heard; with no spoken text the
    message keeps only its (answered) tool calls, and is dropped if it has none.
    Earlier assistant messages of the turn (steps that ended in tool calls) and
    the tool results are unaffected. If a turn has several such events, the last
    one in the list wins. The event may come before or after the turn's final
    `AssistantMessage` in the log.

    Tool results of earlier turns (before the last `UserMessage`) are shortened
    with `shorten_tool_result` to `earlier_result_chars` after their first line,
    since they are context by now and a whole page in every later prompt is
    expensive; results of the latest turn and error results stay whole. None
    keeps every result whole.
    """
    finished_at: dict[str, int] = {}
    spoken: dict[str, str] = {}
    final_reply: dict[str, int] = {}
    for index, event in enumerate(events):
        if isinstance(event, ToolFinished):
            finished_at.setdefault(event.result.call_id, index)
        elif isinstance(event, PlaybackInterrupted) and event.turn_id is not None:
            spoken[event.turn_id] = event.spoken_text
        elif isinstance(event, AssistantMessage) and event.turn_id is not None:
            final_reply[event.turn_id] = index
    replaced = {final_reply[turn]: text for turn, text in spoken.items() if turn in final_reply}
    latest_turn_from = max(
        (i for i, event in enumerate(events) if isinstance(event, UserMessage)), default=0
    )

    messages: list[Message] = []
    open_calls: set[str] = set()
    for index, event in enumerate(events):
        if isinstance(event, UserMessage):
            messages.append(Message(role=Role.USER, content=event.text))
        elif isinstance(event, AssistantMessage):
            calls = [c for c in event.tool_calls if finished_at.get(c.id, -1) > index]
            text = replaced.get(index, event.text)
            if not text and not calls:
                continue
            open_calls.update(c.id for c in calls)
            messages.append(Message(role=Role.ASSISTANT, content=text, tool_calls=calls or None))
        elif isinstance(event, ToolFinished):
            if event.result.call_id in open_calls:
                open_calls.discard(event.result.call_id)
                result = event.result
                if (
                    earlier_result_chars is not None
                    and index < latest_turn_from
                    and not result.is_error
                ):
                    content = shorten_tool_result(result.content, earlier_result_chars)
                    result = result.model_copy(update={"content": content})
                messages.append(tool_message(result))
    return messages
