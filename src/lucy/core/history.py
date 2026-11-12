"""Rebuilding the conversation from the episode log (ADR 0003, 0004)."""

from __future__ import annotations

from collections.abc import Sequence

from lucy.core.events import AnyEvent, AssistantMessage, ToolFinished, UserMessage
from lucy.core.types import Message, Role, ToolResult


def tool_message(result: ToolResult) -> Message:
    """The tool message that feeds `result` back to the LLM."""
    return Message(role=Role.TOOL, content=result.content, tool_call_id=result.call_id)


def events_to_messages(events: Sequence[AnyEvent]) -> list[Message]:
    """Rebuild a valid chat transcript from persisted events, oldest first.

    `UserMessage`, `AssistantMessage` and `ToolFinished` become user, assistant
    and tool messages; every other event is ignored. Interrupted turns can leave
    loose ends, which are removed so the result is always valid:

    - an assistant tool call with no `ToolFinished` later in the list is dropped,
      and an assistant message left with no text and no tool calls is dropped;
    - a `ToolFinished` that answers no earlier kept tool call is dropped (this
      happens when the window of events starts between a call and its result).

    An interrupted `AssistantMessage` keeps the text that was delivered.
    """
    finished_at: dict[str, int] = {}
    for index, event in enumerate(events):
        if isinstance(event, ToolFinished):
            finished_at.setdefault(event.result.call_id, index)

    messages: list[Message] = []
    open_calls: set[str] = set()
    for index, event in enumerate(events):
        if isinstance(event, UserMessage):
            messages.append(Message(role=Role.USER, content=event.text))
        elif isinstance(event, AssistantMessage):
            calls = [c for c in event.tool_calls if finished_at.get(c.id, -1) > index]
            if not event.text and not calls:
                continue
            open_calls.update(c.id for c in calls)
            messages.append(
                Message(role=Role.ASSISTANT, content=event.text, tool_calls=calls or None)
            )
        elif isinstance(event, ToolFinished):
            if event.result.call_id in open_calls:
                open_calls.discard(event.result.call_id)
                messages.append(tool_message(event.result))
    return messages
