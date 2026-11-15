"""The private-data guard (ADR 0022): while a private source's results are in what
the model sees, a call that could send data out of this machine needs the user's yes.

A tool declares, as optional attributes (no attribute means no):

- `private_source: str | None`: its results are private and come from the source of
  that name (a `private = true` MCP server sets the server's name);
- `sends_out: bool`: a call sends its arguments, or something chosen from them,
  outside this machine (`web_search`, `fetch_page`, `wikipedia`, `weather`, every
  MCP tool);
- `destination: str`: where to, in plain words, for the question ("Wikipedia").

The marking is not stored: it is read from the messages the model is given, so it
lasts exactly as long as a private result is in them, shortened or not, and ends
when the history window or the context budget drops it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from lucy.core.ports import Tool
from lucy.core.types import Message, Risk, Role

GUARD_VALUE_CHARS: Final = 60
"""Longest argument value read out in a guard question."""
GUARD_ARGS: Final = 3
"""Arguments named in a guard question; the rest are counted."""


def private_source(tool: object) -> str | None:
    """The private source `tool`'s results come from, or None."""
    source = getattr(tool, "private_source", None)
    return source if isinstance(source, str) and source else None


def sends_out(tool: object) -> bool:
    """Whether a call of `tool` may send data off this machine."""
    return getattr(tool, "sends_out", False) is True


def visible_private_sources(messages: Sequence[Message], tools: Mapping[str, Tool]) -> list[str]:
    """The private sources (sorted) with at least one result among `messages`."""
    names = {call.id: call.name for m in messages for call in m.tool_calls or ()}
    sources = {
        source
        for m in messages
        if m.role == Role.TOOL and m.tool_call_id in names
        if (source := private_source(tools.get(names[m.tool_call_id]))) is not None
    }
    return sorted(sources)


def needs_guard(tool: Tool, sources: Sequence[str]) -> bool:
    """Whether a call of `tool` must be confirmed while `sources` are visible: it sends
    data out, unless it is a READ tool of the only private source visible."""
    if not sources or not sends_out(tool):
        return False
    own = private_source(tool)
    return not (own is not None and set(sources) == {own} and tool.risk == Risk.READ)


def _listed(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def guard_notice(sources: Sequence[str]) -> str:
    """The plain-words opening of every guarded question."""
    return f"Private information from {_listed(list(sources))} is in our conversation."


def _spoken(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    text = re.sub(r"\s+", " ", text).strip().replace('"', "'").replace("?", "")
    if len(text) > GUARD_VALUE_CHARS:
        text = text[: GUARD_VALUE_CHARS - 1].rstrip() + "…"
    return f'"{text}"'


def guard_question(tool: Tool, arguments: Mapping[str, Any], sources: Sequence[str]) -> str:
    """The question for a guarded call of a tool that has no confirmation of its own:
    deterministic from the arguments, naming what would be sent and where."""
    keys = sorted(str(k) for k, v in arguments.items() if v is not None and v != "")
    named = [f"{key.replace('_', ' ')} {_spoken(arguments[key])}" for key in keys[:GUARD_ARGS]]
    if len(keys) > GUARD_ARGS:
        rest = len(keys) - GUARD_ARGS
        named.append(f"{rest} more detail{'s' if rest > 1 else ''}")
    what = _listed(named) if named else "a request"
    destination = getattr(tool, "destination", None)
    where = destination if isinstance(destination, str) and destination else "outside this Mac"
    return f"{guard_notice(sources)} {tool.name} would send {what} to {where}. Shall I go ahead?"


def guarded(question: str, sources: Iterable[str]) -> str:
    """A dangerous tool's own question with the guard notice in front."""
    sources = list(sources)
    return f"{guard_notice(sources)} {question}" if sources else question
