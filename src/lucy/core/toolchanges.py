"""Leaving out exchanges that a newly configured tool made stale (ADR 0025,
amendment 2026-11-17).

The chat model repeats what it said earlier in the conversation. Asked "remember my
face" while no face tool existed, Lucy rightly said she could not; once `enroll_face`
was enabled, gemma4:12b-mlx kept refusing with those exchanges in the history (9 of
9) and called the tool with them left out (7 of 7). No tool wording fixed it.

So the orchestrator records which tools it is configured with (`ToolsConfigured`,
written at the start of a turn when the latest record in the window differs or there
is none), and the prompt history leaves out an earlier exchange when all hold:

(a) its user message switches on a tool group by that group's keywords (the
    selector's matching, `keyword_pattern`), whether or not `[tools] dynamic` is on;
(b) that group has a configured tool now that was not configured when the exchange
    happened, per the latest record before it;
(c) the turn that answered it made no tool call (nor ran or asked to confirm one).

The user message and everything up to the next user message or `TimerFired` go
together, so roles still alternate. The latest user message is never left out. The
event log is untouched: this only shapes the prompt, deterministically and without a
model call, so a turn and the pre-fill before it (`Orchestrator.next_turn_prefix`)
leave out the same exchanges.

Limit: the tool set before the first record in the window is unknown and taken to be
the current one, so nothing is left out of history written before records existed
(or older than the window's first record).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from lucy.core.events import (
    AnyEvent,
    AssistantMessage,
    ConfirmationRequested,
    ConfirmationResolved,
    TimerFired,
    ToolFinished,
    ToolsConfigured,
    ToolStarted,
    UserMessage,
)
from lucy.core.toolsets import ToolGroup, keyword_pattern

_TOOL_ACTIVITY = (ToolStarted, ToolFinished, ConfirmationRequested, ConfirmationResolved)


class ToolChanges:
    """The configured tool names and the groups whose keywords judge staleness."""

    def __init__(self, configured: Iterable[str], groups: Sequence[ToolGroup] = ()) -> None:
        self.names: tuple[str, ...] = tuple(dict.fromkeys(configured))
        self._current = frozenset(self.names)
        self._groups: list[tuple[re.Pattern[str], frozenset[str]]] = []
        for group in groups:
            pattern = keyword_pattern(group.keywords)
            tools = frozenset(group.tools) & self._current
            if pattern is not None and tools:
                self._groups.append((pattern, tools))

    def record(
        self, episodes: Sequence[AnyEvent], session_id: str, turn_id: str | None
    ) -> ToolsConfigured | None:
        """The record to write before a turn's user message, or None when the latest
        record in `episodes` already names the configured tools. With no tools
        configured and no record, nothing is recorded (no exchange can be stale then,
        and logs of tool-less setups stay as they were)."""
        latest = next((e for e in reversed(episodes) if isinstance(e, ToolsConfigured)), None)
        if latest is None and not self.names:
            return None
        if latest is not None and frozenset(latest.names) == self._current:
            return None
        return ToolsConfigured(session_id=session_id, turn_id=turn_id, names=self.names)

    def leave_out_stale(self, episodes: Sequence[AnyEvent]) -> tuple[list[AnyEvent], int]:
        """`episodes` without stale exchanges (module docstring), and how many were
        left out. The last `UserMessage` and its events are always kept."""
        starts = [i for i, e in enumerate(episodes) if isinstance(e, UserMessage)]
        if len(starts) < 2 or not self._groups:
            return list(episodes), 0
        known: list[frozenset[str] | None] = []
        latest: frozenset[str] | None = None
        for event in episodes:
            if isinstance(event, ToolsConfigured):
                latest = frozenset(event.names)
            known.append(latest)
        dropped: set[int] = set()
        count = 0
        for start in starts[:-1]:
            before = known[start]
            if before is None:
                continue  # before the first record: taken to be the current set
            new = self._current - before
            if not new:
                continue
            message = episodes[start]
            assert isinstance(message, UserMessage)
            text = message.text.lower()
            if not any(tools & new and p.search(text) for p, tools in self._groups):
                continue
            end = next(
                (
                    i
                    for i in range(start + 1, len(episodes))
                    if isinstance(episodes[i], UserMessage | TimerFired)
                ),
                len(episodes),
            )
            span = range(start, end)
            if any(_acted(episodes[i]) for i in span):
                continue
            dropped.update(i for i in span if not isinstance(episodes[i], ToolsConfigured))
            count += 1
        if not count:
            return list(episodes), 0
        return [e for i, e in enumerate(episodes) if i not in dropped], count


def _acted(event: AnyEvent) -> bool:
    if isinstance(event, AssistantMessage):
        return bool(event.tool_calls)
    return isinstance(event, _TOOL_ACTIVITY)
