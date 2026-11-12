"""Assembling the prompt for one turn from named, budgeted sections (ADR 0004).

The usable window is the model's context window minus the tokens reserved for
the reply and minus the persona, which is always included in full (with the
current date and time when the builder has a clock). Each other section gets a
fixed fraction of the usable window (`ContextBudgets`). The default fractions
leave some slack for tool schemas and per-message overhead, which are not
counted.

Two layouts (`ContextLayout`) arrange the same sections:

- `"system"`: one system message holding the persona, the date line, the
  profile, the standing instructions and the retrieved memories, then the
  history. The date (to the minute) and the memories change every turn, so
  each turn's prompt differs from the previous one right after the persona and
  the model server reprocesses the whole history.
- `"stable_prefix"` (the default): the system message holds only what rarely
  changes (the persona, then the standing instructions, then the profile),
  followed by the history; what changes every turn (the date line and the
  memories) comes last, as a system message right after the latest user
  message (before that turn's own tool steps). Each turn's prompt then extends
  the previous one up to the new user message, so the model server can reuse
  its cached prefix, and the prefix can be prepared before the user speaks
  (see `prefix_messages`). The block exists only in the prompt: the stored
  conversation is unchanged, and later turns do not replay it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lucy.core.types import MemoryContext, Message, Role

SectionName = Literal["persona", "profile", "instructions", "memories", "history"]

ContextLayout = Literal["system", "stable_prefix"]
"""How the sections are arranged; see the module docstring."""

PROFILE_LABEL = "## About the user"
INSTRUCTIONS_LABEL = "## Standing instructions from the user"
MEMORIES_LABEL = "## Things you remember that may be relevant"


def date_line(now: datetime) -> str:
    """The line telling the model the current date and time, e.g.
    "Today is Thursday, 12 November 2026, 11:42 IST (UTC+05:30).".

    A timezone-aware `now` names its zone (when it has an alphabetic abbreviation)
    and always its UTC offset, so the model can convert to other zones; a naive one
    is labelled "(local time)"."""
    return f"Today is {now:%A}, {now.day} {now:%B %Y, %H:%M} {_zone(now)}."


def _zone(now: datetime) -> str:
    offset = now.utcoffset()
    if offset is None:
        return "(local time)"
    minutes = round(offset.total_seconds() / 60)
    sign = "-" if minutes < 0 else "+"
    hours, mins = divmod(abs(minutes), 60)
    utc = f"(UTC{sign}{hours:02d}:{mins:02d})"
    name = now.tzname() or ""
    if name.isalpha() and name.upper() != "UTC":
        return f"{name} {utc}"
    return utc


def estimate_tokens(text: str) -> int:
    """Estimate the token count of `text` as ceil(characters / 4).

    This is a rough heuristic, not a tokenizer: real counts vary by model and
    language. Pass a model-specific counter to `ContextBuilder` when one exists.
    """
    return math.ceil(len(text) / 4)


class ContextBudgets(BaseModel):
    """Fractions of the usable window given to each section. They must sum to at most 1."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile: float = Field(default=0.10, ge=0, le=1)
    instructions: float = Field(default=0.10, ge=0, le=1)
    memories: float = Field(default=0.25, ge=0, le=1)
    history: float = Field(default=0.50, ge=0, le=1)

    @model_validator(mode="after")
    def _at_most_one(self) -> Self:
        total = self.profile + self.instructions + self.memories + self.history
        if total > 1 + 1e-9:
            raise ValueError(f"context budgets sum to {total:.3f}; they must sum to at most 1")
        return self


class SectionReport(BaseModel):
    """What one section contributed: tokens used, and items included and dropped.

    Items are lines for instructions and memories, messages for history, and the
    whole text for persona and profile. `truncated` is set when the profile was
    cut to fit its budget.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: SectionName
    tokens: int
    included: int
    dropped: int
    truncated: bool = False


class BuiltContext(BaseModel):
    """The messages to send to the LLM, plus a per-section report for the trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    messages: list[Message]
    report: list[SectionReport]


class ContextBuilder:
    """Builds the messages for one LLM call from memory and the conversation."""

    def __init__(
        self,
        persona: str,
        *,
        budgets: ContextBudgets | None = None,
        count_tokens: Callable[[str], int] = estimate_tokens,
        clock: Callable[[], datetime] | None = None,
        layout: ContextLayout = "stable_prefix",
    ) -> None:
        """`clock` returns the current local time (timezone-aware, so the line names
        the zone); with it, every build adds `date_line` (counted with the persona):
        after the persona in the `"system"` layout, in the per-turn block in the
        `"stable_prefix"` layout."""
        self._persona = persona.strip()
        self._budgets = budgets if budgets is not None else ContextBudgets()
        self._count = count_tokens
        self._clock = clock
        self._layout: ContextLayout = layout

    @property
    def layout(self) -> ContextLayout:
        return self._layout

    def _date(self) -> str:
        return "" if self._clock is None else date_line(self._clock())

    def _persona_text(self) -> str:
        """The persona plus the date line: what the persona section counts."""
        date = self._date()
        return f"{self._persona}\n\n{date}" if date else self._persona

    def token_budgets(self, context_window: int, reserve_output: int = 1024) -> dict[str, int]:
        """Token budget of each budgeted section for a model with `context_window` tokens."""
        return self._budgets_for(self._persona_text(), context_window, reserve_output)

    def _budgets_for(
        self, persona: str, context_window: int, reserve_output: int
    ) -> dict[str, int]:
        usable = max(0, context_window - reserve_output - self._count(persona))
        budgets = self._budgets
        return {
            "profile": int(usable * budgets.profile),
            "instructions": int(usable * budgets.instructions),
            "memories": int(usable * budgets.memories),
            "history": int(usable * budgets.history),
        }

    def build(
        self,
        *,
        memory: MemoryContext,
        history: Sequence[Message],
        context_window: int,
        reserve_output: int = 1024,
    ) -> BuiltContext:
        """Assemble the messages for one LLM call.

        `history` must already end with the current user message, which is always
        kept even if it alone exceeds the history budget.
        """
        date = self._date()
        persona = f"{self._persona}\n\n{date}" if date else self._persona
        budgets = self._budgets_for(persona, context_window, reserve_output)
        persona_report = SectionReport(
            name="persona", tokens=self._count(persona), included=1, dropped=0
        )
        profile_text, profile_report = self._profile(memory.profile, budgets["profile"])
        instructions_text, instructions_report = self._lines(
            "instructions", INSTRUCTIONS_LABEL, memory.instructions, budgets["instructions"]
        )
        memories_text, memories_report = self._lines(
            "memories", MEMORIES_LABEL, memory.statements, budgets["memories"]
        )
        kept_history, history_report = self._history(history, budgets["history"])

        if self._layout == "system":
            parts = [persona, profile_text, instructions_text, memories_text]
            messages = [_system(parts), *kept_history]
        else:
            stable = _system([self._persona, instructions_text, profile_text])
            messages = [stable, *self._with_turn_block(kept_history, [date, memories_text])]
        return BuiltContext(
            messages=messages,
            report=[
                persona_report,
                profile_report,
                instructions_report,
                memories_report,
                history_report,
            ],
        )

    @staticmethod
    def _with_turn_block(history: list[Message], parts: Sequence[str]) -> list[Message]:
        """`history` with the per-turn block as a system message right after its latest
        user message; with nothing to say, or no user message, `history` as it is."""
        block = "\n\n".join(p for p in parts if p)
        latest = max((i for i, m in enumerate(history) if m.role is Role.USER), default=None)
        if not block or latest is None:
            return list(history)
        note = Message(role=Role.SYSTEM, content=block)
        return [*history[: latest + 1], note, *history[latest + 1 :]]

    def prefix_messages(self, built: BuiltContext) -> list[Message]:
        """The part of `built` (a `"stable_prefix"` build) that comes before its latest
        user message: what the next turn's prompt is guaranteed to start with, given
        that `built` was made over the same history plus a placeholder user message.

        Used to prepare the model server's cache before the next turn (the
        residency policy's re-warm). Raises `ValueError` for the `"system"`
        layout, whose first message changes every turn, or a build with no user
        message."""
        if self._layout != "stable_prefix":
            raise ValueError("only the stable_prefix layout has a reusable prefix")
        messages = built.messages
        latest = max((i for i, m in enumerate(messages) if m.role is Role.USER), default=None)
        if latest is None:
            raise ValueError("the build has no user message")
        return list(messages[:latest])

    # --- Sections -------------------------------------------------------------

    def _profile(self, profile: str, budget: int) -> tuple[str, SectionReport]:
        profile = profile.strip()
        if not profile:
            return "", SectionReport(name="profile", tokens=0, included=0, dropped=0)
        header = self._count(PROFILE_LABEL)
        body_budget = budget - header
        truncated = False
        if self._count(profile) > body_budget:
            profile = self._truncate(profile, body_budget)
            truncated = True
        if not profile:
            return "", SectionReport(name="profile", tokens=0, included=0, dropped=1)
        tokens = header + self._count(profile)
        report = SectionReport(
            name="profile", tokens=tokens, included=1, dropped=0, truncated=truncated
        )
        return f"{PROFILE_LABEL}\n{profile}", report

    def _truncate(self, text: str, max_tokens: int) -> str:
        """The longest word-boundary prefix of `text` that, with an ellipsis, fits."""
        if max_tokens <= 0:
            return ""
        low, high = 0, len(text)
        while low < high:  # largest n with count(text[:n] + "…") <= max_tokens
            mid = (low + high + 1) // 2
            if self._count(text[:mid] + "…") <= max_tokens:
                low = mid
            else:
                high = mid - 1
        prefix = text[:low]
        if low < len(text) and not text[low].isspace() and " " in prefix:
            prefix = prefix[: prefix.rindex(" ")]
        prefix = prefix.rstrip()
        return f"{prefix}…" if prefix else ""

    def _lines(
        self, name: SectionName, label: str, items: Sequence[str], budget: int
    ) -> tuple[str, SectionReport]:
        items = [item.strip() for item in items if item.strip()]
        if not items:
            return "", SectionReport(name=name, tokens=0, included=0, dropped=0)
        used = self._count(label)
        lines: list[str] = []
        for item in items:
            line = f"- {item}"
            cost = self._count(line)
            if used + cost > budget:
                break
            lines.append(line)
            used += cost
        dropped = len(items) - len(lines)
        if not lines:
            return "", SectionReport(name=name, tokens=0, included=0, dropped=dropped)
        report = SectionReport(name=name, tokens=used, included=len(lines), dropped=dropped)
        return "\n".join([label, *lines]), report

    def _history(
        self, history: Sequence[Message], budget: int
    ) -> tuple[list[Message], SectionReport]:
        groups = _group(history)
        # A tool result without its assistant tool call is never valid on its own.
        orphans = sum(len(g) for g in groups if g[0].role is Role.TOOL)
        groups = [g for g in groups if g[0].role is not Role.TOOL]
        user_indexes = [i for i, g in enumerate(groups) if g[0].role is Role.USER]
        first_kept = user_indexes[-1] if user_indexes else len(groups)
        # The latest user message and everything after it are always kept.
        used = sum(self._message_tokens(m) for g in groups[first_kept:] for m in g)
        for i in range(first_kept - 1, -1, -1):
            cost = sum(self._message_tokens(m) for m in groups[i])
            if used + cost > budget:
                break
            used += cost
            first_kept = i
        kept = [m for g in groups[first_kept:] for m in g]
        dropped = orphans + sum(len(g) for g in groups[:first_kept])
        report = SectionReport(name="history", tokens=used, included=len(kept), dropped=dropped)
        return kept, report

    def _message_tokens(self, message: Message) -> int:
        tokens = self._count(message.content)
        for call in message.tool_calls or ():
            tokens += self._count(call.name) + self._count(json.dumps(call.arguments))
        return tokens


def _system(parts: Sequence[str]) -> Message:
    return Message(role=Role.SYSTEM, content="\n\n".join(p for p in parts if p))


def _group(history: Sequence[Message]) -> list[list[Message]]:
    """Split history into groups, keeping an assistant tool call with its results."""
    groups: list[list[Message]] = []
    for message in history:
        if (
            message.role is Role.TOOL
            and groups
            and groups[-1][0].role is Role.ASSISTANT
            and groups[-1][0].tool_calls
        ):
            groups[-1].append(message)
        else:
            groups.append([message])
    return groups
