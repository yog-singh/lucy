"""Who a turn is with: the owner or a guest (ADR 0027).

With `[people] enabled`, every user turn is with the owner (`[people] owner`) or with
a guest, decided in code from the session's events up to and including the turn's
`UserMessage`, by the first of these rules that decides:

(a) a typed message (`source` "text", `lucy chat`; also a memory correction) is
    the owner's ("typed");
(b) a voice tag on the message decides: the owner's name is the owner, any other
    enrolled name a guest with that name ("voice");
(c) no voice tag, but a glance on the message: the owner in view and nobody else is
    the owner; faces in view without the owner are a guest, named when exactly one
    recognised person and no unknown face is in view; the owner with others in view
    decides nothing ("face");
(d) otherwise the latest earlier message of the session that rules (a) to (c)
    decided is carried over if it is at most `carry_minutes` older than this one
    ("carried"); an older one, or none, makes a guest ("none").

Carrying looks only at decisions rules (a) to (c) made, so a carried decision never
carries itself on: the window counts from the last real decision. A `TimerFired`
(a reminder, a timer, an enrolment outcome) is not a user message and decides
nothing. Ages are differences of the events' own timestamps, never the wall clock,
so the same log always gives the same answer, however much later it is computed.
Names compare ignoring case.

Not authentication (ADR 0027, Security; ADR 0021): a recording or a photo of the
owner passes. The orchestrator writes the result as a `TurnWith` record; what acts
on it (run B: guest restrictions, what is learned) reads that record.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import timedelta
from typing import Final

from pydantic import BaseModel, ConfigDict

from lucy.core.events import (
    AnyEvent,
    PresenceBasis,
    PresenceRole,
    TurnWith,
    UserMessage,
)


class Presence(BaseModel):
    """Who one turn is with, and which rule decided it (for the trace)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: PresenceRole
    name: str | None = None
    basis: PresenceBasis

    @property
    def guest(self) -> bool:
        return self.role == "guest"

    def label(self) -> str:
        """Short words for a screen: "owner", "guest", "guest: Asha"."""
        if self.role == "guest" and self.name:
            return f"guest: {self.name}"
        return self.role

    def record(self, session_id: str, turn_id: str | None, message_id: str) -> TurnWith:
        return TurnWith(
            session_id=session_id,
            turn_id=turn_id,
            role=self.role,
            name=self.name,
            basis=self.basis,
            message_id=message_id,
        )

    @classmethod
    def from_record(cls, record: TurnWith) -> Presence:
        return cls(role=record.role, name=record.name, basis=record.basis)


class PresencePolicy:
    """Decides who a turn is with (module docstring). Pure: no I/O, no clock."""

    def __init__(self, owner: str, carry_minutes: float = 5.0) -> None:
        if not owner.strip():
            raise ValueError("the owner's name is required")
        if carry_minutes < 0:
            raise ValueError("carry_minutes must not be negative")
        self.owner = owner
        self._owner = owner.casefold()
        self._carry = timedelta(minutes=carry_minutes)

    def decide(self, events: Sequence[AnyEvent], message: UserMessage) -> Presence:
        """Who the turn of `message` is with. `events` are the session's events, oldest
        first; they may already include `message` (anything after it is ignored)."""
        own = self._own(message)
        if own is not None:
            return own
        earlier = _before(events, message)
        for event in reversed(earlier):
            if not isinstance(event, UserMessage) or event.session_id != message.session_id:
                continue
            decided = self._own(event)
            if decided is None:
                continue  # it was carried or undecided itself: look further back
            if message.ts - event.ts <= self._carry:
                return Presence(role=decided.role, name=decided.name, basis="carried")
            break  # the latest real decision is too old; older ones are older still
        return Presence(role="guest", name=None, basis="none")

    def _own(self, message: UserMessage) -> Presence | None:
        """Rules (a) to (c): what `message` alone says, or None."""
        if message.source in ("text", "correction"):
            return Presence(role="owner", name=self.owner, basis="typed")
        if message.speaker is not None:
            if self._is_owner(message.speaker):
                return Presence(role="owner", name=message.speaker, basis="voice")
            return Presence(role="guest", name=message.speaker, basis="voice")
        seen = message.seen
        if seen is None:
            return None
        owners = [p.name for p in seen.people if self._is_owner(p.name)]
        others = [p.name for p in seen.people if not self._is_owner(p.name)]
        if owners:
            if not others and seen.unknown == 0:
                return Presence(role="owner", name=owners[0], basis="face")
            return None  # the owner with others: it could be anyone speaking
        if others or seen.unknown:
            name = others[0] if len(others) == 1 and seen.unknown == 0 else None
            return Presence(role="guest", name=name, basis="face")
        return None  # nobody in view

    def _is_owner(self, name: str) -> bool:
        return name.casefold() == self._owner


def _before(events: Sequence[AnyEvent], message: UserMessage) -> Sequence[AnyEvent]:
    """`events` before `message` (all of them if it is not among them)."""
    for index in range(len(events) - 1, -1, -1):
        if events[index].id == message.id:
            return events[:index]
    return events


def guest_run(events: Sequence[AnyEvent], name: str | None) -> set[str]:
    """The turn ids of the current run of guest turns: the latest `TurnWith` records of
    `events`, newest first, while each is a guest's with the same `name` (ignoring
    case). An owner's turn, or a guest with another name (or none where there was
    one), ends the run. Proactive turns have no record and neither end nor join it.
    Empty when the latest record is not such a guest's. (Run B, ADR 0027: a guest's
    prompt holds only these turns' exchanges.)"""
    folded = name.casefold() if name else None
    run: set[str] = set()
    for event in reversed(events):
        if not isinstance(event, TurnWith):
            continue
        other = event.name.casefold() if event.name else None
        if event.role != "guest" or other != folded or event.turn_id is None:
            break
        run.add(event.turn_id)
    return run


def latest_turn_with(events: Sequence[AnyEvent]) -> Presence | None:
    """The latest recorded decision in `events`, or None. `next_turn_prefix` assumes
    the next turn is with the same person (ADR 0027, run B)."""
    for event in reversed(events):
        if isinstance(event, TurnWith):
            return Presence.from_record(event)
    return None


_CAPABILITIES: Final[tuple[tuple[frozenset[str], str], ...]] = (
    (frozenset({"wikipedia", "web_search", "fetch_page"}), "look things up"),
    (frozenset({"time"}), "tell the time"),
    (frozenset({"weather"}), "tell the weather"),
    (frozenset({"set_timer"}), "set a timer"),
    (frozenset({"look"}), "say what the camera shows"),
)


def guest_line(owner: str, guest: str | None, tools: Iterable[str]) -> str:
    """The per-turn line telling the model it is talking with a guest (ADR 0027): who,
    what it can do for them (only what `tools`, the guest's configured tools, allow)
    and that the owner's things are only for the owner."""
    names = set(tools)
    can = [words for group, words in _CAPABILITIES if group & names]
    who = f"a guest, {guest}," if guest else "a guest"
    line = f"You are talking with {who} not {owner}."
    if can:
        listed = can[0] if len(can) == 1 else f"{', '.join(can[:-1])} and {can[-1]}"
        line += f" You can {listed}."
    return (
        f"{line} Anything about {owner} ({owner}'s calendar, reminders, email, or what you "
        f"know about {owner}) and remembering a voice or a face are only for {owner}: if "
        f"asked, say briefly and kindly that it is only for {owner}, do not offer anything "
        "else, and do not guess."
    )


def turn_with(events: Sequence[AnyEvent], turn_id: str) -> Presence | None:
    """The recorded decision for `turn_id`, or None (`[people]` off, or a log from
    before it). Run B and memory extraction read this rather than deciding again."""
    for event in reversed(events):
        if isinstance(event, TurnWith) and event.turn_id == turn_id:
            return Presence.from_record(event)
    return None
