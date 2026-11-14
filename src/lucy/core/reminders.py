"""Reminders and timers: the model and the `ReminderStore` port.

A reminder (or timer: a reminder set as a length of time) is stored as PENDING
with the instant it is due. Delivery, done by a scheduler, goes in three steps
so that several processes on one database (`lucy chat` and `lucy listen`) never
speak the same reminder twice and a crashed one does not lose it:

1. `claim_due` atomically moves every due PENDING row, and every FIRING row whose
   claim is older than the lease, to FIRING owned by the caller;
2. `begin_delivery`, just before speaking, renews the claim and returns only the
   rows the caller still owns (not cancelled, not re-claimed after a lapsed lease);
3. `mark_delivered` records them as DELIVERED with the session and turn.

`ReminderService` drives that for a client (ADR 0020): `wait_due` returns when
something is due, the client calls `fire` at a moment it knows is idle and
publishes the `TimerFired` it returns, and the orchestrator's proactive turn
calls `handle` (registered as the "reminder" timer handler), says the words and
then calls the reply's `delivered`. Marking after speaking means a crash can
repeat a reminder but never lose it.

Everything here is storage-agnostic; the SQLite store lives in `lucy.memory`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from pydantic import AwareDatetime, BaseModel, ConfigDict

from lucy.core.confirm import open_request
from lucy.core.events import AnyEvent, TimerFired

logger = logging.getLogger(__name__)

DEFAULT_LEASE: Final = timedelta(seconds=60)
"""How long a claim holds before another process may take the reminder over."""


class ReminderKind(StrEnum):
    REMINDER = "reminder"
    TIMER = "timer"


class ReminderStatus(StrEnum):
    PENDING = "pending"
    FIRING = "firing"
    """Claimed by a scheduler that is about to deliver it."""
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class Reminder(BaseModel):
    """One reminder or timer, as stored. Times are UTC."""

    model_config = ConfigDict(frozen=True)

    id: int
    kind: ReminderKind
    text: str
    """What to remind about; a timer's optional label (empty when none)."""
    due_at: AwareDatetime
    status: ReminderStatus
    created_at: AwareDatetime
    updated_at: AwareDatetime
    repeat: str | None = None
    """Reserved for repeating reminders; always None for now."""
    session_id: str | None = None
    claimed_by: str | None = None
    claimed_at: AwareDatetime | None = None
    delivered_at: AwareDatetime | None = None
    delivered_session: str | None = None
    delivered_turn: str | None = None

    @property
    def label(self) -> str:
        """The id as the tools show it, e.g. "r12"."""
        return f"r{self.id}"

    @property
    def length(self) -> timedelta:
        """For a timer: how long it was set for (due minus created)."""
        return self.due_at - self.created_at


@runtime_checkable
class ReminderStore(Protocol):
    """Durable reminders. `claim_due` must be atomic across connections and processes."""

    async def add(
        self, kind: ReminderKind, text: str, due_at: datetime, *, session_id: str | None = None
    ) -> Reminder:
        """Store a PENDING reminder and return it. Ids are never reused."""
        ...

    async def get(self, reminder_id: int) -> Reminder | None: ...

    async def upcoming(self) -> list[Reminder]:
        """Reminders not yet delivered or cancelled (PENDING or FIRING), soonest first."""
        ...

    async def cancel(self, reminder_id: int) -> Reminder | None:
        """Cancel a reminder not yet delivered; the cancelled reminder, or None if
        there is no such reminder or it was already delivered or cancelled."""
        ...

    async def claim_due(
        self, now: datetime, owner: str, lease: timedelta = DEFAULT_LEASE
    ) -> list[Reminder]:
        """Claim (as FIRING, for `owner`) every PENDING reminder due by `now` and every
        FIRING one claimed at or before `now - lease`; soonest first."""
        ...

    async def begin_delivery(self, ids: Sequence[int], owner: str) -> list[Reminder]:
        """Renew `owner`'s claim on `ids`; returns those still FIRING for `owner`."""
        ...

    async def mark_delivered(
        self, ids: Sequence[int], owner: str, session_id: str | None, turn_id: str | None
    ) -> list[Reminder]:
        """Mark `owner`'s FIRING reminders among `ids` DELIVERED; returns them."""
        ...

    async def next_due(self, lease: timedelta = DEFAULT_LEASE) -> datetime | None:
        """When `claim_due` next has something to claim (possibly in the past), or None."""
        ...


# --- Delivery ---------------------------------------------------------------------

REMINDER_TIMER: Final = "reminder"
"""`TimerFired.name` of a reminder delivery."""

POLL_SECONDS: Final = 5.0
"""`wait_due` looks at the store at least this often, so reminders set by another
process (or a hold that has ended) are noticed."""


@dataclass(frozen=True)
class TimerReply:
    """What a timer handler wants said in a proactive turn.

    The orchestrator says `text` (code-authored, no model call), then awaits
    `delivered`, if any, to record that it was said."""

    text: str
    delivered: Callable[[], Awaitable[object]] | None = None


TimerHandler = Callable[[TimerFired, str], Awaitable[TimerReply | None]]
"""Handles one `TimerFired` in the turn with the given turn id. None: say nothing."""


class EpisodeLog(Protocol):
    """The part of `MemoryStore` the service reads (for the confirmation hold)."""

    async def recent_episodes(self, session_id: str, limit: int) -> list[AnyEvent]: ...


Announce = Callable[[Sequence[Reminder], datetime], str]
"""The words for reminders delivered at a moment (`lucy.tools.reminders.announcement`)."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def default_owner() -> str:
    """A claim owner unique to this process and service: host, pid and a random part."""
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


class ReminderService:
    """Fires due reminders for a client and delivers them in proactive turns (ADR 0020).

    The service never speaks and never publishes. A client runs, in a task:

        while True:
            await service.wait_due()
            # wait until the client is idle (no turn of its own, nobody speaking)
            event = await service.fire(session_id)
            if event is not None:
                await bus.publish(event)   # then show or speak that turn's reply

    `fire` returns None (and the next `wait_due` waits `poll` seconds) while a
    confirmation request of the session is open, or when nothing was claimed
    because another process got there first. `handle` is the orchestrator's
    "reminder" timer handler.
    """

    def __init__(
        self,
        store: ReminderStore,
        episodes: EpisodeLog,
        announce: Announce,
        *,
        clock: Callable[[], datetime] = _utc_now,
        owner: str | None = None,
        lease: timedelta = DEFAULT_LEASE,
        poll: float = POLL_SECONDS,
        history_limit: int = 200,
    ) -> None:
        self.store = store
        self._episodes = episodes
        self._announce = announce
        self._clock = clock
        self.owner = owner or default_owner()
        self.lease = lease
        self.poll = poll
        self._history_limit = history_limit
        self._backoff_until: datetime | None = None

    async def wait_due(self) -> None:
        """Return when a reminder is due (at once if one already is).

        Sleeps at most `poll` seconds between looks at the store. After a `fire`
        that delivered nothing, it first waits `poll` seconds, so a held or lost
        claim does not spin."""
        while True:
            now = self._clock()
            if self._backoff_until is not None and now < self._backoff_until:
                await asyncio.sleep(min((self._backoff_until - now).total_seconds(), self.poll))
                continue
            self._backoff_until = None
            due = await self.store.next_due(self.lease)
            now = self._clock()
            if due is not None and due <= now:
                return
            wait = self.poll if due is None else min((due - now).total_seconds(), self.poll)
            await asyncio.sleep(max(wait, 0.01))

    async def fire(self, session_id: str) -> TimerFired | None:
        """Claim every due reminder for a proactive turn of `session_id`.

        Call it only when the client is idle. Returns the `TimerFired` to publish,
        or None: while a confirmation request of the session is open (unresolved,
        unanswered, unexpired), or when there was nothing to claim."""
        now = self._clock()
        episodes = await self._episodes.recent_episodes(session_id, self._history_limit)
        if open_request(episodes, now) is not None:
            self._backoff_until = now + timedelta(seconds=self.poll)
            return None
        claimed = await self.store.claim_due(now, self.owner, self.lease)
        if not claimed:
            self._backoff_until = now + timedelta(seconds=self.poll)
            return None
        return TimerFired(
            session_id=session_id,
            name=REMINDER_TIMER,
            payload={"ids": [r.id for r in claimed], "owner": self.owner},
        )

    async def handle(self, event: TimerFired, turn_id: str) -> TimerReply | None:
        """The "reminder" timer handler: what to say for `event`, or None when none of
        its reminders is still ours (cancelled, or taken over after a lapsed lease)."""
        ids = [i for i in event.payload.get("ids", []) if isinstance(i, int)]
        owner = event.payload.get("owner")
        if not ids or not isinstance(owner, str):
            logger.warning("reminder timer %s has no ids or owner", event.id)
            return None
        rows = await self.store.begin_delivery(ids, owner)
        if not rows:
            return None
        text = self._announce(rows, self._clock())
        mine = [r.id for r in rows]

        async def delivered() -> None:
            await self.store.mark_delivered(mine, owner, event.session_id, turn_id)

        return TimerReply(text=text, delivered=delivered)
