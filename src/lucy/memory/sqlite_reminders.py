"""SQLite implementation of the `ReminderStore` port (`lucy.core.reminders`).

Reminders live in the `reminders` table of the same file as memory, on a
connection of their own in autocommit mode. Every operation is a single SQL
statement, so each is atomic even with several connections or processes on the
file (`lucy chat` and `lucy listen` at once): two claimers never get the same
reminder, and a cancel and a claim cannot both win.

Timestamps are fixed-width UTC ISO 8601 text (always with microseconds), so
comparing the text compares the instants.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import aiosqlite

from lucy.core.reminders import (
    DEFAULT_LEASE,
    Reminder,
    ReminderKind,
    ReminderStatus,
)
from lucy.memory.db import open_database

_COLUMNS = (
    "id, kind, text, due_at, status, repeat, session_id, created_at, updated_at, claimed_by,"
    " claimed_at, delivered_at, delivered_session, delivered_turn"
)
_OPEN = "('pending', 'firing')"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _ts(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError(f"naive datetime {value!r}; reminder times must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _dt(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _reminder(row: Any) -> Reminder:
    return Reminder(
        id=row[0],
        kind=ReminderKind(row[1]),
        text=row[2],
        due_at=datetime.fromisoformat(row[3]),
        status=ReminderStatus(row[4]),
        repeat=row[5],
        session_id=row[6],
        created_at=datetime.fromisoformat(row[7]),
        updated_at=datetime.fromisoformat(row[8]),
        claimed_by=row[9],
        claimed_at=_dt(row[10]),
        delivered_at=_dt(row[11]),
        delivered_session=row[12],
        delivered_turn=row[13],
    )


def _sorted(rows: list[Any]) -> list[Reminder]:
    return sorted((_reminder(r) for r in rows), key=lambda r: (r.due_at, r.id))


class SQLiteReminderStore:
    """Reminders in the Lucy SQLite file. Connects lazily; safe to share between coroutines.

    Usable as an async context manager. `clock` supplies `created_at`,
    `updated_at`, and the claim time `begin_delivery` stamps.
    """

    def __init__(
        self, path: str | os.PathLike[str], *, clock: Callable[[], datetime] = _utc_now
    ) -> None:
        self.path = Path(path)
        self._clock = clock
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._statement_lock = asyncio.Lock()

    async def _connection(self) -> aiosqlite.Connection:
        if self._conn is not None:
            return self._conn
        async with self._lock:
            if self._conn is None:
                self._conn = await open_database(self.path)
            return self._conn

    async def close(self) -> None:
        """Close the connection. The store reconnects if used again."""
        async with self._lock:
            if self._conn is not None:
                conn, self._conn = self._conn, None
                await conn.close()

    async def __aenter__(self) -> Self:
        await self._connection()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def _run(self, sql: str, params: Sequence[Any] = ()) -> list[Any]:
        """Run one statement to completion and return all its rows."""
        conn = await self._connection()
        async with self._statement_lock:
            async with conn.execute(sql, params) as cursor:
                return list(await cursor.fetchall())  # drain, so RETURNING completes

    async def add(
        self, kind: ReminderKind, text: str, due_at: datetime, *, session_id: str | None = None
    ) -> Reminder:
        now = _ts(self._clock())
        rows = await self._run(
            "INSERT INTO reminders (kind, text, due_at, status, session_id, created_at,"
            f" updated_at) VALUES (?, ?, ?, 'pending', ?, ?, ?) RETURNING {_COLUMNS}",
            (ReminderKind(kind).value, text, _ts(due_at), session_id, now, now),
        )
        return _reminder(rows[0])

    async def get(self, reminder_id: int) -> Reminder | None:
        rows = await self._run(f"SELECT {_COLUMNS} FROM reminders WHERE id = ?", (reminder_id,))
        return _reminder(rows[0]) if rows else None

    async def upcoming(self) -> list[Reminder]:
        rows = await self._run(
            f"SELECT {_COLUMNS} FROM reminders WHERE status IN {_OPEN} ORDER BY due_at, id"
        )
        return [_reminder(r) for r in rows]

    async def cancel(self, reminder_id: int) -> Reminder | None:
        rows = await self._run(
            "UPDATE reminders SET status = 'cancelled', updated_at = ?"
            f" WHERE id = ? AND status IN {_OPEN} RETURNING {_COLUMNS}",
            (_ts(self._clock()), reminder_id),
        )
        return _reminder(rows[0]) if rows else None

    async def claim_due(
        self, now: datetime, owner: str, lease: timedelta = DEFAULT_LEASE
    ) -> list[Reminder]:
        # One statement takes the write lock before reading, so concurrent claimers
        # on other connections serialise and each row goes to exactly one of them.
        rows = await self._run(
            "UPDATE reminders SET status = 'firing', claimed_by = ?, claimed_at = ?,"
            " updated_at = ?"
            " WHERE (status = 'pending' AND due_at <= ?)"
            " OR (status = 'firing' AND claimed_at <= ?)"
            f" RETURNING {_COLUMNS}",
            (owner, _ts(now), _ts(now), _ts(now), _ts(now - lease)),
        )
        return _sorted(rows)

    async def begin_delivery(self, ids: Sequence[int], owner: str) -> list[Reminder]:
        now = _ts(self._clock())
        rows = await self._run(
            "UPDATE reminders SET claimed_at = ?, updated_at = ?"
            " WHERE id IN (SELECT value FROM json_each(?))"
            " AND status = 'firing' AND claimed_by = ?"
            f" RETURNING {_COLUMNS}",
            (now, now, json.dumps(list(ids)), owner),
        )
        return _sorted(rows)

    async def mark_delivered(
        self, ids: Sequence[int], owner: str, session_id: str | None, turn_id: str | None
    ) -> list[Reminder]:
        now = _ts(self._clock())
        rows = await self._run(
            "UPDATE reminders SET status = 'delivered', delivered_at = ?, delivered_session = ?,"
            " delivered_turn = ?, updated_at = ?"
            " WHERE id IN (SELECT value FROM json_each(?))"
            " AND status = 'firing' AND claimed_by = ?"
            f" RETURNING {_COLUMNS}",
            (now, session_id, turn_id, now, json.dumps(list(ids)), owner),
        )
        return _sorted(rows)

    async def next_due(self, lease: timedelta = DEFAULT_LEASE) -> datetime | None:
        rows = await self._run(
            "SELECT (SELECT min(due_at) FROM reminders WHERE status = 'pending'),"
            " (SELECT min(claimed_at) FROM reminders WHERE status = 'firing')"
        )
        due, claimed = rows[0]
        times = [datetime.fromisoformat(due)] if due else []
        if claimed:
            times.append(datetime.fromisoformat(claimed) + lease)
        return min(times) if times else None
