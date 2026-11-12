"""SQLite implementation of the `MemoryStore` port (ADR 0010).

It holds the append-only episode log and one trace per turn (it also
implements the `TraceSink` port). The knowledge graph lives in the same file
but is managed by `lucy.memory.graph_store.SQLiteGraphStore` on a connection
of its own; `retrieve` delegates to a `Retriever` once one is attached with
`use_retriever`. Connection setup and migrations live in `lucy.memory.db`,
shared with the job store.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from datetime import UTC
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

import aiosqlite

from lucy.core import registry
from lucy.core.events import AnyEvent, event_from_json, event_to_json
from lucy.core.trace import TurnTrace
from lucy.core.types import MemoryContext
from lucy.memory.db import MemoryStoreError, apply_migrations, open_database

__all__ = ["MemoryStoreError", "SQLiteMemoryStore", "apply_migrations"]


class _Retriever(Protocol):
    async def retrieve(self, query: str, budget: int) -> MemoryContext: ...


@registry.register("memory", "sqlite")
class SQLiteMemoryStore:
    """Memory in one SQLite file. Connects lazily; safe to share between coroutines.

    Usable as an async context manager, which connects on entry and closes on exit.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._retriever: _Retriever | None = None

    def use_retriever(self, retriever: _Retriever | None) -> None:
        """Answer `retrieve` with `retriever` (a `lucy.memory.retrieval.Retriever`);
        None detaches it again."""
        self._retriever = retriever

    async def _connection(self) -> aiosqlite.Connection:
        """Return the open connection, opening and migrating the database if needed."""
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

    async def append_episode(self, event: AnyEvent) -> None:
        """Append `event` to the log; raises `MemoryStoreError` if its id is already there."""
        conn = await self._connection()
        try:
            await conn.execute(
                "INSERT INTO episodes (event_id, ts, session_id, turn_id, type, schema_version,"
                " payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    event.id,
                    event.ts.isoformat(),
                    event.session_id,
                    event.turn_id,
                    event.type,
                    event.schema_version,
                    event_to_json(event),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise MemoryStoreError(f"episode {event.id!r} is already in the log") from exc

    async def episode_id(self, event_id: str) -> int | None:
        """The log row id of the event `event_id` (the id statements cite), or None."""
        conn = await self._connection()
        async with conn.execute("SELECT id FROM episodes WHERE event_id = ?", (event_id,)) as cur:
            row = await cur.fetchone()
        return None if row is None else int(row[0])

    async def recent_episodes(self, session_id: str, limit: int) -> list[AnyEvent]:
        """The last `limit` events of `session_id`, oldest first; none if `limit` <= 0."""
        if limit <= 0:
            return []
        conn = await self._connection()
        async with conn.execute(
            "SELECT payload FROM (SELECT id, payload FROM episodes WHERE session_id = ?"
            " ORDER BY id DESC LIMIT ?) ORDER BY id",
            (session_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [event_from_json(row[0]) for row in rows]

    async def record(self, trace: TurnTrace) -> None:
        """Store the trace of one turn (`TraceSink` port).

        Raises `MemoryStoreError` if a trace for the same turn is already stored.
        """
        conn = await self._connection()
        try:
            await conn.execute(
                "INSERT INTO turn_traces (turn_id, session_id, started_at, ended_at, outcome,"
                " payload) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    trace.turn_id,
                    trace.session_id,
                    trace.started_at.astimezone(UTC).isoformat(),
                    trace.ended_at.astimezone(UTC).isoformat(),
                    trace.outcome,
                    trace.model_dump_json(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise MemoryStoreError(f"a trace for turn {trace.turn_id!r} is already stored") from exc

    async def get_trace(self, turn_id: str) -> TurnTrace | None:
        """The stored trace of `turn_id`, or None if there is none."""
        conn = await self._connection()
        async with conn.execute(
            "SELECT payload FROM turn_traces WHERE turn_id = ?", (turn_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return TurnTrace.model_validate_json(row[0]) if row else None

    async def retrieve(self, query: str, budget: int) -> MemoryContext:
        """Memory for `query` from the attached retriever; empty if none is attached."""
        if self._retriever is None:
            return MemoryContext()
        return await self._retriever.retrieve(query, budget)
