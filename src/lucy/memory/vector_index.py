"""The vector index: statement and entity embeddings in sqlite-vec (ADR 0010).

Vectors are derived data. Each (embedding model, dimensions) pair is a *space*,
registered in the `vector_spaces` table, with two `vec0` virtual tables of its
own: `vec_statements_<space>` (rowid = statement id) and `vec_entities_<space>`
(rowid = entity id). So every vector records the model and dimensions that
produced it, a search only ever compares vectors within the current embedder's
space, and switching embedder simply starts a new, empty space; the old one is
ignored (and still maintained, so switching back is safe) until
`drop_all` removes it.

Statement vectors carry three metadata columns so that KNN search can filter
exactly, without over-fetching: `subject_id`, `kind` and `invalid_us` (the
statement's `invalid_at` in microseconds since the epoch, or `NEVER`). They are
denormalised copies of the `statements` columns, kept in sync explicitly by
`SQLiteGraphStore` in the same transaction as each change, in every space.

The methods here take a connection and run inside the caller's transaction;
they do not commit. The connection must have sqlite-vec loaded (`load_vec`).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import aiosqlite
import sqlite_vec

#: `invalid_us` for a statement that is still true.
NEVER: Final = 1 << 62
#: sqlite-vec's largest `k` for one KNN query.
MAX_K: Final = 4096

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)


async def load_vec(conn: aiosqlite.Connection) -> None:
    """Load the sqlite-vec extension into `conn` (extension loading is disabled again after)."""
    await conn.enable_load_extension(True)
    try:
        await conn.load_extension(sqlite_vec.loadable_path())
    finally:
        await conn.enable_load_extension(False)


def to_us(value: datetime | None) -> int:
    """`value` as integer microseconds since the epoch; `NEVER` for None."""
    if value is None:
        return NEVER
    delta = value.astimezone(UTC) - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _blob(vector: Sequence[float]) -> bytes:
    return sqlite_vec.serialize_float32(list(vector))


def _ids(ids: Iterable[int]) -> str:
    return json.dumps(sorted(set(ids)))


@dataclass(frozen=True)
class Space:
    """One embedding space: a model id and a dimension count."""

    id: int
    model: str
    dimensions: int

    @property
    def statements(self) -> str:
        return f"vec_statements_{self.id}"

    @property
    def entities(self) -> str:
        return f"vec_entities_{self.id}"


async def _all(conn: aiosqlite.Connection, sql: str, params: Sequence[Any] = ()) -> list[Any]:
    async with conn.execute(sql, params) as cursor:
        return list(await cursor.fetchall())


class VectorIndex:
    """Vector storage and search for one embedder (`model`, `dimensions`)."""

    def __init__(self, model: str, dimensions: int) -> None:
        self.model = model
        self.dimensions = dimensions

    # --- Spaces ---------------------------------------------------------------

    async def spaces(self, conn: aiosqlite.Connection) -> list[Space]:
        """Every registered space, current model or not."""
        rows = await _all(conn, "SELECT id, model, dimensions FROM vector_spaces ORDER BY id")
        return [Space(*row) for row in rows]

    async def current(self, conn: aiosqlite.Connection) -> Space | None:
        """The space of this embedder, or None if it has no vectors yet."""
        rows = await _all(
            conn,
            "SELECT id, model, dimensions FROM vector_spaces WHERE model = ? AND dimensions = ?",
            (self.model, self.dimensions),
        )
        return Space(*rows[0]) if rows else None

    async def ensure_current(self, conn: aiosqlite.Connection, now: str) -> Space:
        """The space of this embedder, creating it and its tables if needed (writes)."""
        await conn.execute(
            "INSERT INTO vector_spaces (model, dimensions, created_at) VALUES (?, ?, ?)"
            " ON CONFLICT (model, dimensions) DO NOTHING",
            (self.model, self.dimensions, now),
        )
        space = await self.current(conn)
        assert space is not None
        await conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {space.statements} USING vec0("
            f"embedding float[{space.dimensions}] distance_metric=cosine,"
            " subject_id integer, kind text, invalid_us integer)"
        )
        await conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {space.entities} USING vec0("
            f"embedding float[{space.dimensions}] distance_metric=cosine)"
        )
        return space

    async def drop_all(self, conn: aiosqlite.Connection) -> int:
        """Drop every space and its tables (writes). Returns the number of spaces dropped."""
        spaces = await self.spaces(conn)
        for space in spaces:
            await conn.execute(f"DROP TABLE IF EXISTS {space.statements}")
            await conn.execute(f"DROP TABLE IF EXISTS {space.entities}")
        await conn.execute("DELETE FROM vector_spaces")
        return len(spaces)

    # --- Writes ---------------------------------------------------------------

    async def put_statements(
        self,
        conn: aiosqlite.Connection,
        space: Space,
        rows: Sequence[tuple[int, Sequence[float], int, str, int]],
    ) -> None:
        """Store `(statement_id, vector, subject_id, kind, invalid_us)` rows in `space`,
        replacing any existing vector for the same statement."""
        if not rows:
            return
        await conn.execute(
            f"DELETE FROM {space.statements} WHERE rowid IN (SELECT value FROM json_each(?))",
            (_ids(row[0] for row in rows),),
        )
        await conn.executemany(
            f"INSERT INTO {space.statements} (rowid, embedding, subject_id, kind, invalid_us)"
            " VALUES (?, ?, ?, ?, ?)",
            [(sid, _blob(vec), subject, kind, inv) for sid, vec, subject, kind, inv in rows],
        )

    async def put_entities(
        self, conn: aiosqlite.Connection, space: Space, rows: Sequence[tuple[int, Sequence[float]]]
    ) -> None:
        """Store `(entity_id, vector)` rows in `space`, replacing existing ones."""
        if not rows:
            return
        await conn.execute(
            f"DELETE FROM {space.entities} WHERE rowid IN (SELECT value FROM json_each(?))",
            (_ids(row[0] for row in rows),),
        )
        await conn.executemany(
            f"INSERT INTO {space.entities} (rowid, embedding) VALUES (?, ?)",
            [(eid, _blob(vec)) for eid, vec in rows],
        )

    async def delete_statements(self, conn: aiosqlite.Connection, ids: Iterable[int]) -> None:
        """Remove the vectors of statements `ids` from every space."""
        payload = _ids(ids)
        for space in await self.spaces(conn):
            await conn.execute(
                f"DELETE FROM {space.statements} WHERE rowid IN (SELECT value FROM json_each(?))",
                (payload,),
            )

    async def delete_entities(self, conn: aiosqlite.Connection, ids: Iterable[int]) -> None:
        """Remove the vectors of entities `ids` from every space."""
        payload = _ids(ids)
        for space in await self.spaces(conn):
            await conn.execute(
                f"DELETE FROM {space.entities} WHERE rowid IN (SELECT value FROM json_each(?))",
                (payload,),
            )

    async def set_subject(
        self, conn: aiosqlite.Connection, ids: Iterable[int], subject_id: int
    ) -> None:
        """Update the `subject_id` metadata of statements `ids` in every space."""
        payload = _ids(ids)
        for space in await self.spaces(conn):
            await conn.execute(
                f"UPDATE {space.statements} SET subject_id = ?"
                " WHERE rowid IN (SELECT value FROM json_each(?))",
                (subject_id, payload),
            )

    async def set_invalid(
        self, conn: aiosqlite.Connection, ids: Iterable[int], invalid_at: datetime | None
    ) -> None:
        """Update the `invalid_us` metadata of statements `ids` in every space."""
        payload = _ids(ids)
        for space in await self.spaces(conn):
            await conn.execute(
                f"UPDATE {space.statements} SET invalid_us = ?"
                " WHERE rowid IN (SELECT value FROM json_each(?))",
                (to_us(invalid_at), payload),
            )

    # --- Reads ----------------------------------------------------------------

    async def search_statements(
        self,
        conn: aiosqlite.Connection,
        vector: Sequence[float],
        k: int,
        *,
        subject_id: int | None = None,
        kinds: Sequence[str] | None = None,
        current_at: datetime | None = None,
    ) -> list[tuple[int, float]]:
        """The `k` nearest statements as `(id, cosine similarity)`, best first.

        Filters are applied inside the KNN scan: `subject_id`, `kinds` (an
        allow-list) and `current_at` (only statements whose `invalid_at` is
        after it). Returns [] if this embedder has no space yet.
        """
        space = await self.current(conn)
        if space is None or k <= 0 or (kinds is not None and not kinds):
            return []
        where = ["embedding MATCH ?", "k = ?"]
        params: list[Any] = [_blob(vector), min(k, MAX_K)]
        if subject_id is not None:
            where.append("subject_id = ?")
            params.append(subject_id)
        if kinds is not None:
            where.append(f"kind IN ({', '.join('?' for _ in kinds)})")
            params.extend(kinds)
        if current_at is not None:
            where.append("invalid_us > ?")
            params.append(to_us(current_at))
        rows = await _all(
            conn,
            f"SELECT rowid, distance FROM {space.statements} WHERE {' AND '.join(where)}"
            " ORDER BY distance",
            params,
        )
        return [(row[0], 1.0 - row[1]) for row in rows if row[1] is not None]

    async def search_entities(
        self, conn: aiosqlite.Connection, vector: Sequence[float], k: int
    ) -> list[tuple[int, float]]:
        """The `k` nearest entities as `(id, cosine similarity)`, best first."""
        space = await self.current(conn)
        if space is None or k <= 0:
            return []
        rows = await _all(
            conn,
            f"SELECT rowid, distance FROM {space.entities} WHERE embedding MATCH ? AND k = ?"
            " ORDER BY distance",
            (_blob(vector), min(k, MAX_K)),
        )
        return [(row[0], 1.0 - row[1]) for row in rows if row[1] is not None]

    async def missing_statements(
        self, conn: aiosqlite.Connection, limit: int | None, *, after_id: int = 0
    ) -> list[int]:
        """Ids above `after_id` of statements without a vector in this space, ascending."""
        space = await self.current(conn)
        lim = -1 if limit is None else limit
        if space is None:
            rows = await _all(
                conn, "SELECT id FROM statements WHERE id > ? ORDER BY id LIMIT ?", (after_id, lim)
            )
        else:
            rows = await _all(
                conn,
                f"SELECT s.id FROM statements s WHERE s.id > ? AND NOT EXISTS"
                f" (SELECT 1 FROM {space.statements} v WHERE v.rowid = s.id) ORDER BY s.id LIMIT ?",
                (after_id, lim),
            )
        return [row[0] for row in rows]

    async def missing_entities(
        self, conn: aiosqlite.Connection, limit: int | None, *, after_id: int = 0
    ) -> list[int]:
        """Ids above `after_id` of unmerged entities without a vector in this space, ascending."""
        space = await self.current(conn)
        lim = -1 if limit is None else limit
        if space is None:
            rows = await _all(
                conn,
                "SELECT id FROM entities WHERE merged_into IS NULL AND id > ? ORDER BY id LIMIT ?",
                (after_id, lim),
            )
        else:
            rows = await _all(
                conn,
                "SELECT e.id FROM entities e WHERE e.merged_into IS NULL AND e.id > ?"
                f" AND NOT EXISTS (SELECT 1 FROM {space.entities} v WHERE v.rowid = e.id)"
                " ORDER BY e.id LIMIT ?",
                (after_id, lim),
            )
        return [row[0] for row in rows]

    async def has_statement(self, conn: aiosqlite.Connection, space: Space, sid: int) -> bool:
        rows = await _all(conn, f"SELECT 1 FROM {space.statements} WHERE rowid = ?", (sid,))
        return bool(rows)

    async def has_entity(self, conn: aiosqlite.Connection, space: Space, eid: int) -> bool:
        rows = await _all(conn, f"SELECT 1 FROM {space.entities} WHERE rowid = ?", (eid,))
        return bool(rows)


__all__ = ["MAX_K", "NEVER", "Space", "VectorIndex", "load_vec", "to_us"]
