"""SQLite implementation of the `JobStore` port (ADR 0005).

Jobs live in the `jobs` table of the same file as memory, on a connection of
their own. Every operation is a single SQL statement, so each is atomic even
with several connections or processes on the file: in particular two claimers
never get the same job, and the partial unique index `jobs_dedupe_active`
guarantees at most one pending or running job per dedupe key.

Timestamps are stored as fixed-width UTC ISO 8601 text (always with
microseconds), so comparing the text compares the instants.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from collections.abc import Callable, Collection, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import aiosqlite

from lucy.core.jobs import Job, JobStateError, JobStatus, utc_now
from lucy.memory.db import open_database

_COLUMNS = (
    "id, kind, payload, status, attempts, max_attempts, run_after, created_at, updated_at,"
    " last_error, dedupe_key"
)
_ACTIVE = "('pending', 'running')"


def _ts(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError(f"naive datetime {value!r}; job times must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _job(row: sqlite3.Row | tuple[Any, ...]) -> Job:
    return Job(
        id=row[0],
        kind=row[1],
        payload=json.loads(row[2]),
        status=JobStatus(row[3]),
        attempts=row[4],
        max_attempts=row[5],
        run_after=datetime.fromisoformat(row[6]),
        created_at=datetime.fromisoformat(row[7]),
        updated_at=datetime.fromisoformat(row[8]),
        last_error=row[9],
        dedupe_key=row[10],
    )


class SQLiteJobStore:
    """The job queue in the Lucy SQLite file. Connects lazily; safe to share between coroutines.

    Usable as an async context manager, which connects on entry and closes on exit.
    `clock` supplies `created_at`/`updated_at` and the default `run_after`.
    """

    def __init__(
        self, path: str | os.PathLike[str], *, clock: Callable[[], datetime] = utc_now
    ) -> None:
        self.path = Path(path)
        self._clock = clock
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        # Serialises statements on the one connection: a statement must be fully
        # read before the next one runs, or a write (e.g. `complete`) can change
        # the table under another coroutine's half-read SELECT (`counts` then
        # returned rows with a NULL status).
        self._statement_lock = asyncio.Lock()

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

    async def _run(self, sql: str, params: Sequence[Any] = ()) -> tuple[list[Any], int]:
        """Run one statement to completion: (all result rows, rowcount)."""
        conn = await self._connection()
        async with self._statement_lock:
            async with conn.execute(sql, params) as cursor:
                rows = await cursor.fetchall()  # drain, so a RETURNING statement completes
                return list(rows), cursor.rowcount

    async def _fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        rows, _ = await self._run(sql, params)
        return rows[0] if rows else None

    async def _transition(self, job_id: int, sql: str, params: tuple[Any, ...]) -> None:
        """Run an UPDATE of the RUNNING job `job_id`; `sql` ends in `WHERE id = ?`."""
        _, changed = await self._run(f"{sql} AND status = 'running'", (*params, job_id))
        if changed == 0:
            raise JobStateError(f"job {job_id} does not exist or is not running")

    async def enqueue(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        dedupe_key: str | None = None,
        run_after: datetime | None = None,
        max_attempts: int = 5,
    ) -> Job:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        now = _ts(self._clock())
        values = (
            kind,
            json.dumps(payload),
            max_attempts,
            now if run_after is None else _ts(run_after),
            now,
            now,
            dedupe_key,
        )
        insert = (
            "INSERT INTO jobs (kind, payload, status, attempts, max_attempts, run_after,"
            " created_at, updated_at, dedupe_key)"
            " SELECT ?, ?, 'pending', 0, ?, ?, ?, ?, ?"
        )
        if dedupe_key is None:
            return _job(await self._fetchone(f"{insert} RETURNING {_COLUMNS}", values))
        while True:
            try:
                row = await self._fetchone(
                    f"{insert} WHERE NOT EXISTS (SELECT 1 FROM jobs WHERE dedupe_key = ?"
                    f" AND status IN {_ACTIVE}) RETURNING {_COLUMNS}",
                    (*values, dedupe_key),
                )
            except sqlite3.IntegrityError:
                row = None  # lost a race to another connection; fetch its job below
            if row is not None:
                return _job(row)
            existing = await self._fetchone(
                f"SELECT {_COLUMNS} FROM jobs WHERE dedupe_key = ? AND status IN {_ACTIVE}",
                (dedupe_key,),
            )
            if existing is not None:
                return _job(existing)
            # The blocking job finished in between: try inserting again.

    async def claim(self, now: datetime) -> Job | None:
        # One statement: the write lock is taken before the subquery reads, so
        # concurrent claimers on other connections serialise and never share a job.
        row = await self._fetchone(
            "UPDATE jobs SET status = 'running', attempts = attempts + 1, updated_at = ?"
            " WHERE id = (SELECT id FROM jobs WHERE status = 'pending' AND run_after <= ?"
            " ORDER BY run_after, id LIMIT 1)"
            f" RETURNING {_COLUMNS}",
            (_ts(self._clock()), _ts(now)),
        )
        return None if row is None else _job(row)

    async def complete(self, job_id: int) -> None:
        await self._transition(
            job_id,
            "UPDATE jobs SET status = 'done', updated_at = ? WHERE id = ?",
            (_ts(self._clock()),),
        )

    async def fail(self, job_id: int, error: str, retry_at: datetime | None) -> None:
        if retry_at is None:
            await self._transition(
                job_id,
                "UPDATE jobs SET status = 'failed', last_error = ?, updated_at = ? WHERE id = ?",
                (error, _ts(self._clock())),
            )
        else:
            await self._transition(
                job_id,
                "UPDATE jobs SET status = 'pending', last_error = ?, run_after = ?,"
                " updated_at = ? WHERE id = ?",
                (error, _ts(retry_at), _ts(self._clock())),
            )

    async def defer(self, job_id: int, run_after: datetime) -> None:
        await self._transition(
            job_id,
            "UPDATE jobs SET status = 'pending', attempts = max(attempts - 1, 0), run_after = ?,"
            " updated_at = ? WHERE id = ?",
            (_ts(run_after), _ts(self._clock())),
        )

    async def recover(self) -> int:
        _, changed = await self._run(
            "UPDATE jobs SET status = 'pending', attempts = max(attempts - 1, 0), updated_at = ?"
            " WHERE status = 'running'",
            (_ts(self._clock()),),
        )
        return changed

    async def next_due(self) -> datetime | None:
        row = await self._fetchone("SELECT min(run_after) FROM jobs WHERE status = 'pending'")
        return None if row is None or row[0] is None else datetime.fromisoformat(row[0])

    async def counts(self) -> dict[JobStatus, int]:
        rows, _ = await self._run("SELECT status, count(*) FROM jobs GROUP BY status")
        result = dict.fromkeys(JobStatus, 0)
        result.update({JobStatus(status): count for status, count in rows})
        return result

    async def postpone(self, job_id: int, run_after: datetime) -> bool:
        """Move a PENDING job's `run_after` later to `run_after` (never earlier).

        Returns whether it changed (not pending, missing, or already due later:
        False). Not part of the port: used to debounce the profile rewrite.
        """
        _, changed = await self._run(
            "UPDATE jobs SET run_after = ?, updated_at = ?"
            " WHERE id = ? AND status = 'pending' AND run_after < ?",
            (_ts(run_after), _ts(self._clock()), job_id, _ts(run_after)),
        )
        return changed > 0

    async def requeue_unhandled(self, kinds: Collection[str]) -> list[Job]:
        """Put back jobs of `kinds` that failed for good only because no handler was
        registered for their kind (the runner's "no handler registered for job kind"
        error), now that one is.

        Per kind, the newest such job becomes PENDING again (attempts reset, due
        now, error cleared) unless a job with its dedupe key is already pending or
        running; the other such jobs are deleted (they were the same missed run).
        Returns the requeued jobs. Not part of the port.
        """
        requeued: list[Job] = []
        for kind in sorted(set(kinds)):
            error = f"no handler registered for job kind {kind!r}"
            rows, _ = await self._run(
                f"SELECT {_COLUMNS} FROM jobs WHERE kind = ? AND status = 'failed'"
                " AND last_error = ? ORDER BY id DESC",
                (kind, error),
            )
            if not rows:
                continue
            newest, *older = [_job(row) for row in rows]
            now = _ts(self._clock())
            revived, _ = await self._run(
                "UPDATE jobs SET status = 'pending', attempts = 0, last_error = NULL,"
                " run_after = ?, updated_at = ? WHERE id = ? AND status = 'failed'"
                " AND NOT EXISTS (SELECT 1 FROM jobs AS j WHERE j.dedupe_key = jobs.dedupe_key"
                f" AND j.status IN {_ACTIVE})"
                f" RETURNING {_COLUMNS}",
                (now, now, newest.id),
            )
            requeued.extend(_job(row) for row in revived)
            stale = [job.id for job in older] + ([] if revived else [newest.id])
            await self._run(
                "DELETE FROM jobs WHERE status = 'failed' AND id IN"
                " (SELECT value FROM json_each(?))",
                (json.dumps(stale),),
            )
        return requeued

    async def get(self, job_id: int) -> Job | None:
        """The job with `job_id`, or None (for inspection; not part of the port)."""
        row = await self._fetchone(f"SELECT {_COLUMNS} FROM jobs WHERE id = ?", (job_id,))
        return None if row is None else _job(row)

    async def list_jobs(
        self,
        *,
        statuses: Collection[JobStatus] | None = None,
        with_error: bool = False,
        limit: int | None = None,
    ) -> list[Job]:
        """Jobs, most recently updated first (for inspection; not part of the port).

        Optionally only those in `statuses`, and only those with a `last_error`.
        """
        where, params = ["1"], []
        if statuses is not None:
            where.append("status IN (SELECT value FROM json_each(?))")
            params.append(json.dumps(sorted(JobStatus(s).value for s in statuses)))
        if with_error:
            where.append("last_error IS NOT NULL")
        params.append(-1 if limit is None else limit)
        rows, _ = await self._run(
            f"SELECT {_COLUMNS} FROM jobs WHERE {' AND '.join(where)}"
            " ORDER BY updated_at DESC, id DESC LIMIT ?",
            params,
        )
        return [_job(row) for row in rows]
