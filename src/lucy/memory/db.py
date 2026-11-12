"""Opening the Lucy database: one SQLite file shared by the memory and job stores.

Every store opens its own connection through `open_database`, which sets the
pragmas the stores rely on and brings the schema up to date, so several
connections (or processes) can use the same file safely.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import aiosqlite

from lucy.memory.migrations import MIGRATIONS


class MemoryStoreError(RuntimeError):
    """The memory store cannot do what was asked (duplicate event, bad schema, ...)."""


async def _user_version(conn: aiosqlite.Connection) -> int:
    async with conn.execute("PRAGMA user_version") as cursor:
        row = await cursor.fetchone()
    return int(row[0]) if row else 0


async def apply_migrations(
    conn: aiosqlite.Connection, migrations: Sequence[str] = MIGRATIONS
) -> int:
    """Bring the schema of `conn` up to `len(migrations)` and return the resulting version.

    Each migration runs in its own `BEGIN IMMEDIATE` transaction together with the
    `user_version` bump, so a failed migration leaves the previous version intact
    and concurrent openers cannot apply the same step twice. `conn` must be in
    autocommit mode (`autocommit=True`). Raises `MemoryStoreError` if the database
    is newer than `migrations` or a migration fails.
    """
    known = len(migrations)
    while True:
        await conn.execute("BEGIN IMMEDIATE")
        try:
            version = await _user_version(conn)
            if version > known:
                raise MemoryStoreError(
                    f"database schema version {version} is newer than this version of Lucy "
                    f"supports ({known}); upgrade Lucy or restore an older backup"
                )
            if version == known:
                await conn.execute("COMMIT")
                return version
            try:
                await conn.executescript(migrations[version])
                await conn.execute(f"PRAGMA user_version = {version + 1}")
            except sqlite3.Error as exc:
                raise MemoryStoreError(f"schema migration {version + 1} failed: {exc}") from exc
            await conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                await conn.execute("ROLLBACK")
            raise


async def open_database(path: str | os.PathLike[str]) -> aiosqlite.Connection:
    """Open the database at `path` in autocommit mode, ready for use.

    Creates the parent directories, enables WAL, foreign keys and a 5 s busy
    timeout, and applies pending migrations. Raises `MemoryStoreError` if the
    file cannot be opened or migrated; no connection is left open then.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(path, autocommit=True)
    except (OSError, sqlite3.Error) as exc:
        raise MemoryStoreError(f"cannot open memory database {path}: {exc}") from exc
    try:
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.execute("PRAGMA busy_timeout = 5000")
        await apply_migrations(conn)
    except BaseException as exc:
        await conn.close()
        if isinstance(exc, sqlite3.Error):
            raise MemoryStoreError(f"cannot open memory database {path}: {exc}") from exc
        raise
    return conn
