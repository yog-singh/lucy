"""Cancelling a graph-store call while it waits for its transaction to begin.

aiosqlite runs a statement to completion in its worker thread even when the
awaiting coroutine is cancelled. A cancellation while `BEGIN` waits (here, for
another connection's write lock) must not leave the store's connection inside
a transaction: that held the database write lock until the store was closed
(other connections then timed out with "database is locked", which made
`LucyApp.stop` slow and lost the deferral of an interrupted job), and made the
store's next call fail.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from lucy.memory.graph_store import SQLiteGraphStore
from tests.graph_support import HashingEmbedder


async def test_cancelled_while_beginning_leaves_no_transaction(tmp_path: Path) -> None:
    path = tmp_path / "lucy.db"
    async with SQLiteGraphStore(path, HashingEmbedder()) as graph:
        await graph.stats()  # connected and migrated
        blocker = sqlite3.connect(path, isolation_level=None, timeout=5)
        try:
            blocker.execute("BEGIN EXCLUSIVE")  # makes the store's BEGIN wait
            task = asyncio.create_task(graph.set_meta("k", "v"))
            await asyncio.sleep(0.1)  # let the BEGIN reach the worker thread and block
            task.cancel()
            blocker.execute("ROLLBACK")  # the BEGIN now goes through in the thread
            with pytest.raises(asyncio.CancelledError):
                async with asyncio.timeout(5):
                    await task
            conn = await graph._connection()
            assert not conn.in_transaction
            # Another connection can write at once, and the store still works.
            blocker.execute("PRAGMA busy_timeout = 100")
            blocker.execute("BEGIN IMMEDIATE")
            blocker.execute("ROLLBACK")
        finally:
            blocker.close()
        async with asyncio.timeout(5):
            await graph.set_meta("k2", "v2")
            assert await graph.get_meta("k2") == "v2"
