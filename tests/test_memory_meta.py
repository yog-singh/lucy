"""Migration 5 (`memory_meta`) and the graph-store methods added for maintenance."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite
import pytest

from lucy.core.events import UserMessage
from lucy.memory import migrations
from lucy.memory.db import apply_migrations
from lucy.memory.graph_store import DERIVED_META_PREFIX, SQLiteGraphStore
from lucy.memory.sqlite_jobs import SQLiteJobStore
from lucy.memory.sqlite_store import SQLiteMemoryStore
from lucy.memory.vector_index import load_vec
from tests.graph_support import T0, Clock, HashingEmbedder, log, new, query


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "lucy.db"


@pytest.fixture
async def graph(path: Path, clock: Clock) -> AsyncIterator[SQLiteGraphStore]:
    async with SQLiteGraphStore(path, HashingEmbedder(), clock=clock) as g:
        yield g


async def test_meta_get_set_delete(graph: SQLiteGraphStore, path: Path, clock: Clock) -> None:
    assert await graph.get_meta("a") is None
    await graph.set_meta("a", "1")
    assert await graph.get_meta("a") == "1"
    clock.advance(seconds=5)
    await graph.set_meta("a", "2")
    assert await graph.get_meta("a") == "2"
    [(updated,)] = query(path, "SELECT updated_at FROM memory_meta WHERE key = 'a'")
    assert updated.startswith("2026-11-12T09:00:05")
    assert await graph.delete_meta("a") is True
    assert await graph.delete_meta("a") is False
    assert await graph.get_meta("a") is None


async def test_meta_prefix_listing_and_paging(graph: SQLiteGraphStore) -> None:
    await graph.write_meta({"x.a": "1", "x.b": "2", "x.c": "3", "xa": "no", "w": "no", "y": "no"})
    assert await graph.list_meta("x.") == [("x.a", "1"), ("x.b", "2"), ("x.c", "3")]
    assert await graph.list_meta("x.", limit=2) == [("x.a", "1"), ("x.b", "2")]
    assert await graph.list_meta("x.", after="x.b") == [("x.c", "3")]
    assert len(await graph.list_meta()) == 6
    assert await graph.get_meta_many(["x.a", "x.c", "missing"]) == {"x.a": "1", "x.c": "3"}
    await graph.write_meta({"x.a": None, "x.b": "20"})
    assert await graph.list_meta("x.") == [("x.b", "20"), ("x.c", "3")]


async def test_set_confidences_and_meta_is_atomic(graph: SQLiteGraphStore) -> None:
    s = await graph.add_statement(new("User likes tea", confidence=0.8))
    assert await graph.set_confidences_and_meta({s.id: 0.5, 999: 0.1}, {"k": "v"}) == 1
    assert (await graph.get_statement(s.id)).confidence == 0.5  # type: ignore[union-attr]
    assert await graph.get_meta("k") == "v"
    with pytest.raises(ValueError):
        await graph.set_confidences_and_meta({s.id: 1.5}, {"k": "changed"})
    assert await graph.get_meta("k") == "v"
    assert (await graph.get_statement(s.id)).confidence == 0.5  # type: ignore[union-attr]


async def test_reset_graph_clears_only_derived_meta(graph: SQLiteGraphStore) -> None:
    await graph.write_meta(
        {f"{DERIVED_META_PREFIX}decay.s.1": "{}", f"{DERIVED_META_PREFIX}profile": "h"}
    )
    await graph.set_meta("maintenance.enqueued.memory.decay", T0.isoformat())
    await graph.set_meta("derivedX", "kept: not under the prefix")
    await graph.reset_graph()
    assert await graph.list_meta(DERIVED_META_PREFIX) == []
    assert await graph.get_meta("maintenance.enqueued.memory.decay") == T0.isoformat()
    assert await graph.get_meta("derivedX") is not None


async def test_entity_statement_fingerprints(graph: SQLiteGraphStore, clock: Clock) -> None:
    priya = await graph.create_entity("person", "Priya")
    lonely = await graph.create_entity("person", "Nobody")
    a = await graph.add_statement(new("User's sister is Priya", object_id=priya.id))
    before = await graph.entity_statement_fingerprints([1, priya.id, lonely.id, 999])
    assert set(before) == {1, priya.id, lonely.id}
    assert before[lonely.id][0] == 0
    assert before[priya.id][0] == 1 and before[1][0] == 1
    # Confidence changes do not change the fingerprint.
    await graph.set_confidence(a.id, 0.3)
    assert await graph.entity_statement_fingerprints([priya.id]) == {priya.id: before[priya.id]}
    # A new statement does.
    b = await graph.add_statement(new("Priya lives in Pune", subject_id=priya.id))
    after = await graph.entity_statement_fingerprints([priya.id])
    assert after[priya.id][0] == 2 and after[priya.id] != before[priya.id]
    # Invalidation counts once its time has passed.
    await graph.invalidate(b.id, at=T0.replace(year=2027))
    assert (await graph.entity_statement_fingerprints([priya.id]))[priya.id] == after[priya.id]
    clock.advance(days=400)
    assert (await graph.entity_statement_fingerprints([priya.id]))[priya.id] == before[priya.id]
    # A statement with the entity as both subject and object counts once.
    await graph.add_statement(
        new("Priya talks to herself", subject_id=priya.id, object_id=priya.id)
    )
    assert (await graph.entity_statement_fingerprints([priya.id]))[priya.id][0] == 2


async def _v4_connection(path: Path) -> aiosqlite.Connection:
    """A connection to `path` that does not migrate (it stays at version 4)."""
    conn = await aiosqlite.connect(path, autocommit=True)
    await conn.execute("PRAGMA foreign_keys = ON")
    await load_vec(conn)
    return conn


async def test_migration_4_to_5_keeps_everything(path: Path, clock: Clock) -> None:
    async with aiosqlite.connect(path, autocommit=True) as conn:
        assert await apply_migrations(conn, migrations.MIGRATIONS[:4]) == 4
    # Write every kind of data through the stores' own SQL against the version-4 schema.
    memory = SQLiteMemoryStore(path)
    memory._conn = await _v4_connection(path)
    episode = await log(memory, UserMessage(session_id="s", turn_id="t", text="hi"))
    await memory.close()
    graph = SQLiteGraphStore(path, HashingEmbedder(), clock=clock)
    graph._conn = await _v4_connection(path)
    priya = await graph.create_entity("person", "Priya", aliases=["my sister"])
    statement = await graph.add_statement(
        new("User's sister is Priya", object_id=priya.id), [episode]
    )
    await graph.save_profile("A profile.")
    await graph.close()
    jobs = SQLiteJobStore(path)
    jobs._conn = await _v4_connection(path)
    await jobs.enqueue("extract", {"n": 1})
    await jobs.close()
    assert query(path, "PRAGMA user_version")[0][0] == 4

    async with SQLiteGraphStore(path, HashingEmbedder(), clock=clock) as upgraded:
        assert query(path, "PRAGMA user_version")[0][0] == 5
        assert await upgraded.get_statement(statement.id) == statement
        assert await upgraded.get_entity(priya.id) == priya
        assert (await upgraded.find_entities("my sister"))[0].id == priya.id
        assert (await upgraded.current_profile()).content == "A profile."  # type: ignore[union-attr]
        hits = await upgraded.similar_statements("sister Priya")
        assert [h.statement.id for h in hits] == [statement.id]
        assert (await upgraded.get_episodes([episode]))[0].event.text == "hi"  # type: ignore[union-attr]
        await upgraded.set_meta("k", "v")
        assert await upgraded.get_meta("k") == "v"
    async with SQLiteJobStore(path) as reopened:
        job = await reopened.get(1)
        assert job is not None and job.payload == {"n": 1}
