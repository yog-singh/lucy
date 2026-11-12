import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import aiosqlite
import pytest

from lucy.core import registry
from lucy.core.context import SectionReport
from lucy.core.events import (
    AssistantMessage,
    Interrupt,
    TimerFired,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    UserMessage,
)
from lucy.core.ports import MemoryStore, TraceSink
from lucy.core.trace import StepTrace, TurnTrace
from lucy.core.types import FinishReason, MemoryContext, Message, Role, ToolCall, ToolResult
from lucy.memory import migrations
from lucy.memory.sqlite_store import MemoryStoreError, SQLiteMemoryStore, apply_migrations


@pytest.fixture
async def store(tmp_path: Path):
    async with SQLiteMemoryStore(tmp_path / "lucy.db") as s:
        yield s


def _version(path: Path) -> int:
    with sqlite3.connect(path) as conn:
        return conn.execute("PRAGMA user_version").fetchone()[0]


#: Every table of the current schema, including the FTS5 shadow tables.
LATEST_TABLES = {
    "episodes",
    "turn_traces",
    "jobs",
    "entities",
    "statements",
    "statement_sources",
    "profile_versions",
    "vector_spaces",
    "memory_meta",
    *(
        f"{t}_fts{s}"
        for t in ("statements", "entities")
        for s in ("", "_config", "_data", "_docsize", "_idx")
    ),
}


def _tables(path: Path) -> set[str]:
    with sqlite3.connect(path) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}


def test_satisfies_port_and_is_registered() -> None:
    assert isinstance(SQLiteMemoryStore("unused.db"), MemoryStore)
    assert registry.get("memory", "sqlite") is SQLiteMemoryStore


def test_reimport_does_not_fail_registration() -> None:
    registry.load_adapters(["lucy.memory.sqlite_store", "lucy.memory.sqlite_store"])
    # Registering the same class object again is a no-op.
    registry.register("memory", "sqlite")(SQLiteMemoryStore)
    assert registry.get("memory", "sqlite") is SQLiteMemoryStore
    # A different class under the same name is still an error.
    with pytest.raises(registry.RegistryError, match="already registered"):
        registry.register("memory", "sqlite")(type("Other", (), {}))


async def test_registry_create(tmp_path: Path) -> None:
    store = registry.create("memory", "sqlite", path=str(tmp_path / "x.db"))
    assert isinstance(store, SQLiteMemoryStore)
    await store.close()


async def test_round_trip_preserves_order_and_equality(store: SQLiteMemoryStore) -> None:
    call = ToolCall(id="c1", name="echo", arguments={"text": "hi", "n": [1, 2]})
    events = [
        UserMessage(session_id="s", turn_id="t1", text="hello", source="voice"),
        ToolStarted(session_id="s", turn_id="t1", call=call),
        ToolFinished(session_id="s", turn_id="t1", result=ToolResult(call_id="c1", content="hi")),
        AssistantMessage(session_id="s", turn_id="t1", text="hi", tool_calls=[call]),
        TurnEnded(session_id="s", turn_id="t1", reason="completed"),
        TimerFired(session_id="s", name="tea", payload={"minutes": 4}),
        Interrupt(session_id="s"),
    ]
    for event in events:
        await store.append_episode(event)
    assert await store.recent_episodes("s", 100) == events


async def test_sessions_are_isolated(store: SQLiteMemoryStore) -> None:
    a = UserMessage(session_id="a", text="one")
    b = UserMessage(session_id="b", text="two")
    await store.append_episode(a)
    await store.append_episode(b)
    assert await store.recent_episodes("a", 10) == [a]
    assert await store.recent_episodes("b", 10) == [b]
    assert await store.recent_episodes("c", 10) == []


async def test_limit_returns_most_recent_oldest_first(store: SQLiteMemoryStore) -> None:
    events = [UserMessage(session_id="s", text=str(i)) for i in range(5)]
    for event in events:
        await store.append_episode(event)
    assert await store.recent_episodes("s", 2) == events[3:]
    assert await store.recent_episodes("s", 0) == []
    assert await store.recent_episodes("s", -1) == []


async def test_duplicate_event_id_raises(store: SQLiteMemoryStore) -> None:
    event = UserMessage(session_id="s", text="once")
    await store.append_episode(event)
    with pytest.raises(MemoryStoreError, match="already"):
        await store.append_episode(event)
    assert await store.recent_episodes("s", 10) == [event]


async def test_columns_are_filled(tmp_path: Path) -> None:
    path = tmp_path / "lucy.db"
    event = UserMessage(session_id="s", turn_id="t", text="hi")
    async with SQLiteMemoryStore(path) as store:
        await store.append_episode(event)
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT event_id, ts, session_id, turn_id, type, schema_version, extracted_at"
            " FROM episodes"
        ).fetchone()
    assert row == (event.id, event.ts.isoformat(), "s", "t", "user_message", 1, None)
    assert row[1].endswith("+00:00")


async def test_persists_across_close_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "lucy.db"
    event = UserMessage(session_id="s", text="remember me")
    store = SQLiteMemoryStore(path)
    await store.append_episode(event)  # connects lazily, creating the directories
    await store.close()
    await store.close()  # idempotent
    async with SQLiteMemoryStore(path) as reopened:
        assert await reopened.recent_episodes("s", 1) == [event]
    # The closed store reconnects on use.
    assert await store.recent_episodes("s", 1) == [event]
    await store.close()


async def test_wal_and_foreign_keys(store: SQLiteMemoryStore) -> None:
    conn = await store._connection()
    async with conn.execute("PRAGMA journal_mode") as cur:
        assert (await cur.fetchone())[0] == "wal"
    async with conn.execute("PRAGMA foreign_keys") as cur:
        assert (await cur.fetchone())[0] == 1


async def test_retrieve_is_empty_for_now(store: SQLiteMemoryStore) -> None:
    assert await store.retrieve("anything", 1000) == MemoryContext()


async def test_concurrent_appends_all_land(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "lucy.db")  # not yet connected: races the lazy open too
    events = [UserMessage(session_id="s", text=str(i)) for i in range(200)]
    await asyncio.gather(*(store.append_episode(e) for e in events))
    stored = await store.recent_episodes("s", 1000)
    await store.close()
    assert len(stored) == 200
    assert {e.id for e in stored} == {e.id for e in events}


# --- Migrations ---------------------------------------------------------------


async def test_fresh_database_gets_latest_schema(tmp_path: Path) -> None:
    path = tmp_path / "lucy.db"
    async with SQLiteMemoryStore(path):
        pass
    assert _version(path) == len(migrations.MIGRATIONS) == 5
    assert _tables(path) == LATEST_TABLES


async def test_reopen_is_a_noop(tmp_path: Path) -> None:
    path = tmp_path / "lucy.db"
    event = UserMessage(session_id="s", text="kept")
    async with SQLiteMemoryStore(path) as store:
        await store.append_episode(event)
    async with SQLiteMemoryStore(path) as store:
        assert await store.recent_episodes("s", 10) == [event]
    assert _version(path) == 5


async def test_newer_database_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "lucy.db"
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version = 99")
    store = SQLiteMemoryStore(path)
    with pytest.raises(MemoryStoreError, match="newer"):
        await store.append_episode(UserMessage(session_id="s", text="x"))
    assert store._conn is None
    assert _version(path) == 99


async def test_migrations_apply_in_order_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    steps = ["CREATE TABLE a (x INTEGER);", "CREATE TABLE b (y INTEGER); INSERT INTO a VALUES (1);"]
    async with aiosqlite.connect(path, autocommit=True) as conn:
        assert await apply_migrations(conn, steps[:1]) == 1
        assert await apply_migrations(conn, steps) == 2  # only step 2 runs
        assert await apply_migrations(conn, steps) == 2  # no-op
    assert _tables(path) == {"a", "b"}
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM a").fetchone()[0] == 1


async def test_failed_migration_rolls_back(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    steps = ["CREATE TABLE a (x INTEGER);", "CREATE TABLE b (y INTEGER); THIS IS NOT SQL;"]
    async with aiosqlite.connect(path, autocommit=True) as conn:
        with pytest.raises(MemoryStoreError, match="migration 2"):
            await apply_migrations(conn, steps)
        assert not conn.in_transaction
    assert _version(path) == 1
    assert _tables(path) == {"a"}


async def test_database_at_version_1_upgrades_with_episodes_intact(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    events = [
        UserMessage(session_id="s", turn_id="t", text="from the old schema"),
        AssistantMessage(session_id="s", turn_id="t", text="still here"),
    ]
    async with aiosqlite.connect(path, autocommit=True) as conn:
        assert await apply_migrations(conn, migrations.MIGRATIONS[:1]) == 1
    old = SQLiteMemoryStore(path)
    # Write through the store's own SQL against the version-1 schema.
    old._conn = await aiosqlite.connect(path, autocommit=True)
    for event in events:
        await old.append_episode(event)
    await old.close()
    assert _version(path) == 1 and _tables(path) == {"episodes"}

    async with SQLiteMemoryStore(path) as store:
        assert await store.recent_episodes("s", 10) == events
        await store.record(_trace("t"))
        assert await store.get_trace("t") == _trace("t")
    assert _version(path) == 5
    assert _tables(path) == LATEST_TABLES


# --- Traces -------------------------------------------------------------------

_T0 = datetime(2026, 11, 11, 9, 30, tzinfo=UTC)


def _trace(turn_id: str, session_id: str = "s", outcome: str = "completed") -> TurnTrace:
    step = StepTrace(
        role="chat",
        messages=[
            Message(role=Role.SYSTEM, content="persona"),
            Message(role=Role.USER, content="hi"),
        ],
        tools=["echo"],
        started_at=_T0,
        first_token_at=_T0 + timedelta(milliseconds=350),
        ended_at=_T0 + timedelta(seconds=1),
        finish_reason=FinishReason.STOP,
    )
    return TurnTrace(
        turn_id=turn_id,
        session_id=session_id,
        started_at=_T0,
        ended_at=_T0 + timedelta(seconds=1),
        steps=[step],
        context=[SectionReport(name="persona", tokens=10, included=1, dropped=0)],
        outcome=outcome,  # type: ignore[arg-type]
        error="boom" if outcome == "error" else None,
    )


def test_satisfies_trace_sink_port() -> None:
    assert isinstance(SQLiteMemoryStore("unused.db"), TraceSink)


async def test_trace_round_trip(store: SQLiteMemoryStore) -> None:
    trace = _trace("t1", outcome="error")
    await store.record(trace)
    assert await store.get_trace("t1") == trace
    assert await store.get_trace("missing") is None


async def test_trace_columns_are_filled(tmp_path: Path) -> None:
    path = tmp_path / "lucy.db"
    # A non-UTC timestamp is stored in UTC.
    trace = _trace("t1").model_copy(
        update={"started_at": _T0.astimezone(timezone(timedelta(hours=5)))}
    )
    async with SQLiteMemoryStore(path) as store:
        await store.record(trace)
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT turn_id, session_id, started_at, ended_at, outcome FROM turn_traces"
        ).fetchone()
        indexes = {r[1] for r in conn.execute("PRAGMA index_list(turn_traces)").fetchall()}
    assert row == (
        "t1",
        "s",
        _T0.isoformat(),
        (_T0 + timedelta(seconds=1)).isoformat(),
        "completed",
    )
    assert "turn_traces_session" in indexes


async def test_duplicate_trace_raises(store: SQLiteMemoryStore) -> None:
    await store.record(_trace("t1"))
    with pytest.raises(MemoryStoreError, match="already stored"):
        await store.record(_trace("t1"))
