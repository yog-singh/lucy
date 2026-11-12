"""The knowledge-graph store: schema, entities, statements, episodes, profile, FTS sync."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import aiosqlite
import pydantic
import pytest

from lucy.core.events import (
    AssistantMessage,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    UserMessage,
)
from lucy.core.types import ToolCall, ToolResult
from lucy.memory import migrations
from lucy.memory.db import apply_migrations
from lucy.memory.graph_store import GraphNotFoundError, GraphStoreError, SQLiteGraphStore
from lucy.memory.models import (
    USER_ENTITY_ID,
    NewStatement,
    Origin,
    StatementKind,
)
from lucy.memory.sqlite_jobs import SQLiteJobStore
from lucy.memory.sqlite_store import SQLiteMemoryStore
from tests.graph_support import (
    T0,
    Clock,
    HashingEmbedder,
    fts_consistent,
    log,
    new,
    query,
    vector_ids,
)
from tests.test_sqlite_store import _trace


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "lucy.db"


@pytest.fixture
async def graph(
    path: Path, embedder: HashingEmbedder, clock: Clock, make_graph_store
) -> AsyncIterator[SQLiteGraphStore]:
    async with make_graph_store(path, embedder, clock=clock) as g:
        yield g


@pytest.fixture
async def memory(path: Path) -> AsyncIterator[SQLiteMemoryStore]:
    async with SQLiteMemoryStore(path) as m:
        yield m


def count(path: Path, table: str) -> int:
    return query(path, f"SELECT count(*) FROM {table}")[0][0]


# --- Schema and migration -----------------------------------------------------


async def test_fresh_database_has_graph_schema_and_user(graph: SQLiteGraphStore, path: Path):
    assert query(path, "PRAGMA user_version")[0][0] == len(migrations.MIGRATIONS) == 5
    user = await graph.user_entity()
    assert (user.id, user.type, user.name, user.aliases, user.summary) == (
        USER_ENTITY_ID,
        "person",
        "User",
        [],
        None,
    )
    assert user.created_at.tzinfo is not None
    fts_consistent(path)


async def test_migration_3_to_4_preserves_episodes_traces_and_jobs(path: Path, clock: Clock):
    async with aiosqlite.connect(path, autocommit=True) as conn:
        assert await apply_migrations(conn, migrations.MIGRATIONS[:3]) == 3
        await conn.execute(
            "INSERT INTO jobs (kind, payload, status, max_attempts, run_after, created_at,"
            " updated_at) VALUES ('extract', '{\"n\": 1}', 'pending', 3, ?, ?, ?)",
            ((T0.isoformat(timespec="microseconds"),) * 3),
        )
    events = [
        UserMessage(session_id="s", turn_id="t", text="my sister lives in Pune"),
        AssistantMessage(session_id="s", turn_id="t", text="Noted."),
    ]
    old = SQLiteMemoryStore(path)
    old._conn = await aiosqlite.connect(path, autocommit=True)  # stay on version 3
    for event in events:
        await old.append_episode(event)
    await old.record(_trace("t"))
    await old.close()
    assert query(path, "PRAGMA user_version")[0][0] == 3

    async with SQLiteGraphStore(path, HashingEmbedder(), clock=clock) as graph:
        assert (await graph.user_entity()).name == "User"
        unextracted = await graph.unextracted_episodes()
        assert [e.event for e in unextracted] == events
    assert query(path, "PRAGMA user_version")[0][0] == 5
    async with SQLiteMemoryStore(path) as store:
        assert await store.recent_episodes("s", 10) == events
        assert await store.get_trace("t") == _trace("t")
    async with SQLiteJobStore(path) as jobs:
        job = await jobs.get(1)
        assert job is not None and job.kind == "extract" and job.payload == {"n": 1}


async def test_stores_without_sqlite_vec_still_work_on_the_file(
    graph: SQLiteGraphStore, path: Path
):
    await graph.add_statement(new("User likes tea"))
    assert await vector_ids(graph, "statements")  # vec0 tables now exist in the file
    async with SQLiteMemoryStore(path) as store, SQLiteJobStore(path) as jobs:
        event = UserMessage(session_id="s", text="hi")
        await store.append_episode(event)
        assert await store.recent_episodes("s", 5) == [event]
        await jobs.enqueue("k", {})
    # A plain sqlite3 connection can read and delete base rows too.
    assert query(path, "SELECT text FROM statements") == [("User likes tea",)]


async def test_schema_constraints(path: Path, graph: SQLiteGraphStore):
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        base = (
            "INSERT INTO statements (subject_id, predicate, object_id, object_value, text, kind,"
            " origin, confidence, recorded_at, last_observed_at, valid_from, invalid_at)"
            " VALUES (1, 'p', ?, ?, 't', ?, ?, ?, 'x', 'x', ?, ?)"
        )
        good = (None, None, "fact", "stated", 0.5, None, None)
        conn.execute(base, good)
        bad = [
            (1, "v", "fact", "stated", 0.5, None, None),  # both objects
            (None, None, "rumour", "stated", 0.5, None, None),
            (None, None, "fact", "guessed", 0.5, None, None),
            (None, None, "fact", "stated", 1.5, None, None),
            (None, None, "fact", "stated", -0.1, None, None),
            (None, None, "fact", "stated", 0.5, "2026-02", "2026-01"),
            (99, None, "fact", "stated", 0.5, None, None),  # unknown object entity
        ]
        for params in bad:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(base, params)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE statements SET superseded_by = id")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE entities SET aliases = '{}'")


# --- Entities -----------------------------------------------------------------


async def test_create_and_get_entity(graph: SQLiteGraphStore, clock: Clock):
    priya = await graph.create_entity(
        " person ", " Priya ", aliases=["my sister", "PRIYA", " ", "My Sister"], summary="Sister"
    )
    assert (priya.type, priya.name, priya.aliases, priya.summary) == (
        "person",
        "Priya",
        ["my sister"],
        "Sister",
    )
    assert priya.created_at == clock.now
    assert await graph.get_entity(priya.id) == priya
    assert await graph.get_entity(9999) is None
    assert await graph.get_entities([priya.id, 9999, USER_ENTITY_ID]) == [
        await graph.user_entity(),
        priya,
    ]
    with pytest.raises(ValueError):
        await graph.create_entity("person", "  ")


async def test_find_entities_by_name_or_alias_case_insensitively(graph: SQLiteGraphStore):
    priya = await graph.create_entity("person", "Priya", aliases=["my sister"])
    other = await graph.create_entity("person", "Priya Sharma", aliases=["Priya"])
    assert [e.id for e in await graph.find_entities("priya")] == [priya.id, other.id]
    assert await graph.find_entities("MY  SISTER") == [priya]
    assert await graph.find_entities("sister") == []
    assert await graph.find_entities("Pri") == []
    for hostile in ["", "   ", '"', "*", "NEAR(a b)", "a OR b", "-", "😀", "name:x"]:
        assert await graph.find_entities(hostile) == []
    emoji = await graph.create_entity("thing", "😀")
    assert await graph.find_entities("😀") == [emoji]


async def test_entities_mentioned_in_text(graph: SQLiteGraphStore):
    priya = await graph.create_entity("person", "Priya", aliases=["my sister"])
    pune = await graph.create_entity("place", "Pune")
    await graph.create_entity("person", "Priyanka")
    found = await graph.entities_mentioned_in("How is my sister doing in Pune?")
    assert [e.id for e in found] == [priya.id, pune.id]  # longest matched name first
    assert await graph.entities_mentioned_in("Priyas and Punekars") == []
    assert await graph.entities_mentioned_in("") == []
    assert await graph.entities_mentioned_in('"*) OR NOT (') == []


async def test_similar_entities_uses_name_and_summary(graph: SQLiteGraphStore):
    priya = await graph.create_entity("person", "Priya", summary="sister, an architect in Pune")
    await graph.create_entity("project", "Lucy", summary="voice assistant software")
    hits = await graph.similar_entities("architect")
    assert hits[0].entity == priya and hits[0].score > 0
    assert hits[0].score > hits[1].score
    assert await graph.similar_entities("architect", k=0) == []


async def test_add_alias(graph: SQLiteGraphStore, clock: Clock):
    priya = await graph.create_entity("person", "Priya")
    clock.advance(seconds=1)
    updated = await graph.add_alias(priya.id, "my sister")
    assert updated.aliases == ["my sister"] and updated.updated_at == clock.now
    assert await graph.add_alias(priya.id, "My Sister") == updated
    assert await graph.add_alias(priya.id, "priya") == updated
    assert await graph.find_entities("my sister") == [updated]
    with pytest.raises(GraphNotFoundError):
        await graph.add_alias(999, "x")


async def test_rename_entity_reembeds_and_keeps_old_alias(graph: SQLiteGraphStore):
    user = await graph.rename_entity(USER_ENTITY_ID, "Yogesh")
    assert (user.id, user.name, user.aliases) == (USER_ENTITY_ID, "Yogesh", ["User"])
    assert await graph.find_entities("user") == [user]
    assert (await graph.similar_entities("Yogesh"))[0].entity.id == USER_ENTITY_ID
    again = await graph.rename_entity(USER_ENTITY_ID, "Yogi", keep_old_as_alias=False)
    assert (again.name, again.aliases) == ("Yogi", ["User"])
    back = await graph.rename_entity(USER_ENTITY_ID, "user")
    assert (back.name, back.aliases) == ("user", ["Yogi"])  # the new name leaves the aliases


async def test_update_summary_reembeds(graph: SQLiteGraphStore):
    priya = await graph.create_entity("person", "Priya")
    await graph.create_entity("person", "Rahul", summary="plays cricket")
    assert (await graph.similar_entities("architect"))[0].score <= 0
    updated = await graph.update_entity_summary(priya.id, "an architect")
    assert updated.summary == "an architect"
    hits = await graph.similar_entities("architect")
    assert hits[0].entity.id == priya.id and hits[0].score > 0.5
    assert (await graph.update_entity_summary(priya.id, "  ")).summary is None


async def test_merge_repoints_statements_aliases_and_vectors(graph: SQLiteGraphStore, path: Path):
    sister = await graph.create_entity("person", "my sister", aliases=["sis"], summary="family")
    priya = await graph.create_entity("person", "Priya")
    pune = await graph.create_entity("place", "Pune")
    lives = await graph.add_statement(
        new("Sister lives in Pune", subject_id=sister.id, object_id=pune.id)
    )
    loves = await graph.add_statement(new("User loves his sister", object_id=sister.id))
    merged = await graph.merge_entities(sister.id, priya.id)
    assert merged.id == priya.id and merged.aliases == ["my sister", "sis"]
    source = await graph.get_entity(sister.id)
    assert source is not None and source.merged_into == priya.id
    assert (await graph.get_entity(sister.id, resolve=True)) == merged
    assert (await graph.get_statement(lives.id)).subject_id == priya.id  # type: ignore[union-attr]
    assert (await graph.get_statement(loves.id)).object_id == priya.id  # type: ignore[union-attr]
    assert await graph.find_entities("sis") == [merged]
    assert await graph.entities_mentioned_in("how is my sister") == [merged]
    assert [s.id for s in await graph.current_statements(sister.id)] == [lives.id]
    assert {s.id for s in await graph.neighbours(sister.id)} == {lives.id, loves.id}
    # Vectors: the source's entity vector is gone; statement metadata follows the subject.
    assert sister.id not in await vector_ids(graph, "entities")
    hits = await graph.similar_statements("lives in Pune", subject_id=priya.id)
    assert [h.statement.id for h in hits] == [lives.id]
    assert all(h.entity.id != sister.id for h in await graph.similar_entities("my sister family"))
    assert count(path, "statements") == 2
    fts_consistent(path)


async def test_merge_chains_and_merged_targets(graph: SQLiteGraphStore):
    a = await graph.create_entity("person", "A")
    b = await graph.create_entity("person", "B")
    c = await graph.create_entity("person", "C")
    d = await graph.create_entity("person", "D")
    x = await graph.create_entity("person", "X")
    s = await graph.add_statement(new("A is tall", subject_id=a.id))
    await graph.merge_entities(a.id, b.id)
    await graph.merge_entities(b.id, c.id)
    # Path compression: A now points straight at C.
    assert (await graph.get_entity(a.id)).merged_into == c.id  # type: ignore[union-attr]
    # Merging into an entity that was itself merged lands on the survivor.
    survivor = await graph.merge_entities(x.id, a.id)
    assert survivor.id == c.id
    assert (await graph.get_entity(x.id)).merged_into == c.id  # type: ignore[union-attr]
    # Merging a merged source merges its survivor.
    final = await graph.merge_entities(b.id, d.id)
    assert final.id == d.id and set(final.aliases) == {"A", "B", "C", "X"}
    for entity in (a, b, c, x):
        assert (await graph.get_entity(entity.id, resolve=True)).id == d.id  # type: ignore[union-attr]
    assert (await graph.get_statement(s.id)).subject_id == d.id  # type: ignore[union-attr]
    assert await graph.merge_entities(a.id, d.id) == final  # already one entity: no-op
    assert [e.id for e in await graph.list_entities(type="person")] == [USER_ENTITY_ID, d.id]
    assert len(await graph.list_entities(include_merged=True)) == 6


async def test_user_entity_absorbs_but_is_never_merged_away(graph: SQLiteGraphStore):
    me = await graph.create_entity("person", "Yogesh")
    s = await graph.add_statement(new("Yogesh codes", subject_id=me.id))
    with pytest.raises(GraphStoreError, match="user entity"):
        await graph.merge_entities(USER_ENTITY_ID, me.id)
    user = await graph.merge_entities(me.id, USER_ENTITY_ID)
    assert user.id == USER_ENTITY_ID and "Yogesh" in user.aliases
    assert (await graph.get_statement(s.id)).subject_id == USER_ENTITY_ID  # type: ignore[union-attr]


# --- Statements ---------------------------------------------------------------


async def test_add_statement_with_sources(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, clock: Clock, path: Path
):
    e1 = await log(memory, UserMessage(session_id="s", turn_id="t", text="I drink decaf"))
    pune = await graph.create_entity("place", "Pune")
    observed = T0 - timedelta(hours=1)
    s = await graph.add_statement(
        new(
            "User lives in Pune",
            predicate="lives in",
            object_id=pune.id,
            valid_from=T0 - timedelta(days=30),
            observed_at=observed,
        ),
        [e1, e1],
    )
    assert s.subject_id == USER_ENTITY_ID and s.object_id == pune.id and s.object_value is None
    assert (s.kind, s.origin, s.confidence, s.evidence_count) == (
        StatementKind.FACT,
        Origin.STATED,
        0.9,
        1,
    )
    assert (s.recorded_at, s.last_observed_at, s.sources) == (T0, observed, [e1])
    assert (s.invalid_at, s.superseded_by) == (None, None)
    assert await graph.get_statement(s.id) == s
    assert await graph.get_statement(999) is None
    literal = await graph.add_statement(new("User is 40", predicate="age", object_value="40"))
    assert literal.object_value == "40" and literal.sources == []
    assert await graph.get_statements([literal.id, 999, s.id]) == [literal, s]
    assert {s.id, literal.id} == await vector_ids(graph, "statements")
    fts_consistent(path)


async def test_add_statement_failures_store_nothing(
    graph: SQLiteGraphStore, embedder: HashingEmbedder, path: Path
):
    with pytest.raises(GraphStoreError):
        await graph.add_statement(new("User likes tea"), [12345])  # unknown episode
    with pytest.raises(GraphNotFoundError):
        await graph.add_statement(new("Ghost likes tea", subject_id=777))
    with pytest.raises(GraphStoreError):  # one bad item fails the whole batch
        await graph.add_statements([(new("ok one"), []), (new("bad", object_id=888), [])])
    embedder.fail = True
    with pytest.raises(RuntimeError):
        await graph.add_statement(new("User likes coffee"))
    assert count(path, "statements") == 0 and count(path, "statement_sources") == 0
    assert await vector_ids(graph, "statements") == set()
    fts_consistent(path)


async def test_new_statement_validation():
    with pytest.raises(pydantic.ValidationError, match="at most one"):
        new("x", object_id=2, object_value="y")
    with pytest.raises(pydantic.ValidationError, match="before valid_from"):
        new("x", valid_from=T0, invalid_at=T0 - timedelta(days=1))
    with pytest.raises(pydantic.ValidationError):
        new("x", confidence=1.2)
    with pytest.raises(pydantic.ValidationError):
        NewStatement(
            subject_id=1, predicate="p", text="  ", kind="fact", origin="stated", confidence=1
        )


async def test_statement_about_merged_entity_resolves_subject(graph: SQLiteGraphStore):
    a = await graph.create_entity("person", "A")
    b = await graph.create_entity("person", "B")
    await graph.merge_entities(a.id, b.id)
    s = await graph.add_statement(new("A is kind", subject_id=a.id, object_id=a.id))
    assert (s.subject_id, s.object_id) == (b.id, b.id)


async def test_reinforce(graph: SQLiteGraphStore, memory: SQLiteMemoryStore, clock: Clock):
    e1 = await log(memory, UserMessage(session_id="s", text="tea again"))
    e2 = await log(memory, UserMessage(session_id="s", text="more tea"))
    s = await graph.add_statement(
        new("User likes tea", origin=Origin.INFERRED, confidence=0.3), [e1]
    )
    clock.advance(days=1)
    r = await graph.reinforce(s.id, [e2], confidence=0.5)
    assert (r.evidence_count, r.confidence, r.sources) == (2, 0.5, [e1, e2])
    assert r.last_observed_at == clock.now and r.recorded_at == T0
    # A retried reinforcement with the same episodes does not count twice.
    again = await graph.reinforce(s.id, [e2], confidence=0.5)
    assert again.evidence_count == 2
    # An observation without provenance (e.g. reflection) counts; confidence kept.
    clock.advance(days=1)
    r = await graph.reinforce(s.id, observed_at=clock.now - timedelta(days=5))
    assert (r.evidence_count, r.confidence) == (3, 0.5)
    assert r.last_observed_at == T0 + timedelta(days=1)  # never moves backwards
    with pytest.raises(GraphNotFoundError):
        await graph.reinforce(999, [])
    with pytest.raises(ValueError):
        await graph.reinforce(s.id, confidence=2)
    with pytest.raises(GraphStoreError):
        await graph.reinforce(s.id, [4242])
    assert (await graph.get_statement(s.id)).evidence_count == 3  # type: ignore[union-attr]


async def test_supersede_closes_old_at_new_valid_from(graph: SQLiteGraphStore, clock: Clock):
    old = await graph.add_statement(
        new("User drinks dark roast", predicate="drinks", valid_from=T0 - timedelta(days=400))
    )
    switch = T0 - timedelta(days=3)
    clock.advance(days=1)
    nu = await graph.supersede(
        old.id, new("User drinks decaf", predicate="drinks", valid_from=switch), []
    )
    closed = await graph.get_statement(old.id)
    assert closed is not None
    assert (closed.invalid_at, closed.superseded_by) == (switch, nu.id)
    assert nu.recorded_at == clock.now and nu.invalid_at is None
    assert [s.id for s in await graph.current_statements(USER_ENTITY_ID)] == [nu.id]
    hits = await graph.similar_statements("drinks roast")
    assert [h.statement.id for h in hits] == [nu.id]
    hits = await graph.similar_statements("drinks roast", include_invalid=True)
    assert hits[0].statement.id == old.id


async def test_supersede_without_valid_from_ends_now_and_keeps_earlier_end(
    graph: SQLiteGraphStore, clock: Clock
):
    a = await graph.add_statement(new("User works at Acme"))
    nu = await graph.supersede(a.id, new("User works at Initech"))
    assert (await graph.get_statement(a.id)).invalid_at == clock.now  # type: ignore[union-attr]
    ended = T0 - timedelta(days=10)
    b = await graph.add_statement(new("User lived in Delhi"))
    await graph.invalidate(b.id, ended)
    await graph.supersede(b.id, new("User lives in Pune"))
    closed = await graph.get_statement(b.id)
    assert closed is not None and closed.invalid_at == ended and closed.superseded_by
    assert nu.id != closed.superseded_by


async def test_supersede_is_atomic(graph: SQLiteGraphStore, embedder: HashingEmbedder, path: Path):
    old = await graph.add_statement(new("User drinks tea", valid_from=T0))
    first = await graph.supersede(old.id, new("User drinks coffee"))
    before = count(path, "statements")
    with pytest.raises(GraphStoreError, match="already superseded"):
        await graph.supersede(old.id, new("User drinks water"))
    other = await graph.add_statement(new("User owns a bike", valid_from=T0))
    with pytest.raises(GraphStoreError, match="before"):
        await graph.supersede(other.id, new("User owns a car", valid_from=T0 - timedelta(days=1)))
    with pytest.raises(GraphNotFoundError):
        await graph.supersede(9999, new("x"))
    embedder.fail = True
    with pytest.raises(RuntimeError):
        await graph.supersede(other.id, new("User owns a car"))
    embedder.fail = False
    assert count(path, "statements") == before + 1  # only `other` was added
    assert (await graph.get_statement(other.id)).invalid_at is None  # type: ignore[union-attr]
    assert (await graph.get_statement(old.id)).superseded_by == first.id  # type: ignore[union-attr]
    fts_consistent(path)


async def test_future_valid_from_keeps_old_current_until_then(
    graph: SQLiteGraphStore, clock: Clock
):
    old = await graph.add_statement(new("User lives in Pune"))
    move = T0 + timedelta(days=30)
    nu = await graph.supersede(old.id, new("User lives in Berlin", valid_from=move))
    current = {s.id for s in await graph.current_statements(USER_ENTITY_ID)}
    assert current == {old.id, nu.id}
    assert {h.statement.id for h in await graph.similar_statements("lives", k=5)} == current
    clock.advance(days=31)
    assert [s.id for s in await graph.current_statements(USER_ENTITY_ID)] == [nu.id]
    assert [h.statement.id for h in await graph.similar_statements("lives", k=5)] == [nu.id]


async def test_invalidate_and_confidence(graph: SQLiteGraphStore, clock: Clock):
    s = await graph.add_statement(new("User is training for a marathon", kind=StatementKind.GOAL))
    t = await graph.add_statement(new("User likes tea", origin=Origin.INFERRED, confidence=0.4))
    closed = await graph.invalidate(s.id)
    assert closed.invalid_at == clock.now and closed.superseded_by is None
    assert [x.id for x in await graph.current_statements(USER_ENTITY_ID)] == [t.id]
    await graph.set_confidence(t.id, 0.25)
    assert (await graph.get_statement(t.id)).confidence == 0.25  # type: ignore[union-attr]
    assert await graph.set_confidences({t.id: 0.2, s.id: 0.1, 999: 0.5}) == 2
    assert (await graph.get_statement(s.id)).confidence == 0.1  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        await graph.set_confidences({t.id: 0.3, s.id: 1.1})
    assert (await graph.get_statement(t.id)).confidence == 0.2  # type: ignore[union-attr]
    with pytest.raises(GraphNotFoundError):
        await graph.set_confidence(999, 0.5)
    with pytest.raises(GraphNotFoundError):
        await graph.invalidate(999)
    with pytest.raises(ValueError, match="naive"):
        await graph.invalidate(t.id, T0.replace(tzinfo=None))
    later = await graph.add_statement(new("User owns a bike", valid_from=T0))
    with pytest.raises(GraphStoreError):
        await graph.invalidate(later.id, T0 - timedelta(days=1))


async def test_listing_filters(graph: SQLiteGraphStore, clock: Clock):
    priya = await graph.create_entity("person", "Priya")
    stated = await graph.add_statement(new("User likes tea", kind=StatementKind.PREFERENCE))
    old_inference = await graph.add_statement(
        new(
            "User skips lunch on Fridays",
            kind=StatementKind.HABIT,
            origin=Origin.INFERRED,
            confidence=0.5,
            observed_at=T0 - timedelta(days=60),
        )
    )
    fresh_inference = await graph.add_statement(
        new("User walks daily", kind=StatementKind.HABIT, origin=Origin.INFERRED, confidence=0.7)
    )
    rule = await graph.add_statement(
        new("Don't read URLs aloud", kind=StatementKind.INSTRUCTION, confidence=1.0)
    )
    gone = await graph.add_statement(new("Never use emoji", kind=StatementKind.INSTRUCTION))
    await graph.invalidate(gone.id)
    hers = await graph.add_statement(new("Priya lives in Pune", subject_id=priya.id))

    assert [s.id for s in await graph.active_instructions()] == [rule.id]
    habits = await graph.current_statements(USER_ENTITY_ID, kinds=[StatementKind.HABIT])
    assert [s.id for s in habits] == [old_inference.id, fresh_inference.id]
    stale = await graph.list_statements(
        origin=Origin.INFERRED, observed_before=T0 - timedelta(days=30)
    )
    assert [s.id for s in stale] == [old_inference.id]
    weak = await graph.list_statements(origin="inferred", confidence_below=0.6)
    assert [s.id for s in weak] == [old_inference.id]
    assert [s.id for s in await graph.list_statements(min_confidence=0.95)] == [rule.id]
    everything = await graph.list_statements(current_only=False)
    assert len(everything) == 6
    page = await graph.list_statements(after_id=stated.id, limit=2)
    assert [s.id for s in page] == [old_inference.id, fresh_inference.id]
    assert [s.id for s in await graph.list_statements(subject_id=priya.id)] == [hers.id]
    assert await graph.list_statements(kinds=[]) == []


async def test_neighbours(graph: SQLiteGraphStore):
    priya = await graph.create_entity("person", "Priya")
    pune = await graph.create_entity("place", "Pune")
    a = await graph.add_statement(new("Priya lives in Pune", subject_id=priya.id, confidence=0.6))
    b = await graph.add_statement(new("User visits Priya", object_id=priya.id, confidence=0.95))
    c = await graph.add_statement(new("Priya was a nurse", subject_id=priya.id, confidence=0.99))
    await graph.invalidate(c.id)
    goal = await graph.add_statement(
        new("Priya wants a dog", subject_id=priya.id, kind=StatementKind.GOAL, confidence=0.5)
    )
    assert [s.id for s in await graph.neighbours(priya.id)] == [b.id, a.id, goal.id]
    assert [s.id for s in await graph.neighbours(priya.id, limit=1)] == [b.id]
    assert [s.id for s in await graph.neighbours(priya.id, kinds=["goal"])] == [goal.id]
    assert [s.id for s in await graph.neighbours(pune.id)] == []
    with pytest.raises(GraphNotFoundError):
        await graph.neighbours(999)


async def test_similar_statements_filters(graph: SQLiteGraphStore, embedder: HashingEmbedder):
    priya = await graph.create_entity("person", "Priya")
    mine = await graph.add_statement(new("User drinks coffee daily"))
    hers = await graph.add_statement(new("Priya drinks coffee daily", subject_id=priya.id))
    goal = await graph.add_statement(new("User wants less coffee", kind=StatementKind.GOAL))
    hits = await graph.similar_statements("drinks coffee")
    assert hits[0].statement.id in {mine.id, hers.id} and len(hits) == 3
    assert hits[0].score > hits[-1].score
    hits = await graph.similar_statements("drinks coffee", subject_id=priya.id)
    assert [h.statement.id for h in hits] == [hers.id]
    hits = await graph.similar_statements("coffee", kinds=[StatementKind.GOAL])
    assert [h.statement.id for h in hits] == [goal.id]
    assert await graph.similar_statements("coffee", k=0) == []
    assert embedder.calls[-1] == (["coffee"], "document")
    await graph.similar_statements("coffee", kind="query")
    assert embedder.calls[-1] == (["coffee"], "query")


async def test_keyword_statements_handles_hostile_input(graph: SQLiteGraphStore):
    s = await graph.add_statement(new("User runs marathons in the summer"))
    hits = await graph.keyword_statements("running")  # porter stemming: runs ~ running
    assert [h.statement.id for h in hits] == [s.id] and hits[0].score > 0
    hostile = [
        "",
        "   ",
        '"',
        '""',
        "'",
        "*",
        "^runs",
        "runs*",
        "NOT runs",
        "runs AND",
        "OR",
        "NEAR(runs summer, 2)",
        "text:runs",
        "(runs",
        "runs)",
        "{runs}",
        "-runs",
        "+",
        "runs; DROP TABLE statements; --",
        "\x00",
        "😀 runs",
        "_",
        "AND OR NOT",
        "a" * 5000,
    ]
    for text in hostile:
        hits = await graph.keyword_statements(text)
        assert all(h.statement.id == s.id for h in hits)
    assert [h.statement.id for h in await graph.keyword_statements("NOT runs")] == [s.id]
    assert [h.statement.id for h in await graph.keyword_statements("runs; DROP")] == [s.id]


async def test_set_predicate(graph: SQLiteGraphStore):
    a = await graph.add_statement(new("User likes tea", predicate="likes"))
    b = await graph.add_statement(new("User enjoys tea", predicate="enjoys"))
    assert await graph.set_predicate([a.id, b.id, 99], "likes") == 2
    assert {s.predicate for s in await graph.get_statements([a.id, b.id])} == {"likes"}


# --- Episodes -----------------------------------------------------------------


async def test_episode_access_for_the_learning_loop(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, clock: Clock
):
    call = ToolCall(id="c", name="weather", arguments={})
    events = [
        UserMessage(session_id="s", turn_id="t1", text="hi"),
        ToolStarted(session_id="s", turn_id="t1", call=call),
        ToolFinished(session_id="s", turn_id="t1", result=ToolResult(call_id="c", content="sun")),
        AssistantMessage(session_id="s", turn_id="t1", text="It is sunny."),
        TurnEnded(session_id="s", turn_id="t1", reason="completed"),
        UserMessage(session_id="other", turn_id="x", text="elsewhere"),
        UserMessage(session_id="s", turn_id="t2", text="thanks"),
    ]
    ids = [await log(memory, e) for e in events]
    content = [ids[0], ids[2], ids[3], ids[5], ids[6]]
    batch = await graph.unextracted_episodes(limit=3)
    assert [r.id for r in batch] == content[:3]
    assert batch[0].event == events[0] and batch[0].extracted_at is None
    assert await graph.mark_extracted([r.id for r in batch]) == 3
    rest = await graph.unextracted_episodes()
    assert [r.id for r in rest] == content[3:]
    assert await graph.unextracted_episodes(limit=0) == []

    got = await graph.get_episodes([ids[3], ids[0], 9999])
    assert [r.id for r in got] == [ids[0], ids[3]] and got[0].extracted_at == clock.now
    assert [r.id for r in await graph.turn_episodes("t1")] == [ids[0], ids[2], ids[3]]
    assert [r.id for r in await graph.turn_episodes("t1", types=None)] == ids[:5]
    before = await graph.episodes_before(ids[6], 2)
    assert [r.id for r in before] == [ids[2], ids[3]]  # same session, content only
    assert [r.id for r in await graph.episodes_before(ids[6], 2, types=None)] == ids[3:5]
    assert await graph.episodes_before(ids[0], 5) == []
    assert await graph.episodes_before(ids[6], 0) == []
    with pytest.raises(GraphNotFoundError):
        await graph.episodes_before(9999, 3)


# --- Profile ------------------------------------------------------------------


async def test_profile_versions(graph: SQLiteGraphStore, clock: Clock):
    assert await graph.current_profile() is None
    first = await graph.save_profile("Yogesh, software engineer.")
    clock.advance(days=1)
    second = await graph.save_profile("Yogesh, software engineer in Pune.")
    assert await graph.current_profile() == second
    assert second.created_at == clock.now
    assert await graph.profile_versions() == [second, first]
    assert await graph.profile_versions(limit=1) == [second]


# --- Full-text index sync -----------------------------------------------------


async def test_fts_stays_in_sync(graph: SQLiteGraphStore, memory: SQLiteMemoryStore, path: Path):
    episode = await log(memory, UserMessage(session_id="s", turn_id="t", text="x"))
    a = await graph.create_entity("person", "Alpha", aliases=["first"])
    b = await graph.create_entity("person", "Beta")
    s = await graph.add_statement(new("Alpha builds boats", subject_id=a.id), [episode])
    fts_consistent(path)
    await graph.add_alias(a.id, "skipper")
    await graph.rename_entity(b.id, "Gamma")
    fts_consistent(path)
    assert await graph.find_entities("skipper") == [await graph.get_entity(a.id)]
    assert await graph.find_entities("Beta") != []  # kept as alias
    await graph.merge_entities(a.id, b.id)
    fts_consistent(path)
    assert [h.statement.id for h in await graph.keyword_statements("boats")] == [s.id]
    await graph.forget_statements([s.id])
    fts_consistent(path)
    assert await graph.keyword_statements("boats") == []
    await graph.add_statement(new("Gamma sails"))
    await graph.reset_graph()
    fts_consistent(path)
    assert await graph.keyword_statements("sails") == []
    assert await graph.find_entities("Gamma") == []
    await graph.rebuild_fts()
    fts_consistent(path)


async def test_fts_rows_survive_close_and_reopen(path: Path, clock: Clock):
    async with SQLiteGraphStore(path, HashingEmbedder(), clock=clock) as graph:
        await graph.add_statement(new("User plays chess"))
    async with SQLiteGraphStore(path, HashingEmbedder(), clock=clock) as graph:
        assert len(await graph.keyword_statements("chess")) == 1
    assert json.loads(query(path, "SELECT aliases FROM entities WHERE id = 1")[0][0]) == []
