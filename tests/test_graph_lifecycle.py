"""Forget, reset, the vector index across embedders, and concurrency on one file."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from lucy.core.events import AssistantMessage, TurnEnded, UserMessage
from lucy.memory.graph_store import GraphStoreError, SQLiteGraphStore
from lucy.memory.models import USER_ENTITY_ID, ForgetResult
from lucy.memory.sqlite_jobs import SQLiteJobStore
from lucy.memory.sqlite_store import SQLiteMemoryStore
from tests.graph_support import (
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


def trace_with(turn_id: str, text: str):
    """A trace whose prompt contains `text` (as memory or history would)."""
    base = _trace(turn_id)
    step = base.steps[0]
    messages = [step.messages[0].model_copy(update={"content": f"Memories:\n- {text}"})]
    return base.model_copy(update={"steps": [step.model_copy(update={"messages": messages})]})


async def turn(memory: SQLiteMemoryStore, turn_id: str, user: str, reply: str) -> list[int]:
    """Log one turn; returns the episode ids [user, assistant, turn_ended]."""
    return [
        await log(memory, UserMessage(session_id="s", turn_id=turn_id, text=user)),
        await log(memory, AssistantMessage(session_id="s", turn_id=turn_id, text=reply)),
        await log(memory, TurnEnded(session_id="s", turn_id=turn_id, reason="completed")),
    ]


# --- Forget -------------------------------------------------------------------


async def test_forget_deletes_statement_turn_traces_and_indexes(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, path: Path
):
    t1 = await turn(memory, "t1", "My sister Priya lives in Pune", "Noted, Priya is in Pune.")
    t2 = await turn(memory, "t2", "I like green tea", "Green tea it is.")
    await memory.record(trace_with("t1", "irrelevant"))
    await memory.record(trace_with("t2", "nothing"))
    await memory.record(trace_with("t3", "User's sister Priya lives in Pune"))  # read back later
    await memory.record(trace_with("t4", "Priya"))  # too short to search for: kept
    s = await graph.add_statement(new("User's sister Priya lives in Pune"), [t1[0]])
    tea = await graph.add_statement(new("User likes green tea"), [t2[0]])

    result = await graph.forget_statements([s.id, 4242])
    assert result.statement_ids == [s.id]
    assert result.episode_ids == t1  # the whole turn, including the assistant's echo
    assert result.trace_turn_ids == ["t1", "t3"]
    assert await graph.get_statement(s.id) is None
    assert await graph.get_statement(tea.id) is not None
    assert await memory.get_trace("t2") is not None and await memory.get_trace("t4") is not None
    assert [e.id for e in await graph.get_episodes(t1 + t2)] == t2
    assert await graph.keyword_statements("Pune") == []
    assert await graph.similar_statements("sister Priya lives in Pune", k=5) != []
    assert all(h.statement.id == tea.id for h in await graph.similar_statements("Pune"))
    assert await vector_ids(graph, "statements") == {tea.id}
    fts_consistent(path)
    # Rebuilding from the log cannot bring it back.
    await graph.reset_graph()
    texts = [r.event.text for r in await graph.unextracted_episodes()]  # type: ignore[union-attr]
    assert not any("Pune" in t for t in texts)
    assert await graph.forget_statements([]) == ForgetResult()


async def test_forget_with_shared_and_unshared_sources(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore
):
    t1 = await turn(memory, "t1", "Priya moved to Pune and got a dog", "Nice!")
    t2 = await turn(memory, "t2", "Priya's dog is called Bruno", "Cute.")
    moved = await graph.add_statement(new("Priya lives in Pune"), [t1[0]])
    orphan = await graph.add_statement(new("Priya has a dog"), [t1[0]])  # same episode only
    shared = await graph.add_statement(new("Priya's dog exists"), [t1[0], t2[0]])
    by_reply = await graph.add_statement(new("Priya is nice"), [t1[1]])  # same turn, other ep
    unsourced = await graph.add_statement(new("User is a person"))

    result = await graph.forget_statements([moved.id])
    assert result.statement_ids == sorted([moved.id, orphan.id, by_reply.id])
    survivor = await graph.get_statement(shared.id)
    assert survivor is not None and survivor.sources == [t2[0]]
    assert await graph.get_statement(unsourced.id) is not None


async def test_forget_keeps_supersession_consistent(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, clock: Clock
):
    e = [(await turn(memory, f"t{i}", f"statement number {i} said", "ok"))[0] for i in range(5)]
    # Forgetting a correction reopens what it corrected.
    roast = await graph.add_statement(new("User drinks dark roast"), [e[0]])
    decaf = await graph.supersede(roast.id, new("User drinks decaf"), [e[1]])
    result = await graph.forget_statements([decaf.id])
    assert result.reopened_statement_ids == [roast.id]
    reopened = await graph.get_statement(roast.id)
    assert reopened is not None and (reopened.invalid_at, reopened.superseded_by) == (None, None)
    hits = await graph.similar_statements("drinks dark roast")  # vector metadata reopened too
    assert [h.statement.id for h in hits] == [roast.id]

    # Forgetting the middle of a chain repoints to the surviving successor.
    a = await graph.add_statement(new("User lives in Delhi"), [e[2]])
    clock.advance(days=1)
    b = await graph.supersede(a.id, new("User lives in Mumbai"), [e[3]])
    ended = (await graph.get_statement(a.id)).invalid_at  # type: ignore[union-attr]
    clock.advance(days=1)
    c = await graph.supersede(b.id, new("User lives in Pune"), [e[4]])
    result = await graph.forget_statements([b.id])
    assert result.repointed_statement_ids == [a.id] and result.reopened_statement_ids == []
    first = await graph.get_statement(a.id)
    assert first is not None and (first.superseded_by, first.invalid_at) == (c.id, ended)
    assert (await graph.get_statement(c.id)).invalid_at is None  # type: ignore[union-attr]

    # Forgetting a statement that was superseded leaves its successor alone.
    result = await graph.forget_statements([a.id])
    assert result.statement_ids == [a.id]
    assert await graph.get_statement(c.id) is not None


async def test_forget_chain_with_all_successors_forgotten_reopens(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore
):
    e = (await turn(memory, "t1", "x", "y"))[0]
    keep = await graph.add_statement(new("User has a cat"))
    b = await graph.supersede(keep.id, new("User has a dog"), [e])
    c = await graph.supersede(b.id, new("User has a parrot"), [e])
    result = await graph.forget_statements([b.id])  # c shares the episode, so it goes too
    assert result.statement_ids == [b.id, c.id] and result.reopened_statement_ids == [keep.id]
    assert (await graph.get_statement(keep.id)).invalid_at is None  # type: ignore[union-attr]


async def test_forget_cleans_up_entities_and_profile(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, clock: Clock
):
    e1 = (await turn(memory, "t1", "about Priya", "ok"))[0]
    e2 = (await turn(memory, "t2", "about Rahul", "ok"))[0]
    early = await graph.save_profile("before anything was learned")
    clock.advance(hours=1)
    priya = await graph.create_entity("person", "Priya", summary="sister in Pune")
    alias = await graph.create_entity("person", "my sister")
    await graph.merge_entities(alias.id, priya.id)
    rahul = await graph.create_entity("person", "Rahul", summary="friend who knows Priya")
    s = await graph.add_statement(new("Priya lives in Pune", subject_id=priya.id), [e1])
    knows = await graph.add_statement(
        new("Rahul knows Priya", subject_id=rahul.id, object_id=priya.id), [e1]
    )
    keeps = await graph.add_statement(new("Rahul plays chess", subject_id=rahul.id), [e2])
    about_user = await graph.add_statement(new("User loves Priya", object_id=priya.id), [e1])
    clock.advance(hours=1)
    await graph.save_profile("mentions Priya in Pune")

    result = await graph.forget_statements([s.id])
    assert set(result.statement_ids) == {s.id, knows.id, about_user.id}
    assert result.entity_ids == sorted([priya.id, alias.id])
    assert result.cleared_summary_entity_ids == [rahul.id]
    assert await graph.get_entity(priya.id) is None and await graph.get_entity(alias.id) is None
    assert await graph.find_entities("my sister") == []
    rahul_now = await graph.get_entity(rahul.id)
    assert rahul_now is not None and rahul_now.summary is None
    assert rahul.id not in await vector_ids(graph, "entities")  # re-embedded later by name
    assert priya.id not in await vector_ids(graph, "entities")
    assert await graph.get_statement(keeps.id) is not None
    assert await graph.get_entity(USER_ENTITY_ID) is not None
    assert result.profile_versions_deleted == 1
    assert await graph.current_profile() == early


async def test_forget_dry_run_changes_nothing(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, path: Path
):
    e = await turn(memory, "t1", "I like tea", "ok")
    await memory.record(trace_with("t1", "x"))
    s = await graph.add_statement(new("User likes tea"), [e[0]])
    preview = await graph.forget_statements([s.id], dry_run=True)
    assert preview.statement_ids == [s.id] and preview.episode_ids == e
    assert preview.trace_turn_ids == ["t1"]
    assert await graph.get_statement(s.id) is not None
    assert len(await graph.get_episodes(e)) == 3 and await memory.get_trace("t1") is not None
    assert await graph.forget_statements([s.id]) == preview
    fts_consistent(path)


async def test_forget_is_all_or_nothing(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, monkeypatch: pytest.MonkeyPatch
):
    e = await turn(memory, "t1", "I like tea", "ok")
    await memory.record(trace_with("t1", "x"))
    s = await graph.add_statement(new("User likes tea"), [e[0]])

    async def crash(*args: object, **kwargs: object) -> None:
        raise RuntimeError("power cut")

    await graph.create_entity("thing", "tea")
    t = await graph.add_statement(new("tea is a drink", subject_id=2), [e[0]])
    monkeypatch.setattr(graph._vectors, "delete_entities", crash)  # late in the transaction
    with pytest.raises(RuntimeError, match="power cut"):
        await graph.forget_statements([s.id])
    assert len(await graph.get_statements([s.id, t.id])) == 2
    assert len(await graph.get_episodes(e)) == 3 and await memory.get_trace("t1") is not None
    assert await vector_ids(graph, "statements") == {s.id, t.id}


# --- Reset --------------------------------------------------------------------


async def test_reset_graph_clears_derived_data_only(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, path: Path
):
    e = await turn(memory, "t1", "I'm Yogesh and I like tea", "Hi Yogesh")
    await memory.record(_trace("t1"))
    async with SQLiteJobStore(path) as jobs:
        await jobs.enqueue("extract", {})
    await graph.mark_extracted(e[:2])
    await graph.rename_entity(USER_ENTITY_ID, "Yogesh")
    await graph.update_entity_summary(USER_ENTITY_ID, "likes tea")
    tea = await graph.create_entity("thing", "tea")
    other = await graph.create_entity("thing", "chai")
    await graph.merge_entities(other.id, tea.id)
    s = await graph.add_statement(new("User likes tea", object_id=tea.id), [e[0]])
    await graph.supersede(s.id, new("User likes coffee"), [e[0]])
    await graph.save_profile("Yogesh likes tea")

    result = await graph.reset_graph()
    assert (result.statements_deleted, result.entities_deleted) == (2, 2)
    assert (result.profile_versions_deleted, result.episodes_reset) == (1, 2)
    user = await graph.user_entity()
    assert (user.id, user.name, user.aliases, user.summary) == (1, "User", [], None)
    assert await graph.list_statements(current_only=False) == []
    assert await graph.list_entities(include_merged=True) == [user]
    assert await graph.current_profile() is None
    assert query(path, "SELECT count(*) FROM vector_spaces") == [(0,)]
    assert [r.id for r in await graph.unextracted_episodes()] == e[:2]
    assert await memory.get_trace("t1") is not None
    assert query(path, "SELECT count(*) FROM jobs") == [(1,)]
    fts_consistent(path)
    status = await graph.vector_status()
    assert (status.statements_missing, status.entities_missing) == (0, 1)
    # The graph is usable again straight away.
    again = await graph.add_statement(new("User likes tea"), [e[0]])
    assert [h.statement.id for h in await graph.similar_statements("tea")] == [again.id]


# --- Vector index -------------------------------------------------------------


async def test_switching_embedder_model_isolates_vectors(path: Path, clock: Clock):
    old_model = HashingEmbedder(model="old")
    async with SQLiteGraphStore(path, old_model, clock=clock) as old:
        tea = await old.add_statement(new("User likes green tea"))
        priya = await old.create_entity("person", "Priya")

    new_model = HashingEmbedder(model="new", dimensions=128)
    async with SQLiteGraphStore(path, new_model, clock=clock) as graph:
        assert await graph.similar_statements("green tea") == []  # old vectors ignored
        assert await graph.similar_entities("Priya") == []
        status = await graph.vector_status()
        assert (status.model, status.dimensions) == ("new", 128)
        assert (status.statements_missing, status.entities_missing) == (1, 2)
        assert await graph.missing_vectors() == ([tea.id], [USER_ENTITY_ID, priya.id])
        coffee = await graph.add_statement(new("User likes black coffee"))
        assert [h.statement.id for h in await graph.similar_statements("tea", k=5)] == [coffee.id]
        assert await graph.reembed_missing() == 3
        hits = await graph.similar_statements("green tea", k=1)
        assert [h.statement.id for h in hits] == [tea.id]
        # Changes made under the new model keep the old space's metadata in sync.
        await graph.invalidate(tea.id)

    # Same model name, different dimensions: a separate space, no crash.
    async with SQLiteGraphStore(path, HashingEmbedder(model="old", dimensions=64)) as odd:
        assert await odd.similar_statements("tea") == []

    async with SQLiteGraphStore(path, old_model, clock=clock) as old:
        assert await old.similar_statements("green tea") == []  # invalidated, in every space
        hits = await old.similar_statements("green tea", include_invalid=True)
        assert hits[0].statement.id == tea.id
        assert (await old.vector_status()).statements_missing == 1  # only coffee
    # Searching never creates a space; only storing vectors does.
    assert query(path, "SELECT model, dimensions FROM vector_spaces ORDER BY id") == [
        ("old", 256),
        ("new", 128),
    ]


async def test_reembed_missing_is_resumable(path: Path, clock: Clock):
    async with SQLiteGraphStore(path, HashingEmbedder(model="a"), clock=clock) as old:
        await old.add_statements([(new(f"User fact number {i}"), []) for i in range(7)])
    embedder = HashingEmbedder(model="b")
    async with SQLiteGraphStore(path, embedder, clock=clock) as graph:
        assert await graph.reembed_missing(limit=3, batch_size=2) == 3
        assert (await graph.vector_status()).statements_missing == 4

        # Interrupted part-way: the batches already stored are kept.
        real_embed = embedder.embed
        calls = 0

        async def flaky(texts, *, kind="document"):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("embedder crashed")
            return await real_embed(texts, kind=kind)

        embedder.embed = flaky  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="crashed"):
            await graph.reembed_missing(batch_size=2)
        status = await graph.vector_status()
        assert (status.statements_missing, status.entities_missing) == (2, 1)
        embedder.embed = real_embed  # type: ignore[method-assign]

        assert await graph.reembed_missing(batch_size=2) == 3  # 2 statements + the user
        assert await graph.reembed_missing() == 0
        status = await graph.vector_status()
        assert (status.statements_missing, status.entities_missing) == (0, 0)
        with pytest.raises(ValueError):
            await graph.reembed_missing(batch_size=0)


async def test_reembed_skips_items_changed_while_embedding(graph: SQLiteGraphStore, path: Path):
    priya = await graph.create_entity("person", "Priya")
    await graph.drop_vector_index()
    other = SQLiteGraphStore(path, HashingEmbedder(), clock=Clock())
    embedder: HashingEmbedder = graph.embedder  # type: ignore[assignment]
    embedder.gate = asyncio.Event()
    task = asyncio.create_task(graph.reembed_missing())
    await embedder.entered.wait()  # reembed has read the texts and is embedding them
    await other.rename_entity(priya.id, "Priya Sharma")  # stores its own fresh vector
    embedder.gate.set()
    assert await task == 1  # only the user; Priya's stale text was not written over
    await other.close()
    hits = await graph.similar_entities("Sharma")
    assert hits[0].entity.id == priya.id and hits[0].score > 0


async def test_drop_and_rebuild_vector_index(graph: SQLiteGraphStore, path: Path):
    priya = await graph.create_entity("person", "Priya", summary="sister")
    statements = await graph.add_statements(
        [(new("User likes tea"), []), (new("Priya lives in Pune", subject_id=priya.id), [])]
    )
    coffee = await graph.supersede(statements[0].id, new("User likes coffee"))
    assert await graph.drop_vector_index() == 1
    assert await graph.similar_statements("tea", include_invalid=True) == []
    assert query(path, "SELECT count(*) FROM sqlite_master WHERE name GLOB 'vec_*'") == [(0,)]
    assert await graph.rebuild_vector_index(batch_size=2) == 5  # 3 statements + 2 entities
    ids = [h.statement.id for h in await graph.similar_statements("likes", k=5)]
    assert ids == [coffee.id, statements[1].id]  # metadata rebuilt: tea is closed
    hits = await graph.similar_statements("Pune", subject_id=priya.id)
    assert [h.statement.id for h in hits] == [statements[1].id]
    assert (await graph.similar_entities("sister"))[0].entity.id == priya.id


async def test_embedder_returning_wrong_dimensions_is_rejected(path: Path):
    class Broken(HashingEmbedder):
        async def embed(self, texts, *, kind="document"):  # type: ignore[no-untyped-def]
            return [[1.0, 0.0] for _ in texts]

    async with SQLiteGraphStore(path, Broken()) as graph:
        with pytest.raises(GraphStoreError, match="dimensions"):
            await graph.add_statement(new("User likes tea"))
        assert await graph.list_statements() == []


# --- Concurrency --------------------------------------------------------------


async def test_graph_and_memory_stores_write_concurrently(path: Path, clock: Clock):
    memory = SQLiteMemoryStore(path)
    graph = SQLiteGraphStore(path, HashingEmbedder(), clock=clock)
    other = SQLiteGraphStore(path, HashingEmbedder(), clock=clock)
    episodes = [UserMessage(session_id="s", text=f"message {i}") for i in range(60)]
    seed = await graph.add_statement(new("User likes tea"))

    async def add(i: int) -> None:
        await (graph if i % 2 else other).add_statement(new(f"User fact {i}"))

    async def reinforce(i: int) -> None:
        await (graph if i % 2 else other).reinforce(seed.id)

    await asyncio.wait_for(
        asyncio.gather(
            *(memory.append_episode(e) for e in episodes),
            *(add(i) for i in range(40)),
            *(reinforce(i) for i in range(30)),
        ),
        timeout=30,
    )
    assert len(await memory.recent_episodes("s", 100)) == 60
    assert len(await graph.list_statements()) == 41
    assert (await other.get_statement(seed.id)).evidence_count == 31  # type: ignore[union-attr]
    assert (await graph.vector_status()).statements_missing == 0
    for store in (memory, graph, other):
        await store.close()
    fts_consistent(path)


async def test_slow_embedder_does_not_hold_the_write_lock(path: Path, clock: Clock):
    slow = HashingEmbedder()
    slow.gate = asyncio.Event()
    graph = SQLiteGraphStore(path, slow, clock=clock)
    other = SQLiteGraphStore(path, HashingEmbedder(), clock=clock)
    memory = SQLiteMemoryStore(path)
    await graph.user_entity()  # connect first
    pending = asyncio.create_task(graph.add_statement(new("User likes slow things")))
    await slow.entered.wait()
    assert not pending.done()  # waiting on the embedder
    # Other connections, and this store's own reads, are not blocked meanwhile.
    await asyncio.wait_for(memory.append_episode(UserMessage(session_id="s", text="x")), 2)
    await asyncio.wait_for(other.add_statement(new("User likes fast things")), 2)
    await asyncio.wait_for(graph.list_statements(), 2)
    slow.gate.set()
    await asyncio.wait_for(pending, 2)
    assert len(await graph.list_statements()) == 2
    for store in (memory, graph, other):
        await store.close()


async def test_store_reconnects_after_close(path: Path, clock: Clock):
    graph = SQLiteGraphStore(path, HashingEmbedder(), clock=clock)
    await graph.add_statement(new("User likes tea"))
    await graph.close()
    await graph.close()
    assert len(await graph.similar_statements("tea")) == 1  # extension loaded again
    await graph.close()


async def test_forget_from_one_store_is_seen_by_another(
    path: Path, clock: Clock, memory: SQLiteMemoryStore
):
    e = (await turn(memory, "t", "I like tea", "ok"))[0]
    async with (
        SQLiteGraphStore(path, HashingEmbedder(), clock=clock) as a,
        SQLiteGraphStore(path, HashingEmbedder(), clock=clock) as b,
    ):
        s = await a.add_statement(new("User likes tea"), [e])
        await b.forget_statements([s.id])
        assert await a.get_statement(s.id) is None
        assert await a.similar_statements("tea") == []
        clock.advance(days=1)
        assert await a.unextracted_episodes() == []
