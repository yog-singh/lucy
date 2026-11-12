"""Per-turn retrieval over the knowledge graph."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest

from lucy.core.types import MemoryContext
from lucy.memory.graph_store import SQLiteGraphStore
from lucy.memory.models import USER_ENTITY_ID, Origin, Statement, StatementKind
from lucy.memory.retrieval import UNCERTAIN_MARKER, Retriever, render
from lucy.memory.sqlite_store import SQLiteMemoryStore
from tests.graph_support import HASHING_RETRIEVAL, T0, Clock, HashingEmbedder, new


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(synonyms={"espresso": "coffee", "cappuccino": "coffee"})


@pytest.fixture
async def graph(
    tmp_path: Path, embedder: HashingEmbedder, clock: Clock
) -> AsyncIterator[SQLiteGraphStore]:
    async with SQLiteGraphStore(tmp_path / "lucy.db", embedder, clock=clock) as g:
        yield g


@pytest.fixture
def retriever(graph: SQLiteGraphStore, embedder: HashingEmbedder, clock: Clock) -> Retriever:
    return Retriever(graph, embedder, clock=clock, **HASHING_RETRIEVAL)


DISTRACTORS = [
    "User owns a red bicycle",
    "User works as a software engineer",
    "User reads science fiction novels",
    "User was born in Nagpur",
]


async def add_distractors(graph: SQLiteGraphStore) -> None:
    await graph.add_statements([(new(text), []) for text in DISTRACTORS])


def words(text: str) -> int:
    return len(text.split())


# --- Always-present sections and degenerate inputs ----------------------------


async def test_empty_graph(retriever: Retriever):
    assert await retriever.retrieve("what do you know about me?", 500) == MemoryContext()


async def test_profile_and_instructions_are_always_included_and_not_budgeted(
    graph: SQLiteGraphStore, retriever: Retriever
):
    await graph.save_profile("old profile")
    await graph.save_profile("Yogesh, an engineer in Pune.")
    urls = await graph.add_statement(
        new("Don't read URLs aloud", kind=StatementKind.INSTRUCTION, confidence=1)
    )
    emoji = await graph.add_statement(new("Never use emoji", kind=StatementKind.INSTRUCTION))
    stale = await graph.add_statement(new("Speak slowly", kind=StatementKind.INSTRUCTION))
    await graph.invalidate(stale.id)
    await graph.add_statement(new("User reads URLs on screen"))

    expected = ["Don't read URLs aloud", "Never use emoji"]
    for budget in (0, 1, -5):
        context = await retriever.retrieve("read the URLs", budget)
        assert context.profile == "Yogesh, an engineer in Pune."
        assert context.instructions == expected
        assert context.statements == []
    context = await retriever.retrieve("read the URLs", 1000)
    assert context.instructions == expected
    assert context.statements == ["User reads URLs on screen"]  # instructions not repeated
    assert urls.id != emoji.id


async def test_blank_query_skips_search(
    graph: SQLiteGraphStore, retriever: Retriever, embedder: HashingEmbedder
):
    await graph.add_statement(new("User likes tea"))
    embedder.calls.clear()
    assert await retriever.retrieve("   \n", 500) == MemoryContext()
    assert embedder.calls == []


# --- The three channels -------------------------------------------------------


async def test_vector_channel_finds_meaning_without_shared_words(
    graph: SQLiteGraphStore, retriever: Retriever, embedder: HashingEmbedder
):
    await add_distractors(graph)
    await graph.add_statement(new("User drinks coffee every morning"))
    query = "espresso?"
    assert await graph.keyword_statements(query) == []  # keyword search cannot find it
    context = await retriever.retrieve(query, 500)
    assert context.statements == ["User drinks coffee every morning"]
    assert (["espresso?"], "query") in embedder.calls


async def test_keyword_channel_finds_word_forms_vectors_miss(
    graph: SQLiteGraphStore, retriever: Retriever, embedder: HashingEmbedder
):
    await add_distractors(graph)
    await graph.add_statement(new("User runs marathons"))
    hits = await graph.similar_statements("running", kind="query")
    assert all(h.score < retriever.vector_min_similarity for h in hits)  # vectors miss it
    context = await retriever.retrieve("running", 500)
    assert context.statements == ["User runs marathons"]


async def test_entity_channel_expands_named_entities(graph: SQLiteGraphStore, retriever: Retriever):
    await add_distractors(graph)
    priya = await graph.create_entity("person", "Priya", aliases=["my sister"])
    await graph.add_statement(new("She is an architect", subject_id=priya.id))
    await graph.add_statement(new("User visits her monthly", object_id=priya.id))
    query = "how is my sister?"
    assert await graph.keyword_statements(query) == []
    assert await graph.similar_statements(query, kind="query", k=1) != []
    assert all(
        h.score < retriever.vector_min_similarity
        for h in await graph.similar_statements(query, kind="query")
    )
    context = await retriever.retrieve(query, 500)
    assert set(context.statements) == {"She is an architect", "User visits her monthly"}


async def test_entity_channel_matches_entities_by_vector(
    graph: SQLiteGraphStore, retriever: Retriever
):
    await add_distractors(graph)
    lucy = await graph.create_entity("project", "Lucy", summary="voice assistant project")
    await graph.add_statement(new("Ships in January", subject_id=lucy.id, kind=StatementKind.GOAL))
    context = await retriever.retrieve("voice assistant progress", 500)
    assert context.statements == ["Ships in January"]


async def test_user_entity_is_not_expanded_by_default(
    graph: SQLiteGraphStore, embedder: HashingEmbedder, clock: Clock
):
    await add_distractors(graph)
    await graph.rename_entity(USER_ENTITY_ID, "Yogesh")
    plain = Retriever(graph, embedder, clock=clock, **HASHING_RETRIEVAL)
    assert (await plain.retrieve("what about Yogesh", 500)).statements == []
    expanding = Retriever(graph, embedder, clock=clock, expand_user=True, **HASHING_RETRIEVAL)
    assert set((await expanding.retrieve("what about Yogesh", 500)).statements) == set(DISTRACTORS)


async def test_invalidated_statements_are_excluded(graph: SQLiteGraphStore, retriever: Retriever):
    tea = await graph.add_statement(new("User drinks black tea", predicate="drinks"))
    await graph.supersede(tea.id, new("User drinks green tea", predicate="drinks"))
    gone = await graph.add_statement(new("User drinks tea at noon"))
    await graph.invalidate(gone.id)
    context = await retriever.retrieve("tea", 500)
    assert context.statements == ["User drinks green tea"]


# --- Fusion and weighting -----------------------------------------------------


def statement(
    sid: int,
    *,
    text: str | None = None,
    subject: int = 1,
    predicate: str = "p",
    confidence: float = 0.9,
    origin: Origin = Origin.STATED,
    age: timedelta = timedelta(0),
) -> Statement:
    return Statement(
        id=sid,
        subject_id=subject,
        predicate=predicate,
        text=text or f"statement {sid}",
        kind=StatementKind.FACT,
        origin=origin,
        confidence=confidence,
        evidence_count=1,
        recorded_at=T0 - age,
        last_observed_at=T0 - age,
    )


def order(retriever: Retriever, lists: list[list[Statement]]) -> list[int]:
    return [r.statement.id for r in retriever.fuse(lists)]


def test_fusion_rewards_agreement_between_channels(retriever: Retriever):
    a, b, c = statement(1, predicate="a"), statement(2, predicate="b"), statement(3)
    # b is second in two lists; a is first in one only.
    assert order(retriever, [[a, b], [c, b]]) == [2, 1, 3]


def test_confidence_breaks_equal_relevance(retriever: Retriever):
    strong = statement(1, predicate="x", confidence=0.95)
    weak = statement(2, predicate="y", confidence=0.3)
    assert order(retriever, [[weak, strong], [strong, weak]]) == [1, 2]
    assert order(retriever, [[strong, weak], [weak, strong]]) == [1, 2]


def test_recency_breaks_equal_relevance(retriever: Retriever):
    fresh = statement(1, predicate="x")
    stale = statement(2, predicate="y", age=timedelta(days=365))
    assert order(retriever, [[stale, fresh], [fresh, stale]]) == [1, 2]


def test_stated_outranks_inferred_on_same_subject_and_predicate(retriever: Retriever):
    inferred = statement(
        1, predicate="Prefers", confidence=0.99, origin=Origin.INFERRED, text="likes short"
    )
    stated = statement(2, predicate="prefers ", confidence=0.6, text="likes long")
    other = statement(3, predicate="other", origin=Origin.INFERRED, confidence=0.99)
    assert order(retriever, [[inferred, other, stated]]) == [3, 2, 1]
    # On a different subject, the inference keeps its higher rank.
    elsewhere = statement(4, subject=7, predicate="prefers", origin=Origin.INFERRED)
    assert order(retriever, [[elsewhere, stated]]) == [4, 2]


def test_duplicates_are_removed(retriever: Retriever):
    a = statement(1, text="User likes tea")
    b = statement(2, text="user likes  TEA", predicate="q", confidence=0.4)
    assert order(retriever, [[a, a, b], [b, a]]) == [1]


async def test_stated_outranks_inferred_end_to_end(graph: SQLiteGraphStore, retriever: Retriever):
    inferred = new(
        "User prefers short answers about short topics",
        predicate="prefers answer length",
        origin=Origin.INFERRED,
        confidence=0.95,
    )
    stated = new("User prefers detailed answers", predicate="prefers answer length", confidence=0.8)
    await graph.add_statements([(inferred, []), (stated, [])])
    ranked = [r.statement.text for r in await retriever.rank("short answers")]
    assert ranked == [stated.text, inferred.text]


async def test_recency_and_confidence_end_to_end(
    graph: SQLiteGraphStore, retriever: Retriever, clock: Clock
):
    await graph.add_statement(new("User plays chess online", observed_at=T0 - timedelta(days=400)))
    await graph.add_statement(new("User plays chess at the club"))
    await graph.add_statement(new("User plays chess daily", origin=Origin.INFERRED, confidence=0.2))
    texts = (await retriever.retrieve("chess", 500)).statements
    assert texts[0] == "User plays chess at the club"
    assert texts[-1].startswith("User plays chess daily")


# --- Packing and rendering ----------------------------------------------------


async def test_budget_packing_in_rank_order(graph: SQLiteGraphStore, embedder: HashingEmbedder):
    retriever = Retriever(graph, embedder, count_tokens=words, **HASHING_RETRIEVAL)
    await graph.add_statement(new("User likes tea", confidence=1.0))
    await graph.add_statement(
        new("User likes tea with a great deal of milk and sugar in it", confidence=0.9)
    )
    await graph.add_statement(new("User likes tea hot", confidence=0.5))
    ranked = [r.statement.text for r in await retriever.rank("likes tea")]
    assert len(ranked) == 3 and ranked[0] == "User likes tea"
    long = next(t for t in ranked if "milk" in t)
    short = next(t for t in ranked if t.endswith("hot"))
    # "- " counts as a word: 4 + 5 = 9 tokens fit, the 14-token one does not.
    context = await retriever.retrieve("likes tea", 9)
    assert context.statements == ["User likes tea", short]
    context = await retriever.retrieve("likes tea", 4)
    assert context.statements == ["User likes tea"]
    context = await retriever.retrieve("likes tea", 100)
    assert context.statements == ranked and long in ranked


def test_render_marks_uncertain_inferences_and_dates():
    base = statement(1, text="Goes to the gym  on Mondays")
    now = T0
    assert render(base, now=now) == "Goes to the gym on Mondays"
    weak = base.model_copy(update={"origin": Origin.INFERRED, "confidence": 0.4})
    assert render(weak, now=now) == f"Goes to the gym on Mondays [{UNCERTAIN_MARKER}]"
    firm = weak.model_copy(update={"confidence": 0.8})
    assert render(firm, now=now) == "Goes to the gym on Mondays"
    assert render(weak, now=now, uncertain_below=0.3) == "Goes to the gym on Mondays"
    shaky_but_stated = base.model_copy(update={"confidence": 0.1})
    assert render(shaky_but_stated, now=now) == "Goes to the gym on Mondays"
    dated = weak.model_copy(
        update={"valid_from": T0 - timedelta(days=10), "invalid_at": T0 + timedelta(days=20)}
    )
    assert render(dated, now=now) == (
        f"Goes to the gym on Mondays [{UNCERTAIN_MARKER}; since 2026-11-02, until 2026-12-02]"
    )
    future = base.model_copy(update={"valid_from": T0 + timedelta(days=30)})
    assert render(future, now=now) == "Goes to the gym on Mondays [from 2026-12-12]"


async def test_uncertain_inference_is_marked_in_context(
    graph: SQLiteGraphStore, retriever: Retriever
):
    await graph.add_statement(
        new("User skips lunch on Fridays", origin=Origin.INFERRED, confidence=0.35)
    )
    context = await retriever.retrieve("lunch on Friday", 500)
    assert context.statements == [f"User skips lunch on Fridays [{UNCERTAIN_MARKER}]"]


# --- Robustness ---------------------------------------------------------------


HOSTILE = [
    '"',
    "'",
    "*",
    "tea*",
    "NOT tea",
    "tea AND",
    "OR OR OR",
    "NEAR(tea coffee)",
    "text:tea",
    "(((",
    ")",
    "^",
    "-tea",
    "tea; DROP TABLE statements; --",
    "\x00\x01",
    "😀😀",
    "ﬁ ligature",
    "a" * 10_000,
    "\n\t",
    "%_%",
    "\\",
]


@pytest.mark.parametrize("query", HOSTILE)
async def test_hostile_queries_do_not_raise(
    graph: SQLiteGraphStore, retriever: Retriever, query: str
):
    await graph.create_entity("thing", "tea")
    await graph.add_statement(new("User likes tea"))
    context = await retriever.retrieve(query, 500)
    assert all(s == "User likes tea" for s in context.statements)


async def test_hostile_query_keeps_literal_meaning(graph: SQLiteGraphStore, retriever: Retriever):
    await graph.add_statement(new("User likes tea"))
    # "NOT" is a word here, not an operator excluding tea.
    assert (await retriever.retrieve("NOT tea", 500)).statements == ["User likes tea"]


async def test_failing_embedder_degrades_to_keyword_and_names(
    graph: SQLiteGraphStore,
    retriever: Retriever,
    embedder: HashingEmbedder,
    caplog: pytest.LogCaptureFixture,
):
    priya = await graph.create_entity("person", "Priya", aliases=["my sister"])
    await graph.add_statement(new("She is an architect", subject_id=priya.id))
    await graph.add_statement(new("User drinks coffee every morning"))
    await graph.save_profile("profile")
    embedder.fail = True
    with caplog.at_level(logging.WARNING, logger="lucy.memory.retrieval"):
        context = await retriever.retrieve("coffee for my sister", 500)
    assert set(context.statements) == {"User drinks coffee every morning", "She is an architect"}
    assert context.profile == "profile"
    assert "query embedding failed" in caplog.text
    assert (await retriever.retrieve("espresso", 500)).statements == []  # vector-only: lost


async def test_wrong_dimension_query_vector_degrades(
    graph: SQLiteGraphStore, retriever: Retriever, embedder: HashingEmbedder
):
    await graph.add_statement(new("User likes tea"))

    async def short(texts, *, kind="document"):  # type: ignore[no-untyped-def]
        return [[1.0] for _ in texts]

    embedder.embed = short  # type: ignore[method-assign]
    assert (await retriever.retrieve("tea", 500)).statements == ["User likes tea"]


# --- Gating (the thresholds measured for the real embedder) ----------------------


def test_defaults_are_the_measured_values(graph: SQLiteGraphStore):
    r = Retriever(graph, graph.embedder)
    assert (r.vector_min_similarity, r.vector_margin) == (0.68, 0.06)
    assert (r.keyword_min_similarity, r.entity_min_similarity, r.max_statements) == (0.62, 0.75, 8)


async def test_vector_margin_keeps_only_matches_near_the_best(
    graph: SQLiteGraphStore, embedder: HashingEmbedder, clock: Clock
):
    # No shared keyword with the query: vectors only (espresso is a synonym of coffee).
    await graph.add_statement(new("User drinks coffee daily"))  # query cosine 0.50
    await graph.add_statement(new("User drinks coffee or juice or water or milk"))  # 0.41
    loose = Retriever(graph, embedder, clock=clock, **HASHING_RETRIEVAL)
    assert len((await loose.retrieve("espresso", 500)).statements) == 2
    params = HASHING_RETRIEVAL | {"vector_margin": 0.05}
    strict = Retriever(graph, embedder, clock=clock, **params)
    assert (await strict.retrieve("espresso", 500)).statements == ["User drinks coffee daily"]


async def test_absolute_floor_drops_weak_matches(
    graph: SQLiteGraphStore, embedder: HashingEmbedder, clock: Clock
):
    await graph.add_statement(new("User likes tea and cake and biscuits and jam"))
    params = HASHING_RETRIEVAL | {"vector_min_similarity": 0.6}
    strict = Retriever(graph, embedder, clock=clock, **params)
    assert await graph.keyword_statements("tea") != []
    hits = await graph.similar_statements("tea", kind="query")
    assert hits[0].score < 0.6
    # The keyword channel still finds it (no keyword similarity check here).
    assert (await strict.retrieve("tea", 500)).statements != []
    params["keyword_min_similarity"] = 0.6
    stricter = Retriever(graph, embedder, clock=clock, **params)
    assert (await stricter.retrieve("tea", 500)).statements == []


async def test_keyword_hits_need_vector_support(
    graph: SQLiteGraphStore, embedder: HashingEmbedder, clock: Clock
):
    """A shared common word ("works") is not relevance: keyword hits whose vector
    similarity is below `keyword_min_similarity` are dropped."""
    await graph.add_statement(new("Dan works as a paramedic in Glasgow every weekend"))
    params = HASHING_RETRIEVAL | {"vector_min_similarity": 0.9, "keyword_min_similarity": 0.5}
    retriever = Retriever(graph, embedder, clock=clock, **params)
    assert await graph.keyword_statements("photosynthesis works") != []
    assert (await retriever.retrieve("photosynthesis works", 500)).statements == []
    params["keyword_min_similarity"] = None
    unchecked = Retriever(graph, embedder, clock=clock, **params)
    assert (await unchecked.retrieve("photosynthesis works", 500)).statements != []


async def test_keyword_hits_are_kept_without_a_query_vector(
    graph: SQLiteGraphStore, embedder: HashingEmbedder, clock: Clock
):
    await graph.add_statement(new("User runs marathons"))
    params = HASHING_RETRIEVAL | {"keyword_min_similarity": 0.99}
    retriever = Retriever(graph, embedder, clock=clock, **params)
    embedder.fail = True
    assert (await retriever.retrieve("marathons", 500)).statements == ["User runs marathons"]


async def test_stopword_only_queries_skip_the_keyword_channel(
    graph: SQLiteGraphStore, retriever: Retriever, embedder: HashingEmbedder
):
    await graph.add_statement(new("You are what you were"))
    embedder.fail = True  # keyword channel only
    assert await graph.keyword_statements("how are you") != []  # the store falls back
    assert (await retriever.retrieve("how are you", 500)).statements == []


async def test_max_statements_caps_the_context(
    graph: SQLiteGraphStore, embedder: HashingEmbedder, clock: Clock
):
    for i in range(5):
        await graph.add_statement(new(f"User drinks tea number {i}"))
    params = HASHING_RETRIEVAL | {"max_statements": 2}
    capped = Retriever(graph, embedder, clock=clock, **params)
    context = await capped.retrieve("tea", 500)
    assert len(context.statements) == 2
    ranked = await capped.rank("tea")
    assert len(ranked) == 5  # rank is not capped
    assert context.statements == [r.statement.text for r in ranked[:2]]


async def test_entity_floor(graph: SQLiteGraphStore, embedder: HashingEmbedder, clock: Clock):
    lucy = await graph.create_entity("project", "Lucy", summary="voice assistant project")
    await graph.add_statement(new("Ships in January", subject_id=lucy.id, kind=StatementKind.GOAL))
    params = HASHING_RETRIEVAL | {"entity_min_similarity": 0.95}
    strict = Retriever(graph, embedder, clock=clock, **params)
    assert (await strict.retrieve("voice assistant progress", 500)).statements == []
    # An entity named in the query is expanded whatever its similarity.
    assert (await strict.retrieve("how is Lucy going", 500)).statements == ["Ships in January"]


async def test_embedder_failure_is_logged_once_and_recovery_once(
    graph: SQLiteGraphStore,
    retriever: Retriever,
    embedder: HashingEmbedder,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.DEBUG, logger="lucy.memory.retrieval")
    await graph.add_statement(new("User likes tea"))
    embedder.fail = True
    for _ in range(3):
        assert (await retriever.retrieve("tea", 500)).statements == ["User likes tea"]
    embedder.fail = False
    for _ in range(2):
        await retriever.retrieve("tea", 500)
    embedder.fail = True
    await retriever.retrieve("tea", 500)
    records = [(r.levelno, r.getMessage()) for r in caplog.records if r.levelno >= logging.INFO]
    assert [level for level, _ in records] == [logging.WARNING, logging.INFO, logging.WARNING]
    assert "query embedding failed (RuntimeError: embedder is down)" in records[0][1]
    assert records[1][1] == "query embedding works again; retrieving with vectors"


def test_parameter_validation(graph: SQLiteGraphStore):
    with pytest.raises(ValueError, match="vector_margin"):
        Retriever(graph, graph.embedder, vector_margin=-0.1)
    with pytest.raises(ValueError, match="max_statements"):
        Retriever(graph, graph.embedder, max_statements=0)


def test_mismatched_embedder_is_rejected(graph: SQLiteGraphStore):
    with pytest.raises(ValueError, match="not comparable"):
        Retriever(graph, HashingEmbedder(model="other"))
    with pytest.raises(ValueError):
        Retriever(graph, graph.embedder, recency_half_life=timedelta(0))
    with pytest.raises(ValueError):
        Retriever(graph, graph.embedder, confidence_weight=2)


# --- The memory store port ----------------------------------------------------


async def test_memory_store_delegates_to_attached_retriever(
    tmp_path: Path, graph: SQLiteGraphStore, retriever: Retriever
):
    await graph.add_statement(new("User likes tea"))
    await graph.save_profile("profile")
    async with SQLiteMemoryStore(graph.path) as store:
        assert await store.retrieve("tea", 500) == MemoryContext()
        store.use_retriever(retriever)
        assert await store.retrieve("tea", 500) == MemoryContext(
            profile="profile", statements=["User likes tea"]
        )
        store.use_retriever(None)
        assert await store.retrieve("tea", 500) == MemoryContext()
