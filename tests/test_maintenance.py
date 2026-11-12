"""Entity summaries, duplicate merging, maintenance jobs and scheduling (maintenance.py)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest

from lucy.core.gateway import LLMGateway, Preempted, Priority
from lucy.core.jobs import JobRunner
from lucy.core.types import LLMChunk
from lucy.memory.decay import DecayPolicy
from lucy.memory.graph_store import SQLiteGraphStore
from lucy.memory.maintenance import (
    DECAY_JOB,
    MERGE_JOB,
    PROFILE_JOB,
    REEMBED_CONTINUATION_KEY,
    REEMBED_JOB,
    SCHEDULE_PREFIX,
    SUMMARIES_JOB,
    MergePolicy,
    decay_handler,
    merge_candidates,
    merge_duplicate_entities,
    merge_handler,
    profile_handler,
    reembed_handler,
    refresh_entity_summaries,
    rejected_key,
    schedule_maintenance,
    summaries_handler,
)
from lucy.memory.models import USER_ENTITY_ID, Entity, Origin, StatementKind
from lucy.memory.profile import ProfileWriter
from lucy.memory.reflection import REFLECT_JOB
from lucy.memory.sqlite_jobs import SQLiteJobStore
from tests.fakes import FakeLLM, Pause
from tests.graph_support import T0, Clock, HashingEmbedder, new

DAY = timedelta(days=1)
#: HashingEmbedder: "Priya" vs "Priya Sharma" scores 1/sqrt(2) ~ 0.71.
LOW_THRESHOLD = MergePolicy(name_similarity_threshold=0.6)


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


@pytest.fixture
async def jobs(path: Path, clock: Clock) -> AsyncIterator[SQLiteJobStore]:
    async with SQLiteJobStore(path, clock=clock) as store:
        yield store


def text(value: str) -> list[LLMChunk]:
    return [LLMChunk(text=value)]


def verdict(same: bool, reason: str = "because") -> list[LLMChunk]:
    return [LLMChunk(text=json.dumps({"reason": reason, "same": same}))]


def gateway(*responses: list[LLMChunk], role: str = "summary") -> tuple[LLMGateway, FakeLLM]:
    llm = FakeLLM(list(responses))
    return LLMGateway({role: llm, "chat": FakeLLM([text("hi")] * 5)}), llm


async def person(graph: SQLiteGraphStore, name: str, *facts: str, **kwargs: object) -> Entity:
    entity = await graph.create_entity("person", name, **kwargs)  # type: ignore[arg-type]
    for fact in facts:
        await graph.add_statement(new(fact, subject_id=entity.id))
    return entity


# --- Entity summaries -------------------------------------------------------------


async def test_summaries_are_written_from_neighbour_statements(graph: SQLiteGraphStore) -> None:
    priya = await person(graph, "Priya", "Priya lives in Pune", aliases=["my sister"])
    await graph.add_statement(new("User's sister is Priya", object_id=priya.id))
    await graph.add_statement(
        new("Never mention Priya's job", object_id=priya.id, kind=StatementKind.INSTRUCTION)
    )
    empty = await person(graph, "Nobody")
    await graph.add_statement(new("User likes tea"))  # the user entity has statements too
    gw, llm = gateway(text("- Priya is the user's **sister** and lives in Pune.\n"))
    result = await refresh_entity_summaries(gw, graph)
    assert result.written == [priya.id] and result.complete and result.flagged == {}
    summary = (await graph.get_entity(priya.id)).summary  # type: ignore[union-attr]
    assert summary == "Priya is the user's sister and lives in Pune."
    assert (await graph.get_entity(empty.id)).summary is None  # type: ignore[union-attr]
    assert (await graph.user_entity()).summary is None
    [messages] = llm.calls
    prompt = messages[1].content
    assert "Name: Priya (also called: my sister)" in prompt and "Type: person" in prompt
    assert "- Priya lives in Pune" in prompt and "Never mention" not in prompt
    # The summary is embedded: the entity is now found by it.
    hits = await graph.similar_entities("Priya sister Pune", k=1)
    assert hits[0].entity.id == priya.id


async def test_summaries_are_rewritten_only_when_statements_change(
    graph: SQLiteGraphStore,
) -> None:
    priya = await person(graph, "Priya", "Priya lives in Pune")
    rahul = await person(graph, "Rahul", "Rahul plays chess")
    gw, llm = gateway(
        text("Priya lives in Pune."),
        text("Rahul plays chess."),
        text("Priya lives in Pune and is a doctor."),
        text("Priya Sharma lives in Pune and is a doctor."),
    )
    await refresh_entity_summaries(gw, graph)
    assert len(llm.calls) == 2
    again = await refresh_entity_summaries(gw, graph)
    assert again.written == [] and len(llm.calls) == 2
    # A confidence change is not a change of content.
    [statement] = await graph.current_statements(priya.id)
    await graph.set_confidence(statement.id, 0.5)
    assert (await refresh_entity_summaries(gw, graph)).written == []
    # A new statement is.
    await graph.add_statement(new("Priya is a doctor", subject_id=priya.id))
    assert (await refresh_entity_summaries(gw, graph)).written == [priya.id]
    # So is a rename.
    await graph.rename_entity(priya.id, "Priya Sharma")
    assert (await refresh_entity_summaries(gw, graph)).written == [priya.id]
    assert rahul.id not in (await refresh_entity_summaries(gw, graph)).written


async def test_summary_is_removed_when_no_current_statements_remain(
    graph: SQLiteGraphStore, clock: Clock
) -> None:
    priya = await person(graph, "Priya", "Priya lives in Pune")
    gw, llm = gateway(text("Priya lives in Pune."))
    await refresh_entity_summaries(gw, graph)
    [statement] = await graph.current_statements(priya.id)
    await graph.invalidate(statement.id)
    clock.advance(seconds=1)
    result = await refresh_entity_summaries(gw, graph)
    assert result.cleared == [priya.id] and result.written == []
    assert (await graph.get_entity(priya.id)).summary is None  # type: ignore[union-attr]
    assert len(llm.calls) == 1


async def test_uncertain_inferences_do_not_count_for_summaries(graph: SQLiteGraphStore) -> None:
    jazz = await graph.create_entity("topic", "Jazz")
    await graph.add_statement(
        new("User may like Jazz", object_id=jazz.id, origin=Origin.INFERRED, confidence=0.3)
    )
    gw, llm = gateway()
    result = await refresh_entity_summaries(gw, graph)
    assert result.written == [] and llm.calls == []


async def test_summaries_are_bounded_and_resume(graph: SQLiteGraphStore) -> None:
    people = [await person(graph, f"P{i}", f"P{i} plays chess") for i in range(5)]
    gw, llm = gateway(*(text(f"P{i} plays chess.") for i in range(5)))
    first = await refresh_entity_summaries(gw, graph, limit=2)
    assert first.written == [people[0].id, people[1].id] and not first.complete
    second = await refresh_entity_summaries(gw, graph, limit=2)
    assert second.written == [people[2].id, people[3].id]
    third = await refresh_entity_summaries(gw, graph, limit=2)
    assert third.written == [people[4].id] and third.complete
    assert len(llm.calls) == 5


async def test_summary_invention_is_flagged_and_retried(graph: SQLiteGraphStore) -> None:
    priya = await person(graph, "Priya", "Priya lives in Pune")
    gw, llm = gateway(text("Priya is a surgeon at Ruby Hall."), text("Priya lives in Pune."))
    result = await refresh_entity_summaries(gw, graph)
    assert result.flagged == {}
    assert "Ruby, Hall" in llm.calls[1][-1].content
    assert (await graph.get_entity(priya.id)).summary == "Priya lives in Pune."  # type: ignore[union-attr]

    await graph.add_statement(new("Priya is a doctor", subject_id=priya.id))
    gw, _ = gateway(text("Priya works at Ruby Hall."), text("Priya works at Ruby Hall in 2020."))
    result = await refresh_entity_summaries(gw, graph)
    assert result.flagged == {priya.id: ["Ruby", "Hall"]}  # the attempt with fewer flags


async def test_summary_is_limited_to_two_sentences(graph: SQLiteGraphStore) -> None:
    priya = await person(graph, "Priya", "Priya lives in Pune")
    gw, _ = gateway(text("Priya lives in Pune. She lives in Pune. Pune is where Priya lives."))
    await refresh_entity_summaries(gw, graph)
    summary = (await graph.get_entity(priya.id)).summary  # type: ignore[union-attr]
    assert summary == "Priya lives in Pune. She lives in Pune."


async def test_preempted_summary_run_keeps_earlier_progress(graph: SQLiteGraphStore) -> None:
    a = await person(graph, "Asha", "Asha plays chess")
    b = await person(graph, "Bela", "Bela plays chess")
    pause = Pause()
    gw, llm = gateway(text("Asha plays chess."), [LLMChunk(text="Bela"), pause])
    task = asyncio.create_task(refresh_entity_summaries(gw, graph))
    await pause.reached.wait()
    async for _ in gw.stream("chat", [], priority=Priority.FOREGROUND):
        pass
    with pytest.raises(Preempted):
        await task
    assert (await graph.get_entity(a.id)).summary == "Asha plays chess."  # type: ignore[union-attr]
    assert (await graph.get_entity(b.id)).summary is None  # type: ignore[union-attr]
    gw, llm = gateway(text("Bela plays chess."))
    assert (await refresh_entity_summaries(gw, graph)).written == [b.id]
    assert len(llm.calls) == 1


# --- Duplicate entities -------------------------------------------------------------


async def test_merge_candidates(graph: SQLiteGraphStore) -> None:
    priya = await person(graph, "Priya")
    full = await person(graph, "Priya Sharma")
    sister = await person(graph, "my sister")
    aliased = await person(graph, "P. S.", aliases=["Priya"])
    place = await graph.create_entity("place", "Priya")  # another type
    other = await person(graph, "Rahul")
    found = await merge_candidates(graph, priya, policy=LOW_THRESHOLD)
    assert found[0].id == aliased.id  # exact name/alias matches come first
    ids = {e.id for e in found}
    assert full.id in ids and aliased.id in ids
    assert place.id not in ids and other.id not in ids and sister.id not in ids
    strict = await merge_candidates(graph, priya, policy=MergePolicy(name_similarity_threshold=0.9))
    assert [e.id for e in strict] == [aliased.id]
    # Name equal to another's alias, from the other side.
    priya = await graph.add_alias(priya.id, "my sister")
    assert sister.id in {e.id for e in await merge_candidates(graph, priya, policy=LOW_THRESHOLD)}


async def test_confirmed_duplicate_is_merged_into_the_one_with_more_statements(
    graph: SQLiteGraphStore,
) -> None:
    short = await person(graph, "Priya", "Priya lives in Pune")
    full = await person(graph, "Priya Sharma", "Priya Sharma is a doctor", "Priya Sharma is 34")
    gw, llm = gateway(verdict(True), role="extraction")
    result = await merge_duplicate_entities(gw, graph, policy=LOW_THRESHOLD)
    assert result.merged == [(short.id, full.id)] and result.asked == 1
    survivor = await graph.get_entity(short.id, resolve=True)
    assert survivor is not None and survivor.id == full.id and "Priya" in survivor.aliases
    assert len(await graph.current_statements(full.id)) == 3
    prompt = llm.calls[0][1].content
    assert 'A: name "Priya"' in prompt and 'B: name "Priya Sharma"' in prompt
    assert "- Priya lives in Pune" in prompt and "- Priya Sharma is a doctor" in prompt
    assert llm.schemas_seen[0] is not None  # structured output


async def test_merge_tie_keeps_the_older_entity(graph: SQLiteGraphStore) -> None:
    first = await person(graph, "Priya", "Priya lives in Pune")
    second = await person(graph, "Priya Sharma", "Priya Sharma is a doctor")
    gw, _ = gateway(verdict(True), role="extraction")
    result = await merge_duplicate_entities(gw, graph, policy=LOW_THRESHOLD)
    assert result.merged == [(second.id, first.id)]


async def test_rejected_pairs_are_not_asked_again_until_new_statements(
    graph: SQLiteGraphStore,
) -> None:
    sister = await person(graph, "Priya", "User's sister is Priya")
    colleague = await person(graph, "Priya", "Priya is User's colleague at Acme")
    gw, llm = gateway(verdict(False), verdict(False), role="extraction")
    result = await merge_duplicate_entities(gw, graph, policy=LOW_THRESHOLD)
    assert result.rejected == [(sister.id, colleague.id)] and result.merged == []
    assert await graph.get_meta(rejected_key(colleague.id, sister.id)) is not None
    again = await merge_duplicate_entities(gw, graph, policy=LOW_THRESHOLD)
    assert again.asked == 0 and len(llm.calls) == 1
    await graph.add_statement(new("Priya lives in Pune", subject_id=colleague.id))
    third = await merge_duplicate_entities(gw, graph, policy=LOW_THRESHOLD)
    assert third.asked == 1 and len(llm.calls) == 2
    assert (await graph.get_entity(colleague.id)).merged_into is None  # type: ignore[union-attr]


async def test_user_entity_is_never_merged_away(graph: SQLiteGraphStore) -> None:
    await graph.rename_entity(USER_ENTITY_ID, "Yogesh", keep_old_as_alias=False)
    await graph.add_statement(new("Yogesh lives in Bangalore"))
    twin = await person(
        graph, "Yogesh", "Yogesh is a programmer", "Yogesh likes tea", "Yogesh owns a bike"
    )
    gw, llm = gateway(verdict(False), role="extraction")
    assert (await merge_duplicate_entities(gw, graph)).merged == []  # asked, said no
    assert len(llm.calls) == 1
    await graph.add_statement(new("Yogesh plays chess", subject_id=twin.id))
    gw, _ = gateway(verdict(True), role="extraction")
    result = await merge_duplicate_entities(gw, graph)
    assert result.merged == [(twin.id, USER_ENTITY_ID)]  # despite having more statements
    user = await graph.user_entity()
    assert user.merged_into is None
    assert len(await graph.current_statements(USER_ENTITY_ID)) == 5


async def test_merging_is_bounded_and_resumes(graph: SQLiteGraphStore) -> None:
    pairs = []
    for name in ("Asha", "Bela", "Chitra"):
        pairs.append((await person(graph, name, f"{name} plays chess"), None))
    for name in ("Asha", "Bela", "Chitra"):
        await person(graph, name, f"{name} sings")
    gw, llm = gateway(*(verdict(False) for _ in range(3)), role="extraction")
    first = await merge_duplicate_entities(gw, graph, limit=2)
    assert first.asked == 2 and not first.complete
    second = await merge_duplicate_entities(gw, graph, limit=2)
    assert second.asked == 1 and second.complete
    third = await merge_duplicate_entities(gw, graph, limit=2)
    assert third.asked == 0 and len(llm.calls) == 3


async def test_merge_preempted_propagates(graph: SQLiteGraphStore) -> None:
    await person(graph, "Asha", "Asha plays chess")
    await person(graph, "Asha", "Asha sings")
    pause = Pause()
    gw, _ = gateway([LLMChunk(text='{"reason"'), pause], role="extraction")
    task = asyncio.create_task(merge_duplicate_entities(gw, graph))
    await pause.reached.wait()
    async for _ in gw.stream("chat", [], priority=Priority.FOREGROUND):
        pass
    with pytest.raises(Preempted):
        await task
    assert await graph.list_meta("derived.merge.rejected.") == []


# --- Job handlers -----------------------------------------------------------------


def job(store_job_kind: str, payload: dict[str, object] | None = None, key: str | None = None):  # type: ignore[no-untyped-def]
    from lucy.core.jobs import Job, JobStatus

    return Job(
        id=1,
        kind=store_job_kind,
        payload=payload or {},
        status=JobStatus.RUNNING,
        attempts=1,
        max_attempts=5,
        run_after=T0,
        created_at=T0,
        updated_at=T0,
        dedupe_key=key,
    )


async def test_decay_handler_is_idempotent(graph: SQLiteGraphStore, clock: Clock) -> None:
    s = await graph.add_statement(
        new(
            "User likes jazz",
            origin=Origin.INFERRED,
            kind=StatementKind.PREFERENCE,
            confidence=0.8,
            observed_at=T0,
        )  # fmt: skip
    )
    clock.now = T0 + 104 * DAY
    handle = decay_handler(
        graph, policy=DecayPolicy(grace=14 * DAY, half_life=90 * DAY), clock=clock
    )
    await handle(job(DECAY_JOB))
    await handle(job(DECAY_JOB))
    assert (await graph.get_statement(s.id)).confidence == pytest.approx(0.4)  # type: ignore[union-attr]


async def test_profile_handler_forces_on_request(graph: SQLiteGraphStore, clock: Clock) -> None:
    await graph.add_statement(new("User lives in Pune"))
    gw, llm = gateway(text("The user lives in Pune."), text("The user lives in Pune."))
    handle = profile_handler(ProfileWriter(gw, graph, clock=clock))
    await handle(job(PROFILE_JOB))
    await handle(job(PROFILE_JOB))  # unchanged: skipped
    assert len(llm.calls) == 1
    await handle(job(PROFILE_JOB, {"force": True}))
    assert len(llm.calls) == 2 and len(await graph.profile_versions()) == 2


async def test_summaries_and_merge_handlers_are_bounded_and_safe_to_repeat(
    graph: SQLiteGraphStore,
) -> None:
    for name in ("Asha", "Bela", "Chitra"):
        await person(graph, name, f"{name} plays chess")
    gw, llm = gateway(*(text("A person who plays chess.") for _ in range(3)))
    handle = summaries_handler(gw, graph, limit=2)
    await handle(job(SUMMARIES_JOB))
    assert len(llm.calls) == 2
    await handle(job(SUMMARIES_JOB))
    await handle(job(SUMMARIES_JOB))
    assert len(llm.calls) == 3

    await person(graph, "Asha", "Asha sings")
    gw, llm = gateway(verdict(False), role="extraction")
    handle = merge_handler(gw, graph, limit=1)
    await handle(job(MERGE_JOB))
    await handle(job(MERGE_JOB))
    assert len(llm.calls) == 1


async def test_handlers_let_preempted_through_and_the_runner_defers(
    graph: SQLiteGraphStore, jobs: SQLiteJobStore, clock: Clock
) -> None:
    await person(graph, "Asha", "Asha plays chess")
    pause = Pause()
    gw, _ = gateway([LLMChunk(text="Asha"), pause])
    runner = JobRunner(
        jobs, {SUMMARIES_JOB: summaries_handler(gw, graph)}, defer_on=(Preempted,), clock=clock
    )
    queued = await jobs.enqueue(SUMMARIES_JOB, {})
    task = asyncio.create_task(runner.run_until_idle())
    await pause.reached.wait()
    async for _ in gw.stream("chat", [], priority=Priority.FOREGROUND):
        pass
    assert await task == 0
    deferred = await jobs.get(queued.id)
    assert deferred is not None and deferred.status == "pending" and deferred.attempts == 0


async def test_reembed_handler_continues_itself_until_done(
    graph: SQLiteGraphStore, jobs: SQLiteJobStore, clock: Clock
) -> None:
    for i in range(5):
        await graph.add_statement(new(f"User fact {i}"))
    await graph.drop_vector_index()
    status = await graph.vector_status()
    assert status.statements_missing == 5 and status.entities_missing == 1
    first = await jobs.enqueue(REEMBED_JOB, {}, dedupe_key=REEMBED_JOB)
    runner = JobRunner(jobs, {REEMBED_JOB: reembed_handler(graph, jobs, batch=2)}, clock=clock)
    assert await runner.run_until_idle() == 3  # 2 + 2 + 2 vectors
    status = await graph.vector_status()
    assert status.statements_missing == 0 and status.entities_missing == 0
    second = await jobs.get(first.id + 1)
    assert second is not None and second.dedupe_key == REEMBED_CONTINUATION_KEY
    third = await jobs.get(first.id + 2)
    assert third is not None and third.dedupe_key == REEMBED_JOB
    assert await jobs.get(first.id + 3) is None
    # Nothing missing: a run is a no-op and does not continue.
    await reembed_handler(graph, jobs, batch=2)(job(REEMBED_JOB, key=REEMBED_JOB))
    assert await jobs.get(first.id + 3) is None


# --- Scheduling ------------------------------------------------------------------


async def test_schedule_enqueues_due_jobs_in_order_once_per_interval(
    graph: SQLiteGraphStore, jobs: SQLiteJobStore, clock: Clock
) -> None:
    await graph.reembed_missing()  # a fresh file's user entity has no vector yet
    first = await schedule_maintenance(jobs, graph, now=T0)
    assert [j.kind for j in first] == [
        DECAY_JOB,
        MERGE_JOB,
        REFLECT_JOB,
        SUMMARIES_JOB,
        PROFILE_JOB,
    ]
    assert all(j.dedupe_key == j.kind for j in first)
    assert await graph.get_meta(f"{SCHEDULE_PREFIX}{DECAY_JOB}") == T0.isoformat()
    # Not due again within the interval, even though the jobs are still pending.
    assert await schedule_maintenance(jobs, graph, now=T0 + timedelta(hours=23)) == []
    # Due again after it; the still-pending jobs are returned, not duplicated.
    again = await schedule_maintenance(jobs, graph, now=T0 + timedelta(hours=24))
    assert [j.id for j in again] == [j.id for j in first]
    # Claimed in order: merge, then reflection, then summaries and the profile.
    clock.now = T0 + DAY
    claimed = []
    while (j := await jobs.claim(clock.now)) is not None:
        claimed.append(j.kind)
        await jobs.complete(j.id)
    assert claimed == [DECAY_JOB, MERGE_JOB, REFLECT_JOB, SUMMARIES_JOB, PROFILE_JOB]
    later = await schedule_maintenance(jobs, graph, now=T0 + 2 * DAY)
    assert len(later) == 5 and {j.id for j in later}.isdisjoint(j.id for j in first)


async def test_schedule_uses_a_custom_interval_and_survives_a_clock_going_back(
    graph: SQLiteGraphStore, jobs: SQLiteJobStore
) -> None:
    await graph.reembed_missing()
    await schedule_maintenance(jobs, graph, now=T0, interval=timedelta(hours=6))
    assert await schedule_maintenance(jobs, graph, now=T0 + timedelta(hours=5)) == []
    six = timedelta(hours=6)
    assert len(await schedule_maintenance(jobs, graph, now=T0 + six, interval=six)) == 5
    # A recorded time in the future (clock went back) does not block scheduling forever.
    assert len(await schedule_maintenance(jobs, graph, now=T0 - DAY)) == 5


async def test_schedule_enqueues_reembed_whenever_vectors_are_missing(
    graph: SQLiteGraphStore, jobs: SQLiteJobStore
) -> None:
    first = await schedule_maintenance(jobs, graph, now=T0)
    assert [j.kind for j in first][0] == REEMBED_JOB  # the user entity has no vector yet
    await graph.reembed_missing()
    await graph.add_statement(new("User likes tea"))
    assert await schedule_maintenance(jobs, graph, now=T0 + timedelta(minutes=1)) == []
    await graph.drop_vector_index()
    soon = await schedule_maintenance(jobs, graph, now=T0 + timedelta(minutes=5))
    assert [j.kind for j in soon] == [REEMBED_JOB]
    again = await schedule_maintenance(jobs, graph, now=T0 + timedelta(minutes=6))
    assert [j.id for j in again] == [soon[0].id]  # deduplicated while pending


async def test_candidate_threshold_compares_names_not_summaries(graph: SQLiteGraphStore) -> None:
    """A long summary dilutes the stored vector; the name-to-name score is what counts."""
    priya = await person(graph, "Priya")
    full = await person(graph, "Priya Sharma")
    await graph.update_entity_summary(
        full.id, "A paediatrician in Pune with two children who loves gardening and chess"
    )
    [hit] = [h for h in await graph.similar_entities("Priya", k=5) if h.entity.id == full.id]
    assert hit.score < 0.6  # the stored vector alone would miss it
    found = await merge_candidates(graph, priya, policy=LOW_THRESHOLD)  # name score ~0.71
    assert [e.id for e in found] == [full.id]
