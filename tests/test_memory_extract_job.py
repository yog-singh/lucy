"""The `memory.extract` job: bounded batches, continuation, pre-emption, poison turns."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lucy.core.gateway import LLMGateway, Preempted
from lucy.core.jobs import Job, JobRunner, JobStatus
from lucy.core.types import LLMChunk
from lucy.memory.extraction import (
    CONTINUE_DEDUPE_KEY,
    EXTRACT_JOB_KIND,
    Extractor,
    TurnFailedError,
    extraction_handler,
    previous_failure,
    request_extraction,
)
from lucy.memory.graph_store import SQLiteGraphStore
from lucy.memory.policy import MemoryPolicy
from lucy.memory.sqlite_jobs import SQLiteJobStore
from lucy.memory.sqlite_store import SQLiteMemoryStore
from tests.fakes import FakeLLM, Raise
from tests.graph_support import Clock, HashingEmbedder
from tests.memory_extraction_support import TEST_POLICY_VALUES, cand, extraction, turn

DAY1 = datetime(2026, 11, 2, 8, 0, tzinfo=UTC)
INVALID = [LLMChunk(text="not json")]


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
async def graph(tmp_path: Path, clock: Clock) -> AsyncIterator[SQLiteGraphStore]:
    async with SQLiteGraphStore(tmp_path / "lucy.db", HashingEmbedder(), clock=clock) as g:
        yield g


@pytest.fixture
async def memory(tmp_path: Path) -> AsyncIterator[SQLiteMemoryStore]:
    async with SQLiteMemoryStore(tmp_path / "lucy.db") as m:
        yield m


@pytest.fixture
async def jobs(tmp_path: Path, clock: Clock) -> AsyncIterator[SQLiteJobStore]:
    async with SQLiteJobStore(tmp_path / "lucy.db", clock=clock) as store:
        yield store


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM([])


@pytest.fixture
def extractor(graph: SQLiteGraphStore, llm: FakeLLM, clock: Clock) -> Extractor:
    # A high floor: these tests are about the job, not about reconciliation.
    values = TEST_POLICY_VALUES | {"statement_similarity_floor": 0.9}
    policy = MemoryPolicy(**values)
    return Extractor(graph, LLMGateway({"extraction": llm}), policy=policy, clock=clock)


def script(llm: FakeLLM, *responses: list[LLMChunk]) -> None:
    llm._responses.extend(list(r) for r in responses)


def runner(jobs: SQLiteJobStore, handler: object, clock: Clock) -> JobRunner:
    return JobRunner(
        jobs,
        {EXTRACT_JOB_KIND: handler},  # type: ignore[dict-item]
        defer_on=(Preempted,),
        defer_delay=0,
        backoff=lambda attempts: 0.0,
        clock=clock,
    )


async def unextracted(graph: SQLiteGraphStore) -> list[int]:
    return [r.id for r in await graph.unextracted_episodes(100)]


async def all_jobs(jobs: SQLiteJobStore) -> list[Job]:
    found = []
    job_id = 1
    while (job := await jobs.get(job_id)) is not None:
        found.append(job)
        job_id += 1
    return found


async def test_request_extraction_is_deduplicated(jobs: SQLiteJobStore) -> None:
    first = await request_extraction(jobs)
    second = await request_extraction(jobs)
    assert first.id == second.id
    assert (first.kind, first.dedupe_key, first.payload) == (EXTRACT_JOB_KIND, EXTRACT_JOB_KIND, {})


async def test_bounded_batches_continue_until_done(
    memory: SQLiteMemoryStore,
    graph: SQLiteGraphStore,
    extractor: Extractor,
    jobs: SQLiteJobStore,
    llm: FakeLLM,
    clock: Clock,
) -> None:
    for i in range(5):
        await turn(memory, f"t{i}", f"I like thing {i}", ts=DAY1)
    script(llm, *[extraction(cand(f"The user likes thing {i}")) for i in range(5)])
    handler = extraction_handler(extractor, jobs, max_turns=2)
    await request_extraction(jobs)

    first = await jobs.claim(clock())
    assert first is not None
    await handler(first)
    # Two turns per run; the running job cannot re-enqueue under its own key.
    assert len(llm.calls) == 2
    follow = [j for j in await all_jobs(jobs) if j.status is JobStatus.PENDING]
    assert [j.dedupe_key for j in follow] == [CONTINUE_DEDUPE_KEY]
    await jobs.complete(first.id)

    assert await runner(jobs, handler, clock).run_until_idle() == 2
    assert len(llm.calls) == 5
    assert await unextracted(graph) == []
    assert len(await graph.list_statements()) == 5
    counts = await jobs.counts()
    assert counts[JobStatus.PENDING] == 0 and counts[JobStatus.DONE] == 3


async def test_no_follow_up_when_only_a_running_turn_is_left(
    memory: SQLiteMemoryStore,
    extractor: Extractor,
    jobs: SQLiteJobStore,
    llm: FakeLLM,
    clock: Clock,
) -> None:
    await turn(memory, "t1", "done turn", ts=DAY1)
    await turn(memory, "t2", "still running", ts=DAY1, ended=False)
    script(llm, extraction())
    await request_extraction(jobs)
    assert await runner(jobs, extraction_handler(extractor, jobs), clock).run_until_idle() == 1
    assert len(llm.calls) == 1
    assert (await jobs.counts())[JobStatus.PENDING] == 0


async def test_preempted_propagates_and_the_job_is_deferred(
    memory: SQLiteMemoryStore,
    graph: SQLiteGraphStore,
    extractor: Extractor,
    jobs: SQLiteJobStore,
    llm: FakeLLM,
    clock: Clock,
) -> None:
    await turn(memory, "t1", "I like jazz", ts=DAY1)
    error = Preempted("foreground started")
    script(llm, [Raise(error)])
    handler = extraction_handler(extractor, jobs)
    job = await request_extraction(jobs)
    claimed = await jobs.claim(clock())
    assert claimed is not None
    with pytest.raises(Preempted) as caught:
        await handler(claimed)
    assert caught.value is error
    assert await unextracted(graph) != []
    await jobs.defer(job.id, clock())

    # Through the runner: deferred without using up an attempt, then done.
    script(llm, [Raise(Preempted("again"))], extraction(cand("The user likes jazz")))
    jr = runner(jobs, handler, clock)
    assert await jr.run_until_idle() == 1
    done = await jobs.get(job.id)
    assert done is not None and done.status is JobStatus.DONE and done.attempts == 1
    assert await unextracted(graph) == []


async def test_a_poison_turn_is_skipped_after_repeated_failures(
    memory: SQLiteMemoryStore,
    graph: SQLiteGraphStore,
    extractor: Extractor,
    jobs: SQLiteJobStore,
    llm: FakeLLM,
    clock: Clock,
) -> None:
    poison = await turn(memory, "t1", "something odd", ts=DAY1)
    await turn(memory, "t2", "I like jazz", ts=DAY1)
    # Each failed generate call uses two responses (one retry on invalid output).
    script(llm, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID)
    script(llm, extraction(cand("The user likes jazz")))
    handler = extraction_handler(extractor, jobs, max_turn_failures=3)
    job = await request_extraction(jobs, max_attempts=5)
    assert await runner(jobs, handler, clock).run_until_idle() == 1
    done = await jobs.get(job.id)
    assert done is not None and done.status is JobStatus.DONE and done.attempts == 3
    assert done.last_error is not None and "turn turn:t1 failed 2 time(s)" in done.last_error
    assert await unextracted(graph) == []
    [statement] = await graph.list_statements()
    assert statement.text == "The user likes jazz"
    assert not set(statement.sources) & set(poison)


async def test_the_failing_turn_is_recorded_in_the_error(
    memory: SQLiteMemoryStore,
    extractor: Extractor,
    jobs: SQLiteJobStore,
    llm: FakeLLM,
    clock: Clock,
) -> None:
    await turn(memory, "t1", "something odd", ts=DAY1)
    script(llm, INVALID, INVALID)
    handler = extraction_handler(extractor, jobs)
    await request_extraction(jobs)
    job = await jobs.claim(clock())
    assert job is not None
    with pytest.raises(TurnFailedError) as caught:
        await handler(job)
    assert caught.value.key == "turn:t1" and caught.value.failures == 1
    await jobs.fail(job.id, f"TurnFailedError: {caught.value}", clock())
    again = await jobs.get(job.id)
    assert again is not None and previous_failure(again) == ("turn:t1", 1)


async def test_failures_of_different_turns_do_not_add_up(
    memory: SQLiteMemoryStore,
    graph: SQLiteGraphStore,
    extractor: Extractor,
    jobs: SQLiteJobStore,
    llm: FakeLLM,
    clock: Clock,
) -> None:
    await turn(memory, "t1", "first", ts=DAY1)
    await turn(memory, "t2", "second", ts=DAY1)
    # t1 fails once, then succeeds; t2 fails once, then succeeds: nothing skipped.
    script(llm, INVALID, INVALID, extraction(cand("The user said first")))
    script(llm, INVALID, INVALID, extraction(cand("The user said second")))
    handler = extraction_handler(extractor, jobs, max_turn_failures=2)
    await request_extraction(jobs, max_attempts=5)
    assert await runner(jobs, handler, clock).run_until_idle() == 1
    assert sorted(s.text for s in await graph.list_statements()) == [
        "The user said first",
        "The user said second",
    ]


async def test_the_last_attempt_skips_the_turn_instead_of_failing(
    memory: SQLiteMemoryStore,
    graph: SQLiteGraphStore,
    extractor: Extractor,
    jobs: SQLiteJobStore,
    llm: FakeLLM,
    clock: Clock,
) -> None:
    await turn(memory, "t1", "something odd", ts=DAY1)
    script(llm, [Raise(RuntimeError("model crashed"))])
    handler = extraction_handler(extractor, jobs, max_turn_failures=3)
    job = await request_extraction(jobs, max_attempts=1)
    assert await runner(jobs, handler, clock).run_until_idle() == 1
    done = await jobs.get(job.id)
    assert done is not None and done.status is JobStatus.DONE
    assert await unextracted(graph) == []


async def test_transient_errors_never_skip_a_turn(
    memory: SQLiteMemoryStore,
    graph: SQLiteGraphStore,
    extractor: Extractor,
    jobs: SQLiteJobStore,
    llm: FakeLLM,
    clock: Clock,
) -> None:
    await turn(memory, "t1", "I like jazz", ts=DAY1)
    script(llm, *[[Raise(ConnectionRefusedError("server down"))] for _ in range(3)])
    handler = extraction_handler(extractor, jobs, max_turn_failures=1)
    job = await request_extraction(jobs, max_attempts=3)
    assert await runner(jobs, handler, clock).run_until_idle() == 0
    failed = await jobs.get(job.id)
    assert failed is not None and failed.status is JobStatus.FAILED
    assert await unextracted(graph) != []
    # The next request starts afresh once the server is back.
    script(llm, extraction(cand("The user likes jazz")))
    await request_extraction(jobs)
    assert await runner(jobs, handler, clock).run_until_idle() == 1
    assert await unextracted(graph) == []


async def test_custom_transient_errors(
    memory: SQLiteMemoryStore,
    extractor: Extractor,
    jobs: SQLiteJobStore,
    llm: FakeLLM,
    clock: Clock,
) -> None:
    class ServerError(RuntimeError):
        pass

    await turn(memory, "t1", "I like jazz", ts=DAY1)
    script(llm, [Raise(ServerError("502"))])
    handler = extraction_handler(extractor, jobs, transient_errors=(ServerError,))
    await request_extraction(jobs)
    job = await jobs.claim(clock())
    assert job is not None
    with pytest.raises(ServerError):
        await handler(job)


def test_handler_arguments_are_checked(extractor: Extractor, jobs: SQLiteJobStore) -> None:
    with pytest.raises(ValueError):
        extraction_handler(extractor, jobs, max_turns=0)
    with pytest.raises(ValueError):
        extraction_handler(extractor, jobs, max_turn_failures=0)


async def test_empty_queue_is_a_no_op(
    extractor: Extractor, jobs: SQLiteJobStore, llm: FakeLLM, clock: Clock
) -> None:
    await request_extraction(jobs)
    assert await runner(jobs, extraction_handler(extractor, jobs), clock).run_until_idle() == 1
    assert llm.calls == []
