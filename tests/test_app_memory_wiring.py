"""App wiring of the learning loop: reflection, profile refresh, clock, settings, cleanup."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from lucy.core.config import LearningSettings, RetrievalSettings
from lucy.core.jobs import Job, JobStatus
from lucy.memory.decay import DecayPolicy
from lucy.memory.extraction import EXTRACT_JOB_KIND
from lucy.memory.maintenance import DECAY_JOB, PROFILE_JOB, request_profile
from lucy.memory.models import Origin, StatementKind
from lucy.memory.reflection import REFLECT_JOB
from lucy.memory.retrieval import VECTOR_MIN_SIMILARITY
from lucy.memory.sqlite_jobs import SQLiteJobStore
from tests.graph_support import Clock, new
from tests.learning_support import Models, eventually, make_app, reply, run_turn
from tests.memory_extraction_support import cand, extraction


async def idle(app) -> bool:  # type: ignore[no-untyped-def]
    counts = await app.jobs.counts()
    return counts[JobStatus.PENDING] == counts[JobStatus.RUNNING] == 0


def fact(text: str, value: str) -> list:
    return extraction(cand(text, predicate=text, value=value))


async def test_many_quick_turns_give_one_profile_rewrite(tmp_path: Path) -> None:
    """Each finished extraction pushes the pending rewrite back; it runs once, after the
    user has been quiet for `profile_delay`, from everything extracted."""
    texts = ["The user lives in Leeds", "The user has a cat", "The user plays the cello"]
    models = Models(
        chat=[reply("OK.")] * 3,
        extraction=[fact(t, t.split()[-1]) for t in texts],
        summary=[reply("The user lives in Leeds, has a cat and plays the cello.")],
    )
    async with make_app(tmp_path / "lucy.db", models, profile_delay=2.0) as app:
        assert app.jobs is not None and app.graph is not None
        jobs = app.jobs
        await eventually(lambda: idle(app))  # startup maintenance (empty graph)
        for text in texts:
            await run_turn(app, text + ".")

        async def waiting_rewrite() -> bool:
            if len(await app.graph.list_statements()) < 3:  # type: ignore[union-attr]
                return False
            if await jobs.list_jobs(statuses=[JobStatus.RUNNING]):
                return False
            waiting = await jobs.list_jobs(statuses=[JobStatus.PENDING])
            return [j.kind for j in waiting] == [PROFILE_JOB]

        await eventually(waiting_rewrite)
        [pending] = await jobs.list_jobs(statuses=[JobStatus.PENDING])
        assert pending.run_after > datetime.now(UTC)  # waiting for quiet
        assert models.summary_llm.calls == []
        await eventually(lambda: idle(app), timeout=10)
        versions = await app.graph.profile_versions()
    assert [v.content for v in versions] == [
        "The user lives in Leeds, has a cat and plays the cello."
    ]
    assert len(models.summary_llm.calls) == 1
    # The rewrite saw all three statements.
    assert all(t in models.summary_llm.calls[0][1].content for t in texts)


async def test_no_profile_request_while_extraction_is_still_waiting(tmp_path: Path) -> None:
    models = Models()
    async with make_app(tmp_path / "lucy.db", models, background=False) as app:
        assert app.jobs is not None
        handler = app.job_handlers[EXTRACT_JOB_KIND]
        waiting = await app.jobs.enqueue(EXTRACT_JOB_KIND, {}, dedupe_key=EXTRACT_JOB_KIND)
        running = Job(
            id=999,
            kind=EXTRACT_JOB_KIND,
            payload={},
            status=JobStatus.RUNNING,
            attempts=1,
            max_attempts=5,
            run_after=waiting.run_after,
            created_at=waiting.created_at,
            updated_at=waiting.updated_at,
        )
        await handler(running)  # nothing to extract; another extraction job is pending
        assert [j.kind for j in await app.jobs.list_jobs()] == [EXTRACT_JOB_KIND]
        claimed = await app.jobs.claim(datetime.now(UTC))
        assert claimed is not None
        await handler(claimed)  # the backlog is drained now
        kinds = [j.kind for j in await app.jobs.list_jobs(statuses=[JobStatus.PENDING])]
        assert kinds == [PROFILE_JOB]


async def test_reflection_is_registered_and_runs(tmp_path: Path) -> None:
    async with make_app(tmp_path / "lucy.db", Models(), background=False) as app:
        assert app.jobs is not None
        assert REFLECT_JOB in app.job_handlers
        job = await app.jobs.enqueue(REFLECT_JOB, {}, dedupe_key=REFLECT_JOB)
        claimed = await app.jobs.claim(datetime.now(UTC))
        assert claimed is not None and claimed.id == job.id
        await app.job_handlers[REFLECT_JOB](claimed)  # too few messages: no model call


async def test_jobs_that_failed_for_want_of_a_handler_are_requeued_once(tmp_path: Path) -> None:
    db = tmp_path / "lucy.db"
    error = f"no handler registered for job kind {REFLECT_JOB!r}"
    async with SQLiteJobStore(db) as jobs:
        ids = []
        for _ in range(2):
            job = await jobs.enqueue(REFLECT_JOB, {}, dedupe_key=REFLECT_JOB)
            await jobs.claim(datetime.now(UTC))
            await jobs.fail(job.id, error, None)
            ids.append(job.id)
        other = await jobs.enqueue("memory.unknown", {})
        await jobs.claim(datetime.now(UTC))
        await jobs.fail(other.id, "no handler registered for job kind 'memory.unknown'", None)
    async with make_app(db, Models(), background=False) as app:
        assert app.jobs is not None
        remaining = {j.id: j for j in await app.jobs.list_jobs()}
        assert ids[0] not in remaining  # the older duplicate is gone
        revived = remaining[ids[1]]
        assert revived.status is JobStatus.PENDING and revived.attempts == 0
        assert revived.last_error is None
        assert remaining[other.id].status is JobStatus.FAILED  # still no handler
    async with make_app(db, Models(), background=False) as again:  # nothing more to do
        assert again.jobs is not None
        assert len(await again.jobs.list_jobs()) == 2


async def test_clock_drives_the_memory_components(tmp_path: Path) -> None:
    clock = Clock(datetime(2025, 3, 3, 9, 0, tzinfo=UTC))
    models = Models(
        chat=[reply("OK.")],
        extraction=[fact("The user lives in Leeds", "Leeds")],
        summary=[reply("The user lives in Leeds.")],
    )
    async with make_app(tmp_path / "lucy.db", models, background=False, clock=clock) as app:
        assert app.graph is not None and app.jobs is not None and app.clock is clock
        await run_turn(app, "I live in Leeds.")
        await app.schedule_maintenance()
        assert app.runner is not None
        await app.runner.run_until_idle()
        [statement] = await app.graph.list_statements()
        assert statement.recorded_at == clock.now
        profile = await app.graph.current_profile()
        assert profile is not None and profile.created_at == clock.now
        when = await app.graph.get_meta(f"maintenance.enqueued.{DECAY_JOB}")
        assert when == clock.now.isoformat()
        # The extraction prompt dated "today" by the simulated clock.
        assert "Today: 2025-03-03" in models.extraction_llm.calls[0][1].content


async def test_retrieval_and_decay_settings_are_applied(tmp_path: Path) -> None:
    clock = Clock(datetime(2026, 1, 1, tzinfo=UTC))
    retrieval = RetrievalSettings(vector_min_similarity=0.4, max_statements=3)
    learning = LearningSettings(decay_grace_days=0, decay_half_life_days=10)
    app = make_app(
        tmp_path / "lucy.db",
        Models(),
        background=False,
        clock=clock,
        retrieval=retrieval,
        learning=learning,
    )
    async with app:
        assert app.retriever is not None and app.graph is not None and app.jobs is not None
        assert (app.retriever.vector_min_similarity, app.retriever.max_statements) == (0.4, 3)
        assert app.retriever.entity_min_similarity == 0.75  # not given: the default
        statement = await app.graph.add_statement(
            new(
                "User likes jazz",
                kind=StatementKind.PREFERENCE,
                origin=Origin.INFERRED,
                confidence=0.8,
                observed_at=clock.now,
            )
        )
        clock.advance(days=10)
        job = await app.jobs.enqueue(DECAY_JOB, {})
        await app.job_handlers[DECAY_JOB](job)
        after = await app.graph.get_statement(statement.id)
        assert after is not None and abs(after.confidence - 0.4) < 1e-9  # one half-life
    defaults = make_app(
        tmp_path / "other.db", Models(), background=False, retrieval=RetrievalSettings()
    )
    async with defaults:
        assert defaults.retriever is not None
        assert defaults.retriever.vector_min_similarity == VECTOR_MIN_SIMILARITY
    assert DecayPolicy().grace == timedelta(days=30)


async def test_profile_requests_are_debounced(tmp_path: Path) -> None:
    async with SQLiteJobStore(tmp_path / "lucy.db") as jobs:
        first = await request_profile(jobs, delay=60)
        again = await request_profile(jobs, delay=120)
        assert again.id == first.id
        stored = await jobs.get(first.id)
        assert stored is not None and stored.run_after > first.run_after  # pushed back
        earlier = await request_profile(jobs, delay=0)
        stored_again = await jobs.get(first.id)
        assert earlier.id == first.id and stored_again == stored  # never brought forward
