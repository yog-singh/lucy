"""The app with memory learning wired in: fake models, real SQLite stores in tmp_path."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path

import pytest

from lucy.app import JobBacklog, LucyApp
from lucy.core.config import AdapterConfig, LearningSettings, LLMSettings, Settings
from lucy.core.events import TurnEnded
from lucy.core.jobs import JobStatus
from lucy.core.registry import Registry
from lucy.core.types import Role
from lucy.memory.extraction import EXTRACT_JOB_KIND
from lucy.memory.maintenance import (
    DECAY_JOB,
    MERGE_JOB,
    PROFILE_JOB,
    REEMBED_JOB,
    SUMMARIES_JOB,
)
from lucy.memory.reflection import REFLECT_JOB
from lucy.memory.retrieval import Retriever
from lucy.memory.sqlite_store import SQLiteMemoryStore
from tests.fakes import Pause, Raise
from tests.learning_support import (
    Models,
    eventually,
    make_app,
    reply,
    run_turn,
    text_of,
)
from tests.memory_extraction_support import cand, extraction

DOG = "The user has a dog named Rex"


def dog() -> list:
    return extraction(cand(DOG, predicate="has dog named", value="Rex"))


async def statements(app: LucyApp) -> list[str]:
    assert app.graph is not None
    return [s.text for s in await app.graph.list_statements()]


async def has_statement(app: LucyApp, text: str = DOG) -> bool:
    return text in await statements(app)


async def idle(app: LucyApp) -> bool:
    assert app.jobs is not None
    counts = await app.jobs.counts()
    return counts[JobStatus.PENDING] == counts[JobStatus.RUNNING] == 0


def no_other_tasks() -> bool:
    current = asyncio.current_task()
    return [t for t in asyncio.all_tasks() if t is not current] == []


async def test_learns_in_the_background_and_recalls_in_another_session(tmp_path: Path) -> None:
    models = Models(
        chat=[reply("What a good name."), reply("Your dog is called Rex.")],
        extraction=[dog(), extraction()],
        summary=[reply("The user has a dog named Rex.")],
    )
    async with make_app(tmp_path / "lucy.db", models) as app:
        assert app.learning_active and app.learning_off_reason is None
        assert isinstance(app.retriever, Retriever) and app.runner is not None
        assert set(app.job_handlers) == {
            EXTRACT_JOB_KIND,
            DECAY_JOB,
            PROFILE_JOB,
            SUMMARIES_JOB,
            MERGE_JOB,
            REFLECT_JOB,
            REEMBED_JOB,
        }
        first = await run_turn(app, "I have a dog called Rex.", session_id="a")
        assert text_of(first) == "What a good name."
        await eventually(lambda: has_statement(app))

        await eventually(lambda: idle(app))
        # The profile was rewritten once the extraction backlog drained, without a
        # maintenance run (the one scheduled at startup found an empty graph).
        assert app.graph is not None
        profile = await app.graph.current_profile()
        assert profile is not None and profile.content == "The user has a dog named Rex."

        second = await run_turn(app, "What is my dog called?", session_id="b")
        assert text_of(second) == "Your dog is called Rex."
        await eventually(lambda: idle(app))
        assert len(await app.graph.profile_versions()) == 1  # unchanged input: no rewrite
    assert no_other_tasks()

    # The second turn came from another session, so conversation replay could not
    # supply the fact: it reached the model through retrieval, in the per-turn block
    # after the user's message ("stable_prefix" layout, the default).
    *_, user, block = models.chat_llm.calls[1]
    assert user.role is Role.USER and block.role is Role.SYSTEM
    assert "Things you remember" in block.content and DOG in block.content
    assert user.content == "What is my dog called?" and len(models.chat_llm.calls[1]) == 3
    # The extraction model saw the first turn; the statement's source is its user message.
    assert "I have a dog called Rex." in models.extraction_llm.calls[0][1].content
    assert models.chat_llm.aclose_calls == 1 and models.extraction_llm.aclose_calls == 1


async def test_maintenance_is_scheduled_at_start(tmp_path: Path) -> None:
    models = Models()
    async with make_app(tmp_path / "lucy.db", models) as app:
        assert app.jobs is not None
        kinds = {job.kind for job in await app.jobs.list_jobs()}
        assert {DECAY_JOB, MERGE_JOB, REFLECT_JOB, SUMMARIES_JOB, PROFILE_JOB} <= kinds
        await eventually(lambda: idle(app))
        jobs = await app.jobs.list_jobs()
        assert all(job.status is JobStatus.DONE for job in jobs), jobs  # reflection included
        assert app.graph is not None
        assert await app.graph.get_meta(f"maintenance.enqueued.{REFLECT_JOB}") is not None
        # The profile found nothing to write, so that run does not count as done.
        assert await app.graph.get_meta(f"maintenance.enqueued.{PROFILE_JOB}") is None
    # A second start within the interval schedules only the profile again.
    async with make_app(tmp_path / "lucy.db", Models()) as again:
        assert again.jobs is not None
        pending = await again.jobs.list_jobs(statuses=[JobStatus.PENDING, JobStatus.RUNNING])
        assert [job.kind for job in pending] in ([PROFILE_JOB], [])
        await eventually(lambda: idle(again))
    assert models.summary_llm.calls == []


async def test_no_embedder_gives_the_old_behaviour(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="lucy.app")
    models = Models(chat=[reply("Hello.")])
    async with make_app(tmp_path / "lucy.db", models, embedder=False) as app:
        assert not app.learning_active
        assert app.learning_off_reason == "no [embedder] configured"
        assert app.graph is app.jobs is app.runner is app.retriever is None
        assert dict(app.job_handlers) == {}
        assert await app.pending_jobs() == 0
        assert text_of(await run_turn(app, "Hi.")) == "Hello."
    assert app.backlog is None
    lines = [r.getMessage() for r in caplog.records if "learning is off" in r.getMessage()]
    assert lines == [
        "memory learning is off (no [embedder] configured): chat replays the conversation only"
    ]
    assert not any("Things you remember" in m.content for m in models.chat_llm.calls[0])
    assert no_other_tasks()


async def test_learning_disabled_gives_the_old_behaviour(tmp_path: Path) -> None:
    models = Models(chat=[reply("Hello.")])
    disabled = LearningSettings(enabled=False)
    async with make_app(tmp_path / "lucy.db", models, learning=disabled) as app:
        assert app.learning_off_reason == "disabled in the [learning] settings"
        assert app.graph is None
        assert text_of(await run_turn(app, "Hi.")) == "Hello."
    assert models.extraction_llm.calls == []


async def test_missing_learning_role_switches_learning_off(tmp_path: Path) -> None:
    models = Models(chat=[reply("Hello.")])
    settings = Settings(
        llm=LLMSettings(roles={"chat": AdapterConfig(adapter="chat")}),
        embedder=AdapterConfig(adapter="hash"),
        memory=AdapterConfig(adapter="sqlite", options={"path": str(tmp_path / "lucy.db")}),
    )
    async with LucyApp(settings, registry=models.registry()) as app:
        assert app.learning_off_reason == "no 'extraction', 'summary' LLM role configured"
        assert text_of(await run_turn(app, "Hi.")) == "Hello."


async def test_failing_embedder_does_not_break_turns(tmp_path: Path) -> None:
    models = Models(chat=[reply("Noted."), reply("Still here.")], extraction=[dog()])
    models.embedder.fail = True
    async with make_app(tmp_path / "lucy.db", models) as app:
        assert app.learning_active
        one = await run_turn(app, "I have a dog called Rex.")
        assert text_of(one) == "Noted." and isinstance(one[-1], TurnEnded)
        assert one[-1].reason == "completed"
        assert app.jobs is not None
        jobs = app.jobs

        async def extraction_failed() -> bool:
            failed = await jobs.list_jobs(with_error=True)
            return any(j.kind == EXTRACT_JOB_KIND for j in failed)

        await eventually(extraction_failed)
        two = await run_turn(app, "Are you there?")
        assert text_of(two) == "Still here."
    assert app.backlog is not None and app.backlog.pending >= 1


async def test_hanging_embedder_cannot_stall_a_turn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    models = Models(chat=[reply("Quick answer.")])
    async with make_app(tmp_path / "lucy.db", models, retrieval_timeout=0.05) as app:
        assert app.graph is not None
        await app.graph.save_profile("The user is called Sam.")
        await eventually(lambda: idle(app))
        models.embedder.gate = asyncio.Event()  # every embedding call now waits forever
        turn = await run_turn(app, "Hello?")
        assert text_of(turn) == "Quick answer."
    # The turn still got the profile, without retrieved statements.
    assert "The user is called Sam." in models.chat_llm.calls[0][0].content
    assert any("memory retrieval failed (timed out" in r.getMessage() for r in caplog.records)


async def test_failing_job_handler_does_not_break_turns(tmp_path: Path) -> None:
    models = Models(
        chat=[reply("Noted."), reply("Fine.")],
        extraction=[[Raise(RuntimeError("model crashed"))]],
    )
    async with make_app(tmp_path / "lucy.db", models) as app:
        assert app.jobs is not None
        jobs = app.jobs
        assert text_of(await run_turn(app, "I have a dog called Rex.")) == "Noted."

        async def failed() -> bool:
            return any("model crashed" in (j.last_error or "") for j in await jobs.list_jobs())

        await eventually(failed)
        assert text_of(await run_turn(app, "And you?")) == "Fine."
    assert not await has_statement_in(tmp_path / "lucy.db")


async def has_statement_in(db: Path) -> bool:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT count(*) FROM statements").fetchone()[0] > 0


async def test_stop_during_a_job_defers_it_and_a_new_app_completes_it(tmp_path: Path) -> None:
    db = tmp_path / "lucy.db"
    pause = Pause()
    models = Models(chat=[reply("Noted.")], extraction=[[pause]])
    app = make_app(db, models)
    await app.start()
    try:
        await run_turn(app, "I have a dog called Rex.")
        async with asyncio.timeout(5):
            await pause.reached.wait()  # extraction is mid-generation
    finally:
        async with asyncio.timeout(5):
            await app.stop()
    assert app.backlog == JobBacklog(pending=1, interrupted=EXTRACT_JOB_KIND)
    assert models.extraction_llm.exits == ["cancelled"]
    assert no_other_tasks()

    later = Models(extraction=[dog()], summary=[reply("The user has a dog named Rex.")])
    async with make_app(db, later) as again:
        jobs = again.jobs
        assert jobs is not None
        [job] = [j for j in await jobs.list_jobs() if j.kind == EXTRACT_JOB_KIND]
        await eventually(lambda: has_statement(again))
        await eventually(lambda: idle(again))
        done = await jobs.get(job.id)
        assert done is not None and done.status is JobStatus.DONE
        assert done.attempts == 1  # the interrupted run was not counted


async def test_a_turn_preempts_a_background_job_which_is_retried(tmp_path: Path) -> None:
    pause = Pause()
    models = Models(
        chat=[reply("Noted."), reply("Here is my answer.")],
        extraction=[[pause, *dog()], dog(), extraction()],
        summary=[reply("The user has a dog named Rex.")],
    )
    async with make_app(tmp_path / "lucy.db", models) as app:
        assert app.jobs is not None
        jobs = app.jobs
        await run_turn(app, "I have a dog called Rex.", session_id="a")
        async with asyncio.timeout(5):
            await pause.reached.wait()  # the extraction job is mid-generation

        turn = await run_turn(app, "Tell me something.", session_id="b")
        assert text_of(turn) == "Here is my answer."
        assert isinstance(turn[-1], TurnEnded) and turn[-1].reason == "completed"

        await eventually(lambda: has_statement(app))
        await eventually(lambda: idle(app))
        [job] = [j for j in await jobs.list_jobs() if j.kind == EXTRACT_JOB_KIND]
        assert job.status is JobStatus.DONE and job.attempts == 1 and job.last_error is None
    # The pre-empted stream was stopped, then the retry and the second turn ran.
    assert models.extraction_llm.exits[0] in {"closed", "cancelled"}
    assert len(models.extraction_llm.calls) == 3


async def test_background_false_runs_nothing(tmp_path: Path) -> None:
    models = Models()
    async with make_app(tmp_path / "lucy.db", models, background=False) as app:
        assert app.learning_active and app.jobs is not None
        assert await app.jobs.counts() == dict.fromkeys(JobStatus, 0)
        assert isinstance(app.memory, SQLiteMemoryStore)
    assert no_other_tasks()


async def test_stop_is_idempotent_and_safe_after_a_failed_start(tmp_path: Path) -> None:
    models = Models()
    settings = Settings(
        llm=LLMSettings(
            roles={
                "chat": AdapterConfig(adapter="chat"),
                "extraction": AdapterConfig(adapter="extraction"),
                "summary": AdapterConfig(adapter="no_such_adapter"),
            }
        ),
        embedder=AdapterConfig(adapter="hash"),
        memory=AdapterConfig(adapter="sqlite", options={"path": str(tmp_path / "lucy.db")}),
    )
    app = LucyApp(settings, registry=models.registry())
    with pytest.raises(LookupError, match="no_such_adapter"):
        await app.start()
    await app.stop()
    await app.stop()
    assert no_other_tasks()

    ok = make_app(tmp_path / "lucy.db", Models())
    await ok.start()
    await ok.stop()
    await ok.stop()
    assert ok.graph is None and ok.jobs is None
    assert no_other_tasks()


async def test_a_learning_part_failing_to_start_leaves_plain_chat(tmp_path: Path) -> None:
    models = Models(chat=[reply("Hello.")])
    reg: Registry = models.registry()

    class BrokenEmbedder:
        def __init__(self, **options: object) -> None:
            raise ConnectionError("embedder config is wrong")

    reg._classes["embedder"]["hash"] = BrokenEmbedder
    app = make_app(tmp_path / "lucy.db", models)
    app._registry = reg
    async with app:
        assert not app.learning_active
        assert "ConnectionError: embedder config is wrong" in (app.learning_off_reason or "")
        assert text_of(await run_turn(app, "Hi.")) == "Hello."
    assert no_other_tasks()
