"""`lucy memory ...` commands against a temp database prepared through the store APIs."""

from __future__ import annotations

import asyncio
import io
import sqlite3
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lucy.app import JobBacklog, LucyApp
from lucy.clients import cli, memory_cli
from lucy.clients.cli import ChatLoop, backlog_message, memory_banner
from lucy.core.events import UserMessage
from lucy.core.jobs import JobStatus
from lucy.memory.extraction import EXTRACT_JOB_KIND, request_extraction
from lucy.memory.graph_store import SQLiteGraphStore
from lucy.memory.maintenance import PROFILE_JOB
from lucy.memory.models import USER_ENTITY_ID, Origin, StatementKind
from lucy.memory.sqlite_jobs import SQLiteJobStore
from lucy.memory.sqlite_store import SQLiteMemoryStore
from tests.fakes import Pause, Raise
from tests.graph_support import HashingEmbedder, new
from tests.learning_support import Models, make_app, reply, run_turn
from tests.memory_extraction_support import cand, decision, extraction, turn

TIMEOUT = 5
NOW = datetime.now(UTC).replace(microsecond=0)


@dataclass
class Prepared:
    db: Path
    dog: int
    job: int
    brief: int
    rex: int
    dog_turn: list[int]


async def prepare(db: Path) -> Prepared:
    """Three extracted turns in session "a", their statements, an entity and a profile."""
    async with SQLiteMemoryStore(db) as memory, SQLiteGraphStore(db, HashingEmbedder()) as graph:
        t1 = await turn(
            memory,
            "t1",
            "I have a dog called Rex.",
            "Lovely!",
            ts=NOW - timedelta(hours=3),
            session="a",
        )
        t2 = await turn(
            memory, "t2", "I work as a nurse.", "Noted.", ts=NOW - timedelta(hours=2), session="a"
        )
        t3 = await turn(
            memory,
            "t3",
            "Please answer briefly.",
            "Sure.",
            ts=NOW - timedelta(hours=1),
            session="a",
        )
        await graph.rename_entity(USER_ENTITY_ID, "Sam", keep_old_as_alias=False)
        rex = await graph.create_entity("animal", "Rex", aliases=["my dog"], summary="Sam's dog.")
        dog = await graph.add_statement(
            new("Sam has a dog named Rex", predicate="has dog", object_id=rex.id), [t1[0]]
        )
        job = await graph.add_statement(
            new("Sam works as a nurse", predicate="works as", object_value="nurse"), [t2[0]]
        )
        brief = await graph.add_statement(
            new(
                "Answer Sam briefly",
                predicate="wants answers",
                object_value="brief",
                kind=StatementKind.INSTRUCTION,
            ),
            [t3[0]],
        )
        await graph.mark_extracted(t1 + t2 + t3)
        await graph.save_profile("Sam is a nurse with a dog called Rex.")
    return Prepared(db, dog.id, job.id, brief.id, rex.id, t1)


@pytest.fixture
async def prepared(tmp_path: Path) -> Prepared:
    return await prepare(tmp_path / "lucy.db")


@pytest.fixture
async def open_app(tmp_path: Path) -> AsyncIterator[Callable[..., object]]:
    apps: list[LucyApp] = []

    async def make(models: Models | None = None, **kwargs: object) -> LucyApp:
        app = make_app(tmp_path / "lucy.db", models or Models(), background=False, **kwargs)
        await app.start()
        apps.append(app)
        return app

    yield make
    for app in apps:
        await app.stop()


async def run(command, *args, **kwargs) -> tuple[int, str]:
    out = io.StringIO()
    async with asyncio.timeout(TIMEOUT):
        code = await command(*args[:1], out, *args[1:], **kwargs)
    return code, out.getvalue()


class Answers:
    """A scripted `Confirm`: records the prompts and answers in order."""

    def __init__(self, *answers: bool) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> bool:
        self.prompts.append(prompt)
        return self.answers.pop(0)


def rows(db: Path, sql: str) -> list[tuple]:
    with sqlite3.connect(db) as conn:
        return conn.execute(sql).fetchall()


# --- not configured ------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["status"],
        ["profile"],
        ["search", "dog"],
        ["about", "Rex"],
        ["why", "1"],
        ["correct", "1", "x"],
        ["forget", "1", "--yes"],
        ["work"],
        ["rebuild", "--yes"],
    ],
)
async def test_every_command_says_when_memory_is_not_configured(open_app, argv: list[str]) -> None:
    app = await open_app(embedder=False)
    args = cli._parser().parse_args(["memory", *argv])
    out = io.StringIO()
    assert await cli.run_memory_command(app, args, out) == 2
    assert out.getvalue() == (
        "lucy memory: memory is not configured (no [embedder] configured); the knowledge"
        " graph needs an [embedder] section, the SQLite memory store and the 'extraction'"
        " and 'summary' LLM roles\n"
    )


def test_main_runs_memory_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "lucy.toml"
    config.write_text(
        '[llm.roles]\nchat = { adapter = "ollama", model = "m" }\n\n'
        f'[memory]\nadapter = "sqlite"\npath = "{tmp_path / "lucy.db"}"\n'
    )
    monkeypatch.setattr(cli, "configure_logging", lambda verbose: None)
    assert cli.main(["memory", "status", "--config", str(config)]) == 2
    assert "memory is not configured (no [embedder] configured)" in capsys.readouterr().out


# --- status, profile, search, about, why ------------------------------------------------


async def test_status(prepared: Prepared, open_app) -> None:
    app = await open_app()
    assert app.jobs is not None
    failed = await app.jobs.enqueue("memory.decay", {})
    claimed = await app.jobs.claim(datetime.now(UTC))
    assert claimed is not None and claimed.id == failed.id
    await app.jobs.fail(failed.id, "RuntimeError: disk on fire", None)
    await request_extraction(app.jobs)
    code, out = await run(memory_cli.status, app)
    assert code == 0
    assert "Episodes: 9 (0 awaiting extraction)" in out
    assert "Statements: 3 current, 0 invalidated" in out
    assert "current by kind: fact 2, instruction 1" in out
    assert "Entities: 2\n" in out
    assert "Jobs: 1 pending, 0 running, 0 done, 1 failed" in out
    assert f"failed job {failed.id} (memory.decay): RuntimeError: disk on fire" in out
    assert "Vectors missing for hash-bow (256 dimensions): 0 statements, 0 entities" in out
    assert "Profile: last written " in out and "(1 version)" in out


async def test_profile_and_history(prepared: Prepared, open_app) -> None:
    app = await open_app()
    code, out = await run(memory_cli.profile, app)
    assert code == 0 and "Sam is a nurse with a dog called Rex." in out
    assert app.graph is not None
    await app.graph.save_profile("Sam is a nurse.")
    _, out = await run(memory_cli.profile, app, history=True)
    assert out.index("Sam is a nurse.\n") < out.index("Sam is a nurse with a dog")
    assert "(current)" in out.splitlines()[0]


async def test_profile_when_none(tmp_path: Path, open_app) -> None:
    app = await open_app()
    assert (await run(memory_cli.profile, app))[1] == "No profile has been written yet.\n"


async def test_search_shows_what_a_turn_would_get(prepared: Prepared, open_app) -> None:
    app = await open_app()
    code, out = await run(memory_cli.search, app, "what is my dog called?")
    assert code == 0
    assert "Profile: version" in out
    assert f"#{prepared.brief} [instruction, stated, 0.90] Answer Sam briefly" in out
    assert "1. score " in out
    assert f"#{prepared.dog} [fact, stated, 0.90] Sam has a dog named Rex" in out
    assert "given to the turn (memory budget" in out and "(not given to the turn)" not in out


async def test_about_an_entity(prepared: Prepared, open_app) -> None:
    app = await open_app()
    code, out = await run(memory_cli.about, app, "my dog")
    assert code == 0
    assert f"#{prepared.rex} Rex (animal), also called: my dog" in out
    assert "Summary: Sam's dog." in out
    assert f"#{prepared.dog} [fact, stated, 0.90] Sam has a dog named Rex" in out
    code, out = await run(memory_cli.about, app, "Priyanka")
    assert code == 1 and "No entity is called 'Priyanka'." in out
    assert app.graph is not None
    await app.graph.rename_entity(USER_ENTITY_ID, "Sam Okoro", keep_old_as_alias=False)
    code, out = await run(memory_cli.about, app, "sam")  # a whole word of the name
    assert code == 0 and "#1 Sam Okoro (person)\n" in out
    assert "Sam works as a nurse" in out


async def test_why_shows_fields_and_sources(prepared: Prepared, open_app) -> None:
    app = await open_app()
    code, out = await run(memory_cli.why, app, prepared.dog)
    assert code == 0
    assert f"Statement #{prepared.dog}: Sam has a dog named Rex" in out
    assert "subject:    #1 Sam (person)" in out
    assert f"object:     #{prepared.rex} Rex (animal)" in out
    assert "origin:     stated" in out and "confidence: 0.90 (1 observation)" in out
    assert "valid:      from unknown, current" in out
    assert "Learned from (1 episode):" in out
    assert "session 'a'] User: I have a dog called Rex." in out
    assert (await run(memory_cli.why, app, 999))[0] == 1


# --- correct -------------------------------------------------------------------------


async def test_correct_supersedes_with_a_stated_statement(prepared: Prepared, open_app) -> None:
    app = await open_app()
    assert app.graph is not None and app.jobs is not None
    code, out = await run(memory_cli.correct, app, prepared.job, "Sam works as a paramedic")
    assert code == 0
    old = await app.graph.get_statement(prepared.job)
    assert old is not None and old.superseded_by is not None and old.invalid_at is not None
    replacement = await app.graph.get_statement(old.superseded_by)
    assert replacement is not None
    assert replacement.text == "Sam works as a paramedic"
    assert (replacement.subject_id, replacement.predicate, replacement.kind) == (
        old.subject_id,
        old.predicate,
        old.kind,
    )
    assert replacement.object_id is None and replacement.object_value is None
    assert replacement.origin is Origin.STATED
    assert replacement.confidence == memory_cli.CORRECTION_CONFIDENCE
    # The correction is a user message in the log (a complete turn), already extracted.
    [source] = replacement.sources
    [record] = await app.graph.get_episodes([source])
    message = record.event
    assert isinstance(message, UserMessage) and message.source == "correction"
    assert message.session_id == memory_cli.CORRECTIONS_SESSION
    assert message.text == (
        'Correction: what you noted, "Sam works as a nurse", is wrong. Sam works as a paramedic.'
    )
    assert record.extracted_at is not None
    assert message.turn_id is not None
    ended = await app.graph.turn_episodes(message.turn_id, types=["turn_ended"])
    assert len(ended) == 1
    assert await app.graph.unextracted_episodes() == []
    assert f"Now:  #{replacement.id} [fact, stated, 0.95] Sam works as a paramedic" in out
    [job] = await app.jobs.list_jobs()
    assert (job.kind, job.payload) == (PROFILE_JOB, {"force": True})

    # The old one says what replaced it; the new one says what it replaced.
    _, why_old = await run(memory_cli.why, app, prepared.job)
    assert f"superseded by: #{replacement.id}" in why_old and "no longer current" in why_old
    _, why_new = await run(memory_cli.why, app, replacement.id)
    assert f"supersedes:    #{prepared.job}" in why_new
    assert "session 'corrections'] User: Correction: what you noted" in why_new

    # A replaced statement cannot be corrected again; nor can a missing one.
    code, out = await run(memory_cli.correct, app, prepared.job, "Sam is a doctor")
    assert code == 1 and f"already replaced by #{replacement.id}" in out
    assert (await run(memory_cli.correct, app, 999, "x"))[0] == 1
    assert (await run(memory_cli.correct, app, prepared.dog, "   "))[0] == 2


def test_correction_text() -> None:
    assert memory_cli.correction_text("Maya lives in Lisbon.", "Maya lives in  Porto") == (
        'Correction: what you noted, "Maya lives in Lisbon", is wrong. Maya lives in Porto.'
    )
    assert memory_cli.correction_text("x", "Yes!").endswith("Yes!")


async def test_a_correction_survives_a_rebuild(tmp_path: Path, open_app) -> None:
    """Correct, rebuild from the log with a scripted extraction model: the graph ends
    up holding the corrected value, learned from the logged correction."""
    db = tmp_path / "lucy.db"
    nurse = "The user works night shifts as a nurse at Leeds General hospital"
    medic = "The user works night shifts as a paramedic at Leeds General hospital"
    async with SQLiteMemoryStore(db) as memory, SQLiteGraphStore(db, HashingEmbedder()) as graph:
        [said, *_] = await turn(
            memory, "t1", "I work nights as a nurse at Leeds General.", "OK.", ts=NOW, session="a"
        )
        wrong = await graph.add_statement(
            new(nurse, predicate="works as", object_value="nurse"), [said]
        )
        await graph.mark_extracted([said, said + 1])
    app = await open_app(
        Models(
            extraction=[
                extraction(cand(nurse, predicate="works as", value="nurse")),
                extraction(cand(medic, predicate="works as", value="paramedic")),
                decision("supersede", 1),
            ],
            summary=[reply("The user works night shifts as a paramedic.")] * 2,
        )
    )
    assert app.graph is not None
    code, _ = await run(memory_cli.correct, app, wrong.id, medic)
    assert code == 0
    assert [s.text for s in await app.graph.list_statements()] == [medic]

    code, out = await run(memory_cli.rebuild, app, confirm=Answers(True))
    assert code == 0 and "are in the episode log and are learned again" in out
    assert await app.graph.list_statements() == []
    code, out = await run(memory_cli.work, app)
    assert code == 0
    [current] = await app.graph.list_statements()
    assert current.text == medic and current.origin is Origin.STATED
    [record] = await app.graph.get_episodes(current.sources)
    assert isinstance(record.event, UserMessage) and record.event.source == "correction"
    [old] = await app.graph.statements_superseded_by(current.id)
    assert old.text == nurse and old.invalid_at is not None
    # The extraction model saw the correction as the user's own words.
    prompt = models_prompt(app)
    assert 'User: Correction: what you noted, "' + nurse + '", is wrong.' in prompt


def models_prompt(app: LucyApp) -> str:
    llm = app.gateway.for_role("extraction")
    return "\n".join(m.content for call in llm.calls for m in call)  # type: ignore[attr-defined]


def test_times_are_local_with_a_zone_and_bare_dates_are_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time as time_module

    monkeypatch.setenv("TZ", "Asia/Kolkata")
    time_module.tzset()
    try:
        midnight = datetime(2026, 3, 1, tzinfo=UTC)
        assert memory_cli._moment(midnight) == "2026-03-01"  # not 2026-03-01 05:30
        assert memory_cli._day(midnight) == "2026-03-01"
        moment = datetime(2026, 3, 1, 4, 0, tzinfo=UTC)
        assert memory_cli._when(moment) == "2026-03-01 09:30 IST"
        assert memory_cli._moment(moment) == "2026-03-01 09:30 IST"
        monkeypatch.setenv("TZ", "America/New_York")
        time_module.tzset()
        assert memory_cli._day(midnight) == "2026-03-01"  # not the evening before
    finally:
        monkeypatch.undo()
        time_module.tzset()


# --- forget --------------------------------------------------------------------------


async def test_forget_previews_and_respects_no(prepared: Prepared, open_app) -> None:
    app = await open_app()
    answers = Answers(False)
    code, out = await run(memory_cli.forget, app, [prepared.dog], confirm=answers)
    assert code == 1 and out.endswith("Nothing was deleted.\n")
    assert answers.prompts == ["Delete all of this? [y/N] "]
    assert "This will permanently delete:" in out
    assert f"#{prepared.dog} [fact, stated, 0.90] Sam has a dog named Rex" in out
    assert "3 episodes (every event of the turns they came from):" in out
    assert "User: I have a dog called Rex." in out and "Lucy: Lovely!" in out
    assert f"1 entity left with no statements: #{prepared.rex} Rex (animal)" in out
    assert "1 profile version written since" in out
    assert rows(prepared.db, "SELECT count(*) FROM statements") == [(3,)]
    assert rows(prepared.db, "SELECT count(*) FROM episodes") == [(9,)]


async def test_forget_deletes_after_confirmation(prepared: Prepared, open_app) -> None:
    app = await open_app()
    assert app.jobs is not None
    code, out = await run(memory_cli.forget, app, [prepared.dog], confirm=Answers(True))
    assert code == 0
    assert "Forgotten: 1 statement, 3 episodes, 0 traces, 1 entity, 1 profile version." in out
    assert rows(prepared.db, f"SELECT count(*) FROM statements WHERE id = {prepared.dog}") == [(0,)]
    texts = [r[0] for r in rows(prepared.db, "SELECT payload FROM episodes")]
    assert not any("Rex" in text for text in texts)
    assert rows(prepared.db, "SELECT count(*) FROM profile_versions") == [(0,)]
    [job] = await app.jobs.list_jobs(statuses=[JobStatus.PENDING])
    assert (job.kind, job.payload) == (PROFILE_JOB, {"force": True})


async def test_forget_warns_about_turns_that_repeat_the_information(
    prepared: Prepared, open_app
) -> None:
    async with SQLiteMemoryStore(prepared.db) as memory:
        await turn(memory, "t4", "Rex chewed my shoe again.", "Oh no.", ts=NOW, session="b")
    app = await open_app()
    code, out = await run(memory_cli.forget, app, [prepared.dog], confirm=Answers(False))
    assert memory_cli.LEFTOVER_NOTE in out
    assert "'Rex': 1 episode, e.g." in out and "User: Rex chewed my shoe again." in out


async def test_forget_by_query_with_yes(prepared: Prepared, open_app) -> None:
    app = await open_app()
    args = cli._parser().parse_args(["memory", "forget", "--query", "nurse", "--yes"])
    out = io.StringIO()
    assert await cli.run_memory_command(app, args, out) == 0
    text = out.getvalue()
    assert "Current statements matching 'nurse':" in text
    assert f"#{prepared.job} [fact, stated, 0.90] Sam works as a nurse" in text
    assert "Forgotten: 1 statement," in text
    remaining = [r[0] for r in rows(prepared.db, "SELECT text FROM statements")]
    assert "Sam works as a nurse" not in remaining and "Sam has a dog named Rex" in remaining


async def test_forget_with_nothing_to_forget(prepared: Prepared, open_app) -> None:
    app = await open_app()
    code, out = await run(memory_cli.forget, app, [999], confirm=Answers())
    assert code == 1 and "No such statement: #999" in out
    code, out = await run(memory_cli.forget, app, [], query="zebra", confirm=Answers())
    assert code == 1 and "No current statement matches 'zebra'." in out
    assert (await run(memory_cli.forget, app, [], confirm=Answers()))[0] == 2


# --- work and rebuild ------------------------------------------------------------------


async def test_work_drains_the_queue(tmp_path: Path, open_app) -> None:
    db = tmp_path / "lucy.db"
    async with SQLiteMemoryStore(db) as memory:
        await turn(memory, "t1", "I have a cat called Tom.", "Nice.", ts=NOW, session="a")
    models = Models(
        extraction=[extraction(cand("The user has a cat named Tom", value="Tom"))],
        summary=[reply("The user has a cat named Tom.")],
    )
    app = await open_app(models)
    code, out = await run(memory_cli.work, app)
    assert code == 0
    lines = out.splitlines()
    assert any(": done in " in line and EXTRACT_JOB_KIND in line for line in lines)
    assert any("memory.profile: done in" in line for line in lines)
    assert "Memory now: 1 current statements, 0 episodes awaiting extraction." in out
    assert app.jobs is not None
    counts = await app.jobs.counts()
    assert counts[JobStatus.PENDING] == counts[JobStatus.RUNNING] == 0
    assert app.graph is not None
    assert [s.text for s in await app.graph.list_statements()] == ["The user has a cat named Tom"]
    # The extraction ran before the profile rewrite, which therefore saw the statement.
    profile = await app.graph.current_profile()
    assert profile is not None and profile.content == "The user has a cat named Tom."


async def test_work_reports_failures_and_retries(tmp_path: Path, open_app) -> None:
    db = tmp_path / "lucy.db"
    async with SQLiteMemoryStore(db) as memory:
        await turn(memory, "t1", "I have a cat called Tom.", "Nice.", ts=NOW, session="a")
    app = await open_app(Models(extraction=[[Raise(RuntimeError("model crashed"))]]))
    code, out = await run(memory_cli.work, app)
    assert code == 0
    assert f"{EXTRACT_JOB_KIND}: failed, will retry after" in out
    assert "model crashed" in out and "1 job waiting to retry (next at" in out


async def test_work_puts_back_interrupted_jobs(tmp_path: Path, open_app) -> None:
    app = await open_app()
    assert app.jobs is not None
    await app.jobs.enqueue("memory.decay", {})
    await app.jobs.claim(datetime.now(UTC))  # left RUNNING, as by a crash
    _, out = await run(memory_cli.work, app)
    assert out.startswith("Put back 1 interrupted job.\n")
    assert "memory.decay: done in" in out


async def test_rebuild_resets_and_requests_extraction(prepared: Prepared, open_app) -> None:
    app = await open_app()
    assert app.jobs is not None
    code, out = await run(memory_cli.rebuild, app, confirm=Answers(False))
    assert code == 1 and "This deletes all 3 statements, 1 entities" in out
    assert rows(prepared.db, "SELECT count(*) FROM statements") == [(3,)]

    code, out = await run(memory_cli.rebuild, app, confirm=Answers(True))
    assert code == 0 and "Cleared 3 statements and 1 entity; 6 episodes will be re-extracted" in out
    assert rows(prepared.db, "SELECT count(*) FROM statements") == [(0,)]
    assert rows(prepared.db, "SELECT name FROM entities") == [("User",)]
    [job] = await app.jobs.list_jobs()
    assert job.kind == EXTRACT_JOB_KIND and job.status is JobStatus.PENDING


# --- chat ------------------------------------------------------------------------------


async def test_chat_banner_says_memory_is_active_and_counts_pending_jobs(
    tmp_path: Path,
) -> None:
    async with SQLiteJobStore(tmp_path / "lucy.db") as jobs:
        await jobs.enqueue("memory.unknown", {}, run_after=NOW + timedelta(days=1))
    models = Models(chat=[reply("Hi.")])
    async with make_app(tmp_path / "lucy.db", models) as app:
        err = io.StringIO()

        lines = iter(["hello"])

        async def read_line() -> str | None:
            return next(lines, None)

        chat = ChatLoop(app, "cli", read_line=read_line, out=io.StringIO(), err=err)
        async with asyncio.timeout(TIMEOUT):
            assert await chat.run() == 0
    banner = err.getvalue().splitlines()
    assert banner[1].startswith("Memory: active.")
    assert banner[2].endswith("pending; they run in the background while you chat.")


async def test_chat_banner_when_memory_is_off(tmp_path: Path) -> None:
    async with make_app(tmp_path / "lucy.db", Models(), embedder=False) as app:
        assert memory_banner(app) == (
            "Memory: off (no [embedder] configured). Lucy sees only this session's conversation."
        )


def test_backlog_message() -> None:
    assert backlog_message(None) is None
    assert backlog_message(JobBacklog(pending=0)) is None
    assert backlog_message(JobBacklog(pending=1)) == (
        "1 background job left; they resume next time, or run `lucy memory work`."
    )
    assert backlog_message(JobBacklog(pending=3, interrupted="memory.extract")) == (
        "3 background jobs left (a memory.extract job was interrupted); they resume next"
        " time, or run `lucy memory work`."
    )


async def test_exit_is_prompt_with_a_job_in_flight(tmp_path: Path) -> None:
    pause = Pause()
    models = Models(chat=[reply("Noted.")], extraction=[[pause]])
    app = make_app(tmp_path / "lucy.db", models)
    await app.start()
    await run_turn(app, "I have a cat.")
    async with asyncio.timeout(TIMEOUT):
        await pause.reached.wait()
    async with asyncio.timeout(1):  # never waits for the job to finish
        await app.stop()
    message = backlog_message(app.backlog)
    assert message is not None and "a memory.extract job was interrupted" in message


async def test_forget_query_keeps_only_close_matches(prepared: Prepared, open_app) -> None:
    app = await open_app()
    assert app.graph is not None
    # Without a shared keyword, only vector matches near the best one are kept.
    found = await memory_cli.matching_statements(app.graph, "Rex the dog", limit=5)
    assert [s.id for s in found] == [prepared.dog]
    # A query sharing nothing with any statement finds nothing.
    assert await memory_cli.matching_statements(app.graph, "zebra", limit=5) == []
