"""The SQLite job store, and one semantic suite run over it and the in-memory double."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from lucy.core.events import AssistantMessage, UserMessage
from lucy.core.jobs import Job, JobRunner, JobStateError, JobStatus, JobStore
from lucy.memory import migrations
from lucy.memory.db import apply_migrations, open_database
from lucy.memory.sqlite_jobs import SQLiteJobStore
from lucy.memory.sqlite_store import SQLiteMemoryStore
from tests.test_jobs import T0, FakeClock, InMemoryJobStore
from tests.test_sqlite_store import _trace

ACTIVE = (JobStatus.PENDING, JobStatus.RUNNING)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture(params=["memory", "sqlite"])
async def store(
    request: pytest.FixtureRequest, clock: FakeClock, tmp_path: Path
) -> AsyncIterator[Any]:
    if request.param == "memory":
        yield InMemoryJobStore(clock)
    else:
        async with SQLiteJobStore(tmp_path / "lucy.db", clock=clock) as s:
            yield s


async def get(store: Any, job_id: int) -> Job:
    job = await store.get(job_id)
    assert job is not None
    return job


async def claim_one(store: Any, clock: FakeClock) -> Job:
    job = await store.claim(clock())
    assert job is not None
    return job


async def with_status(store: Any, clock: FakeClock, status: JobStatus, key: str) -> Job:
    """A job with dedupe key `key`, driven to `status`."""
    job = await store.enqueue("k", {}, dedupe_key=key)
    if status is not JobStatus.PENDING:
        assert (await claim_one(store, clock)).id == job.id
        if status is JobStatus.DONE:
            await store.complete(job.id)
        elif status is JobStatus.FAILED:
            await store.fail(job.id, "x", None)
    return await get(store, job.id)


# --- Shared semantics ---------------------------------------------------------


def test_satisfies_port(store: Any) -> None:
    assert isinstance(store, JobStore)


async def test_enqueue_returns_pending_job(store: Any, clock: FakeClock) -> None:
    job = await store.enqueue("extract", {"episode": 3, "tags": ["a"], "x": None})
    assert job.kind == "extract"
    assert job.payload == {"episode": 3, "tags": ["a"], "x": None}
    assert job.status is JobStatus.PENDING
    assert job.attempts == 0 and job.max_attempts == 5
    assert job.run_after == job.created_at == job.updated_at == T0
    assert job.last_error is None and job.dedupe_key is None
    assert await get(store, job.id) == job
    second = await store.enqueue("extract", {}, max_attempts=2)
    assert second.id > job.id and second.max_attempts == 2


async def test_enqueue_run_after_is_kept_in_utc(store: Any) -> None:
    later = (T0 + timedelta(hours=1)).astimezone(timezone(timedelta(hours=5, minutes=30)))
    job = await store.enqueue("k", {}, run_after=later)
    assert job.run_after == T0 + timedelta(hours=1)
    assert job.run_after.utcoffset() == timedelta(0)


async def test_enqueue_rejects_bad_max_attempts(store: Any) -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        await store.enqueue("k", {}, max_attempts=0)


@pytest.mark.parametrize("status", list(JobStatus))
async def test_dedupe_blocks_only_while_active(
    store: Any, clock: FakeClock, status: JobStatus
) -> None:
    first = await with_status(store, clock, status, "episode-1")
    again = await store.enqueue("k", {"other": True}, dedupe_key="episode-1")
    if status in ACTIVE:
        assert again == first
        assert sum((await store.counts()).values()) == 1
    else:
        assert again.id != first.id and again.status is JobStatus.PENDING
        assert again.payload == {"other": True}
        assert sum((await store.counts()).values()) == 2


async def test_dedupe_keys_are_independent(store: Any) -> None:
    a = await store.enqueue("k", {}, dedupe_key="a")
    b = await store.enqueue("k", {}, dedupe_key="b")
    n1 = await store.enqueue("k", {})
    n2 = await store.enqueue("k", {})
    assert len({a.id, b.id, n1.id, n2.id}) == 4
    assert await store.enqueue("k", {}, dedupe_key="a") == a


async def test_claim_takes_due_jobs_by_run_after_then_id(store: Any, clock: FakeClock) -> None:
    assert await store.claim(clock()) is None
    late = await store.enqueue("k", {"n": "late"}, run_after=T0 + timedelta(seconds=1))
    b = await store.enqueue("k", {"n": "b"}, run_after=T0)
    a = await store.enqueue("k", {"n": "a"}, run_after=T0 - timedelta(seconds=1))
    c = await store.enqueue("k", {"n": "c"}, run_after=T0)
    claimed = [await claim_one(store, clock) for _ in range(3)]
    assert [j.id for j in claimed] == [a.id, b.id, c.id]
    assert all(j.status is JobStatus.RUNNING and j.attempts == 1 for j in claimed)
    assert await store.claim(clock()) is None  # `late` is not due; claimed ones are not pending
    clock.advance(1)  # run_after == now counts as due
    assert (await claim_one(store, clock)).id == late.id
    assert await get(store, a.id) == claimed[0]


async def test_claim_uses_the_given_time(store: Any) -> None:
    job = await store.enqueue("k", {}, run_after=T0 + timedelta(minutes=1))
    assert await store.claim(T0) is None
    claimed = await store.claim(T0 + timedelta(minutes=2))
    assert claimed is not None and claimed.id == job.id


async def test_complete(store: Any, clock: FakeClock) -> None:
    job = await store.enqueue("k", {})
    await claim_one(store, clock)
    clock.advance(5)
    await store.complete(job.id)
    done = await get(store, job.id)
    assert done.status is JobStatus.DONE and done.attempts == 1
    assert done.updated_at == T0 + timedelta(seconds=5)


async def test_transitions_require_a_running_job(store: Any, clock: FakeClock) -> None:
    pending = await store.enqueue("k", {})
    for job_id in (pending.id, 999):
        with pytest.raises(JobStateError):
            await store.complete(job_id)
        with pytest.raises(JobStateError):
            await store.fail(job_id, "x", None)
        with pytest.raises(JobStateError):
            await store.defer(job_id, T0)
    await claim_one(store, clock)
    await store.complete(pending.id)
    with pytest.raises(JobStateError):
        await store.complete(pending.id)
    assert (await get(store, pending.id)).status is JobStatus.DONE


async def test_fail_with_retry_reschedules(store: Any, clock: FakeClock) -> None:
    job = await store.enqueue("k", {})
    await claim_one(store, clock)
    await store.fail(job.id, "RuntimeError: boom", T0 + timedelta(seconds=30))
    retry = await get(store, job.id)
    assert retry.status is JobStatus.PENDING and retry.attempts == 1
    assert retry.last_error == "RuntimeError: boom"
    assert retry.run_after == T0 + timedelta(seconds=30)
    assert await store.claim(clock()) is None
    clock.advance(30)
    again = await claim_one(store, clock)
    assert again.attempts == 2 and again.last_error == "RuntimeError: boom"


async def test_fail_without_retry_is_permanent(store: Any, clock: FakeClock) -> None:
    job = await store.enqueue("k", {})
    await claim_one(store, clock)
    await store.fail(job.id, "ValueError: no", None)
    failed = await get(store, job.id)
    assert failed.status is JobStatus.FAILED and failed.last_error == "ValueError: no"
    clock.advance(10**6)
    assert await store.claim(clock()) is None
    assert await store.next_due() is None


async def test_defer_does_not_count_the_attempt(store: Any, clock: FakeClock) -> None:
    job = await store.enqueue("k", {})
    await claim_one(store, clock)
    await store.fail(job.id, "first", T0)
    await claim_one(store, clock)  # attempts == 2
    await store.defer(job.id, T0 + timedelta(seconds=1))
    deferred = await get(store, job.id)
    assert deferred.status is JobStatus.PENDING and deferred.attempts == 1
    assert deferred.run_after == T0 + timedelta(seconds=1)
    assert deferred.last_error == "first"


async def test_recover_resets_running_jobs(store: Any, clock: FakeClock) -> None:
    assert await store.recover() == 0
    running = [await store.enqueue("k", {"n": n}) for n in range(2)]
    done = await store.enqueue("k", {})
    pending = await store.enqueue("k", {}, run_after=T0 + timedelta(hours=1))
    for _ in range(3):
        await claim_one(store, clock)
    await store.complete(done.id)
    assert await store.recover() == 2
    for job in running:
        recovered = await get(store, job.id)
        assert recovered.status is JobStatus.PENDING and recovered.attempts == 0
    assert (await get(store, done.id)).status is JobStatus.DONE
    assert await get(store, pending.id) == pending
    assert await store.recover() == 0


async def test_next_due_is_earliest_pending(store: Any, clock: FakeClock) -> None:
    assert await store.next_due() is None
    first = await store.enqueue("k", {}, run_after=T0 - timedelta(minutes=1))
    await store.enqueue("k", {}, run_after=T0 + timedelta(minutes=2))
    await store.enqueue("k", {}, run_after=T0 + timedelta(minutes=1))
    assert await store.next_due() == T0 - timedelta(minutes=1)
    assert (await claim_one(store, clock)).id == first.id  # running jobs do not count
    assert await store.next_due() == T0 + timedelta(minutes=1)


async def test_counts_cover_every_status(store: Any, clock: FakeClock) -> None:
    assert await store.counts() == dict.fromkeys(JobStatus, 0)
    for status, key in [  # pending last: driving a job claims the oldest due one
        (JobStatus.RUNNING, "r"),
        (JobStatus.DONE, "d"),
        (JobStatus.FAILED, "f"),
        (JobStatus.PENDING, "p1"),
        (JobStatus.PENDING, "p2"),
    ]:
        await with_status(store, clock, status, key)
    counts = await store.counts()
    assert counts == {
        JobStatus.PENDING: 2,
        JobStatus.RUNNING: 1,
        JobStatus.DONE: 1,
        JobStatus.FAILED: 1,
    }
    assert all(type(k) is JobStatus for k in counts)


async def test_runner_end_to_end(store: Any, clock: FakeClock) -> None:
    seen: list[int] = []

    async def handler(job: Job) -> None:
        seen.append(job.payload["n"])
        if job.payload["n"] == 2 and job.attempts == 1:
            raise RuntimeError("once")

    for n in range(3):
        await store.enqueue("k", {"n": n})
    runner = JobRunner(store, {"k": handler}, clock=clock, backoff=lambda a: 60.0)
    assert await runner.run_until_idle() == 2
    clock.advance(60)
    assert await runner.run_until_idle() == 1
    assert seen == [0, 1, 2, 2]
    assert (await store.counts())[JobStatus.DONE] == 3


# --- SQLite specifics ---------------------------------------------------------


def _version(path: Path) -> int:
    with sqlite3.connect(path) as conn:
        return conn.execute("PRAGMA user_version").fetchone()[0]


async def test_connects_lazily_and_reconnects(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "nested" / "lucy.db"
    store = SQLiteJobStore(path, clock=clock)
    assert not path.exists()
    job = await store.enqueue("k", {"x": 1})  # connects, creating the directories
    await store.close()
    await store.close()  # idempotent
    assert await store.get(job.id) == job  # the closed store reconnects on use
    await store.close()


async def test_round_trip_after_reopen(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "lucy.db"
    async with SQLiteJobStore(path, clock=clock) as store:
        a = await store.enqueue("extract", {"episode": 1}, dedupe_key="e1")
        b = await store.enqueue("reflect", {}, run_after=T0 + timedelta(days=1), max_attempts=2)
        claimed = await claim_one(store, clock)
    assert claimed.id == a.id
    async with SQLiteJobStore(path, clock=clock) as store:
        assert await store.get(a.id) == claimed
        assert await store.get(b.id) == b
        assert await store.enqueue("extract", {}, dedupe_key="e1") == claimed
        assert await store.recover() == 1
        recovered = await get(store, a.id)
        assert recovered.status is JobStatus.PENDING and recovered.attempts == 0


async def test_columns_are_utc_text_and_json(tmp_path: Path) -> None:
    path = tmp_path / "lucy.db"
    clock = FakeClock(T0.replace(microsecond=0))
    async with SQLiteJobStore(path, clock=clock) as store:
        await store.enqueue("k", {"b": [1, 2], "a": "é"}, dedupe_key="key")
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT kind, payload, status, run_after, created_at, updated_at FROM jobs"
        ).fetchone()
        indexes = {r[1] for r in conn.execute("PRAGMA index_list(jobs)").fetchall()}
    kind, payload, status, *times = row
    assert (kind, status) == ("k", "pending")
    assert json.loads(payload) == {"b": [1, 2], "a": "é"}
    # Fixed width, microseconds always present, so text order is time order.
    assert times == ["2026-11-12T09:00:00.000000+00:00"] * 3
    assert {"jobs_due", "jobs_dedupe_active"} <= indexes


async def test_naive_datetimes_are_rejected(tmp_path: Path) -> None:
    async with SQLiteJobStore(tmp_path / "lucy.db") as store:
        with pytest.raises(ValueError, match="naive"):
            await store.enqueue("k", {}, run_after=datetime(2026, 1, 1))
        with pytest.raises(ValueError, match="naive"):
            await store.claim(datetime(2026, 1, 1))


async def test_dedupe_is_enforced_by_the_schema(tmp_path: Path) -> None:
    path = tmp_path / "lucy.db"
    async with SQLiteJobStore(path) as store:
        await store.enqueue("k", {}, dedupe_key="d")
    with sqlite3.connect(path) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO jobs (kind, payload, status, max_attempts, run_after, created_at,"
            " updated_at, dedupe_key) VALUES ('k', '{}', 'running', 1, 'x', 'x', 'x', 'd')"
        )


async def test_concurrent_claimers_never_share_a_job(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "lucy.db"
    async with (
        SQLiteJobStore(path, clock=clock) as one,
        SQLiteJobStore(path, clock=clock) as two,
    ):
        jobs = [await one.enqueue("k", {"n": n}) for n in range(60)]

        async def drain(store: SQLiteJobStore) -> list[int]:
            got = []
            while (job := await store.claim(clock())) is not None:
                got.append(job.id)
            return got

        async with asyncio.timeout(10):
            results = await asyncio.gather(*(drain(s) for s in (one, two, one, two)))
        claimed = [job_id for result in results for job_id in result]
        assert sorted(claimed) == [j.id for j in jobs]  # each exactly once
        assert (await one.counts())[JobStatus.RUNNING] == 60


async def test_concurrent_dedupe_inserts_one_job(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "lucy.db"
    async with (
        SQLiteJobStore(path, clock=clock) as one,
        SQLiteJobStore(path, clock=clock) as two,
    ):
        async with asyncio.timeout(10):
            jobs = await asyncio.gather(
                *(s.enqueue("k", {}, dedupe_key="same") for s in [one, two] * 10)
            )
        assert len({j.id for j in jobs}) == 1
        assert sum((await one.counts()).values()) == 1


async def test_shares_the_file_with_the_memory_store(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "lucy.db"
    event = UserMessage(session_id="s", text="hi")
    async with SQLiteMemoryStore(path) as memory, SQLiteJobStore(path, clock=clock) as jobs:
        await asyncio.gather(
            memory.append_episode(event),
            jobs.enqueue("extract", {"event_id": event.id}),
        )
        assert await memory.recent_episodes("s", 1) == [event]
        job = await claim_one(jobs, clock)
        assert job.payload == {"event_id": event.id}
    assert _version(path) == len(migrations.MIGRATIONS) == 5


async def test_migration_2_to_3_keeps_episodes_and_traces(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "old.db"
    events = [
        UserMessage(session_id="s", turn_id="t", text="before the job queue"),
        AssistantMessage(session_id="s", turn_id="t", text="still here"),
    ]
    async with aiosqlite.connect(path, autocommit=True) as conn:
        assert await apply_migrations(conn, migrations.MIGRATIONS[:2]) == 2
    old = SQLiteMemoryStore(path)
    old._conn = await aiosqlite.connect(path, autocommit=True)  # write at version 2
    for event in events:
        await old.append_episode(event)
    await old.record(_trace("t"))
    await old.close()
    assert _version(path) == 2

    async with SQLiteJobStore(path, clock=clock) as jobs:  # the job store migrates the file
        job = await jobs.enqueue("extract", {"turn": "t"})
        assert (await claim_one(jobs, clock)).id == job.id
    assert _version(path) == 5
    async with SQLiteMemoryStore(path) as memory:
        assert await memory.recent_episodes("s", 10) == events
        assert await memory.get_trace("t") == _trace("t")


async def test_open_database_sets_pragmas(tmp_path: Path) -> None:
    conn = await open_database(tmp_path / "a" / "b.db")
    try:
        async with conn.execute("PRAGMA journal_mode") as cur:
            assert (await cur.fetchone())[0] == "wal"
        async with conn.execute("PRAGMA foreign_keys") as cur:
            assert (await cur.fetchone())[0] == 1
        async with conn.execute("PRAGMA busy_timeout") as cur:
            assert (await cur.fetchone())[0] == 5000
        async with conn.execute("PRAGMA user_version") as cur:
            assert (await cur.fetchone())[0] == len(migrations.MIGRATIONS)
    finally:
        await conn.close()


async def test_default_clock_is_utc(tmp_path: Path) -> None:
    async with SQLiteJobStore(tmp_path / "lucy.db") as store:
        job = await store.enqueue("k", {})
    assert job.created_at.utcoffset() == timedelta(0)
    assert job.created_at.tzinfo is not None


async def test_concurrent_coroutines_never_see_half_written_results(tmp_path: Path) -> None:
    """Regression: `counts` while another coroutine completed jobs on the same
    connection returned rows with a NULL status (the UPDATE ran between the
    SELECT's execute and its fetch). Statements are now serialised per store."""
    errors: list[BaseException] = []
    async with SQLiteJobStore(tmp_path / "lucy.db") as store:

        async def count() -> None:
            for _ in range(300):
                try:
                    counts = await store.counts()
                    assert set(counts) == set(JobStatus)
                    await store.list_jobs(limit=5)
                except Exception as exc:  # collected, so every failure is reported
                    errors.append(exc)

        async def work() -> None:
            for _ in range(100):
                await store.enqueue("k", {})
                job = await store.claim(datetime.now(UTC))
                assert job is not None
                await store.complete(job.id)

        async with asyncio.timeout(30):
            await asyncio.gather(count(), work())
    assert errors == []


async def test_list_jobs_filters(tmp_path: Path) -> None:
    async with SQLiteJobStore(tmp_path / "lucy.db") as store:
        first = await store.enqueue("a", {})
        second = await store.enqueue("b", {})
        job = await store.claim(datetime.now(UTC) + timedelta(seconds=1))
        assert job is not None and job.id == first.id
        await store.fail(job.id, "boom", None)
        assert [j.id for j in await store.list_jobs()] == [first.id, second.id]
        assert [j.id for j in await store.list_jobs(statuses=[JobStatus.FAILED])] == [first.id]
        assert [j.id for j in await store.list_jobs(with_error=True)] == [first.id]
        assert [j.id for j in await store.list_jobs(statuses=[JobStatus.PENDING])] == [second.id]
        assert len(await store.list_jobs(limit=1)) == 1
