"""The job runner, against an in-memory `JobStore` double (core/jobs.py).

`InMemoryJobStore` and `FakeClock` are also used by tests/test_sqlite_jobs.py,
which runs one semantic suite over this double and the SQLite store.
No test waits on wall-clock time: the clock is fake, and synchronisation uses
events and the store's call log, with `asyncio.timeout` guards against hangs.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from lucy.core.jobs import (
    Job,
    JobRunner,
    JobStateError,
    JobStatus,
    JobStore,
    exponential_backoff,
)

T0 = datetime(2026, 11, 12, 9, 0, tzinfo=UTC)
GUARD = 2.0  # seconds before a hang fails the test


class FakeClock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class InMemoryJobStore:
    """A `JobStore` in a dict. Each operation runs without yielding, so it is atomic.

    It logs the name of every call and lets tests wait for calls (`wait_for`).
    """

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or FakeClock()
        self._jobs: dict[int, Job] = {}
        self._next_id = 1
        self.calls: list[str] = []
        self._changed = asyncio.Condition()

    async def _called(self, name: str) -> None:
        self.calls.append(name)
        async with self._changed:
            self._changed.notify_all()

    async def wait_for(self, name: str, count: int = 1) -> None:
        """Wait until `name` has been called at least `count` times."""
        async with asyncio.timeout(GUARD), self._changed:
            await self._changed.wait_for(lambda: self.calls.count(name) >= count)

    def _update(self, job_id: int, **changes: Any) -> None:
        job = self._jobs.get(job_id)
        if job is None or job.status is not JobStatus.RUNNING:
            raise JobStateError(f"job {job_id} does not exist or is not running")
        self._jobs[job_id] = job.model_copy(update={**changes, "updated_at": self._clock()})

    async def get(self, job_id: int) -> Job | None:
        return self._jobs.get(job_id)

    async def enqueue(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        dedupe_key: str | None = None,
        run_after: datetime | None = None,
        max_attempts: int = 5,
    ) -> Job:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        result = None
        if dedupe_key is not None:
            result = next(
                (
                    j
                    for j in self._jobs.values()
                    if j.dedupe_key == dedupe_key
                    and j.status in (JobStatus.PENDING, JobStatus.RUNNING)
                ),
                None,
            )
        if result is None:
            now = self._clock()
            result = Job(
                id=self._next_id,
                kind=kind,
                payload=dict(payload),
                status=JobStatus.PENDING,
                attempts=0,
                max_attempts=max_attempts,
                run_after=(run_after or now).astimezone(UTC),
                created_at=now,
                updated_at=now,
                last_error=None,
                dedupe_key=dedupe_key,
            )
            self._jobs[result.id] = result
            self._next_id += 1
        await self._called("enqueue")
        return result

    async def claim(self, now: datetime) -> Job | None:
        due = [
            j for j in self._jobs.values() if j.status is JobStatus.PENDING and j.run_after <= now
        ]
        job = None
        if due:
            first = min(due, key=lambda j: (j.run_after, j.id))
            job = first.model_copy(
                update={
                    "status": JobStatus.RUNNING,
                    "attempts": first.attempts + 1,
                    "updated_at": self._clock(),
                }
            )
            self._jobs[job.id] = job
        await self._called("claim")
        return job

    async def complete(self, job_id: int) -> None:
        self._update(job_id, status=JobStatus.DONE)
        await self._called("complete")

    async def fail(self, job_id: int, error: str, retry_at: datetime | None) -> None:
        if retry_at is None:
            self._update(job_id, status=JobStatus.FAILED, last_error=error)
        else:
            self._update(job_id, status=JobStatus.PENDING, last_error=error, run_after=retry_at)
        await self._called("fail")

    async def defer(self, job_id: int, run_after: datetime) -> None:
        attempts = max(self._jobs[job_id].attempts - 1, 0) if job_id in self._jobs else 0
        self._update(job_id, status=JobStatus.PENDING, run_after=run_after, attempts=attempts)
        await self._called("defer")

    async def recover(self) -> int:
        running = [j for j in self._jobs.values() if j.status is JobStatus.RUNNING]
        for job in running:
            self._update(job.id, status=JobStatus.PENDING, attempts=max(job.attempts - 1, 0))
        await self._called("recover")
        return len(running)

    async def next_due(self) -> datetime | None:
        pending = [j.run_after for j in self._jobs.values() if j.status is JobStatus.PENDING]
        await self._called("next_due")
        return min(pending, default=None)

    async def counts(self) -> dict[JobStatus, int]:
        result = dict.fromkeys(JobStatus, 0)
        for job in self._jobs.values():
            result[job.status] += 1
        await self._called("counts")
        return result


# --- Fixtures and helpers -----------------------------------------------------


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(clock: FakeClock) -> InMemoryJobStore:
    return InMemoryJobStore(clock)


class Recorder:
    """Handlers that record the jobs they ran."""

    def __init__(self) -> None:
        self.ran: list[Job] = []
        self.done = asyncio.Event()

    async def ok(self, job: Job) -> None:
        self.ran.append(job)
        self.done.set()


class Preempted(Exception):
    pass


def runner(store: JobStore, clock: FakeClock, handlers: dict[str, Any], **kw: Any) -> JobRunner:
    kw.setdefault("backoff", lambda attempts: 10.0 * attempts)
    return JobRunner(store, handlers, clock=clock, **kw)


async def stop(task: asyncio.Task[None]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(GUARD):
            await task


def test_store_double_satisfies_port() -> None:
    assert isinstance(InMemoryJobStore(), JobStore)


def test_default_backoff_is_exponential_and_capped() -> None:
    assert [exponential_backoff(n) for n in (1, 2, 3, 4)] == [5.0, 10.0, 20.0, 40.0]
    assert exponential_backoff(11) == 3600.0
    assert exponential_backoff(100) == 3600.0


# --- Per-job outcomes (run_until_idle) ----------------------------------------


async def test_success_completes_the_job(store: InMemoryJobStore, clock: FakeClock) -> None:
    rec = Recorder()
    job = await store.enqueue("extract", {"episode": 7})
    assert await runner(store, clock, {"extract": rec.ok}).run_until_idle() == 1
    assert [j.payload for j in rec.ran] == [{"episode": 7}]
    assert rec.ran[0].status is JobStatus.RUNNING and rec.ran[0].attempts == 1
    done = await store.get(job.id)
    assert done is not None and done.status is JobStatus.DONE and done.attempts == 1


async def test_retry_with_backoff_then_success(store: InMemoryJobStore, clock: FakeClock) -> None:
    calls = 0

    async def flaky(job: Job) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("model unavailable")

    r = runner(store, clock, {"extract": flaky})
    job = await store.enqueue("extract", {})
    assert await r.run_until_idle() == 0
    retry = await store.get(job.id)
    assert retry is not None
    assert retry.status is JobStatus.PENDING and retry.attempts == 1
    assert retry.last_error == "RuntimeError: model unavailable"
    assert retry.run_after == T0 + timedelta(seconds=10)  # backoff(1)

    clock.advance(9)
    assert await r.run_until_idle() == 0  # not due yet
    assert calls == 1
    clock.advance(1)
    assert await r.run_until_idle() == 1
    done = await store.get(job.id)
    assert done is not None and done.status is JobStatus.DONE and done.attempts == 2


async def test_permanent_failure_after_max_attempts(
    store: InMemoryJobStore, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    async def broken(job: Job) -> None:
        raise ValueError(f"bad payload on attempt {job.attempts}")

    r = runner(store, clock, {"extract": broken})
    job = await store.enqueue("extract", {}, max_attempts=3)
    for _ in range(3):
        assert await r.run_until_idle() == 0
        clock.advance(3600)
    failed = await store.get(job.id)
    assert failed is not None
    assert failed.status is JobStatus.FAILED and failed.attempts == 3
    assert failed.last_error == "ValueError: bad payload on attempt 3"
    assert await r.run_until_idle() == 0  # never claimed again
    assert "failed permanently" in caplog.text


async def test_unknown_kind_fails_permanently(store: InMemoryJobStore, clock: FakeClock) -> None:
    job = await store.enqueue("mystery", {})
    assert await runner(store, clock, {}).run_until_idle() == 0
    failed = await store.get(job.id)
    assert failed is not None and failed.status is JobStatus.FAILED
    assert failed.last_error is not None and "'mystery'" in failed.last_error
    assert "no handler" in failed.last_error


async def test_defer_on_exception_defers_without_counting(
    store: InMemoryJobStore, clock: FakeClock
) -> None:
    preempt = True

    async def handler(job: Job) -> None:
        if preempt:
            raise Preempted

    r = runner(store, clock, {"extract": handler}, defer_on=(Preempted,), defer_delay=2.5)
    job = await store.enqueue("extract", {}, max_attempts=1)
    for _ in range(3):  # deferring never uses up the single attempt
        assert await r.run_until_idle() == 0
        deferred = await store.get(job.id)
        assert deferred is not None
        assert deferred.status is JobStatus.PENDING and deferred.attempts == 0
        assert deferred.run_after == clock() + timedelta(seconds=2.5)
        assert deferred.last_error is None
        clock.advance(2.5)
    preempt = False
    assert await r.run_until_idle() == 1
    done = await store.get(job.id)
    assert done is not None and done.status is JobStatus.DONE and done.attempts == 1


async def test_run_until_idle_leaves_future_jobs_and_counts_completed(
    store: InMemoryJobStore, clock: FakeClock
) -> None:
    rec = Recorder()

    async def boom(job: Job) -> None:
        raise RuntimeError("x")

    await store.enqueue("ok", {"n": 1})
    await store.enqueue("ok", {"n": 2}, run_after=T0 - timedelta(seconds=1))
    await store.enqueue("ok", {"n": 3}, run_after=T0 + timedelta(seconds=1))
    await store.enqueue("boom", {})
    assert await runner(store, clock, {"ok": rec.ok, "boom": boom}).run_until_idle() == 2
    assert [j.payload["n"] for j in rec.ran] == [2, 1]
    assert await store.counts() == {
        JobStatus.PENDING: 2,  # the future job and the retry
        JobStatus.RUNNING: 0,
        JobStatus.DONE: 2,
        JobStatus.FAILED: 0,
    }
    assert "recover" not in store.calls


# --- The run loop -------------------------------------------------------------


async def test_run_recovers_interrupted_jobs_at_start(
    store: InMemoryJobStore, clock: FakeClock
) -> None:
    rec = Recorder()
    job = await store.enqueue("extract", {})
    await store.claim(clock())  # a previous process died while running it
    task = asyncio.create_task(runner(store, clock, {"extract": rec.ok}).run())
    async with asyncio.timeout(GUARD):
        await rec.done.wait()
    await stop(task)
    assert store.calls[:3] == ["enqueue", "claim", "recover"]
    assert store.calls.count("recover") == 1
    done = await store.get(job.id)
    assert done is not None and done.status is JobStatus.DONE
    assert done.attempts == 1  # the interrupted run was not counted


async def test_notify_wakes_an_idle_run(store: InMemoryJobStore, clock: FakeClock) -> None:
    rec = Recorder()
    r = runner(store, clock, {"extract": rec.ok}, poll_interval=3600)
    task = asyncio.create_task(r.run())
    await store.wait_for("next_due")  # idle: nothing to claim
    await store.enqueue("extract", {"n": 1})
    r.notify()
    async with asyncio.timeout(GUARD):
        await rec.done.wait()
    await stop(task)
    assert [j.payload for j in rec.ran] == [{"n": 1}]


async def test_future_job_runs_when_the_clock_reaches_it(
    store: InMemoryJobStore, clock: FakeClock
) -> None:
    rec = Recorder()
    r = runner(store, clock, {"extract": rec.ok}, poll_interval=3600)
    await store.enqueue("extract", {}, run_after=T0 + timedelta(minutes=5))
    task = asyncio.create_task(r.run())
    await store.wait_for("next_due")
    assert rec.ran == []
    clock.advance(299)
    r.notify()
    await store.wait_for("next_due", 2)  # woke, found nothing due, went idle again
    assert rec.ran == []
    clock.advance(1)
    r.notify()
    async with asyncio.timeout(GUARD):
        await rec.done.wait()
    await stop(task)
    assert len(rec.ran) == 1


async def test_idle_wait_is_bounded_by_next_due_and_poll_interval(
    store: InMemoryJobStore, clock: FakeClock
) -> None:
    # Real time does not move the fake clock, so a short real poll interval makes
    # the loop re-check without notify(); it must still not run the future job.
    rec = Recorder()
    await store.enqueue("extract", {}, run_after=T0 + timedelta(hours=1))
    task = asyncio.create_task(runner(store, clock, {"extract": rec.ok}, poll_interval=0).run())
    await store.wait_for("next_due", 3)
    assert rec.ran == []
    clock.advance(3600)  # no notify: the zero poll interval picks it up
    async with asyncio.timeout(GUARD):
        await rec.done.wait()
    await stop(task)


async def test_cancel_mid_handler_defers_and_reraises(
    store: InMemoryJobStore, clock: FakeClock
) -> None:
    started = asyncio.Event()
    handler_cancelled = False

    async def slow(job: Job) -> None:
        nonlocal handler_cancelled
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            handler_cancelled = True
            raise

    job = await store.enqueue("extract", {}, max_attempts=1)
    task = asyncio.create_task(runner(store, clock, {"extract": slow}).run())
    async with asyncio.timeout(GUARD):
        await started.wait()
    clock.advance(42)
    await stop(task)
    assert handler_cancelled
    deferred = await store.get(job.id)
    assert deferred is not None
    assert deferred.status is JobStatus.PENDING and deferred.attempts == 0
    assert deferred.run_after == T0 + timedelta(seconds=42)  # due now


class SlowDeferStore(InMemoryJobStore):
    """`defer` waits for `release`, to test that the runner shields the write."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__(clock)
        self.release = asyncio.Event()
        self.deferring = asyncio.Event()

    async def defer(self, job_id: int, run_after: datetime) -> None:
        self.deferring.set()
        await self.release.wait()
        await super().defer(job_id, run_after)


async def test_defer_on_cancel_is_written_despite_repeated_cancellation(clock: FakeClock) -> None:
    store = SlowDeferStore(clock)
    started = asyncio.Event()

    async def slow(job: Job) -> None:
        started.set()
        await asyncio.Event().wait()

    job = await store.enqueue("extract", {})
    task = asyncio.create_task(runner(store, clock, {"extract": slow}).run())
    async with asyncio.timeout(GUARD):
        await started.wait()
        task.cancel()
        await store.deferring.wait()
        task.cancel()  # cancelled again while the defer is in flight
        await asyncio.sleep(0)
        assert not task.done()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    deferred = await store.get(job.id)
    assert deferred is not None
    assert deferred.status is JobStatus.PENDING and deferred.attempts == 0


class FlakyStore(InMemoryJobStore):
    """`claim` and `next_due` raise the first time each is called."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__(clock)
        self.broken = {"claim", "next_due"}

    async def claim(self, now: datetime) -> Job | None:
        if "claim" in self.broken:
            self.broken.discard("claim")
            await self._called("claim")
            raise OSError("disk I/O error")
        return await super().claim(now)

    async def next_due(self) -> datetime | None:
        if "next_due" in self.broken:
            self.broken.discard("next_due")
            await self._called("next_due")
            raise OSError("database is locked")
        return await super().next_due()


async def test_store_error_does_not_kill_run(
    clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR, logger="lucy.core.jobs")
    store = FlakyStore(clock)
    rec = Recorder()
    r = runner(store, clock, {"extract": rec.ok}, poll_interval=3600, error_delay=0)
    task = asyncio.create_task(r.run())
    await store.wait_for("next_due", 2)  # survived the claim and the next_due failures
    await store.enqueue("extract", {})
    r.notify()
    async with asyncio.timeout(GUARD):
        await rec.done.wait()
    assert not task.done()
    await stop(task)
    assert caplog.text.count("job store error") == 2


async def test_jobs_run_one_at_a_time_in_due_order(
    store: InMemoryJobStore, clock: FakeClock
) -> None:
    order: list[int] = []
    active = 0
    peak = 0
    all_done = asyncio.Event()

    async def work(job: Job) -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        for _ in range(5):
            await asyncio.sleep(0)  # yield, giving a concurrent runner a chance
        order.append(job.payload["n"])
        active -= 1
        if len(order) == 4:
            all_done.set()

    # Ids ascend 1..4 but due times put them in the order 3, 1, 4, 2.
    for n, offset in [(1, -2), (2, 0), (3, -3), (4, -2)]:
        await store.enqueue("work", {"n": n}, run_after=T0 + timedelta(seconds=offset))
    r = runner(store, clock, {"work": work})
    task = asyncio.create_task(r.run())
    for _ in range(3):
        r.notify()  # extra wake-ups must not start a second job
    async with asyncio.timeout(GUARD):
        await all_done.wait()
    await stop(task)
    assert order == [3, 1, 4, 2]
    assert peak == 1
