"""Background job queue: the job model, the `JobStore` port and the runner (ADR 0005).

Memory extraction, embedding and nightly reflection run as jobs. A job is
stored before it runs, so it survives restarts; the runner takes one due job at
a time, calls the handler registered for its kind and records the outcome:
done, retried later with backoff, failed for good, or deferred without counting
the attempt (pre-empted by foreground work, or interrupted by shutdown).

Everything here is storage-agnostic; the SQLite store lives in `lucy.memory`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import AwareDatetime, BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class JobStatus(StrEnum):
    """Where a job is in its life."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Job(BaseModel):
    """One unit of background work, as stored."""

    model_config = ConfigDict(frozen=True)

    id: int
    kind: str
    payload: dict[str, Any]
    status: JobStatus
    attempts: int
    """Times a run has been started and counted (deferred runs are not counted)."""
    max_attempts: int
    run_after: AwareDatetime
    """Not eligible to run before this (UTC)."""
    created_at: AwareDatetime
    updated_at: AwareDatetime
    last_error: str | None = None
    dedupe_key: str | None = None


class JobStateError(RuntimeError):
    """A transition was asked for a job that does not exist or is not RUNNING."""


@runtime_checkable
class JobStore(Protocol):
    """Durable queue of jobs. Implementations must make `claim` atomic."""

    async def enqueue(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        dedupe_key: str | None = None,
        run_after: datetime | None = None,
        max_attempts: int = 5,
    ) -> Job:
        """Add a PENDING job due at `run_after` (default: now) and return it.

        If `dedupe_key` is given and a PENDING or RUNNING job with that key exists,
        return that job instead and add nothing.
        """
        ...

    async def claim(self, now: datetime) -> Job | None:
        """Take the oldest PENDING job with `run_after <= now` (by `run_after`, then id).

        It becomes RUNNING with `attempts` incremented. None if nothing is due.
        """
        ...

    async def complete(self, job_id: int) -> None:
        """RUNNING -> DONE."""
        ...

    async def fail(self, job_id: int, error: str, retry_at: datetime | None) -> None:
        """Record `error`; RUNNING -> PENDING at `retry_at`, or -> FAILED if it is None."""
        ...

    async def defer(self, job_id: int, run_after: datetime) -> None:
        """RUNNING -> PENDING at `run_after`, without counting the attempt."""
        ...

    async def recover(self) -> int:
        """Every RUNNING job -> PENDING without counting the attempt; return how many.

        For startup, after a process died mid-job.
        """
        ...

    async def next_due(self) -> datetime | None:
        """The earliest `run_after` among PENDING jobs, or None."""
        ...

    async def counts(self) -> dict[JobStatus, int]:
        """Number of jobs in each status (every status present, zero included)."""
        ...


Handler = Callable[[Job], Awaitable[None]]
"""Runs one job. Returning normally means done; raising means retry or fail."""


def exponential_backoff(attempts: int) -> float:
    """Seconds before retry: 5s after the first attempt, doubling, capped at one hour."""
    return min(5.0 * 2.0 ** max(attempts - 1, 0), 3600.0)


def utc_now() -> datetime:
    """The current time in UTC."""
    return datetime.now(UTC)


class JobRunner:
    """Runs due jobs from a `JobStore`, one at a time, in due order."""

    def __init__(
        self,
        store: JobStore,
        handlers: Mapping[str, Handler],
        *,
        defer_on: tuple[type[BaseException], ...] = (),
        defer_delay: float = 1.0,
        backoff: Callable[[int], float] = exponential_backoff,
        clock: Callable[[], datetime] = utc_now,
        poll_interval: float = 30.0,
        error_delay: float = 1.0,
    ) -> None:
        """`defer_on`: handler exceptions that defer the job by `defer_delay` seconds
        without counting the attempt (e.g. the gateway's pre-emption).
        `backoff(attempts)`: seconds before retrying a job that failed its
        `attempts`-th run. `poll_interval`: the longest an idle `run` sleeps.
        `error_delay`: the pause after the store itself fails inside `run`.
        """
        self._store = store
        self._handlers = dict(handlers)
        self._defer_on = defer_on
        self._defer_delay = defer_delay
        self._backoff = backoff
        self._clock = clock
        self._poll_interval = poll_interval
        self._error_delay = error_delay
        self._wake = asyncio.Event()
        self._running = 0

    @property
    def running(self) -> bool:
        """Whether a job is being claimed or run now (by `run` or `run_until_idle`)."""
        return self._running > 0

    async def has_due(self) -> bool:
        """Whether a pending job is due now (it may not have been claimed yet)."""
        due = await self._store.next_due()
        return due is not None and due <= self._clock()

    def notify(self) -> None:
        """Wake an idle `run` now (call after enqueueing, or when time has moved on)."""
        self._wake.set()

    async def run(self) -> None:
        """Recover interrupted jobs, then run due jobs forever, until cancelled."""
        while True:
            try:
                recovered = await self._store.recover()
            except Exception:
                logger.exception("job store failed to recover interrupted jobs; retrying")
                await asyncio.sleep(self._error_delay)
                continue
            if recovered:
                logger.info("recovered %d interrupted job(s)", recovered)
            break
        while True:
            # Clear before claiming: a notify() from here on wakes the idle wait below.
            self._wake.clear()
            try:
                job = await self._claim()
                if job is not None:
                    await self._run_one(job)
                    continue
                timeout = await self._idle_timeout()
            except Exception:
                logger.exception("job store error; pausing before trying again")
                await asyncio.sleep(self._error_delay)
                continue
            try:
                async with asyncio.timeout(timeout):
                    await self._wake.wait()
            except TimeoutError:
                pass

    async def run_until_idle(self) -> int:
        """Run due jobs until none is due; return how many completed.

        Jobs due in the future (such as retries waiting out their backoff) stay
        pending. Store errors propagate.
        """
        completed = 0
        while (job := await self._claim()) is not None:
            if await self._run_one(job):
                completed += 1
        return completed

    async def _idle_timeout(self) -> float:
        timeout = self._poll_interval
        due = await self._store.next_due()
        if due is not None:
            timeout = min(timeout, max((due - self._clock()).total_seconds(), 0.0))
        return timeout

    async def _claim(self) -> Job | None:
        """Claim the next due job. `running` holds from here until `_run_one` ends."""
        self._running += 1
        try:
            job = await self._store.claim(self._clock())
        except BaseException:
            self._running -= 1
            raise
        if job is None:
            self._running -= 1
        return job

    async def _run_one(self, job: Job) -> bool:
        """Run `job` (claimed by `_claim`) and record the outcome; True if it completed."""
        try:
            return await self._run_claimed(job)
        finally:
            self._running -= 1

    async def _run_claimed(self, job: Job) -> bool:
        handler = self._handlers.get(job.kind)
        if handler is None:
            error = f"no handler registered for job kind {job.kind!r}"
            logger.error("job %d failed permanently: %s", job.id, error)
            await self._store.fail(job.id, error, None)
            return False
        try:
            await handler(job)
        except BaseException as exc:
            await self._handle_exception(job, exc)
            return False
        await self._store.complete(job.id)
        return True

    async def _handle_exception(self, job: Job, exc: BaseException) -> None:
        """Record why the handler of `job` did not finish; re-raise if we are stopping."""
        task = asyncio.current_task()
        if isinstance(exc, asyncio.CancelledError):
            # Our own cancellation, as opposed to a CancelledError leaking from the handler.
            stopping = task is not None and task.cancelling() > 0
        else:
            stopping = not isinstance(exc, Exception)  # KeyboardInterrupt, SystemExit
        if stopping:
            # Cancelled (or interrupted) mid-job: put it back as due now, uncounted.
            await self._write_despite_cancellation(job, self._store.defer(job.id, self._clock()))
            raise exc
        if isinstance(exc, self._defer_on):
            logger.info("job %d (%s) deferred: %r", job.id, job.kind, exc)
            await self._store.defer(job.id, self._clock() + timedelta(seconds=self._defer_delay))
            return
        error = f"{type(exc).__name__}: {exc}"
        if job.attempts >= job.max_attempts:
            logger.error(
                "job %d (%s) failed permanently after %d attempt(s): %s",
                job.id,
                job.kind,
                job.attempts,
                error,
                exc_info=exc,
            )
            await self._store.fail(job.id, error, None)
        else:
            delay = self._backoff(job.attempts)
            logger.warning(
                "job %d (%s) failed (attempt %d of %d), retrying in %.0fs: %s",
                job.id,
                job.kind,
                job.attempts,
                job.max_attempts,
                delay,
                error,
                exc_info=exc,
            )
            await self._store.fail(job.id, error, self._clock() + timedelta(seconds=delay))

    @staticmethod
    async def _write_despite_cancellation(job: Job, write: Awaitable[None]) -> None:
        """Wait for `write` to finish even if we are cancelled again meanwhile."""
        future = asyncio.ensure_future(write)
        while not future.done():
            try:
                await asyncio.shield(future)
            except BaseException:
                pass  # cancelled again (keep waiting), or the write failed (logged below)
        if not future.cancelled() and (error := future.exception()) is not None:
            # Not lost: `recover` puts the job back at the next start.
            logger.error("could not put back interrupted job %d", job.id, exc_info=error)
