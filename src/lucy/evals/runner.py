"""Running scenarios against a real app on a temporary database (see `lucy.evals`)."""

from __future__ import annotations

import asyncio
import io
import itertools
import logging
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from lucy.app import LucyApp
from lucy.clients import memory_cli
from lucy.core.config import AdapterConfig, Settings, ToolsSettings
from lucy.core.events import TextDelta, TurnEnded, UserMessage
from lucy.core.jobs import JobStatus
from lucy.core.registry import Registry
from lucy.core.types import Role
from lucy.evals.matching import Match, has_keyword
from lucy.evals.scenario import (
    START,
    AbsentCheck,
    Advance,
    AnswerCheck,
    Correct,
    CountCheck,
    EntityCheck,
    Forget,
    History,
    Jobs,
    Maintenance,
    ProfileCheck,
    Rebuild,
    Say,
    Scenario,
    StatementCheck,
    SupersededCheck,
)
from lucy.memory.extraction import request_extraction
from lucy.memory.graph_store import SQLiteGraphStore
from lucy.memory.models import Statement
from lucy.memory.sqlite_store import SQLiteMemoryStore

log = logging.getLogger(__name__)

#: Seconds a chat turn may take (the model server may be busy with other work).
TURN_TIMEOUT: Final = 300.0
#: Seconds one `jobs` or `maintenance` step may take in total.
JOBS_TIMEOUT: Final = 3600.0
#: Seconds retrieval may take per turn (the app's default is 3; a shared, busy
#: embedding server queues requests).
RETRIEVAL_TIMEOUT: Final = 60.0
#: Statements shown as "found" in a failed check at most.
SHOWN: Final = 8


class SimClock:
    """The scenario's "now": set by steps, never by the wall clock."""

    def __init__(self, now: datetime = START) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def move_to(self, when: datetime) -> None:
        """Move forward to `when` (never back)."""
        self.now = max(self.now, when)


@dataclass
class CheckResult:
    """The outcome of one check step."""

    step: int
    check: str
    description: str
    passed: bool
    expected: str
    actual: str


@dataclass
class ScenarioResult:
    """The outcome of one scenario."""

    name: str
    category: str
    description: str
    checks: list[CheckResult] = field(default_factory=list)
    error: str | None = None
    duration_s: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
    """Seconds spent per step kind (say, jobs, maintenance, ...)."""
    transcript: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.error is None and bool(self.checks) and all(c.passed for c in self.checks)


def settings_for(settings: Settings, db: Path) -> Settings:
    """`settings` with the memory store on `db` (the configured store is never touched)
    and every tool off, so scenarios measure memory without network lookups."""
    memory = AdapterConfig(adapter="sqlite", options={"path": str(db)})
    return settings.model_copy(update={"memory": memory, "tools": ToolsSettings()})


def describe_statement(statement: Statement, now: datetime) -> str:
    line = (
        f"#{statement.id} [{statement.kind.value}, {statement.origin.value},"
        f" {statement.confidence:.2f}, {len(statement.sources)} source(s)] {statement.text}"
    )
    if not statement.is_current(now):
        line += " (ended)"
    return line


class ScenarioRunner:
    """Runs one scenario in a fresh app on `db` with simulated time."""

    def __init__(
        self,
        scenario: Scenario,
        settings: Settings,
        db: Path,
        *,
        registry: Registry | None = None,
        progress: Callable[[str], None] = lambda _: None,
        turn_timeout: float = TURN_TIMEOUT,
        jobs_timeout: float = JOBS_TIMEOUT,
        retrieval_timeout: float = RETRIEVAL_TIMEOUT,
    ) -> None:
        self.scenario = scenario
        self.settings = settings_for(settings, db)
        self.db = db
        self.registry = registry
        self.progress = progress
        self.turn_timeout = turn_timeout
        self.jobs_timeout = jobs_timeout
        self.retrieval_timeout = retrieval_timeout
        self.clock = SimClock()
        self._asks = itertools.count(1)
        self._app: LucyApp | None = None

    @property
    def app(self) -> LucyApp:
        assert self._app is not None
        return self._app

    @property
    def graph(self) -> SQLiteGraphStore:
        assert self.app.graph is not None
        return self.app.graph

    async def run(self) -> ScenarioResult:
        scenario = self.scenario
        result = ScenarioResult(scenario.name, scenario.category, scenario.description)
        started = time.monotonic()
        app = LucyApp(
            self.settings,
            registry=self.registry,
            background=False,
            clock=self.clock,
            retrieval_timeout=self.retrieval_timeout,
        )
        self._app = app
        try:
            await app.start()
            if not app.learning_active:
                raise RuntimeError(f"memory learning is off: {app.learning_off_reason}")
            for number, step in enumerate(scenario.steps, start=1):
                kind = getattr(step, "do", None) or f"check {step.check}"
                self.progress(f"  step {number}: {kind}")
                step_started = time.monotonic()
                outcome = await self._step(number, step, result)
                elapsed = time.monotonic() - step_started
                key = kind.split()[0] if not kind.startswith("check") else "checks"
                if isinstance(step, AnswerCheck):
                    key = "answers"
                result.timings[key] = result.timings.get(key, 0.0) + elapsed
                if isinstance(outcome, CheckResult):
                    result.checks.append(outcome)
                    mark = "ok" if outcome.passed else "FAILED"
                    self.progress(f"    {mark}: {outcome.description}")
                    if not outcome.passed:
                        self.progress(f"      found: {outcome.actual}")
        except Exception as exc:
            log.exception("scenario %s failed", scenario.name)
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            await app.stop()
            result.duration_s = time.monotonic() - started
        return result

    # --- Steps ------------------------------------------------------------------

    async def _step(self, number: int, step: Any, result: ScenarioResult) -> CheckResult | None:
        match step:
            case Say():
                self.clock.move_to(step.at(self.clock.now))
                reply = await self._chat(step.session, step.text)
                result.transcript += [f"[{step.session}] user: {step.text}", f"lucy: {reply}"]
            case History():
                await self._history(step)
            case Jobs():
                await self._drain()
            case Maintenance():
                self.clock.move_to(self.clock.now + timedelta(days=step.advance_days))
                await self.app.schedule_maintenance(force=True)
                await self._drain()
            case Advance():
                self.clock.move_to(self.clock.now + timedelta(days=step.days))
            case Correct():
                return await self._correct(number, step)
            case Forget():
                return await self._forget(number, step)
            case Rebuild():
                await self._command(memory_cli.rebuild, confirm=lambda _: True)
                await self._drain()
                await self.app.schedule_maintenance(force=True)
                await self._drain()
            case StatementCheck():
                return await self._check_statement(number, step)
            case SupersededCheck():
                return await self._check_superseded(number, step)
            case EntityCheck():
                return await self._check_entity(number, step)
            case CountCheck():
                return await self._check_count(number, step)
            case AbsentCheck():
                return self._check_absent(number, step)
            case ProfileCheck():
                return await self._check_profile(number, step)
            case AnswerCheck():
                return await self._check_answer(number, step, result)
            case _:
                raise TypeError(f"unknown step {step!r}")
        return None

    async def _chat(self, session: str, text: str) -> str:
        """One turn through the orchestrator at the simulated time; returns the reply."""
        bus = self.app.bus
        parts: list[str] = []
        async with asyncio.timeout(self.turn_timeout):
            async with bus.subscribe(TextDelta, TurnEnded) as events:
                await bus.publish(UserMessage(session_id=session, text=text, ts=self.clock.now))
                async for event in events:
                    if event.session_id != session:
                        continue
                    if isinstance(event, TextDelta):
                        parts.append(event.text)
                    elif isinstance(event, TurnEnded):
                        if event.reason != "completed":
                            raise RuntimeError(f"turn ended with {event.reason}: {event.error}")
                        break
        self.clock.move_to(self.clock.now + timedelta(seconds=30))
        return "".join(parts).strip()

    async def _history(self, step: History) -> None:
        memory = self.app.memory
        assert isinstance(memory, SQLiteMemoryStore)
        skip: list[int] = []
        for message in step.messages:
            self.clock.move_to(message.at(self.clock.now))
            ids = {"session_id": step.session, "turn_id": str(uuid.uuid4()), "ts": self.clock.now}
            event = UserMessage(text=message.text, **ids)
            await memory.append_episode(event)
            await memory.append_episode(TurnEnded(reason="completed", **ids))
            if not step.extract:
                episode = await memory.episode_id(event.id)
                assert episode is not None
                skip.append(episode)
        if skip:
            await self.graph.mark_extracted(skip, at=self.clock.now)

    async def _drain(self) -> None:
        """Run jobs until none is pending (waiting out retries), extraction first."""
        app = self.app
        assert app.jobs is not None and app.runner is not None
        if await self.graph.unextracted_episodes(limit=1):
            await request_extraction(app.jobs)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.jobs_timeout
        while True:
            await app.runner.run_until_idle()
            counts = await app.jobs.counts()
            if counts[JobStatus.PENDING] == 0:
                return
            due = await app.jobs.next_due()
            wait = 1.0 if due is None else (due - datetime.now(UTC)).total_seconds()
            if loop.time() + max(wait, 0) > deadline:
                raise TimeoutError(f"background jobs still pending after {self.jobs_timeout:g}s")
            await asyncio.sleep(min(max(wait, 0.05), 30.0))

    async def _command(self, command: Any, *args: Any, **kwargs: Any) -> str:
        out = io.StringIO()
        code = await command(self.app, out, *args, **kwargs)
        text = out.getvalue()
        if code != 0:
            raise RuntimeError(f"{command.__name__} exited with {code}: {text.strip()}")
        return text

    async def _matching(self, match: Match, *, current: bool = True) -> list[Statement]:
        statements = await self.graph.list_statements(current_only=current)
        return [s for s in statements if match.matches(s.text)]

    async def _near(self, match: Match, *, current: bool = True) -> str:
        """What the graph holds near `match`, for a failed check."""
        statements = await self.graph.list_statements(current_only=current)
        now = self.clock.now
        near = [s for s in statements if match.mentions(s.text)] or statements
        if not near:
            return "no statements"
        shown = [describe_statement(s, now) for s in near[:SHOWN]]
        more = f" (+{len(near) - SHOWN} more)" if len(near) > SHOWN else ""
        return "; ".join(shown) + more

    async def _correct(self, number: int, step: Correct) -> CheckResult | None:
        found = await self._matching(step.find)
        if not found:
            return CheckResult(
                number,
                "correct",
                f"a current statement to correct {step.find.describe()}",
                False,
                "a statement to correct",
                await self._near(step.find),
            )
        target = max(found, key=lambda s: (s.confidence, s.id))
        await self._command(memory_cli.correct, target.id, step.text)
        await self._drain()
        return None

    async def _forget(self, number: int, step: Forget) -> CheckResult | None:
        found = await self._matching(step.find)
        if not found:
            return CheckResult(
                number,
                "forget",
                f"current statements to forget {step.find.describe()}",
                False,
                "statements to forget",
                await self._near(step.find),
            )
        ids = [s.id for s in found]
        await self._command(memory_cli.forget, ids, confirm=lambda _: True)
        await self._drain()
        return None

    # --- Checks -----------------------------------------------------------------

    async def _check_statement(self, number: int, step: StatementCheck) -> CheckResult:
        match = step.match
        found = [
            s
            for s in await self._matching(match, current=step.current)
            if (step.kind is None or s.kind.value in step.kind)
            and (step.origin is None or s.origin.value == step.origin)
            and (step.min_confidence is None or s.confidence >= step.min_confidence)
            and (step.max_confidence is None or s.confidence <= step.max_confidence)
            and (step.min_sources is None or len(s.sources) >= step.min_sources)
        ]
        what = [("a" if step.exists else "no") + (" current" if step.current else "")]
        if step.origin:
            what.append(step.origin)
        what.append("/".join(step.kind) if step.kind else "statement")
        what.append(match.describe())
        if step.min_confidence is not None:
            what.append(f"confidence >= {step.min_confidence:g}")
        if step.max_confidence is not None:
            what.append(f"confidence <= {step.max_confidence:g}")
        if step.min_sources is not None:
            what.append(f"at least {step.min_sources} source episodes")
        description = " ".join(what)
        passed = bool(found) == step.exists
        now = self.clock.now
        if step.exists:
            actual = (
                describe_statement(found[0], now)
                if found
                else await self._near(match, current=step.current)
            )
        else:
            actual = "; ".join(describe_statement(s, now) for s in found[:SHOWN]) or "none"
        return CheckResult(number, "statement", description, passed, description, actual)

    async def _check_superseded(self, number: int, step: SupersededCheck) -> CheckResult:
        match = step.match
        now = self.clock.now
        candidates = [
            s
            for s in await self._matching(match, current=False)
            if not s.is_current(now) and s.invalid_at is not None
        ]
        passed = bool(candidates)
        if passed and step.replaced_by is not None:
            replaced = []
            for statement in candidates:
                if statement.superseded_by is None:
                    continue
                successor = await self.graph.get_statement(statement.superseded_by)
                if successor is not None and step.replaced_by.matches(successor.text):
                    replaced.append(statement)
            candidates, passed = replaced, bool(replaced)
        description = f"an ended statement {match.describe()}"
        if step.replaced_by is not None:
            description += f", replaced by one {step.replaced_by.describe()}"
        actual = (
            describe_statement(candidates[0], now)
            if passed
            else await self._near(match, current=False)
        )
        return CheckResult(number, "superseded", description, passed, description, actual)

    async def _check_entity(self, number: int, step: EntityCheck) -> CheckResult:
        found = await self.graph.find_entities(step.name)
        passed = bool(found) == step.exists
        description = f"{'an' if step.exists else 'no'} entity called {step.name!r}"
        actual = ", ".join(f"#{e.id} {e.name} ({e.type})" for e in found) or "none"
        if not found:
            entities = await self.graph.list_entities()
            actual = "entities: " + ", ".join(e.name for e in entities[:20])
        return CheckResult(number, "entity", description, passed, description, actual)

    async def _check_count(self, number: int, step: CountCheck) -> CheckResult:
        statements = [
            s
            for s in await self.graph.list_statements()
            if step.kind is None or s.kind.value in step.kind
        ]
        n = len(statements)
        passed = n >= step.min and (step.max is None or n <= step.max)
        bound = f"between {step.min} and {step.max}" if step.max is not None else f">= {step.min}"
        kind = "/".join(step.kind) if step.kind else "current"
        description = f"{kind} statements: {bound}"
        now = self.clock.now
        listed = "; ".join(describe_statement(s, now) for s in statements[:SHOWN])
        actual = f"{n}" + (f": {listed}" if listed else "")
        return CheckResult(number, "count", description, passed, description, actual)

    def _check_absent(self, number: int, step: AbsentCheck) -> CheckResult:
        hits = find_everywhere(self.db, step.words)
        description = "absent from every table: " + ", ".join(repr(w) for w in step.words)
        actual = "; ".join(f"{place}: {sample}" for place, sample in hits[:SHOWN]) or "nowhere"
        return CheckResult(number, "absent", description, not hits, description, actual)

    async def _check_profile(self, number: int, step: ProfileCheck) -> CheckResult:
        profile = await self.graph.current_profile()
        text = profile.content if profile else ""
        description = f"the profile {step.match.describe()}"
        passed = profile is not None and step.match.matches(text)
        problems = step.match.failures(text) if profile else ["no profile"]
        actual = (text or "(no profile)") + (f" [{'; '.join(problems)}]" if problems else "")
        return CheckResult(number, "profile", description, passed, description, actual)

    async def _check_answer(
        self, number: int, step: AnswerCheck, result: ScenarioResult
    ) -> CheckResult:
        session = step.session or f"eval-ask-{next(self._asks)}"
        reply = await self._chat(session, step.question)
        result.transcript += [f"[{session}] user: {step.question}", f"lucy: {reply}"]
        memories = await self._memories(session)
        description = f"answer to {step.question!r} {step.match.describe()}"
        problems = step.match.failures(reply)
        actual = reply + (f" [{'; '.join(problems)}]" if problems else "")
        if memories is not None:
            actual += f" (memory given: {memories})"
        return CheckResult(number, "answer", description, not problems, description, actual)

    async def _memories(self, session: str) -> str | None:
        """The memory lines the last turn of `session` was given, from its trace."""
        memory = self.app.memory
        assert isinstance(memory, SQLiteMemoryStore)
        events = await memory.recent_episodes(session, 50)
        turn_ids = [e.turn_id for e in events if e.turn_id]
        if not turn_ids:
            return None
        trace = await memory.get_trace(turn_ids[-1])
        if trace is None or not trace.steps or not trace.steps[0].messages:
            return None
        # The memories are in the first system message or, with the "stable_prefix"
        # layout, in the one after the user's message.
        system = "\n\n".join(m.content for m in trace.steps[0].messages if m.role is Role.SYSTEM)
        marker = "## Things you remember"
        if marker not in system:
            return "none"
        section = system.split(marker, 1)[1].split("\n## ", 1)[0]
        lines = [line[2:] for line in section.splitlines() if line.startswith("- ")]
        return " | ".join(lines) if lines else "none"


def find_everywhere(db: Path, words: list[str]) -> list[tuple[str, str]]:
    """Where any of `words` occurs in the database: (table.column, excerpt) pairs.

    Reads every text column of every ordinary table with a plain connection.
    Virtual tables (the vector and full-text indexes) and their shadow tables are
    skipped: they index the statements and entities, which are read directly
    (a full-text index can keep a deleted term in its segments until they are
    merged; that is not checked). Matching is `has_keyword`, so it ignores case.
    """
    found: list[tuple[str, str]] = []
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
        virtual = [t for t, sql in tables if sql and sql.upper().startswith("CREATE VIRTUAL")]
        for table, _ in tables:
            if table in virtual or any(table.startswith(f"{v}_") for v in virtual):
                continue
            columns = [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]
            for column in columns:
                rows = conn.execute(f'SELECT "{column}" FROM "{table}"').fetchall()
                for (value,) in rows:
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="ignore")
                    if not isinstance(value, str):
                        continue
                    for word in words:
                        if has_keyword(value, word):
                            at = value.lower().find(word.lower())
                            excerpt = value[max(0, at - 40) : at + len(word) + 40]
                            found.append((f"{table}.{column}", " ".join(excerpt.split())))
                            break
    return found
