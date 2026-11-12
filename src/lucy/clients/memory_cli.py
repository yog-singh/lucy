"""`lucy memory ...`: inspect and control what Lucy has learned.

Each command is an async function taking a started `LucyApp` (built with
`background=False`, so nothing runs on its own), an output stream and its
arguments, and returning the exit code. Confirmation prompts go through a
`Confirm` callable, so the commands are testable without a terminal.

Ids are printed as `#12` (statements and entities have separate id spaces;
commands that take an id say which). Times are shown in local time with the
zone named ("2026-11-12 09:30 IST"); a validity date given without a time (as
extraction stores "since March" or "in 2021", at midnight UTC) is shown as a
date only.

How `correct` builds the replacement: the user gives only free text, which
cannot be parsed reliably into a predicate and an object. The new statement
keeps the old one's subject, predicate and kind (so it stays grouped with it,
and the stated-beats-inferred rule applies per subject and predicate), carries
the new text (the retrieval surface: retrieval and the profile read `text`),
and has no object: the old object or value is what was wrong. It is stated,
with `CORRECTION_CONFIDENCE` and no validity dates; the old statement is
superseded and so ends now. A rewrite of the profile is requested.

Corrections are part of the episode log, the source of truth: `correct` first
appends one turn to the session `CORRECTIONS_SESSION`, a user message (source
"correction") saying in plain words that the old statement is wrong and what
is right (`correction_text`), and a `TurnEnded`, so extraction treats it as a
complete turn. The message is marked as already extracted (the replacement is
written directly, citing it as its source), so it is not extracted again now;
`lucy memory rebuild` re-extracts it with the rest of the log, after the turn
the old statement came from, and so learns the correction again.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from datetime import time as day_start
from typing import Final, TextIO

from lucy.app import LucyApp
from lucy.core.context import ContextBuilder
from lucy.core.events import AssistantMessage, ToolFinished, TurnEnded, UserMessage
from lucy.core.gateway import Preempted
from lucy.core.jobs import Handler, Job, JobRunner, JobStatus, utc_now
from lucy.memory import fts
from lucy.memory.extraction import request_extraction
from lucy.memory.graph_store import GraphStoreError, SQLiteGraphStore
from lucy.memory.maintenance import PROFILE_JOB
from lucy.memory.models import (
    Entity,
    EpisodeRecord,
    NewStatement,
    Origin,
    Statement,
    StatementKind,
)
from lucy.memory.retrieval import Retriever, render
from lucy.memory.sqlite_jobs import SQLiteJobStore
from lucy.memory.sqlite_store import SQLiteMemoryStore
from lucy.prompts import load_prompt

Confirm = Callable[[str], bool]
"""Asks a yes/no question (the prompt text) and returns the answer."""

#: Confidence of a statement written by `lucy memory correct`: the user said so
#: explicitly, which is stronger than a statement extracted from conversation (0.9).
CORRECTION_CONFIDENCE: Final = 0.95
#: The session that `correct` logs corrections in.
CORRECTIONS_SESSION: Final = "corrections"
#: Statements listed by `forget --query` at most (unless `--limit` says otherwise).
FORGET_QUERY_LIMIT: Final = 5
#: `forget --query` keeps vector matches scoring at least the best match minus this.
#: Measured with embeddinggemma-2 (query prefix "query: "): statements unrelated
#: to a query scored 0.63 to 0.72 and related ones 0.73 to 0.83, with the right
#: statement usually more than 0.05 ahead of unrelated ones ("half marathon":
#: 0.81 against at most 0.66), so an absolute threshold cannot separate them.
#: The preview and confirmation remain the safeguard.
FORGET_SIMILARITY_MARGIN: Final = 0.05
#: `forget --query` ignores vector matches below this whatever the margin (a sanity
#: floor, far below any related statement; retrieval's own floors are stricter).
FORGET_MIN_SIMILARITY: Final = 0.3
#: Longest excerpt of an episode's text shown in previews.
EXCERPT_CHARS: Final = 160
#: Forgotten values shorter than this are not searched for in other turns.
MIN_LEFTOVER_NEEDLE: Final = 3

LEFTOVER_NOTE: Final = (
    "Known limit: turns that repeated this information without producing a statement"
    " are not removed"
)


class NotConfigured(RuntimeError):
    """The knowledge graph is not available in this configuration."""


def _require(app: LucyApp) -> tuple[SQLiteGraphStore, SQLiteJobStore, Retriever]:
    if app.graph is None or app.jobs is None or app.retriever is None:
        raise NotConfigured(
            f"memory is not configured ({app.learning_off_reason}); the knowledge graph needs"
            " an [embedder] section, the SQLite memory store and the 'extraction' and"
            " 'summary' LLM roles"
        )
    return app.graph, app.jobs, app.retriever


# --- Formatting ---------------------------------------------------------------------


def _date_only(value: datetime) -> bool:
    """Whether `value` is a bare date as stored (midnight UTC), not a moment."""
    return value.astimezone(UTC).timetz().replace(tzinfo=None) == day_start()


def _when(value: datetime) -> str:
    """A moment in local time with its zone, e.g. "2026-11-12 09:30 IST"."""
    return value.astimezone().strftime("%Y-%m-%d %H:%M %Z").rstrip()


def _day(value: datetime) -> str:
    """The local date of a moment, or the date itself for a bare date."""
    if _date_only(value):
        return value.astimezone(UTC).strftime("%Y-%m-%d")
    return value.astimezone().strftime("%Y-%m-%d")


def _moment(value: datetime) -> str:
    """A validity time: a bare date as a date, anything else as `_when`."""
    return _day(value) if _date_only(value) else _when(value)


def _excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def statement_line(statement: Statement, *, now: datetime | None = None) -> str:
    """`#12 [fact, stated, 0.90] The text.` plus the end date if it is no longer current."""
    now = now or utc_now()
    line = (
        f"#{statement.id} [{statement.kind.value}, {statement.origin.value},"
        f" {statement.confidence:.2f}] {' '.join(statement.text.split())}"
    )
    if statement.invalid_at is not None and not statement.is_current(now):
        line += f" (ended {_day(statement.invalid_at)})"
    return line


def entity_line(entity: Entity) -> str:
    line = f"#{entity.id} {entity.name} ({entity.type})"
    if entity.aliases:
        line += f", also called: {', '.join(entity.aliases)}"
    return line


def episode_text(record: EpisodeRecord) -> str:
    """One episode as "Speaker: text", for previews and provenance."""
    event = record.event
    if isinstance(event, UserMessage):
        return f"User: {event.text}"
    if isinstance(event, AssistantMessage):
        return f"Lucy: {event.text}" if event.text else "Lucy: (tool calls only)"
    if isinstance(event, ToolFinished):
        return f"Tool result: {event.result.content}"
    return f"({event.type})"


def episode_line(record: EpisodeRecord, *, full: bool = False) -> str:
    event = record.event
    text = episode_text(record)
    return (
        f"[{_when(event.ts)}, session {event.session_id!r}] "
        f"{' '.join(text.split()) if full else _excerpt(text)}"
    )


def _plural(n: int, word: str, plural: str | None = None) -> str:
    return f"{n} {word if n == 1 else (plural or word + 's')}"


def _write(out: TextIO, line: str = "") -> None:
    out.write(line + "\n")
    out.flush()


# --- status ----------------------------------------------------------------------


async def status(app: LucyApp, out: TextIO) -> int:
    """Counts of episodes, statements, entities and jobs; vectors; the profile's age."""
    graph, jobs, _ = _require(app)
    stats = await graph.stats()
    _write(out, f"Database: {graph.path}")
    _write(
        out,
        f"Episodes: {stats.episodes} ({stats.episodes_unextracted} awaiting extraction)",
    )
    _write(
        out,
        f"Statements: {stats.statements_current} current,"
        f" {stats.statements_invalidated} invalidated",
    )
    if stats.current_by_kind:
        kinds = ", ".join(
            f"{kind.value} {stats.current_by_kind[kind]}"
            for kind in StatementKind
            if kind in stats.current_by_kind
        )
        _write(out, f"  current by kind: {kinds}")
    merged = f" ({stats.entities_merged} merged away)" if stats.entities_merged else ""
    _write(out, f"Entities: {stats.entities}{merged}")

    counts = await jobs.counts()
    _write(out, "Jobs: " + ", ".join(f"{counts[s]} {s.value}" for s in JobStatus))
    for job in await jobs.list_jobs(statuses=[JobStatus.FAILED], limit=5):
        _write(out, f"  failed job {job.id} ({job.kind}): {_excerpt(job.last_error or '')}")
    for job in await jobs.list_jobs(statuses=[JobStatus.PENDING], with_error=True, limit=5):
        _write(
            out,
            f"  retrying job {job.id} ({job.kind}, {job.attempts} of {job.max_attempts}"
            f" attempts used, next {_when(job.run_after)}): {_excerpt(job.last_error or '')}",
        )

    vectors = await graph.vector_status()
    _write(
        out,
        f"Vectors missing for {vectors.model} ({vectors.dimensions} dimensions):"
        f" {_plural(vectors.statements_missing, 'statement')},"
        f" {_plural(vectors.entities_missing, 'entity', 'entities')}",
    )
    profile = await graph.current_profile()
    if profile is None:
        _write(out, "Profile: not written yet")
    else:
        _write(
            out,
            f"Profile: last written {_when(profile.created_at)}"
            f" ({_plural(stats.profile_versions, 'version')})",
        )
    return 0


# --- profile ---------------------------------------------------------------------


async def profile(app: LucyApp, out: TextIO, *, history: bool = False) -> int:
    """The current profile, or every version with `history`."""
    graph, _, _ = _require(app)
    if history:
        versions = await graph.profile_versions()
        if not versions:
            _write(out, "No profile has been written yet.")
            return 0
        for index, version in enumerate(versions):
            label = " (current)" if index == 0 else ""
            _write(out, f"Version {version.id}, {_when(version.created_at)}{label}:")
            _write(out, version.content)
            _write(out)
        return 0
    current = await graph.current_profile()
    if current is None:
        _write(out, "No profile has been written yet.")
        return 0
    _write(out, f"Profile version {current.id}, written {_when(current.created_at)}:")
    _write(out, current.content)
    return 0


# --- search ----------------------------------------------------------------------


def _memory_budget(app: LucyApp) -> int:
    """The memories budget a chat turn has (as the orchestrator computes it)."""
    window = app.gateway.for_role("chat").capabilities.context_window
    return ContextBuilder(load_prompt("persona")).token_budgets(window)["memories"]


async def search(app: LucyApp, out: TextIO, query: str, *, limit: int = 20) -> int:
    """What retrieval would give a turn whose user message is `query`."""
    graph, _, retriever = _require(app)
    budget = _memory_budget(app)
    context = await retriever.retrieve(query, budget)
    ranked = await retriever.rank(query)
    current = await graph.current_profile()
    if current is None:
        _write(out, "Profile: none yet")
    else:
        _write(out, f"Profile: version {current.id} ({len(current.content.split())} words)")
    instructions = await graph.active_instructions()
    _write(out, f"Standing instructions ({len(instructions)}, always included):")
    for statement in instructions:
        _write(out, f"  {statement_line(statement)}")
    included = set(context.statements)
    now = app.clock()
    fits = 0
    lines = []
    for rank, item in enumerate(ranked, start=1):
        rendered = render(item.statement, now=now, uncertain_below=retriever.uncertain_below)
        mark = "" if rendered in included else "  (not given to the turn)"
        fits += rendered in included
        if rank <= limit:
            lines.append(
                f"  {rank}. score {item.score:.4f}  {statement_line(item.statement)}{mark}"
            )
    cap = f", at most {retriever.max_statements}" if retriever.max_statements else ""
    _write(
        out,
        f"Retrieved statements for {query!r}: {len(ranked)} found, {fits} given to the"
        f" turn (memory budget {budget} tokens{cap})",
    )
    for line in lines:
        _write(out, line)
    if len(ranked) > limit:
        _write(out, f"  ... {len(ranked) - limit} more")
    return 0


# --- about -----------------------------------------------------------------------


async def about(app: LucyApp, out: TextIO, name: str) -> int:
    """The entities called `name`, with summary, aliases and current statements.

    Exact names and aliases first (ignoring case); failing that, entities with
    `name` as whole words in a name ("Mira" finds "Mira Okafor"); failing that,
    the nearest entities by meaning are suggested.
    """
    graph, _, _ = _require(app)
    entities = await graph.find_entities(name)
    key = fts.normalise(name)
    if not entities and key:
        pattern = re.compile(rf"(?<!\w){re.escape(key)}(?!\w)")
        entities = [
            e
            for e in await graph.list_entities()
            if any(pattern.search(fts.normalise(n)) for n in e.names)
        ]
    if not entities:
        _write(out, f"No entity is called {name!r}.")
        try:
            near = await graph.similar_entities(name, k=3, kind="query")
        except Exception as exc:
            _write(out, f"(similar names could not be searched: {type(exc).__name__}: {exc})")
            return 1
        if near:
            _write(out, "Closest by meaning:")
            for hit in near:
                _write(out, f"  {entity_line(hit.entity)} (similarity {hit.score:.2f})")
        return 1
    for index, entity in enumerate(entities):
        if index:
            _write(out)
        _write(out, entity_line(entity))
        if entity.summary:
            _write(out, f"  Summary: {entity.summary}")
        statements = await graph.neighbours(entity.id)
        _write(out, f"  Current statements ({len(statements)}):")
        for statement in statements:
            _write(out, f"    {statement_line(statement)}")
    return 0


# --- why -------------------------------------------------------------------------


async def why(app: LucyApp, out: TextIO, statement_id: int) -> int:
    """One statement in full, its supersession chain links and its source episodes."""
    graph, _, _ = _require(app)
    statement = await graph.get_statement(statement_id)
    if statement is None:
        _write(out, f"There is no statement #{statement_id}.")
        return 1
    now = app.clock()
    ids = [statement.subject_id] + ([statement.object_id] if statement.object_id else [])
    entities = {e.id: e for e in await graph.get_entities(ids)}

    def name(entity_id: int) -> str:
        entity = entities.get(entity_id)
        return f"#{entity_id} {entity.name} ({entity.type})" if entity else f"#{entity_id}"

    _write(out, f"Statement #{statement.id}: {statement.text}")
    _write(out, f"  subject:    {name(statement.subject_id)}")
    _write(out, f"  predicate:  {statement.predicate}")
    if statement.object_id is not None:
        _write(out, f"  object:     {name(statement.object_id)}")
    elif statement.object_value is not None:
        _write(out, f"  object:     {statement.object_value!r}")
    _write(out, f"  kind:       {statement.kind.value}")
    _write(out, f"  origin:     {statement.origin.value}")
    _write(
        out,
        f"  confidence: {statement.confidence:.2f}"
        f" ({_plural(statement.evidence_count, 'observation')})",
    )
    start = _moment(statement.valid_from) if statement.valid_from else "unknown"
    if statement.invalid_at is None:
        validity = f"from {start}, current"
    else:
        state = "current" if statement.is_current(now) else "no longer current"
        validity = f"from {start} until {_moment(statement.invalid_at)} ({state})"
    _write(out, f"  valid:      {validity}")
    _write(
        out,
        f"  recorded:   {_when(statement.recorded_at)};"
        f" last observed {_when(statement.last_observed_at)}",
    )
    if statement.superseded_by is not None:
        successor = await graph.get_statement(statement.superseded_by)
        text = statement_line(successor, now=now) if successor else f"#{statement.superseded_by}"
        _write(out, f"  superseded by: {text}")
    for earlier in await graph.statements_superseded_by(statement.id):
        _write(out, f"  supersedes:    {statement_line(earlier, now=now)}")
    episodes = await graph.get_episodes(statement.sources)
    if not episodes:
        _write(out, "No source episodes (it was not learned from conversation).")
        return 0
    _write(out, f"Learned from ({_plural(len(episodes), 'episode')}):")
    for record in episodes:
        _write(out, f"  {episode_line(record, full=True)}")
    return 0


# --- correct ---------------------------------------------------------------------


def correction_text(old: str, new: str) -> str:
    """The user message logged for a correction: says the old statement is wrong and
    what is right, in plain words, so extraction learns the correction from it."""
    old = " ".join(old.split()).rstrip(".")
    new = " ".join(new.split())
    if new[-1:] not in ".!?":
        new += "."
    return f'Correction: what you noted, "{old}", is wrong. {new}'


async def correct(app: LucyApp, out: TextIO, statement_id: int, text: str) -> int:
    """Replace a statement with a stated correction, logged as a user message (see the
    module docstring)."""
    graph, jobs, _ = _require(app)
    text = " ".join(text.split())
    if not text:
        _write(out, "The corrected text must not be empty.")
        return 2
    old = await graph.get_statement(statement_id)
    if old is None:
        _write(out, f"There is no statement #{statement_id}.")
        return 1
    if old.superseded_by is not None:
        _write(
            out,
            f"Statement #{old.id} was already replaced by #{old.superseded_by};"
            f" correct that one instead (`lucy memory why {old.superseded_by}`).",
        )
        return 1
    now = app.clock()
    if old.valid_from is not None and old.valid_from > now:
        _write(out, f"Cannot correct statement #{old.id}: it only becomes valid later.")
        return 1
    memory = app.memory
    if not isinstance(memory, SQLiteMemoryStore):
        raise NotConfigured("corrections need the SQLite memory store")
    ids = {"session_id": CORRECTIONS_SESSION, "turn_id": str(uuid.uuid4()), "ts": now}
    message = UserMessage(text=correction_text(old.text, text), source="correction", **ids)
    await memory.append_episode(message)
    await memory.append_episode(TurnEnded(reason="completed", **ids))
    episode = await memory.episode_id(message.id)
    assert episode is not None
    await graph.mark_extracted([episode])
    new = NewStatement(
        subject_id=old.subject_id,
        predicate=old.predicate,
        text=text,
        kind=old.kind,
        origin=Origin.STATED,
        confidence=CORRECTION_CONFIDENCE,
        observed_at=now,
    )
    try:
        replacement = await graph.supersede(old.id, new, [episode])
    except GraphStoreError as exc:
        _write(out, f"Cannot correct statement #{old.id}: {exc}")
        return 1
    await jobs.enqueue(PROFILE_JOB, {"force": True}, dedupe_key=PROFILE_JOB)
    _write(out, f"Was:  {statement_line(old, now=now)}")
    _write(out, f"Now:  {statement_line(replacement, now=now)}")
    _write(
        out,
        "The old statement is kept as no longer current. The correction is logged in"
        f" the {CORRECTIONS_SESSION!r} session, so a rebuild keeps it. A profile rewrite"
        " is queued; it runs during the next chat or with `lucy memory work`.",
    )
    return 0


# --- forget ----------------------------------------------------------------------


async def matching_statements(
    graph: SQLiteGraphStore, query: str, *, limit: int
) -> list[Statement]:
    """Current statements of any kind matching `query`, best first, at most `limit`.

    Keyword matches, plus vector matches within `FORGET_SIMILARITY_MARGIN` of the
    best vector match (and at least `FORGET_MIN_SIMILARITY`), fused by
    reciprocal rank. A relative margin, because
    absolute scores sit in a narrow band (see `FORGET_SIMILARITY_MARGIN`).
    """
    scores: dict[int, float] = {}
    statements: dict[int, Statement] = {}
    lists: list[list[Statement]] = []
    try:
        hits = await graph.similar_statements(query, k=limit * 4, kind="query")
    except Exception:
        hits = []  # the embedder is unavailable: keyword matches only
    if hits:
        floor = max(FORGET_MIN_SIMILARITY, hits[0].score - FORGET_SIMILARITY_MARGIN)
        lists.append([h.statement for h in hits if h.score >= floor])
    lists.append([h.statement for h in await graph.keyword_statements(query, k=limit * 4)])
    for ranked in lists:
        for rank, statement in enumerate(ranked, start=1):
            statements[statement.id] = statement
            scores[statement.id] = scores.get(statement.id, 0.0) + 1 / (60 + rank)
    order = sorted(scores, key=lambda sid: (-scores[sid], sid))
    return [statements[sid] for sid in order][:limit]


async def _leftovers(
    graph: SQLiteGraphStore,
    statements: Sequence[Statement],
    removed_entities: Sequence[Entity],
    deleted_episodes: set[int],
) -> dict[str, list[EpisodeRecord]]:
    """Other episodes still mentioning a forgotten value or entity name."""
    needles = {s.object_value.strip() for s in statements if s.object_value}
    for entity in removed_entities:
        needles.update(entity.names)
    found: dict[str, list[EpisodeRecord]] = {}
    for needle in sorted(n for n in needles if len(n.strip()) >= MIN_LEFTOVER_NEEDLE):
        records = [
            r
            for r in await graph.episodes_mentioning(needle, limit=20)
            if r.id not in deleted_episodes
        ]
        if records:
            found[needle] = records
    return found


async def forget(
    app: LucyApp,
    out: TextIO,
    statement_ids: Sequence[int] = (),
    *,
    query: str | None = None,
    limit: int = FORGET_QUERY_LIMIT,
    confirm: Confirm,
) -> int:
    """Delete statements and the turns they came from, after a preview and confirmation."""
    graph, jobs, _ = _require(app)
    ids = list(dict.fromkeys(statement_ids))
    if query is not None:
        matches = await matching_statements(graph, query, limit=limit)
        if not matches:
            _write(out, f"No current statement matches {query!r}.")
            return 1
        _write(out, f"Current statements matching {query!r}:")
        for statement in matches:
            _write(out, f"  {statement_line(statement)}")
        _write(out, "(To forget only some of them, pass their ids instead of --query.)")
        ids.extend(s.id for s in matches if s.id not in ids)
    if not ids:
        _write(out, "Nothing to forget: give statement ids or --query TEXT.")
        return 2

    preview = await graph.forget_statements(ids, dry_run=True)
    missing = [i for i in ids if i not in preview.statement_ids]
    if missing:
        _write(out, "No such statement: " + ", ".join(f"#{i}" for i in missing))
    if not preview.statement_ids:
        return 1
    statements = await graph.get_statements(preview.statement_ids)
    episodes = await graph.get_episodes(preview.episode_ids)
    removed = await graph.get_entities(preview.entity_ids)
    _write(out, "This will permanently delete:")
    _write(out, f"  {_plural(len(statements), 'statement')}:")
    for statement in statements:
        also = "" if statement.id in ids else "  (left without a source)"
        _write(out, f"    {statement_line(statement)}{also}")
    _write(out, f"  {_plural(len(episodes), 'episode')} (every event of the turns they came from):")
    for record in episodes:
        _write(out, f"    {episode_line(record)}")
    _write(out, f"  {_plural(len(preview.trace_turn_ids), 'turn trace')}")
    if removed:
        names = ", ".join(entity_line(e) for e in removed)
        count = _plural(len(removed), "entity", "entities")
        _write(out, f"  {count} left with no statements: {names}")
    if preview.cleared_summary_entity_ids:
        count = _plural(len(preview.cleared_summary_entity_ids), "entity", "entities")
        _write(out, f"  the summaries of {count} (rewritten later)")
    if preview.reopened_statement_ids:
        reopened = ", ".join(f"#{i}" for i in preview.reopened_statement_ids)
        _write(out, f"  and reopen as current the statements they had replaced: {reopened}")
    if preview.repointed_statement_ids:
        repointed = ", ".join(f"#{i}" for i in preview.repointed_statement_ids)
        _write(out, f"  and repoint the replacement links of: {repointed}")
    _write(
        out,
        f"  {_plural(preview.profile_versions_deleted, 'profile version')}"
        " written since (the profile will be rewritten)",
    )
    leftovers = await _leftovers(graph, statements, removed, set(preview.episode_ids))
    if leftovers:
        _write(out, f"{LEFTOVER_NOTE}. Still mentioned in other turns:")
        for needle, records in leftovers.items():
            _write(out, f"  {needle!r}: {_plural(len(records), 'episode')}, e.g.")
            for record in records[:2]:
                _write(out, f"    {episode_line(record)}")
        _write(out, "  A rebuild could learn it again from them.")

    if not confirm("Delete all of this? [y/N] "):
        _write(out, "Nothing was deleted.")
        return 1
    result = await graph.forget_statements(ids)
    await jobs.enqueue(PROFILE_JOB, {"force": True}, dedupe_key=PROFILE_JOB)
    _write(
        out,
        f"Forgotten: {_plural(len(result.statement_ids), 'statement')},"
        f" {_plural(len(result.episode_ids), 'episode')},"
        f" {_plural(len(result.trace_turn_ids), 'trace')},"
        f" {_plural(len(result.entity_ids), 'entity', 'entities')},"
        f" {_plural(result.profile_versions_deleted, 'profile version')}."
        " A profile rewrite is queued (next chat, or `lucy memory work`).",
    )
    return 0


# --- work ------------------------------------------------------------------------


def _reporting(kind: str, handler: Handler, out: TextIO) -> Handler:
    async def run(job: Job) -> None:
        started = time.monotonic()
        try:
            await handler(job)
        except BaseException as exc:
            elapsed = time.monotonic() - started
            final = job.attempts >= job.max_attempts or not isinstance(exc, Exception)
            outcome = "failed" if final else "failed, will retry"
            _write(
                out,
                f"job {job.id} {kind}: {outcome} after {elapsed:.1f}s:"
                f" {_excerpt(f'{type(exc).__name__}: {exc}')}",
            )
            raise
        _write(out, f"job {job.id} {kind}: done in {time.monotonic() - started:.1f}s")

    return run


async def _drain(app: LucyApp, jobs: SQLiteJobStore, out: TextIO) -> int:
    handlers = {kind: _reporting(kind, h, out) for kind, h in app.job_handlers.items()}
    runner = JobRunner(jobs, handlers, defer_on=(Preempted,))
    return await runner.run_until_idle()


async def work(app: LucyApp, out: TextIO, *, maintenance: bool = False) -> int:
    """Run background jobs in the foreground until none is due.

    Due maintenance is scheduled first (and extraction requested if episodes
    wait for it). With `maintenance`, every maintenance job is then run once
    more after the queue has drained, so the profile and summaries reflect
    everything extracted. Do not run this while `lucy chat` uses the same database.
    """
    graph, jobs, _ = _require(app)
    recovered = await jobs.recover()
    if recovered:
        _write(out, f"Put back {_plural(recovered, 'interrupted job')}.")
    await app.schedule_maintenance()
    completed = await _drain(app, jobs, out)
    if maintenance:
        await app.schedule_maintenance(force=True)
        completed += await _drain(app, jobs, out)
    counts = await jobs.counts()
    waiting = counts[JobStatus.PENDING]
    stats = await graph.stats()
    summary = f"{_plural(completed, 'job')} completed"
    if waiting:
        due = await jobs.next_due()
        when = f" (next at {_when(due)})" if due else ""
        summary += f"; {_plural(waiting, 'job')} waiting to retry{when}"
    if counts[JobStatus.FAILED]:
        summary += f"; {counts[JobStatus.FAILED]} failed in total (see `lucy memory status`)"
    _write(out, summary + ".")
    _write(
        out,
        f"Memory now: {stats.statements_current} current statements,"
        f" {stats.episodes_unextracted} episodes awaiting extraction.",
    )
    return 0


# --- rebuild ---------------------------------------------------------------------


async def rebuild(app: LucyApp, out: TextIO, *, confirm: Confirm) -> int:
    """Clear the graph and re-extract the whole episode log with the current models."""
    graph, jobs, _ = _require(app)
    stats = await graph.stats()
    _write(
        out,
        f"This deletes all {stats.statements_current + stats.statements_invalidated}"
        f" statements, {stats.entities - 1} entities (the user entity is reset) and"
        f" {_plural(stats.profile_versions, 'profile version')}, then re-extracts all"
        f" {stats.episodes} episodes with the current models. Episodes and traces are kept."
        " Corrections made with `lucy memory correct` are in the episode log and are"
        " learned again (how well depends on the extraction model).",
    )
    if not confirm("Rebuild the knowledge graph? [y/N] "):
        _write(out, "Nothing was changed.")
        return 1
    result = await graph.reset_graph()
    await request_extraction(jobs)
    _write(
        out,
        f"Cleared {_plural(result.statements_deleted, 'statement')} and"
        f" {_plural(result.entities_deleted, 'entity', 'entities')};"
        f" {_plural(result.episodes_reset, 'episode')} will be re-extracted."
        " Run `lucy memory work --maintenance` (or chat) to rebuild; --maintenance"
        " also rewrites the profile and summaries at the end.",
    )
    return 0
