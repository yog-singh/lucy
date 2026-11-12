"""Periodic ("nightly") memory maintenance that needs no new inference about the user
(docs/memory.md "3. Reflect" housekeeping, "4. Decay", "5. Rewrite the profile").

- `refresh_entity_summaries`: rewrite the one- or two-sentence summary of
  entities whose current statements changed since their summary was written.
- `merge_duplicate_entities`: find entities that look like duplicates, ask the
  model, and merge confirmed pairs.
- Job handlers for these, for decay, the profile rewrite and re-embedding, and
  `schedule_maintenance`, which enqueues whatever is due.

Staleness of summaries
----------------------
`GraphStore.entity_statement_fingerprints` gives, per entity, the number
of its current statements (as subject or object) and a fingerprint of that set
(count, sum and maximum of their ids), in one query per page of entities. When
a summary is written, the fingerprint and the entity's name are stored under
`derived.summary.e.<id>`; the entity is stale when they differ now, or when it
has statements but no summary (e.g. "forget" cleared it). Confidence changes
(decay, reinforcement) do not make a summary stale: they do not change what is
true about the entity. Scanning resumes from `derived.summary.cursor` and wraps
around, so a run bounded by `limit` continues where the last one stopped.

Duplicate entities
------------------
Candidates for an entity are entities of the same type that either share a
name or alias with it (`find_entities`), or are among its nearest entities by
vector (`similar_entities_by_vector`) and have a name or alias whose own
embedding is at least `name_similarity_threshold` similar to one of its names
(see `merge_candidates` for why names are compared alone). Each pair is
put to the model with both entities' names, aliases, summaries and a few
statements; on "same", the entity with fewer current statements is merged into
the other (ties: the older one survives). The user entity is never merged away:
it is always the target, and only after the model's check, like every merge.
A rejected pair is remembered under `derived.merge.rejected.<a>.<b>` with both
entities' statement fingerprints, and is asked again only once either
fingerprint changes (a statement added, removed or moved), since that is new
evidence.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Final

from pydantic import BaseModel, Field

from lucy.core.gateway import LLMGateway, Priority
from lucy.core.jobs import Handler, Job, JobStatus, JobStore, utc_now
from lucy.core.types import Message, Role
from lucy.memory.decay import DecayPolicy, decay_inferred
from lucy.memory.graph_store import DERIVED_META_PREFIX
from lucy.memory.models import USER_ENTITY_ID, Entity, Origin, Statement, StatementKind
from lucy.memory.ports import GraphStore
from lucy.memory.profile import (
    ProfileWriter,
    complete_text,
    plain_text,
    sentences,
    truncate_to_tokens,
    unsupported_terms,
)
from lucy.memory.reflection import REFLECT_JOB
from lucy.memory.sqlite_jobs import SQLiteJobStore
from lucy.prompts import load_prompt

log = logging.getLogger(__name__)

# --- Job kinds and meta keys ----------------------------------------------------

DECAY_JOB: Final = "memory.decay"
PROFILE_JOB: Final = "memory.profile"
SUMMARIES_JOB: Final = "memory.summaries"
MERGE_JOB: Final = "memory.merge_entities"
REEMBED_JOB: Final = "memory.reembed"
#: Interval-scheduled maintenance, in the order it should run: decay first (it
#: changes confidences), merges before reflection and summaries (a merge
#: changes the survivor's statements), reflection (`lucy.memory.reflection`)
#: before summaries and the profile (so they include what it inferred), and
#: the profile last.
SCHEDULED_JOBS: Final = (DECAY_JOB, MERGE_JOB, REFLECT_JOB, SUMMARIES_JOB, PROFILE_JOB)
#: The re-embed job continues itself under this dedupe key while its first run
#: (keyed by `REEMBED_JOB`) is still running, and vice versa.
REEMBED_CONTINUATION_KEY: Final = f"{REEMBED_JOB}.continue"

SUMMARY_PREFIX: Final = f"{DERIVED_META_PREFIX}summary.e."
SUMMARY_CURSOR_KEY: Final = f"{DERIVED_META_PREFIX}summary.cursor"
MERGE_REJECTED_PREFIX: Final = f"{DERIVED_META_PREFIX}merge.rejected."
MERGE_CURSOR_KEY: Final = f"{DERIVED_META_PREFIX}merge.cursor"
#: When each scheduled kind was last enqueued (not derived: kept by reset_graph).
SCHEDULE_PREFIX: Final = "maintenance.enqueued."

#: Name-to-name cosine similarity at or above which two entities are put to the
#: model. Measured with embeddinggemma-2 ("passage: " prefix): name variants of
#: one thing scored 0.90 to 0.98 (Priya / my sister Priya 0.906, Bob / Robert
#: 0.901), unrelated names at most 0.887 (Priya / Pooja), and look-alikes that
#: are different things 0.91 to 0.96 (Rohan / Rohan Iyer 0.910, Priya Sharma /
#: Priya Nair 0.951): those are what the model check is for. Relations without a
#: shared name (my wife / Meera, 0.825) are left to aliases. Specific to the
#: embedder; re-measure when it changes.
NAME_SIMILARITY_THRESHOLD: Final = 0.9


def summary_key(entity_id: int) -> str:
    return f"{SUMMARY_PREFIX}{entity_id:012d}"


def rejected_key(a: int, b: int) -> str:
    low, high = sorted((a, b))
    return f"{MERGE_REJECTED_PREFIX}{low:012d}.{high:012d}"


def _usable(statement: Statement, min_inferred_confidence: float) -> bool:
    return statement.kind is not StatementKind.INSTRUCTION and not (
        statement.origin is Origin.INFERRED and statement.confidence < min_inferred_confidence
    )


class _Scanner:
    """Walks surviving entities by id from a saved cursor, wrapping around once."""

    def __init__(self, graph: GraphStore, cursor_key: str, page_size: int) -> None:
        self._graph = graph
        self._key = cursor_key
        self._page = page_size
        self.start = 0
        self.wrapped = False
        self.exhausted = False

    async def load(self) -> None:
        raw = await self._graph.get_meta(self._key)
        self.start = int(raw) if raw and raw.isdigit() else 0

    async def pages(self) -> AsyncIterator[list[Entity]]:
        """Yield pages of entities: ids after the cursor, then (wrapped) up to it."""
        after = self.start
        while True:
            page = await self._graph.list_entities(after_id=after, limit=self._page)
            if self.wrapped:
                page = [e for e in page if e.id <= self.start]
            if not page:
                if self.wrapped or self.start == 0:
                    self.exhausted = True
                    return
                self.wrapped, after = True, 0
                continue
            yield page
            after = page[-1].id

    async def save(self, entity_id: int) -> None:
        await self._graph.set_meta(self._key, str(entity_id))


# --- Entity summaries -----------------------------------------------------------


@dataclass(frozen=True)
class SummaryPolicy:
    """How entity summaries are written."""

    statements: int = 12
    """Neighbour statements given to the model, strongest first."""
    max_tokens: int = 80
    max_sentences: int = 2
    min_inferred_confidence: float = 0.6
    scan_limit: int = 2000
    """Entities examined per run at most (the fingerprint check is cheap)."""
    page_size: int = 200


@dataclass(frozen=True)
class SummaryResult:
    written: list[int] = field(default_factory=list)
    cleared: list[int] = field(default_factory=list)
    """Entities whose summary was removed because they have no usable statements."""
    flagged: dict[int, list[str]] = field(default_factory=dict)
    """Per written entity: capitalised words or numbers found in none of its statements."""
    scanned: int = 0
    complete: bool = False
    """True when every entity was examined (nothing stale is left unless just changed)."""


def _summary_input(entity: Entity, statements: Sequence[Statement]) -> str:
    names = entity.name
    if entity.aliases:
        names += f" (also called: {', '.join(entity.aliases)})"
    lines = [f"Name: {names}", f"Type: {entity.type}", "Statements:"]
    lines += [f"- {s.text.strip()}" for s in statements]
    return "\n".join(lines) + "\n"


def _clip_sentences(text: str, max_sentences: int, max_tokens: int) -> str:
    return truncate_to_tokens(" ".join(sentences(text)[:max_sentences]), max_tokens)


async def write_entity_summary(
    gateway: LLMGateway,
    entity: Entity,
    statements: Sequence[Statement],
    *,
    role: str = "summary",
    policy: SummaryPolicy | None = None,
) -> tuple[str, list[str], int]:
    """Generate a summary of `entity` from `statements`: (text, unsupported terms, attempts).

    Retries once when the text mentions unsupported terms; keeps the attempt
    with fewer. `Preempted` propagates.
    """
    policy = policy or SummaryPolicy()
    sources = [s.text for s in statements] + entity.names
    messages = [
        Message(role=Role.SYSTEM, content=load_prompt("entity_summary")),
        Message(role=Role.USER, content=_summary_input(entity, statements)),
    ]
    first = _clip_sentences(
        plain_text(await complete_text(gateway, role, messages)),
        policy.max_sentences,
        policy.max_tokens,
    )
    flags = unsupported_terms(first, sources)
    if first and not flags:
        return first, [], 1
    problem = (
        "It mentions " + ", ".join(flags) + ", which appear in none of the statements."
        if first
        else "It was empty."
    )
    retry = [
        *messages,
        Message(role=Role.ASSISTANT, content=first),
        Message(
            role=Role.USER,
            content=f"{problem} Write one or two sentences using only the statements.",
        ),
    ]
    second = _clip_sentences(
        plain_text(await complete_text(gateway, role, retry)),
        policy.max_sentences,
        policy.max_tokens,
    )
    second_flags = unsupported_terms(second, sources)
    if second and (not first or len(second_flags) <= len(flags)):
        return second, second_flags, 2
    return first, flags, 2


async def refresh_entity_summaries(
    gateway: LLMGateway,
    graph: GraphStore,
    *,
    role: str = "summary",
    limit: int = 20,
    policy: SummaryPolicy | None = None,
) -> SummaryResult:
    """Rewrite the summaries of up to `limit` stale entities (see the module doc).

    The user entity is skipped (the profile describes the user). An entity with
    no usable current statements (instructions and uncertain inferences do not
    count) gets no summary; an existing one is removed. Progress is saved after
    each entity, so an interrupted run (`Preempted` propagates) loses at most
    the summary being written.
    """
    policy = policy or SummaryPolicy()
    scanner = _Scanner(graph, SUMMARY_CURSOR_KEY, policy.page_size)
    await scanner.load()
    written: list[int] = []
    cleared: list[int] = []
    flagged: dict[int, list[str]] = {}
    scanned = 0
    stopped = False
    async for page in scanner.pages():
        prints = await graph.entity_statement_fingerprints(e.id for e in page)
        stored = await graph.get_meta_many(summary_key(e.id) for e in page)
        for entity in page:
            if len(written) >= limit or scanned >= policy.scan_limit:
                stopped = True
                break
            scanned += 1
            if entity.id == USER_ENTITY_ID or entity.id not in prints:
                continue
            count, fingerprint = prints[entity.id]
            signature = f"{fingerprint}|{entity.name}"
            key = summary_key(entity.id)
            if stored.get(key) == signature and (entity.summary is not None or count == 0):
                continue
            statements = []
            if count:
                statements = [
                    s
                    for s in await graph.neighbours(entity.id, limit=policy.statements * 3)
                    if _usable(s, policy.min_inferred_confidence)
                ][: policy.statements]
            if not statements:
                if entity.summary is not None:
                    await graph.update_entity_summary(entity.id, None)
                    cleared.append(entity.id)
                await graph.write_meta({key: signature})
                await scanner.save(entity.id)
                continue
            text, flags, _ = await write_entity_summary(
                gateway, entity, statements, role=role, policy=policy
            )
            if flags:
                log.warning("summary of entity %d mentions unsupported terms: %s", entity.id, flags)
                flagged[entity.id] = flags
            if text:
                await graph.update_entity_summary(entity.id, text)
                written.append(entity.id)
            await graph.write_meta({key: signature})
            await scanner.save(entity.id)
        if stopped:
            break
    return SummaryResult(
        written=written,
        cleared=cleared,
        flagged=flagged,
        scanned=scanned,
        complete=not stopped,
    )


# --- Duplicate entities ---------------------------------------------------------


class MergeDecision(BaseModel):
    """The model's verdict on a candidate pair."""

    reason: str = Field(description="One short sentence.")
    same: bool = Field(description="True if A and B are the same real-world thing.")


@dataclass(frozen=True)
class MergePolicy:
    """How duplicate entities are found and checked."""

    name_similarity_threshold: float = NAME_SIMILARITY_THRESHOLD
    similar_k: int = 20
    """Nearest entities (by stored name+summary vector) shortlisted per name."""
    statements: int = 5
    """Statements shown to the model per entity."""
    scan_limit: int = 500
    page_size: int = 100


@dataclass(frozen=True)
class MergeResult:
    merged: list[tuple[int, int]] = field(default_factory=list)
    """(merged-away id, surviving id) per merge."""
    rejected: list[tuple[int, int]] = field(default_factory=list)
    asked: int = 0
    scanned: int = 0
    complete: bool = False


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


def _same_type(a: Entity, b: Entity) -> bool:
    return a.type.strip().lower() == b.type.strip().lower()


async def merge_candidates(
    graph: GraphStore, entity: Entity, *, policy: MergePolicy | None = None
) -> list[Entity]:
    """Surviving entities of the same type that may be the same as `entity`.

    Those sharing a name or alias with it first; then, from the
    `policy.similar_k` nearest entities to each of its names
    (`similar_entities_by_vector`, the shortlist), those whose best
    name-to-name similarity (any name or alias of one against any of the
    other, embedded alone) is at least `policy.name_similarity_threshold`,
    best first.

    The threshold is applied to names alone because stored entity vectors
    include the summary: measured with embeddinggemma-2, a bare name scores
    0.70 to 0.81 against every person's name+summary vector, duplicates or
    not, while name-to-name scores separate them (see `NAME_SIMILARITY_THRESHOLD`).
    """
    policy = policy or MergePolicy()
    found: dict[int, Entity] = {}
    for name in entity.names:
        for other in await graph.find_entities(name):
            if other.id != entity.id and _same_type(other, entity):
                found.setdefault(other.id, other)
    vectors = await graph.embedder.embed(entity.names, kind="document")
    shortlist: dict[int, Entity] = {}
    for vector in vectors:
        for hit in await graph.similar_entities_by_vector(vector, policy.similar_k + 1):
            other = hit.entity
            if other.id != entity.id and other.id not in found and _same_type(other, entity):
                shortlist.setdefault(other.id, other)
    if not shortlist:
        return list(found.values())
    their_names = [(other, name) for other in shortlist.values() for name in other.names]
    their_vectors = await graph.embedder.embed([n for _, n in their_names], kind="document")
    scored: dict[int, tuple[float, Entity]] = {}
    for (other, _), theirs in zip(their_names, their_vectors, strict=True):
        score = max(_cosine(mine, theirs) for mine in vectors)
        if score >= policy.name_similarity_threshold and (
            other.id not in scored or scored[other.id][0] < score
        ):
            scored[other.id] = (score, other)
    ranked = sorted(scored.values(), key=lambda item: (-item[0], item[1].id))
    return [*found.values(), *(other for _, other in ranked)]


def _describe(label: str, entity: Entity, statements: Sequence[Statement]) -> str:
    lines = [f'{label}: name "{entity.name}"']
    if entity.aliases:
        lines.append("other names: " + ", ".join(f'"{a}"' for a in entity.aliases))
    lines.append(f"type: {entity.type}")
    if entity.summary:
        lines.append(f"description: {entity.summary}")
    lines.append("statements:")
    lines += [f"- {s.text.strip()}" for s in statements] or ["- (none)"]
    return "\n".join(lines)


async def ask_same_entity(
    gateway: LLMGateway,
    graph: GraphStore,
    a: Entity,
    b: Entity,
    *,
    role: str = "extraction",
    statements: int = 5,
) -> MergeDecision:
    """Ask the `role` model whether `a` and `b` are the same thing. `Preempted` propagates."""
    shown = [await graph.neighbours(e.id, limit=statements) for e in (a, b)]
    content = (
        _describe("A", a, shown[0])
        + "\n\n"
        + _describe("B", b, shown[1])
        + "\n\nAre A and B the same real-world thing?"
    )
    messages = [
        Message(role=Role.SYSTEM, content=load_prompt("entity_merge")),
        Message(role=Role.USER, content=content),
    ]
    return await gateway.generate(role, messages, MergeDecision, priority=Priority.BACKGROUND)


async def merge_duplicate_entities(
    gateway: LLMGateway,
    graph: GraphStore,
    *,
    role: str = "extraction",
    limit: int = 10,
    policy: MergePolicy | None = None,
) -> MergeResult:
    """Find duplicate entities, confirm each pair with the model, merge confirmed ones.

    At most `limit` pairs are put to the model per run; scanning resumes from
    `derived.merge.cursor`. Each decision is saved before the next question,
    so an interrupted run (`Preempted` propagates) repeats at most one
    question. See the module doc for candidates, direction and memory of
    rejected pairs.
    """
    policy = policy or MergePolicy()
    scanner = _Scanner(graph, MERGE_CURSOR_KEY, policy.page_size)
    await scanner.load()
    merged: list[tuple[int, int]] = []
    rejected: list[tuple[int, int]] = []
    asked = scanned = 0
    stopped = False
    gone: set[int] = set()
    async for page in scanner.pages():
        for entity in page:
            if asked >= limit or scanned >= policy.scan_limit:
                stopped = True
                break
            scanned += 1
            if entity.id in gone:
                continue
            current = await graph.get_entity(entity.id)
            if current is None or current.merged_into is not None:
                continue
            for other in await merge_candidates(graph, current, policy=policy):
                if asked >= limit:
                    stopped = True
                    break
                if other.id in gone:
                    continue
                prints = await graph.entity_statement_fingerprints([current.id, other.id])
                if current.id not in prints or other.id not in prints:
                    continue
                pair = {current.id: prints[current.id][1], other.id: prints[other.id][1]}
                key = rejected_key(current.id, other.id)
                raw = await graph.get_meta(key)
                if raw is not None and json.loads(raw) == {str(k): v for k, v in pair.items()}:
                    continue
                asked += 1
                decision = await ask_same_entity(
                    gateway, graph, current, other, role=role, statements=policy.statements
                )
                if not decision.same:
                    log.info(
                        "entities %d and %d are different: %s",
                        current.id,
                        other.id,
                        decision.reason,
                    )
                    await graph.set_meta(key, json.dumps({str(k): v for k, v in pair.items()}))
                    rejected.append((current.id, other.id))
                    continue
                source, target = _direction(current, other, prints)
                log.info("merging entity %d into %d: %s", source, target, decision.reason)
                survivor = await graph.merge_entities(source, target)
                await graph.delete_meta(key)
                merged.append((source, target))
                gone.add(source)
                if source == current.id:
                    break
                current = survivor
            if stopped:
                break
            await scanner.save(entity.id)
        if stopped:
            break
    return MergeResult(
        merged=merged, rejected=rejected, asked=asked, scanned=scanned, complete=not stopped
    )


def _direction(a: Entity, b: Entity, prints: dict[int, tuple[int, str]]) -> tuple[int, int]:
    """(source, target): the user always survives; else the one with more current
    statements survives, ties going to the older (lower id)."""
    if USER_ENTITY_ID in (a.id, b.id):
        other = b.id if a.id == USER_ENTITY_ID else a.id
        return other, USER_ENTITY_ID
    ranked = sorted((a, b), key=lambda e: (-prints[e.id][0], e.id))
    return ranked[1].id, ranked[0].id


# --- Jobs -----------------------------------------------------------------------


def decay_handler(
    graph: GraphStore,
    *,
    policy: DecayPolicy | None = None,
    clock: Callable[[], datetime] = utc_now,
) -> Handler:
    """Handler for `DECAY_JOB`: `decay_inferred` at the clock's time."""

    async def handle(job: Job) -> None:
        await decay_inferred(graph, now=clock(), policy=policy)

    return handle


def profile_handler(writer: ProfileWriter, *, graph: GraphStore | None = None) -> Handler:
    """Handler for `PROFILE_JOB`; payload `{"force": true}` rewrites even if unchanged.

    With `graph`, a run that found nothing to write (an empty graph) does not
    count as the scheduled run: its schedule entry is removed, so the next
    `schedule_maintenance` enqueues the profile again instead of a day later.
    """

    async def handle(job: Job) -> None:
        result = await writer.rewrite(force=bool(job.payload.get("force", False)))
        if result.status == "empty" and graph is not None:
            await graph.delete_meta(f"{SCHEDULE_PREFIX}{PROFILE_JOB}")

    return handle


async def request_profile(job_store: SQLiteJobStore, *, delay: float = 0.0) -> Job:
    """Ask for a profile rewrite `delay` seconds from now (real time, as the queue uses).

    Deduplicated with the scheduled profile job: if one is already pending, it is
    pushed back to that time (never brought forward), so repeated requests while
    the user is active give one rewrite once they have been quiet for `delay`.
    """
    due = utc_now() + timedelta(seconds=delay)
    job = await job_store.enqueue(PROFILE_JOB, {}, dedupe_key=PROFILE_JOB, run_after=due)
    if job.status is JobStatus.PENDING and job.run_after < due:
        await job_store.postpone(job.id, due)
    return job


def summaries_handler(
    gateway: LLMGateway,
    graph: GraphStore,
    *,
    role: str = "summary",
    limit: int = 20,
    policy: SummaryPolicy | None = None,
) -> Handler:
    """Handler for `SUMMARIES_JOB`: one bounded `refresh_entity_summaries` run."""

    async def handle(job: Job) -> None:
        await refresh_entity_summaries(gateway, graph, role=role, limit=limit, policy=policy)

    return handle


def merge_handler(
    gateway: LLMGateway,
    graph: GraphStore,
    *,
    role: str = "extraction",
    limit: int = 10,
    policy: MergePolicy | None = None,
) -> Handler:
    """Handler for `MERGE_JOB`: one bounded `merge_duplicate_entities` run."""

    async def handle(job: Job) -> None:
        await merge_duplicate_entities(gateway, graph, role=role, limit=limit, policy=policy)

    return handle


def reembed_handler(graph: GraphStore, job_store: JobStore, *, batch: int = 256) -> Handler:
    """Handler for `REEMBED_JOB`: embed up to `batch` missing vectors, then enqueue a
    continuation while vectors are still missing and this run made progress.

    The continuation alternates between two dedupe keys, because a running
    job's own key cannot be enqueued again until it completes.
    """

    async def handle(job: Job) -> None:
        stored = await graph.reembed_missing(batch)
        status = await graph.vector_status()
        if stored and (status.statements_missing or status.entities_missing):
            key = REEMBED_CONTINUATION_KEY if job.dedupe_key == REEMBED_JOB else REEMBED_JOB
            await job_store.enqueue(REEMBED_JOB, {}, dedupe_key=key)

    return handle


async def schedule_maintenance(
    job_store: JobStore,
    graph: GraphStore,
    *,
    now: datetime,
    interval: timedelta = timedelta(hours=24),
) -> list[Job]:
    """Enqueue the maintenance jobs that are due and return them, in run order.

    Each kind in `SCHEDULED_JOBS` is due when it was last enqueued at least
    `interval` before `now` (or never, or apparently in the future); the
    enqueue time is kept in `memory_meta` under `SCHEDULE_PREFIX + kind`.
    `REEMBED_JOB` is enqueued whenever vectors are missing, regardless of
    the interval. Dedupe keys (the kind) mean a pending job is never
    duplicated: enqueueing returns it instead. All jobs are due at `now` and
    enqueued in order (re-embed, decay, merge, reflect, summaries, profile), so a
    runner that takes due jobs in id order runs them in that order.

    For an app that is not always running, call this at startup and then
    periodically; "nightly" means at most once per `interval`.
    """
    jobs: list[Job] = []
    status = await graph.vector_status()
    if status.statements_missing or status.entities_missing:
        jobs.append(await job_store.enqueue(REEMBED_JOB, {}, dedupe_key=REEMBED_JOB, run_after=now))
    keys = [f"{SCHEDULE_PREFIX}{kind}" for kind in SCHEDULED_JOBS]
    last = await graph.get_meta_many(keys)
    enqueued: dict[str, str | None] = {}
    for kind, key in zip(SCHEDULED_JOBS, keys, strict=True):
        if key in last:
            try:
                previous = datetime.fromisoformat(last[key])
            except ValueError:
                previous = None
            if previous is not None and previous <= now and now - previous < interval:
                continue
        jobs.append(await job_store.enqueue(kind, {}, dedupe_key=kind, run_after=now))
        enqueued[key] = now.isoformat()
    await graph.write_meta(enqueued)
    return jobs
