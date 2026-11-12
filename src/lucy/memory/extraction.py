"""Extraction: turning what the user said into candidate statements (docs/memory.md "1. Extract").

After each conversation turn, a background job (kind `"memory.extract"`) reads
the turns whose episodes are not yet extracted, oldest turn first, and for each:

1. builds a prompt (`extraction`) with today's date and the turn's date, what
   the user is called, the known entities mentioned in the turn, about six
   earlier content episodes of the session as context, and the turn itself
   (the user's messages, Lucy's replies and tool results);
2. asks the `extraction` role for candidates in one structured call;
3. renames the user entity if the user stated their name;
4. drops candidates that fail value sanity checks or the rules below (logged
   with the reason);
5. resolves subject and object entities (`resolution.EntityResolver`);
6. reconciles each candidate (`reconcile.Reconciler`), with the turn's user
   messages as the pass's episodes (so siblings are not reconciled against
   each other);
7. checks that every entity created in the turn is connected to the user or
   to something already known, and logs a warning for each one that is not
   (`TurnReport.unlinked`; nothing is dropped or invented);
8. once the user's name is known, rewrites stored texts that still say "The
   user" (`rewrite_user_texts`);
9. marks every content episode of the turn extracted, last.

Statements come only from what the user said: the source episodes of every
statement are the turn's user-message episodes. Lucy's replies and tool
results are shown for understanding only.

Rules enforced in code, whatever the model says:

- Extraction never infers. Candidates with origin "inferred" are dropped:
  a habit or preference concluded from one turn ("prefers vegetarian food"
  after one lasagne) is a guess, and repeated choices are reflection's job.
  In the live evaluation every expected item was stated, and every inferred
  candidate was wrong or trivial.
- A plan is not a goal. A goal whose text or predicate says the user plans,
  is going to or intends to do something is dropped; a goal is what the user
  wants to achieve or is working towards. A plan produces nothing.
- A change is stored as the new state ("Tobias lives in Porto"), and that is
  what replaces the old state. A change event ("moved to Porto", "switched to
  decaf", "left Hartley", "started cycling"; by its predicate, or a text such
  as "has switched to") is reconciled after the turn's other candidates, so
  the new state supersedes the old one first, and it is dropped when another
  candidate of the turn has the same subject and object (it only repeats the
  new state). Otherwise it is reconciled as a change (`change=True`): the
  reconciler looks wider for the state it replaces and asks the model, in its
  one call, for the new state in the form of the old one ("The user drinks
  decaf coffee"), which is what is stored. The model often writes a change
  event although the prompt asks for the new state, so this does not rely on
  the prompt.

A turn is ready once it has a `turn_ended` episode, or once a later content
episode exists (a turn cut short by a crash); the newest turn is left alone
while it is still running.

Safe to run twice: entity resolution finds entities created by an earlier run,
reconciliation skips statements already stored from the same episodes, and
episodes are marked extracted only after everything else.

Job failures (`extraction_handler`): `Preempted` propagates untouched (the
runner defers the job). Any other failure of one turn makes the job fail and
be retried by the runner, and the turn and the consecutive failure count are
recorded in the job's error text (`last_error`, e.g. "memory.extract: turn
turn:abc failed 2 time(s): ..."), which the next attempt reads back: no schema
change and no payload rewrite is needed. When the same turn has failed
`max_turn_failures` times in a row, or on the job's last attempt, the turn is
skipped (its episodes are marked extracted and the failure logged) so later
turns proceed. Errors listed as transient (server unreachable, unknown role)
never skip a turn: they only fail the job, so an outage does not discard turns.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from typing import Any, Final, Literal

from pydantic import BaseModel

from lucy.core.events import AssistantMessage, ToolFinished, TurnEnded, UserMessage
from lucy.core.gateway import GatewayError, LLMGateway, Preempted, Priority
from lucy.core.jobs import Handler, Job, JobStore, utc_now
from lucy.core.types import Message, Role
from lucy.memory import fts
from lucy.memory.models import (
    DEFAULT_USER_NAME,
    USER_ENTITY_ID,
    Entity,
    EpisodeRecord,
    NewStatement,
    Origin,
    StatementKind,
)
from lucy.memory.policy import MemoryPolicy
from lucy.memory.ports import GraphStore
from lucy.memory.reconcile import Outcome, Reconciler, ReconcileResult
from lucy.memory.resolution import EntityResolver
from lucy.prompts import load_prompt

log = logging.getLogger(__name__)

#: The job kind, and the dedupe key of requested extraction jobs.
EXTRACT_JOB_KIND: Final = "memory.extract"
#: The dedupe key a running extraction job uses to enqueue its continuation:
#: a job cannot enqueue itself under its own key while it is running (the
#: queue returns the running job instead).
CONTINUE_DEDUPE_KEY: Final = "memory.extract:continue"

#: Earlier content episodes of the session shown as context.
CONTEXT_EPISODES: Final = 6
#: Longest statement text accepted, in characters.
MAX_TEXT_CHARS: Final = 300
#: Longest entity name accepted, in characters.
MAX_NAME_CHARS: Final = 80
#: Known entities listed in the prompt at most.
MAX_ENTITIES_SHOWN: Final = 20
#: Tool results and assistant replies are cut to this many characters in the prompt.
MAX_CONTEXT_CHARS: Final = 600
#: Dates outside [EARLIEST_YEAR, today's year + LATEST_YEARS_AHEAD] are implausible.
EARLIEST_YEAR: Final = 1900
LATEST_YEARS_AHEAD: Final = 30
#: Predicates of a statement giving the user's own name.
_NAME_PREDICATE: Final = re.compile(
    r"^(?:has )?(?:the )?(?:first |full )?name(?: is)?$|^is (?:named|called)$"
)
#: A goal worded as a plan or intention ("plans to cycle to work next week").
_PLAN: Final = re.compile(
    r"\b(?:plans?|planning|planned|(?:is|are|am) going to|intends?|intending)\b", re.I
)
#: A predicate describing a change event rather than a state.
_CHANGE_EVENT: Final = re.compile(
    r"^(?:has |have )?(?:moved|relocated|switched|swapped|changed|left|started|took up|joined)\b",
    re.I,
)
#: A text describing a change event ("The user has switched to decaf").
_CHANGE_TEXT: Final = re.compile(
    r"\b(?:has |have )?(?:moved|relocated|switched|swapped|changed) (?:to|from|over|back)\b", re.I
)
#: "The user" as written before the user's name was known.
_THE_USER: Final = re.compile(r"\b[Tt]he user\b")
#: Entity types that need no link to the user (a decision is about the user by nature).
_UNLINKED_OK: Final = frozenset({"decision"})
#: Statements whose text still says "The user" rewritten per turn at most.
MAX_REWRITES: Final = 200


def with_name(text: str, name: str) -> str:
    """`text` with "The user" / "the user" replaced by `name`."""
    return _THE_USER.sub(name, text)


# --- What the model returns ---------------------------------------------------


class EntityMention(BaseModel):
    """An entity as the model names it."""

    name: str
    type: str


class ExtractedStatement(BaseModel):
    """One candidate statement as the model produces it (values not yet checked)."""

    subject: EntityMention
    predicate: str
    object_entity: EntityMention | None = None
    object_value: str | None = None
    text: str
    kind: Literal["fact", "preference", "habit", "decision", "goal", "instruction"]
    origin: Literal["stated", "inferred"]
    valid_from: str | None = None
    valid_until: str | None = None


class Extraction(BaseModel):
    """The model's output for one turn."""

    user_name: str | None = None
    statements: list[ExtractedStatement]


# --- Results ------------------------------------------------------------------


@dataclass(frozen=True)
class Dropped:
    """A candidate rejected by the value checks, and why."""

    candidate: ExtractedStatement
    reason: str


@dataclass
class TurnReport:
    """What processing one turn did."""

    key: str
    episode_ids: list[int]
    source_ids: list[int]
    extraction: Extraction | None = None
    dropped: list[Dropped] = field(default_factory=list)
    results: list[ReconcileResult] = field(default_factory=list)
    reconciled: list[ExtractedStatement] = field(default_factory=list)
    """The candidates reconciled, in order: `results[i]` is for `reconciled[i]`."""
    renamed_user: str | None = None
    skipped: str | None = None
    """Set when the turn was marked extracted without extracting (and why)."""
    unlinked: list[str] = field(default_factory=list)
    """Names of entities created in this turn that no statement connects to the
    user or to anything already known (logged; see the module docstring)."""
    rewritten: int = 0
    """Stored texts reworded from "The user" to the user's name."""


@dataclass
class RunReport:
    """What one `Extractor.run` did. `more` is whether ready turns remain."""

    turns: list[TurnReport] = field(default_factory=list)
    more: bool = False


def turn_key(episodes: Sequence[EpisodeRecord]) -> str:
    """A stable identifier of a turn: its turn id, or its first episode's id if it has none."""
    turn_id = episodes[0].event.turn_id
    return f"turn:{turn_id}" if turn_id is not None else f"episode:{episodes[0].id}"


# --- Value checks -----------------------------------------------------------------

_DATE_FORMS: Final = (
    (re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$"), 3),
    (re.compile(r"^(\d{4})-(\d{1,2})$"), 2),
    (re.compile(r"^(\d{4})$"), 1),
)


def parse_date(value: str | None, *, today: date) -> date | None:
    """Parse a model-written date (YYYY-MM-DD, YYYY-MM or YYYY; the first day is assumed).

    Returns None for an empty value; raises `ValueError` for anything else that
    is not a real, plausible date.
    """
    if value is None or not value.strip() or value.strip().lower() in {"null", "none"}:
        return None
    text = value.strip()
    for pattern, parts in _DATE_FORMS:
        match = pattern.match(text)
        if match:
            numbers = [int(g) for g in match.groups()] + [1] * (3 - parts)
            result = date(*numbers)  # raises ValueError for month 59 etc.
            if not EARLIEST_YEAR <= result.year <= today.year + LATEST_YEARS_AHEAD:
                raise ValueError(f"implausible year in {text!r}")
            return result
    raise ValueError(f"not a date: {text!r}")


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def check_candidate(
    candidate: ExtractedStatement, *, today: date, max_text_chars: int = MAX_TEXT_CHARS
) -> str | None:
    """Why `candidate` must be dropped, or None if its values are plausible and it
    follows the rules in the module docstring."""
    text = _clean(candidate.text)
    if not text:
        return "empty text"
    if candidate.origin != "stated":
        return "extraction stores only what the user said (origin inferred)"
    if candidate.kind == "goal" and (
        _PLAN.search(text) or _PLAN.search(_clean(candidate.predicate))
    ):
        return "a plan is not a goal"
    if len(text) > max_text_chars:
        return f"text longer than {max_text_chars} characters"
    if not _clean(candidate.predicate):
        return "empty predicate"
    subject = _clean(candidate.subject.name)
    if not subject or len(subject) > MAX_NAME_CHARS:
        return "subject cannot be resolved"
    obj = candidate.object_entity
    has_entity = obj is not None and bool(_clean(obj.name))
    if has_entity and _clean(candidate.object_value):
        return "both an object entity and an object value"
    if has_entity and obj is not None and len(_clean(obj.name)) > MAX_NAME_CHARS:
        return "object entity name too long"
    try:
        start = parse_date(candidate.valid_from, today=today)
        end = parse_date(candidate.valid_until, today=today)
    except ValueError as exc:
        return f"implausible date ({exc})"
    if start is not None and end is not None and end < start:
        return "valid_until before valid_from"
    return None


def is_change_event(candidate: ExtractedStatement) -> bool:
    """Whether `candidate` describes a change ("moved to Porto") rather than a state."""
    return bool(
        _CHANGE_EVENT.match(_clean(candidate.predicate)) or _CHANGE_TEXT.search(candidate.text)
    )


def _object_key(candidate: ExtractedStatement) -> str:
    obj = candidate.object_entity
    name = obj.name if obj is not None else candidate.object_value
    return fts.normalise(_clean(name))


def _calls_by_name(candidate: ExtractedStatement, user: Entity) -> bool:
    """Whether `candidate` is an instruction to call the user by their own name
    ("Lucy should call Rohan Rohan" after "Hi, I'm Rohan"): nothing to store."""
    if candidate.kind != "instruction" or user.name == DEFAULT_USER_NAME:
        return False
    key = _object_key(candidate)
    return bool(key) and key in {fts.normalise(n) for n in user.names}


def order_candidates(candidates: Sequence[ExtractedStatement]) -> list[ExtractedStatement]:
    """The turn's candidates, change events last (stable otherwise)."""
    return sorted(candidates, key=is_change_event)


def repeats_new_state(
    candidate: ExtractedStatement, siblings: Sequence[ExtractedStatement]
) -> str | None:
    """A reason to drop a change event that another candidate of the turn already
    states as the new state (same subject and object), else None."""
    if not is_change_event(candidate) or not _object_key(candidate):
        return None
    subject = fts.normalise(_clean(candidate.subject.name))
    for other in siblings:
        if (
            other is not candidate
            and not is_change_event(other)
            and fts.normalise(_clean(other.subject.name)) == subject
            and _object_key(other) == _object_key(candidate)
        ):
            return f"a change event; the new state is stored as {_clean(other.text)!r}"
    return None


def _midnight(day: date | None) -> datetime | None:
    return None if day is None else datetime.combine(day, time(), tzinfo=UTC)


# --- Prompt rendering -------------------------------------------------------------


def _day(value: datetime | date) -> str:
    return f"{value.isoformat()[:10]} ({value.strftime('%A')})"


def _cut(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def render_episode(record: EpisodeRecord) -> str | None:
    """One episode as a transcript line, or None if it has no content to show."""
    event = record.event
    if isinstance(event, UserMessage):
        return f"User: {event.text.strip()}"
    if isinstance(event, AssistantMessage):
        if not event.text.strip():
            return None
        return f"Lucy: {_cut(event.text)}"
    if isinstance(event, ToolFinished):
        return f"Tool result: {_cut(event.result.content)}"
    return None


# --- The extractor ----------------------------------------------------------------


class Extractor:
    """Runs extraction and reconciliation over unextracted turns.

    `resolver` and `reconciler` default to ones built from the same graph,
    gateway, policy, role and priority.
    """

    def __init__(
        self,
        graph: GraphStore,
        gateway: LLMGateway,
        *,
        policy: MemoryPolicy | None = None,
        role: str = "extraction",
        priority: Priority = Priority.BACKGROUND,
        clock: Callable[[], datetime] = utc_now,
        context_episodes: int = CONTEXT_EPISODES,
        max_text_chars: int = MAX_TEXT_CHARS,
        resolver: EntityResolver | None = None,
        reconciler: Reconciler | None = None,
    ) -> None:
        self._graph = graph
        self._gateway = gateway
        self.policy = policy or MemoryPolicy()
        self.role = role
        self.priority = priority
        self._clock = clock
        self.context_episodes = context_episodes
        self.max_text_chars = max_text_chars
        self.resolver = resolver or EntityResolver(
            graph, gateway, policy=self.policy, role=role, priority=priority
        )
        self.reconciler = reconciler or Reconciler(
            graph, gateway, policy=self.policy, role=role, priority=priority, clock=clock
        )
        self._prompt = load_prompt("extraction")

    # --- Finding work ---------------------------------------------------------

    async def ready_turns(self, limit: int) -> list[list[EpisodeRecord]]:
        """Up to `limit` unextracted turns that are ready, oldest first.

        Each turn is the list of its content episodes (complete, even if some
        were fetched beyond the first batch), oldest first.
        """
        if limit <= 0:
            return []
        pending = await self._graph.unextracted_episodes(limit=max(50, limit * 20))
        groups: dict[str, list[EpisodeRecord]] = {}
        for record in pending:
            groups.setdefault(turn_key([record]), []).append(record)
        keys = list(groups)
        turns: list[list[EpisodeRecord]] = []
        for index, key in enumerate(keys):
            if len(turns) >= limit:
                break
            first = groups[key][0]
            turn_id = first.event.turn_id
            if turn_id is None:
                turns.append(groups[key])
                continue
            everything = await self._graph.turn_episodes(turn_id, types=None)
            ended = any(isinstance(r.event, TurnEnded) for r in everything)
            if not ended and index == len(keys) - 1:
                break  # the newest turn is still running
            content = [r for r in everything if r.extracted_at is None and is_content(r)]
            turns.append(content or groups[key])
        return turns

    # --- One turn -------------------------------------------------------------

    async def build_messages(self, episodes: Sequence[EpisodeRecord]) -> list[Message]:
        """The extraction prompt for the turn made of `episodes`."""
        user_messages = [r for r in episodes if isinstance(r.event, UserMessage)]
        turn_time = (user_messages or list(episodes))[0].event.ts
        user = await self._graph.user_entity()
        before = await self._graph.episodes_before(episodes[0].id, self.context_episodes)
        turn_text = "\n".join(
            r.event.text
            for r in episodes
            if isinstance(r.event, UserMessage | AssistantMessage) and r.event.text
        )
        known = [
            e for e in await self._graph.entities_mentioned_in(turn_text) if e.id != USER_ENTITY_ID
        ][:MAX_ENTITIES_SHOWN]

        lines = [
            f"Today: {_day(self._clock())}",
            f"Date of this turn: {_day(turn_time)}",
        ]
        if user.name == DEFAULT_USER_NAME:
            lines.append('The user\'s name: not known yet (write "The user" in texts)')
        else:
            also = f" (also called {', '.join(user.aliases)})" if user.aliases else ""
            lines.append(f"The user's name: {user.name}{also}")
        if known:
            lines += ["", "Known people and things mentioned:"]
            for entity in known:
                also = f"; also called {', '.join(entity.aliases)}" if entity.aliases else ""
                lines.append(f"- {entity.name} ({entity.type}{also})")
        rendered_before = [line for r in before if (line := render_episode(r))]
        if rendered_before:
            lines += ["", "Earlier conversation (context only):", *rendered_before]
        lines += ["", "This turn:", *[line for r in episodes if (line := render_episode(r))]]
        lines += ["", "Extract statements from what the user said in this turn."]
        return [
            Message(role=Role.SYSTEM, content=self._prompt),
            Message(role=Role.USER, content="\n".join(lines)),
        ]

    async def extract_turn(self, episodes: Sequence[EpisodeRecord]) -> TurnReport:
        """Extract, resolve, reconcile and mark one turn (its content episodes)."""
        episodes = sorted(episodes, key=lambda r: r.id)
        ids = [r.id for r in episodes]
        sources = [r.id for r in episodes if isinstance(r.event, UserMessage)]
        report = TurnReport(key=turn_key(episodes), episode_ids=ids, source_ids=sources)
        if not sources:
            report.skipped = "no user message"
            await self._graph.mark_extracted(ids)
            return report

        self.resolver.reset()
        messages = await self.build_messages(episodes)
        extraction = await self._gateway.generate(
            self.role, messages, Extraction, priority=self.priority
        )
        report.extraction = extraction
        observed = max(r.event.ts for r in episodes if r.id in sources)
        today = self._clock().date()

        name = extraction.user_name or await self._name_statement(extraction)
        if name:
            report.renamed_user = await self._rename_user(name)
        user = await self._graph.user_entity()
        user_name = None if user.name == DEFAULT_USER_NAME else user.name

        stored: list[NewStatement] = []
        for candidate in order_candidates(extraction.statements):
            reason = check_candidate(candidate, today=today, max_text_chars=self.max_text_chars)
            if reason is None:
                reason = repeats_new_state(candidate, extraction.statements)
            if reason is None and (
                await self._is_name_statement(candidate) or _calls_by_name(candidate, user)
            ):
                reason = "the user's own name (the user entity carries it)"
            if reason is None:
                new = await self._resolve(candidate, observed, today, user_name)
                if new is None:
                    reason = "subject cannot be resolved"
            if reason is not None:
                log.info("dropped candidate %r: %s", candidate.text, reason)
                report.dropped.append(Dropped(candidate, reason))
                continue
            assert new is not None
            result = await self.reconciler.reconcile(
                new, sources, pass_episodes=sources, change=is_change_event(candidate)
            )
            log.info(
                "turn %s: %s %r (%s)", report.key, result.outcome.value, new.text, result.reason
            )
            report.results.append(result)
            report.reconciled.append(candidate)
            if result.outcome is not Outcome.IGNORE:
                stored.append(new)

        report.unlinked = self._unlinked(stored, user)
        for entity_name in report.unlinked:
            log.warning(
                "turn %s: %r is new and nothing links it to the user or to anything known",
                report.key,
                entity_name,
            )
        if user_name:
            report.rewritten = await self.rewrite_user_texts(user_name)
        await self._graph.mark_extracted(ids)
        return report

    async def skip_turn(self, episodes: Sequence[EpisodeRecord], reason: str) -> TurnReport:
        """Mark a turn extracted without extracting it (a poison turn)."""
        ids = sorted(r.id for r in episodes)
        report = TurnReport(
            key=turn_key(sorted(episodes, key=lambda r: r.id)),
            episode_ids=ids,
            source_ids=[r.id for r in episodes if isinstance(r.event, UserMessage)],
            skipped=reason,
        )
        await self._graph.mark_extracted(ids)
        return report

    async def run(self, max_turns: int) -> RunReport:
        """Process up to `max_turns` ready turns, oldest first."""
        report = RunReport()
        for _ in range(max_turns):
            turns = await self.ready_turns(1)
            if not turns:
                return report
            report.turns.append(await self.extract_turn(turns[0]))
        report.more = bool(await self.ready_turns(1))
        return report

    # --- Helpers --------------------------------------------------------------

    async def _name_statement(self, extraction: Extraction) -> str | None:
        """The user's name from a stated "the user's name is X" statement, if any.

        Models sometimes state the name as a statement instead of `user_name`.
        """
        for candidate in extraction.statements:
            if not await self._is_name_statement(candidate):
                continue
            obj = candidate.object_entity
            value = _clean(obj.name if obj is not None else candidate.object_value)
            if value:
                return value
        return None

    async def _is_name_statement(self, candidate: ExtractedStatement) -> bool:
        """Whether `candidate` states the user's own name ("the user is named Rohan")."""
        if candidate.origin != "stated" or not _NAME_PREDICATE.match(
            _clean(candidate.predicate).lower()
        ):
            return False
        return await self.resolver.is_user(_clean(candidate.subject.name))

    def _unlinked(self, stored: Sequence[NewStatement], user: Entity) -> list[str]:
        """Names of entities created this turn that `stored` does not connect to the
        user or to an entity that existed before the turn (directly or through
        other new entities). A statement about a new entity whose text names the
        user ("Biscuit is Dana's tortoise") counts as a connection."""
        new = {e.id: e for e in self.resolver.created if e.type not in _UNLINKED_OK}
        if not new:
            return []
        names = [fts.normalise(n) for n in user.names if n != DEFAULT_USER_NAME]
        names.append("the user")
        linked: set[int] = set()
        edges = [(s.subject_id, s.object_id) for s in stored if s.object_id is not None]
        for s in stored:
            if s.subject_id in new and any(n in fts.normalise(s.text) for n in names):
                linked.add(s.subject_id)
        changed = True
        while changed:
            changed = False
            for a, b in edges:
                for x, y in ((a, b), (b, a)):
                    if x in new and x not in linked and (y not in new or y in linked):
                        linked.add(x)
                        changed = True
        return [e.name for e in new.values() if e.id not in linked]

    async def rewrite_user_texts(self, name: str, *, limit: int = MAX_REWRITES) -> int:
        """Reword up to `limit` stored statements that say "The user" to say `name`.

        Statements written before the user's name was known say "The user".
        Every statement is reworded, current or not: the substitution keeps the
        meaning, and the reconciler compares new statements with old ones too.
        Texts are re-embedded. Idempotent (a reworded text no longer matches);
        bounded per call, so a large backlog is finished over later turns.
        Returns how many statements were reworded.
        """
        graph = self._graph
        hits = await graph.keyword_statements("user", k=limit, include_invalid=True)
        texts = {
            h.statement.id: with_name(h.statement.text, name)
            for h in hits
            if _THE_USER.search(h.statement.text)
        }
        if not texts:
            return 0
        changed = await graph.set_statement_texts(texts)
        log.info('reworded %d statement(s) from "The user" to %r', changed, name)
        return changed

    async def _rename_user(self, name: str) -> str | None:
        name = _clean(name)
        key = fts.normalise(name)
        if not name or len(name) > MAX_NAME_CHARS or key in {"user", "the user", "lucy"}:
            log.info("ignored implausible user name %r", name)
            return None
        user = await self._graph.user_entity()
        if fts.normalise(user.name) == key:
            return None
        keep = user.name != DEFAULT_USER_NAME
        await self._graph.rename_entity(USER_ENTITY_ID, name, keep_old_as_alias=keep)
        log.info("the user is now called %r (was %r)", name, user.name)
        return name

    async def _resolve(
        self,
        candidate: ExtractedStatement,
        observed: datetime,
        today: date,
        user_name: str | None = None,
    ) -> NewStatement | None:
        text = _clean(candidate.text)
        if user_name:
            text = with_name(text, user_name)
        subject_name = _clean(candidate.subject.name)
        try:
            subject = await self.resolver.resolve(
                subject_name, candidate.subject.type, context=text
            )
        except ValueError:
            return None
        object_id = None
        obj = candidate.object_entity
        if obj is not None and _clean(obj.name):
            resolved = await self.resolver.resolve(_clean(obj.name), obj.type, context=text)
            object_id = resolved.entity.id
        origin = Origin(candidate.origin)
        return NewStatement(
            subject_id=subject.entity.id,
            predicate=_clean(candidate.predicate),
            object_id=object_id,
            object_value=_clean(candidate.object_value) or None,
            text=text,
            kind=StatementKind(candidate.kind),
            origin=origin,
            confidence=self.policy.initial_confidence(origin),
            valid_from=_midnight(parse_date(candidate.valid_from, today=today)),
            invalid_at=_midnight(parse_date(candidate.valid_until, today=today)),
            observed_at=observed,
        )


def is_content(record: EpisodeRecord) -> bool:
    """Whether `record` is a content episode (user, assistant or tool result)."""
    return isinstance(record.event, UserMessage | AssistantMessage | ToolFinished)


# --- The job ----------------------------------------------------------------------

_FAILURE: Final = re.compile(r"memory\.extract: turn (\S+) failed (\d+) time")

#: Errors that never cause a turn to be skipped: they say nothing about the turn.
DEFAULT_TRANSIENT_ERRORS: Final[tuple[type[BaseException], ...]] = (
    GatewayError,
    OSError,
    TimeoutError,
)


class TurnFailedError(RuntimeError):
    """Extraction of one turn failed; the message records the turn and the failure count."""

    def __init__(self, key: str, failures: int, error: BaseException) -> None:
        super().__init__(
            f"memory.extract: turn {key} failed {failures} time(s): {type(error).__name__}: {error}"
        )
        self.key = key
        self.failures = failures


def previous_failure(job: Job) -> tuple[str, int] | None:
    """The turn and consecutive failure count recorded by this job's last failed run."""
    match = _FAILURE.search(job.last_error or "")
    return None if match is None else (match.group(1), int(match.group(2)))


async def request_extraction(job_store: JobStore, *, max_attempts: int = 5) -> Job:
    """Enqueue an extraction job, unless one is already pending or running."""
    return await job_store.enqueue(
        EXTRACT_JOB_KIND, {}, dedupe_key=EXTRACT_JOB_KIND, max_attempts=max_attempts
    )


def extraction_handler(
    extractor: Extractor,
    job_store: JobStore,
    *,
    max_turns: int = 4,
    max_turn_failures: int = 3,
    transient_errors: tuple[type[BaseException], ...] = DEFAULT_TRANSIENT_ERRORS,
) -> Handler:
    """The handler for job kind `"memory.extract"`.

    Each run processes at most `max_turns` ready turns, oldest first; if ready
    turns remain it enqueues a follow-up job. See the module docstring for how
    failures are retried and when a turn is skipped. Pass the adapters' error
    types (e.g. the LLM and embedder adapters' errors for an unreachable
    server) in `transient_errors` so an outage never skips turns.
    """
    if max_turns < 1 or max_turn_failures < 1:
        raise ValueError("max_turns and max_turn_failures must be at least 1")

    async def handle(job: Job) -> None:
        previous = previous_failure(job)
        for _ in range(max_turns):
            turns = await extractor.ready_turns(1)
            if not turns:
                return
            turn = turns[0]
            key = turn_key(turn)
            try:
                await extractor.extract_turn(turn)
            except Preempted:
                raise
            except transient_errors:
                raise
            except Exception as exc:
                failures = 1 + (previous[1] if previous and previous[0] == key else 0)
                if failures < max_turn_failures and job.attempts < job.max_attempts:
                    raise TurnFailedError(key, failures, exc) from exc
                log.error(
                    "skipping turn %s after %d failure(s): %s: %s",
                    key,
                    failures,
                    type(exc).__name__,
                    exc,
                    exc_info=exc,
                )
                await extractor.skip_turn(turn, f"failed {failures} time(s): {exc}")
        if await extractor.ready_turns(1):
            await _continue(job_store, job)

    return handle


async def _continue(job_store: JobStore, job: Job) -> Any:
    """Enqueue a follow-up extraction job (see `CONTINUE_DEDUPE_KEY`)."""
    follow = await job_store.enqueue(
        EXTRACT_JOB_KIND, {}, dedupe_key=EXTRACT_JOB_KIND, max_attempts=job.max_attempts
    )
    if follow.id == job.id:
        follow = await job_store.enqueue(
            EXTRACT_JOB_KIND, {}, dedupe_key=CONTINUE_DEDUPE_KEY, max_attempts=job.max_attempts
        )
    return follow
