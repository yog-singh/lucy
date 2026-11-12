"""Reflection: learning from many conversations at once (docs/memory.md "3. Reflect").

Extraction (per turn) stores what the user says. Reflection is a periodic
background job (kind `"memory.reflect"`) that looks across many of the user's
messages for what no single turn shows, in two independent parts.

Pattern inference (`Reflector.infer_patterns`)
----------------------------------------------
Input: the user's own messages (`user_message` episodes) from the last
`policy.lookback`, each numbered 1..n and shown with its weekday, date and
local time (`policy.timezone`), grouped by weekday (all Mondays, then all
Tuesdays...; oldest first within a weekday) so that weekly habits sit
together, plus the user's current habit and preference
statements: stated ones as context ("do not report again"), and inferred
ones as numbered guesses K1..Kn, each with a required evidence field in the
answer schema, so every earlier guess is checked against the messages
(measured: asked to re-report known guesses in a free list, gemma4:12b
skipped a continuing habit, which then decayed like a stopped one). A guess
with cited messages is checked and submitted like a new inference with the
guess's own wording, so it reinforces the guess. Numbers are positions in
the prompt, never database ids.

Budget: the messages (each cut to `policy.max_message_chars`) must fit in
`policy.pattern_input_tokens` (estimated tokens). When they do not, an evenly
spaced sample is shown: the largest k such that messages at k evenly spaced
positions of the chronological list fit. A pattern needs repeated evidence
across the window, so thinning everywhere is better than keeping only the
newest days; the prompt says the messages are a sample.

Run condition: nothing happens (no model call) unless at least
`policy.min_new_messages` user messages have been logged since the last
successful run (by episode id, under `PATTERNS_CURSOR_KEY`, a `derived.` key so
a graph rebuild resets it).

One structured call returns inferences (habit or preference, a self-contained
third-person text, predicate, optional object value, and the numbers of the
supporting messages). Evidence rules are then enforced in code, whatever the
model says, and each drop is logged with its reason:

- every cited number must be one that was shown (otherwise the inference is dropped);
- a habit needs `policy.habit_min_messages` supporting messages on
  `policy.habit_min_days` different local calendar days; a preference needs
  `policy.preference_min_messages`;
- a pattern the user stated outright is not an inference: an inference is
  dropped when a current *stated* habit or preference of the user is
  near-identical to it, or is similar (at least the reconciler's
  `statement_similarity_floor`), was learned from cited messages, and the
  other cited messages would not meet the evidence rule above on their own
  (the user said it; the rest is not a pattern of its own). With enough
  independent evidence it goes on to the reconciler, where stated wins;
- value checks as in extraction: non-empty text of bounded length, written in
  the third person, a non-empty predicate, a short object value; repeats of an
  earlier inference in the same answer are dropped;
- at most `policy.max_inferences` are written per run.

Each surviving inference becomes a `NewStatement` about the user, origin
`inferred`, at the reconciler policy's initial inferred confidence, observed
at the newest cited message, with the cited episodes as sources, and goes
through `Reconciler.reconcile`, whose guard rails apply: an inference never
supersedes a stated statement, matching an existing inferred statement
reinforces it (raising confidence one step per run that brings new evidence;
decay lowers it when the evidence stops), and a stated statement wins.

An inference is submitted with all the messages it cites, including ones
an earlier version of it already had: the reconciler compares it with that
earlier version (it leaves out only statements of the same extraction pass,
and reflection has none), and a reinforcement links only the new ones. An
inference whose cited messages all support a similar current habit or
preference already is skipped without asking the reconciler (no new
evidence).

Goals and decisions (`Reflector.review_progress`)
-------------------------------------------------
Items: the user's current goal statements, and decision entities (type
"decision") with no current statement whose predicate mentions "outcome".
Each item has a cursor (`derived.reflect.progress.<goal|decision>.<id>`): the
last episode id reviewed for it, starting at the newest episode it was learned
from. Items with newer user messages (within `policy.lookback`) are
candidates; the `policy.max_progress_items` most recently observed are
reviewed in one structured call, the rest wait for the next run. The model
sees those items, the statements recorded since the last review (context),
and the user's messages since the oldest selected cursor, numbered 1..n (if
over `policy.progress_input_tokens`, messages sharing a word with an item are
kept first, then the newest).

- goal: open, achieved or dropped. Achieved or dropped needs at least one
  cited message, all of them shown; then a fact recording the outcome (origin
  `stated` only if the model says the user said it outright) goes through
  the reconciler with the cited episodes as sources, and the goal is ended
  as of the newest cited message (kept, with an end date and a pointer to the
  outcome). The outcome is written first: if the run stops in between, the
  next run finds the goal still open and the outcome already stored (a
  duplicate). A goal extraction already closed ("I've given up on Italian"
  superseding it with a fact) is no longer current and so is never reviewed:
  the two never both record an outcome. If reflection closes a goal first,
  extraction later finds the outcome through the goal's pointer, or as a
  similar statement, and reinforces or replaces it instead of adding another.
- decision: when the user reported how it turned out (with cited messages),
  an "outcome" statement is added to the decision entity.

Nothing is invalidated that is not a current goal. No citation, no change.
Cursors of the reviewed items move to the newest message considered.

Both parts are safe to run twice, bounded per run, and let `Preempted`
propagate untouched; what was written before an interruption stays written.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any, Final, Literal

from pydantic import BaseModel, Field, create_model

from lucy.core.context import estimate_tokens
from lucy.core.events import UserMessage
from lucy.core.gateway import LLMGateway, Preempted, Priority
from lucy.core.jobs import Handler, Job, utc_now
from lucy.core.types import Message, Role
from lucy.memory import fts
from lucy.memory.models import (
    DEFAULT_USER_NAME,
    USER_ENTITY_ID,
    Entity,
    EpisodeRecord,
    NewStatement,
    Origin,
    Statement,
    StatementKind,
)
from lucy.memory.ports import DERIVED_META_PREFIX, GraphStore
from lucy.memory.reconcile import Outcome, Reconciler, ReconcileResult
from lucy.prompts import load_prompt

log = logging.getLogger(__name__)

#: The job kind (and its dedupe key when scheduled).
REFLECT_JOB: Final = "memory.reflect"

#: Id of the newest user-message episode seen by the last successful pattern run.
PATTERNS_CURSOR_KEY: Final = f"{DERIVED_META_PREFIX}reflect.patterns.last_episode"
#: When pattern inference last completed (informational).
PATTERNS_LAST_RUN_KEY: Final = f"{DERIVED_META_PREFIX}reflect.patterns.last_run"
#: Per-item review cursors: `PROGRESS_PREFIX + "goal." / "decision." + zero-padded id`.
PROGRESS_PREFIX: Final = f"{DERIVED_META_PREFIX}reflect.progress."
#: When the progress review last completed (statements recorded since are shown).
PROGRESS_LAST_RUN_KEY: Final = f"{DERIVED_META_PREFIX}reflect.last_progress_run"

#: Longest predicate and object value accepted, in characters.
MAX_PREDICATE_CHARS: Final = 80
MAX_VALUE_CHARS: Final = 120
#: First- and second-person words: a statement text must be in the third person.
_FIRST_PERSON_I: Final = re.compile(r"(?<![\w'])I(?:'m|'ve|'d|'ll)?(?![\w'])")
_PERSONAL: Final = re.compile(r"\b(?:me|my|mine|myself|you|your|yours)\b", re.I)


def progress_key(kind: Literal["goal", "decision"], item_id: int) -> str:
    """The `memory_meta` key of a goal's or decision's review cursor."""
    return f"{PROGRESS_PREFIX}{kind}.{item_id:012d}"


# --- Policy ---------------------------------------------------------------------


@dataclass(frozen=True)
class ReflectionPolicy:
    """Windows, budgets and evidence thresholds of reflection."""

    lookback: timedelta = timedelta(days=60)
    """How far back the user's messages are read."""
    min_new_messages: int = 10
    """Pattern inference runs only after this many new user messages."""
    pattern_input_tokens: int = 6000
    """Estimated tokens of messages shown for pattern inference (sampled beyond)."""
    max_message_chars: int = 300
    """Each message is cut to this many characters in the prompt."""
    max_known_patterns: int = 30
    """Current habits and preferences shown, strongest first."""
    max_inferences: int = 8
    """Inferences written per run at most."""
    habit_min_messages: int = 3
    habit_min_days: int = 2
    preference_min_messages: int = 2
    max_text_chars: int = 300
    progress_input_tokens: int = 4000
    """Estimated tokens of messages shown for the goal and decision review."""
    max_progress_items: int = 8
    """Goals and decisions reviewed per run at most."""
    max_recent_statements: int = 15
    """Statements recorded since the last review shown as context."""
    timezone: tzinfo = UTC
    """The user's time zone, for weekdays, dates and times shown to the model."""

    def __post_init__(self) -> None:
        if self.lookback <= timedelta(0):
            raise ValueError("lookback must be positive")
        for name in (
            "min_new_messages",
            "pattern_input_tokens",
            "max_message_chars",
            "habit_min_messages",
            "habit_min_days",
            "preference_min_messages",
            "max_text_chars",
            "progress_input_tokens",
            "max_progress_items",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        for name in ("max_known_patterns", "max_inferences", "max_recent_statements"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")


# --- What the model returns -------------------------------------------------------


class InferredPattern(BaseModel):
    """One habit or preference as the model reports it (not yet checked)."""

    evidence: list[int] = Field(description="Numbers of the messages that show the pattern.")
    kind: Literal["habit", "preference"]
    text: str
    predicate: str
    object_value: str | None = None


class KnownCheck(BaseModel):
    """The model's answer for one known inferred pattern ("K1"...)."""

    evidence: list[int] = Field(description="Numbers of the messages that show it; [] if none.")


class PatternAnswer(BaseModel):
    """The model's answer: evidence per known inferred pattern (by its number), new patterns."""

    known: dict[int, list[int]] = Field(default_factory=dict)
    patterns: list[InferredPattern]


def pattern_schema(known: int) -> type[BaseModel]:
    """The answer schema: a required "K<i>" evidence field per known inferred pattern
    (as for the progress review, so none is skipped), then a list of new patterns."""
    if not known:
        return create_model("PatternAnswer", patterns=(list[InferredPattern], ...))
    checks = create_model(
        "KnownPatterns", **{f"K{i}": (KnownCheck, ...) for i in range(1, known + 1)}
    )
    return create_model("PatternAnswer", known=(checks, ...), patterns=(list[InferredPattern], ...))


class GoalStatus(BaseModel):
    """The model's answer for one goal."""

    evidence: list[int] = Field(description="Numbers of the messages that show the status.")
    status: Literal["open", "achieved", "dropped"]
    said_outright: bool = False
    text: str | None = None


class DecisionOutcome(BaseModel):
    """The model's answer for one decision."""

    evidence: list[int] = Field(description="Numbers of the messages reporting the outcome.")
    outcome: str | None = None
    said_outright: bool = False
    text: str | None = None


def progress_schema(goals: int, decisions: int) -> type[BaseModel]:
    """The answer schema for one review: a required field per item ("G1".., "D1"..).

    One required field per item makes schema-constrained decoding answer every
    item; with a list, gemma4:12b answered only the first goal of three.
    """
    fields: dict[str, Any] = {f"G{i}": (GoalStatus, ...) for i in range(1, goals + 1)}
    fields |= {f"D{i}": (DecisionOutcome, ...) for i in range(1, decisions + 1)}
    return create_model("ProgressAnswer", **fields)


class GoalReview(GoalStatus):
    goal: int
    """The goal's number in the prompt (G1 is 1)."""


class DecisionReview(DecisionOutcome):
    decision: int
    """The decision's number in the prompt (D1 is 1)."""


class ProgressAnswer(BaseModel):
    """The model's answers, by item."""

    goals: list[GoalReview]
    decisions: list[DecisionReview]

    @classmethod
    def from_answer(cls, answer: BaseModel, goals: int, decisions: int) -> ProgressAnswer:
        data = answer.model_dump()
        return cls(
            goals=[GoalReview(goal=i, **data[f"G{i}"]) for i in range(1, goals + 1)],
            decisions=[
                DecisionReview(decision=i, **data[f"D{i}"]) for i in range(1, decisions + 1)
            ],
        )


# --- Results ----------------------------------------------------------------------


@dataclass(frozen=True)
class DroppedInference:
    """An inference rejected by the evidence or value checks, and why."""

    pattern: InferredPattern
    reason: str


@dataclass
class PatternResult:
    """What one `infer_patterns` run did."""

    skipped: str | None = None
    """Set when the model was not asked (and why)."""
    new_messages: int = 0
    """User messages logged since the last successful run."""
    window_messages: int = 0
    shown_messages: int = 0
    sampled: bool = False
    answer: PatternAnswer | None = None
    dropped: list[DroppedInference] = field(default_factory=list)
    results: list[tuple[InferredPattern, ReconcileResult]] = field(default_factory=list)


@dataclass(frozen=True)
class ProgressChange:
    """A goal closed or a decision outcome recorded."""

    item: Literal["goal", "decision"]
    item_id: int
    """The goal statement's id, or the decision entity's id."""
    status: Literal["achieved", "dropped", "outcome"]
    statement: Statement | None
    """The outcome statement written (None if the reconciler ignored it)."""
    evidence: list[int]
    """Episode ids cited."""
    at: datetime


@dataclass
class ProgressResult:
    """What one `review_progress` run did."""

    skipped: str | None = None
    reviewed_goals: list[int] = field(default_factory=list)
    reviewed_decisions: list[int] = field(default_factory=list)
    waiting: int = 0
    """Items with new messages left for the next run."""
    shown_messages: int = 0
    answer: ProgressAnswer | None = None
    changes: list[ProgressChange] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    """Model answers that changed nothing, with the reason."""


# --- Rendering and checks ---------------------------------------------------------


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def render_message(record: EpisodeRecord, tz: tzinfo, max_chars: int) -> str:
    """A user message as shown to the model, without its number."""
    event = record.event
    assert isinstance(event, UserMessage)
    local = event.ts.astimezone(tz)
    when = local.strftime("%a %Y-%m-%d %H:%M")
    return f"{when}: {_cut(_clean(event.text), max_chars)}"


def _evenly_spaced(n: int, k: int) -> list[int]:
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n - 1]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def sample_evenly(costs: Sequence[int], budget: int) -> list[int]:
    """Indices of items to keep so their total cost fits `budget`, spread evenly.

    All of them if they fit; otherwise the largest k for which k evenly spaced
    positions (first and last included) fit. Deterministic.
    """
    n = len(costs)
    if sum(costs) <= budget:
        return list(range(n))

    def cost(k: int) -> int:
        return sum(costs[i] for i in _evenly_spaced(n, k))

    k = max(0, min(n, n * budget // max(sum(costs), 1)))
    while k > 0 and cost(k) > budget:
        k -= 1
    while k < n and cost(k + 1) <= budget:
        k += 1
    return _evenly_spaced(n, k)


def _number(lines: Sequence[str]) -> list[str]:
    return [f"[{i}] {line}" for i, line in enumerate(lines, start=1)]


def text_problem(text: str, max_chars: int) -> str | None:
    """Why a statement text written by the model is unusable, or None."""
    if not text:
        return "empty text"
    if len(text) > max_chars:
        return f"text longer than {max_chars} characters"
    if _FIRST_PERSON_I.search(text) or _PERSONAL.search(text):
        return "text is not in the third person"
    return None


def evidence_problem(
    kind: str, records: Sequence[EpisodeRecord], policy: ReflectionPolicy
) -> str | None:
    """Why `records` (distinct messages) are not enough evidence for a `kind` pattern."""
    if kind == "habit":
        if len(records) < policy.habit_min_messages:
            return (
                f"a habit needs {policy.habit_min_messages} supporting messages,"
                f" {len(records)} cited"
            )
        days = {r.event.ts.astimezone(policy.timezone).date() for r in records}
        if len(days) < policy.habit_min_days:
            return (
                f"a habit needs messages on {policy.habit_min_days} different days, got {len(days)}"
            )
    elif len(records) < policy.preference_min_messages:
        return (
            f"a preference needs {policy.preference_min_messages} supporting messages,"
            f" {len(records)} cited"
        )
    return None


def check_pattern(
    pattern: InferredPattern,
    shown: Sequence[EpisodeRecord],
    policy: ReflectionPolicy,
) -> str | None:
    """Why `pattern` breaks an evidence or value rule, or None.

    `shown` are the messages as numbered in the prompt (number i is shown[i-1]).
    """
    numbers = list(dict.fromkeys(pattern.evidence))
    if not numbers:
        return "no supporting messages cited"
    unknown = [n for n in numbers if not 1 <= n <= len(shown)]
    if unknown:
        return f"cites message(s) {unknown} that were not shown"
    problem = evidence_problem(pattern.kind, [shown[n - 1] for n in numbers], policy)
    if problem:
        return problem
    problem = text_problem(_clean(pattern.text), policy.max_text_chars)
    if problem:
        return problem
    predicate = _clean(pattern.predicate)
    if not predicate:
        return "empty predicate"
    if len(predicate) > MAX_PREDICATE_CHARS:
        return "predicate too long"
    if len(_clean(pattern.object_value)) > MAX_VALUE_CHARS:
        return "object value too long"
    return None


def _cited(numbers: Iterable[int], shown: Sequence[EpisodeRecord]) -> list[EpisodeRecord] | None:
    """The cited messages, or None if the citation is empty or cites an unshown number."""
    numbers = list(dict.fromkeys(numbers))
    if not numbers or any(not 1 <= n <= len(shown) for n in numbers):
        return None
    return [shown[n - 1] for n in numbers]


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", fts.normalise(text)) if len(w) > 3}


def _is_outcome(statement: Statement) -> bool:
    return "outcome" in fts.normalise(statement.predicate)


# --- The reflector ----------------------------------------------------------------


@dataclass
class _Item:
    kind: Literal["goal", "decision"]
    id: int
    cursor: int
    relevant_at: datetime
    statement: Statement | None = None
    entity: Entity | None = None
    statements: list[Statement] = field(default_factory=list)

    @property
    def parts(self) -> list[Statement]:
        """The statements this item consists of."""
        return [self.statement] if self.statement is not None else self.statements


class Reflector:
    """Infers habits and preferences, and reviews goals and decisions (module docstring)."""

    def __init__(
        self,
        graph: GraphStore,
        gateway: LLMGateway,
        reconciler: Reconciler,
        *,
        role: str = "extraction",
        policy: ReflectionPolicy | None = None,
        clock: Callable[[], datetime] = utc_now,
        priority: Priority = Priority.BACKGROUND,
    ) -> None:
        self._graph = graph
        self._gateway = gateway
        self.reconciler = reconciler
        self.role = role
        self.policy = policy or ReflectionPolicy()
        self.priority = priority
        self._clock = clock
        self._patterns_prompt = load_prompt("reflection_patterns")
        self._progress_prompt = load_prompt("reflection_progress")

    async def _user_name(self) -> str | None:
        user = await self._graph.user_entity()
        return None if user.name == DEFAULT_USER_NAME else user.name

    def _today(self, now: datetime) -> str:
        return now.astimezone(self.policy.timezone).strftime("%a %Y-%m-%d")

    # --- Pattern inference ------------------------------------------------------

    async def build_pattern_messages(
        self, window: Sequence[EpisodeRecord], known: Sequence[Statement], now: datetime
    ) -> tuple[list[Message], list[EpisodeRecord], bool]:
        """The prompt, the messages shown (in numbering order), and whether they were sampled."""
        policy = self.policy
        lines = [render_message(r, policy.timezone, policy.max_message_chars) for r in window]
        # The number prefix "[123] " costs about two tokens.
        keep = sample_evenly(
            [estimate_tokens(line) + 2 for line in lines], policy.pattern_input_tokens
        )
        sampled = len(keep) < len(window)
        # Grouped by weekday (Monday first), oldest first within a day: weekly
        # habits then sit together (measured: gemma4:12b found a Sunday-evening
        # habit this way that it missed in plain time order).
        tz = policy.timezone
        keep.sort(key=lambda i: (window[i].event.ts.astimezone(tz).weekday(), window[i].event.ts))
        shown = [window[i] for i in keep]
        name = await self._user_name()
        who = name or "The user"
        start = (now - policy.lookback).astimezone(policy.timezone).strftime("%Y-%m-%d")
        content = [f"Today: {self._today(now)}"]
        if name:
            content.append(f"The user's name: {name}")
        else:
            content.append('The user\'s name: not known (write "The user" in texts)')
        stated = [s for s in known if s.origin is Origin.STATED]
        guesses = [s for s in known if s.origin is not Origin.STATED]
        content += ["", "Habits and preferences the user told Lucy:"]
        content += [f"- {s.text.strip()} ({s.kind.value})" for s in stated] or ["- none"]
        content += ["", "Lucy's earlier guesses, to check against the messages:"]
        content += [
            f"[K{i}] {s.text.strip()} ({s.kind.value})" for i, s in enumerate(guesses, start=1)
        ] or ["- none"]
        header = (
            f"{who}'s messages since {start}, grouped by weekday and oldest first within"
            f" each weekday ({len(window)} messages"
        )
        if sampled:
            header += f"; {len(shown)} of them are shown, spread evenly over the period"
        content += ["", header + "):", *_number([lines[i] for i in keep])]
        content += ["", "List the habits and preferences these messages show, if any."]
        messages = [
            Message(role=Role.SYSTEM, content=self._patterns_prompt),
            Message(role=Role.USER, content="\n".join(content)),
        ]
        return messages, shown, sampled

    async def _known_patterns(self) -> list[Statement]:
        known = await self._graph.list_statements(
            subject_id=USER_ENTITY_ID, kinds=[StatementKind.HABIT, StatementKind.PREFERENCE]
        )
        known.sort(key=lambda s: (s.origin is not Origin.STATED, -s.confidence, s.id))
        return known[: self.policy.max_known_patterns]

    async def infer_patterns(self) -> PatternResult:
        """Infer habits and preferences from the user's recent messages (module docstring)."""
        policy = self.policy
        result = PatternResult()
        raw = await self._graph.get_meta(PATTERNS_CURSOR_KEY)
        last = int(raw) if raw and raw.isdigit() else 0
        result.new_messages = await self._graph.count_episodes_after(last)
        if result.new_messages < policy.min_new_messages:
            result.skipped = (
                f"{result.new_messages} new user message(s), fewer than {policy.min_new_messages}"
            )
            return result
        now = self._clock()
        window = await self._graph.episodes_between(now - policy.lookback, now)
        window = [r for r in window if isinstance(r.event, UserMessage)]
        result.window_messages = len(window)
        if not window:
            result.skipped = "no user messages in the look-back window"
            return result
        known = await self._known_patterns()
        messages, shown, sampled = await self.build_pattern_messages(window, known, now)
        result.shown_messages, result.sampled = len(shown), sampled
        guesses = [s for s in known if s.origin is not Origin.STATED]
        raw_answer = await self._gateway.generate(
            self.role, messages, pattern_schema(len(guesses)), priority=self.priority
        )
        data = raw_answer.model_dump()
        answer = PatternAnswer(
            known={i: data["known"][f"K{i}"]["evidence"] for i in range(1, len(guesses) + 1)},
            patterns=[InferredPattern(**p) for p in data["patterns"]],
        )
        result.answer = answer
        # Evidence for a known guess is checked and submitted like a new
        # inference with the guess's own wording, so it reinforces the guess.
        candidates = [
            InferredPattern(
                evidence=answer.known[i],
                kind="habit" if s.kind is StatementKind.HABIT else "preference",
                text=s.text,
                predicate=s.predicate,
                object_value=s.object_value,
            )
            for i, s in enumerate(guesses, start=1)
            if answer.known[i]
        ]

        seen_texts: set[str] = set()
        seen_evidence: set[tuple[str, frozenset[int]]] = set()
        written = 0
        for pattern in [*candidates, *answer.patterns]:
            reason = check_pattern(pattern, shown, policy)
            evidence = frozenset(pattern.evidence)
            text_key = fts.normalise(_clean(pattern.text))
            if reason is None and (
                text_key in seen_texts or (pattern.kind, evidence) in seen_evidence
            ):
                reason = "repeats an earlier inference in the same answer"
            if reason is None and written >= policy.max_inferences:
                reason = f"over the limit of {policy.max_inferences} inferences per run"
            cited = _cited(pattern.evidence, shown) if reason is None else None
            episodes = sorted(r.id for r in cited or [])
            if reason is None:
                assert cited is not None
                reason = await self._stated_outright(pattern, cited)
            if reason is None:
                reason = await self._no_new_evidence(pattern, episodes)
            if reason is not None:
                log.info("reflection dropped %s %r: %s", pattern.kind, pattern.text, reason)
                result.dropped.append(DroppedInference(pattern, reason))
                continue
            assert cited is not None
            seen_texts.add(text_key)
            seen_evidence.add((pattern.kind, evidence))
            written += 1
            new = self._pattern_statement(pattern, cited)
            outcome = await self.reconciler.reconcile(new, episodes)
            log.info(
                "reflection: %s inferred %s %r (%s)",
                outcome.outcome.value,
                pattern.kind,
                new.text,
                outcome.reason,
            )
            result.results.append((pattern, outcome))

        await self._graph.write_meta(
            {
                PATTERNS_CURSOR_KEY: str(max(last, *(r.id for r in window))),
                PATTERNS_LAST_RUN_KEY: now.isoformat(),
            }
        )
        return result

    def _pattern_statement(
        self, pattern: InferredPattern, cited: Sequence[EpisodeRecord]
    ) -> NewStatement:
        return NewStatement(
            subject_id=USER_ENTITY_ID,
            predicate=_clean(pattern.predicate),
            object_value=_clean(pattern.object_value) or None,
            text=_clean(pattern.text),
            kind=StatementKind(pattern.kind),
            origin=Origin.INFERRED,
            confidence=self.reconciler.policy.initial_confidence(Origin.INFERRED),
            observed_at=max(r.event.ts for r in cited),
        )

    async def _similar_patterns(self, text: str) -> list[tuple[Statement, float]]:
        hits = await self._graph.similar_statements(
            _clean(text),
            subject_id=USER_ENTITY_ID,
            k=10,
            kinds=[StatementKind.HABIT, StatementKind.PREFERENCE],
        )
        floor = self.reconciler.policy.statement_similarity_floor
        return [(h.statement, h.score) for h in hits if h.score >= floor]

    async def _stated_outright(
        self, pattern: InferredPattern, cited: Sequence[EpisodeRecord]
    ) -> str | None:
        """A reason to drop `pattern` if the user already stated it, else None.

        Dropped when a similar stated habit or preference is near-identical, or
        was learned from cited messages and the other cited messages would not
        be enough evidence on their own (see the module docstring).
        """
        near = self.reconciler.policy.near_identical_similarity
        for statement, score in await self._similar_patterns(pattern.text):
            if statement.origin is not Origin.STATED:
                continue
            rest = [r for r in cited if r.id not in set(statement.sources)]
            if score >= near or (
                len(rest) < len(cited) and evidence_problem(pattern.kind, rest, self.policy)
            ):
                return f"the user stated it outright (statement {statement.id})"
        return None

    async def _no_new_evidence(self, pattern: InferredPattern, episodes: list[int]) -> str | None:
        """A reason to skip `pattern` if a similar habit or preference already has all
        of its cited messages (nothing for the reconciler to add), else None."""
        for statement, _ in await self._similar_patterns(pattern.text):
            if statement.sources and set(episodes) <= set(statement.sources):
                return f"no new evidence for statement {statement.id}"
        return None

    # --- Goals and decisions ----------------------------------------------------

    async def _items(self) -> list[_Item]:
        goals = await self._graph.list_statements(
            subject_id=USER_ENTITY_ID, kinds=[StatementKind.GOAL]
        )
        decisions: list[tuple[Entity, list[Statement]]] = []
        for entity in await self._graph.list_entities(type="decision"):
            statements = await self._graph.current_statements(entity.id)
            if statements and not any(_is_outcome(s) for s in statements):
                decisions.append((entity, statements))
        keys = [progress_key("goal", g.id) for g in goals]
        keys += [progress_key("decision", e.id) for e, _ in decisions]
        cursors = await self._graph.get_meta_many(keys)

        def cursor(key: str, sources: Iterable[int]) -> int:
            raw = cursors.get(key)
            if raw is not None and raw.isdigit():
                return int(raw)
            return max(sources, default=0)

        items = [
            _Item(
                "goal",
                g.id,
                cursor(progress_key("goal", g.id), g.sources),
                g.last_observed_at,
                statement=g,
            )
            for g in goals
        ]
        for entity, statements in decisions:
            sources = [i for s in statements for i in s.sources]
            items.append(
                _Item(
                    "decision",
                    entity.id,
                    cursor(progress_key("decision", entity.id), sources),
                    max(s.last_observed_at for s in statements),
                    entity=entity,
                    statements=statements,
                )
            )
        return items

    async def _sweep_cursors(self, items: Sequence[_Item]) -> None:
        live = {progress_key(i.kind, i.id) for i in items}
        stale = {k: None for k, _ in await self._graph.list_meta(PROGRESS_PREFIX) if k not in live}
        if stale:
            await self._graph.write_meta(stale)

    def _select_messages(
        self, messages: Sequence[EpisodeRecord], items: Sequence[_Item]
    ) -> list[EpisodeRecord]:
        """Messages to show, in time order, within the progress budget."""
        policy = self.policy
        lines = [render_message(r, policy.timezone, policy.max_message_chars) for r in messages]
        costs = [estimate_tokens(line) + 2 for line in lines]
        if sum(costs) <= policy.progress_input_tokens:
            return list(messages)
        words: set[str] = set()
        for item in items:
            texts = [item.statement.text] if item.statement else [s.text for s in item.statements]
            if item.entity is not None:
                texts.append(item.entity.name)
            for text in texts:
                words |= _words(text)
        texts = [r.event.text if isinstance(r.event, UserMessage) else "" for r in messages]
        order = sorted(range(len(messages)), key=lambda i: (not (_words(texts[i]) & words), -i))
        keep: list[int] = []
        used = 0
        for i in order:
            if used + costs[i] <= policy.progress_input_tokens:
                keep.append(i)
                used += costs[i]
        return [messages[i] for i in sorted(keep)]

    async def build_progress_messages(
        self,
        goals: Sequence[_Item],
        decisions: Sequence[_Item],
        recent: Sequence[Statement],
        shown: Sequence[EpisodeRecord],
        since: datetime,
        now: datetime,
    ) -> list[Message]:
        """The prompt for reviewing `goals` and `decisions` against `shown` messages."""
        tz = self.policy.timezone
        name = await self._user_name()
        who = name or "The user"
        content = [f"Today: {self._today(now)}"]
        if name:
            content.append(f"The user's name: {name}")
        else:
            content.append('The user\'s name: not known (write "The user" in texts)')
        sources = {i for item in [*goals, *decisions] for s in item.parts for i in s.sources}
        said = {r.id: r.event.ts for r in await self._graph.get_episodes(sources)}

        def first_said(statements: Sequence[Statement]) -> str:
            times = [said[i] for s in statements for i in s.sources if i in said]
            when = min(times, default=min(s.recorded_at for s in statements))
            return when.astimezone(tz).strftime("%Y-%m-%d")

        content += ["", "Goals (still open as far as Lucy knows):"]
        for i, item in enumerate(goals, start=1):
            assert item.statement is not None
            day = first_said([item.statement])
            content.append(f"[G{i}] {item.statement.text.strip()} (noted {day})")
        if not goals:
            content.append("- none")
        content += ["", "Decisions whose outcome Lucy does not know yet:"]
        for i, item in enumerate(decisions, start=1):
            assert item.entity is not None
            day = first_said(item.statements)
            details = " ".join(s.text.strip() for s in item.statements)
            content.append(f"[D{i}] {item.entity.name} (decided {day}): {details}")
        if not decisions:
            content.append("- none")
        if recent:
            content += ["", "Recently noted (for context only):"]
            content += [f"- {s.text.strip()}" for s in recent]
        start = since.astimezone(tz).strftime("%Y-%m-%d")
        lines = [render_message(r, tz, self.policy.max_message_chars) for r in shown]
        content += ["", f"{who}'s messages since {start}, oldest first:", *_number(lines)]
        content += ["", "For each goal and decision, say what these messages show."]
        return [
            Message(role=Role.SYSTEM, content=self._progress_prompt),
            Message(role=Role.USER, content="\n".join(content)),
        ]

    async def review_progress(self) -> ProgressResult:
        """Check open goals and decisions against the user's messages (module docstring)."""
        policy = self.policy
        result = ProgressResult()
        now = self._clock()
        items = await self._items()
        await self._sweep_cursors(items)
        if not items:
            result.skipped = "no open goals or decisions"
            return result
        floor = min(i.cursor for i in items)
        window = await self._graph.episodes_between(now - policy.lookback, now, after_id=floor)
        window = [r for r in window if isinstance(r.event, UserMessage)]
        candidates = [i for i in items if any(r.id > i.cursor for r in window)]
        if not candidates:
            result.skipped = "no new user messages about open goals or decisions"
            return result
        candidates.sort(key=lambda i: (i.relevant_at, i.id), reverse=True)
        selected = candidates[: policy.max_progress_items]
        result.waiting = len(candidates) - len(selected)
        since_id = min(i.cursor for i in selected)
        considered = [r for r in window if r.id > since_id]
        shown = self._select_messages(considered, selected)
        result.shown_messages = len(shown)
        goals = [i for i in selected if i.kind == "goal"]
        decisions = [i for i in selected if i.kind == "decision"]
        result.reviewed_goals = [i.id for i in goals]
        result.reviewed_decisions = [i.id for i in decisions]

        raw = await self._graph.get_meta(PROGRESS_LAST_RUN_KEY)
        try:
            since = datetime.fromisoformat(raw) if raw else now - policy.lookback
        except ValueError:
            since = now - policy.lookback
        # Context only: other open goals are left out (they wait for their turn).
        goal_ids = {i.id for i in items if i.kind == "goal"}
        recent = [
            s
            for s in await self._graph.statements_recorded_since(
                since, subject_id=USER_ENTITY_ID, limit=policy.max_recent_statements * 2
            )
            if s.id not in goal_ids and s.kind is not StatementKind.INSTRUCTION
        ][: policy.max_recent_statements]
        first_shown = shown[0].event.ts if shown else now
        messages = await self.build_progress_messages(
            goals, decisions, recent, shown, first_shown, now
        )
        schema = progress_schema(len(goals), len(decisions))
        raw_answer = await self._gateway.generate(
            self.role, messages, schema, priority=self.priority
        )
        answer = ProgressAnswer.from_answer(raw_answer, len(goals), len(decisions))
        result.answer = answer

        answered: set[int] = set()
        for review in answer.goals:
            note = await self._apply_goal(review, goals, shown, answered, result)
            if note:
                result.dropped.append(note)
                log.info("reflection: goal review changed nothing: %s", note)
        answered = set()
        for decision in answer.decisions:
            note = await self._apply_decision(decision, decisions, shown, answered, result)
            if note:
                result.dropped.append(note)
                log.info("reflection: decision review changed nothing: %s", note)
        closed = {progress_key(change.item, change.item_id) for change in result.changes}
        newest = max((r.id for r in considered), default=since_id)
        cursors: dict[str, str | None] = {}
        for item in selected:
            key = progress_key(item.kind, item.id)
            cursors[key] = None if key in closed else str(max(item.cursor, newest))
        cursors[PROGRESS_LAST_RUN_KEY] = now.isoformat()
        await self._graph.write_meta(cursors)
        return result

    async def _apply_goal(
        self,
        review: GoalReview,
        goals: Sequence[_Item],
        shown: Sequence[EpisodeRecord],
        done: set[int],
        result: ProgressResult,
    ) -> str | None:
        if not 1 <= review.goal <= len(goals):
            return f"unknown goal G{review.goal}"
        if review.goal in done:
            return f"G{review.goal} answered twice; the first answer counts"
        done.add(review.goal)
        if review.status == "open":
            return None
        cited = _cited(review.evidence, shown)
        if cited is None:
            return f"G{review.goal} {review.status} without valid cited messages"
        item = goals[review.goal - 1]
        goal = await self._graph.get_statement(item.id)
        now = self._clock()
        if goal is None or goal.kind is not StatementKind.GOAL or not goal.is_current(now):
            return f"G{review.goal} is no longer a current goal"
        at = max(r.event.ts for r in cited)
        if goal.valid_from is not None and at < goal.valid_from:
            at = goal.valid_from
        name = await self._user_name() or "The user"
        text = _clean(review.text)
        if text_problem(text, self.policy.max_text_chars):
            verb = "achieved" if review.status == "achieved" else "gave up"
            text = f"{name} {verb} the goal: {goal.text.strip().rstrip('.')}."
        origin = Origin.STATED if review.said_outright else Origin.INFERRED
        new = NewStatement(
            subject_id=USER_ENTITY_ID,
            predicate="achieved goal" if review.status == "achieved" else "dropped goal",
            object_id=goal.object_id,
            object_value=None if goal.object_id is not None else goal.object_value,
            text=text,
            kind=StatementKind.FACT,
            origin=origin,
            confidence=self.reconciler.policy.initial_confidence(origin),
            valid_from=at,
            observed_at=at,
        )
        episodes = sorted({r.id for r in cited})
        outcome = await self.reconciler.reconcile(new, episodes)
        log.info(
            "reflection: goal %d %s; outcome %s %r (%s)",
            goal.id,
            review.status,
            outcome.outcome.value,
            text,
            outcome.reason,
        )
        statement = outcome.statement if outcome.outcome is not Outcome.IGNORE else None
        if statement is not None and statement.id == goal.id:
            # The reconciler took the outcome for the goal itself: record it separately.
            statement = await self._graph.add_statement(new, episodes)
        # The reconciler may already have closed the goal (superseded by the outcome).
        current = await self._graph.get_statement(goal.id)
        if current is not None and current.is_current(self._clock()):
            pointer = (
                statement.id if statement is not None and current.superseded_by is None else None
            )
            await self._graph.end_statement(goal.id, at, superseded_by=pointer)
        result.changes.append(
            ProgressChange("goal", goal.id, review.status, statement, episodes, at)
        )
        return None

    async def _apply_decision(
        self,
        review: DecisionReview,
        decisions: Sequence[_Item],
        shown: Sequence[EpisodeRecord],
        done: set[int],
        result: ProgressResult,
    ) -> str | None:
        if not 1 <= review.decision <= len(decisions):
            return f"unknown decision D{review.decision}"
        if review.decision in done:
            return f"D{review.decision} answered twice; the first answer counts"
        done.add(review.decision)
        outcome_value = _cut(_clean(review.outcome), MAX_VALUE_CHARS)
        text = _clean(review.text)
        if not outcome_value and not text:
            return None
        cited = _cited(review.evidence, shown)
        if cited is None:
            return f"D{review.decision} outcome without valid cited messages"
        item = decisions[review.decision - 1]
        assert item.entity is not None
        current = await self._graph.current_statements(item.entity.id)
        if any(_is_outcome(s) for s in current):
            return f"D{review.decision} already has an outcome"
        problem = text_problem(text, self.policy.max_text_chars)
        if problem:
            if not outcome_value:
                return f"D{review.decision} outcome unusable: {problem}"
            text = f"Outcome of the {item.entity.name.strip()}: {outcome_value}."
        at = max(r.event.ts for r in cited)
        origin = Origin.STATED if review.said_outright else Origin.INFERRED
        new = NewStatement(
            subject_id=item.entity.id,
            predicate="outcome",
            object_value=outcome_value or None,
            text=text,
            kind=StatementKind.DECISION,
            origin=origin,
            confidence=self.reconciler.policy.initial_confidence(origin),
            valid_from=at,
            observed_at=at,
        )
        episodes = sorted({r.id for r in cited})
        statement = await self._graph.add_statement(new, episodes)
        log.info("reflection: decision %d outcome %r", item.entity.id, text)
        result.changes.append(
            ProgressChange("decision", item.entity.id, "outcome", statement, episodes, at)
        )
        return None


# --- The job ----------------------------------------------------------------------


class ReflectionError(RuntimeError):
    """One or both parts of a reflection run failed; the other part still ran."""


def reflection_handler(reflector: Reflector) -> Handler:
    """The handler for job kind `"memory.reflect"`.

    Runs pattern inference, then the goal and decision review. Each part runs
    whatever happens to the other: a failure in one is logged and, after the
    other has run, re-raised (as `ReflectionError`) so the runner retries the
    job; both parts are safe to repeat and whatever a part wrote stays
    written. `Preempted` propagates untouched at once (the runner defers the
    job, and the rerun repeats only unfinished work).
    """

    async def handle(job: Job) -> None:
        failures: list[str] = []
        for name, part in (
            ("patterns", reflector.infer_patterns),
            ("progress", reflector.review_progress),
        ):
            try:
                await part()
            except Preempted:
                raise
            except Exception as exc:
                log.error(
                    "reflection %s failed: %s: %s", name, type(exc).__name__, exc, exc_info=exc
                )
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        if failures:
            raise ReflectionError("memory.reflect: " + "; ".join(failures))

    return handle
