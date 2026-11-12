"""Reconciliation: merging a candidate statement into the graph (docs/memory.md "2. Reconcile").

For each candidate, existing statements about the same subject are found by
embedding similarity (invalidated ones included).

Statements of the same pass. The caller says which episodes the pass that
produced the candidate read (`pass_episodes`: for extraction, the turn's user
messages). Statements learned only from those episodes are its siblings (or
come from an interrupted earlier run of the same turn) and are not reconciled
against: the extraction model returned them as separate statements, and small
models were observed "superseding" one option of a decision with the choice
made in the same sentence. An older statement whose sources merely happen to
be among the candidate's evidence (reflection re-observing an inference with
all of its earlier messages plus new ones) is compared as usual. A sibling
that says the same thing as the candidate in other words (same kind,
predicate and object, similar text) makes the candidate a duplicate: that is
what a re-run after a crash produces when the model words its output
differently.

If no statement is at least `MemoryPolicy.statement_similarity_floor` similar,
the candidate is added without asking the model. Otherwise one structured call
(prompt `reconcile`) chooses add, reinforce(target), supersede(target) or
ignore, plus, for a supersession, other listed statements the candidate also
ends (`also_ends`: the user gave up an activity, so the habit of doing it ends
too). These guard rails are then enforced in code, whatever the model said:

- the target must be one of the statements shown, else the candidate is added;
- the target must have the candidate's subject, else the candidate is added;
- reinforce and supersede need a current target: a superseded target is
  replaced by the current end of its replacement chain (so restating an old
  belief the user has since changed supersedes the newer one), or the
  candidate is added if the chain has no current end;
- a candidate that restates a statement that was corrected by a stated one is
  ignored unless the candidate is itself stated;
- an inferred candidate never supersedes (or ends) a stated statement;
- a stated candidate that reinforces an inferred statement supersedes it
  instead, so the belief becomes stated;
- a goal or habit is closed by a fact: a "decision" candidate that
  supersedes or ends one ("gave up X") is stored as a fact, as reflection's
  goal review records a dropped goal;
- a candidate the model wants to add, but which is near-identical to a
  current statement, reinforces it (and a near-identical corrected statement
  is treated as above);
- `also_ends` only ends current habits and goals of the same subject that
  were shown, other than the target, and only with a supersession.

Dates: a superseded (or ended) statement ends when the replacement starts:
the date the user gave for the change (`valid_from`), or else the time of the
message it was learned from (`observed_at`), never the time reconciliation
runs. A replacement without a date of its own therefore starts at that
message. The clock is used only when the candidate has neither.

Nothing is deleted: a superseded statement keeps its text and gets an end
date and a pointer to its replacement.

Safe to run twice: a candidate near-identical to a statement that already has
all of the candidate's source episodes (and the same object) was stored by an
earlier run and is skipped; reinforcing with sources already linked is skipped
too, so evidence is never counted twice.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from lucy.core.gateway import LLMGateway, Priority
from lucy.core.jobs import utc_now
from lucy.core.types import Message, Role
from lucy.memory import fts
from lucy.memory.graph_store import GraphStoreError
from lucy.memory.models import NewStatement, Origin, ScoredStatement, Statement, StatementKind
from lucy.memory.policy import MemoryPolicy
from lucy.memory.ports import GraphStore
from lucy.prompts import load_prompt

log = logging.getLogger(__name__)

#: Similar statements fetched per candidate.
SIMILAR_K: Final = 10
#: Similar statements shown to the model at most.
SHOWN_MAX: Final = 6
#: Longest replacement chain followed to find the current version of a statement.
_MAX_CHAIN: Final = 32
#: Kinds `also_ends` may end: what the user does or works towards.
_ENDABLE: Final = frozenset({StatementKind.HABIT, StatementKind.GOAL})
#: Statements one candidate may end besides its target, at most.
MAX_ALSO_ENDS: Final = 3
#: A predicate saying the user stopped something ("gave up", "stopped", "quit").
_ENDING: Final = re.compile(r"\b(?:gave up|give up|given up|stopped|quit|dropped|no longer)\b")


@runtime_checkable
class EndsStatements(Protocol):
    """A store that can end a statement because of another (`SQLiteGraphStore.end_statement`).

    Not (yet) part of `GraphStore`; without it, `also_ends` is logged and skipped.
    """

    async def end_statement(
        self, statement_id: int, at: datetime, *, superseded_by: int | None = None
    ) -> Statement: ...


class Outcome(StrEnum):
    """What reconciliation did with a candidate."""

    ADD = "add"
    REINFORCE = "reinforce"
    SUPERSEDE = "supersede"
    IGNORE = "ignore"
    DUPLICATE = "duplicate"
    """Already stored from the same source episodes (a re-run); nothing changed."""


class ReconcileDecision(BaseModel):
    """The model's choice for a candidate. `target` and `also_ends` are numbers of listed
    statements (mapped to statement ids, -1 for a number not shown, after the call)."""

    action: Literal["add", "reinforce", "supersede", "ignore"]
    target: int | None = None
    also_ends: list[int] = Field(default_factory=list)


@dataclass(frozen=True)
class ReconcileResult:
    """The outcome for one candidate.

    `statement` is the statement added, reinforced or added as the replacement
    (None for ignore); `target_id` is the existing statement reinforced,
    superseded or duplicated; `asked_model` says whether the model was asked;
    `reason` explains guard-rail decisions, for logs and tests.
    """

    outcome: Outcome
    statement: Statement | None = None
    target_id: int | None = None
    asked_model: bool = False
    reason: str = ""
    ended_ids: list[int] = field(default_factory=list)
    """Statements ended besides the target (`also_ends`), by id."""


def _same_object(a: NewStatement, b: Statement) -> bool:
    value_a = fts.normalise(a.object_value or "")
    value_b = fts.normalise(b.object_value or "")
    return a.object_id == b.object_id and value_a == value_b


def _same_predicate(a: NewStatement, b: Statement) -> bool:
    return fts.normalise(a.predicate) == fts.normalise(b.predicate)


def _describe(statement: Statement) -> str:
    return f"{statement.kind.value}, {statement.origin.value}"


class Reconciler:
    """Decides and applies add / reinforce / supersede / ignore for candidates."""

    def __init__(
        self,
        graph: GraphStore,
        gateway: LLMGateway,
        *,
        policy: MemoryPolicy | None = None,
        role: str = "extraction",
        priority: Priority = Priority.BACKGROUND,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._graph = graph
        self._gateway = gateway
        self.policy = policy or MemoryPolicy()
        self.role = role
        self.priority = priority
        self._clock = clock
        self._prompt = load_prompt("reconcile")

    async def reconcile(
        self,
        new: NewStatement,
        episode_ids: Sequence[int],
        *,
        pass_episodes: Collection[int] = (),
    ) -> ReconcileResult:
        """Merge the candidate `new`, learned from `episode_ids`, into the graph.

        `pass_episodes` are the episodes the pass that produced `new` read (for
        extraction, the turn's user messages; empty for reflection):
        statements learned only from them are siblings from the same pass and
        are not reconciled against (module docstring).
        """
        episodes = sorted(set(episode_ids))
        same_pass = set(pass_episodes)
        hits = await self._graph.similar_statements(
            new.text, subject_id=new.subject_id, k=SIMILAR_K, include_invalid=True
        )
        policy = self.policy

        for hit in hits:
            s = hit.statement
            if (
                episodes
                and set(episodes) <= set(s.sources)
                and _same_object(new, s)
                and (
                    hit.score >= policy.near_identical_similarity
                    or fts.normalise(s.text) == fts.normalise(new.text)
                )
            ):
                return ReconcileResult(
                    Outcome.DUPLICATE, s, s.id, reason="already stored from these episodes"
                )

        def sibling(s: Statement) -> bool:
            return bool(same_pass) and bool(s.sources) and set(s.sources) <= same_pass

        for hit in hits:
            s = hit.statement
            if (
                sibling(s)
                and hit.score >= policy.statement_similarity_floor
                and s.kind is new.kind
                and _same_predicate(new, s)
                and _same_object(new, s)
                and (new.object_id is not None or bool(new.object_value))
            ):
                # The same kind, predicate and object from the same turn, in
                # other words: a re-run (or the model repeating itself).
                return ReconcileResult(
                    Outcome.DUPLICATE,
                    s,
                    s.id,
                    reason="same statement from this turn in other words",
                )

        # Statements learned only from this pass's episodes (siblings from this
        # turn, or from an interrupted earlier run of it) are not reconciled
        # against: the user does not contradict or repeat themselves within one
        # turn, and the model returned them as separate statements.
        shown = [
            h
            for h in hits
            if h.score >= policy.statement_similarity_floor and not sibling(h.statement)
        ][:SHOWN_MAX]
        if not shown:
            result = await self._add(new, episodes, asked=False, reason="nothing similar")
        else:
            decision = await self._ask(new, shown)
            result = await self._apply(new, episodes, same_pass, shown, decision)
        return await self._end_related(new, result, same_pass, [h.statement.id for h in shown])

    # --- The model's decision -------------------------------------------------

    async def _ask(self, new: NewStatement, shown: list[ScoredStatement]) -> ReconcileDecision:
        now = self._clock()
        number = {h.statement.id: i for i, h in enumerate(shown, start=1)}
        lines = [
            "Known statements:",
        ]
        for i, hit in enumerate(shown, start=1):
            s = hit.statement
            if s.is_current(now):
                status = "current"
            else:
                assert s.invalid_at is not None
                status = f"no longer true since {s.invalid_at.date().isoformat()}"
                if s.superseded_by in number:
                    status += f", replaced by [{number[s.superseded_by]}]"
                elif s.superseded_by is not None:
                    status += ", replaced"
            lines.append(f"[{i}] {s.text} ({_describe(s)}; {status})")
        lines += [
            "",
            f"New statement: {new.text} ({new.kind.value}, {new.origin.value})",
        ]
        messages = [
            Message(role=Role.SYSTEM, content=self._prompt),
            Message(role=Role.USER, content="\n".join(lines)),
        ]
        decision = await self._gateway.generate(
            self.role, messages, ReconcileDecision, priority=self.priority
        )

        def to_id(number: int) -> int:
            # Map a shown number back to the statement id (-1: not shown).
            return shown[number - 1].statement.id if 1 <= number <= len(shown) else -1

        return decision.model_copy(
            update={
                "target": None if decision.target is None else to_id(decision.target),
                "also_ends": [to_id(n) for n in dict.fromkeys(decision.also_ends)],
            }
        )

    # --- Guard rails ------------------------------------------------------------

    async def _head(self, statement: Statement) -> Statement | None:
        """The current end of `statement`'s replacement chain, or None."""
        now = self._clock()
        current = statement
        for _ in range(_MAX_CHAIN):
            if current.is_current(now):
                return current
            if current.superseded_by is None:
                return None
            nxt = await self._graph.get_statement(current.superseded_by)
            if nxt is None:
                return None
            current = nxt
        return None

    async def _corrected_by_stated(self, statement: Statement) -> bool:
        """Whether `statement` was superseded by a stated statement."""
        if statement.superseded_by is None:
            return False
        successor = await self._graph.get_statement(statement.superseded_by)
        return successor is not None and successor.origin is Origin.STATED

    async def _apply(
        self,
        new: NewStatement,
        episodes: list[int],
        same_pass: set[int],
        shown: list[ScoredStatement],
        decision: ReconcileDecision,
    ) -> ReconcileResult:
        now = self._clock()
        by_id = {h.statement.id: h.statement for h in shown}
        action = decision.action
        stated = new.origin is Origin.STATED
        notes: list[str] = [f"model said {action}"]

        if action == "add":
            near = [
                h.statement
                for h in shown
                if h.score >= self.policy.near_identical_similarity
                and _same_object(new, h.statement)
            ]
            if near:
                action, target = "reinforce", near[0]
                notes.append(f"near-identical to {target.id}: reinforce")
            else:
                return await self._add(new, episodes, asked=True, reason="; ".join(notes))
        elif action == "ignore":
            return ReconcileResult(Outcome.IGNORE, asked_model=True, reason="model said ignore")
        else:
            if decision.target not in by_id:
                notes.append("target not among the shown statements: add")
                return await self._add(new, episodes, asked=True, reason="; ".join(notes))
            target = by_id[decision.target]

        if target.subject_id != new.subject_id:
            notes.append(f"{target.id} is about another subject: add")
            return await self._add(new, episodes, asked=True, reason="; ".join(notes))

        if not target.is_current(now):
            if not stated and await self._corrected_by_stated(target):
                notes.append(f"restates {target.id}, which the user corrected")
                return ReconcileResult(
                    Outcome.IGNORE, target_id=target.id, asked_model=True, reason="; ".join(notes)
                )
            head = await self._head(target)
            if head is None:
                notes.append(f"{target.id} is not current and has no current successor: add")
                return await self._add(new, episodes, asked=True, reason="; ".join(notes))
            if same_pass and set(head.sources) <= same_pass:
                notes.append(f"{target.id} was already replaced from this turn by {head.id}: add")
                return await self._add(new, episodes, asked=True, reason="; ".join(notes))
            if action == "reinforce":
                if not stated:
                    notes.append(f"restates {target.id}, which has since changed")
                    return ReconcileResult(
                        Outcome.IGNORE,
                        target_id=target.id,
                        asked_model=True,
                        reason="; ".join(notes),
                    )
                # The user restates a belief that has changed since: it is true again.
                action = "supersede"
            notes.append(f"{target.id} is not current: its current successor is {head.id}")
            target = head

        if action == "supersede" and not stated and target.origin is Origin.STATED:
            notes.append("an inferred candidate never supersedes a stated statement")
            return ReconcileResult(
                Outcome.IGNORE, target_id=target.id, asked_model=True, reason="; ".join(notes)
            )
        if action == "reinforce" and stated and target.origin is Origin.INFERRED:
            notes.append("stated candidate replaces the inferred statement")
            action = "supersede"

        if action == "reinforce":
            return await self._reinforce(new, episodes, target, "; ".join(notes))
        also = self._also_ends(new, target, decision.also_ends, by_id, notes)
        if new.kind is StatementKind.DECISION and any(s.kind in _ENDABLE for s in (target, *also)):
            notes.append("a goal or habit is closed by a fact")
            new = new.model_copy(update={"kind": StatementKind.FACT})
        result = await self._supersede(new, episodes, target, "; ".join(notes))
        if also and result.statement is not None:
            result = await self._end_also(result, also)
        return result

    async def _end_related(
        self,
        new: NewStatement,
        result: ReconcileResult,
        same_pass: set[int],
        shown: Collection[int],
    ) -> ReconcileResult:
        """The second hop of `also_ends`, for a candidate that ended something.

        Runs when a fact (or decision) superseded a habit or goal, or when the
        candidate's predicate says something ended ("gave up", "stopped",
        "quit"), whether it was added or superseded something. Not when a habit,
        goal or preference replaced another ("swims on Thursdays" for "swims on
        Tuesdays"): that changes one thing. Current habits and goals of the same subject
        that were not shown are collected: those similar to the replaced
        statement, and those sharing a word with the candidate's object
        (keyword search). Embedding similarity alone misses them: "wants to be
        fluent in Italian by summer 2027" scored 0.834 against "gave up
        learning Italian" and 0.850 against "wants to get a pilot's licence by
        2028". The model is asked once more, about these only, with the same
        prompt; what it would supersede or lists in `also_ends` is ended
        (after the same guard rails). Nothing else is looked at: statements
        that are neither habits nor goals, or that share no word and are not
        similar, stay as they are.
        """
        replacement = result.statement
        if replacement is None or result.outcome not in (Outcome.ADD, Outcome.SUPERSEDE):
            return result
        target = None
        if result.outcome is Outcome.SUPERSEDE and result.target_id is not None:
            target = await self._graph.get_statement(result.target_id)
        ending = bool(_ENDING.search(fts.normalise(new.predicate)))
        closes = (
            target is not None
            and target.kind in _ENDABLE
            and new.kind in (StatementKind.FACT, StatementKind.DECISION)
        )
        if not (ending or closes):
            return result
        skip = {replacement.id, *shown, *result.ended_ids}
        if target is not None:
            skip.add(target.id)
        found: dict[int, ScoredStatement] = {}
        if target is not None:
            for hit in await self._graph.similar_statements(
                target.text, subject_id=new.subject_id, k=SIMILAR_K, kinds=sorted(_ENDABLE)
            ):
                if hit.score >= self.policy.statement_similarity_floor:
                    found.setdefault(hit.statement.id, hit)
        words = new.object_value
        if words is None and new.object_id is not None:
            entity = await self._graph.get_entity(new.object_id, resolve=True)
            words = entity.name if entity is not None else None
        if words:
            for hit in await self._graph.keyword_statements(
                words, k=SIMILAR_K * 2, kinds=sorted(_ENDABLE)
            ):
                if hit.statement.subject_id == new.subject_id:
                    found.setdefault(hit.statement.id, hit)
        now = self._clock()
        related = [
            h
            for h in found.values()
            if h.statement.id not in skip
            and h.statement.is_current(now)
            and not (same_pass and set(h.statement.sources) <= same_pass)
        ][:SHOWN_MAX]
        if not related:
            return result
        decision = await self._ask(new, related)
        if decision.action != "supersede":
            log.info(
                "nothing related ends with %d (model said %s)", replacement.id, decision.action
            )
            return result
        numbers = [n for n in (decision.target, *decision.also_ends) if n is not None]
        notes: list[str] = []
        chosen = self._also_ends(
            new,
            target or replacement,
            numbers,
            {h.statement.id: h.statement for h in related},
            notes,
        )
        log.info("related to %d: %s", replacement.id, "; ".join(notes) or "nothing ends")
        if not chosen:
            return result
        return await self._end_also(result, chosen)

    def _also_ends(
        self,
        new: NewStatement,
        target: Statement,
        numbers: Sequence[int],
        by_id: dict[int, Statement],
        notes: list[str],
    ) -> list[Statement]:
        """The statements to end besides `target`, after the guard rails (module docstring)."""
        now = self._clock()
        chosen: list[Statement] = []
        for sid in numbers:
            other = by_id.get(sid)
            if other is None or other.id == target.id or other in chosen:
                continue
            problem = None
            if other.subject_id != new.subject_id:
                problem = "another subject"
            elif other.kind not in _ENDABLE:
                problem = f"a {other.kind.value}"
            elif not other.is_current(now):
                problem = "not current"
            elif new.origin is not Origin.STATED and other.origin is Origin.STATED:
                problem = "stated, and the candidate is inferred"
            if problem:
                notes.append(f"not ending {other.id}: {problem}")
                continue
            chosen.append(other)
        if len(chosen) > MAX_ALSO_ENDS:
            notes.append(f"ending only {MAX_ALSO_ENDS} of {len(chosen)} other statements")
            chosen = chosen[:MAX_ALSO_ENDS]
        if chosen:
            notes.append("also ends " + ", ".join(str(s.id) for s in chosen))
        return chosen

    async def _end_also(self, result: ReconcileResult, also: list[Statement]) -> ReconcileResult:
        replacement = result.statement
        assert replacement is not None
        graph = self._graph
        if not isinstance(graph, EndsStatements):
            log.warning(
                "the graph store cannot end statements; not ending %s", [s.id for s in also]
            )
            return result
        closed = None
        if result.outcome is Outcome.SUPERSEDE and result.target_id is not None:
            closed = await self._graph.get_statement(result.target_id)
        at = (
            (closed.invalid_at if closed else None)
            or replacement.valid_from
            or replacement.last_observed_at
        )
        ended: list[int] = list(result.ended_ids)
        for other in also:
            try:
                await graph.end_statement(other.id, at, superseded_by=replacement.id)
            except GraphStoreError as exc:
                log.warning("could not end statement %d: %s", other.id, exc)
                continue
            ended.append(other.id)
        return ReconcileResult(
            result.outcome,
            result.statement,
            result.target_id,
            asked_model=result.asked_model,
            reason=result.reason,
            ended_ids=ended,
        )

    # --- Writes -----------------------------------------------------------------

    async def _add(
        self, new: NewStatement, episodes: list[int], *, asked: bool, reason: str
    ) -> ReconcileResult:
        statement = await self._graph.add_statement(new, episodes)
        return ReconcileResult(Outcome.ADD, statement, asked_model=asked, reason=reason)

    async def _reinforce(
        self, new: NewStatement, episodes: list[int], target: Statement, reason: str
    ) -> ReconcileResult:
        if episodes and set(episodes) <= set(target.sources):
            return ReconcileResult(
                Outcome.DUPLICATE,
                target,
                target.id,
                asked_model=True,
                reason=reason + "; already reinforced from these episodes",
            )
        confidence = self.policy.reinforced(target.confidence, target.origin)
        statement = await self._graph.reinforce(
            target.id, episodes, confidence=confidence, observed_at=new.observed_at
        )
        return ReconcileResult(
            Outcome.REINFORCE, statement, target.id, asked_model=True, reason=reason
        )

    async def _supersede(
        self, new: NewStatement, episodes: list[int], target: Statement, reason: str
    ) -> ReconcileResult:
        if new.valid_from is None and new.observed_at is not None:
            # The old statement ends when the user said it changed, not now.
            new = new.model_copy(update={"valid_from": new.observed_at})
        try:
            statement = await self._graph.supersede(target.id, new, episodes)
        except GraphStoreError as exc:
            if new.valid_from is None or target.superseded_by is not None:
                raise
            # The new statement claims to start before the old one did; the
            # dates are unreliable. Start it when it was said (or, if even
            # that is before the old one began, when the old one began), or
            # leave the start unknown if that is not known either.
            start = new.observed_at
            if start is not None and target.valid_from is not None:
                start = max(start, target.valid_from)
            log.warning("superseding %d with dates %s; starting at %s", target.id, exc, start)
            new = new.model_copy(update={"valid_from": start})
            statement = await self._graph.supersede(target.id, new, episodes)
            reason += "; valid_from moved (it was before the old statement began)"
        return ReconcileResult(
            Outcome.SUPERSEDE, statement, target.id, asked_model=True, reason=reason
        )
