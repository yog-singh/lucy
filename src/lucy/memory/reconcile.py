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

Change events. Extraction marks a candidate that describes a change rather
than a state ("The user switched to decaf"; `change=True`). Such a text is
worded unlike the state it replaces ("The user drinks dark roast coffee":
0.81 with `embeddinggemma-2`, under the similarity floor), so it would be
added beside the old state, which stays current. For a change event:

- current statements of the subject at least
  `MemoryPolicy.change_similarity_floor` similar are shown too (same subject,
  siblings left out, at most `SHOWN_MAX` in all);
- the model is always asked, exactly once, with the candidate marked "(a
  change)" and a schema (`ChangeDecision`) that first asks for the new state
  in the form of the statement it replaces ("The user drinks decaf coffee"),
  then for the action. The new state replaces the candidate's text and
  predicate when it is usable (`_new_state`); a reworded change that
  supersedes a fact, preference or habit takes that statement's kind;
- the decision then goes through the same guard rails as any other. A
  reworded change that repeats a sibling is a duplicate. A change that was
  not reworded and whose target was already replaced from this turn is
  ignored (a sibling stored the new state: "moved from Lisbon" beside "lives
  in Porto"); a reworded one is added ("takes notes in Obsidian" beside "gave
  up Evernote").

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
from typing import Final, Literal

from pydantic import BaseModel, Field

from lucy.core.gateway import LLMGateway, Priority
from lucy.core.jobs import utc_now
from lucy.core.types import Message, Role
from lucy.memory import fts
from lucy.memory.models import NewStatement, Origin, ScoredStatement, Statement, StatementKind
from lucy.memory.policy import MemoryPolicy
from lucy.memory.ports import GraphStore, GraphStoreError
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
#: Words of a change event that a statement of the new state must not use.
_CHANGE_WORDS: Final = re.compile(
    r"\b(?:switch(?:ed|es)?|mov(?:ed|es)|relocated|chang(?:ed|es)|swapp?(?:ed|s)?|"
    r"start(?:ed|s)|took up|joined)\b",
    re.I,
)
#: A negated state ("does not work at Brightline"): only says what ended.
_NEGATED: Final = re.compile(
    r"\b(?:does not|doesn't|do not|don't|is not|isn't|no longer|not any ?more)\b", re.I
)
#: Longest new-state text accepted from the model, in characters.
_MAX_NEW_TEXT: Final = 300
#: A word starting with a capital letter (a name, or the first word of a sentence).
_CAPITALISED: Final = re.compile(r"\b[A-Z][\w'&-]*")
#: A word.
_WORD: Final = re.compile(r"[\w'&-]+")
#: Kinds a reworded change takes over from the statement it replaces.
_SAME_FORM: Final = frozenset({StatementKind.FACT, StatementKind.PREFERENCE, StatementKind.HABIT})


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


class ChangeDecision(BaseModel):
    """The model's choice for a change event, after the new state it led to.

    `new_text` / `new_predicate` say what is true now, in the form of the
    statement the change replaces; written first so the decision is about them.
    """

    new_text: str | None = None
    new_predicate: str | None = None
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
        change: bool = False,
    ) -> ReconcileResult:
        """Merge the candidate `new`, learned from `episode_ids`, into the graph.

        `pass_episodes` are the episodes the pass that produced `new` read (for
        extraction, the turn's user messages; empty for reflection):
        statements learned only from them are siblings from the same pass and
        are not reconciled against (module docstring).

        `change`: `new` describes a change ("switched to decaf") rather than a
        state; see "Change events" in the module docstring.
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
        now = self._clock()
        shown = [
            h
            for h in hits
            if not sibling(h.statement)
            and (
                h.score >= policy.statement_similarity_floor
                or (
                    change
                    and h.score >= policy.change_similarity_floor
                    and h.statement.is_current(now)
                )
            )
        ][:SHOWN_MAX]
        if change:
            change_decision = await self._ask_change(new, shown)
            new, rewritten = self._new_state(new, change_decision, shown)
            decision = ReconcileDecision(
                action=change_decision.action,
                target=change_decision.target,
                also_ends=change_decision.also_ends,
            )
            repeated = await self._repeats_sibling(new, sibling) if rewritten else None
            if repeated is not None:
                return repeated
            result = await self._apply(
                new, episodes, same_pass, shown, decision, change=True, reworded=rewritten
            )
        elif not shown:
            result = await self._add(new, episodes, asked=False, reason="nothing similar")
        else:
            decision = await self._ask(new, shown)
            result = await self._apply(new, episodes, same_pass, shown, decision)
        return await self._end_related(new, result, same_pass, [h.statement.id for h in shown])

    # --- The model's decision -------------------------------------------------

    async def _ask(self, new: NewStatement, shown: list[ScoredStatement]) -> ReconcileDecision:
        messages = self._messages(new, shown)
        decision = await self._gateway.generate(
            self.role, messages, ReconcileDecision, priority=self.priority
        )
        return self._ids(decision, shown)

    async def _ask_change(self, new: NewStatement, shown: list[ScoredStatement]) -> ChangeDecision:
        """One call for a change event: the new state, and what it does to `shown`."""
        messages = self._messages(new, shown, change=True)
        decision = await self._gateway.generate(
            self.role, messages, ChangeDecision, priority=self.priority
        )
        return self._ids(decision, shown)

    def _messages(
        self, new: NewStatement, shown: list[ScoredStatement], *, change: bool = False
    ) -> list[Message]:
        now = self._clock()
        number = {h.statement.id: i for i, h in enumerate(shown, start=1)}
        lines = ["Known statements:" if shown else "Known statements: none"]
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
        marked = " (a change)" if change else ""
        lines += [
            "",
            f"New statement{marked}: {new.text} ({new.kind.value}, {new.origin.value})",
        ]
        return [
            Message(role=Role.SYSTEM, content=self._prompt),
            Message(role=Role.USER, content="\n".join(lines)),
        ]

    @staticmethod
    def _ids[D: (ReconcileDecision, ChangeDecision)](
        decision: D, shown: list[ScoredStatement]
    ) -> D:
        """`decision` with shown numbers mapped back to statement ids (-1: not shown)."""

        def to_id(number: int) -> int:
            return shown[number - 1].statement.id if 1 <= number <= len(shown) else -1

        return decision.model_copy(
            update={
                "target": None if decision.target is None else to_id(decision.target),
                "also_ends": [to_id(n) for n in dict.fromkeys(decision.also_ends)],
            }
        )

    # --- Change events ----------------------------------------------------------

    def _new_state(
        self, new: NewStatement, decision: ChangeDecision, shown: list[ScoredStatement]
    ) -> tuple[NewStatement, bool]:
        """`new` reworded as the state the change led to, and whether it was.

        The model's wording is used only if it is a plausible statement of a
        state: not empty or overlong, without the words of a change, not a
        negation the candidate did not make ("left Brightline" stays as it
        is rather than becoming "does not work at Brightline"), and
        naming no one that neither the candidate nor a shown statement names
        (so an example's "Sam" cannot slip in). Otherwise `new` is kept as it
        is. The predicate falls back to the replaced statement's (or the
        candidate's) when the model's is missing or still describes a change.
        """
        text = " ".join((decision.new_text or "").split())
        problem = None
        if not text:
            problem = "no new state given"
        elif len(text) > _MAX_NEW_TEXT:
            problem = "new state too long"
        elif _CHANGE_WORDS.search(text):
            problem = "new state still describes a change"
        elif _NEGATED.search(text) and not _NEGATED.search(new.text):
            # "left Brightline" says what ended, not what is true instead;
            # "does not work at Brightline" would read as a statement about
            # working there.
            problem = "the change only says what ended"
        else:
            known = {
                word.lower()
                for source in (new.text, *(h.statement.text for h in shown))
                for word in _WORD.findall(source)
            }
            invented = [w for w in _CAPITALISED.findall(text) if w.lower() not in known]
            if invented:
                problem = f"new state names {', '.join(invented)}"
        if problem is not None:
            log.info("change event %r kept as it is: %s", new.text, problem)
            return new, False
        predicate = " ".join((decision.new_predicate or "").split())
        if not predicate or _CHANGE_WORDS.search(predicate):
            target = next((h.statement for h in shown if h.statement.id == decision.target), None)
            predicate = target.predicate if target is not None else new.predicate
        log.info("change event %r stored as the new state %r", new.text, text)
        return new.model_copy(update={"text": text, "predicate": predicate}), True

    async def _repeats_sibling(
        self, new: NewStatement, sibling: Callable[[Statement], bool]
    ) -> ReconcileResult | None:
        """A duplicate result if the reworded change says what a statement from the
        same pass already says (same predicate, similar text), else None."""
        hits = await self._graph.similar_statements(
            new.text, subject_id=new.subject_id, k=SIMILAR_K, include_invalid=True
        )
        for hit in hits:
            s = hit.statement
            if (
                sibling(s)
                and hit.score >= self.policy.statement_similarity_floor
                and _same_predicate(new, s)
            ):
                return ReconcileResult(
                    Outcome.DUPLICATE,
                    s,
                    s.id,
                    asked_model=True,
                    reason="the new state of this change is already stored from this turn",
                )
        return None

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
        *,
        change: bool = False,
        reworded: bool = False,
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
                if change and not reworded:
                    # The turn already stored that the old state ended, and the
                    # change was not reworded into a new state: it says nothing
                    # more ("moved from Lisbon" beside "lives in Porto"). A
                    # reworded one ("takes notes in Obsidian" beside "gave up
                    # Evernote") is new and is added below; one that repeats a
                    # sibling was already caught (`_repeats_sibling`).
                    notes.append(
                        f"{target.id} was already replaced from this turn by {head.id}: ignore"
                    )
                    return ReconcileResult(
                        Outcome.IGNORE,
                        target_id=target.id,
                        asked_model=True,
                        reason="; ".join(notes),
                    )
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
        if reworded and target.kind in _SAME_FORM and new.kind is not target.kind:
            # The new state of a change, in the form of the state it replaces.
            notes.append(f"stored as a {target.kind.value}, like {target.id}")
            new = new.model_copy(update={"kind": target.kind})
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
