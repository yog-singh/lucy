"""Reconciliation of candidate statements, the guard rails, and the confidence policy."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from lucy.core.events import UserMessage
from lucy.core.gateway import LLMGateway
from lucy.core.types import LLMChunk
from lucy.memory.graph_store import GraphNotFoundError, GraphStoreError, SQLiteGraphStore
from lucy.memory.models import NewStatement, Origin, ScoredStatement, Statement, StatementKind
from lucy.memory.policy import MemoryPolicy
from lucy.memory.reconcile import SHOWN_MAX, Outcome, Reconciler
from lucy.memory.sqlite_store import SQLiteMemoryStore
from tests.fakes import FakeLLM
from tests.graph_support import T0, Clock, HashingEmbedder, log, new
from tests.memory_extraction_support import TEST_POLICY_VALUES, decision, reply

POLICY = MemoryPolicy(**TEST_POLICY_VALUES)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
async def graph(tmp_path: Path, clock: Clock) -> AsyncIterator[SQLiteGraphStore]:
    async with SQLiteGraphStore(tmp_path / "lucy.db", HashingEmbedder(), clock=clock) as g:
        yield g


@pytest.fixture
async def memory(tmp_path: Path) -> AsyncIterator[SQLiteMemoryStore]:
    async with SQLiteMemoryStore(tmp_path / "lucy.db") as m:
        yield m


class Harness:
    """A reconciler on a scripted model, plus helpers to create episodes."""

    def __init__(self, graph: SQLiteGraphStore, memory: SQLiteMemoryStore, clock: Clock) -> None:
        self.graph = graph
        self.memory = memory
        self.clock = clock
        self.llm = FakeLLM([])
        self.reconciler = Reconciler(
            graph, LLMGateway({"extraction": self.llm}), policy=POLICY, clock=clock
        )
        self._n = 0

    def script(self, *responses: list[LLMChunk]) -> None:
        self.llm._responses.extend(list(r) for r in responses)

    async def episode(self) -> int:
        self._n += 1
        return await log(
            self.memory, UserMessage(session_id="s", turn_id=f"t{self._n}", text=f"m{self._n}")
        )

    async def existing(self, text: str, **kwargs: Any) -> Statement:
        """A statement stored from an earlier episode."""
        kwargs.setdefault("confidence", POLICY.initial_confidence(kwargs.get("origin", "stated")))
        return await self.graph.add_statement(new(text, **kwargs), [await self.episode()])

    def prompt(self, call: int = -1) -> str:
        return self.llm.calls[call][-1].content


@pytest.fixture
def h(graph: SQLiteGraphStore, memory: SQLiteMemoryStore, clock: Clock) -> Harness:
    return Harness(graph, memory, clock)


def cand(text: str, *, origin: Origin = Origin.STATED, **kwargs: Any) -> NewStatement:
    return new(text, origin=origin, confidence=POLICY.initial_confidence(origin), **kwargs)


# --- Policy -------------------------------------------------------------------


def test_policy_defaults_and_reinforcement() -> None:
    policy = MemoryPolicy()
    assert policy.initial_confidence("stated") == 0.9
    assert policy.initial_confidence(Origin.INFERRED) == 0.4
    assert policy.reinforced(0.4, Origin.INFERRED) == pytest.approx(0.55)
    assert policy.reinforced(0.55, Origin.INFERRED) == pytest.approx(0.6625)
    assert policy.reinforced(0.9, Origin.STATED) == pytest.approx(0.925)
    # A stated statement never drops below the floor (e.g. after a decay elsewhere).
    assert policy.reinforced(0.3, Origin.STATED) == 0.9
    assert policy.reinforced(1.0, Origin.INFERRED) == 1.0


def test_policy_rejects_inconsistent_thresholds() -> None:
    with pytest.raises(ValidationError):
        MemoryPolicy(entity_ambiguous_similarity=0.97, entity_match_similarity=0.9)
    with pytest.raises(ValidationError):
        MemoryPolicy(statement_similarity_floor=0.99, near_identical_similarity=0.9)
    with pytest.raises(ValidationError):
        MemoryPolicy(stated_confidence=1.5)


# --- The four outcomes ----------------------------------------------------------


async def test_nothing_similar_is_added_without_asking(h: Harness) -> None:
    await h.existing("Arjun plays the cello")
    ep = await h.episode()
    result = await h.reconciler.reconcile(cand("Arjun drinks dark roast coffee"), [ep])
    assert result.outcome is Outcome.ADD and not result.asked_model
    assert h.llm.calls == []
    assert result.statement is not None
    assert result.statement.sources == [ep] and result.statement.confidence == 0.9


async def test_the_model_sees_candidate_and_similar_statements(h: Harness) -> None:
    old = await h.existing("Arjun drinks dark roast coffee", kind=StatementKind.PREFERENCE)
    replaced = await h.existing("Arjun drinks instant coffee", origin=Origin.INFERRED)
    await h.graph.supersede(replaced.id, new("Arjun drinks filter coffee"), [await h.episode()])
    h.script(decision("add"))
    await h.reconciler.reconcile(cand("Arjun drinks decaf coffee"), [await h.episode()])
    prompt = h.prompt()
    assert "Arjun drinks dark roast coffee (preference, stated; current)" in prompt
    assert "Arjun drinks instant coffee (fact, inferred; no longer true since 2026-11-12" in prompt
    assert "replaced by [" in prompt
    assert prompt.rstrip().endswith("New statement: Arjun drinks decaf coffee (fact, stated)")
    assert h.llm.calls[-1][0].content.startswith("You keep Lucy's memory")
    # Statements are shown by number, starting at 1, not by id.
    assert "[1] " in prompt and "[2] " in prompt and old.id == 1


async def test_reinforce_raises_confidence_and_links_the_source(h: Harness) -> None:
    old = await h.existing("Arjun drinks dark roast coffee", origin=Origin.INFERRED)
    h.script(decision("reinforce", 1))
    ep = await h.episode()
    h.clock.advance(days=1)
    result = await h.reconciler.reconcile(
        cand("Arjun often drinks dark roast coffee", origin=Origin.INFERRED), [ep]
    )
    assert result.outcome is Outcome.REINFORCE and result.target_id == old.id
    stored = await h.graph.get_statement(old.id)
    assert stored is not None
    assert stored.evidence_count == 2 and stored.confidence == pytest.approx(0.55)
    assert stored.sources == [*old.sources, ep]
    assert len(await h.graph.list_statements()) == 1


async def test_stated_reinforcement_stays_at_least_at_the_floor(h: Harness) -> None:
    old = await h.existing("Arjun drinks dark roast coffee")
    await h.graph.set_confidence(old.id, 0.5)
    h.script(decision("reinforce", 1))
    await h.reconciler.reconcile(cand("Arjun drinks dark roast coffee daily"), [await h.episode()])
    stored = await h.graph.get_statement(old.id)
    assert stored is not None and stored.confidence == 0.9


async def test_supersede_keeps_the_old_statement_with_an_end_date(h: Harness) -> None:
    old = await h.existing("Arjun drinks dark roast coffee", kind=StatementKind.PREFERENCE)
    h.script(decision("supersede", 1))
    h.clock.advance(days=3)
    result = await h.reconciler.reconcile(
        cand("Arjun drinks decaf coffee", kind=StatementKind.PREFERENCE), [await h.episode()]
    )
    assert result.outcome is Outcome.SUPERSEDE and result.statement is not None
    closed = await h.graph.get_statement(old.id)
    assert closed is not None
    assert closed.superseded_by == result.statement.id
    assert closed.invalid_at == T0 + timedelta(days=3)
    assert [s.text for s in await h.graph.list_statements()] == ["Arjun drinks decaf coffee"]


async def test_supersede_ends_the_old_statement_when_the_new_one_starts(h: Harness) -> None:
    old = await h.existing("Arjun lives in Bristol city")
    h.script(decision("supersede", 1))
    start = T0 - timedelta(days=30)
    result = await h.reconciler.reconcile(
        cand("Arjun lives in Leeds city", valid_from=start), [await h.episode()]
    )
    closed = await h.graph.get_statement(old.id)
    assert result.outcome is Outcome.SUPERSEDE
    assert closed is not None and closed.invalid_at == start


async def test_supersede_drops_a_start_before_the_old_statement_began(h: Harness) -> None:
    old = await h.existing("Arjun lives in Bristol city", valid_from=T0 - timedelta(days=10))
    h.script(decision("supersede", 1))
    result = await h.reconciler.reconcile(
        cand("Arjun lives in Leeds city", valid_from=T0 - timedelta(days=400)),
        [await h.episode()],
    )
    assert result.outcome is Outcome.SUPERSEDE and result.statement is not None
    assert result.statement.valid_from is None
    closed = await h.graph.get_statement(old.id)
    assert closed is not None and closed.invalid_at == T0


async def test_ignore_stores_nothing(h: Harness) -> None:
    await h.existing("Arjun drinks dark roast coffee")
    h.script(decision("ignore"))
    result = await h.reconciler.reconcile(cand("Arjun drinks coffee"), [await h.episode()])
    assert result.outcome is Outcome.IGNORE and result.asked_model
    assert len(await h.graph.list_statements(current_only=False)) == 1


# --- Guard rails ------------------------------------------------------------------


@pytest.mark.parametrize("target", [None, 0, 2, 99])
async def test_a_target_that_was_not_shown_means_add(h: Harness, target: int | None) -> None:
    old = await h.existing("Arjun drinks dark roast coffee")
    h.script(decision("supersede", target))
    result = await h.reconciler.reconcile(cand("Arjun drinks decaf coffee"), [await h.episode()])
    assert result.outcome is Outcome.ADD
    assert "not among the shown" in result.reason
    stored = await h.graph.get_statement(old.id)
    assert stored is not None and stored.invalid_at is None


async def test_inferred_never_supersedes_stated(h: Harness) -> None:
    old = await h.existing("Arjun drinks dark roast coffee")
    h.script(decision("supersede", 1))
    result = await h.reconciler.reconcile(
        cand("Arjun drinks decaf coffee", origin=Origin.INFERRED), [await h.episode()]
    )
    assert result.outcome is Outcome.IGNORE and result.target_id == old.id
    stored = await h.graph.get_statement(old.id)
    assert stored is not None and stored.invalid_at is None
    assert len(await h.graph.list_statements(current_only=False)) == 1


async def test_inferred_may_supersede_inferred(h: Harness) -> None:
    await h.existing("Arjun drinks dark roast coffee", origin=Origin.INFERRED)
    h.script(decision("supersede", 1))
    result = await h.reconciler.reconcile(
        cand("Arjun drinks decaf coffee", origin=Origin.INFERRED), [await h.episode()]
    )
    assert result.outcome is Outcome.SUPERSEDE
    assert result.statement is not None and result.statement.confidence == 0.4


async def test_stated_candidate_replaces_a_matching_inference(h: Harness) -> None:
    old = await h.existing("Arjun drinks dark roast coffee", origin=Origin.INFERRED)
    h.script(decision("reinforce", 1))
    result = await h.reconciler.reconcile(
        cand("Arjun really drinks dark roast coffee"), [await h.episode()]
    )
    assert result.outcome is Outcome.SUPERSEDE and result.statement is not None
    assert result.statement.origin is Origin.STATED and result.statement.confidence == 0.9
    closed = await h.graph.get_statement(old.id)
    assert closed is not None and closed.superseded_by == result.statement.id


async def corrected(h: Harness) -> tuple[Statement, Statement]:
    """The "dark roast" statement (stated), then corrected to "decaf" by a stated statement."""
    old = await h.existing("Arjun drinks dark roast coffee")
    replacement = await h.graph.supersede(
        old.id, new("Arjun drinks decaf coffee"), [await h.episode()]
    )
    return old, replacement


async def test_inferred_restatement_of_a_corrected_belief_is_ignored(h: Harness) -> None:
    old, replacement = await corrected(h)
    text = "Arjun drinks dark roast coffee often"
    h.script(decision("reinforce", (await shown_lines(h, text)).index(old.text) + 1))
    result = await h.reconciler.reconcile(cand(text, origin=Origin.INFERRED), [await h.episode()])
    assert result.outcome is Outcome.IGNORE and "corrected" in result.reason
    after = await h.graph.get_statement(old.id)
    assert after is not None and after.evidence_count == 1
    assert [s.id for s in await h.graph.list_statements()] == [replacement.id]


async def test_the_model_saying_add_does_not_resurrect_a_corrected_belief(h: Harness) -> None:
    old, replacement = await corrected(h)
    h.script(decision("add"))
    result = await h.reconciler.reconcile(
        cand("Arjun drinks dark roast coffee", origin=Origin.INFERRED), [await h.episode()]
    )
    assert result.outcome is Outcome.IGNORE
    assert "corrected" in result.reason
    assert [s.id for s in await h.graph.list_statements()] == [replacement.id]
    stored = await h.graph.get_statement(old.id)
    assert stored is not None and stored.evidence_count == 1


async def test_stated_restatement_of_a_changed_belief_supersedes_the_current_one(
    h: Harness,
) -> None:
    old, replacement = await corrected(h)
    text = "Arjun drinks dark roast coffee again"
    h.script(decision("reinforce", (await shown_lines(h, text)).index(old.text) + 1))
    result = await h.reconciler.reconcile(cand(text), [await h.episode()])
    assert result.outcome is Outcome.SUPERSEDE and result.target_id == replacement.id
    assert result.statement is not None
    current = await h.graph.list_statements()
    assert [s.id for s in current] == [result.statement.id]
    first = await h.graph.get_statement(old.id)
    assert first is not None and first.superseded_by == replacement.id  # history kept


async def shown_lines(h: Harness, text: str) -> list[str]:
    """The statements the model would be shown for `text`, in order."""
    hits = await h.graph.similar_statements(text, subject_id=1, k=10, include_invalid=True)
    return [x.statement.text for x in hits if x.score >= POLICY.statement_similarity_floor]


async def test_superseding_an_old_version_retargets_the_current_one(h: Harness) -> None:
    old, replacement = await corrected(h)
    number = (await shown_lines(h, "Arjun drinks green tea coffee")).index(old.text) + 1
    h.script(decision("supersede", number))
    result = await h.reconciler.reconcile(
        cand("Arjun drinks green tea coffee"), [await h.episode()]
    )
    assert result.outcome is Outcome.SUPERSEDE and result.target_id == replacement.id


async def test_an_ended_target_without_successor_means_add(h: Harness) -> None:
    old = await h.existing("Arjun drinks dark roast coffee")
    await h.graph.invalidate(old.id, T0 - timedelta(days=1))
    h.script(decision("supersede", 1))
    result = await h.reconciler.reconcile(cand("Arjun drinks decaf coffee"), [await h.episode()])
    assert result.outcome is Outcome.ADD and "no current successor" in result.reason


async def test_add_of_a_near_identical_statement_reinforces_it(h: Harness) -> None:
    old = await h.existing("Arjun drinks dark roast coffee")
    h.script(decision("add"))
    result = await h.reconciler.reconcile(
        cand("Arjun drinks dark roast coffee."), [await h.episode()]
    )
    assert result.outcome is Outcome.REINFORCE and result.target_id == old.id


async def test_statements_from_the_same_turn_are_not_reconciled_against(h: Harness) -> None:
    ep = await h.episode()
    await h.reconciler.reconcile(
        cand("Arjun considered the Kindle reader"), [ep], pass_episodes=[ep]
    )
    result = await h.reconciler.reconcile(
        cand("Arjun considered the Kobo reader"), [ep], pass_episodes=[ep]
    )
    assert result.outcome is Outcome.ADD and not result.asked_model
    assert h.llm.calls == []
    assert len(await h.graph.list_statements()) == 2


async def test_a_belief_already_replaced_in_this_turn_is_not_replaced_again(h: Harness) -> None:
    old = await h.existing("Arjun drinks dark roast coffee")
    ep = await h.episode()
    h.script(decision("supersede", 1))
    first = await h.reconciler.reconcile(
        cand("Arjun drinks decaf coffee"), [ep], pass_episodes=[ep]
    )
    assert first.outcome is Outcome.SUPERSEDE
    # A sibling candidate also "replaces" dark roast, which already ended this turn.
    h.script(decision("supersede", 1))
    second = await h.reconciler.reconcile(
        cand("Arjun quit dark roast coffee for sleep"), [ep], pass_episodes=[ep]
    )
    assert second.outcome is Outcome.ADD and "already replaced" in second.reason
    assert {s.text for s in await h.graph.list_statements()} == {
        "Arjun drinks decaf coffee",
        "Arjun quit dark roast coffee for sleep",
    }
    closed = await h.graph.get_statement(old.id)
    assert closed is not None and closed.superseded_by == first.statement.id  # type: ignore[union-attr]


# --- Running twice ----------------------------------------------------------------


async def test_rerun_of_an_added_statement_is_a_duplicate(h: Harness) -> None:
    ep = await h.episode()
    first = await h.reconciler.reconcile(cand("Arjun drinks dark roast coffee"), [ep])
    again = await h.reconciler.reconcile(cand("Arjun drinks dark roast coffee"), [ep])
    assert first.outcome is Outcome.ADD and again.outcome is Outcome.DUPLICATE
    assert h.llm.calls == []
    assert len(await h.graph.list_statements()) == 1


async def test_rerun_of_a_reinforcement_does_not_count_twice(h: Harness) -> None:
    old = await h.existing("Arjun drinks dark roast coffee", origin=Origin.INFERRED)
    ep = await h.episode()
    h.script(decision("reinforce", 1), decision("reinforce", 1))
    text = "Arjun usually drinks dark roast coffee"
    first = await h.reconciler.reconcile(cand(text, origin=Origin.INFERRED), [ep])
    again = await h.reconciler.reconcile(cand(text, origin=Origin.INFERRED), [ep])
    assert first.outcome is Outcome.REINFORCE and again.outcome is Outcome.DUPLICATE
    stored = await h.graph.get_statement(old.id)
    assert stored is not None
    assert stored.evidence_count == 2 and stored.confidence == pytest.approx(0.55)


async def test_rerun_of_a_supersession_does_not_supersede_twice(h: Harness) -> None:
    old = await h.existing("Arjun drinks dark roast coffee")
    ep = await h.episode()
    h.script(decision("supersede", 1))
    first = await h.reconciler.reconcile(cand("Arjun drinks decaf coffee"), [ep])
    again = await h.reconciler.reconcile(cand("Arjun drinks decaf coffee"), [ep])
    assert first.outcome is Outcome.SUPERSEDE and again.outcome is Outcome.DUPLICATE
    assert len(await h.graph.list_statements(current_only=False)) == 2
    closed = await h.graph.get_statement(old.id)
    assert first.statement is not None
    assert closed is not None and closed.superseded_by == first.statement.id


# --- Statements of the same pass and older ones -------------------------------------


async def test_an_older_statement_whose_sources_are_all_in_the_evidence_is_compared(
    h: Harness,
) -> None:
    """Reflection re-observes an inference with all its earlier messages plus a new one."""
    a, b = await h.episode(), await h.episode()
    text = "Arjun usually goes to the gym on Mondays"
    old = await h.graph.add_statement(
        new(text, kind=StatementKind.HABIT, origin=Origin.INFERRED, confidence=0.4), [a, b]
    )
    c = await h.episode()
    h.script(decision("reinforce", 1))
    result = await h.reconciler.reconcile(
        cand("Arjun goes to the gym most Mondays", origin=Origin.INFERRED), [a, b, c]
    )
    assert result.outcome is Outcome.REINFORCE and result.target_id == old.id
    assert len(h.llm.calls) == 1
    stored = await h.graph.get_statement(old.id)
    assert stored is not None and stored.sources == [a, b, c] and stored.evidence_count == 2
    assert len(await h.graph.list_statements(current_only=False)) == 1


async def test_without_a_pass_nothing_is_left_out(h: Harness) -> None:
    """The same episodes but no `pass_episodes`: the statement is shown to the model."""
    ep = await h.episode()
    await h.graph.add_statement(new("Arjun considered the Kindle reader"), [ep])
    h.script(decision("add"))
    result = await h.reconciler.reconcile(cand("Arjun considered the Kobo reader"), [ep])
    assert result.outcome is Outcome.ADD and result.asked_model


async def test_a_sibling_in_other_words_is_a_duplicate(h: Harness) -> None:
    """A re-run after a crash, with the model wording the same statement differently."""
    ep = await h.episode()
    first = await h.reconciler.reconcile(
        cand("Arjun lives in Leeds city", predicate="lives in", object_value="Leeds"),
        [ep],
        pass_episodes=[ep],
    )
    again = await h.reconciler.reconcile(
        cand("Arjun's home is in Leeds city", predicate="Lives in", object_value="leeds"),
        [ep],
        pass_episodes=[ep],
    )
    assert first.outcome is Outcome.ADD
    assert again.outcome is Outcome.DUPLICATE and again.target_id == first.statement.id  # type: ignore[union-attr]
    assert h.llm.calls == []
    assert len(await h.graph.list_statements(current_only=False)) == 1


@pytest.mark.parametrize(
    ("predicate", "value", "kind"),
    [
        ("chose", "Kobo", StatementKind.DECISION),  # another predicate (the choice vs an option)
        ("option", "Kindle", StatementKind.DECISION),  # another object
        ("option", "Kobo", StatementKind.FACT),  # another kind
        ("option", None, StatementKind.DECISION),  # no object to compare
    ],
)
async def test_siblings_that_differ_are_kept(
    h: Harness, predicate: str, value: str | None, kind: StatementKind
) -> None:
    ep = await h.episode()
    base = dict(predicate="option", kind=StatementKind.DECISION)
    await h.reconciler.reconcile(
        cand("Arjun considered the Kobo reader", object_value="Kobo", **base),
        [ep],
        pass_episodes=[ep],
    )
    other = await h.reconciler.reconcile(
        cand(
            "Arjun considered the Kobo reader too",
            object_value=value,
            predicate=predicate,
            kind=kind,
        ),
        [ep],
        pass_episodes=[ep],
    )
    assert other.outcome is Outcome.ADD and not other.asked_model
    assert len(await h.graph.list_statements()) == 2


async def test_a_sibling_in_other_words_from_another_pass_is_not_a_duplicate(h: Harness) -> None:
    """Reflection (no pass) with the same object and predicate still reaches the model."""
    ep = await h.episode()
    await h.graph.add_statement(
        new("Arjun lives in Leeds city", predicate="lives in", object_value="Leeds"), [ep]
    )
    h.script(decision("reinforce", 1))
    result = await h.reconciler.reconcile(
        cand("Arjun's home is in Leeds city", predicate="lives in", object_value="Leeds"),
        [ep, await h.episode()],
    )
    assert result.outcome is Outcome.REINFORCE and result.asked_model


# --- Dates of a supersession -------------------------------------------------------


async def test_supersede_ends_the_old_statement_when_the_user_said_it(h: Harness) -> None:
    old = await h.existing("Arjun works at Hartley and Sons")
    said = T0 - timedelta(days=5)
    h.script(decision("supersede", 1))
    result = await h.reconciler.reconcile(
        cand("Arjun works at Nordwind and Sons", observed_at=said), [await h.episode()]
    )
    assert result.outcome is Outcome.SUPERSEDE and result.statement is not None
    closed = await h.graph.get_statement(old.id)
    assert closed is not None and closed.invalid_at == said  # not the clock (T0)
    assert result.statement.valid_from == said


async def test_a_start_before_the_old_statement_moves_to_when_it_was_said(h: Harness) -> None:
    old = await h.existing("Arjun lives in Bristol city", valid_from=T0 - timedelta(days=10))
    said = T0 - timedelta(days=2)
    h.script(decision("supersede", 1))
    result = await h.reconciler.reconcile(
        cand("Arjun lives in Leeds city", valid_from=T0 - timedelta(days=400), observed_at=said),
        [await h.episode()],
    )
    assert result.statement is not None and result.statement.valid_from == said
    closed = await h.graph.get_statement(old.id)
    assert closed is not None and closed.invalid_at == said


# --- Goals and activities that end -------------------------------------------------------


async def goal_and_habit(h: Harness) -> tuple[Statement, Statement]:
    goal = await h.existing("Arjun wants to learn Italian", kind=StatementKind.GOAL)
    habit = await h.existing("Arjun practises to learn Italian", kind=StatementKind.HABIT)
    return goal, habit


def number_of(lines: list[str], text: str) -> int:
    return lines.index(text) + 1


async def test_a_goal_given_up_is_closed_by_a_fact_and_ends_the_habit(h: Harness) -> None:
    goal, habit = await goal_and_habit(h)
    text = "Arjun gave up wanting to learn Italian"
    lines = await shown_lines(h, text)
    said = T0 - timedelta(days=3)
    h.script(decision("supersede", number_of(lines, goal.text), [number_of(lines, habit.text)]))
    result = await h.reconciler.reconcile(
        cand(text, kind=StatementKind.DECISION, observed_at=said), [await h.episode()]
    )
    assert result.outcome is Outcome.SUPERSEDE and result.statement is not None
    assert result.statement.kind is StatementKind.FACT  # a goal is closed by a fact
    assert "a goal or habit is closed by a fact" in result.reason
    assert result.ended_ids == [habit.id]
    for old in (goal, habit):
        closed = await h.graph.get_statement(old.id)
        assert closed is not None
        assert closed.invalid_at == said and closed.superseded_by == result.statement.id
    assert "also_ends" in h.llm.calls[-1][0].content  # the prompt explains the field


async def test_also_ends_guard_rails(h: Harness) -> None:
    goal, habit = await goal_and_habit(h)
    fact = await h.existing("Arjun learns Italian from a book")
    inferred_habit = await h.existing(
        "Arjun practises Italian on Duolingo to learn",
        kind=StatementKind.HABIT,
        origin=Origin.INFERRED,
    )
    text = "Arjun gave up wanting to learn Italian"
    lines = await shown_lines(h, text)
    numbers = [number_of(lines, s.text) for s in (fact, goal, habit, inferred_habit)] + [99]
    h.script(decision("supersede", number_of(lines, inferred_habit.text), numbers))
    # An inferred candidate: it may end inferred statements, not stated ones.
    result = await h.reconciler.reconcile(cand(text, origin=Origin.INFERRED), [await h.episode()])
    assert result.outcome is Outcome.SUPERSEDE and result.ended_ids == []
    assert f"not ending {fact.id}: a fact" in result.reason
    assert f"not ending {goal.id}: stated, and the candidate is inferred" in result.reason
    current = {s.id for s in await h.graph.list_statements()}
    assert {goal.id, habit.id, fact.id} <= current and inferred_habit.id not in current


async def test_also_ends_is_ignored_unless_superseding(h: Harness) -> None:
    goal, habit = await goal_and_habit(h)
    text = "Arjun wants to learn Italian slowly"
    lines = await shown_lines(h, text)
    h.script(decision("add", None, [number_of(lines, habit.text)]))
    result = await h.reconciler.reconcile(cand(text), [await h.episode()])
    assert result.outcome is Outcome.ADD and result.ended_ids == []
    assert len(await h.graph.list_statements()) == 3


async def test_a_goal_closed_by_reflection_is_found_through_its_outcome(h: Harness) -> None:
    """Reflection closed the goal first (pointing at its outcome); extraction then
    reports the same news: it replaces the outcome instead of adding a second one."""
    goal, _ = await goal_and_habit(h)
    outcome = await h.graph.add_statement(
        new("Arjun gave up the goal: learning Italian", origin=Origin.INFERRED, confidence=0.4),
        [await h.episode()],
    )
    await h.graph.end_statement(goal.id, T0 - timedelta(days=1), superseded_by=outcome.id)
    text = "Arjun gave up wanting to learn Italian"
    lines = await shown_lines(h, text)
    h.script(decision("supersede", number_of(lines, goal.text)))
    result = await h.reconciler.reconcile(cand(text), [await h.episode()])
    assert result.outcome is Outcome.SUPERSEDE and result.target_id == outcome.id
    current = [s.text for s in await h.graph.list_statements() if "gave up" in s.text]
    assert current == [text]


# --- Supersession targets -------------------------------------------------------------


async def test_a_target_about_another_subject_is_never_superseded(h: Harness) -> None:
    other = await h.graph.create_entity("person", "Tobias")
    theirs = await h.graph.add_statement(
        new("Tobias lives in Lisbon city", subject_id=other.id), [await h.episode()]
    )
    original = h.graph.similar_statements

    async def leaky(*args: object, **kwargs: object) -> list[ScoredStatement]:
        hits = await original(*args, **kwargs)  # type: ignore[arg-type]
        return [*hits, ScoredStatement(statement=theirs, score=0.99)]

    h.graph.similar_statements = leaky  # type: ignore[method-assign]
    h.script(decision("supersede", 1))
    result = await h.reconciler.reconcile(cand("Arjun lives in Porto city"), [await h.episode()])
    assert result.outcome is Outcome.ADD and "another subject" in result.reason
    kept = await h.graph.get_statement(theirs.id)
    assert kept is not None and kept.invalid_at is None


# --- Store support ----------------------------------------------------------------------


async def test_end_statement_keeps_an_earlier_end_and_checks_its_arguments(h: Harness) -> None:
    a = await h.existing("Arjun drinks dark roast coffee", valid_from=T0 - timedelta(days=9))
    b = await h.existing("Arjun drinks decaf coffee")
    c = await h.existing("Arjun drinks tea")
    with pytest.raises(GraphStoreError):
        await h.graph.end_statement(a.id, T0 - timedelta(days=10))
    with pytest.raises(GraphStoreError):
        await h.graph.end_statement(a.id, T0, superseded_by=a.id)
    with pytest.raises(GraphNotFoundError):
        await h.graph.end_statement(a.id, T0, superseded_by=999)
    ended = await h.graph.end_statement(a.id, T0 - timedelta(days=2), superseded_by=b.id)
    assert ended.invalid_at == T0 - timedelta(days=2) and ended.superseded_by == b.id
    again = await h.graph.end_statement(a.id, T0, superseded_by=b.id)
    assert again.invalid_at == T0 - timedelta(days=2)  # the earlier end stays
    with pytest.raises(GraphStoreError):
        await h.graph.end_statement(a.id, T0, superseded_by=c.id)
    assert a.id not in [x.statement.id for x in await h.graph.similar_statements("dark roast")]


async def test_the_second_hop_finds_a_related_goal_the_first_did_not_show(h: Harness) -> None:
    """The goal is not similar to the candidate (not shown), but shares its object's
    words: after the habit is superseded, the model is asked about it once more."""
    habit = await h.existing("Arjun practises to learn Italian", kind=StatementKind.HABIT)
    goal = await h.existing("Arjun dreams of fluent Italiano Italian", kind=StatementKind.GOAL)
    other = await h.existing("Arjun wants to run a marathon", kind=StatementKind.GOAL)
    text = "Arjun gave up wanting to learn Italian"
    assert goal.text not in await shown_lines(h, text)
    said = T0 - timedelta(days=2)
    h.script(decision("supersede", 1), decision("supersede", 1))
    result = await h.reconciler.reconcile(
        cand(text, predicate="gave up", object_value="learning Italian", observed_at=said),
        [await h.episode()],
    )
    assert result.outcome is Outcome.SUPERSEDE and result.target_id == habit.id
    assert result.ended_ids == [goal.id]
    second = h.prompt()
    assert "Italiano" in second and "marathon" not in second and "practises" not in second
    closed = await h.graph.get_statement(goal.id)
    assert closed is not None and closed.invalid_at == said
    assert closed.superseded_by == result.statement.id  # type: ignore[union-attr]
    kept = await h.graph.get_statement(other.id)
    assert kept is not None and kept.invalid_at is None


async def test_an_added_ending_statement_also_asks_about_related_ones(h: Harness) -> None:
    """Nothing similar enough to compare with, but the predicate says something ended."""
    goal = await h.existing("Arjun dreams of fluent Italiano Italian", kind=StatementKind.GOAL)
    h.script(decision("ignore"))
    result = await h.reconciler.reconcile(
        cand("Arjun quit the evening class", predicate="gave up", object_value="Italian"),
        [await h.episode()],
    )
    assert result.outcome is Outcome.ADD and result.ended_ids == []
    assert len(h.llm.calls) == 1 and "Italiano" in h.prompt()  # asked; said no
    kept = await h.graph.get_statement(goal.id)
    assert kept is not None and kept.invalid_at is None


async def test_no_second_hop_for_an_ordinary_change(h: Harness) -> None:
    leeds = await h.existing("Arjun lives in Leeds city")
    await h.existing("Arjun wants to visit York minster", kind=StatementKind.GOAL)
    text = "Arjun lives in York city"
    h.script(decision("supersede", (await shown_lines(h, text)).index(leeds.text) + 1))
    result = await h.reconciler.reconcile(
        cand(text, predicate="lives in", object_value="York"), [await h.episode()]
    )
    assert result.outcome is Outcome.SUPERSEDE and result.target_id == leeds.id
    assert len(h.llm.calls) == 1  # a fact replaced a fact: no second hop


async def test_no_second_hop_when_a_habit_replaces_a_habit(h: Harness) -> None:
    old = await h.existing("Arjun swims every Tuesday morning", kind=StatementKind.HABIT)
    await h.existing("Arjun wants to swim a 5k open water swim", kind=StatementKind.GOAL)
    text = "Arjun swims every Thursday morning"
    h.script(decision("supersede", (await shown_lines(h, text)).index(old.text) + 1))
    result = await h.reconciler.reconcile(
        cand(text, kind=StatementKind.HABIT, predicate="swims", object_value="swim"),
        [await h.episode()],
    )
    assert result.outcome is Outcome.SUPERSEDE and result.target_id == old.id
    assert len(h.llm.calls) == 1


# --- Change events ------------------------------------------------------------------

#: The real gap with a change event: under the usual floor, above the change floor.
WIDE = MemoryPolicy(
    **TEST_POLICY_VALUES | {"statement_similarity_floor": 0.95, "change_similarity_floor": 0.0}
)


def change_reply(
    new_text: str | None,
    action: str,
    target: int | None = None,
    *,
    predicate: str | None = None,
) -> list[LLMChunk]:
    """A scripted reply to a change event's call (`target` is a shown number)."""
    return reply(
        {"new_text": new_text, "new_predicate": predicate, "action": action, "target": target}
    )


async def test_a_change_event_supersedes_the_old_state_with_the_new_state(h: Harness) -> None:
    """ "switched to decaf" shares no words with "drinks dark roast": it is shown anyway, and
    the new state, in the old statement's form and kind, replaces it from the message date."""
    h.reconciler.policy = WIDE
    old = await h.existing(
        "Arjun drinks dark roast coffee every morning",
        kind=StatementKind.PREFERENCE,
        predicate="drinks",
    )
    said = T0 + timedelta(days=21)
    h.script(change_reply("Arjun drinks decaf coffee.", "supersede", 1, predicate="drinks"))
    result = await h.reconciler.reconcile(
        cand("Arjun switched to decaf for good", predicate="switched to", observed_at=said),
        [await h.episode()],
        change=True,
    )
    assert result.outcome is Outcome.SUPERSEDE and result.target_id == old.id
    assert len(h.llm.calls) == 1
    prompt = h.prompt()
    assert "[1] Arjun drinks dark roast coffee every morning (preference" in prompt
    assert "New statement (a change): Arjun switched to decaf for good (fact, stated)" in prompt
    stored = result.statement
    assert stored is not None
    assert (stored.text, stored.predicate) == ("Arjun drinks decaf coffee.", "drinks")
    assert stored.kind is StatementKind.PREFERENCE and stored.valid_from == said
    closed = await h.graph.get_statement(old.id)
    assert closed is not None
    assert closed.superseded_by == stored.id and closed.invalid_at == said
    assert [s.text for s in await h.graph.list_statements(current_only=False)] == [
        "Arjun drinks dark roast coffee every morning",
        "Arjun drinks decaf coffee.",
    ]


async def test_a_change_event_ends_the_old_state_on_the_date_the_user_gave(h: Harness) -> None:
    h.reconciler.policy = WIDE
    old = await h.existing("Arjun drives to work", kind=StatementKind.HABIT, valid_from=T0)
    since = T0 + timedelta(days=10)
    h.script(change_reply("Arjun cycles to work.", "supersede", 1, predicate="cycles"))
    result = await h.reconciler.reconcile(
        cand(
            "Arjun started cycling to work",
            predicate="started",
            valid_from=since,
            observed_at=since + timedelta(days=4),
        ),
        [await h.episode()],
        change=True,
    )
    assert result.statement is not None and result.statement.kind is StatementKind.HABIT
    closed = await h.graph.get_statement(old.id)
    assert closed is not None and closed.invalid_at == result.statement.valid_from == since


async def test_a_change_event_with_nothing_known_is_stored_as_the_new_state(h: Harness) -> None:
    h.script(change_reply("Arjun takes notes in Obsidian.", "add", predicate="takes notes in"))
    result = await h.reconciler.reconcile(
        cand("Arjun switched to Obsidian for notes", predicate="switched to"),
        [await h.episode()],
        change=True,
    )
    assert result.outcome is Outcome.ADD and result.asked_model
    assert len(h.llm.calls) == 1  # one call, even with nothing to show
    assert "Known statements: none" in h.prompt()
    assert [s.text for s in await h.graph.list_statements()] == ["Arjun takes notes in Obsidian."]


async def test_each_change_event_costs_exactly_one_call(h: Harness) -> None:
    h.reconciler.policy = WIDE
    for i in range(8):
        await h.existing(f"Arjun fact number {i} about things")
    h.script(change_reply("Arjun lives in Porto city.", "add"))
    await h.reconciler.reconcile(
        cand("Arjun moved to Porto city", predicate="moved to"), [await h.episode()], change=True
    )
    assert len(h.llm.calls) == 1
    shown = [line for line in h.prompt().splitlines() if line.startswith("[")]
    assert len(shown) == SHOWN_MAX


@pytest.mark.parametrize(
    ("new_text", "why"),
    [
        (None, "nothing given"),
        ("Arjun has switched to decaf coffee.", "still a change"),
        ("Sam drinks decaf coffee.", "a name from nowhere (the prompt's example)"),
        ("x" * 301, "too long"),
    ],
)
async def test_an_unusable_new_state_keeps_the_change_event(
    h: Harness, new_text: str | None, why: str
) -> None:
    h.reconciler.policy = WIDE
    old = await h.existing("Arjun drinks dark roast coffee", kind=StatementKind.PREFERENCE)
    h.script(change_reply(new_text, "supersede", 1, predicate="drinks"))
    result = await h.reconciler.reconcile(
        cand("Arjun switched to decaf", predicate="switched to"),
        [await h.episode()],
        change=True,
    )
    assert result.outcome is Outcome.SUPERSEDE and result.target_id == old.id, why
    assert result.statement is not None
    assert (result.statement.text, result.statement.predicate) == (
        "Arjun switched to decaf",
        "switched to",
    )
    assert result.statement.kind is StatementKind.FACT  # not reworded: its own kind


async def test_a_change_that_only_says_what_ended_is_not_reworded_as_a_negation(
    h: Harness,
) -> None:
    h.reconciler.policy = WIDE
    old = await h.existing("Arjun works at Brightline studio", predicate="works at")
    h.script(change_reply("Arjun does not work at Brightline.", "supersede", 1))
    result = await h.reconciler.reconcile(
        cand("Arjun left Brightline studio", predicate="left"), [await h.episode()], change=True
    )
    assert result.outcome is Outcome.SUPERSEDE and result.target_id == old.id
    assert result.statement is not None and result.statement.text == "Arjun left Brightline studio"


async def test_a_change_event_never_touches_another_subjects_statements(h: Harness) -> None:
    h.reconciler.policy = WIDE
    tobias = await h.graph.create_entity("person", "Tobias")
    theirs = await h.graph.add_statement(
        new("Tobias drinks dark roast coffee", subject_id=tobias.id), [await h.episode()]
    )
    h.script(change_reply("Arjun drinks decaf coffee.", "add"))
    await h.reconciler.reconcile(
        cand("Arjun switched to decaf coffee", predicate="switched to"),
        [await h.episode()],
        change=True,
    )
    assert "Tobias" not in h.prompt()  # only the subject's own statements are shown
    # Even if the store returned it and the model named it, it is not superseded.
    original = h.graph.similar_statements

    async def leaky(*args: object, **kwargs: object) -> list[ScoredStatement]:
        hits = await original(*args, **kwargs)  # type: ignore[arg-type]
        return [ScoredStatement(statement=theirs, score=0.99), *hits]

    h.graph.similar_statements = leaky  # type: ignore[method-assign]
    h.script(change_reply("Arjun drinks oat milk lattes.", "supersede", 1))
    result = await h.reconciler.reconcile(
        cand("Arjun switched to oat milk lattes", predicate="switched to"),
        [await h.episode()],
        change=True,
    )
    assert result.outcome is Outcome.ADD and "another subject" in result.reason
    kept = await h.graph.get_statement(theirs.id)
    assert kept is not None and kept.invalid_at is None


async def test_an_inferred_change_never_supersedes_a_stated_statement(h: Harness) -> None:
    h.reconciler.policy = WIDE
    old = await h.existing("Arjun drinks dark roast coffee")
    h.script(change_reply("Arjun drinks decaf coffee.", "supersede", 1))
    result = await h.reconciler.reconcile(
        cand("Arjun switched to decaf", predicate="switched to", origin=Origin.INFERRED),
        [await h.episode()],
        change=True,
    )
    assert result.outcome is Outcome.IGNORE
    kept = await h.graph.get_statement(old.id)
    assert kept is not None and kept.invalid_at is None


async def test_a_change_whose_target_this_turn_already_replaced_is_ignored(h: Harness) -> None:
    """ "Tobias moved from Lisbon": the turn's "lives in Porto" already replaced Lisbon."""
    old = await h.existing("Arjun lives in Lisbon city", predicate="lives in")
    ep = await h.episode()
    h.script(decision("supersede", 1))
    first = await h.reconciler.reconcile(
        cand("Arjun lives in Porto city", predicate="lives in"), [ep], pass_episodes=[ep]
    )
    assert first.outcome is Outcome.SUPERSEDE
    h.script(change_reply(None, "supersede", 1))
    second = await h.reconciler.reconcile(
        cand("Arjun moved from Lisbon city", predicate="moved from"),
        [ep],
        pass_episodes=[ep],
        change=True,
    )
    assert second.outcome is Outcome.IGNORE and "already replaced" in second.reason
    assert [s.text for s in await h.graph.list_statements()] == ["Arjun lives in Porto city"]
    closed = await h.graph.get_statement(old.id)
    assert closed is not None and closed.superseded_by == first.statement.id  # type: ignore[union-attr]


async def test_a_reworded_change_beside_an_ending_from_this_turn_is_added(h: Harness) -> None:
    """ "I moved my notes over to Obsidian": the turn's "gave up Evernote" already ended the
    old habit; the new state is still worth keeping."""
    old = await h.existing("Arjun keeps notes in Evernote", kind=StatementKind.HABIT)
    ep = await h.episode()
    await h.graph.supersede(old.id, cand("Arjun gave up using Evernote"), [ep])
    h.script(change_reply("Arjun keeps notes in Obsidian.", "supersede", 1, predicate="keeps"))
    result = await h.reconciler.reconcile(
        cand("Arjun moved his notes to Obsidian", predicate="moved"),
        [ep],
        pass_episodes=[ep],
        change=True,
    )
    assert result.outcome is Outcome.ADD and "already replaced" in result.reason
    assert {s.text for s in await h.graph.list_statements()} == {
        "Arjun gave up using Evernote",
        "Arjun keeps notes in Obsidian.",
    }
    closed = await h.graph.get_statement(old.id)
    assert closed is not None and closed.invalid_at is not None


async def test_a_reworded_change_that_repeats_a_sibling_is_a_duplicate(h: Harness) -> None:
    ep = await h.episode()
    sibling = await h.reconciler.reconcile(
        cand("Arjun works at Kestrel Studio", predicate="works at"), [ep], pass_episodes=[ep]
    )
    h.script(change_reply("Arjun works at Kestrel Studio.", "add", predicate="works at"))
    result = await h.reconciler.reconcile(
        cand("Arjun joined Kestrel Studio as a designer", predicate="joined"),
        [ep],
        pass_episodes=[ep],
        change=True,
    )
    assert result.outcome is Outcome.DUPLICATE and result.target_id == sibling.statement.id  # type: ignore[union-attr]
    assert len(await h.graph.list_statements()) == 1


def test_the_change_floor_may_not_exceed_the_statement_floor() -> None:
    with pytest.raises(ValidationError):
        MemoryPolicy(change_similarity_floor=0.9, statement_similarity_floor=0.84)
