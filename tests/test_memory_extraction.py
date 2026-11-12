"""Extraction of statements from conversation turns (memory/extraction.py)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from lucy.core.events import UserMessage
from lucy.core.gateway import LLMGateway
from lucy.core.types import LLMChunk
from lucy.memory.extraction import (
    ExtractedStatement,
    Extractor,
    check_candidate,
    parse_date,
    turn_key,
)
from lucy.memory.graph_store import SQLiteGraphStore
from lucy.memory.models import USER_ENTITY_ID, Origin, StatementKind
from lucy.memory.policy import MemoryPolicy
from lucy.memory.reconcile import Outcome
from lucy.memory.sqlite_store import SQLiteMemoryStore
from lucy.prompts import load_prompt
from tests.fakes import FakeLLM
from tests.graph_support import T0, Clock, HashingEmbedder, log
from tests.memory_extraction_support import (
    TEST_POLICY_VALUES,
    cand,
    decision,
    extraction,
    turn,
)

POLICY = MemoryPolicy(**TEST_POLICY_VALUES)
DAY1 = datetime(2026, 11, 2, 8, 0, tzinfo=UTC)  # a Monday


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


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM([])


@pytest.fixture
def extractor(graph: SQLiteGraphStore, llm: FakeLLM, clock: Clock) -> Extractor:
    return Extractor(graph, LLMGateway({"extraction": llm}), policy=POLICY, clock=clock)


def script(llm: FakeLLM, *responses: list[LLMChunk]) -> None:
    llm._responses.extend(list(r) for r in responses)


def prompt_of(llm: FakeLLM, call: int = 0) -> str:
    return llm.calls[call][-1].content


async def unextracted(graph: SQLiteGraphStore) -> list[int]:
    return [r.id for r in await graph.unextracted_episodes(100)]


# --- Units of work and context ----------------------------------------------------


async def test_turns_are_processed_oldest_first_with_their_dates(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "First thing", "Reply one", ts=DAY1)
    await turn(memory, "t2", "Second thing", "Reply two", ts=DAY1 + timedelta(days=3))
    script(llm, extraction(), extraction())
    report = await extractor.run(5)
    assert [t.key for t in report.turns] == ["turn:t1", "turn:t2"] and not report.more
    first, second = prompt_of(llm, 0), prompt_of(llm, 1)
    assert "This turn:\nUser: First thing\nLucy: Reply one" in first
    assert "Date of this turn: 2026-11-02 (Monday)" in first
    assert "Date of this turn: 2026-11-05 (Thursday)" in second
    assert "Today: 2026-11-12 (Thursday)" in first
    # The earlier turn is context for the later one, not part of it.
    assert "Earlier conversation (context only):\nUser: First thing\nLucy: Reply one" in second
    assert "This turn:\nUser: Second thing" in second
    assert llm.calls[0][0].content == load_prompt("extraction")
    assert await unextracted(graph) == []


async def test_context_is_bounded_and_from_the_same_session(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    for i in range(5):
        await turn(memory, f"old{i}", f"old message {i}", f"old reply {i}", ts=DAY1)
    await turn(memory, "other", "other session", ts=DAY1, session="s2")
    await graph.mark_extracted(await unextracted(graph))
    await turn(memory, "now", "the new message", ts=DAY1)
    script(llm, extraction())
    await extractor.run(1)
    prompt = prompt_of(llm)
    assert "other session" not in prompt
    # Six earlier content episodes: the last three turns' user and assistant messages.
    assert "old message 1" not in prompt
    assert "old message 2" in prompt and "old reply 4" in prompt


async def test_known_entities_and_the_users_name_are_shown(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await graph.rename_entity(USER_ENTITY_ID, "Arjun", keep_old_as_alias=False)
    await graph.add_alias(USER_ENTITY_ID, "Arj")
    await graph.create_entity("person", "Meera", aliases=["my sister"])
    await graph.create_entity("place", "Leeds")
    await graph.create_entity("place", "Paris")
    await turn(memory, "t1", "My sister is visiting", "Is she coming from Leeds?", ts=DAY1)
    script(llm, extraction())
    await extractor.run(1)
    prompt = prompt_of(llm)
    assert "The user's name: Arjun (also called Arj)" in prompt
    assert "Known people and things mentioned:" in prompt
    assert "- Meera (person; also called my sister)" in prompt
    assert "- Leeds (place)" in prompt
    assert "Paris" not in prompt


async def test_unknown_user_name_is_said(
    memory: SQLiteMemoryStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "Hello", ts=DAY1)
    script(llm, extraction())
    await extractor.run(1)
    assert "The user's name: not known yet" in prompt_of(llm)


async def test_only_what_the_user_said_is_a_source(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    ids = await turn(
        memory, "t1", "What's on? I love jazz.", "Here is the list.", ts=DAY1, tool="Jazz night"
    )
    script(llm, extraction(cand("The user loves jazz", kind="preference")))
    report = await extractor.run(1)
    prompt = prompt_of(llm)
    assert "Tool result: Jazz night" in prompt and "Lucy: Here is the list." in prompt
    [statement] = await graph.list_statements()
    assert statement.sources == [ids[0]]
    assert report.turns[0].source_ids == [ids[0]]
    assert sorted(report.turns[0].episode_ids) == sorted(ids)
    assert await unextracted(graph) == []


async def test_empty_result_marks_the_turn_extracted(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "How are you?", "Fine, thanks!", ts=DAY1)
    script(llm, extraction())
    report = await extractor.run(3)
    assert len(report.turns) == 1 and report.turns[0].results == []
    assert await unextracted(graph) == [] and await graph.list_statements() == []


async def test_a_turn_without_a_user_message_is_marked_without_a_model_call(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", None, "Reminder: dentist at 4.", ts=DAY1)
    report = await extractor.run(1)
    assert report.turns[0].skipped == "no user message" and llm.calls == []
    assert await unextracted(graph) == []


async def test_the_running_turn_waits_until_it_ends(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "Unfinished", ts=DAY1, ended=False)
    assert await extractor.ready_turns(5) == []
    assert (await extractor.run(5)).turns == []
    # A later turn shows the first one is over (e.g. it was cut short by a crash).
    await turn(memory, "t2", "Next", ts=DAY1, ended=False)
    ready = await extractor.ready_turns(5)
    assert [turn_key(t) for t in ready] == ["turn:t1"]
    await log(memory, UserMessage(session_id="s1", turn_id=None, text="no turn", ts=DAY1))
    ready = await extractor.ready_turns(5)
    assert [turn_key(t) for t in ready][:2] == ["turn:t1", "turn:t2"]
    assert turn_key(ready[2]).startswith("episode:")
    script(llm, extraction(), extraction(), extraction())
    assert len((await extractor.run(5)).turns) == 3
    assert await unextracted(graph) == []


async def test_a_bounded_run_reports_more_work(
    memory: SQLiteMemoryStore, extractor: Extractor, llm: FakeLLM
) -> None:
    for i in range(3):
        await turn(memory, f"t{i}", f"message {i}", ts=DAY1)
    script(llm, extraction(), extraction(), extraction())
    first = await extractor.run(2)
    assert len(first.turns) == 2 and first.more
    second = await extractor.run(2)
    assert len(second.turns) == 1 and not second.more


# --- Statements --------------------------------------------------------------------


async def test_a_turn_becomes_statements_with_entities_and_dates(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    [user_ep, _] = await turn(memory, "t1", "My sister Meera moved to Leeds in March", ts=DAY1)
    script(
        llm,
        extraction(
            cand("Meera is the user's sister", predicate="sister", entity=("Meera", "person")),
            cand(
                "Meera lives in Leeds",
                subject=("Meera", "person"),
                predicate="lives in",
                entity=("Leeds", "city"),
                valid_from="2026-03",
            ),
            cand("The user is vegetarian", value="vegetarian"),
        ),
    )
    report = await extractor.run(1)
    assert [r.outcome for r in report.turns[0].results] == [Outcome.ADD] * 3
    [meera] = await graph.find_entities("Meera")
    [leeds] = await graph.find_entities("Leeds")
    assert (meera.type, leeds.type) == ("person", "city")
    sister, lives, veg = await graph.list_statements()
    assert (sister.subject_id, sister.object_id) == (USER_ENTITY_ID, meera.id)
    assert (lives.subject_id, lives.object_id, lives.predicate) == (meera.id, leeds.id, "lives in")
    assert lives.valid_from == datetime(2026, 3, 1, tzinfo=UTC)
    assert lives.last_observed_at == DAY1 and lives.recorded_at == T0
    assert (veg.object_value, veg.origin, veg.confidence) == ("vegetarian", Origin.STATED, 0.9)
    assert sister.confidence == 0.9 and sister.sources == [user_ep]
    assert report.turns[0].unlinked == []


async def test_valid_until_becomes_invalid_at(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "I'm on leave until next Friday", ts=DAY1)
    script(llm, extraction(cand("The user is on leave", valid_until="2026-11-13")))
    await extractor.run(1)
    [statement] = await graph.list_statements()
    assert statement.invalid_at == datetime(2026, 11, 13, tzinfo=UTC)


async def test_instruction_is_active_every_turn(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "Don't read URLs aloud", ts=DAY1)
    script(llm, extraction(cand("Lucy should not read URLs aloud", kind="instruction")))
    await extractor.run(1)
    assert [s.text for s in await graph.active_instructions()] == [
        "Lucy should not read URLs aloud"
    ]


async def test_decision_structure(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "Kindle or Kobo? I went with the Kobo for library books", ts=DAY1)
    choice = ("E-reader choice", "decision")
    kw: dict[str, Any] = {"subject": choice, "kind": "decision"}
    script(
        llm,
        extraction(
            cand("The user considered the Kindle", predicate="option", value="Kindle", **kw),
            cand("The user considered the Kobo", predicate="option", value="Kobo", **kw),
            cand("The user chose the Kobo", predicate="chose", value="Kobo", **kw),
            cand("The Kobo lends library books", predicate="reason", value="library", **kw),
        ),
    )
    await extractor.run(1)
    [entity] = await graph.find_entities("e-reader choice")
    assert entity.type == "decision"
    statements = await graph.current_statements(entity.id)
    assert [(s.predicate, s.object_value) for s in statements] == [
        ("option", "Kindle"),
        ("option", "Kobo"),
        ("chose", "Kobo"),
        ("reason", "library"),
    ]
    assert {s.kind for s in statements} == {StatementKind.DECISION}
    assert llm.calls[1:] == []  # siblings from one turn are not reconciled against each other


# --- The user's name ---------------------------------------------------------------


async def test_the_user_entity_is_renamed(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "I'm Arjun", ts=DAY1)
    await turn(memory, "t2", "Actually, my name is Arjun Rao", ts=DAY1)
    await turn(memory, "t3", "Arjun Rao, as I said", ts=DAY1)
    script(
        llm,
        extraction(user_name="Arjun"),
        extraction(user_name="Arjun Rao"),
        extraction(user_name="arjun rao"),
    )
    report = await extractor.run(3)
    assert [t.renamed_user for t in report.turns] == ["Arjun", "Arjun Rao", None]
    user = await graph.user_entity()
    # The placeholder "User" is not kept as an alias; a real earlier name is.
    assert (user.name, user.aliases) == ("Arjun Rao", ["Arjun"])


async def test_a_name_statement_also_renames_the_user(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "My name's Rohan", ts=DAY1)
    script(llm, extraction(cand("The user is named Rohan", predicate="is named", value="Rohan")))
    report = await extractor.run(1)
    assert report.turns[0].renamed_user == "Rohan"
    assert (await graph.user_entity()).name == "Rohan"
    # The name lives on the user entity; no "Rohan is named Rohan" statement.
    assert await graph.list_statements() == []
    assert "the user's own name" in report.turns[0].dropped[0].reason


@pytest.mark.parametrize("name", ["user", "Lucy", "  ", "x" * 200])
async def test_implausible_user_names_are_ignored(
    memory: SQLiteMemoryStore,
    graph: SQLiteGraphStore,
    extractor: Extractor,
    llm: FakeLLM,
    name: str,
) -> None:
    await turn(memory, "t1", "hi", ts=DAY1)
    script(llm, extraction(user_name=name))
    await extractor.run(1)
    assert (await graph.user_entity()).name == "User"


async def test_an_instruction_to_use_the_users_own_name_is_not_stored(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "Hi Lucy, I'm Rohan. Call me Ro.", ts=DAY1)
    own = cand("Lucy should call the user Rohan", kind="instruction", value="Rohan")
    nickname = cand("Lucy should call the user Ro", kind="instruction", value="Ro")
    script(llm, extraction(own, nickname, user_name="Rohan"))
    report = await extractor.run(1)
    assert [s.text for s in await graph.list_statements()] == ["Lucy should call Rohan Ro"]
    assert [d.candidate.text for d in report.turns[0].dropped] == [own["text"]]


# --- Value checks ------------------------------------------------------------------


def test_parse_date_forms() -> None:
    today = date(2026, 11, 12)
    assert parse_date("2026-03-05", today=today) == date(2026, 3, 5)
    assert parse_date("2026-3", today=today) == date(2026, 3, 1)
    assert parse_date("2027", today=today) == date(2027, 1, 1)
    assert parse_date(None, today=today) is None
    assert parse_date(" null ", today=today) is None
    for bad in ["5941-59-59", "2026-02-30", "3000-01-01", "1066-10-14", "next Friday"]:
        with pytest.raises(ValueError):
            parse_date(bad, today=today)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"valid_from": "5941-59-59"}, "implausible date"),
        ({"valid_until": "2999-01-01"}, "implausible date"),
        ({"valid_from": "2026-05-01", "valid_until": "2026-04-01"}, "valid_until before"),
        ({"text": "   "}, "empty text"),
        ({"text": "word " * 100}, "text longer than"),
        ({"predicate": " "}, "empty predicate"),
        ({"subject": {"name": " ", "type": "person"}}, "subject cannot be resolved"),
        (
            {"object_entity": {"name": "Leeds", "type": "place"}, "object_value": "Leeds"},
            "both an object entity and an object value",
        ),
    ],
)
async def test_implausible_candidates_are_dropped_and_logged(
    memory: SQLiteMemoryStore,
    graph: SQLiteGraphStore,
    extractor: Extractor,
    llm: FakeLLM,
    caplog: pytest.LogCaptureFixture,
    overrides: dict[str, Any],
    reason: str,
) -> None:
    await turn(memory, "t1", "something", ts=DAY1)
    bad = cand("The user lives in Leeds") | overrides
    script(llm, extraction(bad, cand("The user plays the cello")))
    with caplog.at_level(logging.INFO, logger="lucy.memory.extraction"):
        report = await extractor.run(1)
    [dropped] = report.turns[0].dropped
    assert dropped.reason.startswith(reason)
    assert any("dropped candidate" in r.message and reason in r.message for r in caplog.records)
    assert [s.text for s in await graph.list_statements()] == ["The user plays the cello"]
    assert await unextracted(graph) == []


def test_check_candidate_accepts_plausible_values() -> None:
    candidate = ExtractedStatement.model_validate(
        cand("Meera lives in Leeds", entity=("Leeds", "place"), valid_from="2026-03-01")
    )
    assert check_candidate(candidate, today=date(2026, 11, 12)) is None
    blank_entity = ExtractedStatement.model_validate(
        cand("Meera is a vet", entity=("", "person"), value="vet")
    )
    assert check_candidate(blank_entity, today=date(2026, 11, 12)) is None


# --- Reconciliation through the extractor -------------------------------------------


async def test_a_contradiction_in_a_later_turn_supersedes(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "I drink dark roast", ts=DAY1)
    await turn(memory, "t2", "I've switched to decaf", ts=DAY1 + timedelta(days=1))
    script(
        llm,
        extraction(cand("The user drinks dark roast coffee", kind="preference")),
        extraction(cand("The user drinks decaf coffee", kind="preference")),
        decision("supersede", 1),
    )
    report = await extractor.run(2)
    assert report.turns[1].results[0].outcome is Outcome.SUPERSEDE
    old, current = await graph.list_statements(current_only=False)
    # It ended when the user said so, not when extraction ran (T0).
    switched = DAY1 + timedelta(days=1)
    assert old.superseded_by == current.id and old.invalid_at == switched
    assert current.valid_from == switched
    assert "[1] The user drinks dark roast coffee (preference, stated; current)" in prompt_of(
        llm, 2
    )


# --- Running twice -------------------------------------------------------------------


class Crash(Exception):
    pass


def crash_once(obj: Any, name: str, *, after: int = 0) -> None:
    """Make `obj.name` raise `Crash` on its call number `after` + 1, once."""
    original = getattr(obj, name)
    calls = {"n": 0}

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == after + 1:
            raise Crash(name)
        return await original(*args, **kwargs)

    setattr(obj, name, wrapper)


THREE = (
    cand("Meera is the user's sister", predicate="sister", entity=("Meera", "person")),
    cand("The user drinks dark roast coffee", kind="preference"),
    cand("The user plays the cello", kind="habit"),
)


@pytest.mark.parametrize(
    ("method", "after"),
    [
        ("create_entity", 0),  # before anything was stored
        ("add_statement", 0),  # after the entity was created
        ("add_statement", 1),  # after the first statement
        ("add_statement", 2),  # after two statements
        ("mark_extracted", 0),  # after everything but the marking
    ],
)
async def test_rerun_after_a_crash_creates_no_duplicates(
    memory: SQLiteMemoryStore,
    graph: SQLiteGraphStore,
    extractor: Extractor,
    llm: FakeLLM,
    method: str,
    after: int,
) -> None:
    await turn(memory, "t1", "facts", ts=DAY1)
    script(llm, extraction(*THREE), extraction(*THREE))
    crash_once(graph, method, after=after)
    with pytest.raises(Crash):
        await extractor.run(1)
    assert await unextracted(graph) != []
    await extractor.run(1)
    statements = await graph.list_statements(current_only=False)
    assert sorted(s.text for s in statements) == sorted(c["text"] for c in THREE)
    assert {s.evidence_count for s in statements} == {1}
    assert len(await graph.find_entities("Meera")) == 1
    assert await unextracted(graph) == []


async def test_rerun_after_a_crash_does_not_count_evidence_twice(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "I like jazz", ts=DAY1)
    await turn(memory, "t2", "Jazz is my favourite", ts=DAY1)
    await turn(memory, "t3", "Coffee: decaf now", ts=DAY1)
    jazz = cand("The user likes jazz music", kind="preference")
    later = cand("The user likes jazz music a lot", kind="preference")
    script(llm, extraction(jazz), extraction(later), decision("reinforce", 1))
    script(llm, extraction(later), decision("reinforce", 1))
    await extractor.run(1)
    crash_once(graph, "mark_extracted")
    with pytest.raises(Crash):
        await extractor.run(1)
    report = await extractor.run(1)
    assert report.turns[0].results[0].outcome is Outcome.DUPLICATE
    [statement] = await graph.list_statements()
    assert statement.evidence_count == 2 and statement.confidence == pytest.approx(0.925)


async def test_rerun_after_a_crash_does_not_supersede_twice(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "dark roast", ts=DAY1)
    await turn(memory, "t2", "decaf now", ts=DAY1)
    roast = cand("The user drinks dark roast coffee", kind="preference")
    decaf = cand("The user drinks decaf coffee", kind="preference")
    script(llm, extraction(roast), extraction(decaf), decision("supersede", 1))
    script(llm, extraction(decaf))
    await extractor.run(1)
    crash_once(graph, "mark_extracted")
    with pytest.raises(Crash):
        await extractor.run(1)
    report = await extractor.run(1)
    assert report.turns[0].results[0].outcome is Outcome.DUPLICATE
    statements = await graph.list_statements(current_only=False)
    assert [s.text for s in statements] == [roast["text"], decaf["text"]]
    assert statements[0].superseded_by == statements[1].id


async def test_rerun_after_a_rename_is_harmless(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "I'm Arjun and I play cello", ts=DAY1)
    out = extraction(cand("Arjun plays the cello"), user_name="Arjun")
    script(llm, out, out)
    crash_once(graph, "add_statement")
    with pytest.raises(Crash):
        await extractor.run(1)
    report = await extractor.run(1)
    assert report.turns[0].renamed_user is None
    user = await graph.user_entity()
    assert (user.name, user.aliases) == ("Arjun", [])
    assert len(await graph.list_statements()) == 1


# --- What extraction stores (rules enforced in code) --------------------------------


async def test_extraction_never_infers(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "Let's go with the vegetarian lasagne tonight.", ts=DAY1)
    script(
        llm,
        extraction(
            cand("The user prefers vegetarian food", kind="preference", origin="inferred"),
            cand("The user has a mother", origin="inferred"),
        ),
    )
    report = await extractor.run(1)
    assert await graph.list_statements() == []
    assert [d.reason for d in report.turns[0].dropped] == [
        "extraction stores only what the user said (origin inferred)"
    ] * 2


@pytest.mark.parametrize(
    "text",
    [
        "The user plans to cycle to work next week",
        "The user is going to repaint the kitchen",
        "The user intends to start running",
    ],
)
async def test_a_plan_is_not_a_goal(
    memory: SQLiteMemoryStore,
    graph: SQLiteGraphStore,
    extractor: Extractor,
    llm: FakeLLM,
    text: str,
) -> None:
    await turn(memory, "t1", "plans", ts=DAY1)
    script(llm, extraction(cand(text, kind="goal", predicate="wants to")))
    report = await extractor.run(1)
    assert await graph.list_statements() == []
    assert report.turns[0].dropped[0].reason == "a plan is not a goal"


async def test_a_goal_is_kept(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "goals", ts=DAY1)
    script(
        llm,
        extraction(
            cand("The user wants to get a pilot's licence by 2028", kind="goal"),
            cand("The user is working towards a 5k in under 20 minutes", kind="goal"),
        ),
    )
    await extractor.run(1)
    assert len(await graph.list_statements(kinds=[StatementKind.GOAL])) == 2


async def test_a_new_entity_nothing_links_to_the_user_is_logged(
    memory: SQLiteMemoryStore,
    graph: SQLiteGraphStore,
    extractor: Extractor,
    llm: FakeLLM,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The half-extracted tortoise: "Biscuit is about forty" and nothing else."""
    await turn(memory, "t1", "We have a tortoise called Biscuit, he's about forty", ts=DAY1)
    script(
        llm,
        extraction(
            cand(
                "Biscuit is about forty years old",
                subject=("Biscuit", "animal"),
                predicate="age",
                value="about forty",
            )
        ),
    )
    with caplog.at_level(logging.WARNING, logger="lucy.memory.extraction"):
        report = await extractor.run(1)
    assert report.turns[0].unlinked == ["Biscuit"]
    assert "'Biscuit' is new and nothing links it" in caplog.text
    # Logged, not acted on: the statement is kept (it is true).
    assert [s.text for s in await graph.list_statements()] == ["Biscuit is about forty years old"]


@pytest.mark.parametrize(
    "link",
    [
        # an edge from the user
        cand("Biscuit is the user's tortoise", predicate="pet", entity=("Biscuit", "animal")),
        # a statement about it whose text names the user
        cand("Biscuit is the user's tortoise", subject=("Biscuit", "animal"), value="tortoise"),
    ],
)
async def test_a_new_entity_linked_to_the_user_is_not_reported(
    memory: SQLiteMemoryStore,
    graph: SQLiteGraphStore,
    extractor: Extractor,
    llm: FakeLLM,
    link: dict[str, Any],
) -> None:
    await turn(memory, "t1", "We have a tortoise called Biscuit, he lives in Kew", ts=DAY1)
    lives = cand("Biscuit lives in Kew", subject=("Biscuit", "animal"), entity=("Kew", "place"))
    script(llm, extraction(link, lives))
    report = await extractor.run(1)
    # Kew is linked through Biscuit, which is linked to the user.
    assert report.turns[0].unlinked == []


async def test_new_entities_linked_only_to_each_other_are_reported(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "Biscuit lives in Kew", ts=DAY1)
    script(
        llm,
        extraction(
            cand("Biscuit lives in Kew", subject=("Biscuit", "animal"), entity=("Kew", "place"))
        ),
    )
    report = await extractor.run(1)
    assert sorted(report.turns[0].unlinked) == ["Biscuit", "Kew"]


async def test_a_change_of_state_replaces_the_old_state_with_the_new_one(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    """ "Tobias moved from Lisbon to Porto last month": the new state supersedes the old
    one, valid from the move; the change event, even listed first, does not."""
    await turn(memory, "t1", "My friend Tobias lives in Lisbon", ts=DAY1)
    await turn(
        memory, "t2", "Tobias moved from Lisbon to Porto last month", ts=DAY1 + timedelta(days=4)
    )
    tobias = ("Tobias", "person")
    script(
        llm,
        extraction(
            cand("Tobias is the user's friend", predicate="friend", entity=tobias),
            cand(
                "Tobias lives in Lisbon city",
                subject=tobias,
                predicate="lives in",
                entity=("Lisbon", "place"),
            ),
        ),
        extraction(
            cand(
                "Tobias moved from Lisbon to Porto city",
                subject=tobias,
                predicate="moved from",
                entity=("Lisbon", "place"),
                valid_from="2026-10",
            ),
            cand(
                "Tobias moved to Porto city",
                subject=tobias,
                predicate="moved to",
                entity=("Porto", "place"),
                valid_from="2026-10",
            ),
            cand(
                "Tobias lives in Porto city",
                subject=tobias,
                predicate="lives in",
                entity=("Porto", "place"),
                valid_from="2026-10",
            ),
        ),
        decision("supersede", 1),  # "lives in Porto" replaces "lives in Lisbon"
        decision("supersede", 1),  # "moved from Lisbon" also wants to replace it
    )
    report = await extractor.run(2)
    second = report.turns[1]
    assert [c.predicate for c in second.reconciled] == ["lives in", "moved from"]
    assert [d.reason for d in second.dropped] == [
        "a change event; the new state is stored as 'Tobias lives in Porto city'"
    ]
    assert second.results[0].outcome is Outcome.SUPERSEDE
    assert "already replaced from this turn" in second.results[1].reason
    lisbon = next(
        s
        for s in await graph.list_statements(current_only=False)
        if "Lisbon city" in s.text and s.predicate == "lives in"
    )
    porto = second.results[0].statement
    assert porto is not None and porto.text == "Tobias lives in Porto city"
    assert lisbon.superseded_by == porto.id
    assert lisbon.invalid_at == porto.valid_from == datetime(2026, 10, 1, tzinfo=UTC)


async def test_giving_up_a_goal_closes_it_and_the_habit_when_the_user_said_so(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "I want to learn Italian, I practise every evening", ts=DAY1)
    said = DAY1 + timedelta(days=6)
    await turn(memory, "t2", "I've given up on Italian", ts=said)
    script(
        llm,
        extraction(
            cand("The user wants to learn Italian", kind="goal"),
            cand("The user practises to learn Italian", kind="habit"),
        ),
        # The model calls it a decision; a goal is closed by a fact.
        extraction(
            cand(
                "The user gave up wanting to learn Italian",
                kind="decision",
                predicate="gave up",
                value="learning Italian",
            )
        ),
        decision("supersede", 1, [2]),
    )
    report = await extractor.run(2)
    [result] = report.turns[1].results
    assert result.outcome is Outcome.SUPERSEDE and result.statement is not None
    assert result.statement.kind is StatementKind.FACT
    goal, habit = await graph.list_statements(
        current_only=False, kinds=[StatementKind.GOAL, StatementKind.HABIT]
    )
    for ended in (goal, habit):
        # Ended at the message's time (not T0, when extraction ran), pointing at the fact.
        assert ended.invalid_at == said and ended.superseded_by == result.statement.id
    assert [s.text for s in await graph.list_statements()] == [
        "The user gave up wanting to learn Italian"
    ]


async def test_the_user_texts_are_reworded_once_the_name_is_known(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "facts", ts=DAY1)
    await turn(memory, "t2", "I'm Mira, and I have a dog", ts=DAY1)
    script(
        llm,
        extraction(
            cand("The user plays the cello", kind="habit"),
            cand(
                "Tobias is the user's best friend", predicate="friend", entity=("Tobias", "person")
            ),
            cand("The user lives in Leeds city"),
        ),
        extraction(cand("The user has a dog"), user_name="Mira"),
    )
    await extractor.run(1)
    [leeds] = [s for s in await graph.list_statements() if "Leeds" in s.text]
    await graph.invalidate(leeds.id, DAY1)  # history is reworded too
    report = await extractor.run(1)
    assert report.turns[0].renamed_user == "Mira"
    assert report.turns[0].rewritten == 3
    texts = sorted(s.text for s in await graph.list_statements(current_only=False))
    assert texts == [
        "Mira has a dog",  # this turn's candidate, already with the name
        "Mira lives in Leeds city",
        "Mira plays the cello",
        "Tobias is Mira's best friend",
    ]
    # Re-embedded and re-indexed: found by the new wording.
    [hit, *_] = await graph.similar_statements("Mira plays the cello", subject_id=USER_ENTITY_ID)
    assert hit.statement.text == "Mira plays the cello" and hit.score > 0.99
    [kw] = await graph.keyword_statements("cello")
    assert kw.statement.text == "Mira plays the cello"
    assert await graph.keyword_statements("user") == []
    # Idempotent.
    assert await extractor.rewrite_user_texts("Mira") == 0


async def test_a_rerun_after_a_crash_in_other_words_creates_no_duplicate(
    memory: SQLiteMemoryStore, graph: SQLiteGraphStore, extractor: Extractor, llm: FakeLLM
) -> None:
    await turn(memory, "t1", "My sister Meera lives in Leeds", ts=DAY1)
    meera = ("Meera", "person")
    first = extraction(
        cand("Meera is the user's sister", predicate="sister", entity=meera),
        cand(
            "Meera lives in Leeds city",
            subject=meera,
            predicate="lives in",
            entity=("Leeds", "place"),
        ),
    )
    reworded = extraction(
        cand("Meera is the sister of the user", predicate="sister", entity=meera),
        cand(
            "Meera's home is in Leeds city",
            subject=meera,
            predicate="lives in",
            entity=("Leeds", "place"),
        ),
    )
    script(llm, first, reworded)
    crash_once(graph, "mark_extracted")
    with pytest.raises(Crash):
        await extractor.run(1)
    report = await extractor.run(1)
    assert [r.outcome for r in report.turns[0].results] == [Outcome.DUPLICATE] * 2
    assert len(await graph.list_statements(current_only=False)) == 2
    assert len(llm.calls) == 2  # no reconciliation call
