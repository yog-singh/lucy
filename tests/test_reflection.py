"""Reflection: pattern inference, goal and decision review, the job (memory/reflection.py)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from lucy.core.events import AssistantMessage, UserMessage
from lucy.core.gateway import LLMGateway, Preempted
from lucy.core.jobs import Job, JobRunner, JobStatus
from lucy.core.types import LLMChunk
from lucy.memory.graph_store import SQLiteGraphStore
from lucy.memory.maintenance import (
    DECAY_JOB,
    MERGE_JOB,
    PROFILE_JOB,
    SUMMARIES_JOB,
    schedule_maintenance,
)
from lucy.memory.models import USER_ENTITY_ID, Origin, Statement, StatementKind
from lucy.memory.policy import MemoryPolicy
from lucy.memory.reconcile import Outcome, Reconciler
from lucy.memory.reflection import (
    PATTERNS_CURSOR_KEY,
    PROGRESS_LAST_RUN_KEY,
    REFLECT_JOB,
    InferredPattern,
    ReflectionError,
    ReflectionPolicy,
    Reflector,
    check_pattern,
    progress_key,
    reflection_handler,
    sample_evenly,
)
from lucy.memory.sqlite_jobs import SQLiteJobStore
from lucy.memory.sqlite_store import SQLiteMemoryStore
from lucy.prompts import load_prompt
from tests.fakes import FakeLLM, Raise
from tests.graph_support import T0, Clock, HashingEmbedder, log, new
from tests.memory_extraction_support import TEST_POLICY_VALUES, decision, reply

POLICY = MemoryPolicy(**TEST_POLICY_VALUES)
DAY = timedelta(days=1)
#: T0 is Thursday 2026-11-12 09:00 UTC; this is Monday 2026-10-12 07:30 UTC.
MONDAY = datetime(2026, 10, 12, 7, 30, tzinfo=UTC)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "lucy.db"


@pytest.fixture
async def graph(path: Path, clock: Clock) -> AsyncIterator[SQLiteGraphStore]:
    async with SQLiteGraphStore(path, HashingEmbedder(), clock=clock) as g:
        yield g


@pytest.fixture
async def memory(path: Path) -> AsyncIterator[SQLiteMemoryStore]:
    async with SQLiteMemoryStore(path) as m:
        yield m


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM([])


def make_reflector(graph: SQLiteGraphStore, llm: FakeLLM, clock: Clock, **policy: Any) -> Reflector:
    gateway = LLMGateway({"extraction": llm})
    reconciler = Reconciler(graph, gateway, policy=POLICY, clock=clock)
    return Reflector(graph, gateway, reconciler, policy=ReflectionPolicy(**policy), clock=clock)


@pytest.fixture
def reflector(graph: SQLiteGraphStore, llm: FakeLLM, clock: Clock) -> Reflector:
    return make_reflector(graph, llm, clock)


def script(llm: FakeLLM, *responses: list[LLMChunk]) -> None:
    llm._responses.extend(list(r) for r in responses)


def prompt_of(llm: FakeLLM, call: int = 0) -> str:
    return llm.calls[call][-1].content


_counter = iter(range(10**6))


async def say(memory: SQLiteMemoryStore, text: str, ts: datetime) -> int:
    """Log a user message at `ts`; returns its episode id."""
    turn = f"t{next(_counter)}"
    return await log(memory, UserMessage(text=text, session_id="s1", turn_id=turn, ts=ts))


async def chatter(memory: SQLiteMemoryStore, n: int, start: datetime = MONDAY) -> list[int]:
    """`n` ordinary messages, one every five hours from `start`."""
    return [
        await say(memory, f"question number {i} about the weather", start + timedelta(hours=5 * i))
        for i in range(n)
    ]


def pattern(
    evidence: Sequence[int],
    text: str = "Rohan usually goes to the gym on Monday mornings.",
    kind: str = "habit",
    predicate: str = "goes to the gym",
    value: str | None = "Monday mornings",
) -> dict[str, Any]:
    return {
        "evidence": list(evidence),
        "kind": kind,
        "text": text,
        "predicate": predicate,
        "object_value": value,
    }


def patterns(*items: dict[str, Any], known: dict[int, list[int]] | None = None) -> list[LLMChunk]:
    """A scripted pattern answer; `known` maps a guess's K number to its evidence."""
    data: dict[str, Any] = {"patterns": list(items)}
    if known is not None:
        data = {"known": {f"K{k}": {"evidence": v} for k, v in known.items()}} | data
    return reply(data)


def progress(
    goals: Sequence[dict[str, Any]] = (), decisions: Sequence[dict[str, Any]] = ()
) -> list[LLMChunk]:
    """A scripted review answer: one field per goal ("G1"...) and decision ("D1"...)."""
    data = {f"G{g['goal']}": {k: v for k, v in g.items() if k != "goal"} for g in goals}
    data |= {
        f"D{d['decision']}": {k: v for k, v in d.items() if k != "decision"} for d in decisions
    }
    return reply(data)


def goal_review(
    goal: int,
    status: str,
    evidence: Sequence[int] = (),
    *,
    outright: bool = True,
    text: str | None = None,
) -> dict[str, Any]:
    return {
        "goal": goal,
        "evidence": list(evidence),
        "status": status,
        "said_outright": outright,
        "text": text,
    }


async def gym_week(memory: SQLiteMemoryStore) -> list[int]:
    """Three Monday-morning gym messages (numbered 1..3 when alone), then 9 others."""
    ids = [
        await say(memory, "back from the gym, what's on today", MONDAY + i * 7 * DAY)
        for i in range(3)
    ]
    await chatter(memory, 9, MONDAY + 15 * DAY)
    return ids


async def inferred(graph: SQLiteGraphStore) -> list[Statement]:
    return await graph.list_statements(subject_id=USER_ENTITY_ID, origin=Origin.INFERRED)


# --- Prompts and pure helpers -----------------------------------------------------


def test_prompts_load_and_make_empty_the_normal_case() -> None:
    patterns_prompt = load_prompt("reflection_patterns")
    assert '{"patterns": []}' in patterns_prompt
    assert "wish, plan or intention" in patterns_prompt
    progress_prompt = load_prompt("reflection_progress")
    assert '"status": "open"' in progress_prompt


def test_sample_evenly_keeps_everything_that_fits_and_spreads_the_rest() -> None:
    assert sample_evenly([10] * 5, 50) == [0, 1, 2, 3, 4]
    kept = sample_evenly([10] * 100, 205)
    assert len(kept) == 20 and kept[0] == 0 and kept[-1] == 99
    gaps = {b - a for a, b in zip(kept, kept[1:], strict=False)}
    assert gaps <= {5, 6}
    assert sample_evenly([10] * 10, 5) == []
    assert sample_evenly([], 5) == []


def test_policy_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        ReflectionPolicy(min_new_messages=0)
    with pytest.raises(ValueError):
        ReflectionPolicy(lookback=timedelta(0))
    with pytest.raises(ValueError):
        ReflectionPolicy(max_inferences=-1)


# --- Store helpers --------------------------------------------------------------


async def test_episodes_between_selects_by_event_time_type_and_id(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore
) -> None:
    exact = await say(memory, "at the start", MONDAY)  # stored without microseconds
    later = await say(memory, "a bit later", MONDAY + timedelta(seconds=1, microseconds=5))
    early = await say(memory, "logged late but earlier", MONDAY - timedelta(microseconds=1))
    await log(memory, AssistantMessage(text="reply", session_id="s1", ts=MONDAY))
    end = await say(memory, "at the end", MONDAY + DAY)
    found = await graph.episodes_between(MONDAY, MONDAY + DAY)
    assert [r.id for r in found] == [exact, later]  # [start, end), user messages only
    everything = await graph.episodes_between(MONDAY - DAY, MONDAY + 2 * DAY, types=None)
    assert [r.id for r in everything][:2] == [early, exact] and len(everything) == 5
    assert [
        r.id for r in await graph.episodes_between(MONDAY, MONDAY + 2 * DAY, after_id=exact)
    ] == [
        later,
        end,
    ]
    assert len(await graph.episodes_between(MONDAY, MONDAY + 2 * DAY, limit=1)) == 1
    with pytest.raises(ValueError):
        await graph.episodes_between(datetime(2026, 1, 1), MONDAY)  # noqa: DTZ001


async def test_count_episodes_after_and_statements_recorded_since(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, clock: Clock
) -> None:
    first = await say(memory, "one", MONDAY)
    await say(memory, "two", MONDAY)
    await log(memory, AssistantMessage(text="reply", session_id="s1", ts=MONDAY))
    assert await graph.count_episodes_after(0) == 2
    assert await graph.count_episodes_after(first) == 1
    assert await graph.count_episodes_after(0, types=None) == 3
    old = await graph.add_statement(new("Rohan likes tea"))
    clock.advance(hours=1)
    goal = await graph.add_statement(new("Rohan wants a boat", kind=StatementKind.GOAL))
    other = await graph.create_entity("person", "Asha")
    await graph.add_statement(new("Asha sings", subject_id=other.id))
    since = await graph.statements_recorded_since(T0 + timedelta(minutes=30))
    assert len(since) == 2 and goal.id in {s.id for s in since}
    assert [s.id for s in await graph.statements_recorded_since(T0, subject_id=1)] == [
        goal.id,
        old.id,
    ]
    assert await graph.statements_recorded_since(T0, kinds=[StatementKind.HABIT]) == []
    assert len(await graph.statements_recorded_since(T0, limit=1)) == 1


# --- Pattern inference: input ---------------------------------------------------


async def test_run_condition_needs_enough_new_messages(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM
) -> None:
    await chatter(memory, 9)
    result = await reflector.infer_patterns()
    assert result.skipped and result.new_messages == 9 and llm.calls == []
    await chatter(memory, 1, MONDAY + 20 * DAY)
    script(llm, patterns())
    result = await reflector.infer_patterns()
    assert result.skipped is None and len(llm.calls) == 1
    assert await graph.get_meta(PATTERNS_CURSOR_KEY) is not None
    # Counted from the last successful run.
    await chatter(memory, 9, MONDAY + 25 * DAY)
    assert (await reflector.infer_patterns()).skipped and len(llm.calls) == 1
    # A graph rebuild resets the counter (a derived key).
    await graph.reset_graph()
    script(llm, patterns())
    assert (await reflector.infer_patterns()).skipped is None and len(llm.calls) == 2


async def test_window_numbering_weekdays_and_known_patterns_are_shown(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, llm: FakeLLM, clock: Clock
) -> None:
    reflector = make_reflector(graph, llm, clock, timezone=ZoneInfo("Asia/Kolkata"))
    await graph.rename_entity(USER_ENTITY_ID, "Rohan")
    await say(memory, "too old to matter", T0 - 61 * DAY)
    late = await say(memory, "logged second, said first", MONDAY - DAY)
    await chatter(memory, 10)
    await say(memory, "from the future", T0 + DAY)
    await log(memory, AssistantMessage(text="Lucy's words", session_id="s1", ts=MONDAY))
    await graph.add_statement(
        new("Rohan swims every Tuesday", kind=StatementKind.HABIT, predicate="swims")
    )
    await graph.add_statement(
        new(
            "Rohan usually orders noodles",
            kind=StatementKind.PREFERENCE,
            origin=Origin.INFERRED,
            confidence=0.4,
        )
    )
    script(llm, patterns(known={1: []}))
    result = await reflector.infer_patterns()
    assert result.window_messages == 11 and result.shown_messages == 11 and not result.sampled
    prompt = prompt_of(llm)
    assert "Today: Thu 2026-11-12" in prompt and "The user's name: Rohan" in prompt
    # Grouped by weekday in the user's time zone (07:30 UTC is 13:00 in India),
    # Monday first, oldest first within a weekday, numbered from 1.
    assert "[1] Mon 2026-10-12 13:00: question number 0 about the weather" in prompt
    assert "[2] Mon 2026-10-12 18:00: question number 1 about the weather" in prompt
    assert "[11] Sun 2026-10-11 13:00: logged second, said first" in prompt
    assert "[12] " not in prompt
    assert "grouped by weekday" in prompt
    for absent in ("too old", "future", "Lucy's words"):
        assert absent not in prompt
    assert "told Lucy:\n- Rohan swims every Tuesday (habit)\n" in prompt
    assert "against the messages:\n[K1] Rohan usually orders noodles (preference)\n" in prompt
    assert llm.schemas_seen[0] is not None and llm.schemas_seen[0]["required"] == [
        "known",
        "patterns",
    ]
    assert late in {r.id for r in await graph.episodes_between(MONDAY - 2 * DAY, T0)}


async def test_over_budget_messages_are_sampled_across_the_whole_window(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, llm: FakeLLM, clock: Clock
) -> None:
    reflector = make_reflector(graph, llm, clock, pattern_input_tokens=200)
    await chatter(memory, 60, T0 - 50 * DAY)  # ~13 tokens each: about 15 fit
    script(llm, patterns())
    result = await reflector.infer_patterns()
    assert result.sampled and 10 <= result.shown_messages < 20
    prompt = prompt_of(llm)
    assert "question number 0 about" in prompt and "question number 59 about" in prompt
    assert f"{result.shown_messages} of them are shown, spread evenly" in prompt
    numbers = [line.split("]")[0] for line in prompt.splitlines() if line.startswith("[")]
    assert numbers == [f"[{i}" for i in range(1, result.shown_messages + 1)]


# --- Pattern inference: evidence rules ---------------------------------------------


def test_check_pattern_rules() -> None:
    policy = ReflectionPolicy()

    def msg(day: int, hour: int = 8) -> Any:
        from lucy.memory.models import EpisodeRecord

        ts = MONDAY + day * DAY + timedelta(hours=hour)
        return EpisodeRecord(
            id=day * 100 + hour, event=UserMessage(text="x", session_id="s", ts=ts)
        )

    shown = [msg(0), msg(0, 12), msg(7), msg(14), msg(0, 10)]

    def check(**fields: Any) -> str | None:
        return check_pattern(InferredPattern(**(pattern([1, 3, 4]) | fields)), shown, policy)

    assert check() is None
    assert "not shown" in (check(evidence=[1, 3, 6]) or "")
    assert "not shown" in (check(evidence=[0, 1, 3]) or "")
    assert "no supporting" in (check(evidence=[]) or "")
    assert "3 supporting messages, 2 cited" in (check(evidence=[1, 3, 3]) or "")
    assert "2 different days, got 1" in (check(evidence=[1, 2, 2, 5]) or "")
    assert check(evidence=[1, 2], kind="preference") is None
    assert "2 supporting" in (check(evidence=[4], kind="preference") or "")
    assert "third person" in (check(text="I go to the gym on Mondays.") or "")
    assert "third person" in (check(text="Rohan wants you to remind him.") or "")
    assert "empty text" in (check(text="  ") or "")
    assert "longer than" in (check(text="Rohan " + "x" * 400) or "")
    assert "empty predicate" in (check(predicate=" ") or "")
    assert "object value" in (check(object_value="x" * 200) or "")


async def test_habit_on_one_day_is_dropped(
    memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM, graph: SQLiteGraphStore
) -> None:
    for hour in (7, 12, 18):
        await say(memory, "at the gym again", MONDAY + timedelta(hours=hour - 7))
    await chatter(memory, 9, MONDAY + DAY)
    script(llm, patterns(pattern([1, 2, 3])))
    result = await reflector.infer_patterns()
    assert [d.reason for d in result.dropped] == [
        "a habit needs messages on 2 different days, got 1"
    ]
    assert await inferred(graph) == []


async def test_evidence_rules_drop_bad_inferences_and_log_them(
    memory: SQLiteMemoryStore,
    reflector: Reflector,
    llm: FakeLLM,
    graph: SQLiteGraphStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    await gym_week(memory)  # 12 messages: the gym ones are [1], [2] and [3]
    script(
        llm,
        patterns(
            pattern([1, 2, 99]),
            pattern([1, 2]),
            pattern([4], kind="preference", text="Rohan prefers rain.", value="rain"),
            pattern([1, 2, 3], text="I go to the gym on Mondays."),
        ),
    )
    caplog.set_level("INFO", logger="lucy.memory.reflection")
    result = await reflector.infer_patterns()
    reasons = [d.reason for d in result.dropped]
    assert reasons == [
        "cites message(s) [99] that were not shown",
        "a habit needs 3 supporting messages, 2 cited",
        "a preference needs 2 supporting messages, 1 cited",
        "text is not in the third person",
    ]
    assert result.results == [] and await inferred(graph) == []
    assert "not shown" in caplog.text
    # The run still counts as done.
    assert await graph.get_meta(PATTERNS_CURSOR_KEY) is not None


async def test_a_pattern_the_user_stated_is_not_inferred(
    memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM, graph: SQLiteGraphStore
) -> None:
    ids = await gym_week(memory)
    await graph.add_statement(
        new(
            "Rohan goes to the gym on Monday mornings",
            kind=StatementKind.HABIT,
            predicate="goes to the gym",
        ),
        [ids[0]],
    )
    script(llm, patterns(pattern([1, 2, 3])))
    result = await reflector.infer_patterns()
    assert len(result.dropped) == 1 and "stated it outright" in result.dropped[0].reason
    assert await inferred(graph) == []


async def test_repeats_and_the_per_run_limit(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, llm: FakeLLM, clock: Clock
) -> None:
    reflector = make_reflector(graph, llm, clock, max_inferences=1)
    await gym_week(memory)
    script(
        llm,
        patterns(
            pattern([1, 2, 3]),
            pattern([1, 2, 3], text="Rohan usually goes to the gym on Monday mornings."),
            pattern([4, 5], kind="preference", text="Rohan prefers sunny weather."),
        ),
    )
    result = await reflector.infer_patterns()
    assert [d.reason for d in result.dropped] == [
        "repeats an earlier inference in the same answer",
        "over the limit of 1 inferences per run",
    ]
    assert len(await inferred(graph)) == 1


# --- Pattern inference: writing through the reconciler ---------------------------


async def test_new_inference_is_added_as_an_inferred_statement(
    memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM, graph: SQLiteGraphStore
) -> None:
    ids = await gym_week(memory)
    script(llm, patterns(pattern([1, 2, 3])))
    result = await reflector.infer_patterns()
    [(_, outcome)] = result.results
    assert outcome.outcome is Outcome.ADD
    [statement] = await inferred(graph)
    assert statement.kind is StatementKind.HABIT and statement.origin is Origin.INFERRED
    assert statement.confidence == POLICY.inferred_confidence
    assert statement.sources == ids
    assert statement.last_observed_at == MONDAY + 14 * DAY
    assert statement.object_value == "Monday mornings"


async def test_reinforcement_with_new_and_partly_overlapping_evidence(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, llm: FakeLLM, clock: Clock
) -> None:
    reflector = make_reflector(graph, llm, clock)
    ids = await gym_week(memory)
    script(llm, patterns(pattern([1, 2, 3])))
    await reflector.infer_patterns()
    [first] = await inferred(graph)

    # Same evidence again (forced re-run), for the guess and as a reworded new
    # pattern: nothing new, nothing changes, no reconciliation call.
    await graph.delete_meta(PATTERNS_CURSOR_KEY)
    reworded = pattern([1, 2, 3], text="Rohan tends to go to the gym on Mondays.")
    script(llm, patterns(reworded, known={1: [1, 2, 3]}))
    result = await reflector.infer_patterns()
    assert [d.reason for d in result.dropped] == [f"no new evidence for statement {first.id}"] * 2
    assert len(llm.calls) == 2

    # The guess with every earlier message plus a new one (a superset): reinforced
    # once, with the new one.
    clock.advance(days=7)
    fourth = await say(memory, "back from the gym, what's on today", MONDAY + 21 * DAY)
    await chatter(memory, 9, MONDAY + 22 * DAY)
    script(llm, patterns(known={1: [1, 2, 3, 4]}), decision("reinforce", 1))  # all Mondays
    result = await reflector.infer_patterns()
    [(_, outcome)] = result.results
    assert outcome.outcome is Outcome.REINFORCE and outcome.target_id == first.id
    assert "New statement: Rohan usually goes" in prompt_of(llm, 3)  # the reconciler was asked
    after = await graph.get_statement(first.id)
    assert after is not None and after.evidence_count == 2
    assert after.confidence == pytest.approx(POLICY.reinforced(0.4, Origin.INFERRED))
    assert after.sources == [*ids, fourth]
    assert after.last_observed_at == MONDAY + 21 * DAY
    assert len(await inferred(graph)) == 1

    # Partly overlapping (two old, two new): the new ones count once.
    clock.advance(days=14)
    fifth = await say(memory, "gym done", MONDAY + 28 * DAY)
    sixth = await say(memory, "gym done again", MONDAY + 35 * DAY)
    await chatter(memory, 8, MONDAY + 29 * DAY)
    window = await graph.episodes_between(clock.now - 60 * DAY, clock.now)
    window.sort(key=lambda r: (r.event.ts.weekday(), r.event.ts))  # as shown
    number = {r.id: i for i, r in enumerate(window, start=1)}
    cited = [number[ids[2]], number[fourth], number[fifth], number[sixth]]
    # Reported as a new pattern in other words: same treatment.
    script(
        llm,
        patterns(pattern(cited, text="Rohan goes to the gym most Mondays."), known={1: []}),
        decision("reinforce", 1),
    )
    result = await reflector.infer_patterns()
    [(_, outcome)] = result.results
    assert outcome.outcome is Outcome.REINFORCE
    after = await graph.get_statement(first.id)
    assert after is not None and after.evidence_count == 3
    assert after.sources == [*ids, fourth, fifth, sixth]
    assert after.confidence == pytest.approx(
        POLICY.reinforced(POLICY.reinforced(0.4, Origin.INFERRED), Origin.INFERRED)
    )


async def test_an_inference_never_supersedes_a_stated_statement(
    memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM, graph: SQLiteGraphStore
) -> None:
    await gym_week(memory)
    other = await say(memory, "I go to the gym on Monday mornings", MONDAY - 30 * DAY)
    stated = await graph.add_statement(
        new("Rohan goes to the gym on Monday mornings", kind=StatementKind.HABIT), [other]
    )
    text = "Rohan usually goes to the gym on Monday Wednesday and Friday mornings."
    script(llm, patterns(pattern([2, 3, 4], text=text)), decision("supersede", 1))
    result = await reflector.infer_patterns()
    [(_, outcome)] = result.results
    assert outcome.outcome is Outcome.IGNORE and outcome.target_id == stated.id
    assert await inferred(graph) == []
    kept = await graph.get_statement(stated.id)
    assert kept is not None and kept.invalid_at is None


async def test_preempted_pattern_run_propagates_and_is_repeated(
    memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM, graph: SQLiteGraphStore
) -> None:
    await gym_week(memory)
    script(llm, [Raise(Preempted("foreground"))])
    with pytest.raises(Preempted):
        await reflector.infer_patterns()
    assert await graph.get_meta(PATTERNS_CURSOR_KEY) is None
    script(llm, patterns(pattern([1, 2, 3])))
    assert len((await reflector.infer_patterns()).results) == 1


# --- Goals and decisions ----------------------------------------------------------


async def add_goal(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, text: str, ts: datetime
) -> Statement:
    source = await say(memory, f"I want to {text}", ts)
    return await graph.add_statement(
        new(f"Rohan wants to {text}", kind=StatementKind.GOAL, observed_at=ts), [source]
    )


async def add_decision(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, ts: datetime
) -> tuple[int, int]:
    source = await say(memory, "went with the ThinkPad over the Mac", ts)
    entity = await graph.create_entity("decision", "Laptop choice")
    for text in ("Rohan considered a Mac.", "Rohan chose the ThinkPad."):
        await graph.add_statement(
            new(text, subject_id=entity.id, kind=StatementKind.DECISION, observed_at=ts), [source]
        )
    return entity.id, source


async def test_goal_achieved_is_closed_with_an_outcome(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM
) -> None:
    goal = await add_goal(graph, memory, "run a half marathon", MONDAY)
    await say(memory, "did a long run today", MONDAY + 5 * DAY)
    done = await say(memory, "I finished the half marathon!", MONDAY + 12 * DAY)
    script(
        llm,
        progress([goal_review(1, "achieved", [2], text="Rohan ran a half marathon.")]),
        decision("add"),
    )
    result = await reflector.review_progress()
    prompt = prompt_of(llm)
    assert "[G1] Rohan wants to run a half marathon (noted 2026-10-12)" in prompt
    assert "[1] Sat 2026-10-17 07:30: did a long run today" in prompt
    assert "I want to run" not in prompt  # the message the goal came from is not shown again
    [change] = result.changes
    assert change.status == "achieved" and change.evidence == [done]
    closed = await graph.get_statement(goal.id)
    assert closed is not None and closed.invalid_at == MONDAY + 12 * DAY
    assert change.statement is not None
    assert closed.superseded_by == change.statement.id  # why it ended
    outcome = await graph.get_statement(change.statement.id)
    assert outcome is not None and outcome.kind is StatementKind.FACT
    assert outcome.origin is Origin.STATED and outcome.confidence == POLICY.stated_confidence
    assert outcome.sources == [done] and outcome.valid_from == MONDAY + 12 * DAY
    assert outcome.text == "Rohan ran a half marathon."
    # Closed: no longer reviewed, no model call.
    await say(memory, "another message", MONDAY + 13 * DAY)
    assert (await reflector.review_progress()).skipped and len(llm.calls) == 2
    assert await graph.get_meta(progress_key("goal", goal.id)) is None


async def test_reconciler_superseding_the_goal_closes_it_once(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM
) -> None:
    await graph.rename_entity(USER_ENTITY_ID, "Rohan")
    goal = await add_goal(graph, memory, "run a half marathon", MONDAY)
    await say(memory, "I ran the half marathon on Sunday", MONDAY + 12 * DAY)
    script(llm, progress([goal_review(1, "achieved", [1])]), decision("supersede", 1))
    result = await reflector.review_progress()
    closed = await graph.get_statement(goal.id)
    assert closed is not None and closed.invalid_at == MONDAY + 12 * DAY
    assert result.changes[0].statement is not None
    assert closed.superseded_by == result.changes[0].statement.id
    assert (
        result.changes[0].statement.text
        == "Rohan achieved the goal: Rohan wants to run a half marathon."
    )


async def test_goal_dropped_is_inferred_when_not_said_outright(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM
) -> None:
    goal = await add_goal(graph, memory, "learn Spanish", MONDAY)
    await say(memory, "no more lessons for me, too busy", MONDAY + 3 * DAY)
    script(
        llm,
        progress(
            [goal_review(1, "dropped", [1], outright=False, text="Rohan gave up learning Spanish.")]
        ),
    )
    result = await reflector.review_progress()
    [change] = result.changes
    assert change.status == "dropped" and change.statement is not None
    assert change.statement.origin is Origin.INFERRED
    assert change.statement.confidence == POLICY.inferred_confidence
    closed = await graph.get_statement(goal.id)
    assert closed is not None and closed.invalid_at == MONDAY + 3 * DAY


async def test_a_goal_extraction_already_closed_is_not_reviewed_again(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM
) -> None:
    """Extraction closed the goal ("I've given up on Spanish"): no second outcome."""
    goal = await add_goal(graph, memory, "learn Spanish", MONDAY)
    said = MONDAY + 3 * DAY
    source = await say(memory, "I've given up on Spanish", said)
    await graph.supersede(
        goal.id,
        new("Rohan gave up learning Spanish", observed_at=said, valid_from=said),
        [source],
    )
    result = await reflector.review_progress()
    assert result.skipped == "no open goals or decisions" and llm.calls == []
    facts = await graph.list_statements(kinds=[StatementKind.FACT])
    assert [s.text for s in facts] == ["Rohan gave up learning Spanish"]


async def test_open_goals_and_uncited_answers_change_nothing(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM
) -> None:
    open_goal = await add_goal(graph, memory, "read 20 books", MONDAY)
    other = await add_goal(graph, memory, "learn Spanish", MONDAY)
    third = await add_goal(graph, memory, "build a shed", MONDAY)
    newest = await say(memory, "what should I read next?", MONDAY + DAY)
    script(
        llm,
        progress(
            [
                goal_review(1, "open"),
                goal_review(2, "achieved", []),
                goal_review(3, "dropped", [7]),
                goal_review(4, "achieved", [1]),  # not asked for: ignored
            ]
        ),
    )
    result = await reflector.review_progress()
    assert result.changes == []
    assert len(result.dropped) == 2
    assert all("without valid cited" in d for d in result.dropped)
    assert llm.schemas_seen[0] is not None
    assert llm.schemas_seen[0]["required"] == ["G1", "G2", "G3"]
    for goal in (open_goal, other, third):
        kept = await graph.get_statement(goal.id)
        assert kept is not None and kept.invalid_at is None
        assert await graph.get_meta(progress_key("goal", goal.id)) == str(newest)
    assert await graph.get_meta(PROGRESS_LAST_RUN_KEY) == T0.isoformat()
    # Nothing new since: no model call.
    assert (await reflector.review_progress()).skipped and len(llm.calls) == 1


async def test_decision_outcome_is_recorded_on_the_decision(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM
) -> None:
    entity_id, _ = await add_decision(graph, memory, MONDAY)
    await say(memory, "what's the weather", MONDAY + DAY)
    happy = await say(memory, "the ThinkPad has been great, no regrets", MONDAY + 20 * DAY)
    review = {
        "decision": 1,
        "evidence": [2],
        "outcome": "happy with it",
        "said_outright": True,
        "text": "Rohan is happy with the ThinkPad he chose.",
    }
    script(llm, progress(decisions=[review]))
    result = await reflector.review_progress()
    prompt = prompt_of(llm)
    assert (
        "[D1] Laptop choice (decided 2026-10-12): Rohan considered a Mac. Rohan chose the ThinkPad."
        in prompt
    )
    [change] = result.changes
    assert change.item == "decision" and change.item_id == entity_id
    statement = await graph.get_statement(change.statement.id if change.statement else 0)
    assert statement is not None and statement.subject_id == entity_id
    assert statement.predicate == "outcome" and statement.object_value == "happy with it"
    assert statement.kind is StatementKind.DECISION and statement.sources == [happy]
    # Has an outcome now: not reviewed again.
    await say(memory, "hello", MONDAY + 21 * DAY)
    assert (await reflector.review_progress()).skipped and len(llm.calls) == 1


async def test_decision_without_outcome_or_citation_is_unchanged(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM
) -> None:
    entity_id, _ = await add_decision(graph, memory, MONDAY)
    await say(memory, "the ThinkPad arrived", MONDAY + DAY)
    nothing = {"decision": 1, "evidence": [], "outcome": None, "said_outright": False, "text": None}
    uncited = nothing | {"outcome": "works well", "text": "Rohan likes the ThinkPad."}
    script(llm, progress(decisions=[nothing]), progress(decisions=[uncited]))
    assert (await reflector.review_progress()).changes == []
    await say(memory, "it's fine", MONDAY + 2 * DAY)
    result = await reflector.review_progress()
    assert result.changes == [] and "without valid cited" in result.dropped[0]
    assert len(await graph.current_statements(entity_id)) == 2


async def test_bounded_items_most_recently_relevant_first_and_the_rest_wait(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, llm: FakeLLM, clock: Clock
) -> None:
    reflector = make_reflector(graph, llm, clock, max_progress_items=1)
    older = await add_goal(graph, memory, "learn Spanish", MONDAY)
    newer = await add_goal(graph, memory, "build a shed", MONDAY + DAY)
    await say(memory, "busy week", MONDAY + 3 * DAY)
    script(llm, progress([goal_review(1, "open")]), progress([goal_review(1, "open")]))
    result = await reflector.review_progress()
    assert result.reviewed_goals == [newer.id] and result.waiting == 1
    assert "build a shed" in prompt_of(llm) and "learn Spanish" not in prompt_of(llm, 0)
    result = await reflector.review_progress()
    assert result.reviewed_goals == [older.id] and result.waiting == 0
    assert (await reflector.review_progress()).skipped


async def test_progress_messages_over_budget_keep_related_ones(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, llm: FakeLLM, clock: Clock
) -> None:
    reflector = make_reflector(graph, llm, clock, progress_input_tokens=60)
    await add_goal(graph, memory, "run a marathon", MONDAY)
    await say(memory, "finally ran the marathon in four hours", MONDAY + DAY)
    await chatter(memory, 10, MONDAY + 2 * DAY)
    script(llm, progress([goal_review(1, "open")]))
    result = await reflector.review_progress()
    prompt = prompt_of(llm)
    assert "[1] Tue 2026-10-13 07:30: finally ran the marathon" in prompt
    assert 1 < result.shown_messages < 11
    assert "question number 9" in prompt  # then the newest


async def test_preempted_progress_review_propagates_and_saves_nothing(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM
) -> None:
    goal = await add_goal(graph, memory, "learn Spanish", MONDAY)
    await say(memory, "hola", MONDAY + DAY)
    script(llm, [Raise(Preempted("foreground"))])
    with pytest.raises(Preempted):
        await reflector.review_progress()
    assert await graph.get_meta(progress_key("goal", goal.id)) is None


# --- The job ----------------------------------------------------------------------


def job() -> Job:
    return Job(
        id=1,
        kind=REFLECT_JOB,
        payload={},
        status=JobStatus.RUNNING,
        attempts=1,
        max_attempts=5,
        run_after=T0,
        created_at=T0,
        updated_at=T0,
    )


INVALID = [LLMChunk(text="not json")]


async def test_handler_runs_progress_even_when_patterns_fail(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM
) -> None:
    goal = await add_goal(graph, memory, "learn Spanish", MONDAY - DAY)
    await gym_week(memory)
    script(llm, INVALID, INVALID, progress([goal_review(1, "open")]))
    with pytest.raises(ReflectionError, match="patterns: StructuredOutputError"):
        await reflection_handler(reflector)(job())
    assert len(llm.calls) == 3
    assert await graph.get_meta(progress_key("goal", goal.id)) is not None


async def test_handler_runs_patterns_and_reports_a_progress_failure(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM
) -> None:
    await add_goal(graph, memory, "learn Spanish", MONDAY - DAY)
    await gym_week(memory)
    script(llm, patterns(pattern([2, 3, 4])), INVALID, INVALID)
    with pytest.raises(ReflectionError, match="progress: StructuredOutputError"):
        await reflection_handler(reflector)(job())
    assert len(await inferred(graph)) == 1


async def test_handler_lets_preempted_through_and_the_runner_defers(
    graph: SQLiteGraphStore,
    memory: SQLiteMemoryStore,
    reflector: Reflector,
    llm: FakeLLM,
    path: Path,
    clock: Clock,
) -> None:
    await add_goal(graph, memory, "learn Spanish", MONDAY - DAY)
    await gym_week(memory)
    script(llm, [Raise(Preempted("foreground"))])
    with pytest.raises(Preempted):
        await reflection_handler(reflector)(job())
    assert len(llm.calls) == 1  # the progress review did not start after the pre-emption
    script(llm, [Raise(Preempted("foreground"))])
    async with SQLiteJobStore(path, clock=clock) as jobs:
        queued = await jobs.enqueue(REFLECT_JOB, {}, dedupe_key=REFLECT_JOB)
        runner = JobRunner(
            jobs,
            {REFLECT_JOB: reflection_handler(reflector)},
            defer_on=(Preempted,),
            defer_delay=60,
            clock=clock,
        )
        assert await runner.run_until_idle() == 0
        deferred = await jobs.get(queued.id)
    assert deferred is not None and deferred.status == JobStatus.PENDING
    assert deferred.attempts == 0


async def test_handler_is_safe_to_run_twice(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM
) -> None:
    goal = await add_goal(graph, memory, "learn Spanish", MONDAY - DAY)
    await gym_week(memory)
    script(llm, patterns(pattern([2, 3, 4])), progress([goal_review(1, "open")]))
    handle = reflection_handler(reflector)
    await handle(job())
    await handle(job())
    assert len(llm.calls) == 2
    assert len(await inferred(graph)) == 1
    kept = await graph.get_statement(goal.id)
    assert kept is not None and kept.invalid_at is None


# --- Scheduling -------------------------------------------------------------------


async def test_reflection_is_scheduled_after_merging_and_before_summaries(
    graph: SQLiteGraphStore, path: Path, clock: Clock
) -> None:
    await graph.reembed_missing()
    async with SQLiteJobStore(path, clock=clock) as jobs:
        first = await schedule_maintenance(jobs, graph, now=T0)
        assert [j.kind for j in first] == [
            DECAY_JOB,
            MERGE_JOB,
            REFLECT_JOB,
            SUMMARIES_JOB,
            PROFILE_JOB,
        ]
        reflect = first[2]
        assert reflect.dedupe_key == REFLECT_JOB
        again = await schedule_maintenance(jobs, graph, now=T0 + DAY)
        assert again[2].id == reflect.id  # still pending: not duplicated
        assert len({j.id for j in again}) == 5
        assert json.dumps([j.kind for j in again]) == json.dumps([j.kind for j in first])


async def test_near_identical_stated_pattern_is_dropped_without_shared_messages(
    memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM, graph: SQLiteGraphStore
) -> None:
    await gym_week(memory)
    elsewhere = await say(memory, "I go to the gym on Monday mornings", MONDAY - 40 * DAY)
    await graph.add_statement(
        new("Rohan usually goes to the gym on Monday mornings.", kind=StatementKind.HABIT),
        [elsewhere],
    )
    script(llm, patterns(pattern([2, 3, 4])))
    result = await reflector.infer_patterns()
    assert "stated it outright" in result.dropped[0].reason


async def test_cursors_of_items_that_are_gone_are_swept(
    graph: SQLiteGraphStore, memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM
) -> None:
    goal = await add_goal(graph, memory, "learn Spanish", MONDAY)
    await say(memory, "hola", MONDAY + DAY)
    script(llm, progress([goal_review(1, "open")]))
    await reflector.review_progress()
    assert await graph.get_meta(progress_key("goal", goal.id)) is not None
    await graph.invalidate(goal.id, MONDAY + 2 * DAY)  # closed some other way
    assert (await reflector.review_progress()).skipped
    assert await graph.get_meta(progress_key("goal", goal.id)) is None


async def test_a_stated_pattern_with_enough_other_evidence_goes_to_the_reconciler(
    memory: SQLiteMemoryStore, reflector: Reflector, llm: FakeLLM, graph: SQLiteGraphStore
) -> None:
    ids = await gym_week(memory)
    fourth = await say(memory, "gym again this morning", MONDAY + 21 * DAY)
    stated = await graph.add_statement(
        new(
            "Rohan goes to the gym on Monday mornings",
            kind=StatementKind.HABIT,
            predicate="goes to the gym",
        ),
        [ids[0]],
    )
    # Three cited messages besides the one the user stated it in: an inference
    # of its own, which the reconciler merges into the stated statement.
    script(llm, patterns(pattern([1, 2, 3, 4])), decision("reinforce", 1))
    result = await reflector.infer_patterns()
    [(_, outcome)] = result.results
    assert outcome.outcome is Outcome.REINFORCE and outcome.target_id == stated.id
    kept = await graph.get_statement(stated.id)
    assert kept is not None and kept.origin is Origin.STATED
    assert kept.sources == [*ids, fourth] and await inferred(graph) == []
