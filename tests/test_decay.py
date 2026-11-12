"""Confidence decay of inferred statements (lucy.memory.decay)."""

from __future__ import annotations

import math
import random
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from lucy.memory.decay import (
    DEFAULT_GRACE,
    DEFAULT_HALF_LIFE,
    LAST_RUN_KEY,
    RECORD_PREFIX,
    DecayPolicy,
    decay_inferred,
    record_key,
)
from lucy.memory.graph_store import SQLiteGraphStore
from lucy.memory.models import Origin, StatementKind
from tests.graph_support import T0, Clock, HashingEmbedder, new

DAY = timedelta(days=1)
#: The arithmetic below uses these values (the defaults until 2026-11-12), not the defaults.
POLICY = DecayPolicy(grace=timedelta(days=14), half_life=timedelta(days=90))  # floor 0.05


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
async def graph(tmp_path: Path, clock: Clock) -> AsyncIterator[SQLiteGraphStore]:
    async with SQLiteGraphStore(tmp_path / "lucy.db", HashingEmbedder(), clock=clock) as g:
        yield g


def expected(c: float, observed: datetime, now: datetime, policy: DecayPolicy = POLICY) -> float:
    """The reference rule: untouched within grace, then halving per half-life, to the floor."""
    unobserved = now - observed - policy.grace
    if unobserved <= timedelta(0) or c <= policy.floor:
        return c
    return max(policy.floor, c * 0.5 ** (unobserved / policy.half_life))


async def inferred(
    graph: SQLiteGraphStore,
    text: str,
    *,
    confidence: float = 0.8,
    observed_at: datetime = T0,
    kind: StatementKind = StatementKind.PREFERENCE,
    **kwargs: Any,
) -> int:
    statement = await graph.add_statement(
        new(
            text,
            kind=kind,
            origin=Origin.INFERRED,
            confidence=confidence,
            observed_at=observed_at,
            **kwargs,
        )
    )
    return statement.id


async def confidence(graph: SQLiteGraphStore, statement_id: int) -> float:
    statement = await graph.get_statement(statement_id)
    assert statement is not None
    return statement.confidence


async def run(graph: SQLiteGraphStore, clock: Clock, now: datetime, **policy: Any) -> Any:
    clock.now = now
    base = {"grace": POLICY.grace, "half_life": POLICY.half_life}
    return await decay_inferred(graph, now=now, policy=DecayPolicy(**(base | policy)))


# --- The rule -------------------------------------------------------------------


async def test_only_inferred_facts_preferences_habits_and_goals_decay(
    graph: SQLiteGraphStore, clock: Clock
) -> None:
    decaying = {
        kind: await inferred(graph, f"inferred {kind}", kind=kind)
        for kind in (
            StatementKind.FACT,
            StatementKind.PREFERENCE,
            StatementKind.HABIT,
            StatementKind.GOAL,
        )
    }
    kept = [
        await inferred(graph, "inferred decision", kind=StatementKind.DECISION),
        await inferred(graph, "inferred instruction", kind=StatementKind.INSTRUCTION),
        (
            await graph.add_statement(
                new("stated preference", kind=StatementKind.PREFERENCE, confidence=0.8)
            )
        ).id,
        await inferred(graph, "no longer true", invalid_at=T0 + DAY),
    ]
    now = T0 + 200 * DAY
    result = await run(graph, clock, now)
    for sid in decaying.values():
        assert await confidence(graph, sid) == pytest.approx(expected(0.8, T0, now))
    for sid in kept:
        assert await confidence(graph, sid) == 0.8
    assert result.lowered == 4 and sorted(result.lowered_ids) == sorted(decaying.values())
    assert await graph.get_meta(LAST_RUN_KEY) == now.isoformat()


async def test_grace_half_life_and_floor(graph: SQLiteGraphStore, clock: Clock) -> None:
    sid = await inferred(graph, "likes jazz", confidence=0.8)
    low = await inferred(graph, "maybe likes opera", confidence=0.04)
    await run(graph, clock, T0 + 14 * DAY)
    assert await confidence(graph, sid) == 0.8  # within grace: untouched
    await run(graph, clock, T0 + 104 * DAY)
    assert await confidence(graph, sid) == pytest.approx(0.4)  # one half-life after grace
    await run(graph, clock, T0 + 194 * DAY)
    assert await confidence(graph, sid) == pytest.approx(0.2)
    await run(graph, clock, T0 + 3000 * DAY)
    assert await confidence(graph, sid) == 0.05  # floor
    assert await confidence(graph, low) == 0.04  # never raised to the floor
    # Nothing is deleted or invalidated.
    statement = await graph.get_statement(sid)
    assert statement is not None and statement.invalid_at is None


async def test_running_twice_at_the_same_time_changes_nothing(
    graph: SQLiteGraphStore, clock: Clock
) -> None:
    ids = [await inferred(graph, f"pref {i}", confidence=0.5 + i / 20) for i in range(5)]
    now = T0 + 120 * DAY
    first = await run(graph, clock, now)
    values = [await confidence(graph, sid) for sid in ids]
    meta = await graph.list_meta(RECORD_PREFIX)
    second = await run(graph, clock, now)
    assert first.lowered == 5 and second.lowered == 0
    assert [await confidence(graph, sid) for sid in ids] == values
    assert await graph.list_meta(RECORD_PREFIX) == meta


async def test_daily_runs_equal_one_run(tmp_path: Path) -> None:
    results = []
    for schedule in ("daily", "once"):
        clock = Clock()
        async with SQLiteGraphStore(
            tmp_path / f"{schedule}.db", HashingEmbedder(), clock=clock
        ) as g:
            ids = [
                await inferred(g, f"habit {i}", confidence=0.3 + i / 10, observed_at=T0 - i * DAY)
                for i in range(6)
            ]
            days = range(1, 401) if schedule == "daily" else [400]
            for day in days:
                await run(g, clock, T0 + day * DAY)
            results.append([await confidence(g, sid) for sid in ids])
    assert results[0] == pytest.approx(results[1], rel=1e-12, abs=1e-12)
    for i, value in enumerate(results[1]):
        assert value == pytest.approx(expected(0.3 + i / 10, T0 - i * DAY, T0 + 400 * DAY))


# --- Reinforcement and other writers ---------------------------------------------


async def test_reinforced_between_runs_restarts_from_its_new_observation(
    graph: SQLiteGraphStore, clock: Clock
) -> None:
    sid = await inferred(graph, "gym on Mondays", kind=StatementKind.HABIT, confidence=0.8)
    await run(graph, clock, T0 + 104 * DAY)
    assert await confidence(graph, sid) == pytest.approx(0.4)
    reinforced_at = T0 + 110 * DAY
    clock.now = reinforced_at
    await graph.reinforce(sid, confidence=0.7, observed_at=reinforced_at)
    await run(graph, clock, reinforced_at + 10 * DAY)
    assert await confidence(graph, sid) == 0.7  # within the new grace period
    later = reinforced_at + 14 * DAY + 90 * DAY
    await run(graph, clock, later)
    assert await confidence(graph, sid) == pytest.approx(0.35)


async def test_late_reinforcement_with_an_old_observation_decays_from_that_observation(
    graph: SQLiteGraphStore, clock: Clock
) -> None:
    """Extraction can lag: the reinforcement is written after a run but observed before it."""
    sid = await inferred(graph, "drinks green tea", confidence=0.8)
    await run(graph, clock, T0 + 200 * DAY)
    observed = T0 + 100 * DAY  # before the run above
    clock.now = T0 + 201 * DAY
    await graph.reinforce(sid, confidence=0.9, observed_at=observed)
    now = T0 + 202 * DAY
    await run(graph, clock, now)
    assert await confidence(graph, sid) == pytest.approx(expected(0.9, observed, now))


async def test_statement_recorded_after_a_run_with_an_older_observation(
    graph: SQLiteGraphStore, clock: Clock
) -> None:
    """E.g. a rebuild from the log: its decay counts from its own observation."""
    await inferred(graph, "first", confidence=0.8)
    await run(graph, clock, T0 + 300 * DAY)
    clock.now = T0 + 301 * DAY
    sid = await inferred(graph, "learned late", confidence=0.8, observed_at=T0 + 50 * DAY)
    now = T0 + 302 * DAY
    await run(graph, clock, now)
    assert await confidence(graph, sid) == pytest.approx(expected(0.8, T0 + 50 * DAY, now))


async def test_confidence_changed_by_another_writer_without_new_observation(
    graph: SQLiteGraphStore, clock: Clock
) -> None:
    """The new value is taken as current at decay's last write; only later time counts."""
    sid = await inferred(graph, "likes hiking", confidence=0.8)
    first = T0 + 104 * DAY
    await run(graph, clock, first)
    await graph.set_confidence(sid, 0.6)
    await run(graph, clock, first)  # same time: nothing to apply
    assert await confidence(graph, sid) == 0.6
    await run(graph, clock, first + 90 * DAY)
    assert await confidence(graph, sid) == pytest.approx(0.3)
    await run(graph, clock, first + 180 * DAY)
    assert await confidence(graph, sid) == pytest.approx(0.15)


async def test_property_any_schedule_gives_the_reference_result(tmp_path: Path) -> None:
    """Random statements, random reinforcements (absolute values, sometimes extracted
    late), random run schedules: the result matches the reference rule and a single run."""
    for seed in range(5):
        rng = random.Random(seed)
        horizon = 500
        n = 8
        initial = [
            (round(rng.uniform(0.02, 1.0), 3), T0 - rng.randint(0, 100) * DAY) for _ in range(n)
        ]
        # (write day, statement index, confidence, observed day offset) in write order.
        events = []
        last_obs = [obs for _, obs in initial]
        for day in sorted(rng.sample(range(1, horizon), 10)):
            i = rng.randrange(n)
            written = T0 + day * DAY
            observed = max(written - rng.randint(0, 40) * DAY, last_obs[i] + timedelta(seconds=1))
            last_obs[i] = observed
            events.append((written, i, round(rng.uniform(0.1, 1.0), 3), observed))
        end = T0 + horizon * DAY
        policy = DecayPolicy(batch_size=3)
        schedules = {
            "random": sorted(T0 + rng.randint(1, horizon) * DAY for _ in range(rng.randint(5, 40))),
            # Every 7th day (daily runs are covered by test_daily_runs_equal_one_run).
            "weekly": [T0 + d * DAY for d in range(1, horizon + 1, 7)],
            "once": [],
        }
        finals = {}
        for name, runs in schedules.items():
            clock = Clock()
            async with SQLiteGraphStore(
                tmp_path / f"{seed}-{name}.db", HashingEmbedder(), clock=clock
            ) as g:
                ids = [
                    await inferred(g, f"statement {i}", confidence=c, observed_at=obs)
                    for i, (c, obs) in enumerate(initial)
                ]
                timeline = [(t, 0, None) for t in runs] + [(e[0], 1, e) for e in events]
                for when, _, event in sorted(timeline, key=lambda x: (x[0], x[1])):
                    clock.now = when
                    if event is None:
                        await decay_inferred(g, now=when, policy=policy)
                    else:
                        _, i, c, observed = event
                        await g.reinforce(ids[i], confidence=c, observed_at=observed)
                clock.now = end
                await decay_inferred(g, now=end, policy=policy)
                await decay_inferred(g, now=end, policy=policy)  # repeat: no change
                finals[name] = [await confidence(g, sid) for sid in ids]
        reference = []
        for i, (c, obs) in enumerate(initial):
            for _, j, value, observed in events:
                if j == i:
                    c, obs = value, observed
            reference.append(expected(c, obs, end, policy))
        for name, values in finals.items():
            assert values == pytest.approx(reference, rel=1e-9, abs=1e-12), (seed, name)


# --- Batching, interruption and records -------------------------------------------


async def test_batches_cover_everything(graph: SQLiteGraphStore, clock: Clock) -> None:
    ids = [await inferred(graph, f"pref {i}") for i in range(7)]
    result = await run(graph, clock, T0 + 104 * DAY, batch_size=2)
    assert result.examined == 7 and result.lowered == 7
    for sid in ids:
        assert await confidence(graph, sid) == pytest.approx(0.4)


async def test_interrupted_run_then_rerun_matches_an_uninterrupted_run(
    graph: SQLiteGraphStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = [await inferred(graph, f"pref {i}") for i in range(6)]
    now = T0 + 104 * DAY
    original = graph.set_confidences_and_meta
    calls = 0

    async def failing(*args: Any, **kwargs: Any) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("crash in the second batch")
        return await original(*args, **kwargs)

    monkeypatch.setattr(graph, "set_confidences_and_meta", failing)
    with pytest.raises(RuntimeError):
        await run(graph, clock, now, batch_size=2)
    partial = [await confidence(graph, sid) for sid in ids]
    assert partial[:2] == [pytest.approx(0.4)] * 2 and partial[2:] == [0.8] * 4
    monkeypatch.setattr(graph, "set_confidences_and_meta", original)
    later = now + 90 * DAY
    await run(graph, clock, later, batch_size=2)
    for sid in ids:
        assert await confidence(graph, sid) == pytest.approx(0.2)


async def test_records_of_gone_or_reinforced_statements_are_swept(
    graph: SQLiteGraphStore, clock: Clock
) -> None:
    kept = await inferred(graph, "kept")
    invalidated = await inferred(graph, "invalidated")
    reinforced = await inferred(graph, "reinforced")
    now = T0 + 104 * DAY
    await run(graph, clock, now)
    assert len(await graph.list_meta(RECORD_PREFIX)) == 3
    await graph.invalidate(invalidated, at=now)
    await graph.reinforce(reinforced, confidence=0.9, observed_at=now)
    result = await run(graph, clock, now + DAY)
    assert result.records_removed == 2
    assert [k for k, _ in await graph.list_meta(RECORD_PREFIX)] == [record_key(kept)]


def test_default_grace_and_half_life() -> None:
    """Chosen from simulated trajectories (see `DEFAULT_GRACE` in lucy.memory.decay)."""
    assert DecayPolicy().grace == DEFAULT_GRACE == timedelta(days=30)
    assert DecayPolicy().half_life == DEFAULT_HALF_LIFE == timedelta(days=30)


async def test_default_policy_separates_stopped_from_continuing_habits(
    graph: SQLiteGraphStore, clock: Clock
) -> None:
    """A habit unobserved for 48 days is uncertain (< 0.6); one seen monthly is untouched."""
    stopped = await inferred(graph, "swims on Saturdays", confidence=0.89, kind=StatementKind.HABIT)
    monthly = await inferred(graph, "volunteers monthly", confidence=0.75, kind=StatementKind.HABIT)
    policy = DecayPolicy()
    await decay_inferred(graph, now=T0 + 30 * DAY, policy=policy)
    assert await confidence(graph, monthly) == 0.75  # a monthly gap stays within grace
    await decay_inferred(graph, now=T0 + 48 * DAY, policy=policy)
    assert await confidence(graph, stopped) < 0.6
    await decay_inferred(graph, now=T0 + 125 * DAY, policy=policy)
    assert await confidence(graph, stopped) < 0.1


async def test_policy_validation() -> None:
    with pytest.raises(ValueError):
        DecayPolicy(half_life=timedelta(0))
    with pytest.raises(ValueError):
        DecayPolicy(floor=1.5)
    with pytest.raises(ValueError):
        DecayPolicy(batch_size=0)
    assert math.isclose(POLICY.floor, 0.05)
