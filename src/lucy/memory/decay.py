"""Confidence decay of inferred statements (docs/memory.md "4. Decay", ADR 0012).

An inferred fact, preference, habit or goal that stops being observed slowly
loses confidence. Stated statements, instructions and decisions never decay.
Decay only lowers `confidence` (and so ranking); it never deletes or
invalidates anything.

The rule, for a statement last observed at L whose confidence was last set by
someone other than decay to `v`:

    target(now) = min(current, max(floor, v * 0.5 ** (max(0, now - (L + grace)) / half_life)))

so a statement is untouched for `grace` after its last observation, then halves
every `half_life` of unobserved time, down to `floor`; one already below the
floor is left alone.

Why per-statement state is stored
---------------------------------
Decay must give the same result however often it runs: twice at the same
`now` changes nothing the second time, and daily for N days equals once after
N days. Applying a factor to the *current* confidence on each run cannot do
this on its own: it needs to know how much decay the current value already
contains. A single "last run" timestamp is not enough either:

- a statement recorded since the last run may carry an older observation time
  (late extraction, or a rebuild from the log), so it needs decay from L +
  grace, not from the last run;
- a reinforcement written after the last run may carry an observation time
  older than the last run (late extraction): its confidence is a fresh value
  as of that observation, so it decays from its own L + grace;
- an interrupted run leaves some batches decayed and others not.

So decay keeps, per statement it has lowered, a record in `memory_meta`
(`derived.decay.s.<id>`) holding the undecayed value `v`, the anchor time `a`
from which it decays, and the `last_observed_at` and confidence it saw and
wrote. Each run computes the target as a pure function of `(v, a, now)`, so
the result does not depend on the schedule, and repeating a run is a no-op.
Each batch's confidences and records are written in one transaction.

On each run, for a statement with current confidence `c` and observation L:

- no record, or the record's L differs (the statement was reinforced: L moved
  forward and its confidence was reset by the reinforcement): `v = c`,
  `a = L + grace`;
- record matches L and `c` is what decay wrote: `v`, `a` from the record;
- record matches L but `c` differs (someone else set the confidence without a
  new observation, presumably starting from the decayed value): `v = c`, with
  `a` = the time of decay's last write, so only the time since then counts.

A reinforcement that moves L but does not reset the confidence keeps the
decayed value as its new starting point; that case depends on when decay last
ran, by design of the reinforcer. Records of statements that are gone, no
longer current, or were reinforced are removed by a sweep at the end of each
run. `derived.decay.last_run` records the last completed run (informational).

Work is done in batches of `policy.batch_size` statements (paging by id), so a
large graph is never loaded at once.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Final

from lucy.memory.graph_store import DERIVED_META_PREFIX
from lucy.memory.models import Origin, Statement, StatementKind
from lucy.memory.ports import GraphStore

log = logging.getLogger(__name__)

#: Kinds whose inferred statements decay.
DECAYING_KINDS: Final = (
    StatementKind.FACT,
    StatementKind.PREFERENCE,
    StatementKind.HABIT,
    StatementKind.GOAL,
)
#: Per-statement decay records: `RECORD_PREFIX + zero-padded statement id`.
RECORD_PREFIX: Final = f"{DERIVED_META_PREFIX}decay.s."
#: When the last decay run completed (ISO 8601 UTC).
LAST_RUN_KEY: Final = f"{DERIVED_META_PREFIX}decay.last_run"


def record_key(statement_id: int) -> str:
    """The `memory_meta` key of a statement's decay record (sorts by id)."""
    return f"{RECORD_PREFIX}{statement_id:012d}"


#: Default grace period and half-life (overridable in `[learning]`). Chosen on
#: 2026-11-12 by simulating daily runs with the real reinforcement rule (an
#: inference starts at 0.40 and each run with new evidence moves it a quarter of
#: the way to 1.0); the earlier 14 / 90 days kept a weekly habit that had stopped
#: at 0.61 four months later and above 0.38 after six. With 30 / 30 days:
#:
#: - a weekly habit that continues: 0.66 after a month, 0.89 after two, then ~1.0;
#:   it never decays (re-observed within the grace period);
#: - a weekly habit that stops (at 0.89): uncertain (below 0.6) 48 days after its
#:   last occurrence, below 0.2 after 95 days, below 0.1 after 125;
#: - a monthly habit that continues is not punished: 0.55, 0.66, 0.75, 0.81 after
#:   its 4th to 7th occurrence (no decay with 30-day gaps; 0.60 by day 150 even
#:   with 35-day gaps, against 0.57 with 14 / 90 days).
DEFAULT_GRACE: Final = timedelta(days=30)
DEFAULT_HALF_LIFE: Final = timedelta(days=30)


@dataclass(frozen=True)
class DecayPolicy:
    """How fast unobserved inferences lose confidence."""

    grace: timedelta = DEFAULT_GRACE
    """Untouched while observed within this long."""
    half_life: timedelta = DEFAULT_HALF_LIFE
    """Confidence halves every `half_life` of unobserved time after the grace period."""
    floor: float = 0.05
    """Decay never takes confidence below this."""
    batch_size: int = 500

    def __post_init__(self) -> None:
        if self.grace < timedelta(0) or self.half_life <= timedelta(0):
            raise ValueError("grace must be >= 0 and half_life > 0")
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError("floor must be within [0, 1]")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True)
class DecayResult:
    """What a decay run did."""

    now: datetime
    examined: int = 0
    """Statements past their grace period that were looked at."""
    lowered: int = 0
    """Statements whose confidence was lowered."""
    records_removed: int = 0
    """Stale decay records swept."""
    lowered_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class _Record:
    value: float  # v: the undecayed confidence
    anchor: datetime  # a: decay counts from here
    observed: datetime  # the statement's last_observed_at when written
    written: float  # the confidence decay wrote
    at: datetime  # when decay wrote it

    def dump(self) -> str:
        return json.dumps(
            {
                "v": self.value,
                "a": self.anchor.isoformat(),
                "l": self.observed.isoformat(),
                "w": self.written,
                "t": self.at.isoformat(),
            }
        )

    @classmethod
    def load(cls, raw: str) -> _Record | None:
        try:
            data = json.loads(raw)
            return cls(
                value=float(data["v"]),
                anchor=datetime.fromisoformat(data["a"]),
                observed=datetime.fromisoformat(data["l"]),
                written=float(data["w"]),
                at=datetime.fromisoformat(data["t"]),
            )
        except (ValueError, KeyError, TypeError):
            return None


def decayed(value: float, anchor: datetime, now: datetime, policy: DecayPolicy) -> float:
    """`value` decayed from `anchor` to `now` (no floor, no clamping to the current value)."""
    elapsed = (now - anchor) / policy.half_life
    if elapsed <= 0:
        return value
    return value * math.pow(0.5, elapsed)


def _start(
    statement: Statement, record: _Record | None, policy: DecayPolicy
) -> tuple[float, datetime]:
    """The undecayed value and anchor of `statement` (see the module docstring)."""
    c, observed = statement.confidence, statement.last_observed_at
    if record is None or record.observed != observed:
        return c, observed + policy.grace
    # Tolerate a last-bit difference from the REAL round trip through SQLite.
    if math.isclose(record.written, c, rel_tol=1e-12, abs_tol=1e-15):
        return record.value, record.anchor
    return c, record.at


async def decay_inferred(
    graph: GraphStore, *, now: datetime, policy: DecayPolicy | None = None
) -> DecayResult:
    """Lower the confidence of inferred statements that have not been observed lately.

    Safe to interrupt and to repeat: each batch is written atomically with its
    records, and the result depends only on the stored state and `now`.
    `Preempted` cannot occur (no model calls).
    """
    policy = policy or DecayPolicy()
    examined = lowered = 0
    lowered_ids: list[int] = []
    after: int | None = None
    while True:
        batch = await graph.list_statements(
            kinds=DECAYING_KINDS,
            origin=Origin.INFERRED,
            observed_before=now - policy.grace,
            after_id=after,
            limit=policy.batch_size,
        )
        if not batch:
            break
        after = batch[-1].id
        batch = [s for s in batch if s.is_current(now)]
        examined += len(batch)
        stored = await graph.get_meta_many(record_key(s.id) for s in batch)
        confidences: dict[int, float] = {}
        records: dict[str, str | None] = {}
        for statement in batch:
            key = record_key(statement.id)
            record = _Record.load(stored[key]) if key in stored else None
            value, anchor = _start(statement, record, policy)
            c = statement.confidence
            target = min(c, max(policy.floor, decayed(value, anchor, now, policy)))
            if target < c:
                confidences[statement.id] = target
                records[key] = _Record(
                    value=value,
                    anchor=anchor,
                    observed=statement.last_observed_at,
                    written=target,
                    at=now,
                ).dump()
        if confidences:
            await graph.set_confidences_and_meta(confidences, records)
            lowered += len(confidences)
            lowered_ids.extend(confidences)
    removed = await _sweep(graph, now)
    await graph.set_meta(LAST_RUN_KEY, now.isoformat())
    if lowered or removed:
        log.info(
            "decay: lowered %d of %d statement(s), swept %d record(s)", lowered, examined, removed
        )
    return DecayResult(
        now=now,
        examined=examined,
        lowered=lowered,
        records_removed=removed,
        lowered_ids=lowered_ids,
    )


async def _sweep(graph: GraphStore, now: datetime, batch_size: int = 500) -> int:
    """Delete records of statements that are gone, not current, or observed since."""
    removed = 0
    after: str | None = None
    while True:
        page = await graph.list_meta(RECORD_PREFIX, after=after, limit=batch_size)
        if not page:
            return removed
        after = page[-1][0]
        ids = {int(key[len(RECORD_PREFIX) :]): raw for key, raw in page}
        statements = {s.id: s for s in await graph.get_statements(sorted(ids))}
        stale: dict[str, str | None] = {}
        for sid, raw in ids.items():
            statement = statements.get(sid)
            record = _Record.load(raw)
            if (
                statement is None
                or record is None
                or not statement.is_current(now)
                or statement.last_observed_at != record.observed
            ):
                stale[record_key(sid)] = None
        if stale:
            await graph.write_meta(stale)
            removed += len(stale)
