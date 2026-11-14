"""The SQLite reminder store: adding, listing, cancelling, and the claim / begin /
deliver steps a scheduler uses, including two connections on one file (as `lucy
chat` and `lucy listen` would be) and the migration onto an existing database."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite

from lucy.core.events import UserMessage
from lucy.core.reminders import ReminderKind, ReminderStatus, ReminderStore
from lucy.memory import migrations
from lucy.memory.db import apply_migrations
from lucy.memory.sqlite_reminders import SQLiteReminderStore
from lucy.memory.sqlite_store import SQLiteMemoryStore

T0 = datetime(2026, 11, 14, 10, 0, tzinfo=UTC)
LEASE = timedelta(seconds=60)


class Clock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


async def test_add_list_get_and_cancel(tmp_path: Path) -> None:
    async with SQLiteReminderStore(tmp_path / "lucy.db", clock=Clock()) as store:
        assert isinstance(store, ReminderStore)
        later = await store.add(ReminderKind.REMINDER, "passport", T0 + timedelta(days=1))
        sooner = await store.add(ReminderKind.TIMER, "", T0 + timedelta(minutes=10))
        assert later.status is ReminderStatus.PENDING and later.label == f"r{later.id}"
        assert sooner.length == timedelta(minutes=10)
        assert [r.id for r in await store.upcoming()] == [sooner.id, later.id]
        assert await store.get(later.id) == later

        cancelled = await store.cancel(later.id)
        assert cancelled is not None and cancelled.status is ReminderStatus.CANCELLED
        assert await store.cancel(later.id) is None  # only once
        assert await store.cancel(9999) is None
        assert [r.id for r in await store.upcoming()] == [sooner.id]


async def test_delivered_reminders_cannot_be_cancelled(tmp_path: Path) -> None:
    async with SQLiteReminderStore(tmp_path / "lucy.db", clock=Clock()) as store:
        r = await store.add(ReminderKind.REMINDER, "bank", T0)
        await store.claim_due(T0, "a")
        await store.mark_delivered([r.id], "a", "s1", "t1")
        assert await store.cancel(r.id) is None
        stored = await store.get(r.id)
        assert stored is not None and stored.status is ReminderStatus.DELIVERED
        assert (stored.delivered_session, stored.delivered_turn) == ("s1", "t1")
        assert await store.upcoming() == []


async def test_claim_takes_every_missed_reminder_in_one_batch(tmp_path: Path) -> None:
    async with SQLiteReminderStore(tmp_path / "lucy.db", clock=Clock()) as store:
        missed = [
            await store.add(ReminderKind.REMINDER, f"r{i}", T0 - timedelta(hours=i))
            for i in (3, 1, 2)
        ]
        future = await store.add(ReminderKind.REMINDER, "later", T0 + timedelta(minutes=5))
        claimed = await store.claim_due(T0, "a")
        assert [r.text for r in claimed] == ["r3", "r2", "r1"]  # oldest first
        assert all(r.status is ReminderStatus.FIRING and r.claimed_by == "a" for r in claimed)
        assert {r.id for r in claimed} == {r.id for r in missed}
        assert await store.claim_due(T0, "a") == []  # a fresh claim is not taken again
        assert [r.id for r in await store.upcoming()][-1] == future.id


async def test_next_due_counts_pending_and_lapsing_claims(tmp_path: Path) -> None:
    async with SQLiteReminderStore(tmp_path / "lucy.db", clock=Clock()) as store:
        assert await store.next_due() is None
        await store.add(ReminderKind.REMINDER, "later", T0 + timedelta(minutes=5))
        assert await store.next_due() == T0 + timedelta(minutes=5)
        await store.add(ReminderKind.REMINDER, "now", T0)
        await store.claim_due(T0, "a")
        assert await store.next_due(LEASE) == T0 + LEASE  # the claim lapses first


async def test_an_expired_lease_is_reclaimed_and_a_fresh_one_is_not(tmp_path: Path) -> None:
    async with SQLiteReminderStore(tmp_path / "lucy.db", clock=Clock()) as store:
        r = await store.add(ReminderKind.REMINDER, "bank", T0)
        assert [x.id for x in await store.claim_due(T0, "a", LEASE)] == [r.id]
        assert await store.claim_due(T0 + timedelta(seconds=59), "b", LEASE) == []
        taken = await store.claim_due(T0 + timedelta(seconds=60), "b", LEASE)
        assert [(x.id, x.claimed_by) for x in taken] == [(r.id, "b")]


async def test_begin_delivery_refuses_a_stolen_or_cancelled_reminder(tmp_path: Path) -> None:
    clock = Clock()
    async with SQLiteReminderStore(tmp_path / "lucy.db", clock=clock) as store:
        stolen = await store.add(ReminderKind.REMINDER, "stolen", T0)
        cancelled = await store.add(ReminderKind.REMINDER, "cancelled", T0)
        kept = await store.add(ReminderKind.REMINDER, "kept", T0)
        ids = [stolen.id, cancelled.id, kept.id]
        assert len(await store.claim_due(T0, "a")) == 3
        # a's claim lapses; b takes over the first, and the user cancels the second.
        later = T0 + timedelta(minutes=2)
        clock.now = later
        await store.begin_delivery([kept.id], "a")  # renews only kept
        assert [x.id for x in await store.claim_due(later, "b")] == [stolen.id, cancelled.id]
        assert await store.begin_delivery([stolen.id], "a") == []
        assert (await store.cancel(cancelled.id)) is not None  # cancel works while firing
        assert [x.id for x in await store.begin_delivery(ids, "b")] == [stolen.id]
        delivering = await store.begin_delivery(ids, "a")
        assert [x.id for x in delivering] == [kept.id]
        assert delivering[0].claimed_at == later
        assert await store.mark_delivered([stolen.id], "a", "s", "t") == []
        done = await store.mark_delivered(ids, "a", "s", "t")
        assert [x.id for x in done] == [kept.id]
        stolen_now = await store.get(stolen.id)
        assert stolen_now is not None and stolen_now.claimed_by == "b"
        assert stolen_now.status is ReminderStatus.FIRING


async def test_two_connections_claiming_at_once_never_share_a_reminder(tmp_path: Path) -> None:
    path = tmp_path / "lucy.db"
    async with (
        SQLiteReminderStore(path, clock=Clock()) as one,
        SQLiteReminderStore(path, clock=Clock()) as two,
    ):
        added = [
            await one.add(ReminderKind.REMINDER, f"r{i}", T0 + timedelta(seconds=i))
            for i in range(40)
        ]

        async def sweep(store: SQLiteReminderStore, owner: str) -> list[int]:
            got: list[int] = []
            for step in range(41):
                got += [r.id for r in await store.claim_due(T0 + timedelta(seconds=step), owner)]
                await asyncio.sleep(0)
            return got

        a, b = await asyncio.gather(sweep(one, "a"), sweep(two, "b"))
        assert sorted(a + b) == sorted(r.id for r in added)
        assert not set(a) & set(b)
        owners = {r.id: r.claimed_by for r in await one.upcoming()}
        assert all(owners[i] == "a" for i in a) and all(owners[i] == "b" for i in b)


async def test_ids_are_never_reused(tmp_path: Path) -> None:
    path = tmp_path / "lucy.db"
    async with SQLiteReminderStore(path, clock=Clock()) as store:
        first = await store.add(ReminderKind.REMINDER, "a", T0)
        second = await store.add(ReminderKind.REMINDER, "b", T0)
    with sqlite3.connect(path) as conn:  # even after the newest row is deleted
        conn.execute("DELETE FROM reminders WHERE id = ?", (second.id,))
    async with SQLiteReminderStore(path, clock=Clock()) as store:
        third = await store.add(ReminderKind.REMINDER, "c", T0)
    assert third.id > second.id > first.id


async def test_migration_adds_reminders_to_an_existing_database(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    async with aiosqlite.connect(path, autocommit=True) as conn:
        assert await apply_migrations(conn, migrations.MIGRATIONS[:5]) == 5
    old = SQLiteMemoryStore(path)
    old._conn = await aiosqlite.connect(path, autocommit=True)  # stay on version 5
    kept = UserMessage(session_id="s", text="kept")
    await old.append_episode(kept)
    await old.close()

    async with SQLiteReminderStore(path, clock=Clock()) as store:
        r = await store.add(ReminderKind.REMINDER, "bank", T0)
        assert (await store.get(r.id)) == r
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == len(migrations.MIGRATIONS)
    async with SQLiteMemoryStore(path) as memory:
        assert await memory.recent_episodes("s", 10) == [kept]
