"""The model residency policy as the app wires it (fakes only)."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from lucy.app import LucyApp
from lucy.core.config import (
    AdapterConfig,
    ContextSettings,
    LLMSettings,
    ResidencySettings,
    Settings,
)
from lucy.core.events import SpeechActivity
from lucy.core.registry import Registry
from lucy.core.types import FinishReason, LLMChunk, Message, Role
from lucy.memory.sqlite_store import SQLiteMemoryStore
from tests.fakes import FakeLLM, ModelServer, ResidentFake, adapter_for
from tests.test_app import run_turn


def reply(text: str) -> list[LLMChunk]:
    return [LLMChunk(text=text), LLMChunk(finish_reason=FinishReason.STOP)]


def make(db: Path, llm: FakeLLM, **residency: object) -> tuple[Settings, Registry]:
    registry = Registry()
    registry.register("llm", "fake")(adapter_for(llm))
    registry.register("memory", "sqlite")(SQLiteMemoryStore)
    settings = Settings(
        llm=LLMSettings(roles={"chat": AdapterConfig(adapter="fake", options={"model": "m"})}),
        memory=AdapterConfig(adapter="sqlite", options={"path": str(db)}),
        context=ContextSettings(layout="stable_prefix"),
        residency=ResidencySettings(**residency),  # type: ignore[arg-type]
    )
    return settings, registry


def rows(db: Path, table: str) -> int:
    with sqlite3.connect(db) as conn:
        return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


async def wait_for_cycle(app: LucyApp, count: int = 1) -> None:
    residency = app.residency
    assert residency is not None
    async with asyncio.timeout(5):
        while len(residency.records) < count:
            await asyncio.sleep(0.01)


async def test_the_app_runs_residency_for_a_capable_chat_adapter(tmp_path: Path) -> None:
    db = tmp_path / "lucy.db"
    server = ModelServer()
    llm = ResidentFake(server, [reply("Hello."), reply("Again.")])
    settings, registry = make(db, llm, idle_seconds=0.1, text_quiet_seconds=0)
    async with LucyApp(settings, registry=registry) as app:
        assert app.residency is not None
        await run_turn(app, "Hi.")
        episodes, traces = rows(db, "episodes"), rows(db, "turn_traces")
        await wait_for_cycle(app)
        # The warm-up left nothing behind: no episodes, no traces, no gateway requests.
        assert (rows(db, "episodes"), rows(db, "turn_traces")) == (episodes, traces)
        assert len(llm.calls) == 1
        await run_turn(app, "Once more.")
    assert server.calls[:2] == ["size", "unload"]
    first, second = server.prefills
    assert first[-1] == Message(role=Role.USER, content="")
    assert second[-1] == Message(role=Role.USER, content="x")
    # The pre-filled prefix is exactly how the next real turn began.
    prefix = first[:-1]
    sent = llm.calls[1]
    assert sent[: len(prefix)] == prefix
    assert sent[len(prefix)] == Message(role=Role.USER, content="Once more.")


async def test_residency_is_off_when_disabled_or_unsupported(tmp_path: Path) -> None:
    llm = ResidentFake(ModelServer(), [])
    settings, registry = make(tmp_path / "a.db", llm, enabled=False)
    async with LucyApp(settings, registry=registry) as app:
        assert app.residency is None
    plain = FakeLLM([])
    settings, registry = make(tmp_path / "b.db", plain)
    async with LucyApp(settings, registry=registry) as app:
        assert app.residency is None


async def test_stop_cancels_a_warm_up_in_flight(tmp_path: Path) -> None:
    from tests.fakes import Pause

    server = ModelServer()
    server.hold["prefill:"] = pause = Pause()
    llm = ResidentFake(server, [reply("Hello.")])
    settings, registry = make(tmp_path / "lucy.db", llm, idle_seconds=0.1, text_quiet_seconds=0)
    app = LucyApp(settings, registry=registry)
    await app.start()
    await run_turn(app, "Hi.")
    async with asyncio.timeout(5):
        await pause.reached.wait()
    await asyncio.wait_for(app.stop(), 2)
    assert app.residency is None
    assert "prefill:x" not in server.calls


async def test_speech_activity_reaches_the_policy_and_is_not_stored(tmp_path: Path) -> None:
    db = tmp_path / "lucy.db"
    server = ModelServer()
    llm = ResidentFake(server, [reply("Hello.")])
    settings, registry = make(db, llm)
    async with LucyApp(settings, registry=registry) as app:
        residency = app.residency
        assert residency is not None
        await app.bus.publish(SpeechActivity(session_id="v", state="speaking"))
        async with asyncio.timeout(5):
            while not residency.records:  # the model was not known loaded: loaded at once
                await asyncio.sleep(0.01)
        assert residency.voice_active
        assert residency.records[-1].reason == "speech"
        assert await residency.busy_reason() == "speech"
        await app.bus.publish(SpeechActivity(session_id="v", state="idle"))
        await run_turn(app, "Hi.", session_id="v")
        await asyncio.sleep(0.05)
        assert await residency.busy_reason() is None
    assert server.calls[0] == "load"
    with sqlite3.connect(db) as conn:
        types = [r[0] for r in conn.execute("SELECT payload FROM episodes")]
    assert not any("speech_activity" in t for t in types)
