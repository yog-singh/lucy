import asyncio
import sqlite3
from pathlib import Path

import pytest

from lucy.app import LucyApp
from lucy.core.config import AdapterConfig, LLMSettings, Settings
from lucy.core.events import Event, TextDelta, TurnEnded, UserMessage
from lucy.core.registry import Registry, RegistryError
from lucy.core.types import FinishReason, LLMChunk, Role
from lucy.memory.sqlite_store import SQLiteMemoryStore
from tests.fakes import FakeLLM, adapter_for


def reply(text: str) -> list[LLMChunk]:
    return [LLMChunk(text=text), LLMChunk(finish_reason=FinishReason.STOP)]


def make_registry(llm: FakeLLM) -> Registry:
    reg = Registry()
    reg.register("llm", "fake")(adapter_for(llm))
    reg.register("memory", "sqlite")(SQLiteMemoryStore)
    return reg


def make_settings(db: Path, **roles: AdapterConfig) -> Settings:
    roles = roles or {"chat": AdapterConfig(adapter="fake", options={"model": "m"})}
    return Settings(
        llm=LLMSettings(roles=roles),
        memory=AdapterConfig(adapter="sqlite", options={"path": str(db)}),
    )


async def run_turn(app: LucyApp, text: str, session_id: str = "s") -> list[Event]:
    """Publish one message and collect this session's events until its TurnEnded."""
    seen: list[Event] = []
    async with asyncio.timeout(5):
        async with app.bus.subscribe() as events:
            await app.bus.publish(UserMessage(session_id=session_id, text=text))
            async for event in events:
                if event.session_id == session_id and event.turn_id is not None:
                    seen.append(event)
                    if isinstance(event, TurnEnded):
                        return seen
    raise AssertionError("bus closed before the turn ended")


def text_of(events: list[Event]) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


async def test_conversation_survives_a_restart(tmp_path: Path) -> None:
    db = tmp_path / "lucy.db"
    first = FakeLLM([reply("Nice to meet you, Sam."), reply("Blue it is.")])
    async with LucyApp(make_settings(db), registry=make_registry(first)) as app:
        assert isinstance(app.memory, SQLiteMemoryStore)
        one = await run_turn(app, "Hi, I am Sam.")
        two = await run_turn(app, "My favourite colour is blue.")
    assert text_of(one) == "Nice to meet you, Sam."
    assert text_of(two) == "Blue it is."
    assert first.aclose_calls == 1

    second = FakeLLM([reply("You are Sam.")])
    app = LucyApp(make_settings(db), registry=make_registry(second))
    await app.start()
    try:
        three = await run_turn(app, "Who am I?")
    finally:
        await app.stop()
    assert text_of(three) == "You are Sam."
    ended = three[-1]
    assert isinstance(ended, TurnEnded) and ended.reason == "completed"

    # The third turn saw the whole earlier conversation, rebuilt from the database.
    (messages,) = second.calls
    assert messages[0].role is Role.SYSTEM and "Lucy" in messages[0].content
    assert [(m.role, m.content) for m in messages[1:]] == [
        (Role.USER, "Hi, I am Sam."),
        (Role.ASSISTANT, "Nice to meet you, Sam."),
        (Role.USER, "My favourite colour is blue."),
        (Role.ASSISTANT, "Blue it is."),
        (Role.USER, "Who am I?"),
    ]

    turn_ids = {e.turn_id for e in one + two + three}
    assert len(turn_ids) == 3
    with sqlite3.connect(db) as conn:
        types = [r[0] for r in conn.execute("SELECT type FROM episodes ORDER BY id")]
        traced = {r[0]: r[1] for r in conn.execute("SELECT turn_id, outcome FROM turn_traces")}
    assert types == ["user_message", "assistant_message", "turn_ended"] * 3
    assert traced == dict.fromkeys(turn_ids, "completed")
    async with SQLiteMemoryStore(db) as store:
        trace = await store.get_trace(ended.turn_id or "")
    assert trace is not None and trace.steps[0].messages == messages


async def test_stop_is_idempotent_and_closes_everything(tmp_path: Path) -> None:
    llm = FakeLLM([])
    app = LucyApp(make_settings(tmp_path / "lucy.db"), registry=make_registry(llm))
    await app.start()
    bus = app.bus
    await app.stop()
    await app.stop()
    assert app.bus is bus and bus.closed
    assert llm.aclose_calls == 1
    with pytest.raises(RuntimeError, match="not running"):
        app.memory  # noqa: B018
    current = asyncio.current_task()
    assert [t for t in asyncio.all_tasks() if t is not current] == []
    with pytest.raises(RuntimeError, match="only be called once"):
        await app.start()


async def test_each_distinct_adapter_is_closed_once(tmp_path: Path) -> None:
    shared, other = FakeLLM([]), FakeLLM([])
    shared_adapter, other_adapter = adapter_for(shared), adapter_for(other)
    reg = Registry()
    reg.register("llm", "fake")(shared_adapter)
    reg.register("llm", "other")(other_adapter)
    reg.register("memory", "sqlite")(SQLiteMemoryStore)
    same = AdapterConfig(adapter="fake", options={"model": "m"})
    settings = make_settings(
        tmp_path / "lucy.db", chat=same, summary=same, extraction=AdapterConfig(adapter="other")
    )
    async with LucyApp(settings, registry=reg):
        pass
    assert shared_adapter.options == [{"model": "m"}]  # chat and summary share one instance
    assert shared.aclose_calls == 1
    assert other.aclose_calls == 1


async def test_stop_after_failed_start(tmp_path: Path) -> None:
    llm = FakeLLM([])
    settings = make_settings(tmp_path / "lucy.db", chat=AdapterConfig(adapter="no_such_adapter"))
    app = LucyApp(settings, registry=make_registry(llm))
    with pytest.raises(RegistryError, match="no_such_adapter"):
        await app.start()
    with pytest.raises(RuntimeError, match="not running"):
        app.bus  # noqa: B018
    await app.stop()
    await app.stop()
    # Nothing is left running.
    current = asyncio.current_task()
    assert [t for t in asyncio.all_tasks() if t is not current] == []


async def test_memory_defaults_to_sqlite_in_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(llm=LLMSettings(roles={"chat": AdapterConfig(adapter="fake")}))
    async with LucyApp(settings, registry=make_registry(FakeLLM([]))) as app:
        assert isinstance(app.memory, SQLiteMemoryStore)
        assert app.memory.path == Path("data/lucy.db")
