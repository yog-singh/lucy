"""Helpers for tests of the app with memory learning wired in (fake models, real SQLite)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lucy.app import LucyApp
from lucy.core.config import (
    AdapterConfig,
    LearningSettings,
    LLMSettings,
    RetrievalSettings,
    Settings,
)
from lucy.core.events import Event, TextDelta, TurnEnded, UserMessage
from lucy.core.registry import Registry
from lucy.core.types import FinishReason, LLMChunk
from lucy.memory.sqlite_store import SQLiteMemoryStore
from tests.fakes import FakeLLM, Script, adapter_for
from tests.graph_support import HASHING_RETRIEVAL, HashingEmbedder

TIMEOUT = 5


def reply(text: str) -> list[LLMChunk]:
    """A scripted plain-text answer."""
    return [LLMChunk(text=text), LLMChunk(finish_reason=FinishReason.STOP)]


@dataclass
class Models:
    """One fake model per role, and the embedder, as the registry hands them out."""

    chat: Sequence[Script] = ()
    extraction: Sequence[Script] = ()
    summary: Sequence[Script] = ()
    embedder: HashingEmbedder = field(default_factory=HashingEmbedder)

    def __post_init__(self) -> None:
        self.chat_llm = FakeLLM(self.chat)
        self.extraction_llm = FakeLLM(self.extraction)
        self.summary_llm = FakeLLM(self.summary)

    def registry(self) -> Registry:
        reg = Registry()
        reg.register("llm", "chat")(adapter_for(self.chat_llm))
        reg.register("llm", "extraction")(adapter_for(self.extraction_llm))
        reg.register("llm", "summary")(adapter_for(self.summary_llm))
        reg.register("embedder", "hash")(adapter_for(self.embedder))
        reg.register("memory", "sqlite")(SQLiteMemoryStore)
        return reg


#: `[retrieval]` settings for `HashingEmbedder` (the defaults suit embeddinggemma-2).
#: `RetrievalSettings` cannot express "no relative floor" or "no keyword check" as
#: None (None means the default), so it uses values that never bite.
HASHING_RETRIEVAL_SETTINGS = RetrievalSettings(
    vector_min_similarity=HASHING_RETRIEVAL["vector_min_similarity"],
    vector_margin=2.0,
    keyword_min_similarity=-1.0,
    entity_min_similarity=HASHING_RETRIEVAL["entity_min_similarity"],
    max_statements=1000,
)


def learning_settings(
    db: Path,
    *,
    embedder: bool = True,
    learning: LearningSettings | None = None,
    retrieval: RetrievalSettings | None = None,
) -> Settings:
    roles = {role: AdapterConfig(adapter=role) for role in ("chat", "extraction", "summary")}
    return Settings(
        llm=LLMSettings(roles=roles),
        embedder=AdapterConfig(adapter="hash") if embedder else None,
        memory=AdapterConfig(adapter="sqlite", options={"path": str(db)}),
        learning=learning or LearningSettings(),
        retrieval=retrieval or HASHING_RETRIEVAL_SETTINGS,
    )


def make_app(db: Path, models: Models, **kwargs: Any) -> LucyApp:
    """An app on `db` with `models`. The profile rewrite after extraction is not
    delayed unless `profile_delay` is given (the real default waits a minute)."""
    names = ("embedder", "learning", "retrieval")
    settings_kwargs = {k: kwargs.pop(k) for k in names if k in kwargs}
    kwargs.setdefault("runner_options", {"defer_delay": 0.01, "backoff": lambda _: 60.0})
    kwargs.setdefault("profile_delay", 0.0)
    return LucyApp(learning_settings(db, **settings_kwargs), registry=models.registry(), **kwargs)


async def run_turn(app: LucyApp, text: str, session_id: str = "s") -> list[Event]:
    """Publish one message and collect this session's events until its TurnEnded."""
    seen: list[Event] = []
    async with asyncio.timeout(TIMEOUT):
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


async def eventually(check: Callable[[], Awaitable[bool]], timeout: float = TIMEOUT) -> None:
    """Wait until `check()` is true (polling the event loop), or fail after `timeout`."""
    async with asyncio.timeout(timeout):
        while not await check():
            await asyncio.sleep(0.005)
