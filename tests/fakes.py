"""Test doubles for the core ports. No models are loaded."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from lucy.core.events import AnyEvent
from lucy.core.trace import TurnTrace
from lucy.core.types import (
    EmbedderCaps,
    LLMCaps,
    LLMChunk,
    MemoryContext,
    Message,
    Risk,
    ToolOutput,
    ToolSpec,
)


@dataclass
class Pause:
    """Script step for `FakeLLM`: set `reached`, then wait until `release` is set."""

    reached: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class Raise:
    """Script step for `FakeLLM`: raise `error` mid-stream."""

    error: Exception


Script = Sequence[LLMChunk | Pause | Raise]


class FakeLLM:
    """Scripted LLM: each `stream` call yields the next scripted response.

    A response may contain `Pause` steps (to hold the stream at a known point)
    and `Raise` steps (to fail mid-stream).

    `exits` records, per started stream, how it ended: "done" (script finished),
    "closed" (closed by the consumer), "cancelled" or "raised". `open_streams`
    counts streams started and not yet ended.
    """

    def __init__(
        self,
        responses: Sequence[Script],
        *,
        context_window: int = 8192,
        json_schema: bool = True,
    ) -> None:
        self.capabilities = LLMCaps(
            native_tools=True,
            json_schema=json_schema,
            streaming=True,
            context_window=context_window,
        )
        self._responses = [list(r) for r in responses]
        self.calls: list[list[Message]] = []
        self.tools_seen: list[list[ToolSpec]] = []
        self.schemas_seen: list[dict[str, Any] | None] = []
        self.exits: list[str] = []
        self.open_streams = 0
        self.aclose_calls = 0

    async def aclose(self) -> None:
        self.aclose_calls += 1

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        # Record eagerly, at call time, then hand back an async generator.
        if not self._responses:
            raise AssertionError("FakeLLM has no scripted responses left")
        self.calls.append(list(messages))
        self.tools_seen.append(list(tools))
        self.schemas_seen.append(response_schema)
        return self._replay(self._responses.pop(0))

    async def _replay(self, script: list[LLMChunk | Pause | Raise]) -> AsyncIterator[LLMChunk]:
        self.open_streams += 1
        outcome = "raised"
        try:
            for step in script:
                if isinstance(step, Pause):
                    step.reached.set()
                    await step.release.wait()
                elif isinstance(step, Raise):
                    raise step.error
                else:
                    yield step
            outcome = "done"
        except GeneratorExit:
            outcome = "closed"
            raise
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            self.open_streams -= 1
            self.exits.append(outcome)


def adapter_for(instance: Any) -> type[Any]:
    """A class to register in a `Registry` whose construction returns `instance`.

    The constructor options from config are recorded in `options` on the class.
    """

    class Adapter:
        options: list[dict[str, Any]] = []

        def __new__(cls, **options: Any) -> Any:
            cls.options.append(options)
            return instance

    return Adapter


class FakeEmbedder:
    """Deterministic small vectors derived from a hash of each text and its kind.

    A query and a document with the same text get different vectors, as with
    real retrieval models that prefix them differently. Calls are recorded in
    `calls` as `(texts, kind)`.
    """

    def __init__(self, dimensions: int = 8) -> None:
        self.capabilities = EmbedderCaps(dimensions=dimensions, model="fake-embedder")
        self.calls: list[tuple[list[str], str]] = []

    async def embed(
        self, texts: Sequence[str], *, kind: Literal["document", "query"] = "document"
    ) -> list[list[float]]:
        self.calls.append((list(texts), kind))
        dims = self.capabilities.dimensions
        vectors = []
        for text in texts:
            digest = hashlib.sha256(f"{kind}:{text}".encode()).digest()
            vectors.append([digest[i % len(digest)] / 255 for i in range(dims)])
        return vectors


class InMemoryMemoryStore:
    """Keeps episodes in a list; retrieval returns an empty context."""

    def __init__(self) -> None:
        self.episodes: list[AnyEvent] = []

    async def append_episode(self, event: AnyEvent) -> None:
        self.episodes.append(event)

    async def retrieve(self, query: str, budget: int) -> MemoryContext:
        return MemoryContext()

    async def recent_episodes(self, session_id: str, limit: int) -> list[AnyEvent]:
        session = [e for e in self.episodes if e.session_id == session_id]
        return session[-limit:] if limit > 0 else []


class InMemoryTraceSink:
    """Keeps recorded turn traces in a list."""

    def __init__(self) -> None:
        self.traces: list[TurnTrace] = []

    async def record(self, trace: TurnTrace) -> None:
        self.traces.append(trace)


class EchoTool:
    """Read-tier tool that returns its `text` argument."""

    name = "echo"
    description = "Repeat the given text."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    risk = Risk.READ

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        return ToolOutput(content=str(args.get("text", "")))
