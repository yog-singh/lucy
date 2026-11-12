"""Ports: the small interfaces the core depends on (ADR 0002).

Adapters implement these structurally; they do not need to subclass them. All
are `runtime_checkable`, which checks only that the members exist, not their
signatures.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from lucy.core.events import AnyEvent
from lucy.core.trace import TurnTrace
from lucy.core.types import (
    AudioChunk,
    EmbedderCaps,
    LLMCaps,
    LLMChunk,
    MemoryContext,
    Message,
    Risk,
    STTCaps,
    ToolOutput,
    ToolSpec,
    Transcript,
    TTSCaps,
)


@runtime_checkable
class LLM(Protocol):
    """A language model that streams normalised chunks."""

    capabilities: LLMCaps

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """Stream the model's response to `messages`, offering `tools`.

        Implementations are async generators (`async def` containing `yield`), so
        callers iterate the result directly with `async for` and do not await
        the call. Tool calls are yielded only once complete.

        If `response_schema` (a JSON Schema) is given, the streamed text must
        form a single JSON document conforming to it. Only adapters whose
        `capabilities.json_schema` is true accept it; others raise. Callers
        normally use `LLMGateway.generate`, which falls back to prompting.
        """
        ...


@runtime_checkable
class STT(Protocol):
    """Speech to text. Input audio is in the core input format."""

    capabilities: STTCaps

    async def transcribe(self, audio: AudioChunk) -> Transcript:
        """Transcribe one utterance."""
        ...


@runtime_checkable
class TTS(Protocol):
    """Text to speech. Output audio is in the core output format."""

    capabilities: TTSCaps

    def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """Synthesise `text`. Implementations are async generators, like `LLM.stream`."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into vectors of `capabilities.dimensions` floats."""

    capabilities: EmbedderCaps

    async def embed(
        self, texts: Sequence[str], *, kind: Literal["document", "query"] = "document"
    ) -> list[list[float]]:
        """Return one vector per input text, in order.

        `kind` says whether the texts are documents to be stored and searched
        (statements, summaries) or search queries; retrieval models embed the
        two differently. Empty input returns `[]`.
        """
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """The episode log plus everything derived from it (see docs/memory.md)."""

    async def append_episode(self, event: AnyEvent) -> None:
        """Append an event to the append-only episode log."""
        ...

    async def retrieve(self, query: str, budget: int) -> MemoryContext:
        """Return memory relevant to `query`, fitting within `budget` tokens."""
        ...

    async def recent_episodes(self, session_id: str, limit: int) -> list[AnyEvent]:
        """The most recent `limit` persisted events of `session_id`, oldest first."""
        ...


@runtime_checkable
class Tool(Protocol):
    """Something the assistant can do, with a JSON-schema input and a risk tier (ADR 0006)."""

    name: str
    description: str
    parameters: dict[str, Any]
    risk: Risk

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        """Run the tool.

        The orchestrator builds the `ToolResult` by attaching the originating
        `ToolCall` id to the returned output.
        """
        ...


@runtime_checkable
class TraceSink(Protocol):
    """Stores one trace per turn, so "why did she say that?" can be answered (ADR 0004)."""

    async def record(self, trace: TurnTrace) -> None:
        """Store `trace`."""
        ...


def tool_spec(tool: Tool) -> ToolSpec:
    """Describe a tool to the LLM."""
    return ToolSpec(name=tool.name, description=tool.description, parameters=tool.parameters)
