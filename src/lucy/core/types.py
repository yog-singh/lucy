"""The core's own normalised data types (ADR 0009).

Adapters translate model- and engine-specific formats into these types and back.
Nothing model-specific (chat templates, thinking tags, native audio formats) may
appear here.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    """Base for immutable core value types."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --- Messages and tools -------------------------------------------------------


class Role(StrEnum):
    """Who authored a message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(_Frozen):
    """A request from the LLM to run a tool with the given arguments."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolOutput(_Frozen):
    """What a tool returns. The orchestrator turns it into a `ToolResult`."""

    content: str
    is_error: bool = False


class ToolResult(_Frozen):
    """The outcome of running a tool, fed back to the LLM."""

    call_id: str
    content: str
    is_error: bool = False


class ToolSpec(_Frozen):
    """What the LLM is told about a tool: its name, purpose and JSON-schema parameters."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class Message(_Frozen):
    """One message in the core's single conversation format.

    An assistant message may carry `tool_calls`; a tool message carries the
    `tool_call_id` it answers.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class Risk(StrEnum):
    """Tool risk tier (ADR 0006). The orchestrator enforces it."""

    READ = "read"
    ACT = "act"
    DANGEROUS = "dangerous"


class FinishReason(StrEnum):
    """Why an LLM stopped generating, normalised across models."""

    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"


class LLMChunk(_Frozen):
    """One streamed piece of LLM output.

    Any combination of fields may be set. `tool_call` is only ever a *completed*
    call: adapters assemble partial tool-call deltas internally. `finish_reason`
    is set on the last chunk of a response.
    """

    text: str | None = None
    tool_call: ToolCall | None = None
    finish_reason: FinishReason | None = None


# --- Audio and speech ---------------------------------------------------------

#: Core input audio format: what STT adapters receive (16 kHz mono 16-bit PCM).
INPUT_SAMPLE_RATE: Final = 16_000
INPUT_CHANNELS: Final = 1
#: Core output audio format: what TTS adapters produce (24 kHz mono 16-bit PCM).
#: 24 kHz is the native rate of common neural TTS engines such as Kokoro.
OUTPUT_SAMPLE_RATE: Final = 24_000
OUTPUT_CHANNELS: Final = 1
#: All core audio is signed 16-bit little-endian PCM.
SAMPLE_WIDTH_BYTES: Final = 2


class AudioChunk(_Frozen):
    """Raw signed 16-bit little-endian PCM audio.

    Adapters resample to the core formats above; the core never resamples.
    Bytes are base64-encoded when serialised to JSON.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", ser_json_bytes="base64", val_json_bytes="base64"
    )

    data: bytes
    sample_rate: int
    channels: int = 1


class Transcript(_Frozen):
    """The text recognised from an utterance."""

    text: str
    language: str | None = None


# --- Memory -------------------------------------------------------------------


class MemoryContext(_Frozen):
    """What memory hands to the context builder for one turn."""

    profile: str = ""
    instructions: list[str] = Field(default_factory=list)
    statements: list[str] = Field(default_factory=list)


# --- Capabilities (ADR 0008) --------------------------------------------------


class LLMCaps(_Frozen):
    """What an LLM adapter's model supports."""

    native_tools: bool
    json_schema: bool
    streaming: bool
    context_window: int = Field(gt=0)


class STTCaps(_Frozen):
    """What an STT adapter supports."""

    streaming: bool


class TTSCaps(_Frozen):
    """What a TTS adapter supports. A non-streaming TTS is sent whole sentences."""

    streaming: bool


class EmbedderCaps(_Frozen):
    """An embedder's vector size and model id; vectors are only comparable within one model."""

    dimensions: int = Field(gt=0)
    model: str
